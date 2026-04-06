import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests


DEFAULT_API_KEY = "sk-1234"
ALLOWED_GPU_IDS = {1, 2, 3}
ASR_NETWORK = "asr-net"


@dataclass
class GpuInfo:
	index: int
	total_mb: int
	used_mb: int

	@property
	def free_mb(self) -> int:
		return self.total_mb - self.used_mb


@dataclass
class StartedModel:
	alias: str
	model_name: str
	container_name: str
	port: int
	service_name: str
	health_endpoint: str
	model_info_endpoint: str


@dataclass
class PublicModel:
	alias: str
	litellm_model: str
	api_key_env: str


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
	return subprocess.run(
		cmd,
		check=True,
		cwd=str(cwd) if cwd else None,
		text=True,
		capture_output=True,
	)


def run_cmd_allow_failure(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
	return subprocess.run(
		cmd,
		check=False,
		cwd=str(cwd) if cwd else None,
		text=True,
		capture_output=True,
	)


def load_gpu_info() -> List[GpuInfo]:
	try:
		result = run_cmd(
			[
				"nvidia-smi",
				"--query-gpu=index,memory.total,memory.used",
				"--format=csv,noheader,nounits",
			]
		)
	except (FileNotFoundError, subprocess.CalledProcessError):
		return []

	gpus: List[GpuInfo] = []
	for line in result.stdout.strip().splitlines():
		parts = [p.strip() for p in line.split(",")]
		if len(parts) != 3:
			continue
		try:
			idx = int(parts[0])
			total = int(parts[1])
			used = int(parts[2])
		except ValueError:
			continue
		gpus.append(GpuInfo(index=idx, total_mb=total, used_mb=used))
	return gpus


def select_gpu(gpus: List[GpuInfo], min_free_mb: int) -> Optional[GpuInfo]:
	candidates = [g for g in gpus if g.index in ALLOWED_GPU_IDS and g.free_mb >= min_free_mb]
	if not candidates:
		return None
	return sorted(candidates, key=lambda g: g.free_mb, reverse=True)[0]


def _parse_env_lines(raw: str) -> Dict[str, str]:
	parsed: Dict[str, str] = {}
	for line in raw.splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		parsed[key.strip()] = value.strip()
	return parsed


def write_env(env_path: Path, env_vars: Dict[str, str]) -> Path:
	existing: Dict[str, str] = {}
	if env_path.exists():
		existing = _parse_env_lines(env_path.read_text(encoding="utf-8"))
	existing.update(env_vars)
	lines = [f"{key}={value}" for key, value in existing.items()]
	env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return env_path


def _normalize_name(name: str) -> str:
	normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "").strip())
	return normalized.strip("-").lower() or "model"


def ensure_network(network_name: str) -> None:
	try:
		run_cmd(["docker", "network", "inspect", network_name])
	except subprocess.CalledProcessError:
		run_cmd(["docker", "network", "create", network_name])


def write_litellm_config(workspace_root: Path, started_models: List[StartedModel], public_models: Optional[List["PublicModel"]] = None) -> Path:
	lines: List[str] = ["model_list:"]
	for model in started_models:
		lines.extend(
			[
				f"  - model_name: {model.alias}",
				"    litellm_params:",
				f"      model: hosted_vllm/{model.model_name}",
				f"      api_base: http://{model.container_name}:{model.port}/v1",
				f"      api_key: \"{DEFAULT_API_KEY}\"",
			]
		)

	for model in (public_models or []):
		lines.extend(
			[
				f"  - model_name: {model.alias}",
				"    litellm_params:",
				f"      model: {model.litellm_model}",
				f"      api_key: os.environ/{model.api_key_env}",
			]
		)

	lines.extend(
		[
			"",
			"general_settings:",
			"  disable_auth: true",
			"  disable_database: true",
			"",
		]
	)

	config_yaml = workspace_root / "config.yaml"
	config_yaml.write_text("\n".join(lines), encoding="utf-8")
	return config_yaml


def ensure_litellm_env(workspace_root: Path, public_models: Optional[List["PublicModel"]] = None) -> Path:
	env_path = workspace_root / ".env"
	# Read existing .env so we never overwrite a real API key with a placeholder
	existing: Dict[str, str] = {}
	if env_path.exists():
		existing = _parse_env_lines(env_path.read_text(encoding="utf-8"))
	env_vars: Dict[str, str] = {"LITELLM_MASTER_KEY": f'"{DEFAULT_API_KEY}"'}
	# Only insert placeholder for keys not already present in .env
	for env_var in sorted({m.api_key_env for m in (public_models or [])}):
		if env_var not in existing:
			env_vars[env_var] = ""
	write_env(env_path, env_vars)
	return env_path


def start_litellm(compose_file: Path) -> None:
	service="litellm"
	cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d", "--build"]
	if service:
		cmd.append(service)
	run_cmd(cmd)


def remove_conflicting_containers(container_names: List[str]) -> None:
	for name in container_names:
		run_cmd_allow_failure(["docker", "rm", "-f", name])


def bring_up_stack(compose_file: Path, container_names: List[str]) -> None:
	run_cmd_allow_failure(["docker", "compose", "-f", str(compose_file), "down", "--remove-orphans"])
	remove_conflicting_containers(container_names)
	start_litellm(compose_file)


def _relpath(path: Path, base: Path) -> str:
	try:
		return str(path.relative_to(base))
	except ValueError:
		return str(path)


def write_compose_stack(workspace_root: Path, entries: List[dict], started: List[StartedModel], public_models: Optional[List["PublicModel"]] = None) -> Path:
	lines: List[str] = ["version: \"3.8\"", "", "services:"]
	depends = [m.service_name for m in started]

	lines.extend(
		[
			"  litellm:",
			"    image: litellm/litellm:latest",
			"    container_name: litellm-gateway",
			"    platform: linux/amd64",
			"    entrypoint: [\"litellm\"]",
			"    command: [\"--config\", \"/app/config.yaml\", \"--port\", \"4000\"]",
			"    environment:",
			"      - LITELLM_DISABLE_DATABASE=true",
			"      - LITELLM_DISABLE_AUTH=true",
			f"      - LITELLM_MASTER_KEY={DEFAULT_API_KEY}",
		]
	)

	# Forward API keys for public models into the LiteLLM container
	for env_var in sorted({m.api_key_env for m in (public_models or [])}):
		lines.append(f"      - {env_var}=${{{env_var}}}")

	lines.extend(
		[
			"    volumes:",
			"      - ./config.yaml:/app/config.yaml:ro",
			"    ports:",
			"      - \"4000:4000\"",
			"    networks:",
			"      - asr-net",
		]
	)
	if depends:
		lines.append("    depends_on:")
		for dep in depends:
			lines.append(f"      - {dep}")

	whisperx_used = False
	for entry in entries:
		alias = str(entry.get("name") or "").strip()
		model_name = str(entry.get("model_name") or "").strip()
		port = int(entry.get("port"))
		gpu_id = int(entry.get("gpu_id"))
		compose_dir = Path(entry.get("compose_dir", ""))
		if not compose_dir.is_absolute():
			compose_dir = (workspace_root / compose_dir).resolve()

		safe_alias = _normalize_name(alias)
		service_name = f"{safe_alias}-srv"
		container_name = service_name
		compose_hint = compose_dir.as_posix().lower()
		context_rel = _relpath(compose_dir, workspace_root)

		lines.append(f"  {service_name}:")
		if "whisperx" in compose_hint:
			whisperx_used = True
			uvicorn_port = int(entry.get("port"))
			lines.extend(
				[
					"    image: whisperx-api-server-cuda",
					"    build:",
					f"      context: {context_rel}",
					"      dockerfile: Dockerfile.cuda",
					f"    container_name: {container_name}",
					"    healthcheck:",
					f"      test: [\"CMD-SHELL\", \"curl --fail http://localhost:{uvicorn_port}/healthcheck || exit 1\"]",
					f"    command: uvicorn --factory whisperx_api_server.main:create_app --host 0.0.0.0 --port {uvicorn_port}",
					"    ports:",
					f"      - \"{uvicorn_port}:{uvicorn_port}\"",
					"    environment:",
					"      - UVICORN_HOST=0.0.0.0",
					f"      - UVICORN_PORT={uvicorn_port}",
					f"      - WHISPER__MODEL={model_name}",
					"      - WHISPER__INFERENCE_DEVICE=cuda",
					"      - WHISPER__DEVICE_INDEX=0",
					"      - WHISPER__PRELOAD_MODEL=true",
					f"      - CUDA_VISIBLE_DEVICES={gpu_id}",
					"    volumes:",
					"      - hugging_face_cache:/home/ubuntu/.cache/huggingface",
					"      - torch_cache:/home/ubuntu/.cache/torch",
					"    deploy:",
					"      resources:",
					"        reservations:",
					"          devices:",
					"            - driver: nvidia",
					f"              device_ids: [\"{gpu_id}\"]",
					"              capabilities: [gpu]",
					"    networks:",
					"      - asr-net",
				]
			)
			continue

		if "fireredasr2s" in compose_hint:
			firered_model_type = str(entry.get("firered_model_type", "aed")).lower()
			batch_max_size = int(entry.get("batch_max_size", 8))
			batch_window_ms = float(entry.get("batch_window_ms", 100))
			max_concurrent_requests = int(entry.get("max_concurrent_requests", 64))
			lines.extend(
				[
					"    build:",
					"      context: .",
					"      dockerfile: vllm_inference_models/fireredasr2s_aed/Dockerfile",
					f"    container_name: {container_name}",
					"    deploy:",
					"      resources:",
					"        reservations:",
					"          devices:",
					"            - driver: nvidia",
					f"              device_ids: [\"{gpu_id}\"]",
					"              capabilities: [gpu]",
					"    environment:",
					"      - PYTHONUNBUFFERED=1",
					f"      - MODEL_NAME={model_name}",
					f"      - FIRERED_MODEL_TYPE={firered_model_type}",
					f"      - MODEL_DIR=/app/pretrained_models/FireRedASR2-{firered_model_type.upper()}",
					f"      - BATCH_MAX_SIZE={batch_max_size}",
					f"      - BATCH_WINDOW_MS={batch_window_ms}",
					f"      - MAX_CONCURRENT_REQUESTS={max_concurrent_requests}",
					"    volumes:",
					"      - ./fireredasr-api/pretrained_models:/app/pretrained_models",
					"      - /models:/models",
					"      - ~/.cache/huggingface:/root/.cache/huggingface",
					"    ports:",
					f"      - \"{port}:{port}\"",
					f"    command: uvicorn app.main:app --host 0.0.0.0 --port {port}",
					"    networks:",
					"      - asr-net",
				]
			)
			continue

		is_whisper = "whisper" in compose_hint
		command = f"vllm serve {model_name} --host 0.0.0.0 --port {port}"
		if is_whisper:
			dtype = entry.get("dtype", "bfloat16")
			gpu_mem_util = entry.get("gpu_memory_utilization", "0.3")
			command = (
				"python -m app.main --host 0.0.0.0 --port "
				f"{port} --model {model_name} --dtype {dtype} --gpu-memory-utilization {gpu_mem_util}"
			)

		lines.extend(
			[
				"    build:",
				f"      context: {context_rel}",
				f"    container_name: {container_name}",
				"    deploy:",
				"      resources:",
				"        reservations:",
				"          devices:",
				"            - driver: nvidia",
				f"              device_ids: [\"{gpu_id}\"]",
				"              capabilities: [gpu]",
				"    environment:",
				"      - PYTHONUNBUFFERED=1",
				"    volumes:",
				"      - ~/.cache/huggingface:/root/.cache/huggingface",
				"    ports:",
				f"      - \"{port}:{port}\"",
				f"    command: {command}",
				"    networks:",
				"      - asr-net",
			]
		)

	if whisperx_used:
		lines.extend(["", "volumes:", "  hugging_face_cache:", "  torch_cache:"])

	lines.extend(["", "networks:", "  asr-net:", "    external: true", ""])

	compose_path = workspace_root / "docker-compose.yml"
	compose_path.write_text("\n".join(lines), encoding="utf-8")
	return compose_path


def wait_litellm_ready(base_url: str, timeout_sec: int = 180) -> None:
	deadline = time.time() + timeout_sec
	headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
	url = f"{base_url.rstrip('/')}/v1/models"

	last_error = "unknown error"
	while time.time() < deadline:
		try:
			resp = requests.get(url, headers=headers, timeout=10)
			if resp.status_code == 200:
				return
			last_error = f"status={resp.status_code}, body={resp.text[:300]}"
		except Exception as exc:  # noqa: BLE001
			last_error = str(exc)
		time.sleep(2)

	raise RuntimeError(f"LiteLLM not ready at {url}: {last_error}")


def wait_model_ready(host: str, port: int, endpoints: list, timeout_sec: int = 300) -> None:
	base = f"{host.rstrip('/')}:{port}"
	# this is OpenAI compatible port, so these endpoints won't be exist
	# endpoints = ["/v1/models", "/health"]
	# instead, use this
	# endpoints = ["/models/list", "/healthcheck"]
	deadline = time.time() + timeout_sec
	last_error = "unknown error"

	while time.time() < deadline:
		for endpoint in endpoints:
			print(f"[debug] poking endpoint {endpoint} on port {port}")
			url = f"{base}{endpoint}"
			try:
				resp = requests.get(url, timeout=10, headers={"Authorization": f"Bearer {DEFAULT_API_KEY}"})
				if resp.status_code == 200:
					return
				last_error = f"{url} status={resp.status_code}, body={resp.text[:200]}"
			except Exception as exc:  # noqa: BLE001
				last_error = f"{url} error={exc}"
		time.sleep(3)

	raise RuntimeError(f"Model not ready on {base}: {last_error}")


def validate_inference(base_url: str, sample_audio: Path, aliases: List[str]) -> None:
	if not sample_audio.exists():
		raise FileNotFoundError(f"Sample audio not found: {sample_audio}")

	headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
	url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"

	for alias in aliases:
		last_error = "unknown error"
		for attempt in range(1, 7):
			with sample_audio.open("rb") as audio_file:
				files = {"file": (sample_audio.name, audio_file)}
				data = {"model": alias, "response_format": "json"}
				try:
					resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)
				except Exception as exc:  # noqa: BLE001
					last_error = str(exc)
					time.sleep(5)
					continue

			if resp.status_code == 200:
				try:
					payload = resp.json()
				except ValueError:
					last_error = f"non-json response: {resp.text}"
					time.sleep(5)
					continue
				if not payload.get("text"):
					last_error = f"missing text field: {payload}"
					time.sleep(5)
					continue
				print(f"[ok] {alias}: {payload['text'][:80]}")
				break

			last_error = f"status={resp.status_code}, body={resp.text}"
			time.sleep(5)
		else:
			raise RuntimeError(f"Inference failed for '{alias}': {last_error}")


def resolve_gpu(entry: dict, gpus: List[GpuInfo], min_free_mb: int) -> int:
	gpu_id = entry.get("gpu_id")

	if gpu_id is None:
		chosen = select_gpu(gpus, min_free_mb)
		if chosen is None:
			raise RuntimeError(f"No allowed GPU {sorted(ALLOWED_GPU_IDS)} with >= {min_free_mb}MB free")
		return chosen.index

	gpu_id = int(gpu_id)
	if gpu_id not in ALLOWED_GPU_IDS:
		raise RuntimeError(
			f"gpu_id={gpu_id} is not allowed. Allowed GPU IDs are: {sorted(ALLOWED_GPU_IDS)}"
		)

	if not gpus:
		raise RuntimeError("nvidia-smi unavailable; cannot validate gpu_id")

	gpu_info = next((g for g in gpus if g.index == gpu_id), None)
	if gpu_info is None:
		raise RuntimeError(f"GPU {gpu_id} not found by nvidia-smi")

	if gpu_info.free_mb < min_free_mb:
		print(
			f"[warn] GPU {gpu_id} free memory {gpu_info.free_mb}MB < min_free_gpu_mb={min_free_mb}; continuing because gpu_id is explicitly set"
		)

	return gpu_id


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Start enabled ASR models, generate LiteLLM config from config.json, start gateway, and validate inference."
	)
	parser.add_argument("--config", default="config.json", help="Path to config.json")
	parser.add_argument("--sample-audio", default="sample.wav", help="Audio file used to validate inference")
	parser.add_argument("--litellm-url", default="http://localhost:4000", help="LiteLLM base URL")
	args = parser.parse_args()

	config_path = Path(args.config).resolve()
	if not config_path.exists():
		raise SystemExit(f"Config not found: {config_path}")

	workspace_root = config_path.parent
	config = json.loads(config_path.read_text(encoding="utf-8"))
	models = config.get("asr_model_list", [])
	default_min_free_mb = int(config.get("min_free_gpu_mb", 4096))

	enabled = [m for m in models if bool(m.get("enabled", False))]

	public_entries = config.get("public_model_list", [])
	enabled_public: List[PublicModel] = []
	for entry in public_entries:
		if not bool(entry.get("enabled", False)):
			continue
		alias = str(entry.get("name") or "").strip()
		litellm_model = str(entry.get("litellm_model") or "").strip()
		api_key_env = str(entry.get("api_key_env") or "").strip()
		if not alias or not litellm_model or not api_key_env:
			raise RuntimeError(f"Invalid public model entry: {entry}")
		print(f"[start] public model {alias}")
		enabled_public.append(PublicModel(alias=alias, litellm_model=litellm_model, api_key_env=api_key_env))

	if not enabled and not enabled_public:
		raise SystemExit("No enabled models in config.json")

	gpus = load_gpu_info()
	if enabled and not gpus:
		raise SystemExit("nvidia-smi not available; unable to validate required GPU IDs [1,2,3]")

	ensure_network(ASR_NETWORK)

	started_models: List[StartedModel] = []

	for entry in enabled:
		alias = str(entry.get("name") or "").strip()
		model_name = str(entry.get("model_name") or "").strip()
		port = int(entry.get("port"))
		compose_dir = Path(entry.get("compose_dir", ""))
		if not compose_dir.is_absolute():
			compose_dir = (workspace_root / compose_dir).resolve()

		if not alias:
			raise RuntimeError(f"Invalid model entry without name: {entry}")
		if not model_name:
			raise RuntimeError(f"Invalid model '{alias}': missing model_name")
		if not compose_dir.exists():
			raise RuntimeError(f"Invalid model '{alias}': compose_dir not found at {compose_dir}")

		min_free_mb = int(entry.get("min_free_gpu_mb", default_min_free_mb))
		gpu_id = resolve_gpu(entry, gpus, min_free_mb)

		# for poking post container init
		health_endpoint = entry.get("health_endpoint", "")
		model_info_endpoint = entry.get("model_info_endpoint", "")

		safe_alias = _normalize_name(alias)
		container_name = f"{safe_alias}-srv"

		print(f"[start] {alias}: GPU={gpu_id}, PORT={port}, CONTAINER={container_name}")

		started_models.append(
			StartedModel(
				alias=alias,
				model_name=model_name,
				container_name=container_name,
				port=port,
				service_name=container_name,
				health_endpoint=health_endpoint,
				model_info_endpoint=model_info_endpoint
			)
		)

	print("[info] Writing compose stack")
	compose_file = write_compose_stack(workspace_root, enabled, started_models, enabled_public)

	print("[info] Writing litellm config")
	write_litellm_config(workspace_root, started_models, enabled_public)

	print("[info] Validating litellm .env for public models")
	ensure_litellm_env(workspace_root, enabled_public)

	print("[info] Initializing litellm routing and model endpoints")
	bring_up_stack(compose_file, [m.container_name for m in started_models])

	print("[info] Poking litellm router endpoint")
	wait_litellm_ready(args.litellm_url)

	for model in started_models:
		print(f"[info] Poking litellm local model container")
		endpoints = [model.model_info_endpoint, model.health_endpoint]
		wait_model_ready("http://localhost", model.port, endpoints=endpoints)

	sample_audio = Path(args.sample_audio)
	if not sample_audio.is_absolute():
		sample_audio = (workspace_root / sample_audio).resolve()

	local_aliases = [m.alias for m in started_models]
	public_aliases = [m.alias for m in enabled_public]
	if local_aliases:
		try:
			validate_inference(args.litellm_url, sample_audio, aliases=local_aliases)
		except FileNotFoundError:
			raise FileNotFoundError(f"Sample audio not found {sample_audio}! Skipping local validation of inference!")

	if public_aliases:
		print(f"[info] Public models registered (not validated locally): {', '.join(public_aliases)}")
	print("[done] All enabled models are up and inference-able through LiteLLM.")


if __name__ == "__main__":
	main()
