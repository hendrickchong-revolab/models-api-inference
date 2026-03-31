# Models API Inference

Docker-first ASR inference workspace that runs multiple model backends behind a single LiteLLM gateway with OpenAI-compatible transcription endpoints.

## What This Repo Does

- Starts one or more ASR model services from a JSON config.
- Generates `docker-compose.yml` and LiteLLM `config.yaml` automatically.
- Exposes all enabled models through a single gateway at `http://localhost:4000`.
- Verifies readiness and runs a real transcription check against each enabled model.

The main entrypoint is `start_models.py`. That script is the source of truth for orchestration.

## Runtime Architecture

1. `config.json` defines which models are enabled, their aliases, ports, container contexts, and GPU assignments.
2. `start_models.py` validates GPU availability and creates the external Docker network `asr-net` if needed.
3. `start_models.py` rewrites `docker-compose.yml` for the enabled model set.
4. `start_models.py` rewrites `config.yaml` so LiteLLM can route requests by alias.
5. Docker Compose starts the model containers and the LiteLLM gateway.
6. The launcher waits for `/v1/models` and model health endpoints, then sends a transcription request using `sample.wav`.

The result is one gateway with model selection by alias, not by direct container URL.

## Supported Models

### Enabled in the committed config

These are enabled in `config.json` and exposed in the generated `config.yaml` currently:

| Alias | Backend | Model | Port |
| --- | --- | --- | --- |
| `whisper_ms_precise` | custom Whisper OpenAI-compatible server | `mesolitica/Malaysian-whisper-large-v3-turbo-v3` | `7086` |
| `whisperx` | WhisperX API server | `large-v3` | `7090` |
| `qwen_asr` | `vllm serve` | `Qwen/Qwen3-ASR-1.7B` | `7087` |

### Available in `config.json` but disabled by default

| Alias | Backend | Model | Port |
| --- | --- | --- | --- |
| `whisper` | custom Whisper OpenAI-compatible server | `openai/whisper-large-v3` | `7085` |
| `glm_asr` | `vllm serve` | `zai-org/GLM-ASR-Nano-2512` | `7088` |
| `fireredasr2s_aed` | custom FastAPI wrapper | `fireredasr2s-aed` | `7091` |

## Key Files

- `start_models.py`: launcher that validates GPUs, writes generated files, starts containers, and validates inference.
- `config.json`: editable model matrix and deployment settings.
- `config.yaml`: generated LiteLLM config used by the gateway.
- `docker-compose.yml`: generated compose stack for LiteLLM plus enabled model services.
- `docker-compose.template.yml`: minimal compose template for the gateway network shape.
- `test_litellm.py`: simple multipart transcription request against LiteLLM.
- `qwen_requests.py`: direct chat-completions style request for Qwen ASR.
- `models_benchmark/qwen_benchmark_requests.py`: sequential or parallel request benchmark runner.

## Requirements

- Docker with Compose support
- NVIDIA GPU runtime available to Docker
- `nvidia-smi` available on the host
- GPUs matching the launcher constraints in `start_models.py`

Current launcher validation only allows GPU IDs `1`, `2`, and `3`.

## Configuration Model

Edit `config.json`, not `config.yaml`, when changing which models should run.

Example model entry:

```json
{
  "name": "qwen_asr",
  "enabled": true,
  "compose_dir": "vllm_inference_models/qwen_asr",
  "model_name": "Qwen/Qwen3-ASR-1.7B",
  "port": 7087,
  "gpu_id": 2
}
```

Important fields:

- `name`: alias used in LiteLLM requests as the `model` value.
- `enabled`: whether this model is launched.
- `compose_dir`: container build context.
- `model_name`: upstream model identifier or runtime model name.
- `port`: service port exposed by the container.
- `gpu_id`: host GPU assignment.

Additional per-model keys are also used by the launcher for specific backends, such as FireRed batch settings or Whisper dtype and GPU memory utilization.

## Start The Stack

From the repository root:

```bash
python start_models.py
```

Optional flags:

```bash
python start_models.py \
  --config config.json \
  --sample-audio sample.wav \
  --litellm-url http://localhost:4000
```

What the launcher does:

- loads enabled models from `config.json`
- validates GPU availability and memory thresholds
- ensures the external Docker network exists
- generates `docker-compose.yml`
- generates `config.yaml`
- writes `.env` with the LiteLLM master key
- starts the stack with Docker Compose
- waits for gateway and model readiness
- validates transcription for every enabled alias

## Use The Gateway

Base URL:

```bash
http://localhost:4000
```

Default API key used by the generated config:

```bash
sk-1234
```

List routed models:

```bash
curl -s \
  -H "Authorization: Bearer sk-1234" \
  http://localhost:4000/v1/models
```

Send an OpenAI-compatible transcription request:

```bash
curl -s http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-1234" \
  -F file=@sample.wav \
  -F model=qwen_asr \
  -F response_format=json
```

Required multipart fields:

- `file`
- `model`

Common optional fields:

- `response_format`
- `language`
- `prompt`
- `temperature`
- `timestamp_granularities[]`

## Local Test And Benchmark Scripts

Test LiteLLM transcription routing:

```bash
LITELLM_MODEL=qwen_asr python test_litellm.py
```

The script defaults to `fireredasr2s_aed`, which is disabled in the committed config, so setting `LITELLM_MODEL` is recommended unless you enable FireRed.

Send a direct Qwen chat-completions request:

```bash
python qwen_requests.py --audio /path/to/audio.mp3 --url http://localhost:8061/v1/chat/completions
```

Run the Qwen request benchmark:

```bash
python models_benchmark/qwen_benchmark_requests.py \
  --audio-dir audios_test \
  --url http://localhost:8061/v1/chat/completions \
  --parallel 4
```

## Repository Layout

```text
.
├── config.json
├── config.yaml
├── docker-compose.template.yml
├── docker-compose.yml
├── models/
├── models_benchmark/
├── qwen_requests.py
├── start_models.py
├── test_litellm.py
└── vllm_inference_models/
```

Model container directories under `vllm_inference_models/`:

- `whisper/`
- `whisperx/`
- `qwen_asr/`
- `glm_asr/`
- `fireredasr2s_aed/`

## Notes And Caveats

- `config.yaml` is generated. Treat `config.json` as the editable source of truth.
- `docker-compose.yml` is generated by the launcher and will be overwritten.
- LiteLLM runs with auth and database disabled in the generated config, using the fixed master key `sk-1234`.
- FireRed support expects additional mounted model assets under `fireredasr-api/pretrained_models` when enabled.
- WhisperX uses a dedicated CUDA API server path rather than `vllm serve`.

## Troubleshooting

If startup fails:

- check `nvidia-smi` on the host
- confirm Docker can access the NVIDIA runtime
- verify enabled `compose_dir` paths exist
- make sure the configured GPU IDs are valid for this machine

If LiteLLM is up but inference fails:

- call `GET /v1/models` first to confirm the alias is routed
- use one of the enabled aliases from `config.json`
- inspect container logs with Docker Compose for the model backend that failed

## License

MIT. See `LICENSE`.
