import asyncio
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper.audio import decode_audio
from huggingface_hub import snapshot_download
from fireredasr2 import FireRedAsr2, FireRedAsr2Config


DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "fireredasr2s-aed")
FIRERED_MODEL_TYPE = os.getenv("FIRERED_MODEL_TYPE", "aed").lower()
MODEL_DIR = os.getenv("MODEL_DIR", f"/app/pretrained_models/FireRedASR2-{FIRERED_MODEL_TYPE.upper()}")
HF_REPO_ID = os.getenv("HF_REPO_ID", "FireRedTeam/FireRedASR2-AED")
HF_TOKEN = os.getenv("HF_TOKEN")
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "8"))
BATCH_WINDOW_MS = float(os.getenv("BATCH_WINDOW_MS", "100"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "64"))
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "600"))


app = FastAPI(title="FireRedASR2S OpenAI-compatible API")
_model = None
_concurrency_semaphore = None
_batch_queue = None
_batcher_task = None
_batcher_lock = None


@dataclass
class _BatchJob:
    request_id: str
    model_alias: str
    response_format: str
    filename: str
    payload: bytes
    future: asyncio.Future


def _model_dir_candidates() -> list[Path]:
    primary = Path(MODEL_DIR)
    candidates = [
        primary,
        Path(f"/app/pretrained_models/FireRedASR2-{FIRERED_MODEL_TYPE.upper()}"),
        Path(f"/app/pretrained_models/FireRedASR-{FIRERED_MODEL_TYPE.upper()}-L"),
    ]
    # preserve order, remove duplicates
    seen = set()
    uniq: list[Path] = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _is_valid_model_dir(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except Exception:  # noqa: BLE001
        resolved = path
    return resolved.exists() and (resolved / "config.yaml").exists() and (resolved / "dict.txt").exists()


def _ensure_model_dir() -> Path:
    for candidate in _model_dir_candidates():
        if _is_valid_model_dir(candidate):
            return candidate.resolve(strict=False)

    target = Path(MODEL_DIR)
    if target.is_symlink():
        target = Path(f"/app/pretrained_models/FireRedASR2-{FIRERED_MODEL_TYPE.upper()}")
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    if not _is_valid_model_dir(target):
        raise RuntimeError(f"Model directory not found or invalid after download: {target}")
    return target


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model() -> FireRedAsr2:
    global _model
    if _model is not None:
        return _model

    model_root = _ensure_model_dir()
    config = FireRedAsr2Config(use_gpu=torch.cuda.is_available())
    model = FireRedAsr2.from_pretrained(FIRERED_MODEL_TYPE, str(model_root), config=config)
    _model = model
    return _model


def _convert_to_16k_mono_wav(source_path: Path, target_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(target_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr[-500:]}")


def _duration_seconds(wav_path: Path) -> int:
    audio = decode_audio(str(wav_path), sampling_rate=16000)
    if isinstance(audio, tuple):
        audio = audio[0]
    if getattr(audio, "shape", None) is None:
        return 0
    return int(round(float(audio.shape[-1]) / 16000.0))


def _transcribe(wav_path: Path) -> str:
    model = _load_model()
    request_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    result = model.transcribe([request_id], [str(wav_path)])
    if isinstance(result, list) and result:
        return str(result[0].get("text", "")).strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return ""


def _transcribe_batch(batch_ids: list[str], batch_wav_paths: list[str]) -> Dict[str, str]:
    model = _load_model()
    result = model.transcribe(batch_ids, batch_wav_paths)
    out: Dict[str, str] = {rid: "" for rid in batch_ids}
    if isinstance(result, list):
        for item in result:
            rid = str(item.get("uttid", ""))
            if rid in out:
                out[rid] = str(item.get("text", "")).strip()
    elif isinstance(result, dict):
        rid = str(result.get("uttid", ""))
        if rid in out:
            out[rid] = str(result.get("text", "")).strip()
    return out


def _get_concurrency_semaphore() -> asyncio.Semaphore | None:
    global _concurrency_semaphore
    if MAX_CONCURRENT_REQUESTS <= 0:
        return None
    if _concurrency_semaphore is None:
        _concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _concurrency_semaphore


async def _ensure_batcher() -> asyncio.Queue:
    global _batch_queue, _batcher_task, _batcher_lock
    if _batcher_lock is None:
        _batcher_lock = asyncio.Lock()
    async with _batcher_lock:
        if _batch_queue is None:
            _batch_queue = asyncio.Queue()
        if _batcher_task is None or _batcher_task.done():
            _batcher_task = asyncio.create_task(_batch_consumer())
    return _batch_queue


async def _process_batch(batch: list[_BatchJob]) -> None:
    tmp_dirs: list[Path] = []
    batch_ids: list[str] = []
    batch_wavs: list[str] = []
    durations: Dict[str, int] = {}

    try:
        for job in batch:
            tmp_dir = Path(tempfile.mkdtemp(prefix="fireredasr2s-"))
            tmp_dirs.append(tmp_dir)
            raw_path = tmp_dir / f"input-{uuid.uuid4().hex}-{job.filename}"
            wav_path = tmp_dir / "input-16k.wav"
            raw_path.write_bytes(job.payload)
            _convert_to_16k_mono_wav(raw_path, wav_path)

            batch_ids.append(job.request_id)
            batch_wavs.append(str(wav_path))
            durations[job.request_id] = _duration_seconds(wav_path)

        batch_results = await asyncio.to_thread(_transcribe_batch, batch_ids, batch_wavs)

        for job in batch:
            text = str(batch_results.get(job.request_id, "")).strip()
            if not text:
                raise RuntimeError("Empty transcription result")
            if job.response_format == "text":
                payload: Any = text
            else:
                payload = {
                    "text": text,
                    "usage": {"type": "duration", "seconds": int(durations.get(job.request_id, 0))},
                    "model": job.model_alias,
                }
            if not job.future.done():
                job.future.set_result(payload)
    except Exception as exc:  # noqa: BLE001
        for job in batch:
            if not job.future.done():
                job.future.set_exception(exc)
    finally:
        for tmp in tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)


async def _batch_consumer() -> None:
    global _batch_queue
    if _batch_queue is None:
        return

    loop = asyncio.get_running_loop()
    while True:
        job = await _batch_queue.get()
        batch: list[_BatchJob] = [job]
        deadline = loop.time() + (max(0.0, BATCH_WINDOW_MS) / 1000.0)

        while len(batch) < max(1, BATCH_MAX_SIZE):
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                next_job = await asyncio.wait_for(_batch_queue.get(), timeout)
                batch.append(next_job)
            except asyncio.TimeoutError:
                break

        await _process_batch(batch)


async def _enqueue_transcription(
    *,
    file: UploadFile,
    model: str,
    response_format: str,
) -> Any:
    queue = await _ensure_batcher()
    loop = asyncio.get_running_loop()
    request_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    future = loop.create_future()
    payload = await file.read()
    await queue.put(
        _BatchJob(
            request_id=request_id,
            model_alias=model,
            response_format=response_format,
            filename=file.filename,
            payload=payload,
            future=future,
        )
    )
    return await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT_SEC)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_NAME,
                "object": "model",
                "created": now,
                "owned_by": "fireredasr",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_NAME),
    response_format: str = Form("json"),
) -> Any:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    sem = _get_concurrency_semaphore()
    try:
        if sem is None:
            return await _enqueue_transcription(file=file, model=model, response_format=response_format)
        async with sem:
            return await _enqueue_transcription(file=file, model=model, response_format=response_format)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
