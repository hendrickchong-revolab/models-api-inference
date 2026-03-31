import contextlib
import os
import queue
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import gc
import logging
import time
import tempfile
import asyncio
from fastapi import UploadFile

from whisperx import audio as whisperx_audio

from whisperx_api_server.config import (
    Language,
)
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.backends.registry import (
    get_default_transcription_model_name,
    get_alignment_backend,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
)

logger = logging.getLogger(__name__)

config = get_config()

_concurrency_semaphore = None
_io_executor: ThreadPoolExecutor | None = None
_UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_UPLOAD_WRITE_BUFFER_SIZE = 1024 * 1024  # 1 MiB
_batch_queue: asyncio.Queue["_BatchJob"] | None = None
_batcher_task: asyncio.Task | None = None
_batcher_lock: asyncio.Lock | None = None


@dataclass
class _BatchJob:
    future: asyncio.Future
    kwargs: dict


def _get_concurrency_semaphore() -> asyncio.Semaphore | None:
    global _concurrency_semaphore
    if not torch.cuda.is_available():
        return None
    if _concurrency_semaphore is None:
        max_concurrent = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "1"))
        _concurrency_semaphore = asyncio.Semaphore(max_concurrent)
    return _concurrency_semaphore


def _cleanup_cache_only():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_io_executor() -> ThreadPoolExecutor:
    global _io_executor
    if _io_executor is None:
        workers = int(os.getenv("IO_EXECUTOR_WORKERS", "4"))
        _io_executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="whisperx_io")
    return _io_executor


def _get_batch_settings() -> tuple[int, float]:
    max_size = int(os.getenv("BATCH_MAX_SIZE", "10"))
    window_ms = float(os.getenv("BATCH_WINDOW_MS", "100"))
    if max_size < 1:
        max_size = 1
    if window_ms < 0:
        window_ms = 0.0
    return max_size, window_ms


async def _ensure_batcher() -> asyncio.Queue["_BatchJob"]:
    global _batch_queue, _batcher_task, _batcher_lock
    if _batcher_lock is None:
        _batcher_lock = asyncio.Lock()
    async with _batcher_lock:
        if _batch_queue is None:
            _batch_queue = asyncio.Queue()
        if _batcher_task is None or _batcher_task.done():
            _batcher_task = asyncio.create_task(_batch_consumer())
    return _batch_queue


async def _batch_consumer() -> None:
    global _batch_queue
    if _batch_queue is None:
        return
    loop = asyncio.get_running_loop()
    while True:
        job = await _batch_queue.get()
        batch = [job]
        max_size, window_ms = _get_batch_settings()
        deadline = loop.time() + (window_ms / 1000.0)
        while len(batch) < max_size:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                next_job = await asyncio.wait_for(_batch_queue.get(), timeout)
            except asyncio.TimeoutError:
                break
            else:
                batch.append(next_job)

        tasks = [asyncio.create_task(_run_job(j)) for j in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for j, result in zip(batch, results):
            if j.future.cancelled():
                continue
            if isinstance(result, BaseException):
                j.future.set_exception(result)
            else:
                j.future.set_result(result)


async def _run_job(job: "_BatchJob") -> dict[str, Any]:
    return await _transcribe_impl(**job.kwargs)


async def _save_upload_to_temp(audio_file: UploadFile, request_id: str) -> str:
    """Stream upload to temp file in chunks"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as tmp:
        file_path = tmp.name

    chunk_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=2)
    write_error: list[BaseException] = []

    def _writer() -> None:
        try:
            with open(file_path, "wb", buffering=_UPLOAD_WRITE_BUFFER_SIZE) as f:
                while True:
                    chunk = chunk_queue.get()
                    if chunk is None:
                        break
                    f.write(chunk)
        except BaseException as e:
            write_error.append(e)

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    try:
        while True:
            chunk = await audio_file.read(_UPLOAD_STREAM_CHUNK_SIZE)
            if len(chunk) == 0:
                break
            try:
                chunk_queue.put_nowait(chunk)
            except queue.Full:
                await asyncio.to_thread(chunk_queue.put, chunk)
            if len(chunk) < _UPLOAD_STREAM_CHUNK_SIZE:
                break
        try:
            chunk_queue.put_nowait(None)
        except queue.Full:
            await asyncio.to_thread(chunk_queue.put, None)
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Failed to read uploaded file: {e}")
        with contextlib.suppress(queue.Full):
            chunk_queue.put_nowait(None)
        await asyncio.to_thread(writer_thread.join, 5)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise
    finally:
        await asyncio.to_thread(writer_thread.join, 30)

    if write_error:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        logger.error(
            f"Request ID: {request_id} - Failed to write temp file: {write_error[0]}")
        raise write_error[0]

    return file_path


async def _load_audio(file_path: str, request_id: str) -> np.ndarray:
    loop = asyncio.get_running_loop()
    executor = _get_io_executor()
    try:
        audio = await loop.run_in_executor(executor, whisperx_audio.load_audio, file_path)
        logger.info(
            f"Request ID: {request_id} - Audio loaded from {file_path}")
        return audio
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Failed to load audio: {e}")
        raise


def _finalize_text(result: dict[str, Any], align_or_diarize: bool) -> dict[str, Any]:
    segments = result.get("segments", [])
    if align_or_diarize and isinstance(segments, dict):
        segments = segments.get("segments", [])

    result["text"] = '\n'.join([s.get("text", "").strip()
                               for s in segments if s.get("text")])
    return result


async def _transcribe_impl(
    audio_file: UploadFile,
    batch_size: int = config.whisper.batch_size,
    chunk_size: int = config.whisper.chunk_size,
    asr_options: dict | None = None,
    language: Language = config.default_language,
    model_name: str | None = None,
    align: bool = False,
    diarize: bool = False,
    speaker_embeddings: bool = False,
    request_id: str = "",
    task: str = "transcribe",
) -> dict[str, Any]:
    start_time = time.perf_counter()
    file_path = None
    audio = None
    concurrency_sem = _get_concurrency_semaphore()
    semaphore_acquired = False
    profile: dict[str, float] = {}

    try:
        t0 = time.perf_counter()
        file_path = await _save_upload_to_temp(audio_file, request_id)
        profile["upload_save"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Saving uploaded file took {profile['upload_save']:.2f} seconds")

        if concurrency_sem:
            await concurrency_sem.acquire()
            semaphore_acquired = True
            logger.debug(
                f"Request ID: {request_id} - Acquired GPU concurrency semaphore")

        selected_backends = resolve_stage_backends()
        transcription_stage_backend = get_transcription_backend(
            selected_backends.transcription)
        alignment_stage_backend = (
            get_alignment_backend(selected_backends.alignment)
            if (align or diarize)
            else None
        )
        diarization_stage_backend = (
            get_diarization_backend(selected_backends.diarization)
            if diarize
            else None
        )

        if not model_name:
            model_name = get_default_transcription_model_name()

        logger.info(
            f"Request ID: {request_id} - Transcribing {audio_file.filename} with model: {model_name}, options: {asr_options}, language: {language}, task: {task}, stage_backends: {selected_backends}")

        t0 = time.perf_counter()
        audio = await _load_audio(file_path, request_id)
        profile["audio_load"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Loading audio took {profile['audio_load']:.2f} seconds")

        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
            file_path = None

        t0 = time.perf_counter()
        result = await transcription_stage_backend.transcribe(
            model_name=model_name,
            audio=audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            language=language,
            task=task,
            asr_options=asr_options,
            request_id=request_id,
        )
        profile["transcribe"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Transcription took {profile['transcribe']:.2f} seconds")

        if align or diarize:
            if alignment_stage_backend is None:
                raise RuntimeError("Alignment backend is not initialized.")
            t0 = time.perf_counter()
            result = await alignment_stage_backend.align(
                result=result,
                audio=audio,
                request_id=request_id,
            )
            profile["align"] = time.perf_counter() - t0
            logger.debug(
                f"Request ID: {request_id} - Alignment took {profile['align']:.2f} seconds")

        if diarize:
            if diarization_stage_backend is None:
                raise RuntimeError("Diarization backend is not initialized.")
            t0 = time.perf_counter()
            result = await diarization_stage_backend.diarize(
                result=result,
                audio=audio,
                speaker_embeddings=speaker_embeddings,
                request_id=request_id,
            )
            profile["diarize"] = time.perf_counter() - t0
            logger.debug(
                f"Request ID: {request_id} - Diarization took {profile['diarize']:.2f} seconds")

        t0 = time.perf_counter()
        result = _finalize_text(result, align or diarize)
        profile["finalize"] = time.perf_counter() - t0

        total = time.perf_counter() - start_time
        logger.info(
            f"Request ID: {request_id} - Transcription completed for {audio_file.filename}")
        logger.debug(
            f"Request ID: {request_id} - profile: total={total:.2f}s | "
            + " | ".join(f"{k}={v:.2f}s" for k, v in profile.items())
            + f" | (other={total - sum(profile.values()):.2f}s)")

        return result
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Transcription failed for {audio_file.filename} with error: {e}")
        raise
    finally:
        with contextlib.suppress(Exception):
            if concurrency_sem and semaphore_acquired:
                concurrency_sem.release()
        with contextlib.suppress(Exception):
            if file_path is not None and os.path.exists(file_path):
                os.remove(file_path)
        if config.audio_cleanup and audio is not None:
            del audio
            logger.info(f"Request ID: {request_id} - Audio data cleaned up")
        if config.cache_cleanup:
            _cleanup_cache_only()
            logger.info(f"Request ID: {request_id} - Cache cleanup completed")


async def transcribe(
    audio_file: UploadFile,
    batch_size: int = config.whisper.batch_size,
    chunk_size: int = config.whisper.chunk_size,
    asr_options: dict | None = None,
    language: Language = config.default_language,
    model_name: str | None = None,
    align: bool = False,
    diarize: bool = False,
    speaker_embeddings: bool = False,
    request_id: str = "",
    task: str = "transcribe",
) -> dict[str, Any]:
    max_size, window_ms = _get_batch_settings()
    if max_size <= 1 and window_ms <= 0:
        return await _transcribe_impl(
            audio_file=audio_file,
            batch_size=batch_size,
            chunk_size=chunk_size,
            asr_options=asr_options,
            language=language,
            model_name=model_name,
            align=align,
            diarize=diarize,
            speaker_embeddings=speaker_embeddings,
            request_id=request_id,
            task=task,
        )

    queue_obj = await _ensure_batcher()
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    job = _BatchJob(
        future=future,
        kwargs={
            "audio_file": audio_file,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "asr_options": asr_options,
            "language": language,
            "model_name": model_name,
            "align": align,
            "diarize": diarize,
            "speaker_embeddings": speaker_embeddings,
            "request_id": request_id,
            "task": task,
        },
    )
    await queue_obj.put(job)
    return await future
