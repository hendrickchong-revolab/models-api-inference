import base64
import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="ASR Adapter")

CONFIG_PATH = Path(os.environ.get("ADAPTER_CONFIG", "/app/adapter_config.json"))


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    config = load_config()
    litellm_base: str = config["litellm_base"]
    litellm_key: str = config["litellm_key"]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{litellm_base}/v1/models",
            headers={"Authorization": f"Bearer {litellm_key}"},
        )
    return resp.json()


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(...),
    response_format: str = Form("json"),
    language: str = Form(None),
    prompt: str = Form(None),
):
    config = load_config()
    litellm_base: str = config["litellm_base"]
    litellm_key: str = config["litellm_key"]
    chat_audio_models: list = config.get("chat_audio_models", [])

    audio_bytes = await file.read()
    mime_type = file.content_type or "audio/wav"

    if model in chat_audio_models:
        # Gemini-style models don't support /audio/transcriptions — translate to multimodal chat completions
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        transcription_prompt = prompt or "Transcribe this audio exactly. Return only the transcription text, nothing else."
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": transcription_prompt},
                        {"type": "file", "file": {"file_data": f"data:{mime_type};base64,{encoded}"}},
                    ],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{litellm_base}/v1/chat/completions",
                headers={"Authorization": f"Bearer {litellm_key}", "Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        text = resp.json()["choices"][0]["message"]["content"]
        return {"text": text}

    else:
        # Standard transcription models — forward directly to LiteLLM
        form_data: dict = {"model": model, "response_format": response_format}
        if language:
            form_data["language"] = language
        if prompt:
            form_data["prompt"] = prompt
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{litellm_base}/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {litellm_key}"},
                files={"file": (file.filename or "audio.wav", audio_bytes, mime_type)},
                data=form_data,
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
