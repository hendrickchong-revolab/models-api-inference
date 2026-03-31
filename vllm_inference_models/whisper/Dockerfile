FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install --no-cache-dir \
    "vllm==0.11.0" \
    "transformers==4.57.6" \
    librosa \
    silero-vad \
    torchaudio \
    numpy \
    uvloop \
    uvicorn \
    onnxruntime

WORKDIR /workspace
COPY app /workspace/app
