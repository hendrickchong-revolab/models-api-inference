FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY vllm_inference_models/fireredasr2s_aed/requirements.txt /app/requirements.txt
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r /app/requirements.txt

COPY vllm_inference_models/fireredasr2s_aed/app /app/app
COPY vllm_inference_models/fireredasr2s_aed/fireredasr2 /app/fireredasr2

EXPOSE 7091

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7091"]
