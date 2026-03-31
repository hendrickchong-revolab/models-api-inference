FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt update \
    && apt install -y build-essential ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
RUN uv pip install "vllm[audio]"

WORKDIR /workspace
