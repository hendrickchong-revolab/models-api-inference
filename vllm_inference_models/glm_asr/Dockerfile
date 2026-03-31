FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update -o Dir::Cache::archives=/tmp/apt \
    && apt-get install -y --no-install-recommends -o Dir::Cache::archives=/tmp/apt \
        ca-certificates gnupg ubuntu-keyring \
    && apt-get update -o Dir::Cache::archives=/tmp/apt \
    && apt-get install -y --no-install-recommends -o Dir::Cache::archives=/tmp/apt \
        build-essential ffmpeg git \
    && rm -rf /var/lib/apt/lists/* /tmp/apt

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# Install vLLM audio support first, then override Transformers with source
# Finally pin numpy for numba compatibility (ensure it stays <=2.2)
RUN uv pip install --no-cache-dir "vllm[audio]" --torch-backend auto \
    && uv pip install --no-cache-dir --upgrade git+https://github.com/huggingface/transformers.git \
    && uv pip install --no-cache-dir --upgrade "numpy<2.3"

WORKDIR /workspace
