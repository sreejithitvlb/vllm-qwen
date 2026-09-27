# Stage 1: base-vllm
# NVIDIA L4T CUDA base with JetPack 6.x for Jetson AGX Orin
# Includes CUDA 12.6 + cuDNN + TensorRT; no vLLM pre-installed
FROM nvcr.io/nvidia/l4t-cuda:12.6-runtime-jetpack6 AS base-vllm

# Install build dependencies for vLLM compilation on ARM64
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Install vLLM 0.29.0 (stable release for ARM64)
# PyTorch will be pulled as a vLLM dependency
RUN uv pip install --system --no-cache --break-system-packages \
    --index-url https://pypi.org/simple \
    vllm==0.29.0

# Verify vLLM installation
RUN python3 -c "import vllm; print(f'✓ vLLM version: {vllm.__version__}')"


# Stage 2: app
# Qwen2 runtime dependencies and application code
FROM base-vllm AS app

# Qwen2 runtime dependencies not included in the vLLM base image.
# - transformers>=4.40.0 : Qwen2 architecture support added in 4.40
# - accelerate           : device_map / multi-GPU dispatch
# - tiktoken             : Qwen tokenizer backend
# - einops               : tensor ops used by Qwen attention layers
# - transformers_stream_generator : streaming generation helper
# - pandas               : not a Qwen dependency; vllm's CLI entrypoint eagerly
#                           imports vllm.entrypoints.cli.benchmark, which imports
#                           vllm.benchmarks.datasets, which requires pandas even
#                           when only running `vllm serve`.
#
# Install from upstream PyPI, NOT the Jetson wheel index baked into the base image.
# dustynv sets PIP_INDEX_URL=http://jetson.webredirect.org/..., whose host only has
# an IPv6 (AAAA) record; Docker's default bridge is IPv4-only, so the build can't
# resolve it ("Errno -2 Name or service not known"). These deps are pure-Python or
# ship aarch64 wheels on PyPI — the Jetson index is only needed for torch/vllm/CUDA,
# which are already present in the base image.
RUN uv pip install --system --no-cache --break-system-packages \
    --index-url https://pypi.org/simple \
    "transformers>=4.40.0" \
    accelerate \
    tiktoken \
    einops \
    transformers_stream_generator \
    pyyaml \
    pandas

WORKDIR /app

COPY pyproject.toml .
COPY main.py .
COPY config/ ./config/

# HF_HOME is also set in devcontainer.json for the dev environment;
# kept here so the production image behaves the same way.
ENV HF_HOME=/workspace/.cache/huggingface

CMD ["python3", "main.py"]
