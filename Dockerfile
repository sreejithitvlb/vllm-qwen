# Base: dustynv/vllm pre-built for Jetson AGX Orin (L4T R36.4.0 / JetPack 6.x)
# Ships PyTorch + CUDA 12.6 + vLLM compiled for ARM64 — do not swap this base
# without verifying CUDA/driver compatibility with the host JetPack version.
FROM dustynv/vllm:r36.4-cu129-24.04

RUN pip install --no-cache-dir uv

# Upgrade vLLM to latest version available for ARM64
RUN uv pip install --system --no-cache --break-system-packages \
    --upgrade \
    --index-url https://pypi.org/simple \
    vllm

# Print vLLM version for verification
RUN python3 -c "import vllm; print(f'✓ vLLM version: {vllm.__version__}')"

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
