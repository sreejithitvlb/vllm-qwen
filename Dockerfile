# Stage 1: base-vllm
# Locally built vLLM 0.20.0 image for Jetson AGX Orin (JetPack 6.x, CUDA 12.x).
# Newer vLLM releases require a CUDA version JetPack 6 doesn't ship, so vLLM,
# torch and the CUDA stack come pre-installed from this image instead of PyPI.
FROM vllm-jetson:0.20.0 AS base-vllm

# uv may not be present in the base image; install it from upstream PyPI
# (see the note in Stage 2 about the Jetson wheel index).
RUN command -v uv >/dev/null 2>&1 || \
    pip3 install --no-cache-dir --index-url https://pypi.org/simple uv

# The base image's system cv2 (/usr/lib/python3.10/dist-packages) was compiled
# against NumPy 1.x but NumPy 2.2.6 is installed, so `import cv2` fails with
# "_ARRAY_API not found". The PyPI wheel is built for NumPy 2 and installs to
# /usr/local/lib/python3.10/dist-packages, which precedes the system copy on
# sys.path. numpy is pinned so the resolver can't swap out the base image's build.
RUN uv pip install --system --no-cache --break-system-packages \
    --index-url https://pypi.org/simple \
    opencv-python-headless \
    "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')"

# compressed-tensors is a vLLM dependency missing from the base image; vLLM imports
# it while validating any --quantization (including awq_marlin). The Jetson build's
# vllm metadata doesn't list it, so the pin is copied from vLLM v0.20.0's
# requirements/common.txt.
RUN uv pip install --system --no-cache --break-system-packages \
    --index-url https://pypi.org/simple \
    "compressed-tensors==0.15.0.1" \
    "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')"

# Verify vLLM, cv2, numpy and compressed_tensors import cleanly
RUN python3 -c "import numpy, cv2, compressed_tensors, vllm; print(f'✓ vLLM {vllm.__version__} | numpy {numpy.__version__} | cv2 {cv2.__version__} ({cv2.__file__})')"


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

# The base image sets ENTRYPOINT to `vllm`, which would receive CMD as arguments.
ENTRYPOINT []
CMD ["python3", "main.py"]
