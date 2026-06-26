# Base: dustynv/vllm pre-built for Jetson AGX Orin (L4T R36.4.0 / JetPack 6.x)
# Ships PyTorch + CUDA 12.6 + vLLM compiled for ARM64 — do not swap this base
# without verifying CUDA/driver compatibility with the host JetPack version.
FROM dustynv/vllm:r36.4.0

# Qwen2 runtime dependencies not included in the vLLM base image.
# - transformers>=4.40.0 : Qwen2 architecture support added in 4.40
# - accelerate           : device_map / multi-GPU dispatch
# - tiktoken             : Qwen tokenizer backend
# - einops               : tensor ops used by Qwen attention layers
# - transformers_stream_generator : streaming generation helper
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate \
    tiktoken \
    einops \
    transformers_stream_generator

WORKDIR /app

COPY pyproject.toml .
COPY main.py .

# HF_HOME is also set in devcontainer.json for the dev environment;
# kept here so the production image behaves the same way.
ENV HF_HOME=/workspace/.cache/huggingface

CMD ["python3", "main.py"]
