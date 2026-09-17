# vllm-qwen

Serve a **MoE LLM with vLLM on a Jetson AGX Orin 64GB**, developed via a VS Code Dev
Container that runs **on the Jetson** over SSH. See [RESUME.md](RESUME.md) for full status,
the JetPack/disk blockers, the model shortlist, and the benchmarking plan.

## Dev container (milestone 1)

The dev container runs on the Jetson itself (ARM64 + GPU), built from an L4T base image
that matches the installed JetPack (5.1.2 / L4T R35.4.1). It pulls no model weights yet.

### Prerequisites
- VS Code with the **Remote - SSH** and **Dev Containers** extensions.
- Passwordless SSH to the Jetson: `ssh sree@jetson`.

### Steps
1. In VS Code: **Remote-SSH → Connect to Host → `jetson`**.
2. Open this folder on the Jetson.
3. Command Palette → **Dev Containers: Reopen in Container** (the image builds/pulls on the
   Jetson; first pull of the L4T base is several GB).
4. In the container terminal, verify the GPU is visible (use `python3` — `python` in the L4T
   image is Python 2.7, which has no torch):
   ```bash
   python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Expect `True` and a device name containing `Orin`. The same check runs automatically as
   `postCreateCommand`, so look for it in the build log too.

### Notes
- The container requests the GPU via `--runtime=nvidia` (the Jetson's default runtime is
  `runc`). See [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json).
- Disk is tight (~16 GB free). Don't pull model weights into this container — that waits
  for the disk/NVMe decision (see [RESUME.md](RESUME.md)).

## Configuration

### Environment variables

These can be overridden at `docker run` time with `-e VAR=value` or set in the shell before running `main.py` directly.

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | HuggingFace model ID to load and serve. |
| `SERVED_MODEL_NAME` | `qwen2.5-7b` | Model name exposed by the OpenAI-compatible API (`/v1/models`, `model` field). |
| `PORT` | `8000` | TCP port the vLLM HTTP server listens on. |
| `GPU_MEM_UTIL` | `0.83` | Fraction of GPU VRAM (0–1) reserved for model weights and the KV cache. Lower this if the server OOMs; raise it to fit longer contexts. |
| `MAX_MODEL_LEN` | `4096` | Maximum sequence length in tokens (prompt + completion). Reducing this lowers KV cache memory pressure. |
| `MAX_NUM_SEQS` | `32` | Maximum number of sequences processed concurrently. Lower values reduce memory usage at the cost of throughput. |

### Fixed vLLM flags

These are hard-coded in [main.py](main.py) and optimised for the Jetson AGX Orin.

| Flag | Value | Reason |
|---|---|---|
| `--quantization` | `awq_marlin` | 4-bit AWQ quantization using the Marlin CUDA kernel. Required because the model (`-AWQ` suffix) was quantized with AWQ; Marlin gives the best throughput on Jetson's Ampere GPU. |
| `--dtype` | `float16` | FP16 for all activations and compute. Jetson Orin does not support BF16 natively. |
| `--tensor-parallel-size` | `1` | Single GPU — the Orin has one integrated GPU. |
| `--swap-space` | `0` | Disables CPU swap for KV cache pages. Avoids slow PCIe memory paging which would tank latency on Jetson's shared memory architecture. |
| `--host` | `0.0.0.0` | Bind on all interfaces so the API is reachable from the host and the network. |
| `--enable-prefix-caching` | *(flag)* | Reuses KV cache blocks across requests that share a common prompt prefix (e.g. a system prompt). Reduces latency and memory churn in chat workloads. |
| `--trust-remote-code` | *(flag)* | Allows Qwen's custom tokenizer and model code to execute when loaded from HuggingFace. Required for Qwen2.5 models. |
| `--max-log-len` | `100` | Truncates per-request log lines to 100 characters to keep logs readable. |

To learn more about any flag, run:

```bash
# Full reference
vllm serve --help

# Look up a specific flag, e.g.:
vllm serve --help | grep -A 5 -- --quantization
vllm serve --help | grep -A 5 -- --gpu-memory-utilization
vllm serve --help | grep -A 5 -- --max-model-len
```

<!-- sync project locally-->
scp -r sree@jetson:~/projects/vllm-qwen/* C:\Users\sreej\github-projects\jetson-deployment\vllm-qwen\