# Jetson MoE LLM Serving with vLLM — Progress & Resume Notes

_Last updated: 2026-06-23 (milestone 2 — **JetPack 6.2 flashed onto eMMC**; SDK components still to install)_

## Goal
Serve a **Mixture-of-Experts (MoE) LLM** (model still **TBD** — see shortlist) on the
**Jetson AGX Orin 64GB** using **vLLM**, developed through a **VS Code Remote-SSH + Dev
Container that runs on the Jetson** (ARM64 + GPU), then **benchmark** the serving.

### Incremental milestones
1. **Dev container acquaintance** ✅ _DONE (2026-06-14)_ — see "Milestone 1 status" below
2. **JetPack 6.2 flash** 🟡 _OS flashed 2026-06-23 (onto eMMC, not NVMe); SDK components still to install_ ← _current focus_
3. **Serve a DENSE model first with vLLM** — Qwen2.5-7B-Instruct, to validate the serve/benchmark pipeline before MoE _(plan change 2026-06-22: dense-first de-risking)_
4. **Switch to the MoE model** with vLLM — Qwen3-30B-A3B (compare shortlist)
5. Metrics & benchmarking

---

## ✅ Milestone 1 status (dev container — DONE 2026-06-14)

Remote-SSH into the Jetson works from VS Code, and the dev container **builds and runs on the
Jetson** (ARM64 + GPU) from `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`. GPU passthrough
confirmed: `torch 2.0.0a0+…nv23.02`, `cuda available: True`, device `Orin` (verified both by a
direct `docker run --rm --runtime=nvidia … python3 -c "import torch…"` and by the container build).

**Gotchas found & fixed (so they don't recur):**
- **`python` in this image is Python 2.7; torch is on `python3` only.** The `postCreateCommand`
  originally used `python` → failed `ImportError: No module named torch` (non-fatal; container
  still built). Fixed to `python3`.
- **The `r35.4.1` l4t-pytorch tag does not exist** (NGC's newest r35 tag is `r35.2.1`). An early
  build failed pulling `r35.4.1`; image pinned to `r35.2.1` (runs fine on the R35.4.1 host).
- **Two repo clones that drift:** laptop `C:\Users\sreej\github-projects\jetson-deployment\vllm-qwen`
  **and** the Jetson `~/vllm-qwen` (the Remote-SSH workspace). A "Reopen in Container" build reads
  the **Jetson** copy, so edits made only on the laptop don't take effect — **sync via git** to
  avoid hand-patching both.
- **Make sure the build targets the Jetson, not local Docker Desktop.** Reopen-in-Container builds
  against whatever Docker context the window points at; do it from inside the `[SSH: jetson]`
  window so it builds on the Jetson (ARM64), not local WSL2 (x86_64).

---

## Jetson profile (re-probed 2026-06-23, post-flash)

| Item | Value |
|------|-------|
| Board | Jetson **AGX Orin** Developer Kit |
| Memory | 61 GiB unified RAM (~58 GiB free at idle) |
| GPU | Ampere, ~204 GB/s mem bandwidth, **no FP8** (Ampere) |
| CPU | 8 cores, aarch64 |
| JetPack / L4T | **JetPack 6.2.2+b24 / L4T R36.5.0** _(was JP5.1.2 / R35.4.1 — flashed 2026-06-23)_ |
| CUDA | **none on host — by design.** CUDA comes from inside the vLLM container (the `dustynv/vllm:*-r36.4-*` image ships CUDA 12.6). Host `nvidia-jetpack` deliberately **NOT** installed (saves several GB of eMMC; only needed for bare-metal `nvcc`, which we don't do). |
| OS | **Ubuntu 22.04.5 LTS** _(was 20.04.6)_ |
| GPU driver | `nvidia-smi` works (driver **540.5.0**, device `Orin`) — ships with the L4T BSP; this is all the host needs, the container brings CUDA. |
| Docker | **not installed yet** (`docker: command not found`) — install `docker.io` + nvidia-container-toolkit (this is the only host install needed). |
| Disk `/` | **57 G eMMC, 7.7 G used, ~46 G free (15% used)** — flashed to **eMMC** (`/dev/mmcblk0p1`); **NVMe never purchased** |

---

## ✅ Blockers (both resolved by the 2026-06-23 flash)

> **Update 2026-06-23:** the Jetson was flashed to **JetPack 6.2.2 / L4T R36.5.0 (Ubuntu 22.04.5)
> onto the eMMC** (NVMe was never bought). This clears **blocker 1** outright. **Blocker 2** is
> eased — eMMC now shows **~46 G free** (15% used), enough for the dense phase and likely the MoE
> phase too, though an NVMe is still nice-to-have for comfort/HF cache. The history below is kept
> for context.

### 1. ~~JetPack version — upgrade 5.1.2 → 6.2+~~ ✅ DONE (flashed 2026-06-23)
Modern **vLLM + MoE realistically requires JetPack 6.2+ (L4T r36.4, CUDA 12.6)**:
- Prebuilt vLLM wheels/containers target it (`pip` `jp6/cu126` index, `nvcr.io/nvidia/vllm`,
  `dusty-nv/jetson-containers` r36.4). Working **MoE kernels** are tested there.
- On the current **JP5.1.2 / CUDA 11.4** path, vLLM means fragile source builds with
  **uncertain MoE support** — not recommended for the serving phase.
- **Confirmed 2026-06-22:** `dustynv/vllm` publishes **only r36.4 (JP6) tags — no r35/JP5 image
  exists**; NVIDIA forum mod: *"We don't have a vLLM container for the r35 branch."* The only JP5
  route is a from-source build, which produces a ~15–20 GB image that **won't fit in the ~20 GB
  free eMMC**. → **vLLM (dense or MoE) genuinely requires the flash. llama.cpp/ollama would run on
  JP5 today, but the user chose vLLM, so we flash.**
- **Cost of upgrade:** JetPack upgrade = **flash** (= erase-and-reinstall the Jetson's OS;
  rewrites only the target storage, **not** the laptop). Done with **NVIDIA SDK Manager,
  which runs on the Windows laptop directly** (it auto-sets-up the Linux part via WSL2 — no
  separate PC needed).
- **Decision made: flash JP6.2 onto NVMe** (see "JetPack 6.2 + NVMe upgrade plan" below).
  Not blocking the dev-container milestone (that runs on JP5 today).

### 2. Disk too small for model weights — ✅ eased (46 G free after fresh flash)
- **Update 2026-06-23:** the fresh JP6.2 flash left the eMMC with **~46 G free (7.7 G used of 57 G)**.
  That comfortably fits the **dense** phase (Qwen2.5-7B AWQ ~5 GB + vLLM r36.4 image ~15–20 GB),
  and is workable for the **MoE** phase (30B AWQ ~18 GB) though it gets tight with the HF cache.
- **NVMe was never purchased** — flash went to eMMC instead. An NVMe is still a nice-to-have:
  mount an explicit **HF cache** (`HF_HOME`) onto it for the MoE phase if eMMC gets cramped.
- _History (JP5):_ ollama removal had freed ~6 GB to ~22 GB on the old install; moot now.

---

## 🔧 JetPack 6.2 flash — ✅ OS DONE 2026-06-23 (eMMC); SDK install still pending

**What actually happened:** flashed **JetPack 6.2.2 (L4T R36.5.0, Ubuntu 22.04.5) onto the eMMC**
(the NVMe was never purchased). The OS image + GPU driver are up (`nvidia-smi` works); the **JetPack
SDK components — CUDA 12.6 toolkit, cuDNN, TensorRT, Docker, nvidia-container-toolkit — are NOT yet
installed** (`nvidia-jetpack` shows Installed: none, Candidate: 6.2.2). That install is the next step.

> _Original plan was NVMe; reality is eMMC._ The flash erased the old JP5 install — there is no
> JP5 fallback now. The historical NVMe plan is kept below for reference if an NVMe is added later.

### Prerequisites (everything is doable from the Windows laptop — no second PC)
- [ ] **M.2 NVMe SSD — drive selected 2026-06-22, NOT YET PURCHASED.** Chosen:
      **Ediloca EN705 1 TB** (M.2 2280, Key-M, PCIe **Gen4** NVMe, ships with heatsink) — matches all
      requirements (Gen4 is backward-compatible; Orin slot is Gen4 x4). Spec needed: M.2 2280, Key-M,
      **PCIe NVMe** (not SATA), **1 TB**. **Heatsink clearance caveat:** the AGX Orin dev kit's M.2
      Key-M slot is on the carrier-board **underside** — a tall heatsink may not fit; the drive works
      fine bare if so. Install in the Orin's M.2 slot before flashing.
- [ ] **The Windows laptop itself** runs **NVIDIA SDK Manager** (recent versions run on Windows
      and auto-configure the Linux part via **WSL2** — no separate Ubuntu PC needed).
      *Caveat: the Windows/WSL2 flash path works but can be finicky — expect a possible retry.*
- [ ] **NVIDIA Developer account** (free; SDK Manager login).
- [ ] **USB-C cable** laptop ↔ Jetson, and a way to put the Jetson in **Force Recovery** mode
      (button combo on the dev kit).
- [ ] Note current Jetson IP (`10.0.0.74`) and that a fresh OS **resets `~/.ssh/authorized_keys`**
      — SSH key + the `jetson` alias must be re-established after flashing.

### Pre-flash backup (eMMC stays, but capture anything you want)
- [ ] Docker is empty (nothing to save there). Copy off anything in `~` you care about and note
      the ollama models under `/usr/share/ollama` if you want to re-pull them later:
      `scp -r sree@jetson:~/something ./backup/`

### Flash steps (SDK Manager on Windows → NVMe)
1. On the Windows laptop, install & launch **SDK Manager**; log in. On first Jetson flash it
   will offer to set up **WSL2 + Ubuntu + USB bridging** automatically — accept.
2. Physically install the **NVMe** into the Jetson (power off first).
3. Put the Jetson in **Force Recovery** mode and connect the USB-C cable to the laptop; SDK
   Manager should detect it (it attaches the USB device to WSL2 for you).
4. In SDK Manager **Step 1**: target = **Jetson AGX Orin**, version = **JetPack 6.2**.
5. **Step 2 → Storage device = NVMe** (this is the key choice that puts the OS on NVMe, not eMMC).
6. Flash the OS image, then let it install **CUDA 12.6 / cuDNN 9 / TensorRT 10** (over the network
   after first boot, or via SDK Manager).
7. First boot: complete Ubuntu 22.04 setup (recreate user `sree`, hostname).

### Post-flash setup (re-establish the workflow)
- [x] **SSH:** `ssh sree@jetson` works again (verified 2026-06-23).
- [x] **Verify the flash:**
      - `cat /etc/nv_tegra_release` → **R36 (release) REV 5.0** ✅ (L4T 36.5.0)
      - `lsb_release -d` → **Ubuntu 22.04.5 LTS** ✅
      - `nvidia-smi` works (driver 540.5.0, `Orin`) ✅
      - `df -h /` → root on **eMMC** (`/dev/mmcblk0p1`), 46 G free ✅ _(on eMMC, not NVMe)_
- [ ] **CUDA — install on HOST: NO (decided 2026-06-23).** CUDA is provided by the vLLM **container**,
      not the host. The JP6.2 host driver (540.5.0, CUDA 12.6-class) matches the r36.4 image's CUDA 12.6,
      so the container's CUDA runs fine on this host. Skip `nvidia-jetpack` (saves several GB of eMMC) —
      only install it if you ever need bare-metal `nvcc`, which this project doesn't.
- [ ] **Docker + GPU (the only host install needed):**
      ```bash
      sudo apt update && sudo apt install docker.io nvidia-container-toolkit
      sudo usermod -aG docker sree
      sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
      sudo systemctl restart docker   # then log out/in for the group change
      ```
      Verify the **container's** CUDA sees the GPU: `docker run --rm --runtime=nvidia dustynv/vllm:r36.4.0 nvidia-smi`.
- [ ] **Performance:** `sudo nvpmodel -m 0` (MAXN) + `sudo jetson_clocks` for serving/benchmarks.
- [ ] **(Recommended) install `dusty-nv/jetson-containers`** — gives a tested vLLM image for r36.4.

### Update the dev container for JP6
- [ ] In [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json), switch the base
      `image` from the r35.4.1 l4t-pytorch tag to an **r36.4 / vLLM-ready** image (e.g. a
      `dustynv/vllm:*-r36.4.0` tag or jetson-containers vLLM build).
- [ ] Enable the now-commented **HF cache mount** pointing at an NVMe path
      (e.g. `source=/mnt/nvme/hf-cache`).
- [ ] Reopen in container; re-run the GPU sanity check (`torch.cuda.is_available()` → `True`).

### Fallback — OTA upgrade (only if the Windows/WSL2 flash won't cooperate)
NVIDIA/RidgeRun document an in-place **OTA** 5→6 migration run over SSH on the Jetson (no PC at
all). Downsides: higher brick risk, and it **won't relocate the OS to NVMe** (stays on eMMC), so
you'd still need a separate disk fix. Treat as last resort — and note that recovering from a failed
OTA would itself need SDK Manager (the Windows path above), so try the Windows flash first.

---

## Dense warm-up model (milestone 3 — serve this FIRST)

**Plan change 2026-06-22:** before the MoE model, serve a **dense** model with vLLM to validate the
end-to-end serve + benchmark pipeline on the freshly-flashed JP6 stack.

| Model | ~4-bit size | Notes |
|-------|-------------|-------|
| **Qwen2.5-7B-Instruct (AWQ)** | ~5–5.5 GB | **Chosen first deploy.** Dense sibling of the Qwen3-MoE end goal (same tokenizer/prompt family). Official `Qwen/Qwen2.5-7B-Instruct-AWQ` on HF. AWQ INT4 (Ampere has no FP8). Tiny on the 1 TB NVMe; runs easily in the 59 GB RAM. |

## MoE model shortlist (to compare in milestone 4)

**Selection criterion:** Orin is **memory-bandwidth bound (~204 GB/s)**, so decode speed
tracks **active parameters**, not total. Favor **low active-param MoE**. Ampere has **no FP8**
→ quantize with **AWQ / GPTQ INT4**.

| Model | Total / Active | ~4-bit size | Notes |
|-------|----------------|-------------|-------|
| **Qwen3-30B-A3B (AWQ)** | 30B / 3B | ~16–18 GB | **Lead candidate.** Benchmarked on Orin 64GB: ~41 tok/s (user) → ~77 tok/s (NVIDIA-tuned), `--max-model-len 20480 --gpu-memory-utilization 0.9 --tensor-parallel-size 1` |
| Qwen3.5 / 3.6 35B-A3B | 35B / 3B | ~19–20 GB | Newer, native tool calling |
| DeepSeek-V2-Lite | 16B / 2.4B | ~9 GB | Lighter fallback, more headroom |
| gpt-oss-20b | 21B / 3.6B | ~12 GB | Alt fallback |

---

## vLLM Core Concepts (learn these first)

A short map of the ideas you will encounter when reading vLLM docs or tuning flags.
No need to go deep on all of them up front — the sequence below is the right order.

### 1. PagedAttention (the core idea)
vLLM's defining innovation. Normally the KV cache (the key/value tensors produced during
attention for every token in the context) is allocated as one big contiguous block per
sequence — wasteful and hard to share. PagedAttention splits the KV cache into fixed-size
**pages** (blocks of 16 tokens by default) stored non-contiguously, like virtual memory.
This lets vLLM:
- pack sequences of different lengths without fragmentation,
- share KV pages between requests that share a prefix (prefix caching),
- swap pages to CPU RAM when GPU memory is full (`--swap-space`).
**Key takeaway:** `--block-size` controls page size; `--gpu-memory-utilization` controls how
much GPU RAM vLLM reserves for pages.

### 2. Continuous batching
Traditional serving processes one request at a time (or a fixed batch). vLLM uses
**continuous batching** (also called iteration-level scheduling): new requests are inserted
into the running batch as soon as a slot frees, without waiting for the slowest sequence to
finish. This dramatically improves GPU utilization under concurrent load.
`--max-num-seqs` caps the number of sequences in the batch at any moment.

### 3. Prefill vs decode (the two inference phases)
Every request goes through two phases:
- **Prefill:** process all input prompt tokens in one forward pass (compute-heavy, runs fast,
  produces the first output token). Metric: **TTFT** (time-to-first-token).
- **Decode:** generate one token per forward pass, auto-regressively, until done (memory-
  bandwidth-bound). Metric: **TPOT** (time-per-output-token) or **ITL** (inter-token latency).

On the Orin, decode is the bottleneck because it is memory-bandwidth-bound (204 GB/s).
For MoE models, only the active experts' weights are read per token — that is why a 30B MoE
with 3B active parameters decodes at roughly a 3B dense model's speed.

### 4. KV cache sizing
`--gpu-memory-utilization × total_GPU_RAM − model_weights_VRAM = KV cache budget`.
A larger KV cache lets more sequences run in parallel and supports longer `max-model-len`.
If you reduce `--max-model-len`, more of the KV budget is available per concurrent sequence.

### 5. Quantization (AWQ / GPTQ)
Both are **post-training weight quantization** schemes that compress weights to INT4 or INT8:
- **AWQ** (Activation-aware Weight Quantization) — searches for the best per-channel scale
  factors by sampling activations; generally better quality than GPTQ at same bit-width.
  Flag: `--quantization awq`. Ampere supports INT4 GEMM kernels; no FP8.
- **GPTQ** — older method; broader model availability. Flag: `--quantization gptq`.
Only the **weights** are quantized; activations remain in FP16. The model loads faster and
uses ~2× less VRAM than BF16.

### 6. The two vLLM APIs

| API | When to use | Entry point |
|-----|-------------|-------------|
| `vllm.LLM` (offline) | Batch inference in a script; no HTTP | `from vllm import LLM, SamplingParams` |
| `vllm serve` (online) | Persistent HTTP server; OpenAI-compatible | `vllm serve <model> [flags]` |

For serving on the Jetson, you want `vllm serve`. The `LLM` class is useful for one-off experiments.

### 7. SamplingParams (controls text generation)
```python
from vllm import SamplingParams
params = SamplingParams(
    temperature=0.7,      # randomness; 0 = greedy, 1 = sample from full distribution
    top_p=0.9,            # nucleus sampling: keep only the top-p probability mass
    max_tokens=256,       # max output tokens; prevents runaway generation
    stop=["</s>", "\n"],  # stop sequences
    repetition_penalty=1.1,
)
```
In the HTTP API these map to the same-named JSON fields in the request body.

### 8. OpenAI-compatible API
`vllm serve` exposes three endpoints that match the OpenAI API shape:
- `GET  /v1/models` — list loaded model(s)
- `POST /v1/chat/completions` — chat (messages array in, assistant reply out)
- `POST /v1/completions` — raw text completion (legacy prompt-in, text-out)

Any client built for the OpenAI Python SDK works with vLLM with only a `base_url` change:
```python
from openai import OpenAI
client = OpenAI(base_url="http://jetson:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

### 9. Chunked prefill (`--enable-chunked-prefill`)
When a long prompt arrives, the prefill pass can take hundreds of milliseconds and stall
decode for in-flight sequences. Chunked prefill breaks the prefill into smaller chunks
interleaved with decode steps — trades TTFT for smoother TPOT under mixed traffic. Enable
when `max-model-len` is large (≥ 8k) and you care about tail latency, not just throughput.

### 10. Async engine (advanced)
`vllm serve` internally uses `AsyncLLMEngine` — an async wrapper around the core engine that
handles concurrent HTTP requests without blocking. You only touch this if you embed vLLM into
a custom FastAPI app rather than using the built-in server.

---

## Milestone 3 — vLLM Serving Layer (implementation plan)

This section is the hands-on guide for standing up the serving layer once Docker is installed on the Jetson host (the remaining milestone 2 step).

---

### Step 0 — prerequisites (finish milestone 2)

```bash
# On the Jetson host (SSH in)
sudo apt update && sudo apt install -y docker.io nvidia-container-toolkit
sudo usermod -aG docker sree
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
# Log out and back in so the docker group takes effect, then verify:
docker run --rm --runtime=nvidia dustynv/vllm:r36.4.0 nvidia-smi
```

---

### Step 1 — download model weights

Model weights live **outside** the container so they survive rebuilds.
Create a persistent cache directory on the host (or wherever ~46 G of eMMC allows):

```bash
mkdir -p ~/hf-cache
```

Download the dense model (AWQ INT4, ~5 GB):

```bash
docker run --rm \
  -v ~/hf-cache:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  dustynv/vllm:r36.4.0 \
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ
```

If the model is gated, add `-e HUGGING_FACE_HUB_TOKEN=<your-token>`.

---

### Step 2 — launch the vLLM server (Docker)

```bash
docker run --rm \
  --runtime=nvidia \
  --shm-size=8g \
  -v ~/hf-cache:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -p 8000:8000 \
  dustynv/vllm:r36.4.0 \
  vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen2.5-7b \
    --trust-remote-code
```

Server is ready when the log shows: `INFO:     Application startup complete.`

---

### Step 3 — configuration flags (what each one does)

| Flag | Value (dense) | Why |
|------|---------------|-----|
| `--quantization awq` | `awq` | Load weights as AWQ INT4; halves VRAM vs FP16. |
| `--dtype` | `float16` | Compute dtype for non-weight ops. Ampere supports FP16 and BF16; FP16 is safe here. (No FP8 on Ampere.) |
| `--gpu-memory-utilization` | `0.90` | Fraction of GPU memory vLLM reserves for KV cache + model. 0.90 leaves a 10% safety margin. Raise to 0.95 if you need more context; lower if OOM. |
| `--max-model-len` | `8192` | Max total tokens (prompt + output). KV cache is pre-allocated for this length — shorter = more room for concurrent sequences. |
| `--tensor-parallel-size` | `1` | Number of GPUs. Orin has one GPU, so always 1. |
| `--max-num-seqs` | `32` | Max in-flight requests. On a 64 GB Orin this can go higher; start at 32 and tune. |
| `--host` | `0.0.0.0` | Bind on all interfaces so the laptop can reach the Jetson over SSH or LAN. |
| `--port` | `8000` | OpenAI-compatible API port. |
| `--served-model-name` | `qwen2.5-7b` | The name clients use in `model:` fields of API requests (alias — the actual model weights are unchanged). |
| `--trust-remote-code` | _(flag)_ | Qwen models use custom model code on HF; required for Qwen family. |
| `--enforce-eager` | _(flag, add if needed)_ | Disables CUDA graph capture. Lower throughput but more stable if graph capture OOMs or hangs. Add for MoE or long-context runs if startup fails. |
| `--swap-space` | `4` (GiB) | CPU RAM to use as KV cache overflow (paged out when GPU KV cache is full). Add `--swap-space 4` for long-context or high-concurrency. |
| `--block-size` | `16` (default) | Paged-attention block size in tokens. Leave at default unless you are tuning memory fragmentation. |
| `--kv-cache-dtype` | `auto` (default) | `auto` = same as `--dtype`. Do **not** set `fp8` — Ampere has no FP8. |
| `--enable-chunked-prefill` | _(flag, optional)_ | Breaks long prefills into chunks to reduce TTFT jitter under load. Useful when `max-model-len` is large. |
| `--max-log-len` | `100` | Truncates logged prompts. Add to keep server logs readable. |

**Environment variables** (set with `-e` in `docker run` or in the container):

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | HuggingFace cache root — must match the volume mount. |
| `HUGGING_FACE_HUB_TOKEN` | Auth token for gated models. |
| `VLLM_WORKER_MULTIPROC_METHOD` | Set to `spawn` if you hit multiprocessing issues (rare). |
| `CUDA_VISIBLE_DEVICES` | Restrict which GPU(s) vLLM sees. Orin has one, so leave unset. |

---

### Step 4 — smoke-test the API

vLLM exposes an **OpenAI-compatible REST API**. Test with `curl` from the laptop (replace `jetson` with the Jetson's IP/hostname):

```bash
# List loaded models
curl http://jetson:8000/v1/models

# Chat completion
curl http://jetson:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
    "max_tokens": 64
  }'

# Text completion (legacy)
curl http://jetson:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "prompt": "The capital of France is",
    "max_tokens": 16
  }'
```

Expected responses are JSON with `choices[0].message.content` (chat) or `choices[0].text` (completion).

---

### Step 5 — serving script (main.py)

Two patterns — pick one:

**Pattern A: launch the OpenAI-compatible HTTP server programmatically**

```python
import subprocess, sys

def main():
    cmd = [
        "vllm", "serve", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--quantization", "awq",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "8192",
        "--tensor-parallel-size", "1",
        "--max-num-seqs", "32",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", "qwen2.5-7b",
        "--trust-remote-code",
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
```

The container `CMD` already calls `python3 main.py`, so this pattern lets `docker run` start the server directly.

**Pattern B: use vLLM's Python API for offline/batch inference (no HTTP server)**

```python
from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.7, max_tokens=256)
    outputs = llm.generate(["Hello, what is 2+2?"], sampling)
    for o in outputs:
        print(o.outputs[0].text)

if __name__ == "__main__":
    main()
```

Pattern B is useful for batch jobs; Pattern A is what you want for a persistent serving endpoint.

---

### Step 6 — Jetson performance tuning (before benchmarking)

Run these on the Jetson host **before** starting the server container:

```bash
# Maximum power/performance mode (MAXN — all CPU + GPU cores at full clock)
sudo nvpmodel -m 0

# Lock clocks at maximum (eliminates throttling during benchmark)
sudo jetson_clocks

# Confirm GPU clock is pinned
cat /sys/devices/17000000.gpu/devfreq/17000000.gpu/cur_freq
```

`nvpmodel -m 0` + `jetson_clocks` are **required** before any benchmark run — without them, clocks
throttle mid-run and results are not reproducible. Add them to a startup script or cron on the Jetson.

---

### Step 7 — MoE model flags (Milestone 4: Qwen3-30B-A3B)

When switching from the dense 7B to the 30B MoE, change these flags:

```bash
vllm serve Qwen/Qwen3-30B-A3B-AWQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 20480 \          # MoE can handle longer context; 20k is a safe start
  --tensor-parallel-size 1 \
  --max-num-seqs 16 \              # fewer slots — MoE KV cache is still sized for active experts
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-30b-moe \
  --trust-remote-code \
  --enforce-eager                  # add this first run; remove if startup succeeds without it
```

MoE notes:
- The Orin's 204 GB/s memory bandwidth limits **decode** speed to ~active-parameter bandwidth,
  so Qwen3-30B-A3B (3B active) should feel like a 3B dense model at decode time.
- If the HF cache + model weights start crowding the 46 G eMMC, move `HF_HOME` to an NVMe
  (mount as `/mnt/nvme/hf-cache`) — the only reason to buy the NVMe.
- AWQ variant: look for `Qwen/Qwen3-30B-A3B-AWQ` or a community-quantized version; if no AWQ
  exists, use `--quantization gptq` with a GPTQ model instead.

---

### Step 8 — serving checklist

- [ ] Docker + nvidia-container-toolkit installed; `docker run --rm --runtime=nvidia … nvidia-smi` passes
- [ ] `nvidia-ctk runtime configure --runtime=docker --set-as-default` done + docker restarted
- [ ] `~/hf-cache` directory created on host; volume-mounted into container
- [ ] Dense model weights downloaded: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- [ ] vLLM server starts without error; log shows `Application startup complete.`
- [ ] `/v1/models` curl returns the model name
- [ ] `/v1/chat/completions` curl returns a sensible answer
- [ ] `nvpmodel -m 0` + `jetson_clocks` run before any benchmark
- [ ] Dense model benchmarked (token/s, TTFT) — then switch to MoE

---

## Benchmarking plan (milestone 5)

**Stabilize clocks first** (always, before every run):
```bash
sudo nvpmodel -m 0   # MAXN power mode — all CPU + GPU cores unlocked
sudo jetson_clocks   # lock clocks at maximum, prevent mid-run throttle
```

---

### Metrics tooling — what's available and where

#### Already on the Jetson host (ships with L4T — no install needed)

**`tegrastats`** — the primary hardware metrics stream for Jetson.
Outputs a line every interval with: GPU/CPU load, RAM used, die temperatures,
and **power draw per rail** (VDD_CPU, VDD_GPU, VDD_SOC, total board power).
Power is the most important edge-device metric and only `tegrastats` gives it.

```bash
# Stream at 500 ms intervals, log to file (run this in a separate SSH session
# WHILE the benchmark is running in another session)
tegrastats --interval 500 | tee tegrastats_$(date +%Y%m%d_%H%M%S).log

# Sample output line (one line per interval):
# RAM 3212/62833MB (lfb 13x4MB) SWAP 0/31416MB (cached 0MB)
# CPU [4%@729,3%@729,2%@729,2%@729,off,off,off,off] EMC_FREQ 3%@3199
# GR3D_FREQ 78%@1300 VDD_CPU_CV 1253mW VDD_SOC_CV 1876mW VDD_GPU_CV 4912mW
# Tboard@40C Tdiode@41.5C TJ@46.5C
```

Key fields:
| Field | Meaning |
|-------|---------|
| `GR3D_FREQ X%@YMHz` | GPU utilization % and current clock |
| `RAM X/YMB` | Host RAM used / total (unified memory — same pool as GPU) |
| `VDD_GPU_CV XmW` | GPU power draw in milliwatts |
| `VDD_CPU_CV XmW` | CPU power draw |
| `VDD_SOC_CV XmW` | SoC power draw |
| `TJ@XC` | Junction temperature (the thermal throttle reference) |

**`nvidia-smi`** — confirmed working (`nvidia-smi dmon` streams GPU stats too, but `tegrastats` is preferred on Jetson because it adds power and thermal data that `nvidia-smi` omits).

---

#### Inside the vLLM container (available once Docker is installed)

**`vllm bench serve`** — the built-in LLM serving benchmark.
Sends concurrent requests to the running vLLM HTTP server and measures latency/throughput.

```bash
# Run from inside the container (or via docker exec) while the server is up on :8000
# Typical run: 100 prompts, input ~512 tokens, output ~128 tokens, 10 concurrent users
python -m vllm.entrypoints.benchmark_serving \
  --backend vllm \
  --host 0.0.0.0 \
  --port 8000 \
  --model qwen2.5-7b \
  --dataset-name random \
  --num-prompts 100 \
  --input-len 512 \
  --output-len 128 \
  --request-rate 10 \
  --max-concurrency 10
```

Output reports:
- **TTFT** (time-to-first-token, mean + percentiles)
- **TPOT / ITL** (inter-token latency)
- **Throughput** (output tok/s, request/s)
- **E2E latency** (total request time)

Concurrency sweep (compare models at matched load):
```bash
for C in 1 4 8 16; do
  echo "--- concurrency $C ---"
  python -m vllm.entrypoints.benchmark_serving \
    --model qwen2.5-7b --num-prompts 100 \
    --input-len 512 --output-len 128 \
    --max-concurrency $C
done
```

---

#### Nice-to-have: `jtop` (install once on the host)

`jtop` (from the `jetson-stats` package) is a terminal dashboard that combines GPU, CPU,
RAM, clocks, power, and temperature into one readable screen. Easier to eyeball during
a run than raw `tegrastats` log lines.

```bash
# One-time install on the Jetson host (outside Docker)
sudo pip3 install jetson-stats
sudo systemctl restart jtop.service   # starts a background stats collector

# Run the dashboard (Ctrl+C to exit)
jtop
```

`jtop` also has a Python API for programmatic metric collection if you want to log to CSV.

---

#### Optional advanced: `genai-perf` (NVIDIA Triton toolkit)

`genai-perf` gives richer output — concurrency sweeps, p50/p90/p99 latencies, CSV/JSON export,
and profile-by-phase graphs. It requires the Triton `perf_analyzer` binary.

```bash
# Install inside the vLLM container (or a separate Triton container)
pip install genai-perf

genai-perf profile \
  -m qwen2.5-7b \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --endpoint-type chat \
  --url http://localhost:8000 \
  --input-tokens-mean 512 \
  --output-tokens-mean 128 \
  --concurrency 1 4 8 16
```

This is optional for milestone 5 — `vllm bench serve` + `tegrastats` covers the core metrics.

---

### Running both tools in parallel (the standard workflow)

Open **two SSH sessions** to the Jetson simultaneously:

**Terminal 1 — start hardware logging:**
```bash
tegrastats --interval 500 | tee run_dense_$(date +%Y%m%d_%H%M%S).log
```

**Terminal 2 — run the LLM benchmark (inside the container):**
```bash
docker exec -it <vllm-container-name> \
  python -m vllm.entrypoints.benchmark_serving \
    --model qwen2.5-7b --num-prompts 200 \
    --input-len 512 --output-len 128 \
    --max-concurrency 8 \
  | tee bench_dense_$(date +%Y%m%d_%H%M%S).log
```

Stop `tegrastats` (Ctrl+C) when the benchmark finishes. The two log files together give you:
- LLM latency/throughput from `bench_*.log`
- GPU load, power draw, and temperature over the same time window from `run_*.log`

---

### Metrics summary table (what to collect per model)

| Metric | Tool | Target (Qwen3-30B-A3B reference) |
|--------|------|----------------------------------|
| TTFT (p50 / p99) | `vllm bench serve` | < 500 ms / < 1 s at C=1 |
| TPOT / ITL | `vllm bench serve` | ~13–25 ms/token (41–77 tok/s) |
| Output tok/s | `vllm bench serve` | ~41 tok/s baseline, ~77 NVIDIA-tuned |
| GPU utilization | `tegrastats` `GR3D_FREQ` | > 80% during decode |
| GPU power (W) | `tegrastats` `VDD_GPU_CV` | log for efficiency ratio |
| Peak RAM | `tegrastats` `RAM` | should stay well under 61 GB |
| Peak temperature | `tegrastats` `TJ` | watch for thermal throttle (> 90°C) |

---

## ▶️ Next steps to resume

**Milestone 1 (devcontainer acquaintance) — ✅ DONE 2026-06-14.** Remote-SSH + dev container +
GPU passthrough all working on the Jetson (see "Milestone 1 status" above).

**Milestone 2 (flash) — 🟡 OS DONE 2026-06-23.** JetPack 6.2.2 / L4T R36.5.0 / Ubuntu 22.04.5
flashed onto the **eMMC** (46 G free). `nvidia-smi` + SSH work. **Remaining:** install **Docker +
nvidia-container-toolkit** on the host (CUDA comes from the container — **no host `nvidia-jetpack`**).

**Decision locked 2026-06-22:** user wants **vLLM** (not llama.cpp). Sequence is dense-first.
**Decision 2026-06-23:** CUDA via container, **not** host — host only needs Docker + container-toolkit.

**Now (finish milestone 2, then serve):**
1. **Install Docker + GPU runtime on the Jetson (only host install):**
   `sudo apt install docker.io nvidia-container-toolkit` → `usermod -aG docker sree` →
   `nvidia-ctk runtime configure --runtime=docker --set-as-default` → restart docker → log out/in.
   Verify GPU in a container: `docker run --rm --runtime=nvidia dustynv/vllm:r36.4.0 nvidia-smi`.
   **Skip host CUDA / `nvidia-jetpack`** — the container ships CUDA 12.6.
2. **Update the dev container for JP6:** switch [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)
   base from the `r35.2.1` l4t-pytorch tag to an **r36.4 / vLLM-ready** image
   (e.g. `dustynv/vllm:0.9.x-r36.4-cu128-24.04`); reopen in container; re-run the GPU sanity check.
3. **Milestone 3 — serve the DENSE model first:** Qwen2.5-7B-Instruct (AWQ) with vLLM; smoke-test the
   OpenAI-compatible API.
4. **Milestone 4 — switch to MoE:** Qwen3-30B-A3B (AWQ); then benchmark (milestone 5).

_NVMe note: never purchased; flash went to eMMC. Add an NVMe later only if the MoE phase + HF cache
gets tight on the 46 G eMMC._
