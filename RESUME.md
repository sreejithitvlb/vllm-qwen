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

## Benchmarking plan (milestone 5)
- **Tools:** `vllm bench serve` (built-in) and/or NVIDIA **`genai-perf`**.
- **Metrics:** TTFT (time-to-first-token), TPOT / ITL (inter-token latency), output tok/s,
  request throughput, and concurrency scaling.
- **Stabilize clocks first:** `sudo nvpmodel -m 0` (MAXN) + `sudo jetson_clocks` before runs.
- Compare shortlist models head-to-head at matched `max-model-len` / concurrency.

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
