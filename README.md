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
