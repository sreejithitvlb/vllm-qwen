# CUDA Kernels in vLLM (Jetson AGX Orin Reference)

## Attention Kernels
| Kernel | Purpose |
|--------|---------|
| FlashAttention | Fast attention computation, reduces memory reads |
| PagedAttention | vLLM's own kernel for block-based KV cache management |
| FlashInfer | Newer, faster alternative to FlashAttention |

## Matrix Multiplication (GEMM) Kernels
| Kernel | Purpose |
|--------|---------|
| Marlin | Fast INT4/INT8 GEMM — used by `awq_marlin` on Ampere |
| cuBLAS | NVIDIA's standard GEMM library for FP16/BF16 |
| CUTLASS | NVIDIA's template library for custom GEMM ops |

## Quantization Kernels
| Kernel | Purpose |
|--------|---------|
| AWQ | Original INT4 kernel — slower |
| AWQ Marlin | Same INT4 weights, faster Marlin kernel — use this on Jetson |
| GPTQ | For GPTQ quantized models |
| FP8 | Hopper/Ada only — not available on Jetson Ampere |

## Sampling Kernels
| Kernel | Purpose |
|--------|---------|
| Top-p / Top-k | Standard sampling methods |
| Rejection sampler | Used for speculative decoding |

## Jetson-Specific Notes
- **Ampere GPU** — supports INT4 (AWQ/GPTQ) and FP16/BF16; no FP8
- **awq_marlin** — best quantization choice on Jetson; same weights as AWQ, faster kernel
- **Triton not installed** — vLLM falls back to pre-compiled CUDA kernels; not a blocker
- **PagedAttention + Marlin GEMM** — the two most impactful kernels for this setup
