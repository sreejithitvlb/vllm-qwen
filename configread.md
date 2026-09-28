# Config review

Reference for vLLM's `SchedulerConfig` and `CacheConfig` options, and whether each matters for this
project (Qwen2.5-7B-Instruct-AWQ, text-only, `max_model_len: 4096`, running on a Jetson AGX Orin
64 GB, JetPack 6 / L4T R36.5).

`main.py` passes every uncommented key in `config/*.yaml` to vLLM, so use underscores in YAML
(`async_scheduling: true`), not the CLI's dashes.

None of these options are required; all have defaults.

- [SchedulerConfig](#schedulerconfig): batching and throughput
- [CacheConfig](#cacheconfig): KV cache and memory
- [AttentionConfig](#attentionconfig): attention kernels

Booleans: `main.py` passes `true` as `--<flag>` and `false` as `--no-<flag>`, so `false` can turn
off something vLLM enables by default (e.g. `enable_chunked_prefill: false`).

# SchedulerConfig

Settings live in [config/scheduler.yaml](config/scheduler.yaml).

## Current settings

```yaml
max_num_seqs: 32
max_num_batched_tokens: 2048
enable_chunked_prefill: true
```

Constraints to keep in mind:
- `max_num_batched_tokens` must be ≥ `max_num_seqs`.
- If `enable_chunked_prefill` is `false`, `max_num_batched_tokens` must be ≥ `max_model_len` (4096),
  or vLLM refuses to start.

## Summary

| YAML key | Default | Status | Recommendation |
|---|---|---|---|
| `max_num_seqs` | auto | Set (32) | Keep |
| `max_num_batched_tokens` | auto | Set (2048) | Keep; tune with benchmarks |
| `enable_chunked_prefill` | auto (on in V1) | Set (true) | Keep |
| `async_scheduling` | auto | Not set | Try it and benchmark |
| `stream_interval` | 1 | Not set | Raise only if throughput beats smooth streaming |
| `long_prefill_token_threshold` | 0 | Not set | Only if short prompts wait behind long ones |
| `max_num_partial_prefills` | 1 | Not set | Same as above |
| `max_long_partial_prefills` | 1 | Not set | Same as above |
| `scheduler_reserve_full_isl` | true | Not set | Leave the default |
| `scheduling_policy` | fcfs | Not set | Leave the default |
| `scheduler_cls` | built-in | Not set | Leave the default |
| `disable_chunked_mm_input` | false | Not set | Not relevant (text-only model) |
| `disable_hybrid_kv_cache_manager` | auto | Not set | Not relevant (no sliding window) |

## Core settings (in use)

### `max_num_seqs`
Maximum number of sequences processed in a single iteration, i.e. how many requests can be in the
batch at once. Higher values mean more concurrency, but each sequence needs KV cache memory.

### `max_num_batched_tokens`
Maximum number of tokens processed in a single scheduler step, across all sequences. Accepts
human-readable values like `2k`.
- Lower → smoother token generation (lower inter-token latency).
- Higher → faster time to first token and higher total throughput, at the cost of more memory per step.

### `enable_chunked_prefill`
Splits long prompts into chunks that fit the remaining `max_num_batched_tokens`, so one long prompt
doesn't stall generation for everyone else. Already on by default in recent vLLM (V1 engine).

## Worth trying later

### `async_scheduling`
Schedules the next step while the GPU is still running the current one, which avoids gaps in GPU
utilization and can improve latency and throughput. Try `true` and benchmark; if startup fails or
output looks wrong, remove it.

### `stream_interval`
How many tokens to buffer before sending a streaming update. `1` sends every token immediately
(smoothest). A larger value (e.g. `10`) reduces host overhead and may raise throughput, but the
stream looks choppier.

### Partial-prefill tuning
These three work together and only matter with chunked prefill and a mix of long and short prompts
arriving concurrently.

- **`long_prefill_token_threshold`**: a prompt longer than this counts as "long".
- **`max_num_partial_prefills`**: how many prompts can be partially prefilled at the same time.
- **`max_long_partial_prefills`**: how many *long* prompts can be prefilled at the same time. Setting
  this lower than `max_num_partial_prefills` lets short prompts jump ahead of long ones, improving
  their latency.

Example if short prompts are getting stuck behind long ones:

```yaml
long_prefill_token_threshold: 1024
max_num_partial_prefills: 4
max_long_partial_prefills: 1
```

## Leave at default

### `scheduler_reserve_full_isl`
When `true` (default), the scheduler checks that a request's *full* prompt fits in the KV cache
before admitting it, not just the first chunk. This prevents over-admission and KV cache thrashing
with chunked prefill, which matters on a memory-limited device.

### `scheduling_policy`
- `fcfs` (default): requests are handled in arrival order.
- `priority`: requests are handled by a priority value sent with each request (lower = earlier), with
  arrival time breaking ties. Only useful if clients actually send priorities.

### `scheduler_cls`
The scheduler class to use. Default is `vllm.v1.core.sched.scheduler.Scheduler`. Only change this to
plug in a custom scheduler.

## Not relevant to this model

### `disable_chunked_mm_input`
For multimodal models: stops an image from being split across scheduler steps. Qwen2.5-7B-Instruct is
text-only.

### `disable_hybrid_kv_cache_manager`
For models that mix full attention and sliding-window attention: forces the same KV cache size for
all layers. Qwen2.5 doesn't use sliding window by default.

# CacheConfig

Settings live in [config/cache.yaml](config/cache.yaml). This is usually the most important group to
tune, because it controls how much memory vLLM takes and how much of it goes to the KV cache.

## Current settings

```yaml
gpu_memory_utilization: 0.83
enable_prefix_caching: true
# kv_cache_dtype: auto          # Data type: auto, float16, bfloat16, float8
# block_size: 16
```

Note: the `kv_cache_dtype` comment lists `float8`, which vLLM doesn't accept. Valid values are
`auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, and in some versions `bfloat16`.

## Why Orin is different

On Orin the CPU and GPU share the same physical memory (64 GB here). Anything vLLM claims is taken
away from the OS and every other process, and options that move data "to CPU memory" don't free
anything, because it's the same memory.

## Summary

| YAML key | Default | Status | Recommendation |
|---|---|---|---|
| `gpu_memory_utilization` | 0.9 | Set (0.83) | Lower to 0.3–0.4 |
| `enable_prefix_caching` | on in V1 | Set (true) | Keep |
| `kv_cache_dtype` | auto | Not set | Leave at `auto` |
| `block_size` | 16 | Not set | Leave the default |
| `swap_space` | 4 (GiB) | Not set | Don't set on Orin |
| `cpu_offload_gb` | 0 | Not set | Don't set on Orin |

## Settings

### `gpu_memory_utilization`
Fraction of GPU memory vLLM may use for weights, activations and KV cache. Default 0.9. Common
advice for Orin is 0.6–0.75, because of the shared memory.

For this model, even that is more than needed. Rough memory budget:

| Part | Size |
|---|---|
| Model weights (7B, 4-bit AWQ) | ~5 GB |
| KV cache: ~57 KB/token × 32 seqs × 4096 tokens (131k tokens) | ~7.5 GB |
| Runtime overhead | a few GB |
| **Total** | **~15 GB** |

At 0.83, vLLM claims about 53 GB and leaves about 11 GB for the OS and everything else, which can
push the system into swap. **0.3–0.4** (19–26 GB) covers the model with headroom. If you raise
`max_num_seqs` or `max_model_len`, the KV cache grows proportionally, so recheck this.

### `enable_prefix_caching`
Reuses the KV cache for shared prompt prefixes, such as a long system prompt sent with every chat
request. A big win for chatbots, and on by default in recent vLLM. Keep it.

### `kv_cache_dtype`
Data type of the KV cache. `fp8` roughly halves KV memory compared to fp16, where it's supported.
Orin's GPU (Ampere) has no native FP8 hardware, so fp8 may be unsupported, slower, or slightly less
accurate depending on the attention backend. With the memory budget above there's no need for the
savings, so leave it at `auto`.

### `block_size`
Number of tokens per KV cache block. Default 16. Leave it unless a specific attention backend
requires a different value.

### `swap_space`
CPU memory (GiB) reserved for swapping out paused requests. On Orin this is the same physical memory,
so it only wastes RAM. Recent vLLM (V1 engine) recomputes paused requests instead of swapping, so it
is likely ignored anyway. Don't set it.

### `cpu_offload_gb`
Moves part of the model weights to CPU memory to save GPU memory. On shared memory it saves nothing
and slows every step down. Don't set it.

# AttentionConfig

Chooses which attention kernels vLLM runs. Settings would go in
[config/attention.yaml](config/attention.yaml) (all commented out for now).

**Required params: none.** vLLM auto-selects a backend from the GPU, model and dtype, and every
option below has a default. The file exists only so the options are documented next to the others.

Unverified against this build: written from vLLM's AttentionConfig, not from the 0.20.0 Jetson
image. Before relying on a flag, run this in the container and compare:

```bash
vllm serve --help=AttentionConfig
```

## Why Orin limits the choices

Orin's GPU is Ampere (compute capability 8.7). Kernels built for Hopper (FlashAttention 3) or
Blackwell (TRT-LLM, some cuDNN paths) don't run on it, and the Jetson build may not include
FlashInfer. Triton is also not installed in this image (see [kernels_readme.md](kernels_readme.md)),
so Triton-based backends are out too. In practice the backend will be FlashAttention 2.

## Summary

| YAML key | Default | Recommendation |
|---|---|---|
| `attention_backend` | auto | Leave unset; set only to debug or compare backends |
| `flash_attn_version` | auto | Leave unset (Orin only supports 2) |
| `use_prefill_decode_attention` | false | Leave unset |
| `flash_attn_max_num_splits_for_cuda_graph` | 32 | Leave unset |
| `use_cudnn_prefill` | false | Not relevant (Blackwell / MLA models) |
| `use_trtllm_attention` | auto | Not relevant (Blackwell only) |
| `use_trtllm_ragged_deepseek_prefill` | false | Not relevant (DeepSeek on Blackwell) |
| `disable_flashinfer_prefill` | false | Not relevant unless FlashInfer is in use |
| `disable_flashinfer_q_quantization` | false | Not relevant unless FlashInfer is in use |

## Settings

### `attention_backend`
Forces a specific backend instead of auto-selection, e.g. `FLASH_ATTN`, `TRITON_ATTN`,
`FLASHINFER`. Replaces the older `VLLM_ATTENTION_BACKEND` environment variable. The startup log
prints which backend was picked; only set this if that choice fails or you want to benchmark an
alternative. `TRITON_ATTN` needs Triton, which this image doesn't have.

### `flash_attn_version`
Forces FlashAttention 2 or 3. Version 3 needs Hopper, so on Orin the only valid value is `2`, which
auto-selection already picks.

### `use_prefill_decode_attention`
Uses separate prefill and decode kernels instead of one unified kernel, for the Triton backend.
Not usable here, since Triton isn't installed.

### `flash_attn_max_num_splits_for_cuda_graph`
Caps how many ways FlashAttention splits the work during decode when CUDA graphs are used. A
low-level performance knob; leave it.

### `use_cudnn_prefill`, `use_trtllm_attention`, `use_trtllm_ragged_deepseek_prefill`
Alternative kernels for Blackwell GPUs and MLA models like DeepSeek. They don't apply to Orin or
Qwen2.5.

### `disable_flashinfer_prefill`, `disable_flashinfer_q_quantization`
Turn off parts of the FlashInfer backend. Only matter if FlashInfer is the active backend.

# How to verify a change

1. Restart the server and check the startup log for errors or warnings about these values.
2. Send one long prompt (~3–4k tokens) together with a few short ones and confirm nothing stalls.
3. Compare time to first token and tokens/sec before and after the change.
4. For cache changes, check memory while the server runs (`free -h` or `tegrastats`) and confirm
   the system isn't swapping.
