# Apple GPU model enablement — status and integration blockers

Goal: run full models (Llama-3 8B, FLUX.2) on macOS (Apple silicon / Metal,
M5+). This is the live tracker of what has landed and the integration blockers
that remain between here and a model generating correctly on Metal. Update it as
blockers resolve.

## Approach

MAX's Metal runtime is complete (the MLRT Metal device context; compute
capabilities M1–M5), and op→kernel dispatch silently falls back to generic Mojo
kernels where no Apple path exists. So enablement is two things: hardware-MMA
kernels for the hot ops, and closing the ops that either don't run or are wrong
on Metal. Many portable-Mojo ops already work; the open gaps are tracked below.

## Status

| Area                              | Status    | PR     | Notes                      |
|-----------------------------------|-----------|--------|----------------------------|
| matmul (M5 16×16 MMA + M1–M4 8×8) | landed    | #89441 | `AppleM5MatMul` + split-K  |
| FA prefill (MMA) + paged-KV       | in review | #89478 | fires in a real 8B graph   |
| decode (attention-sink fix)       | in review | #89478 | existing decode + sink     |
| conv2d (im2col + MMA)             | in review | #89479 | bf16-only; mem-bound guard |

## Validated end-to-end

Llama-3.1-8B (BF16) compiles and runs all 32 prefill layers on the M5, with the
Apple FA prefill kernel firing (`flash_attention` on `metal:0`, not the naive
fallback). Full text generation is blocked by the sampler (B1 + B2 below).

## Integration blockers (open)

Ordered by impact on running a full model end to end.

### B1 — sampler threadgroup overflow (blocks decode compile)

- Symptom: the `fused_token_sampling` Metal pipeline-state fails to compile —
  threadgroup memory 32800 > the 32768 Metal limit. The first decode step
  crashes before any token prints.
- Root cause: the two-stage top-k `_topk_stage1` (`nn/topk.mojo:937`) sizes
  `topk_sram` as `_APPLE_STATIC_SHMEM_MAX_COUNT[TopK_2[f32]]` = exactly 32768
  bytes (the whole budget), then `_block_reduce_topk` needs ~32 more.
- Fix (specified, not yet applied): Apple-gated headroom in the stage-1/2
  `unsafe_stack_allocation` (`topk.mojo:957`/`1239` + a helper in
  `normalization.mojo`); NVIDIA/AMD take the dynamic-shmem branch and are
  untouched.
- Severity: high — gates decode *compilation* on Metal. Owner: TBD.

### B2 — sampler correctness on Apple (FlashInfer top-k) — MOCO-2673

- Symptom: the live decode path (`top_k` → FlashInfer `topk_sampling_from_prob`)
  samples *outside* the top-K set on Metal. `test_topk_gpu_fi.mojo` is already
  apple-excluded under this ticket.
- Severity: high — even once B1 lets decode compile, this gates *correct*
  generation. Owner: TBD (MOCO-2673).

### B3 — quantized matmul (Q4_K / GGUF)

- No Apple path (`qmatmul_gpu` is NVIDIA-only), so GGUF/Q4_K models fall back
  and fail. Use BF16 weights until an Apple dequant + MMA path exists.
- Severity: medium (BF16 sidesteps it). Deferred.

### B4 — fp16 conv2d

- The im2col conv dispatcher self-gates bf16-only, so fp16 convs fall to the
  naive kernel on Apple. Severity: low. Follow-on to #89479.

### B5 — fused inline-im2col conv (perf)

- The materialize-im2col path loses to naive on memory-bound convs (low output
  channels, high resolution — guarded to naive in #89479). A fused
  DRAM→register im2col loader would recover that regime. Severity: low (perf).

## References

- Design: [`DESIGN.md`](DESIGN.md) (this directory).
- PRs: matmul #89441 (landed); attention #89478; conv2d #89479.
- KB: `kernels/apple-m5-matmul`, `kernels/apple-conv2d-im2col`,
  `patterns/apple-m5-gpu-performance-considerations`,
  `patterns/apple-fa-register-resident-p-fragment`,
  `patterns/apple-paged-kv-prefill-per-sub-tile`.
