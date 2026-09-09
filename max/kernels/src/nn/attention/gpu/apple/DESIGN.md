# Apple GPU attention backend — design

Status: draft (in progress). Last updated: 2026-06-21.

## Scope

Design for the Apple silicon (Metal) GPU attention kernels: the landed
flash-attention **decode** path and the in-progress bf16 flash-attention
(FA2) **prefill** kernel for M5. Both build on the `MmaOpApple` 16×16
simdgroup-MMA foundation shared with the Apple matmul.

Out of scope (tracked separately): quantized (Q4_K) attention/matmul,
convolution (the diffusion VAE), and the M1–M4 8×8 path. See the roadmap
section.

## Status

Live status and the integration blockers between here and a model generating
on Metal are tracked in [`MODEL_ENABLEMENT.md`](MODEL_ENABLEMENT.md). In brief:
the matmul (#89441) landed; the FA prefill + paged-KV + decode sink fix are in
review (#89478) and the prefill fires in a real 8B graph; conv2d is in review
(#89479). Full generation is currently blocked by the sampler — a 32-byte
threadgroup overflow plus the FlashInfer top-k correctness bug (MOCO-2673).

## Context

MAX can already execute graphs on the Apple Metal GPU (the MLRT Metal
device context is complete), and op→kernel dispatch silently falls back
to generic Mojo kernels where no Apple path exists. For attention, the
decode path has a dedicated Apple kernel, but **prefill falls back to
`mha_gpu_naive`** — correct but slow. Prefill is the highest-leverage
attention gap: it gates prompt processing for LLMs and is the *entire*
compute of a diffusion transformer (all image tokens run in parallel).

## Hardware model (M5)

The rules that shape every Apple GPU kernel (see the perf KB entry):

- Hardware **16×16 simdgroup MMA** (`MmaOpApple`, Metal 4 / `cc == 5`),
  bf16/fp16/fp32 in, fp32 accumulate; M1–M4 have only an 8×8 unit.
- Simdgroup = 32 lanes; each lane owns an 8-element **bit-scatter**
  fragment of a 16×16 tile (2 rows × 4 cols). Owned at the SIMD layer,
  not expressible via `distribute`.
- **No shared-memory staging** — it degrades Apple matmul. Operands load
  DRAM→register directly. There is no `s_waitcnt` / `schedule_barrier`
  overlap machinery to pipeline.
- Scalar ALU; do index math in `Int32`.

## What already exists

- **Matmul** — `AppleM5MatMul` (`linalg/matmul/gpu/apple/`), the
  `MmaOpApple` foundation this kernel reuses.
- **Attention decode** — `naive_fa_decode.mojo`: warp-centric, one query
  row, so Q·Kᵀ is a GEMV reduced with `air.simd_sum` (no MMA);
  reduction-free P·V; online softmax; paged KV via `MHAOperand`; no
  barriers, no shared memory.
- **Dispatch** — `flash_attention_dispatch` routes Apple decode under
  `is_token_generation`; the `not is_token_generation` (prefill) branch
  currently falls through to `mha_gpu_naive`.

## Prefill kernel design

Prefill differs from decode *in kind*: many query rows turn Q·Kᵀ and P·V
into real matrix×matrix products, so the kernel is two `MmaOpApple` GEMMs
sandwiching an online softmax — i.e. it reuses the matmul directly. Per
query-row tile (owned by one simdgroup), iterating KV tiles online:

```text
init: m = -inf (or sink), l = 0, O = 0
for each KV tile:
    QK = MmaOpApple(Q, K, transpose_b=True)   # Sq×Sk fp32 score tile
    apply MHAMask(QK) at absolute (b, head, q, k)
    softmax.full(O, QK)                        # m,l update + rescale O
    P  = QK_softmaxed.cast[bf16]
    O += MmaOpApple(P, V)                       # no transpose
epilogue: O /= l; cast to out dtype; masked store
```

Tiling: `Sq = num_m_mmas·16`, KV-tile width `Sk = num_n_mmas·16`; start
at `2×2` (the 32×32 simdgroup subtile already tuned for matmul). KV tiles
load DRAM→register (no SMEM). Hoist the loop-invariant tile base out of
the KV loop (the matmul slab-hoist lesson).

## Online softmax: the `Softmax` struct

A register-resident state object holding running `(m, l)` plus the
per-fragment row-max/row-sum scratch, with the sequence
`qk_max → exp → qk_sum → correction → rescale O → update m,l`. The
algebra ports ~as-is from the AMD `Softmax` struct; the load-bearing
change is generalizing `frag_num_rows` from 1 (AMD's row-vector
fragment) to **2** (the M5 lane owns two rows). Built so prefill and a
later-refactored decode can share it.

## Fragment row-reduction (the crux) — validated

The online softmax needs a per-**row** max/sum over the score tile, but a
row is scattered across the MMA fragment. Layout: lane owns rows
`{row_lo, row_lo+8}` × cols `{col_base..col_base+3}` with
`row_lo = 4·b4 + 2·b2 + b1`, `col_base = 8·b3 + 4·b0`. The 4 lanes
sharing a row differ only in bits `b0, b3`, i.e. they sit at XOR offsets
`{1, 8}`. So the reduction is: per-lane reduce the 4 cols of each
row-half, then a 2-step `shuffle_xor` butterfly over masks `{1, 8}`. fp32
shuffles at width 1 (the only width the Apple shuffle intrinsic allows
for non-half dtypes). No SMEM, no barriers.

This is the one piece that cannot be lifted from the decode kernel
(decode's `air.simd_sum` reduces all 32 lanes — correct for a GEMV, wrong
here because it mixes rows) nor copied from AMD (`lane_group_max` assumes
a contiguous strided lane group; the Apple col-lanes are an XOR group).
**Validated in isolation** (`test/gpu/linalg/test_apple_fa_frag_reduce.mojo`,
PASS on M5) against a host reference before wiring the full kernel — this
is where a silent fragment-layout bug would otherwise hide.

For `num_n_mmas > 1` the full-row reduction first combines the per-row
partials across the `ni` fragments (a register combine, same lane holds
the matching cols of each), then runs the butterfly once.

## AMD → Apple port mapping

The algorithm ports; the AMD data-movement and scheduling do not.

Ports ~as-is:

- The online-softmax algebra and the `Softmax` state-object shape.
- `MHAMask` masking at absolute coords (the decode contract).
- The exp-scaled precision trick (subtract the unscaled max so the max
  element exits at exactly 0).
- Sink-as-init-state (pre-seed `m`/`l` so the body needs no sink branch).
- Mask-derived iteration bounds and full-mask tile skips.

Rewrite for Apple:

- The fragment row-reduction (above): `frag_num_rows = 2`; MFMA →
  16×16 bit-scatter; `lane_group_max` → `shuffle_xor` butterfly `{1, 8}`.
- Q·Kᵀ and P·V via `MmaOpApple` (`transpose_b=True` on Q·Kᵀ).
- P cast fp32→bf16 between softmax and P·V.
- Epilogue `O / l` + the odd-stride vectorized store from the matmul.

Drop entirely:

- LDS / `KVBuffer` staging and double-buffer slot ping-pong → load KV
  tiles DRAM→register.
- `s_waitcnt` / `schedule_barrier` / iglp scheduling — no M5 analog.
- The cross-warp SMEM softmax scratch — one simdgroup owns its row's
  full KV span, so there is no cross-warp stage.
- The `exp2`-no-flush workaround (AMD-specific) — use plain `exp` with a
  `NEG_INF` sentinel, as the decode kernel already does on M5.

## Integration

- Add an Apple branch in `flash_attention_dispatch`'s
  `if not is_token_generation:` path, gated on
  `has_apple_gpu_accelerator()` with an env opt-out mirroring the decode
  flag; otherwise keep falling through to `mha_gpu_naive`.
- New file `apple/fa_prefill.mojo`, sibling to `naive_fa_decode.mojo`,
  with a launcher signature mirroring `mha_gpu_naive` (the
  `MHAOperand` / `MHAMask` / ragged / paged-KV contract and the
  `q_offset` / `score_row` / `cache_len` math lift from decode).

## Test plan

1. Fragment row-reduction unit test — **done**, PASS on M5.
2. End-to-end prefill vs an fp32 host attention reference: NN/NT,
   ragged M/N, causal and sliding-window masks, fp16/bf16 in,
   head dims that are multiples of 16 up to 256.
3. Confirm the landed decode, matmul (55 cases), and split-K suites are
   unaffected.

## Performance expectations

bf16/fp16 in, fp32 accumulation. v1 target is correctness plus clearly
beating the `mha_gpu_naive` fallback, not peak. Large prefill is
compute-bound; the matmul's measured ~50–58 TFLOPS (fp16/bf16, M5 Max)
is the realistic ceiling reference. Tuning (tile sizes, accumulator
count) comes after correctness.

Measured (M5 Max, `bench_apple_fa_prefill`, cache-busted, BF16 32q/8kv
d=128). The wide-threadgroup kernel (16 simdgroups, narrow 1x2 per-simdgroup
tile, no SMEM) sustains ~37-41 TFLOP/s full-work (NullMask) and ~20-50x the
`mha_gpu_naive` fallback -- the "beat naive" bar is met; see the
wide-threadgroup section below for the geometry/tile rationale.

The KV loop skips fully-masked tiles via the mask-derived
`status == FULL_MASK` check (causal takes a monotonic early-exit; other
masks skip per-tile), cutting causal-prefill wall-clock ~2x at large seq.
The bench credits the full square (`2*B*H*seq*num_keys*depth`), so on that
convention the *reported* causal throughput roughly doubles -- this reflects
the removed masked work, NOT a faster MMA rate; the honest rate is the
full-work NullMask number above. See the tile-tuning rationale in
`fa_prefill.mojo`.

## Wide-threadgroup launch (no SMEM)

`fa_prefill_apple` launches `num_simdgroups=16` simdgroups in one threadgroup
(`block_dim = 16*32`, 256 query rows), each independently owning its own 16-row
query block and streaming the KV range from DRAM. There is no threadgroup
memory and no `barrier()`: the simdgroups are co-resident only so they share KV
reads through the L2 -- when all 16 read the same KV tile, the L2 serves the
reuse, so a single DRAM-resident kernel needs no software staging.

We did try classic FlashAttention SMEM staging (stage each KV tile DRAM->SMEM
once per threadgroup, feed all simdgroups from SMEM). A controlled A/B -- the
*same* wide-threadgroup geometry, staging on vs off -- showed the staging was a
net loss at every shape: two `barrier()`s plus a cooperative copy to reproduce
what the L2 already does for co-resident reads. The apparent staged win was
entirely the threadgroup GEOMETRY (256 co-resident query rows -> L2 reuse), not
the SMEM. So the kernel keeps the geometry and drops the staging.

Measured (M5 Max, BF16 32q/8kv, d=128, b1; full-work NullMask TFLOP/s): the
wide-threadgroup kernel beats the old `block_dim=32` single-simdgroup base at
every seq -- ~1.5x at seq4096, ~2x at seq8192, ~3x at seq16384 (NullMask
seq16384 ~37-41 TFLOP/s, the honest full-work rate) -- and beats the SMEM-staged
variant everywhere it was tried. The narrow per-simdgroup tile (`NUM_M_MMAS=1,
NUM_N_MMAS=2`, SK=32) is forced by the 16-simdgroup register budget: widening
the score tile collapses occupancy (a single-simdgroup kernel would want the
opposite, wide 1x8, but the wide threadgroup beats it). This upholds the
matmul-derived "no SMEM / no barrier on Apple" rule for attention too -- it just
needed the wide-threadgroup geometry to realize the L2 reuse.

Dispatch: `fa_prefill_apple` is the single Apple-prefill kernel
(`not is_token_generation`), gated by the comptime KV/depth checks and the
`MODULAR_ENABLE_APPLE_FA_PREFILL` opt-out -- no separate staged path or seq
threshold.

## Open questions / risks

- The `frag_num_rows = 2` generalization ripples through every per-row
  loop in the ported `Softmax` (AMD hard-asserts a row-vector fragment).
- Masking needs the fragment `rb`/`cb` → absolute `(b, head, q, k)`
  mapping; reuse the matmul epilogue's coord mapping.
- Tile shape (`Sq`/`Sk`/`num_m_mmas`/`num_n_mmas`) starts at 2×2; revisit
  after the end-to-end test passes.

## Roadmap (Apple GPU backend, beyond this kernel)

1. Conv2D — an im2col `TileLoader` feeding `AppleM5MatMul` (the FLUX VAE
   blocker); the heavy compute is the matmul we already have.
2. Quantized (Q4_K) matmul/attention — dequant-in-register + MMA (the
   ggml-metal technique), once bf16 paths land.

## References

Source:

- `nn/attention/gpu/apple/naive_fa_decode.mojo` (decode; the contract +
  Apple idioms)
- `nn/attention/gpu/mha.mojo` (`flash_attention_dispatch`)
- `linalg/matmul/gpu/apple/matmul_kernel.mojo` and
  `linalg/arch/apple/mma.mojo` (`MmaOpApple`)
- `nn/attention/gpu/amd_structured/{softmax,attention,mha_prefill}.mojo`
  (the structural reference)

KB entries:

- `kernels/{mha-amd, attention-amd-core, attention-amd-softmax,
  apple-m5-matmul}`
- `patterns/{apple-m5-gpu-performance-considerations,
  aiter-non-materialized-fp32-softmax-register-architecture,
  amd-softmax-exp-scaled-precision-trick, amd-attention-sink-as-init-state,
  hoist-loop-invariant-tile-base-out-of-hot-loop,
  tiletensor-vectorized-store-alignment-odd-stride,
  cpp-to-mojo-idiomatic-porting-translation-table}`
- `exceptions/apple-mma-fragment-is-not-distribute-expressible`
