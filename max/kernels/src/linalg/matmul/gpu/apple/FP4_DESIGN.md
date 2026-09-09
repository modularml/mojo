# Apple M5 NVFP4 matmul design (W4A16)

This note records the design of the weight-only NVFP4 matmul for Apple-silicon
GPUs and the pipeline plumbing that lets a mixed-precision diffusion model run
on it. It complements the per-kernel reference in the kernel knowledge base.

## Problem

NVFP4 stores weights as 4-bit E2M1 values with a per-block (16 elements along K)
fp8_e4m3 scale, roughly a 4× shrink versus bfloat16. On NVIDIA SM100 and AMD
CDNA4 the tensor core has a native block-scaled FP4 path, so packed FP4 feeds
the MMA directly with the activations also quantized.

The Apple M5 MMA is different: its 16×16×16 simdgroup matmul accepts
fp16/bf16/fp32/fp8/int8, but **not FP4**. There is no hardware mode that
consumes packed FP4. So the SM100 approach does not port.

## Approach: weight-only quantization

We run **W4A16**: weights stay packed at 4 bits in DRAM; activations and outputs
are bfloat16. The FP4 weight is dequantized to bfloat16 at the point it is
consumed for the MMA — either in-register per B-fragment (the `matmul2d`
register-C tile) or cooperatively into threadgroup memory once per K-strip (the
cooperative-SMEM paths) — and fed to the existing bfloat16 MMA. Activations are
never quantized.

This buys the thing that matters on a unified-memory Mac — **residency**. The
weight working set stays 4× smaller and hot, instead of expanding to bfloat16 in
memory. It is not a throughput win: dequant-on-load is slower per FLOP than a
dense bf16 GEMM (see "Performance"). The win is fitting the model.

### Why the weight is the B operand (`transpose_b=True`)

For `out = x @ Wᵀ`, the FP4 weight `W` (`[N, K]`) is the B operand, and the
dense `AppleM5MatMul` already runs this layout with `transpose_b=True`. Keeping
the weight as B means only the B producer changes — the dequant — while the
activation A loads and the epilogue stay on the proven dense path. Reshaping the
weight into the A operand instead would swap M↔N across the Morton schedule, the
epilogue coordinates, and the transpose semantics — a transliteration with no
upside.

### Correctness anchor

Each path emits the dequantized B fragment in **exactly the lane→element order
the dense `transpose_b=True` path already produces** (the `_apple_frag_layout`
bit-scatter) and sets `hw_transpose_b=True`. Because the fragment layout and the
MMA flag are identical to the proven dense path, the fused dequant-matmul is
**bit-for-bit identical** to "materialize the weight to bfloat16, then run the
dense GEMM." Correctness rides on the existing dense oracle rather than on a
fresh layout argument; the tests pin exactly this equivalence.

A useful invariant falls out of the layout: each lane's four K-columns start at
an even index, so they are always **two contiguous packed bytes** — the
precondition for the interior fast path (one 2-byte load, four nibbles, one
block scale).

### Dequant math

The kernel applies `E2M1[nibble] * |block_scale_fp8|` only. The per-tensor
`weight_scale_2` is folded as a post-matmul multiply in the graph (in float32,
then cast to bf16), not in the kernel. `input_scale` cancels on Apple: with
bfloat16 activations there is no activation-quant pre-scale and no epilogue
`input_scale` fold, so only `weight_scale_2` survives — unlike the SM100 path,
which scales `x` by `1/input_scale` and folds `input_scale` into the epilogue.

## Memory access: TileTensor and `Fp4WeightLoader`

All DRAM/SMEM/register accesses in these kernels go through **TileTensor**
indexing and slicing — there is no raw pointer arithmetic in the matmul bodies.
The `matmul2d` kernels route their packed-weight, activation, block-scale, SMEM,
and output transitions through a single owner object, **`Fp4WeightLoader`** (a
TileIO expert object, sibling of the dense/conv loaders): `load_a_frag`
(bounds-aware A gather, zero-filling out-of-range rows on ragged M),
`decode_b_frag_regc` (packed → bf16 B fragment in-register), and
`decode_strip_to_smem` (cooperative strip decode into a double-buffered SMEM
view). Because the loader carries the bounds and the SMEM views are sized for
both buffers, the ragged-M and double-buffer edges are in-bounds by
construction rather than by hand-written guards.

The **one** access that stays a SIMD intrinsic is the MMA fragment load/store
inside `MmaOpApple`. The Apple fragment lane→element map is a hardware
bit-scatter (interleaved lane bits) that TileTensor `distribute` cannot express,
and Apple matmul has no shared-memory staging stage where `distribute` would pay
off, so the SIMD-width fragment loads there are the idiomatic implementation
(see the KB entry
`exceptions/apple-mma-fragment-is-not-distribute-expressible`).

## Dispatch

`enqueue_apple_fp4_matmul` selects among three paths by shape (all crossovers
measured on M5 with a thermal-fair A/B):

1. **`matmul2d` deep-K** (the default where it wins): `K ≥ 18432` and
   `M ≥ 1024`, aligned interior, no epilogue → the native 16×32×16 tiling
   (`enqueue_matmul2d_fp4_smem`). This is the real FLUX.2-dev FFN-down (N=6144,
   K=18432) at prefill M ≈ 4096.
2. **materialize→dense**: `M ≥ 1536` → dequant the whole weight to a transient
   bf16 buffer (execution-time only, not a graph constant), run the dense bf16
   GEMM, free it. The dequant is off the MMA critical path (one dense decode
   pass + a dense GEMM, both ~roofline), beating the fused path ~1.25–1.34× at
   large M.
3. **fused cooperative-SMEM** (`AppleM5Fp4MatMul`): smaller M. The threadgroup
   cooperatively decodes each `(BN, BK)` weight sub-tile into a bf16 SMEM tile
   once per K-strip, then runs the square simdgroup MMA (`MmaOpApple`) reading B
   from SMEM and A from DRAM, double-buffered. `BM=128`/`BK=64` for mid-M,
   `BM=64`/`BK=32` for small-M. Deeper `BK` feeds more MMA K-steps per decode;
   `BK=64` is the sweet spot (`BK=128` regresses on SMEM pressure, and a
   >16-lane 16-bit decode trips an M5 codegen crash, which bounds `BK`/`BM`).

## The `matmul2d` deep-K path

Apple's MLX 4-bit `quantized_matmul` reaches **~0.9× of dense** via Metal's
`matmul2d` (16×32×16) tensor-op. That op is **itself a software tiling over
`air.simdgroup_matrix_16x16x16`** (eight MMA call sites over the four
transpose combinations) — **not distinct silicon**. So `matmul2d_fp4.mojo`
reaches the same 16×32×16 tile with **two native `_mma_apple_transposable`
calls per issue** (`matmul2d_mma_regc_bt_native`, one per 16-wide N half); there
is **no backend tensor-op builtin and no KGEN external-symbol stamp**. The
tile's own ceiling is **~120 TF/s** (with four accumulators).

Its one winning regime is **deep-K FFN-down** (N=6144, K ≥ 18432): the
materialize→dense incumbent stages a 226 MB bf16 weight and hits the M5 DRAM
wall (→ ~32 TF/s), while the packed `matmul2d` kernel stays 4-bit-in-DRAM at
**~42 TF/s**. Off that niche `matmul2d` is dominated (materialize wins mid-K
large-M, fused wins small-M), which is why the dispatch gates it on deep K
rather than on large M generally.

## Performance

The value is residency, not TFLOP/s, but the gap to dense has been substantially
closed. Reconciled on one clean harness, with MLX re-run on the same M5 shapes:
the **shipped** dispatch is **~0.85–0.90× MLX-4bit** across the FLUX mix.
Routing deep-K FFN-down to `matmul2d` takes that slice from **0.72× to 0.96×
MLX**, and the whole FLUX W4A16 matmul mix from **0.868× to 0.964× MLX**
(~+11% prefill-effective; FFN-down is ~30% of the transformer matmul FLOPs).

The residual to MLX (~2–10%) is a **measured** ~10 TF/s co-issue gap (loads and
the MMA contend for the per-core issue path) plus MLX's slightly higher dense
ceiling. The levers tried (deeper strip, operand-interleave) did not move it, so
it is the **current limiter — not a proven floor**; async-copy-style feed,
issue-slot scheduling, and prefetch are untried. It is **not** the E2M1 decode
(that costs ~1 TF/s, and MLX's affine-int4 also decodes to bf16 — its edge is
not format-inherent).

## Lowering

The kernel is exposed as the graph op `mo.matmul.weight.only.block.scaled.apple`
(`graph_compiler/builtin_kernels/linalg.mojo`), gated on `is_gpu[target]()` and
`has_apple_gpu_accelerator()`. The op signature mirrors the existing W4A16
sibling `mo.matmul.mxfp4.dequant.fp8` (bf16 `a`, packed-uint8 `b`, rank-2
`b_scales`, no `a_scales`, no `tensor_sf`) rather than the NVFP4
`mo.matmul.dynamic.block.scaled` op, whose FP4-`a` + rank-5-scale shape is
hard-gated to SM100.

On the Python side, `nn/quant_ops.py::_matmul_float4` takes an Apple branch
(`_is_apple_gpu()`) that passes bf16 `x` and rank-2 weight scales straight
through to `_apple_weight_only_block_scaled_matmul` (`nn/kernels.py`), then
multiplies by `weight_scale_2` in float32. No new config flag — the branch
auto-engages on metal.

## Pipeline enablement (mixed-precision diffusion)

A quantized transformer paired with a bf16 text encoder and an f32 VAE breaks
the FLUX.2 pipeline's implicit assumption that all components share one dtype.
The per-component encodings are set explicitly through the existing
`--model-override <component>.quantization_encoding=...` mechanism; two
cross-platform code fixes (in the MAX pipeline layer, not in this kernel) then
let such an assembly run:

1. A dtype cast at the transformer→VAE graph boundary (no-op when matched), so
   an f32 VAE accepts the transformer's bf16 latents.
2. A float32→bfloat16 weight-path fallback, so a bf16-graph component (the VAE)
   can load a checkpoint that ships float32 safetensors.

With the kernel plus these two fixes, FLUX.2-dev with an NVFP4 transformer
renders end to end on an M5.

## Files

- `fp4_dequant.mojo` — E2M1 dequant, lo-nibble first; the stage-1 materialize
  oracle.
- `fp4_utils.mojo` — E2M1 → bf16/f32 decode primitives (branch-free,
  FTZ-safe).
- `matmul2d_fp4.mojo` — the native `matmul2d` W4A16 kernels: `Fp4WeightLoader`
  (the TileIO owner), `matmul2d_mma_regc_bt_native` (the two-`_mma_apple`
  16×32×16 tile), and the register-C (`run`) and cooperative-SMEM
  (`run_smem_decode`) bodies, plus `enqueue_matmul2d_fp4[_smem]`.
- `fp4_matmul.mojo` — `AppleM5Fp4MatMul` (the fused cooperative-SMEM kernel),
  the materialize→dense path, and `enqueue_apple_fp4_matmul` (the three-way
  dispatch); reuses the dense Morton scheduler, geometry, and epilogue.
- `../../../arch/apple/mma.mojo` — `MmaOpApple`, the dense simdgroup MMA op used
  by the fused and materialize→dense paths; owns the SIMD fragment I/O.
- `../../../../graph_compiler/builtin_kernels/linalg.mojo` — graph op
  registration.
- `nn/quant_ops.py`, `nn/kernels.py` — the metal lowering branch.
- `test/gpu/linalg/test_apple_fp4_matmul.mojo`,
  `tests/integration/nn/test_linear_nvfp4_apple_gpu.py` — correctness.
