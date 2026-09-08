# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Apple M5 weight-only FP8 (W8A16) GEMV: `out = x @ dequant(W)^T` at M=1.

Apple silicon GPU (Metal 4, `compute_capability == 5`). The batch-1 decode
regime (`M == 1`) is a matrix-vector product, NOT a matrix-matrix product: there
is exactly one activation row, so there is no MMA to feed. This kernel is the
GEMV -- register-resident, no SMEM, no `barrier()`, no `_mma_apple`.

This is the FP8 sibling of `fp4_gemv.mojo` (the shipped W4A16 GEMV), transliterated
op-for-op and SIMPLER: the weight is stored `float8_e4m3fn` (one byte per element,
NOT two packed E2M1 nibbles), so the FP4 nibble-unpack + E2M1-f16-decode + per-16
FP8 block-scale collapse to a single width load + a native `float8_e4m3fn -> f32`
widening cast. Two FP4-specific M5 hazards vanish with it:
  - No denormal-flush-to-zero footgun: there is no magic-constant f16 decode
    injecting +-0.5 (`patterns/apple-m5-denormal-flush-to-zero`); the E4M3 -> f32
    cast is the hardware's own conversion, exact for every E4M3 code on M5 (the
    same cast the FP4 GEMV already uses on its FP8-E4M3 block scales).
  - No >=24-lane 16-bit-SIMD-arithmetic codegen crash
    (`known-limitations/apple-m5-wide-16bit-simd-codegen-crash`): the only
    16-bit-domain arithmetic in the FP4 kernel was the E2M1 nibble expansion,
    which does not exist here. The FP8 load widens straight to f32; all
    arithmetic is f32 (any width is safe there).

WHY it wins at decode: batch-1 decode on M5 is weight-read-bandwidth-bound (the
profiling campaign measured ~530 GB/s of the ~614 GB/s peak on the bf16 path).
W8A16 reads 1 byte/weight instead of the 2 bytes of a bf16 weight, so the
wall-clock drops roughly with the byte count. The `float8_e4m3fn -> f32` cast
gives NO compute speedup (there is no MMA here); the entire win is reading fewer
bytes. See `Kernels/claude_kb/entries/kernels/apple-m5-fp4-matmul.md` and
`patterns/apple-m5-gpu-perf-model` (the co-issue penalty applies only at core
saturation -- the GEMV is under-occupancy, so it does not bite here).

Structure (mirrors `fp4_gemv.mojo`, itself mirroring
`linalg/gemv.mojo::gemv_kernel_vector`): one warp owns one output column `n`
(= one row of the `transpose_b` weight `W[N, K]`); its 32 lanes stride down K in
whole `TILE_K`-wide chunks (strided by `WARP_SIZE` chunks). Per chunk: one
coalesced width-`TILE_K` FP8 load through the `Fp8WeightLoader` expert object ->
native E4M3 -> f32 widen; one width-`TILE_K` bf16 activation load -> f32; FMA
into an fp32 accumulator. A `warp.sum` reduces the 32 partials to the output
element. A scalar (width-1) tail handles `K % TILE_K != 0` (the non-aligned edge;
empty on the model path, where K is a large power of two).

Scale contract (mirrors the FP4 W4A16 path exactly): the modelopt static FP8
checkpoint carries ONE per-tensor scalar `weight_scale` (and a per-tensor
`input_scale` that cancels for a bf16 activation -- see the routing branch in
`quant_ops._matmul_float8`). This kernel produces the RAW `x @ W_fp8^T`; the
scalar `weight_scale` is folded by the graph lowering as a post-matmul multiply
(the FP8 analog of NVFP4's `weight_scale_2`, applied in
`quant_ops._matmul_float8`). Because `weight_scale` is a scalar it factors out of
the sum, so the post-matmul fold is EXACT (not merely within tolerance).

Every DRAM -> register weight transition has an OWNER (`Fp8WeightLoader`, KB
`new-primitives/amd-tile-io-expert-objects`; the FP8 sibling of `Fp4WeightLoader`
in `matmul2d_fp4.mojo`): it holds the `weight [N, K]` TileTensor view and does
ALL addressing through TileTensor width-loads / indexing -- no raw pointer
arithmetic. The bf16 activation is a plain row-major `[1, K]` tensor; its width
load is an inline TileTensor access (no special decode to own), matching
`fp4_gemv.mojo`.
"""

from std.collections import Optional
from max.gpu import WARP_SIZE, global_idx, lane_id
from max.gpu.host import DeviceContext
from std.math import ceildiv
import max.gpu.primitives.warp as warp
from std.utils import IndexList

from layout import Coord, TensorEngine, TileTensor, TensorLayout
from layout.tile_layout import row_major

from linalg.matmul.gpu.apple.matmul_8x8 import gemm_kernel_apple_8x8
from linalg.matmul.gpu.apple.matmul_kernel import enqueue_apple_matmul
from linalg.matmul.gpu.apple.matmul2d_fp8 import enqueue_matmul2d_fp8
from linalg.utils import elementwise_epilogue_type


# The K-chunk width one lane loads per iteration. 16 mirrors the FP4 GEMV's
# `NVFP4_SF_VECTOR_SIZE = 16` activation width (a proven-on-M5 `<16 x bfloat>`
# activation load; see `fp4_gemv.mojo`). The FP8 weight load is width-16 bytes
# (`float8_e4m3fn` is one byte), 16-byte-aligned on the aligned interior
# (`K % 16 == 0`, true for every FP8-quantized Linear). This is the single
# perf/codegen knob for the deferred M5 microbench: a `<16 x float8_e4m3fn>` load
# + a width-16 E4M3 -> f32 widen are the two idioms not yet measured on-device;
# if either mislowers, drop to 8 (`<8 x bfloat>` is the proven fa_prefill width)
# or 4 (alignment-robust for every dtype) -- correctness is width-independent
# (the scalar tail covers any residue).
comptime _FP8_GEMV_TILE_K = 16


@fieldwise_init
struct Fp8WeightLoader[
    w_layout: TensorLayout,
](ImplicitlyCopyable, Movable):
    """Owner of the FP8 weight -> f32 register transition for the W8A16 GEMV.

    The FP8 sibling of `Fp4WeightLoader` (`matmul2d_fp4.mojo`), specialized for
    the unpacked E4M3 GEMV: there is no nibble unpack and no per-block scale, so a
    column chunk is a single coalesced width-`W` `float8_e4m3fn` load widened to
    f32. Holds the `weight [N, K]` TileTensor view and does ALL addressing via
    TileTensor width-loads -- no raw pointer arithmetic.

    Parameters:
        w_layout: Layout of the FP8 weight `[N, K]` view.
    """

    # Held with `ImmUntrackedOrigin`: struct fields cannot expose `AnyOrigin`
    # (same field-origin rule as `Fp4WeightLoader` / `DenseALoader`). The kernel
    # arg this view derives from outlives the K-loop. Built via
    # `Fp8WeightLoader.from_kernel_args`.
    var weight: TileTensor[.float8_e4m3fn, Self.w_layout, ImmUntrackedOrigin]

    @always_inline
    @staticmethod
    def from_kernel_args(
        weight: TileTensor[
            .float8_e4m3fn, Self.w_layout, ImmutAnyOrigin, Engine=_
        ],
    ) -> Self:
        """Build the loader from the kernel's `AnyOrigin` weight arg.

        Rebases the view onto `ImmUntrackedOrigin` (the field-origin rule; the
        arg outlives the K-loop), preserving layout/shape/stride.
        """
        return Self(
            TileTensor(
                weight.ptr.unsafe_origin_cast[ImmUntrackedOrigin](),
                weight.layout,
            )
        )

    @always_inline
    def load_col_chunk[
        width: Int
    ](self, n: Int, k0: Int) -> SIMD[.float32, width]:
        """This column's `width` FP8 weights at `[n, k0 : k0+width)`, E4M3 -> f32.

        One coalesced width-`width` `float8_e4m3fn` load (adjacent lanes read
        adjacent runs) widened to f32 by the hardware E4M3 conversion (exact on
        M5). `alignment=1` makes no alignment claim (never miscompiles; the FP4
        GEMV's packed-byte load uses the same conservative claim). Callers pass
        `k0 + width <= K` (the aligned interior uses `width = TILE_K`; the K tail
        uses `width = 1`).

        Parameters:
            width: SIMD width of the load (`TILE_K` interior, `1` for the tail).
        """
        # No flat-rank assert: the `Coord`-based `.load` needs no N-D-index
        # evidence (only `t[r, c]` subscripting does), matching `fp4_gemv`. The
        # `[N, K]` contract lives in the docstring; a `self.weight.flat_rank`
        # comptime assert is illegal anyway (dynamic-value read through `self`).
        return self.weight.load[width=width, alignment=1](Coord(n, k0)).cast[
            DType.float32
        ]()


@__name(t"fp8_gemv_{c_type}")
def fp8_gemv_kernel[
    c_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    w_layout: TensorLayout,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    w_engine: TensorEngine,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],  # [1, N]
    a: TileTensor[
        .bfloat16, a_layout, ImmutAnyOrigin, Engine=a_engine
    ],  # [1, K]
    weight: TileTensor[
        .float8_e4m3fn, w_layout, ImmutAnyOrigin, Engine=w_engine
    ],  # [N, K]
    n_arg: Int32,
    k_arg: Int32,
):
    """One warp per output column; 32 lanes stride down K widening FP8 -> f32.

    `c` is `[1, N]`, `a` the bf16 activation `[1, K]`, `weight` the FP8-E4M3
    weight `[N, K]`. Accumulation is fp32. Produces the RAW `x @ W_fp8^T`; the
    per-tensor scalar `weight_scale` is folded post-matmul by the graph lowering.
    """
    var n = Int(n_arg)
    var k = Int(k_arg)
    # No flat-rank asserts: `a.load`/`c.store` are `Coord`-based (not `t[r, c]`
    # subscripts), so no N-D-index evidence is required, matching `fp4_gemv`.
    comptime TILE_K = _FP8_GEMV_TILE_K

    var n_idx = Int(global_idx.x) // WARP_SIZE
    if n_idx >= n:
        return
    var lid = Int(lane_id())

    var loader = Fp8WeightLoader[w_layout].from_kernel_args(weight)

    var acc = Float32(0)

    # Vectorized interior: each lane owns whole `TILE_K` chunks, strided by
    # `WARP_SIZE` chunks. Adjacent lanes read adjacent runs -> coalesced.
    var nchunk = k // TILE_K
    var chunk = lid
    while chunk < nchunk:
        var k0 = chunk * TILE_K
        var wv = loader.load_col_chunk[TILE_K](n_idx, k0)
        var xv = a.load[width=TILE_K](Coord(0, k0)).cast[.float32]()
        acc[0] += (xv * wv).reduce_add()
        chunk += WARP_SIZE

    # Scalar (width-1) tail for `K % TILE_K` (non-aligned edge; empty on the
    # model path). `load_col_chunk[1]` is the width-1 specialization, so the tail
    # reuses the exact same owned-load path -- no separate scalar accessor.
    var ktail0 = nchunk * TILE_K
    var kt = ktail0 + lid
    while kt < k:
        var wv = loader.load_col_chunk[1](n_idx, kt)
        var xv = a.load[width=1](Coord(0, kt)).cast[.float32]()
        acc[0] += (xv * wv).reduce_add()
        kt += WARP_SIZE

    var dot = warp.sum(acc)
    if lid == 0:
        var y = dot.cast[c_type]()

        comptime if elementwise_lambda_fn:
            comptime epilogue = elementwise_lambda_fn.value()
            epilogue[c_type, 1](IndexList[2](0, n_idx), y)
        else:
            c.store(Coord(0, n_idx), y)


@always_inline
def enqueue_apple_fp8_gemv[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    weight: TileTensor[.float8_e4m3fn, ...],
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Enqueue the M=1 W8A16 GEMV: `out = a @ W_fp8^T` (raw, unscaled).

    One warp per output column N (`ceil(N*WARP_SIZE / block_dim)` threadgroups of
    `block_dim` threads). `a` is the bf16 activation `[1, K]`, `weight` the
    FP8-E4M3 weight `[N, K]`. Assumes `M == 1`; the caller
    (`enqueue_apple_fp8_matmul`) gates the `M == 1` route. Any K is correct (the
    kernel's width-1 tail covers `K % TILE_K`).

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue (AMD's `(row, col)`
            contract), applied on the width-1 store.
    """
    comptime BLK = 256  # 8 warps / threadgroup
    var grid = ceildiv(n * WARP_SIZE, BLK)

    comptime kernel = fp8_gemv_kernel[
        c_type,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(weight).LayoutType,
        elementwise_lambda_fn,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(weight).Engine,
    ]
    ctx.enqueue_function[kernel](
        c,
        a,
        weight,
        Int32(n),
        Int32(k),
        grid_dim=grid,
        block_dim=BLK,
    )


@__name(t"apple_fp8_materialize_{out_type}")
def fp8_materialize_kernel[
    out_type: DType,
    w_layout: TensorLayout,
    out_layout: TensorLayout,
    out_engine: TensorEngine,
    w_engine: TensorEngine,
](
    out_w: TileTensor[out_type, out_layout, MutAnyOrigin, Engine=out_engine],
    weight: TileTensor[
        .float8_e4m3fn, w_layout, ImmutAnyOrigin, Engine=w_engine
    ],
):
    """Materializes the FP8 weight into a dense `[N, K]` `out_type` buffer.

    One thread per output element `(n, k)`: `out_w[n, k] = W_fp8[n, k]` widened to
    `out_type` (E4M3 -> `out_type`, exact for `out_type = bf16`; bf16 represents
    every E4M3 value exactly). The M>1 interim path of `enqueue_apple_fp8_matmul`
    dequantizes the weight so the EXISTING dense bf16 `AppleM5MatMul` can consume
    it (the per-tensor `weight_scale` is folded post-matmul by the graph lowering,
    identically to the GEMV path). The FP8 sibling of `fp4_materialize_kernel`,
    minus the nibble unpack + block scale.
    """
    comptime assert out_w.flat_rank == 2, "out_w must be 2D [N, K]"
    comptime assert weight.flat_rank == 2, "weight must be 2D [N, K]"

    var N = Int(out_w.dim[0]())
    var K = Int(out_w.dim[1]())

    var n = Int(global_idx.y)
    var k = Int(global_idx.x)
    if n >= N or k >= K:
        return

    # Rebind the element to a plain `Scalar[float8_e4m3fn]` before the widening
    # cast (mirrors `fp4_materialize_kernel`; the GPU skill's tensor-element read
    # idiom), then rebind the widened value back to the output ElementType. bf16
    # represents every E4M3 value exactly, so this cast is lossless.
    var wv = rebind[Float8_e4m3fn](weight[n, k])
    out_w[n, k] = rebind[out_w.ElementType](wv.cast[out_type]())


@always_inline
def enqueue_fp8_materialize[
    out_type: DType
](
    out_w: TileTensor[mut=True, out_type, ...],
    weight: TileTensor[.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Enqueues `fp8_materialize_kernel`: FP8 weight -> dense `[N, K]` `out_type`.

    `out_w` is `[N, K]`, `weight` is `[N, K]`. The grid is
    `(ceil(K/16), ceil(N/16))` threadgroups of 16x16 threads; bounds are checked
    per thread so ragged K/N are fine. The FP8 sibling of `enqueue_fp4_materialize`.
    """
    var N = Int(out_w.dim[0]())
    var K = Int(out_w.dim[1]())

    comptime BLK = 16
    comptime kernel = fp8_materialize_kernel[
        out_type,
        type_of(weight).LayoutType,
        type_of(out_w).LayoutType,
        type_of(out_w).Engine,
        type_of(weight).Engine,
    ]
    ctx.enqueue_function[kernel](
        out_w,
        weight.as_immut(),
        grid_dim=(ceildiv(K, BLK), ceildiv(N, BLK)),
        block_dim=(BLK, BLK),
    )


@always_inline
def _enqueue_apple_fp8_materialize_dense[
    c_type: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    weight: TileTensor[.float8_e4m3fn, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Materialize the FP8 weight to a transient bf16 buffer, then dense GEMM.

    The M>1 (prefill / interim) path of `enqueue_apple_fp8_matmul`. Allocates a
    transient `[N, K]` bf16 weight buffer at execution time, dequantizes the FP8
    weight into it (`enqueue_fp8_materialize`, the SAME E4M3 -> bf16 cast the GEMV
    uses -- bf16 is exact for E4M3, so this is bit-consistent with the GEMV
    weight values), then runs the stock dense bf16 MMA (`enqueue_apple_matmul`,
    `transpose_b=True`) reading the materialized weight. The optional fused
    epilogue threads straight through. The per-tensor `weight_scale` scalar is
    applied OUTSIDE the kernel by the graph lowering (a post-matmul multiply),
    identically to the GEMV path. Slice 4 will replace this interim with a fused
    cooperative-SMEM FP8 matmul (mirroring the FP4 fused path).

    Buffer lifetime: the dense GEMM reads `wdense` after this function returns
    (`enqueue_*` is async). `DeviceBuffer.__deinit__` schedules a stream-ordered
    free, so it cannot race the GEMM on one in-order stream; the `_ = wdense_dev^`
    pins the handle alive until AFTER both enqueues are issued. Same pattern as
    `_enqueue_apple_fp4_materialize_dense`.
    """
    var wdense_dev = ctx.enqueue_create_buffer[.bfloat16](n * k)
    var wdense_tt = TileTensor(wdense_dev.unsafe_ptr(), row_major(n, k))

    enqueue_fp8_materialize[.bfloat16](wdense_tt, weight, ctx)

    if ctx.compute_capability() == 5:
        enqueue_apple_matmul[
            in_type=DType.bfloat16,
            c_type=c_type,
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, wdense_tt.as_immut(), ctx)
    else:
        comptime apple_kernel = gemm_kernel_apple_8x8[
            c_type,
            DType.bfloat16,
            DType.bfloat16,
            type_of(c).LayoutType,
            type_of(a).LayoutType,
            type_of(wdense_tt).LayoutType,
            type_of(c).Engine,
            type_of(a).Engine,
            type_of(wdense_tt).Engine,
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda_fn,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=16,
            NUM_SIMDGROUPS=4,
        ]
        ctx.enqueue_function[apple_kernel](
            c,
            a,
            wdense_tt.as_immut(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(n, 64), ceildiv(m, 64)),
            block_dim=(4 * WARP_SIZE,),
        )

    # Keep the transient weight alive through the async materialize + GEMM
    # enqueue (see the buffer-lifetime note in the docstring).
    _ = wdense_dev^


@always_inline
def enqueue_apple_fp8_matmul[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    weight: TileTensor[.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the W8A16 matmul: `out = a @ W_fp8^T` (raw, unscaled).

    `a` is the bf16 activation `(M, K)`, `weight` the FP8-E4M3 weight `(N, K)`
    (`transpose_b`). C is `(M, N)`. Produces the RAW matmul; the per-tensor scalar
    `weight_scale` is folded post-matmul by the graph lowering
    (`quant_ops._matmul_float8`, the FP8 analog of NVFP4's `weight_scale_2`).

    Dispatch (all routes produce consistent weight values -- E4M3 -> f32/bf16 is
    exact):

    - Tiled FP8 matmul (`enqueue_matmul2d_fp8`; bf16 A x fp8 B -> fp32 on the
      native Apple MMA, direct DRAM fp8 feed), no fused epilogue, for the wide-N
      decode class (`n > k`: Mamba in_proj, MLP up) at ANY M AND every `M > 1`
      shape including narrow-N (out_proj, MLP down). It amortizes the per-output
      cost across a threadgroup tile, beating both the bf16 matmul and the
      materialize route (measured M5 Max, M=32: 1.20-2.17x the bf16 matmul on all
      four Nemotron FP8 Linears; the prior narrow-N M>1 materialize route ran at
      0.11-0.20x bf16 -- the c32 FP8 regression).
    - Batch-1 decode (`M == 1`) narrow-N (`n <= k`: out_proj, MLP down): the
      register-resident W8A16 GEMV (`enqueue_apple_fp8_gemv`), no MMA. Its long
      per-warp K amortizes the per-output cost, so it wins this class at M=1
      (2.4-2.6x bf16, ahead of tiled's ~1.1x there).
    - `M > 1` WITH a fused epilogue (the tiled interior store can't apply it), or
      any M on pre-M5: MATERIALIZE the weight to a transient `(N, K)` bf16 buffer,
      then run the dense bf16 MMA (hardware-neutral on pre-M5).

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue (AMD's `(row, col)`
            contract), threaded to whichever path runs.
    """
    comptime assert (
        c_type == .float16 or c_type == .bfloat16 or c_type == .float32
    ), "enqueue_apple_fp8_matmul: c_type must be one of {fp16, bf16, fp32}"

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(Int(weight.dim[0]()) == n, "weight must be (N, K)")
    debug_assert(Int(weight.dim[1]()) == k, "weight must be (N, K)")

    var cc = ctx.compute_capability()

    # Pre-M5 (M1-M4): materialize -> dense (hardware-neutral); the M5 GEMV's
    # `float8_e4m3fn` width load / widen is unvalidated pre-M5, and the design
    # targets M5. This keeps pre-M5 correct.
    if cc != 5:
        _enqueue_apple_fp8_materialize_dense[c_type, elementwise_lambda_fn](
            c, a, weight, m, n, k, ctx
        )
        return

    # Tiled FP8 route (bf16 A x fp8 B -> fp32 on the native Apple MMA, direct
    # DRAM fp8 feed). It amortizes the per-output cost across a threadgroup tile
    # and beats BOTH the bf16 matmul and the materialize->dense route whenever
    # the simdgroup MMA has a real activation matrix to amortize over:
    #   - wide-N decode (n > k: Mamba in_proj, MLP up) at ANY M, and
    #   - EVERY M > 1 shape, including narrow-N (out_proj, MLP down).
    # Measured (M5 Max, M=32, weight-read GB/s vs the bf16 matmul the model runs
    # today): in_proj 1.92x, mlp_up 2.17x, out_proj 1.29x, mlp_down 1.20x -- all
    # wins. The OLD n<=k M>1 route materialized the weight to bf16 (~5*N*K bytes)
    # and ran at 0.11-0.20x bf16 (a ~5-9x slowdown) -- the entire c32 FP8
    # regression. Only M == 1 narrow-N stays on the GEMV below: its long per-warp
    # K amortization beats tiled at batch 1 (out_proj 2.38x, mlp_down 2.60x bf16
    # vs tiled's 1.07x/1.13x at M=1). Gated on no fused epilogue (the matmul2d
    # interior store ignores it -- the GEMV / materialize paths below apply it).
    comptime if elementwise_lambda_fn:
        # Fused epilogue present: skip the tiled path (interior store ignores it).
        pass
    else:
        if n > k or m > 1:
            enqueue_matmul2d_fp8[c_type=c_type](c, a, weight, ctx)
            return

    # Batch-1 decode: register-resident W8A16 GEMV, no MMA.
    if m == 1:
        enqueue_apple_fp8_gemv[c_type, elementwise_lambda_fn](
            c, a, weight, n, k, ctx
        )
        return

    # M > 1 WITH a fused epilogue (the tiled interior store can't apply it): the
    # materialize -> dense bf16 MMA path. (Pre-M5 M>1 already returned above.)
    _enqueue_apple_fp8_materialize_dense[c_type, elementwise_lambda_fn](
        c, a, weight, m, n, k, ctx
    )
