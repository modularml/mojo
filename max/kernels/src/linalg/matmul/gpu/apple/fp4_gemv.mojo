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
"""Apple M5 weight-only NVFP4 (W4A16) GEMV: `out = x @ dequant(W)^T` at M=1.

Apple silicon GPU (Metal 4, `compute_capability == 5`). The batch-1 decode
regime (`M == 1`) is a matrix-vector product, NOT a matrix-matrix product: there
is exactly one activation row, so there is no MMA to feed. Routing it through the
prefill cooperative-SMEM MMA path (`AppleM5Fp4MatMul`) wastes the simdgroup
matrix unit on a rank-1 update and is bandwidth-inefficient. This kernel is the
GEMV instead -- register-resident, no SMEM, no `barrier()`, no `_mma_apple`.

WHY it wins at decode: Llama-8B batch-1 decode is weight-read-bandwidth-bound
(the profiling campaign measured ~540 GB/s, ~88% of the M5 ~614 GB/s peak, on
the bf16 path). W4A16 reads ~1/4 the weight bytes (packed nibbles + a small fp8
scale stream vs a full bf16 weight), so the wall-clock drops roughly with the
byte count -- measured 1.53x over a bf16 GEMV at the down-proj shape (N=4096,
K=14336) on M5 Max. The dequant-to-bf16 gives NO compute speedup (there is no
MMA here); the entire win is reading fewer bytes. See
`Kernels/claude_kb/entries/kernels/apple-m5-fp4-matmul.md` and
`patterns/apple-m5-quantized-matmul-ceiling` (the co-issue penalty applies only
at core saturation -- the GEMV is under-occupancy, so it does not bite here).

Structure (mirrors `linalg/gemv.mojo::gemv_kernel_vector`, B-load replaced by an
inline FP4 decode): one warp owns one output column `n` (= one row of the
`transpose_b` weight `W[N, K]`); its 32 lanes stride down K, each lane owning
whole 16-col FP8 scale blocks (strided by `WARP_SIZE` blocks). Per block: one
coalesced width-8 packed-byte load -> expand to 16 E2M1 nibbles ->
`decode_e2m1_to_f16` (F16-domain inject; cast f32) -> `* |block_scale|`
-> FMA into an fp32 accumulator against the activation. A `warp.sum` reduces the
32 partials to the output element. The decode stays in the **float16** domain
because the M5 flushes f32/bf16 denormals to zero on arithmetic inputs (a
magic-constant f32/bf16 decode silently zeroes every +-0.5, see
`patterns/apple-m5-denormal-flush-to-zero`) but **preserves f16 subnormals**, so
the f16 inject decodes +-0.5 exactly (verified on-device).

Two hard M5 constraints, both satisfied by the width-16 decode:
  - The decode's 16-bit SIMD width is exactly `NVFP4_SF_VECTOR_SIZE = 16`
    nibbles per block; the Metal backend crashes on >= 24-lane 16-bit-element
    SIMD arithmetic (see
    `known-limitations/apple-m5-wide-16bit-simd-codegen-crash`). Staying at 16
    lanes dodges it. All accumulation is fp32 (any width is safe there).
  - The dequant arithmetic (`decode_e2m1_to_f16(nibble).cast[f32]() *
    abs(scale)`, fp32 accumulate) is BIT-IDENTICAL to the materialize->dense
    oracle (`fp4_dequant.mojo`) -- all 16 E2M1 values are exact in f16->f32, so
    the f16 decode equals `decode_e2m1_to_f32` bit-for-bit -- so the GEMV is a
    within-tolerance match of the dense path (differing only by the
    fp32-vs-MMA reduction order and the final cast).
"""

from std.collections import Optional
from max.gpu import WARP_SIZE, global_idx, lane_id
from max.gpu.host import DeviceContext
from std.math import ceildiv
import max.gpu.primitives.warp as warp
from std.utils import IndexList

from layout import Coord, TensorEngine, TileTensor, TensorLayout
from layout.tile_layout import row_major

from linalg.fp4_utils import decode_e2m1_to_f16, NVFP4_SF_VECTOR_SIZE
from linalg.utils import elementwise_epilogue_type


@__name(t"fp4_gemv_{c_type}")
def fp4_gemv_kernel[
    c_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    p_layout: TensorLayout,
    s_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    p_engine: TensorEngine,
    s_engine: TensorEngine,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],  # [1, N]
    a: TileTensor[
        .bfloat16, a_layout, ImmutAnyOrigin, Engine=a_engine
    ],  # [1, K]
    packed: TileTensor[
        .uint8, p_layout, ImmutAnyOrigin, Engine=p_engine
    ],  # [N, K//2]
    scales: TileTensor[
        .float8_e4m3fn, s_layout, ImmutAnyOrigin, Engine=s_engine
    ],  # [N,K//16]
    n_arg: Int32,
    k_arg: Int32,
):
    """One warp per output column; 32 lanes stride down K decoding FP4 -> fp32.

    `c` is `[1, N]`, `a` the bf16 activation `[1, K]`, `packed` the FP4 weight
    `[N, K//2]` (lo-nibble first), `scales` the FP8-E4M3 block scales
    `[N, ceil(K/16)]`. Accumulation is fp32.

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        c_layout: `TileTensor` layout of the output `c`.
        a_layout: `TileTensor` layout of the activation `a`.
        p_layout: `TileTensor` layout of the packed FP4 weight.
        s_layout: `TileTensor` layout of the FP8 block scales.
        c_engine: `TensorEngine` of the output `c`.
        a_engine: `TensorEngine` of the activation `a`.
        p_engine: `TensorEngine` of the packed FP4 weight.
        s_engine: `TensorEngine` of the FP8 block scales.
        elementwise_lambda_fn: Optional fused epilogue applied on the
            width-1 store.

    Args:
        c: Output tile tensor `[1, N]` receiving the GEMV result.
        a: Bf16 activation tile tensor `[1, K]`, the single activation row.
        packed: FP4-packed weight tile tensor `[N, K//2]` (lo-nibble first).
        scales: FP8-E4M3 block scales tile tensor `[N, ceil(K/16)]`.
        n_arg: Number of output columns (rows of the transposed weight).
        k_arg: Inner dimension length; must be a multiple of
            `NVFP4_SF_VECTOR_SIZE` (16).
    """
    var n = Int(n_arg)
    var k = Int(k_arg)
    comptime SF = NVFP4_SF_VECTOR_SIZE  # 16 nibbles / scale block
    comptime BYTES = SF // 2  # 8 packed bytes / block

    var n_idx = Int(global_idx.x) // WARP_SIZE
    if n_idx >= n:
        return
    var lid = Int(lane_id())

    # Each lane owns whole 16-col FP8 scale-blocks, strided by WARP_SIZE blocks.
    # (K is a multiple of 16 for every NVFP4-quantized Linear -> no K tail.)
    var acc = Float32(0)
    var nblk = k // SF
    var blk = lid
    while blk < nblk:
        var k0 = blk * SF  # first K col of this block
        var byte0 = blk * BYTES  # first packed byte of this block
        # One coalesced width-8 byte load (adjacent lanes read adjacent runs) --
        # no raw pointer arithmetic (TileTensor width-load).
        var bytes = packed.load[width=BYTES, alignment=1](Coord(n_idx, byte0))
        # Expand 8 bytes -> 16 E2M1 nibbles: element 2*j = lo nibble (even K),
        # 2*j+1 = hi nibble (odd K). Width-16 uint16 arithmetic: M5-safe.
        var nib = SIMD[.uint16, SF](0)

        comptime for j in range(BYTES):
            var bj = UInt16(bytes[j])
            nib[2 * j] = bj & UInt16(0xF)
            nib[2 * j + 1] = (bj >> UInt16(4)) & UInt16(0xF)

        var scale_abs = abs(scales[n_idx, blk][0].cast[.float32]())
        # F16-domain decode (Preston's inject) then cast f32: dodges the M5
        # bf16/f32 subnormal-FTZ trap that zeroes +-0.5 -- f16 subnormals
        # survive on M5 (verified on-device). Bit-identical to
        # `decode_e2m1_to_f32(nib)` (all 16 E2M1 values are exact in f16->f32),
        # so `* scale_abs` still matches the materialize->dense oracle exactly.
        var w_f32 = decode_e2m1_to_f16(nib).cast[.float32]() * scale_abs
        var xv = a.load[width=SF](Coord(0, k0)).cast[.float32]()
        acc[0] += (xv * w_f32).reduce_add()
        blk += WARP_SIZE

    var dot = warp.sum(acc)
    if lid == 0:
        var y = dot.cast[c_type]()

        comptime if elementwise_lambda_fn:
            comptime epilogue = elementwise_lambda_fn.value()
            epilogue[c_type, 1](IndexList[2](0, n_idx), y)
        else:
            c.store[width=1](Coord(0, n_idx), y)


@always_inline
def enqueue_apple_fp4_gemv[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    packed: TileTensor[.uint8, ...],
    scales: TileTensor[.float8_e4m3fn, ...],
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Enqueue the M=1 W4A16 GEMV: `out = a @ dequant(packed, scales)^T`.

    One warp per output column N (`ceil(N*WARP_SIZE / block_dim)` threadgroups of
    `block_dim` threads). `a` is the bf16 activation `[1, K]`, `packed` the FP4
    weight `[N, K//2]` (lo-nibble first), `scales` the FP8-E4M3 block scales
    `[N, ceil(K/16)]`. Assumes `M == 1` and `K % NVFP4_SF_VECTOR_SIZE == 0` (true
    for every NVFP4-quantized Linear); the caller (`enqueue_apple_fp4_matmul`)
    gates the `M == 1` route.

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue (AMD's `(row, col)`
            contract), applied on the width-1 store.

    Args:
        c: Output tile tensor `[1, N]` receiving the GEMV result.
        a: Bf16 activation tile tensor `[1, K]`, the single activation row.
        packed: FP4-packed weight tile tensor `[N, K//2]` (lo-nibble first).
        scales: FP8-E4M3 block scales tile tensor `[N, ceil(K/16)]`.
        n: Number of output columns (rows of the transposed weight).
        k: Inner dimension length; must be a multiple of
            `NVFP4_SF_VECTOR_SIZE` (16).
        ctx: `DeviceContext` used to enqueue the kernel on the device.
    """
    comptime BLK = 256  # 8 warps / threadgroup
    var grid = ceildiv(n * WARP_SIZE, BLK)

    comptime kernel = fp4_gemv_kernel[
        c_type,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(packed).LayoutType,
        type_of(scales).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(packed).Engine,
        type_of(scales).Engine,
        elementwise_lambda_fn,
    ]
    ctx.enqueue_function[kernel](
        c,
        a,
        packed,
        scales,
        Int32(n),
        Int32(k),
        grid_dim=grid,
        block_dim=BLK,
    )
