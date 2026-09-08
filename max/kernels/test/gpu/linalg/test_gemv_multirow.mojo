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
"""Multi-row GEMV vector kernel: bit-exactness against the one-row kernel.

The wide-N shallow-K M=1 GEMV path packs several output rows into one warp to
escape the CTA launch-rate ceiling. The reduction per row is unchanged, so the
new kernel must reproduce the one-row kernel bit for bit, including under an
epilogue lambda and for an N that does not divide the row tile.
"""

from std.math import align_up, ceildiv
from std.sys import simd_width_of

from max.gpu import WARP_SIZE, global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from layout import Coord, Idx, TileTensor, row_major
from linalg.gemv import (
    _GEMV_MULTIROW_ROWS,
    gemv_kernel_vector,
    gemv_kernel_vector_multirow,
)
from linalg.utils import elementwise_epilogue_type
from std.utils import IndexList


def _fill[
    dtype: DType, integral: Bool
](x: MutPointer[Scalar[dtype], MutAnyOrigin], length: Int32, seed: Int32,):
    var i = Int(global_idx.x)
    if i >= Int(length):
        return
    var h = UInt32(i) * UInt32(2654435761) + UInt32(seed) * UInt32(2246822519)
    h ^= h >> 16
    h *= UInt32(2654435761)
    h ^= h >> 13

    comptime if integral:
        # Small integers keep an f32 dot product of K<=1024 terms exact, so a
        # host reference matches the GPU regardless of summation order.
        x[i] = Float32(Int(h % UInt32(9)) - 4).cast[dtype]()
    else:
        x[i] = (
            Float32(Int(h & UInt32(0xFFFF)) - 32768) / Float32(32768.0)
        ).cast[dtype]()


def _fill_launch[
    dtype: DType, integral: Bool = False
](buf: DeviceBuffer[dtype], length: Int, seed: Int, ctx: DeviceContext) raises:
    ctx.enqueue_function[_fill[dtype, integral]](
        buf,
        Int32(length),
        Int32(seed),
        grid_dim=ceildiv(length, 256),
        block_dim=256,
    )


def test_matches_one_row_kernel[
    c_type: DType, dtype: DType, N: Int, K: Int, *, with_epilogue: Bool = False
](ctx: DeviceContext, seed: Int) raises:
    """Runs both kernels on the same inputs and requires identical outputs."""
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime check_bounds_k = K % (WARP_SIZE * simd_width) != 0
    comptime rows_per_warp = _GEMV_MULTIROW_ROWS
    comptime warps_per_block = 8
    # The one-row launch the dispatch performs today.
    comptime one_row_block_dim = min(align_up(K // simd_width, WARP_SIZE), 1024)

    print(
        "N",
        N,
        "K",
        K,
        c_type,
        "seed",
        seed,
        "rows_per_warp",
        rows_per_warp,
        "epilogue",
        with_epilogue,
    )

    var w_dev = ctx.enqueue_create_buffer[dtype](N * K)
    var act_dev = ctx.enqueue_create_buffer[dtype](K)
    var c_dev = ctx.enqueue_create_buffer[c_type](N)
    var bias_dev = ctx.enqueue_create_buffer[c_type](N)
    var ref_host = ctx.enqueue_create_host_buffer[c_type](N)
    var new_host = ctx.enqueue_create_host_buffer[c_type](N)

    _fill_launch[dtype](w_dev, N * K, seed, ctx)
    _fill_launch[dtype](act_dev, K, seed + 1, ctx)
    _fill_launch[c_type](bias_dev, N, seed + 2, ctx)

    var w = TileTensor[mut=False](w_dev, row_major(Coord(Idx[N], Idx[K])))
    var act = TileTensor[mut=False](act_dev, row_major(Coord(Idx[1], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(Idx[1], Idx[N])))
    var bias = TileTensor[mut=False](bias_dev, row_major(Coord(Idx[1], Idx[N])))

    @__parameter
    @always_inline
    @__copy_capture(c, bias)
    def bias_epilogue[
        _dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c.store[width=width](
            (idx[0], idx[1]),
            val.cast[c_type]()
            + bias.load[width=width](Coord(idx)).cast[c_type](),
        )

    comptime epilogue = Optional[elementwise_epilogue_type](
        bias_epilogue
    ) if with_epilogue else None

    comptime one_row_kernel = gemv_kernel_vector[
        c_type,
        dtype,
        dtype,
        type_of(c).LayoutType,
        type_of(w).LayoutType,
        type_of(act).LayoutType,
        type_of(c).Engine,
        type_of(w).Engine,
        type_of(act).Engine,
        simd_width=simd_width,
        transpose_b=True,
        elementwise_lambda_fn=epilogue,
        check_bounds=check_bounds_k,
    ]
    comptime multirow_kernel = gemv_kernel_vector_multirow[
        c_type,
        dtype,
        dtype,
        type_of(c).LayoutType,
        type_of(w).LayoutType,
        type_of(act).LayoutType,
        type_of(c).Engine,
        type_of(w).Engine,
        type_of(act).Engine,
        simd_width=simd_width,
        rows_per_warp=rows_per_warp,
        transpose_b=True,
        elementwise_lambda_fn=epilogue,
        check_bounds=check_bounds_k,
    ]

    # Distinct sentinels: a row no kernel writes shows up as a mismatch.
    c_dev.enqueue_fill(Scalar[c_type](1))
    ctx.enqueue_function[one_row_kernel](
        c,
        w,
        act,
        Int32(N),
        Int32(1),
        Int32(K),
        grid_dim=ceildiv(N, one_row_block_dim // WARP_SIZE),
        block_dim=one_row_block_dim,
    )
    ctx.enqueue_copy(ref_host, c_dev)

    c_dev.enqueue_fill(Scalar[c_type](-1))
    ctx.enqueue_function[multirow_kernel](
        c,
        w,
        act,
        Int32(N),
        Int32(1),
        Int32(K),
        grid_dim=ceildiv(N, rows_per_warp * warps_per_block),
        block_dim=WARP_SIZE * warps_per_block,
    )
    ctx.enqueue_copy(new_host, c_dev)
    ctx.synchronize()

    var mismatches = 0
    for i in range(N):
        if ref_host[i] != new_host[i]:
            mismatches += 1
    print("  mismatches vs one-row kernel:", mismatches, "/", N)
    if mismatches != 0:
        raise "multi-row GEMV is not bit-exact with the one-row kernel"


def test_matches_host_reference[
    c_type: DType, dtype: DType, N: Int, K: Int
](ctx: DeviceContext) raises:
    """Checks a non-dividing N against a host dot product.

    Inputs are small integers, so the f32 accumulation is exact and the host
    result is independent of summation order.
    """
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime check_bounds_k = K % (WARP_SIZE * simd_width) != 0
    comptime rows_per_warp = _GEMV_MULTIROW_ROWS
    comptime warps_per_block = 8

    print("N", N, "K", K, c_type, "host reference, rows", rows_per_warp)

    var w_dev = ctx.enqueue_create_buffer[dtype](N * K)
    var act_dev = ctx.enqueue_create_buffer[dtype](K)
    var c_dev = ctx.enqueue_create_buffer[c_type](N)
    var w_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    var act_host = ctx.enqueue_create_host_buffer[dtype](K)
    var c_host = ctx.enqueue_create_host_buffer[c_type](N)

    _fill_launch[dtype, integral=True](w_dev, N * K, 11, ctx)
    _fill_launch[dtype, integral=True](act_dev, K, 13, ctx)
    c_dev.enqueue_fill(Scalar[c_type](-1))

    var w = TileTensor[mut=False](w_dev, row_major(Coord(Idx[N], Idx[K])))
    var act = TileTensor[mut=False](act_dev, row_major(Coord(Idx[1], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(Idx[1], Idx[N])))

    comptime multirow_kernel = gemv_kernel_vector_multirow[
        c_type,
        dtype,
        dtype,
        type_of(c).LayoutType,
        type_of(w).LayoutType,
        type_of(act).LayoutType,
        type_of(c).Engine,
        type_of(w).Engine,
        type_of(act).Engine,
        simd_width=simd_width,
        rows_per_warp=rows_per_warp,
        transpose_b=True,
        check_bounds=check_bounds_k,
    ]
    ctx.enqueue_function[multirow_kernel](
        c,
        w,
        act,
        Int32(N),
        Int32(1),
        Int32(K),
        grid_dim=ceildiv(N, rows_per_warp * warps_per_block),
        block_dim=WARP_SIZE * warps_per_block,
    )

    ctx.enqueue_copy(w_host, w_dev)
    ctx.enqueue_copy(act_host, act_dev)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    var mismatches = 0
    for row in range(N):
        var acc = Float32(0)
        for i in range(K):
            acc += (
                w_host[row * K + i].cast[.float32]()
                * act_host[i].cast[.float32]()
            )
        if c_host[row] != acc.cast[c_type]():
            mismatches += 1
    print("  mismatches vs host reference:", mismatches, "/", N)
    if mismatches != 0:
        raise "multi-row GEMV disagrees with the host reference"


def main() raises:
    with DeviceContext() as ctx:
        # Production shape of the wide-N shallow-K path (bf16 in, bf16 out).
        for seed in [1, 7, 24301]:
            test_matches_one_row_kernel[.bfloat16, .bfloat16, 262144, 256](
                ctx, seed
            )

        test_matches_one_row_kernel[
            .bfloat16, .bfloat16, 262144, 256, with_epilogue=True
        ](ctx, 1)

        # f32 output, and a K deep enough to halve the row tile.
        test_matches_one_row_kernel[.float32, .bfloat16, 262144, 512](ctx, 1)

        # N divides neither the row tile nor the warp.
        test_matches_one_row_kernel[.bfloat16, .bfloat16, 100003, 256](ctx, 1)
        test_matches_host_reference[.bfloat16, .bfloat16, 100003, 256](ctx)
