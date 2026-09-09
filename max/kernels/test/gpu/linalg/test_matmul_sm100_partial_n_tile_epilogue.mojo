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
"""Coverage for MXF-338: SM100 matmul with a fused compute-lambda epilogue
on a shape where ``block_tile_N`` does not evenly divide ``N``.

Shape: M=64, N=128, K=128, ``block_tile_N=88`` — last N-tile spans
[88, 176), so cols [128, 176) are OOB. ``apply_to_fragment`` must skip
the lambda for those lanes; otherwise, the lambda's load reads past the
broadcast operand and faults whenever the next page is unmapped.

Both ``transpose_c`` branches of ``apply_to_fragment`` are exercised
(``swapAB=False/True``); each branch has its own bounds checks.
"""

from std.collections import Optional
from std.sys import align_of, size_of
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from internal_utils import assert_almost_equal
from std.random import rand, seed
from layout import TileTensor, Coord, CoordLike, row_major, Idx
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from linalg.utils import elementwise_compute_lambda_type
import linalg.matmul.vendor.blas as vendor_blas
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def test_partial_n_tile_compute_epilogue[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    swapAB: Bool = False,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        t"MXF-338 regression: in/out=({a_type},{b_type},{c_type})"
        t" shape=({M},{N},{K}) mma={mma_shape} block_tile={block_tile_shape}"
        t" swapAB={swapAB}"
    )

    comptime assert block_tile_shape[1] == 88, (
        "fixed to block_tile_N=88 (the partial-N-tile path); N must not be"
        " a multiple of 88"
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = M * K
    var b_size = N * K if transpose_b else K * N
    var c_size = M * N

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_host_ref_ptr = alloc[Scalar[c_type]](c_size)
    var c_host_copy_ptr = alloc[Scalar[c_type]](c_size)

    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)
    var c_host_copy = TileTensor(c_host_copy_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)

    var a_tensor = TileTensor(a_device, a_shape)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    var c_tensor_lt = c_tensor.to_layout_tensor()

    @__parameter
    @always_inline
    @__copy_capture(c_tensor_lt)
    def in_bounds_compute_lambda[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        return val + c_tensor_lt.load[width=width](idx).cast[_dtype]()

    seed(1234)
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    for i in range(M):
        for j in range(N):
            comptime assert c_host.flat_rank == 2
            c_host[i, j] = Scalar[c_type](0)
            c_host_copy[i, j] = c_host[i, j]

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        cta_group=cta_group,
        AB_swapped=swapAB,
    )

    comptime optional_lambda_fn = Optional[elementwise_compute_lambda_type](
        in_bounds_compute_lambda
    )

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        elementwise_compute_lambda_fn=optional_lambda_fn,
    ](c_tensor, a_tensor, b_tensor, ctx)

    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    vendor_blas.matmul(
        ctx,
        c_ref_tensor_lt,
        a_lt,
        b_lt,
        c_row_major=True,
        transpose_b=transpose_b,
    )
    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    var c_host_copy_lt = c_host_copy.to_layout_tensor()

    @__parameter
    @always_inline
    @__copy_capture(c_host_copy_lt)
    def in_bounds_compute_lambda_local[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        return val + c_host_copy_lt.load[width=width](idx).cast[_dtype]()

    for i in range(M):
        for j in range(N):
            comptime assert c_host_ref.flat_rank == 2
            c_host_ref[i, j] = in_bounds_compute_lambda_local(
                IndexList[2](i, j), c_host_ref[i, j]
            )

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )

    print("=== MXF-338 regression: TEST PASSED ===")

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    c_host_copy_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main() raises:
    comptime dtype = DType.bfloat16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[dtype]())
    comptime MMA_K = 16

    comptime block_tile_shape_1sm = Index(64, 88, BK)
    comptime mma_shape_1sm = Index(64, 88, MMA_K)

    with DeviceContext() as ctx:
        # transpose_c=False: `if top_col >= self.N: return` branch.
        test_partial_n_tile_compute_epilogue[
            dtype,
            dtype,
            dtype,
            block_tile_shape_1sm,
            mma_shape_1sm,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            cta_group=1,
            swapAB=False,
        ](ctx, Int(64), Idx[128], Idx[128])

        # transpose_c=True: the per-row `top_row`/`bot_row` bound checks.
        test_partial_n_tile_compute_epilogue[
            dtype,
            dtype,
            dtype,
            block_tile_shape_1sm,
            mma_shape_1sm,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            cta_group=1,
            swapAB=True,
        ](ctx, Int(64), Idx[128], Idx[128])
