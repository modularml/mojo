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
"""Coverage for the transposed elementwise epilogue's per-row bounds check.

Under swapAB the kernel's M dimension is the problem's N, so a fragment's
(top_row, bot_row) pair sits 8 apart along N and each row needs its own
bound check. Warp row origins step by 16 or 32 depending on the tcgen05
data-path layout, so any N that is not a multiple of 16 puts one warp's
pairs astride N under every layout.

The cases below cover all three row-origin layouts the epilogue selects
between: layout F (MMA_M 64, cta_group 1), layout A/D (MMA_M 128,
cta_group 1) and layout B (cta_group 2).

test_matmul_sm100_partial_n_tile_epilogue covers the compute-lambda path.
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
from linalg.utils import elementwise_epilogue_type
import linalg.matmul.vendor.blas as vendor_blas
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def test_transposed_epilogue_row_straddle[
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
    swapAB: Bool = True,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        t"in/out=({a_type},{b_type},{c_type}) shape=({M},{N},{K})"
        t" mma={mma_shape} block_tile={block_tile_shape} swapAB={swapAB}"
    )

    comptime assert (
        NType.static_value % 16 != 0
    ), "N must not be a multiple of 16 so a row pair straddles N"
    comptime assert (
        NType.static_value * size_of[c_type]() % 16 == 0
    ), "N must keep C's row stride TMA-aligned"

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

    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

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
    def store_epilogue[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_tensor_lt.store[width=width](idx, val.cast[c_type]())

    seed(1234)
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    # C is zeroed, so a dropped store reads back as 0 against a reference
    # the random operands make nonzero.
    for i in range(M):
        for j in range(N):
            comptime assert c_host.flat_rank == 2
            c_host[i, j] = Scalar[c_type](0)

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

    comptime optional_lambda_fn = Optional[elementwise_epilogue_type](
        store_epilogue
    )

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        elementwise_lambda_fn=optional_lambda_fn,
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

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.0001,
        rtol=1e-2,
    )

    print("=== TEST PASSED ===")

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main() raises:
    comptime dtype = DType.bfloat16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[dtype]())
    comptime MMA_K = 16

    with DeviceContext() as ctx:
        # Layout F: warp row origins step by 16.
        test_transposed_epilogue_row_straddle[
            dtype,
            dtype,
            dtype,
            Index(64, 64, BK),
            Index(64, 64, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            cta_group=1,
        ](ctx, Int(64), Idx[136], Idx[128])

        # Layout A/D: warp row origins step by 32. N and M here are the
        # shape the bug was reported on, which leaves an 8-row partial tile.
        test_transposed_epilogue_row_straddle[
            dtype,
            dtype,
            dtype,
            Index(128, 64, BK),
            Index(128, 64, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            cta_group=1,
        ](ctx, Int(300), Idx[776], Idx[128])

        # Layout B: warp row origins step by 32 within a CTA pair, and the
        # odd warps offset along the column axis instead.
        test_transposed_epilogue_row_straddle[
            dtype,
            dtype,
            dtype,
            Index(64, 32, BK),
            Index(128, 64, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            cta_group=2,
        ](ctx, Int(300), Idx[776], Idx[128])
