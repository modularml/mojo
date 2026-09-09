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
from std.collections import Optional
from std.random import shuffle, seed
from std.sys import align_of, size_of, argv

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from internal_utils import assert_almost_equal
from std.random import rand
from layout import TileTensor, Coord, CoordLike, row_major, Idx
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from linalg.utils import elementwise_compute_lambda_type

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test_matmul_sm100_epilogue[
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
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
    test_lambda_fn: Bool = False,
    register_based_epilogue: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
](
    ctx: DeviceContext,
    m: MType,
    n: NType,
    k: KType,
    is_benchmark: Bool = False,
) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        t"in/out dtypes=({a_type}, {b_type}, {c_type})  problem shape=({M},"
        t" {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape} register_based_epilogue={register_based_epilogue} swapAB={swapAB} k_group_size={k_group_size}"
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * Int(k.value())
    var b_size = Int(n.value()) * Int(k.value())
    var c_size = Int(m.value()) * Int(n.value())

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_copy_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

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
    def test_lambda_add_coords_prod[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        # this function helps us determine if the provided indexes are correct
        # while also testing arithmetic operations
        var x = c_tensor_lt.load[width=width](idx).cast[_dtype]()
        var y = val * x
        return y

    seed(1234)
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    var scales: List[Int32] = [-2, -1, 0, 1, 2]

    for i in range(M):
        for j in range(N):
            shuffle(scales)
            comptime assert c_host.flat_rank == 2
            c_host[i, j] = Scalar[c_type](scales[0])
            c_host_copy[i, j] = c_host[i, j]

    # Move operands to the Device
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
        k_group_size=k_group_size,
        register_based_epilogue=register_based_epilogue,
    )

    comptime optional_lambda_fn = Optional[elementwise_compute_lambda_type](
        test_lambda_add_coords_prod
    ) if test_lambda_fn else None

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {var c_tensor, var a_tensor, var b_tensor, imm}:
        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=matmul_config,
            elementwise_compute_lambda_fn=optional_lambda_fn,
        ](c_tensor, a_tensor, b_tensor, ctx)

    if is_benchmark:
        comptime nrun = 50

        # Warmup
        kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time(kernel_launch, nrun)) / Float64(
            nrun
        )
        var sectime = nstime / 1000000
        print(nrun, "runs avg", sectime, "ms")
    else:
        kernel_launch(ctx)

    if not is_benchmark:
        comptime assert a_type != .float8_e4m3fn or transpose_b, (
            "Testing is only supported for transposed_b==True when"
            " a_type==float8_e4m3fn. Add the non-transposed case if needed."
        )

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
        def test_lambda_add_coords_prod_local[
            _dtype: DType,
            width: SIMDLength,
            *,
            alignment: Int = align_of[SIMD[_dtype, width]](),
        ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
            _dtype, width
        ]:
            return val * c_host_copy_lt.load[width=width](idx).cast[_dtype]()

        comptime if optional_lambda_fn:
            # Apply the compute lambda directly on the reference tensor
            # alias compute_lambda = elementwise_compute_lambda_fn.value()
            for i in range(M):
                for j in range(N):
                    comptime assert c_host_ref.flat_rank == 2
                    c_host_ref[i, j] = test_lambda_add_coords_prod_local(
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

        print("\n=== TEST PASSED ===\n")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main() raises:
    comptime dtype = DType.bfloat16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[dtype]())
    comptime MMA_K = 16
    var is_bench = is_benchmark()

    with DeviceContext() as ctx:
        comptime for register_based_epilogue in [True, False]:
            # swapAB with epilogue tests (2SM)
            comptime for mma_m_scale in range(1, 3):
                comptime for mma_n_scale in range(1, 17):
                    comptime block_tile_shape = Index(
                        64 * mma_m_scale, 8 * mma_n_scale, BK
                    )
                    comptime umma_shape = Index(
                        128 * mma_m_scale, 16 * mma_n_scale, MMA_K
                    )

                    test_matmul_sm100_epilogue[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=2,
                        test_lambda_fn=True,
                        register_based_epilogue=register_based_epilogue,
                        swapAB=True,
                        k_group_size=2,
                    ](
                        ctx,
                        Int(100),
                        Idx[2560],
                        Idx[8192],
                        is_benchmark=is_bench,
                    )

                    test_matmul_sm100_epilogue[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=2,
                        test_lambda_fn=True,
                        register_based_epilogue=register_based_epilogue,
                        swapAB=True,
                    ](
                        ctx,
                        Int(17),
                        Idx[1024],
                        Idx[1024],
                        is_benchmark=is_bench,
                    )

            # swapAB with epilogue tests (1SM)
            # we support all range of mma_n_scales in range(1, 33) but the test will time out so we only test a subset
            comptime for mma_m in [64, 128]:
                comptime for mma_n in [8, 16, 32, 40, 48, 64, 88, 104, 128]:
                    comptime block_tile_shape = Index(mma_m, mma_n, BK)
                    comptime umma_shape = Index(mma_m, mma_n, MMA_K)

                    test_matmul_sm100_epilogue[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=1,
                        test_lambda_fn=True,
                        register_based_epilogue=register_based_epilogue,
                        swapAB=True,
                        k_group_size=2,
                    ](
                        ctx,
                        Int(1000),
                        Idx[1024],
                        Idx[1024],
                        is_benchmark=is_bench,
                    )

                    test_matmul_sm100_epilogue[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                        cta_group=1,
                        test_lambda_fn=True,
                        register_based_epilogue=register_based_epilogue,
                        swapAB=True,
                    ](
                        ctx,
                        Int(512),
                        Idx[4096],
                        Idx[1024 + 16],
                        is_benchmark=is_bench,
                    )
