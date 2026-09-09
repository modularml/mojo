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
from std.random import random_float64
from std.sys import align_of, size_of, get_defined_bool

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from internal_utils import assert_almost_equal
from std.random import rand
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from linalg.utils import elementwise_compute_lambda_type

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


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
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    print(
        String(
            "in/out dtypes=(",
            a_type,
            ", ",
            b_type,
            ", ",
            c_type,
            ") ",
            " problem shape=(",
            Int(m.value()),
            ", ",
            Int(n.value()),
            ", ",
            Int(k.value()),
            ") ",
            "mma_shape=",
            mma_shape,
            " block_tile_shape=",
            block_tile_shape,
            " register_based_epilogue=",
            register_based_epilogue,
            " swapAB=",
            swapAB,
            " k_group_size=",
            k_group_size,
        )
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
    var b_size = (
        Int(n.value())
        * Int(k.value()) if transpose_b else Int(k.value())
        * Int(n.value())
    )
    var c_size = Int(m.value()) * Int(n.value())

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)
    var c_host_copy_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_copy = TileTensor(c_host_copy_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    var c_tensor_lt = c_tensor.to_layout_tensor()

    @__parameter
    @always_inline
    @__copy_capture(c_tensor_lt)
    def test_lambda_add_coords_summ[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        # this function helps us determine if the provided indexes are correct
        # while also testing arithmetic operations
        return val + c_tensor_lt.load[width=width](idx).cast[_dtype]()

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            comptime assert c_host.flat_rank == 2
            c_host[i, j] = Scalar[c_type](random_float64(-1, 1))
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
        test_lambda_add_coords_summ
    ) if test_lambda_fn else None

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        elementwise_compute_lambda_fn=optional_lambda_fn,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        ctx,
    )

    comptime assert a_type != .float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
    )

    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var c_ref_lt = c_ref_tensor.to_layout_tensor()

    vendor_blas.matmul(
        ctx,
        c_ref_lt,
        a_lt,
        b_lt,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    var c_tensor_host_lt = c_host_copy.to_layout_tensor()

    @__parameter
    @always_inline
    @__copy_capture(c_tensor_host_lt)
    def test_lambda_add_coords_summ_local[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        return val + c_tensor_host_lt.load[width=width](idx).cast[_dtype]()

    comptime if optional_lambda_fn:
        # Apply the compute lambda directly on the reference tensor
        # alias compute_lambda = elementwise_compute_lambda_fn.value()
        for i in range(Int(m.value())):
            for j in range(Int(n.value())):
                comptime assert c_host_ref.flat_rank == 2
                c_host_ref[i, j] = test_lambda_add_coords_summ_local(
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

    # Cleanup
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


# Quick mode: reduce test configs for faster iteration
# QUICK_TEST=True: 48 tests (8 configs × 6 sizes) - ~30 seconds
# FASTER_TEST=True: 8 tests (4 configs × 2 sizes) - ~5 seconds
comptime QUICK_TEST = get_defined_bool["QUICK_TEST", False]()
comptime FASTER_TEST = get_defined_bool["FASTER_TEST", False]()


def main() raises:
    comptime dtype = DType.bfloat16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[dtype]())
    comptime MMA_K = 16

    # FASTER mode: only 1 mma_n_scale, QUICK mode: 2-4, full: 1-16
    comptime n_scale_max = 3 if FASTER_TEST else (5 if QUICK_TEST else 17)

    with DeviceContext() as ctx:
        comptime for mma_m_scale in range(1, 3):
            comptime for mma_n_scale in range(1, n_scale_max):
                # Quick/Faster mode: skip odd n_scale values
                comptime if (
                    QUICK_TEST or FASTER_TEST
                ) and mma_n_scale % 2 != 0:
                    continue

                comptime block_tile_shape = Index(
                    64 * mma_m_scale, 8 * mma_n_scale, BK
                )

                comptime umma_shape = Index(
                    128 * mma_m_scale, 16 * mma_n_scale, MMA_K
                )

                comptime for register_based_epilogue in [True, False]:
                    # Helper to run test with varying cluster/k_group/sizes
                    @__parameter
                    def run[
                        MType: CoordLike,
                        NType: CoordLike,
                        KType: CoordLike,
                        //,
                        cluster_m: Int,
                        cluster_n: Int,
                        k_group: Int = 1,
                    ](m: MType, n: NType, k: KType) raises:
                        test_matmul_sm100_epilogue[
                            dtype,
                            dtype,
                            .bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](
                                Int32(cluster_m), Int32(cluster_n), 1
                            ),
                            cta_group=2,
                            test_lambda_fn=True,
                            register_based_epilogue=register_based_epilogue,
                            k_group_size=k_group,
                        ](ctx, m, n, k)

                    # FASTER mode: 2 key test cases only
                    run[4, 4](Int(1000), Idx[1024], Idx[1024])

                    comptime if not FASTER_TEST:
                        run[4, 4](Int(512), Idx[4096], Idx[1024])
                        run[4, 4, k_group=2](Int(500), Idx[2048], Idx[4096])
                        run[8, 2](Int(1024), Idx[256], Idx[128])

                    run[2, 2](Idx[1024], Idx[1024], Idx[2048])

                    comptime if not FASTER_TEST:
                        run[4, 4](Int(8192), Idx[2560], Idx[8192])
