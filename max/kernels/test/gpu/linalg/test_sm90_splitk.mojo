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
from std.collections import OptionalReg

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from std.memory import alloc
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from internal_utils import (
    assert_almost_equal,
    assert_with_measure,
)
from std.random import rand
from internal_utils._measure import relative_difference
from linalg.matmul.gpu.sm90.matmul import warp_specialize_gemm_with_multicasting
from linalg.matmul.gpu.tile_scheduler import RasterOrder
from linalg.utils_gpu import MatmulConfig

from std.utils.index import Index, IndexList


def test_warp_specialize_gemm_with_multicasting[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    block_tile_shape: IndexList[3],
    a_type: DType,
    b_type: DType,
    c_type: DType,
    cluster_shape: IndexList[3],
    num_pipeline_stages: Int = 4,
    transpose_b: Bool = True,
    partitioned_multicast: Bool = False,
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    splits: Int = 2,
](ctx: DeviceContext, m: MType, n: NType, k: KType,) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime CLUSTER_N = cluster_shape[0]
    comptime CLUSTER_M = cluster_shape[1]

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

    # Initialize matmul operands
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    # Move operands to the Device

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    comptime num_consumer: Int = 1 if BM == 64 else 2

    comptime wgmma_shape = Index(
        64, BN, 32
    ) if a_type == .float8_e4m3fn else Index(64, BN, 16)

    print(
        "wgmma_n",
        BN,
        a_type,
        "x",
        b_type,
        "x",
        c_type,
        " : PROBLEM SHAPE (M,N,K): (",
        M,
        "x",
        N,
        "x",
        K,
        ") - ",
        "BLOCKS SHAPE (BM,BN,BK): (",
        BM,
        "x",
        BN,
        "x",
        BK,
        ") - ",
        "CLUSTER DIMS (M,N): (",
        CLUSTER_M,
        "x",
        CLUSTER_N,
        ") NUM CONSUMERS: ",
        num_consumer,
        " NUM PIPELINE STAGES: ",
        num_pipeline_stages,
        " SPLITS: ",
        splits,
        " MULTICAST MODE: ",
        "PARTITIONED" if partitioned_multicast else "BROADCAST",
    )

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=block_tile_shape,
        mma_shape=wgmma_shape,
        cluster_shape=cluster_shape,
        num_pipeline_stages=num_pipeline_stages,
        num_consumer=num_consumer,
        partitioned_multicast=partitioned_multicast,
    )

    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        config=matmul_config,
        use_tma_store=use_tma_store,
        splits=splits,
        raster_order=RasterOrder.AlongN,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        ctx,
    )

    ctx.synchronize()

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

    assert_with_measure[relative_difference](
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        threshold=0.001,
    )

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main() raises:
    with DeviceContext() as ctx:
        # NOTE: please note that cublaslt handle should be used for fp8-e4m3fn and cublas handle for bfloat16
        # because cublas does not support float8-e4m3fn. Also, fp8 tests should be run first and then bfloat16 tests
        # otherwise we will get unhandled exception error.

        print("FLOAT8 GEMM TESTS")
        test_warp_specialize_gemm_with_multicasting[
            Index(64, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            num_pipeline_stages=6,
            partitioned_multicast=False,
            splits=2,
        ](ctx, Int(33), Idx[2304], Idx[2048])

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 2, 1),
            num_pipeline_stages=2,
            partitioned_multicast=False,
            splits=2,
        ](ctx, Int(64), Idx[384], Idx[512])

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            num_pipeline_stages=2,
            partitioned_multicast=False,
            splits=2,
        ](ctx, Int(64), Idx[384], Idx[512])

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 80, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 2, 1),
            num_pipeline_stages=2,
            partitioned_multicast=False,
            splits=4,
        ](ctx, Int(64), Idx[2560], Idx[8192])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=False,
            num_pipeline_stages=4,
            splits=4,
        ](
            ctx,
            Idx[4096],
            Idx[2560],
            Idx[8192],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=False,
            num_pipeline_stages=4,
        ](
            ctx,
            Idx[512],
            Idx[8192],
            Idx[2048],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 112, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            num_pipeline_stages=4,
            partitioned_multicast=False,
        ](
            ctx,
            Idx[512],
            Idx[14336],
            Idx[4096],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=4,
            splits=2,
        ](ctx, Int(199), Idx[512], Idx[1024])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 2, 1),
            partitioned_multicast=False,
            num_pipeline_stages=1,
            splits=2,
        ](ctx, Int(200), Idx[256], Idx[256])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(257), Idx[384], Idx[256])
        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(257), Idx[384], Idx[256])
        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(257), Idx[384], Idx[256])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(255), Idx[384], Idx[256])
        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(255), Idx[384], Idx[256])
        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(255), Idx[384], Idx[256])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(129), Idx[512], Idx[256])
        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(129), Idx[512], Idx[256])
        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(129), Idx[512], Idx[256])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(127), Idx[512], Idx[256])
        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 2, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(127), Idx[512], Idx[256])
        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 128),
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=True,
            num_pipeline_stages=2,
            splits=2,
        ](ctx, Int(127), Idx[512], Idx[256])

        print("BFLOAT16 GEMM TESTS")
        test_warp_specialize_gemm_with_multicasting[
            Index(64, 128, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            num_pipeline_stages=2,
            partitioned_multicast=False,
            splits=2,
        ](ctx, Int(64), Idx[384], Idx[512])

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 80, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            num_pipeline_stages=8,
            partitioned_multicast=False,
            splits=4,
        ](ctx, Int(64), Idx[2560], Idx[8192])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 128, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            num_pipeline_stages=4,
            partitioned_multicast=False,
        ](
            ctx,
            Int(2048),
            Idx[8192],
            Idx[8192],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 80, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            num_pipeline_stages=6,
            partitioned_multicast=False,
        ](
            ctx,
            Int(2048),
            Idx[2560],
            Idx[8192],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 256, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            num_pipeline_stages=4,
            partitioned_multicast=False,
            splits=2,
        ](
            ctx,
            Int(64),
            Idx[2560],
            Idx[8192],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 80, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            num_pipeline_stages=6,
            partitioned_multicast=False,
            splits=4,
        ](
            ctx,
            Int(64),
            Idx[2560],
            Idx[8192],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 128, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            num_pipeline_stages=4,
            partitioned_multicast=False,
            splits=2,
        ](
            ctx,
            Int(64),
            Idx[8192],
            Idx[2048],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 128, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            num_pipeline_stages=2,
            partitioned_multicast=False,
            splits=2,
        ](ctx, Int(64), Idx[384], Idx[512])

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 128, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            num_pipeline_stages=2,
            partitioned_multicast=True,
            splits=2,
        ](ctx, Int(64), Idx[384], Idx[512])

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 80, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            num_pipeline_stages=2,
            partitioned_multicast=True,
            splits=4,
        ](ctx, Int(64), Idx[2560], Idx[8192])

        test_warp_specialize_gemm_with_multicasting[
            Index(64, 80, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            num_pipeline_stages=2,
            partitioned_multicast=True,
            splits=4,
        ](ctx, Int(64), Idx[2560], Idx[8192])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 256, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=False,
            splits=4,
        ](ctx, Int(8192), Idx[8192], Idx[2048])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 256, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=False,
            splits=4,
        ](ctx, Int(4096), Idx[8192], Idx[2048])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 256, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=False,
            use_tma_store=True,
            splits=4,
        ](ctx, Int(4096), Idx[8192], Idx[2048])

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 256, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=False,
            splits=4,
        ](
            ctx,
            Int(128),
            Idx[14336],
            Idx[8192],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 256, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            partitioned_multicast=False,
        ](
            ctx,
            Idx[8192],
            Idx[8192],
            Idx[7168],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 256, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            partitioned_multicast=False,
            splits=2,
        ](
            ctx,
            Idx[8192],
            Idx[8192],
            Idx[7168],
        )

        test_warp_specialize_gemm_with_multicasting[
            Index(128, 256, 64),
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 2, 1),
            partitioned_multicast=False,
            splits=4,
        ](
            ctx,
            Idx[8192],
            Idx[8192],
            Idx[7168],
        )
