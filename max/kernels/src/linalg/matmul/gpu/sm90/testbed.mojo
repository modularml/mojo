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

"""Provides a testbed for validating SM90 warp-specialized matmul kernels against a vendor reference."""

from std.math import ceildiv
from std.sys import align_of

from max.gpu.host import DeviceContext
from std.memory import dealloc
from std.memory.alloc import Layout as AllocLayout
from layout import Coord, CoordLike, Idx, TileTensor, row_major
from std.utils.index import IndexList
from internal_utils import assert_almost_equal, assert_with_measure
from std.random import rand
from internal_utils._measure import relative_difference
from std.collections import OptionalReg
from std.utils.index import IndexList

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import MatmulConfig
from ...vendor.blas import Backend
from ...vendor.blas import matmul as vendor_matmul
from ..tile_scheduler import MatmulSchedule
from .matmul import warp_specialize_gemm_with_multicasting


def test_matmul_sm90[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    cluster_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    num_pipeline_stages: Int = 4,
    transpose_b: Bool = True,
    partitioned_multicast: Bool = False,
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    default_epilogue: Bool = False,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    measure_threshold: Optional[Float64] = None,
    backend: Backend = Backend.CUBLAS,
    k_group_size: Int = 1,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    """Runs an SM90 warp-specialized matmul and validates the result against a vendor reference.

    Allocates host and device buffers for the operands, initializes them with
    random data, launches the warp-specialized GEMM with multicasting, and compares
    the output against a vendor (cuBLAS) reference using an elementwise tolerance
    check.

    Parameters:
        MType: Coordinate-like type encoding the M dimension extent (inferred).
        NType: Coordinate-like type encoding the N dimension extent (inferred).
        KType: Coordinate-like type encoding the K dimension extent (inferred).
        a_type: Element dtype of the left-hand operand.
        b_type: Element dtype of the right-hand operand.
        c_type: Element dtype of the output matrix.
        cluster_shape: Thread block cluster shape as `(N, M, K)`.
        block_tile_shape: Block tile shape as `(BM, BN, BK)`.
        wgmma_shape: WGMMA instruction shape as `(M, N, K)`.
        num_consumer: Number of consumer warps in the warp-specialized pipeline.
        num_pipeline_stages: Number of software pipeline stages.
        transpose_b: Whether the right-hand operand is stored transposed.
        partitioned_multicast: Whether to use partitioned instead of broadcast multicast.
        grid_shape: Optional explicit grid shape override.
        use_tma_store: Whether to use TMA store for the epilogue.
        schedule: Tile scheduler hint controlling launch order.
        default_epilogue: Whether to use the default store-based epilogue.
        elementwise_compute_lambda_fn: Optional elementwise compute lambda applied in the epilogue.
        measure_threshold: Optional relative-difference threshold for a measured assertion.
        backend: Vendor backend used for the reference matmul.
        k_group_size: Group size used for grouped matmul along the K dimension.

    Args:
        ctx: Device context used for device allocations and kernel dispatch.
        m: Number of rows of the output matrix (M dimension).
        n: Number of columns of the output matrix (N dimension).
        k: Contraction dimension shared by the operands (K dimension).
    """
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    comptime CLUSTER_N = cluster_shape[0]
    comptime CLUSTER_M = cluster_shape[1]

    # Calculate sizes
    var a_size = M * K
    var b_size = N * K if transpose_b else K * N
    var c_size = M * N

    # Host allocations
    var a_host_alloc = alloc(
        AllocLayout[Scalar[a_type]](count=a_size)
    ).into_managed()
    var a_host = a_host_alloc.unsafe_ptr()
    var b_host_alloc = alloc(
        AllocLayout[Scalar[b_type]](count=b_size)
    ).into_managed()
    var b_host = b_host_alloc.unsafe_ptr()
    var c_host_alloc = alloc(
        AllocLayout[Scalar[c_type]](count=c_size)
    ).into_managed()
    var c_host = c_host_alloc.unsafe_ptr()
    var c_host_ref_alloc = alloc(
        AllocLayout[Scalar[c_type]](count=c_size)
    ).into_managed()
    var c_host_ref: UnsafePointer[
        Scalar[c_type], origin_of(c_host_ref_alloc)
    ] = c_host_ref_alloc.unsafe_ptr()

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_dev_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)

    # Construct TileTensors for device buffers
    var a_tensor = TileTensor(a_dev_buffer, row_major(Coord(m, k))).as_immut()
    var b_tensor = TileTensor(
        b_dev_buffer,
        row_major(
            Coord(
                Idx[NType.static_value if transpose_b else KType.static_value],
                Idx[KType.static_value if transpose_b else NType.static_value],
            ),
        ),
    ).as_immut()
    var c_tensor = TileTensor(c_dev_buffer, row_major(Coord(m, n)))
    var c_ref_tensor = TileTensor(c_dev_ref_buffer, row_major(Coord(m, n)))

    # Initialize matmul operands
    rand(a_host, a_size)
    rand(b_host, b_size)

    # Move operands to the Device
    ctx.enqueue_copy(a_dev_buffer, a_host)
    ctx.enqueue_copy(b_dev_buffer, b_host)

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    print(
        "wgmma_shape",
        wgmma_shape,
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
        " MULTICAST MODE: ",
        "PARTITIONED" if partitioned_multicast else "BROADCAST",
        "USE TMA STORE: ",
        use_tma_store,
    )

    assert (ceildiv(M, BM) % (CLUSTER_M)) == 0, String(
        "Number of blocks on M axis should be multiple of cluster dim. M",
        "(M // BM=",
        String(M // BM),
        ") CLUSTER SIZE:",
        String(CLUSTER_M),
    )

    assert (ceildiv(N, BN) % (CLUSTER_N)) == 0, String(
        "Number of blocks on M axis should be multiple of cluster dim. N",
        "N // BN=(",
        String(N // BN),
        ") CLUSTER SIZE:",
        String(CLUSTER_N),
    )

    @__parameter
    @always_inline
    @__copy_capture(c_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_tensor.store_linear[alignment=alignment](
            idx, rebind[SIMD[c_type, width]](val)
        )

    comptime elf = Optional[elementwise_epilogue_type](
        epilogue_fn
    ) if default_epilogue and elementwise_compute_lambda_fn is None else None

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=block_tile_shape,
        mma_shape=wgmma_shape,
        cluster_shape=cluster_shape,
        num_pipeline_stages=num_pipeline_stages,
        num_consumer=num_consumer,
        partitioned_multicast=partitioned_multicast,
        k_group_size=k_group_size,
    )

    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        config=matmul_config,
        schedule=schedule,
        grid_shape=grid_shape,
        use_tma_store=use_tma_store,
        elementwise_lambda_fn=elf,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
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

    vendor_matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.enqueue_copy(c_host, c_dev_buffer)
    ctx.enqueue_copy(c_host_ref, c_dev_ref_buffer)
    ctx.synchronize()

    comptime if elementwise_compute_lambda_fn:
        # Apply the compute lambda directly on the reference tensor
        comptime compute_lambda = elementwise_compute_lambda_fn.value()
        for i in range(M):
            for j in range(N):
                c_host_ref[i * N + j] = compute_lambda(
                    IndexList[2](i, j),
                    c_host_ref[i * N + j],
                )

    comptime if measure_threshold:
        assert_with_measure[relative_difference](
            c_host,
            c_host_ref,
            c_size,
            threshold=measure_threshold.value(),
        )

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host,
        c_host_ref,
        c_size,
        atol=0.0001,
        rtol=rtol,
    )

    # Cleanup host pointers
    dealloc(a_host_alloc^)
    dealloc(b_host_alloc^)
    dealloc(c_host_alloc^)
    dealloc(c_host_ref_alloc^)
