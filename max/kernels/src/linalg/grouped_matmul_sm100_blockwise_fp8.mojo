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

"""Provides blockwise-scaled FP8 grouped GEMM kernels for SM100 (B200) GPUs."""

from std.collections import Optional
from std.math import ceildiv, gcd
from std.math.uutils import umod, ufloordiv
from std.sys import align_of, size_of, simd_width_of
from max.gpu.host.info import B200, H100, _is_sm10x_gpu
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from std.collections.string.string_span import get_static_string
from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import (
    block_id_in_cluster,
    thread_idx,
    block_idx,
    lane_id,
    warp_id as get_warp_id,
)
from max.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
    mbarrier_arrive,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    TileTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
)
from layout.layout import zipped_divide
from layout.layout_tensor import upcast
from layout.swizzle import make_swizzle
from layout.runtime_tuple import idx2crd
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
    create_tma_tile,
)
from std.logger import Logger
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from .arch.sm100 import MmaOpSM100_SS
from .matmul.gpu.sm100.config import MatmulConfig
from .matmul.gpu.sm100.matmul import WarpRole, consumer_main_loop, stsm_helper
from .matmul.gpu.sm100.pipeline import ProducerConsumerPipeline
from structured_kernels.tile_types import (
    SMemTileArray2D,
    swizzle_mode_to_bytes,
)
from .grouped_matmul_tile_scheduler import TileScheduler
from .utils import elementwise_epilogue_type

comptime logger = Logger()


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__name(
    t"matmul_sm100_grouped_blockwise_scaled_fp8_1d2d_{a_type}_{b_type}_{c_type}",
)
def matmul_sm100_grouped_blockwise_scaled_fp8_1d2d_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    accum_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_offsets_layout: Layout,
    expert_ids_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    c_static_N: Int,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: Int = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    a_offsets: LayoutTensor[.uint32, a_offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[.int32, expert_ids_layout, MutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, MutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, MutAnyOrigin],
    num_iters: Int32,
):
    var _num_iters = Int(num_iters)
    comptime assert transpose_b, "Only support transposed B"
    comptime assert num_threads == 128
    comptime assert (
        accum_type == .float32
    ), "Only support float32 for accumulator"

    var expert_idx = block_idx.z
    var M = rebind[UInt32](a_offsets[expert_idx + 1]) - rebind[UInt32](
        a_offsets[expert_idx]
    )
    comptime N = c_static_N
    comptime K = a_layout.shape[1].value()

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    comptime assert N % BN == 0, "N must be divisible by BN"
    comptime assert (
        BN <= BK or gcd(BN, BK) == BN - BK
    ), "BN <= BK or gcd(BN, BK) == BN - BK"

    var a_start_row_vec = a_offsets[expert_idx]
    comptime assert a_start_row_vec.length == 1
    var a_start_row = a_start_row_vec[0]

    var expert_vec = expert_ids[expert_idx]
    comptime assert expert_vec.length == 1

    var expert = expert_vec[0]

    var b_start_row = expert * Int32(N)

    var m_start = block_idx.y * BM
    var n_start = block_idx.x * BN
    var a_m_start = Int(a_start_row) + m_start
    var b_n_start = Int(b_start_row) + n_start
    if m_start >= Int(M) or n_start >= N:
        # print("m_start: ", m_start, "n_start: ", n_start, "M: ", M, "N: ", N)
        return

    # make sure A and B scales are compatible
    comptime b_scales_expert = b_scales_layout.shape[0].value()
    comptime b_scales_n = b_scales_layout.shape[1].value()
    comptime b_scales_k = b_scales_layout.shape[2].value()
    comptime a_scales_k = a_scales_layout.shape[0].value()

    var b_scales_2d = LayoutTensor[
        b_scales_type,
        Layout.row_major(b_scales_expert * b_scales_n, b_scales_k),
        b_scales.origin,
        address_space=b_scales.address_space,
    ](b_scales.ptr)

    comptime assert (
        N % b_scales_n == 0 and K % b_scales_k == 0 and K % a_scales_k == 0
    ), "N and K must be divisible by b_scales.shape[1] and b_scales.shape[2]"

    comptime B_SCALING_BLOCK_N = N // b_scales_n
    comptime B_SCALING_BLOCK_K = K // b_scales_k
    comptime A_SCALING_BLOCK = K // a_scales_k
    comptime assert (
        BK == B_SCALING_BLOCK_K == B_SCALING_BLOCK_N == A_SCALING_BLOCK
    ), (
        "Only support SCALING SIZE of 128! got:"
        + String(BK)
        + " "
        + String(B_SCALING_BLOCK_K)
        + " "
        + String(B_SCALING_BLOCK_N)
        + " "
        + String(A_SCALING_BLOCK)
    )

    # Use typed layouts as source of truth; bridge to legacy Layout for
    # LayoutTensor and MMA descriptor pipeline.
    comptime a_smem_layout = tile_layout_k_major_typed[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ].to_layout()
    comptime b_smem_layout = tile_layout_k_major_typed[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ].to_layout() if transpose_b else tile_layout_mn_major_typed[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ].to_layout()

    comptime a_scales_smem_layout = Layout.row_major(1, BM)

    var a_smem = rebind[
        UnsafePointer[Scalar[a_type], MutAnyOrigin, address_space=.SHARED]
    ](
        external_memory[
            Scalar[a_type],
            address_space=.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )

    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime a_scales_smem_tile_t = LayoutTensor[
        a_scales_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]

    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()
    comptime a_scales_size = a_scales_smem_layout.size()

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (a_scales_size * size_of[a_scales_type]()) % 16
    ) == 0, "preserve alignment"

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var warp_id = get_warp_id()
    var elect_one_warp = ufloordiv(thread_idx.x, WARP_SIZE) == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=1,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    # final results accumulator regs for C
    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag = Array[Scalar[accum_type], c_frag_size](
        fill=Scalar[accum_type](0)
    )

    # temporary accumulators for TMEM loads
    comptime total_repeat = BN // 8
    comptime repeat = 1  # a higher repeat will probably get us better performance, but it will increase register pressure
    comptime temp_cfrags_size = 4 * repeat

    comptime assert (
        total_repeat % repeat == 0
    ), "total_repeat must be divisible by repeat"
    var c_frag_temp: Array[Scalar[accum_type], temp_cfrags_size]

    for k_iter in range(_num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            var k_start = k_iter * BK
            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (k_start, a_m_start),
            )

            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (k_start, b_n_start) if transpose_b else (
                    b_n_start,
                    k_start,
                ),
            )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            mma_op.mma(
                lt_to_tt(a_smem_tile),
                lt_to_tt(b_smem_tile),
                tmem_addr,
                init_c=(True),  # Initialize C on first iteration
            )

            mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

        comptime for ld_iter in range(total_repeat // repeat):
            c_frag_temp = tcgen05_ld[
                datapaths=16,
                bits=256,
                repeat=repeat,
                dtype=accum_type,
                pack=False,
                width=temp_cfrags_size,
            ](tmem_addr + UInt32(ld_iter * 8 * repeat))
            tcgen05_load_wait()  # wait for the load to finish

            var b_scale: Scalar[accum_type]
            var b_scale_m_offset = Int(expert * Int32(b_scales_n))

            comptime if BN != BK:
                var global_n = block_idx.x * BN

                var begin_n = min(BN, BK - umod(global_n, BK))
                comptime end_n = BN  # if N % BN !=0 then it should be  min(BN, N - block_idx.x * BN)

                var idx0 = ufloordiv(global_n, BK)
                var next_n = begin_n if begin_n < end_n else BN

                if ld_iter < (next_n // 8):
                    b_scale = rebind[Scalar[b_scales_type]](
                        b_scales_2d[b_scale_m_offset + idx0, k_iter]
                    ).cast[accum_type]()
                else:
                    b_scale = rebind[Scalar[b_scales_type]](
                        b_scales_2d[b_scale_m_offset + idx0 + 1, k_iter]
                    ).cast[accum_type]()

            else:
                b_scale = rebind[Scalar[b_scales_type]](
                    b_scales_2d[b_scale_m_offset + block_idx.x, k_iter]
                ).cast[accum_type]()

            var m_offset = (warp_id * 16) + ufloordiv(lane_id(), 4)

            # TODO: this is an ugly way to calculate the m offset, need to rethink how we can make this more efficient
            comptime for j in range(temp_cfrags_size // 2):
                var local_m = m_offset + (j % 2) * 8
                var a_scale = a_scales[k_iter, a_m_start + local_m].cast[
                    accum_type
                ]()

                var scale = rebind[Scalar[accum_type]](a_scale * b_scale)
                var scale_pair = SIMD[accum_type, 2](scale)

                comptime idx = ld_iter * temp_cfrags_size + 2 * j
                var c_pair = SIMD[accum_type, 2](c_frag[idx], c_frag[idx + 1])
                var t_pair = SIMD[accum_type, 2](
                    c_frag_temp[2 * j], c_frag_temp[2 * j + 1]
                )
                var result = c_pair + t_pair * scale_pair
                c_frag[idx] = result[0]
                c_frag[idx + 1] = result[1]

        barrier()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // WARP_SIZE
    warp_id = ufloordiv(thread_idx.x, WARP_SIZE)

    comptime c_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
    comptime c_gmem_type = LayoutTensor[
        c_type,
        c_gmem_layout,
        MutAnyOrigin,
        layout_int_type=.int32,
        address_space=.GENERIC,
    ]

    # FIXME: A list literal initializer should be enough here, but somehow Mojo fails to infer that.
    var c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout](
        Index(M, N), Index(N, 1)
    )

    var c_by_expert = c_gmem_type(
        c_ptr + a_start_row * UInt32(N), c_gmem_runtime_layout
    )

    var ctile, ctile_coords, _ = c_by_expert.tile_with_offset[BM, BN](
        block_idx.y, block_idx.x
    )
    comptime c_coord_type = type_of(ctile_coords)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            var c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // num_warps, MMA_N](
                    4 * m_mma + warp_id, n_mma
                )
            )
            var c_gmem_warp_tile_coords = ctile_coords + rebind[c_coord_type](
                _c_gmem_warp_tile_coords
            )

            var c_gmem_frag, _c_gmem_frag_coords, _ = (
                c_gmem_warp_tile.vectorize[1, 2]().distribute_with_offset[
                    Layout.row_major(8, 4)
                ](lane_id())
            )
            var new_c_gmem_frag_coords = rebind[c_coord_type](
                _c_gmem_frag_coords
            )
            new_c_gmem_frag_coords[1] *= 2
            var c_gmem_frag_coords = (
                c_gmem_warp_tile_coords + new_c_gmem_frag_coords
            )

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec
                    comptime dst_idx = type_of(c_gmem_frag).layout(
                        IntTuple(m_vec, n_vec)
                    )
                    comptime dst_m_offset, dst_n_offset = divmod(dst_idx, N)
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < M and n < UInt32(N):
                        var c_mn = SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()

                        comptime if elementwise_lambda_fn:
                            comptime alignment = align_of[SIMD[c_type, 2]]()
                            comptime epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=alignment](
                                (Int(a_start_row + m), Int(n)), c_mn
                            )
                        else:
                            c_gmem_frag[m_vec, n_vec] = rebind[
                                c_gmem_frag.element_type
                            ](c_mn)


def grouped_matmul_sm100_blockwise_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    transpose_b: Bool,
    //,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    a_offsets: TileTensor[
        mut=False, a_offsets_type, address_space=.GENERIC, ...
    ],
    expert_ids: TileTensor[
        mut=False, expert_ids_type, address_space=.GENERIC, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launches the basic (non-persistent) SM100 blockwise-scaled FP8 grouped GEMM kernel.

    Converts the input `TileTensor`s to `LayoutTensor`s, builds TMA
    descriptors for A, B, and the C output, and enqueues
    `matmul_sm100_grouped_blockwise_scaled_fp8_1d2d_kernel` with a grid of
    `(N/BN, max_tokens/BM, num_active_experts)` blocks.

    Parameters:
        c_type: Element type of the output `C` tensor (inferred).
        a_type: Element type of the input `A` tensor (inferred). Must be
            `float8_e4m3fn`.
        b_type: Element type of the input `B` tensor (inferred). Must be
            `float8_e4m3fn`.
        a_scales_type: Element type of the `a_scales` tensor (inferred).
        b_scales_type: Element type of the `b_scales` tensor (inferred).
        a_offsets_type: Element type of the `a_offsets` tensor (inferred).
        expert_ids_type: Element type of the `expert_ids` tensor (inferred).
        transpose_b: Whether `B` is stored transposed (inferred). Must be
            `True`.
        config: Matmul configuration specifying block tile shape, MMA
            shape, and TMA swizzle modes.
        elementwise_lambda_fn: Optional epilogue function applied to each
            output element before storing (defaults to `None`).

    Args:
        c: Output tensor of shape `[total_tokens, N]` holding the
            grouped matmul results.
        a: Input activation tensor of shape `[total_tokens, K]` in FP8.
        b: Input weight tensor of shape `[num_experts, N, K]` in FP8.
        a_scales: Per-block scales for `A` of shape `[K // BK, total_tokens]`
            where `BK` is the scaling block size.
        b_scales: Per-block scales for `B` of shape `[num_experts, N // BN,
            K // BK]` where `BN` and `BK` are the scaling block sizes.
        a_offsets: Cumulative row offsets per expert, length
            `num_active_experts + 1`. Entry `i + 1` minus entry `i` gives
            the row count for expert `i`.
        expert_ids: Expert index for each active expert slot, mapping the
            grid Z index to the corresponding row offset in `B`.
        max_num_tokens_per_expert: Maximum number of tokens assigned to any
            single expert, used to size the grid M dimension.
        num_active_experts: Number of active experts in this grouped
            matmul, used to size the grid Z dimension.
        ctx: Device context used to enqueue the kernel.
    """
    comptime assert config.transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn for A and B"

    comptime accum_type = get_accum_type[a_type]()

    # Extract compile-time shapes from TileTensors.
    comptime num_experts = b.static_shape[0]
    comptime N = c.static_shape[1]
    comptime K = a.static_shape[1]

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]

    comptime assert BK == 128, "blockwise scaled fp8 only works with BK = 128"

    var a_scales_1 = Int(a_scales.dim[1]())
    assert a_scales_1 == Int(c.dim[0]()), "a_scales.dim(1) must be equal to M"

    var a_scales_0 = Int(a_scales.dim[0]())
    assert K % a_scales_0 == 0 and (K // a_scales_0) == BK, (
        "K must be divisible by a_scales.dim(0) and BK must be equal to K"
        " // a_scales.dim(0)"
    )

    var b_scales_0 = Int(b_scales.dim[1]())
    var b_scales_1 = Int(b_scales.dim[2]())
    assert (N % b_scales_0 == 0 and (N // b_scales_0) == BK) and (
        K % b_scales_1 == 0 and (K // b_scales_1) == BK
    ), (
        "N must be divisible by b_scales.dim(0) and BK must be equal to N"
        " // b_scales.dim(0) and K must be divisible by b_scales.dim(1) and"
        " BK must be equal to K // b_scales.dim(1)"
    )

    logger.info(
        "Executing SM100 Basic Grouped 1D2D Blockwise Scaled FP8 GEMM"
        " (BLOCK_SCALE_SIZE = 128)"
    )
    logger.info("Max tokens per expert: ", max_num_tokens_per_expert)
    logger.info("Number of active experts: ", num_active_experts)
    logger.info(
        "A Scales Shape: [",
        a_scales.dim[0](),
        ", ",
        a_scales.dim[1](),
        "]",
        sep="",
    )
    logger.info(
        "B Scales Shape: [",
        b_scales.dim[0](),
        ", ",
        b_scales.dim[1](),
        ", ",
        b_scales.dim[2](),
        "]",
        sep="",
    )

    # Convert TileTensors to LayoutTensors at the kernel boundary.
    var a_tensor = a.to_layout_tensor()
    var b_tensor = b.to_layout_tensor()
    var c_tensor = c.to_layout_tensor()
    var a_scales_tensor = a_scales.to_layout_tensor()
    var b_scales_tensor = b_scales.to_layout_tensor()
    var a_offsets_tensor = a_offsets.to_layout_tensor()
    var expert_ids_tensor = expert_ids.to_layout_tensor()

    var a_tma_op = create_tensor_tile[
        Index(BM, BK), swizzle_mode=config.a_swizzle
    ](ctx, a_tensor)

    # Reshape 3D weights to 2D for TMA.
    var b_2d = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        address_space=.GENERIC,
    ](b.ptr.as_unsafe_any_origin())
    var b_tma_op = create_tensor_tile[
        Index(BN, BK) if config.transpose_b else Index(BK, BN),
        swizzle_mode=config.b_swizzle,
    ](ctx, b_2d)

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24 + size_of[a_scales_type]() * BM

    comptime block_dim = 128

    comptime kernel = matmul_sm100_grouped_blockwise_scaled_fp8_1d2d_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        accum_type,
        type_of(a_tensor).layout,
        type_of(b_tensor).layout,
        type_of(a_offsets_tensor).layout,
        type_of(expert_ids_tensor).layout,
        type_of(a_scales_tensor).layout,
        type_of(b_scales_tensor).layout,
        N,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        config.block_tile_shape,
        config.mma_shape,
        transpose_b=config.transpose_b,
        a_swizzle=config.a_swizzle,
        b_swizzle=config.b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        a_offsets_tensor,
        expert_ids_tensor,
        c_tensor.ptr,
        a_scales_tensor,
        b_scales_tensor,
        Int32(ceildiv(K, BK)),
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


@always_inline
def _get_accumulator_size[
    *,
    c_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
]() -> IndexList[2]:
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime stageN = c_smem_layout.shape[1].value()
    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages
    comptime data_paths = 16
    comptime bits = 256
    comptime repeats = stageN // (bits // 32)

    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime num_elements = repeats * fragment_size

    return Index(num_stages, num_elements)


@always_inline
def _tile_fits_full_tma[
    a_scales_type: DType, BM: Int
](m_tile_global_start: Int, total_m: Int) -> Bool:
    """Whether a full BM-row A/a_scales TMA is safe for this tile.

    The bound is the A buffer, not the current expert. A tile may read BM rows
    past this expert's end into the next expert's rows: that read is in-bounds
    as long as the whole BM strip lies within total_m, and the epilogue masks
    the over-read rows with m < M (per-expert count) so they are never written.
    The next expert's own tile recomputes them correctly. That makes the
    expert boundary irrelevant to safety and lets small-M-per-expert decode
    tiles keep the fast TMA path instead of falling onto the partial copy.

    Three conditions must hold:
      * the full BM-row strip stays within total_m, so the TMA load and the
        matching a_scales strip never read past the A buffer,
      * m_tile_global_start is aligned (column offset within a K row), and
      * total_m is aligned, since it is the K-row stride: at any K>0 the scales
        byte offset is k * total_m * size_of(scales), which is only 16-byte
        aligned when the stride itself is. Misalignment faults with
        ILLEGAL_INSTRUCTION.
    """
    comptime SCALES_M_ALIGN = 16 // size_of[a_scales_type]()
    return (
        (total_m - m_tile_global_start) >= BM
        and m_tile_global_start % SCALES_M_ALIGN == 0
        and total_m % SCALES_M_ALIGN == 0
    )


@always_inline
def _copy_partial_a_tile_blockwise_from_gmem[
    a_type: DType,
    a_scales_type: DType,
    a_gmem_layout: Layout,
    a_scales_gmem_layout: Layout,
    *,
    a_smem_layout: Layout,
    a_scales_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle,
](
    a_gmem: LayoutTensor[a_type, a_gmem_layout, ImmutAnyOrigin],
    a_scales_gmem: LayoutTensor[
        a_scales_type, a_scales_gmem_layout, ImmutAnyOrigin
    ],
    a_smem_tile: LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
        ...,
    ],
    a_scales_smem_tile: LayoutTensor[
        a_scales_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
        ...,
    ],
    expert_end_row: Int,
    m_tile_global_start: Int,
    iter_idx: Int,
):
    """Cooperative warp copy of A and matching `a_scales` strip from gmem into
    SMEM. Used when the full-width TMA load is ineligible: the final
    buffer-tail tile whose BM-row strip would run past total_m, or a misaligned
    scale stride that the scales TMA cannot address. Rows at or past the current
    expert's end are zeroed so MMA still sees a full BM×BK tile without reading
    past the activation buffer; the epilogue masks those rows, so the fill only
    needs to be safe, not exact. Writes go to the physical SMEM offset produced
    by `make_swizzle` so the MMA descriptor's swizzle XOR reads back the value
    we just stored. All lanes of the calling warp must execute this."""
    comptime BM = block_tile_shape[0]
    comptime BK = block_tile_shape[2]
    comptime a_sw = make_swizzle[a_type, a_swizzle]()
    comptime VEC = 16 // size_of[a_type]()
    comptime CHUNKS_PER_ROW = BK // VEC
    comptime TOTAL_CHUNKS = BM * CHUNKS_PER_ROW
    comptime assert (
        BK % VEC == 0 and TOTAL_CHUNKS % WARP_SIZE == 0
    ), "partial-tile copy expects BK*BM aligned to 16-byte chunks per warp"
    comptime assert (
        BM % WARP_SIZE == 0
    ), "partial-tile scales copy assumes BM is a multiple of WARP_SIZE"
    comptime CHUNKS_PER_LANE = TOTAL_CHUNKS // WARP_SIZE
    var lane = lane_id()
    var zero_vec = SIMD[a_type, VEC](0)

    comptime for i in range(CHUNKS_PER_LANE):
        var chunk = i * WARP_SIZE + lane
        var row = chunk // CHUNKS_PER_ROW
        var k0 = (chunk % CHUNKS_PER_ROW) * VEC
        var g_row = m_tile_global_start + row
        var av = zero_vec
        if g_row < expert_end_row:
            av = a_gmem.load[width=VEC](g_row, iter_idx * BK + k0)
        (a_smem_tile.ptr + Int(a_sw(Int32(row * BK + k0)))).store(av)

    # a_scales: BM scalars total; each lane writes one row per outer iteration.
    comptime for i in range(BM // WARP_SIZE):
        var row = i * WARP_SIZE + lane
        var g_row = m_tile_global_start + row
        var sv = Scalar[a_scales_type](0)
        if g_row < expert_end_row:
            sv = rebind[Scalar[a_scales_type]](a_scales_gmem[iter_idx, g_row])
        a_scales_smem_tile[0, row] = sv


@always_inline
def load_AB[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    a_scales_tile_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_tile_rank],
    a_scales_desc_shape: IndexList[a_scales_tile_rank],
    num_pipeline_stages: Int,
    expert_ids_layout: Layout,
    /,
    *,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_scales_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    a_scales_tma_op: TMATensorTile[
        a_scales_type,
        a_scales_tile_rank,
        a_scales_tile_shape,
        a_scales_desc_shape,
    ],
    a_smem_base: UnsafePointer[
        mut=True, Scalar[a_type], _, address_space=.SHARED
    ],
    b_smem_base: UnsafePointer[
        mut=True, Scalar[b_type], _, address_space=.SHARED
    ],
    a_scales_smem_base: UnsafePointer[
        mut=True, Scalar[a_scales_type], _, address_space=.SHARED
    ],
    load_mma_pipeline: ProducerConsumerPipeline[num_pipeline_stages],
    peer_cta_coord: Tuple[Int, Int, Int],
    work_tile_coord: Tuple[Int, Int],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: Int,
    elect_one_cta: Bool,
    scheduler: TileScheduler,
    expert_ids: LayoutTensor[.int32, expert_ids_layout, ImmutAnyOrigin],
):
    """Issues multicast TMA loads for A, B, and A scales into the producer stage of the load-MMA pipeline.

    Waits for the consumer (MMA) to release the buffer, then launches
    async multicast TMA copies of A, B, and the matching A scales strip
    into the current pipeline stage's SMEM and signals completion through
    the stage's mbarrier.

    Parameters:
        a_type: Element dtype of the A activation operand (inferred).
        b_type: Element dtype of the B weight operand (inferred).
        a_scales_type: Element dtype of the A per-block scales (inferred).
        a_tile_rank: Rank of the A TMA tile descriptor (inferred).
        a_tile_shape: Shape of each A TMA tile copy (inferred).
        a_desc_shape: Descriptor shape of the A TMA tensor (inferred).
        b_tile_rank: Rank of the B TMA tile descriptor (inferred).
        b_tile_shape: Shape of each B TMA tile copy (inferred).
        b_desc_shape: Descriptor shape of the B TMA tensor (inferred).
        a_scales_tile_rank: Rank of the A scales TMA tile descriptor
            (inferred).
        a_scales_tile_shape: Shape of each A scales TMA tile copy
            (inferred).
        a_scales_desc_shape: Descriptor shape of the A scales TMA
            tensor (inferred).
        num_pipeline_stages: Number of load-MMA pipeline buffer
            stages (inferred).
        expert_ids_layout: Layout of the expert-IDs mapping tensor
            (inferred).
        a_smem_layout: SMEM layout for the A tile buffer.
        b_smem_layout: SMEM layout for the B tile buffer.
        a_scales_smem_layout: SMEM layout for the A scales strip.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
        cta_group: CTA group size, 1 or 2 (defaults to 1).

    Args:
        a_tma_op: TMA descriptor for loading A activation tiles.
        b_tma_op: TMA descriptor for loading B weight tiles.
        a_scales_tma_op: TMA descriptor for loading A scales tiles.
        a_smem_base: Base pointer to the A SMEM buffer.
        b_smem_base: Base pointer to the B SMEM buffer.
        a_scales_smem_base: Base pointer to the A scales SMEM buffer.
        load_mma_pipeline: Producer-consumer pipeline between load and
            MMA warps.
        peer_cta_coord: Peer CTA coordinates as (peer_id, mma_coord_m,
            mma_coord_n).
        work_tile_coord: Work tile coordinates as (m, n).
        a_multicast_mask: Multicast mask selecting CTAs for A TMA loads.
        b_multicast_mask: Multicast mask selecting CTAs for B TMA loads.
        iter_idx: K-iteration index of the current tile.
        elect_one_cta: Whether this CTA is the elected leader for the
            cluster.
        scheduler: Tile scheduler tracking the current group and work
            tile.
        expert_ids: Mapping from group index to expert row offset in B.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    comptime a_scales_expected_bytes = a_scales_smem_layout.size() * size_of[
        a_scales_type
    ]()
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (
        a_expected_bytes + b_expected_bytes + a_scales_expected_bytes
    )

    comptime a_tma_load_size = _idx_product[a_tile_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime a_scales_tma_load_size = _idx_product[
        a_scales_tile_rank, a_scales_desc_shape
    ]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    var stage = load_mma_pipeline.producer_stage()

    # Wait until MMA (consumer) has used the buffer.
    load_mma_pipeline.wait_consumer()

    var a_gmem_slice_coord = peer_cta_coord[2] * a_tma_rows + work_tile_coord[0]
    var expert_id = expert_ids[Int(scheduler.current_group_idx)]
    var b_gmem_slice_coord_vec = type_of(expert_id)(
        peer_cta_coord[1] * b_tma_rows
        + peer_cta_coord[0] * BN
        + work_tile_coord[1]
    ) + expert_id * type_of(expert_id)(scheduler.static_MN)
    comptime assert b_gmem_slice_coord_vec.length == 1
    var b_gmem_slice_coord = Int(b_gmem_slice_coord_vec[0])

    comptime a_smem_tile_size = a_smem_layout.size()
    comptime b_smem_tile_size = b_smem_layout.size()
    comptime a_scales_smem_tile_size = a_scales_smem_layout.size()

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](a_smem_base + Int(stage) * a_smem_tile_size)
    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](b_smem_base + Int(stage) * b_smem_tile_size)
    var a_scales_smem_tile = LayoutTensor[
        a_scales_type,
        a_scales_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](a_scales_smem_base + Int(stage) * a_scales_smem_tile_size)

    var a_smem_slice = type_of(a_smem_tile)(
        a_smem_tile.ptr + peer_cta_coord[2] * a_tma_load_size
    )
    var b_smem_slice = type_of(b_smem_tile)(
        b_smem_tile.ptr + peer_cta_coord[1] * b_tma_load_size
    )
    var tma_mbar = load_mma_pipeline.producer_mbar(stage)

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

        a_tma_op.async_multicast_load[cta_group](
            a_smem_slice,
            tma_mbar[0],
            (iter_idx * BK, a_gmem_slice_coord),
            a_multicast_mask,
        )

        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[0],
            (iter_idx * BK, b_gmem_slice_coord),
            b_multicast_mask,
        )

        a_scales_tma_op.async_copy[cta_group](
            a_scales_smem_tile,
            tma_mbar[0],
            (work_tile_coord[0], iter_idx),
        )


@always_inline
def load_AB_partial[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    num_pipeline_stages: Int,
    expert_ids_layout: Layout,
    a_gmem_layout: Layout,
    a_scales_gmem_layout: Layout,
    /,
    *,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_scales_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    cta_group: Int = 1,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    a_gmem: LayoutTensor[a_type, a_gmem_layout, ImmutAnyOrigin],
    a_scales_gmem: LayoutTensor[
        a_scales_type, a_scales_gmem_layout, ImmutAnyOrigin
    ],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    a_smem_base: UnsafePointer[
        mut=True, Scalar[a_type], _, address_space=.SHARED
    ],
    b_smem_base: UnsafePointer[
        mut=True, Scalar[b_type], _, address_space=.SHARED
    ],
    a_scales_smem_base: UnsafePointer[
        mut=True, Scalar[a_scales_type], _, address_space=.SHARED
    ],
    load_mma_pipeline: ProducerConsumerPipeline[num_pipeline_stages],
    peer_cta_coord: Tuple[Int, Int, Int],
    work_tile_coord: Tuple[Int, Int],
    b_multicast_mask: UInt16,
    iter_idx: Int,
    elect_one_cta: Bool,
    scheduler: TileScheduler,
    expert_ids: LayoutTensor[.int32, expert_ids_layout, ImmutAnyOrigin],
    expert_end_row: Int,
    m_tile_global_start: Int,
):
    """Sibling to `load_AB` for tiles the full-TMA path can't handle: fills A
    and `a_scales` SMEM via a cooperative warp copy from gmem and issues TMA
    only for B.

    Parameters:
        a_type: Element dtype of the A activation operand (inferred).
        b_type: Element dtype of the B weight operand (inferred).
        a_scales_type: Element dtype of the A per-block scales (inferred).
        b_tile_rank: Rank of the B TMA tile descriptor (inferred).
        b_tile_shape: Shape of each B TMA tile copy (inferred).
        b_desc_shape: Descriptor shape of the B TMA tensor (inferred).
        num_pipeline_stages: Number of load-MMA pipeline buffer stages
            (inferred).
        expert_ids_layout: Layout of the expert-IDs mapping tensor
            (inferred).
        a_gmem_layout: Layout of the A activation tensor in global memory
            (inferred).
        a_scales_gmem_layout: Layout of the A per-block scales tensor in
            global memory (inferred).
        a_smem_layout: SMEM layout for the A tile buffer.
        b_smem_layout: SMEM layout for the B tile buffer.
        a_scales_smem_layout: SMEM layout for the A scales strip.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        cta_group: CTA group size, 1 or 2 (defaults to 1).
        a_swizzle: Swizzle mode for the A SMEM buffer, used to compute
            physical store offsets in the cooperative warp copy (defaults
            to `SWIZZLE_NONE`).

    Args:
        a_gmem: A activation tensor in global memory, read by the
            cooperative warp copy.
        a_scales_gmem: A per-block scales tensor in global memory, read by
            the cooperative warp copy.
        b_tma_op: TMA descriptor for loading B weight tiles.
        a_smem_base: Base pointer to the A SMEM buffer.
        b_smem_base: Base pointer to the B SMEM buffer.
        a_scales_smem_base: Base pointer to the A scales SMEM buffer.
        load_mma_pipeline: Producer-consumer pipeline between load and MMA
            warps.
        peer_cta_coord: Peer CTA coordinates as (peer_id, mma_coord_m,
            mma_coord_n).
        work_tile_coord: Work tile coordinates as (m, n).
        b_multicast_mask: Multicast mask selecting CTAs for B TMA loads.
        iter_idx: K-iteration index of the current tile.
        elect_one_cta: Whether this CTA is the elected leader for the
            cluster.
        scheduler: Tile scheduler tracking the current group and work
            tile.
        expert_ids: Mapping from group index to expert row offset in B.
        expert_end_row: Global row index one past the last row of the
            current expert; rows at or past it are zeroed during the
            cooperative copy.
        m_tile_global_start: Global M-row start of the current tile within
            the A buffer.
    """
    # `expect_bytes` below only accounts for B's TMA; A is populated by the
    # calling warp under a `syncwarp` + `fence_async_view_proxy`. A peer CTA
    # in a cluster would see A unsynchronised on the multicast path.
    comptime assert (
        cta_group == 1
    ), "load_AB_partial assumes a single-CTA consumer"
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime b_tma_rows = b_desc_shape[0]

    var stage = load_mma_pipeline.producer_stage()

    # Wait until MMA (consumer) has used the buffer.
    load_mma_pipeline.wait_consumer()

    var expert_id = expert_ids[Int(scheduler.current_group_idx)]
    var b_gmem_slice_coord_vec = type_of(expert_id)(
        peer_cta_coord[1] * b_tma_rows
        + peer_cta_coord[0] * BN
        + work_tile_coord[1]
    ) + expert_id * type_of(expert_id)(scheduler.static_MN)
    comptime assert b_gmem_slice_coord_vec.length == 1
    var b_gmem_slice_coord = Int(b_gmem_slice_coord_vec[0])

    comptime a_smem_tile_size = a_smem_layout.size()
    comptime b_smem_tile_size = b_smem_layout.size()
    comptime a_scales_smem_tile_size = a_scales_smem_layout.size()

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](a_smem_base + Int(stage) * a_smem_tile_size)
    var a_scales_smem_tile = LayoutTensor[
        a_scales_type,
        a_scales_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](a_scales_smem_base + Int(stage) * a_scales_smem_tile_size)
    var b_smem_slice = LayoutTensor[
        b_type,
        b_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](
        b_smem_base
        + Int(stage) * b_smem_tile_size
        + peer_cta_coord[1] * b_tma_load_size
    )
    var tma_mbar = load_mma_pipeline.producer_mbar(stage)

    _copy_partial_a_tile_blockwise_from_gmem[
        a_smem_layout=a_smem_layout,
        a_scales_smem_layout=a_scales_smem_layout,
        block_tile_shape=block_tile_shape,
        a_swizzle=a_swizzle,
    ](
        a_gmem,
        a_scales_gmem,
        a_smem_tile,
        a_scales_smem_tile,
        expert_end_row,
        m_tile_global_start,
        iter_idx,
    )
    syncwarp()
    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[0].expect_bytes(Int32(cta_group * b_expected_bytes))
        # Order the cooperative SMEM writes before the async-proxy TMA.
        fence_async_view_proxy()
        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[0],
            (iter_idx * BK, b_gmem_slice_coord),
            b_multicast_mask,
        )


@always_inline
def multi_stage_reg_epilogue[
    c_tile_rank: Int,
    c_tile_shape: IndexList[c_tile_rank],
    c_desc_shape: IndexList[c_tile_rank],
    accum_type: DType,
    accum_layout: Layout,
    /,
    *,
    c_smem_layout: Layout,
    c_type: DType,
    c_static_N: Int,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    is_lower_frag_required: Bool,
    cta_group: Int,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle,
](
    c_upper_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        ...,
    ],
    c_lower_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        ...,
    ],
    c_smem_base: UnsafePointer[
        Scalar[c_type], MutAnyOrigin, address_space=.SHARED
    ],
    c_tma_op: TMATensorTile[c_type, c_tile_rank, c_tile_shape, c_desc_shape],
    c_ptr: UnsafePointer[mut=True, Scalar[c_type], _],
    c_coord: Tuple[Int, Int],
    elect_one_warp: Bool,
    group_end_idx: UInt32,
):
    """Writes the accumulated C fragments from registers to global memory through a staged shared-memory buffer.

    Casts the upper (and optionally lower) accumulator fragments to the
    output dtype, packs them into SMEM via the STSM helper, and then
    either issues TMA async stores or performs a cooperative warp store
    to global memory, depending on the output dtype and tile bounds.

    Parameters:
        c_tile_rank: Rank of the C TMA tile descriptor (inferred).
        c_tile_shape: Shape of each C TMA tile copy (inferred).
        c_desc_shape: Descriptor shape of the C TMA tensor (inferred).
        accum_type: Accumulator dtype of the register tiles (inferred).
        accum_layout: Layout of the upper and lower accumulator register
            tiles (inferred); shape is `(num_stages, num_elements)`.
        c_smem_layout: SMEM layout for the C staging buffer.
        c_type: Element dtype of the output `C` tensor.
        c_static_N: Compile-time N dimension of the output matrix, used
            as the row stride for global stores.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
        is_lower_frag_required: Whether the lower accumulator fragment
            must be packed and stored to SMEM.
        cta_group: CTA group size, 1 or 2.
        num_output_warps: Number of output warps participating in the
            epilogue store.
        c_swizzle: TMA swizzle mode for the C SMEM buffer.

    Args:
        c_upper_main_tile: Upper accumulator register tile holding the
            staged MMA results to be stored.
        c_lower_main_tile: Lower accumulator register tile holding the
            staged MMA results to be stored.
        c_smem_base: Base pointer to the C SMEM staging buffer.
        c_tma_op: TMA descriptor for storing C output tiles.
        c_ptr: Pointer to the output `C` buffer in global memory.
        c_coord: Output tile coordinates as (m, n) in the `C` matrix.
        elect_one_warp: Whether this warp is the elected leader for TMA
            stores.
        group_end_idx: One-past-the-last row index of the current group;
            rows at or past it are masked out during the store.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime num_stages = accum_layout.shape[0].value()
    comptime num_elements = accum_layout.shape[1].value()

    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime repeats = num_elements // fragment_size
    comptime stageN = repeats * (bits // 32)
    comptime fragments_per_stage = fragment_size * repeats

    comptime swizzle = make_swizzle[c_type, c_swizzle]()

    var warp_id = get_warp_id()

    comptime for stage in range(num_stages):
        var upper_frag = c_upper_main_tile.load[fragments_per_stage](stage, 0)
        var lower_frag = c_lower_main_tile.load[fragments_per_stage](stage, 0)

        # Assume double-buffer for shared memory packing
        comptime c_smem_tile_size = c_smem_layout.size()
        var c_smem_tile = LayoutTensor[
            c_type,
            c_smem_layout,
            MutAnyOrigin,
            address_space=.SHARED,
            alignment=128,
        ](c_smem_base + (stage % 2) * c_smem_tile_size)
        comptime c_smem_tile_m = 32 if cta_group == 2 else BM // num_output_warps
        var c_smem_warp_tt = lt_to_tt(c_smem_tile).tile[c_smem_tile_m, stageN](
            warp_id, 0
        )

        var c_smem_warp_tile_upper = c_smem_warp_tt.tile[data_paths, stageN](
            0, 0
        )
        var upper_st = Array[Scalar[c_type], fragments_per_stage](
            uninitialized=True
        )

        comptime cast_width = 4 // size_of[Scalar[c_type]]()
        comptime for _i in range(fragments_per_stage // cast_width):
            comptime offset = _i * cast_width
            var src = SIMD[accum_type, cast_width]()
            comptime for _j in range(cast_width):
                src[_j] = upper_frag[offset + _j]
            var casted = src.cast[c_type]()
            comptime for _j in range(cast_width):
                upper_st[offset + _j] = casted[_j]
        stsm_helper[swizzle, stageN, swizzle_mode=c_swizzle](
            upper_st, c_smem_warp_tile_upper
        )

        var c_smem_warp_tile_lower = c_smem_warp_tt.tile[data_paths, stageN](
            1, 0
        )

        comptime if is_lower_frag_required:
            var lower_st = Array[Scalar[c_type], fragments_per_stage](
                uninitialized=True
            )

            comptime for _i in range(fragments_per_stage // cast_width):
                comptime offset = _i * cast_width
                var src = SIMD[accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = lower_frag[offset + _j]
                var casted = src.cast[c_type]()
                comptime for _j in range(cast_width):
                    lower_st[offset + _j] = casted[_j]
            stsm_helper[swizzle, stageN, swizzle_mode=c_swizzle](
                lower_st, c_smem_warp_tile_lower
            )

        # Guard the write to shared memory is done.
        named_barrier[Int32(num_output_warps * WARP_SIZE)]()

        var lane = lane_id()

        comptime CG2_TMA_BM = c_smem_tile.layout.shape[
            0
        ].value() if MMA_M == 256 else BM
        comptime CG1_TMA_BM = c_smem_tile.layout.shape[0].value()
        comptime TMA_BM = CG2_TMA_BM if cta_group == 2 else CG1_TMA_BM

        var cg2_elect_one_warp = (
            warp_id == 0 if MMA_M == 256 else ufloordiv(warp_id, 2) == 0
        )
        var cg1_elect_one_warp = warp_id == 0
        var elect_one_warp = (
            cg2_elect_one_warp if cta_group == 2 else cg1_elect_one_warp
        )

        var coord_n_mma_m256 = c_coord[1] + (stage * stageN)
        var coord_n_mma_m128 = (
            c_coord[1] + (stage * stageN) + (BN * ufloordiv(warp_id, 2))
        )

        var cg2_coord_n = coord_n_mma_m256 if MMA_M == 256 else coord_n_mma_m128
        var cg1_coord_n = coord_n_mma_m256
        var coord_n = cg2_coord_n if cta_group == 2 else cg1_coord_n
        var coord_m = c_coord[0]
        var cg2_c_smem_coord_m = 0 if MMA_M == 256 else ufloordiv(warp_id, 2)
        var cg1_c_smem_coord_m = 0
        var c_smem_coord_m = (
            cg2_c_smem_coord_m if cta_group == 2 else cg1_c_smem_coord_m
        )

        if (
            size_of[c_type]() != 2
            or UInt32(coord_m) + UInt32(TMA_BM) > group_end_idx
        ):
            comptime output_threads = num_output_warps * WARP_SIZE
            comptime c_smem_M = c_smem_tile.layout.shape[0].value()
            comptime RLayout32Bits[layout: Layout] = RuntimeLayout[
                layout,
                element_type=.uint32,
                linear_idx_type=.uint32,
            ]
            comptime simd_size = simd_width_of[c_type]()
            comptime alignment = align_of[SIMD[c_type, simd_size]]()
            comptime thread_n = stageN // simd_size
            comptime assert (
                stageN % simd_size == 0
            ), "stageN must be divisible by simd_size"
            comptime assert (
                output_threads % thread_n == 0
            ), "output_threads must be divisible by thread_n"
            comptime thread_layout = Layout.row_major(
                output_threads // thread_n, thread_n
            )

            comptime for i in range(c_smem_M // TMA_BM):
                var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](i, 0)
                comptime split_layout = c_smem_split.layout
                var split_rt = RLayout32Bits[split_layout]()
                comptime zipped = zipped_divide(
                    upcast(split_layout, simd_size), thread_layout
                )
                var zipped_rt = RLayout32Bits[zipped]()

                # zipped.shape[1][1] == 1 by construction
                comptime for j in range(zipped.shape[1][0].value()):
                    var input_crd = RuntimeTuple[
                        IntTuple(UNKNOWN_VALUE, j), element_type=.uint32
                    ](thread_idx.x, j)
                    var linear_idx = zipped_rt(input_crd) * UInt32(simd_size)
                    var linear_tup = RuntimeTuple[
                        IntTuple(UNKNOWN_VALUE), element_type=.uint32
                    ](Int(linear_idx))
                    var cmem_crd = idx2crd(
                        linear_tup, split_rt.shape, split_rt.stride
                    )
                    var local_i = Int(cmem_crd[0].get_int())
                    var local_j = Int(cmem_crd[1].get_int())
                    var global_i: Int = coord_m + local_i
                    var global_j: Int = coord_n + local_j
                    if global_i < Int(group_end_idx):
                        # src_ptr = c_smem_split.ptr + swizzle(linear_idx)
                        var src_ptr = c_smem_split.ptr + (
                            linear_idx if size_of[c_type]()
                            != 2 else swizzle(linear_idx)
                        )
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        var dst_ptr = c_ptr + global_i * c_static_N + global_j
                        dst_ptr.store[width=simd_size, alignment=alignment](src)
        else:
            var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](
                c_smem_coord_m, 0
            )

            if elect_one_warp and lane == 0:
                fence_async_view_proxy()
                c_tma_op.async_store(
                    c_smem_split,
                    (coord_n, coord_m),
                )
                c_tma_op.commit_group()

            # Keep one tma store in fly
            comptime if stage < num_stages - 1:
                c_tma_op.wait_group[1]()
            # Last stage guard all tma store to finish
            else:
                c_tma_op.wait_group[0]()

        comptime if stage > 0 and stage < num_stages - 1:
            # Guard the tma read from shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()


@always_inline
def promote_accumulators[
    pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    accum_type: DType,
    accum_layout: Layout,
    a_scales_type: DType,
    b_scales_type: DType,
    b_scales_layout: Layout,
    expert_ids_layout: Layout,
    /,
    *,
    a_scales_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    CLUSTER_SIZE: Int32,
    is_lower_frag_required: Bool,
    num_output_warps: Int,
](
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, ImmutAnyOrigin],
    b_scales_n: Int,
    a_scales_smem_base: UnsafePointer[
        mut=True, Scalar[a_scales_type], _, address_space=.SHARED
    ],
    c_upper_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        ...,
    ],
    c_lower_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        ...,
    ],
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_pipeline_stages],
    tmem_addr: UInt32,
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    work_tile_coord: Tuple[Int, Int],
    elect_one_warp: Bool,
    stage_stride_cols: Int,
    k_iter: Int,
    problem_shape: StaticTuple[Int32, 3],
    expert_ids: LayoutTensor[.int32, expert_ids_layout, ImmutAnyOrigin],
    scheduler: TileScheduler,
):
    """Loads MMA outputs from TMEM and accumulates them into register C fragments with blockwise FP8 scaling.

    Reads the per-stage TMEM accumulator fragments, fetches the matching
    A scales from SMEM and B scales from global memory, multiplies them
    into a per-fragment scale, and accumulates the scaled products into
    the upper and lower register tiles for the epilogue.

    Parameters:
        pipeline_stages: Number of load-MMA pipeline buffer stages
            (inferred).
        num_accum_pipeline_stages: Number of MMA-output pipeline buffer
            stages (inferred).
        accum_type: Accumulator dtype for the MMA result (inferred);
            must be `float32`.
        accum_layout: Layout of the upper and lower accumulator register
            tiles (inferred); shape is `(num_stages, num_elements)`.
        a_scales_type: Element dtype of the A per-block scales
            (inferred).
        b_scales_type: Element dtype of the B per-block scales
            (inferred).
        b_scales_layout: Layout of the B per-block scales tensor
            (inferred).
        expert_ids_layout: Layout of the expert-IDs mapping tensor
            (inferred).
        a_scales_smem_layout: SMEM layout for the A scales strip.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
        cta_group: CTA group size, 1 or 2.
        CLUSTER_SIZE: Number of CTAs in the cluster, used to gate the
            load-mbar arrive.
        is_lower_frag_required: Whether the lower accumulator fragment
            must be loaded and accumulated.
        num_output_warps: Number of epilogue output warps participating
            in the promotion.

    Args:
        b_scales: Per-block scales for `B` of shape `[num_experts,
            N // BN, K // BK]`, read to fetch the B scale for the
            current K block.
        b_scales_n: N dimension of the B scales tensor (columns per
            expert), used to compute the expert row offset into
            `b_scales`.
        a_scales_smem_base: Base pointer to the A scales SMEM buffer.
        c_upper_main_tile: Upper accumulator register tile of shape
            `(num_stages, num_elements)` that the scaled MMA fragments
            are accumulated into.
        c_lower_main_tile: Lower accumulator register tile of shape
            `(num_stages, num_elements)` that the scaled MMA fragments
            are accumulated into when `is_lower_frag_required`.
        mma_output_pipeline: Producer-consumer pipeline carrying TMEM
            accumulator fragments from the MMA warp.
        tmem_addr: Base TMEM address of the accumulator allocation.
        load_mma_pipeline: Producer-consumer pipeline between the load
            and MMA warps, used to track the A scales SMEM stage.
        work_tile_coord: Work tile coordinates as (m, n); `n` selects
            the B scale index.
        elect_one_warp: Whether this warp is the elected leader warp.
        stage_stride_cols: TMEM column stride between consecutive
            MMA-output pipeline stages.
        k_iter: K-iteration index of the current tile, used to index
            the A and B scales.
        problem_shape: Problem shape as (M, N, K) of the grouped GEMM.
        expert_ids: Mapping from group index to expert row offset in
            `B`.
        scheduler: Tile scheduler tracking the current group and work
            tile.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime assert (
        a_scales_type == b_scales_type and accum_type == .float32
    ), "a_scales_type must equal b_scales_type, and accum_type must be float32"
    # Rows each warp is responsible for:
    # warp_id 0 -> 0-15 upper, 16-31 lower
    # warp_id 1 -> 32-47 upper, 48-63 lower
    # warp_id 2 -> 64-79 upper, 80-95 lower
    # warp_id 3 -> 96-111 upper, 112-127 lower

    var M = problem_shape[0]
    var N = problem_shape[1]
    var K = problem_shape[2]

    comptime num_stages = accum_layout.shape[0].value()
    comptime num_elements = accum_layout.shape[1].value()
    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime assert fragment_size == 4, "fragment_size must be 4"
    comptime repeats = num_elements // fragment_size
    comptime stageN = repeats * (bits // 32)
    comptime load_width = 2

    var bn = work_tile_coord[1]

    var tma_load_stage_index = load_mma_pipeline.consumer_stage()

    # scale_b index calculation when MMA_N != BK(128)
    var b_scale_idx0 = 0
    var b_scale_next_n = 0
    var b_scale_0: Scalar[accum_type]
    var b_scale_1: Scalar[accum_type]
    var expert_id = expert_ids[Int(scheduler.current_group_idx)]
    var b_scale_m_offset = expert_id * type_of(expert_id)(b_scales_n)

    comptime if MMA_N != BK:
        comptime assert stageN <= gcd(MMA_N, BK) and (
            gcd(MMA_N, BK) % stageN == 0
        ), (
            "gcd(MMA_N, BK) must be divisible by stageN. If not then this"
            " step should be updated to support non-divisible case"
            " accordingly"
        )

        var global_bn_start = bn
        var begin_n = min(BK - umod(global_bn_start, BK), MMA_N)
        var end_n = min(N - Int32(global_bn_start), Int32(MMA_N))

        # find the first b_scale index just by dividing by block size (128)
        # we use `b_scale_next_n` to find the second b_scale index later
        b_scale_idx0 = ufloordiv(global_bn_start, BK)
        # If MMA_N > BK (128) then we should use two scales_b in each block. `next_n` determines the border between the two scales_b.
        # Example: N = 960, MMA_N = 192, num_of_b_scales: ceildiv(960, BK) = 8
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <-------------------------128------------------------------>|<----------64------------>
        # <-------------------------block_scales[idx0]--------------->|<--block_scales[idx0+1]-->
        #                                                           next_n(128)

        # this condition determines the border between the two scale_b and whether we have two scale_b in this block or one
        b_scale_next_n = begin_n if begin_n < Int(end_n) else MMA_N
        # Example 1: N = 896, MMA_N = 192, num_of_b_scales: ceildiv(896, BK) = 7
        # This will be the last block on the horizontal axis i.e., work_tile_block[1] == 4
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <------------------------------------------------------------------------------------->|<
        # <-----------------------------------block_scales[6]----------------------------------->|<
        #                                                                                     next_n (192)

        # Example 2: N = 904, MMA_N = 192, num_of_b_scales: ceildiv(N, BK) = 8
        # This will be the last block on the horizontal axis i.e., work_tile_block[1] == 4
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <-------------------------128------------------------------>|<----------64------------>
        # <-------------------------block_scales[6]------------------>|<-----block_scales[7]---->
        #                                                           next_n(128)

        # prefetch b scales
        b_scale_0 = rebind[Scalar[accum_type]](
            b_scales[
                b_scale_m_offset + type_of(b_scale_m_offset)(b_scale_idx0),
                k_iter,
            ].cast[accum_type]()
        )
        # this mean in this block we have two scale_b
        if b_scale_next_n < MMA_N:
            b_scale_1 = rebind[Scalar[accum_type]](
                b_scales[
                    b_scale_m_offset
                    + type_of(b_scale_m_offset)(b_scale_idx0)
                    + 1,
                    k_iter,
                ].cast[accum_type]()
            )
        else:
            b_scale_1 = 0.0

    else:
        # when MMA_N == BK == 128 we only have one scale_b per block
        b_scale_0 = rebind[Scalar[accum_type]](
            b_scales[
                b_scale_m_offset
                + type_of(b_scale_m_offset)(ufloordiv(bn, MMA_N)),
                k_iter,
            ].cast[accum_type]()
        )
        b_scale_1 = 0.0

    var warp_id = get_warp_id()

    # we update the column offset to include the current stage
    var staged_c_row: Int
    var staged_c_col: Int

    comptime if MMA_M == 256 or (MMA_M == 128 and cta_group == 1):
        # based on layout A/D (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-a)
        staged_c_row = warp_id * WARP_SIZE
        staged_c_col = 0
    elif MMA_M == 64 and cta_group == 1:
        # based on layout F (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-f)
        staged_c_row = warp_id * (WARP_SIZE // 2)
        staged_c_col = 0
    else:
        # based on layout B (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-b)
        staged_c_row = umod(warp_id, 2) * WARP_SIZE
        staged_c_col = BN * ufloordiv(warp_id, 2)

    # this is the tensor memory layout
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-fragments-shape-16256b
    # we use it to figure out the starting coordinate
    comptime threads_per_row = (
        stageN // repeats // load_width
    )  # 4 threads per row
    var top_frag_upper_coord = StaticTuple[UInt32, 2](
        UInt32(ufloordiv(lane_id(), threads_per_row)),
        UInt32(umod(lane_id(), threads_per_row) * load_width),
    )

    # getting the other 3 coordinates is straightforward. Each fragment is spaced out by 16 rows
    # and within each fragment the elements are spaced out by 8 rows(this can be seen by the tv layout).
    var bottom_frag_upper_coord = StaticTuple[UInt32, 2](
        top_frag_upper_coord[0] + 8, top_frag_upper_coord[1]
    )

    var top_frag_lower_coord = StaticTuple[UInt32, 2](
        top_frag_upper_coord[0] + 16, top_frag_upper_coord[1]
    )

    var bottom_frag_lower_coord = StaticTuple[UInt32, 2](
        top_frag_lower_coord[0] + 8, top_frag_lower_coord[1]
    )

    var mma_output_stage = mma_output_pipeline.consumer_stage()
    var tmem_offset = mma_output_stage * UInt32(stage_stride_cols) + tmem_addr
    mma_output_pipeline.wait_producer()

    comptime a_scales_smem_tile_size = a_scales_smem_layout.size()
    var a_scales_smem = LayoutTensor[
        a_scales_type,
        a_scales_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](a_scales_smem_base + Int(tma_load_stage_index) * a_scales_smem_tile_size)
    # load a_scales from SMEM
    var upper_sfa0_smem = a_scales_smem[
        0, UInt32(staged_c_row) + top_frag_upper_coord[0]
    ].cast[accum_type]()
    var upper_sfa1_smem = a_scales_smem[
        0, UInt32(staged_c_row) + bottom_frag_upper_coord[0]
    ].cast[accum_type]()

    var lower_sfa0_smem = Scalar[accum_type]()
    var lower_sfa1_smem = Scalar[accum_type]()

    comptime if is_lower_frag_required:
        lower_sfa0_smem = rebind[Scalar[accum_type]](
            a_scales_smem[
                0, UInt32(staged_c_row) + top_frag_lower_coord[0]
            ].cast[accum_type]()
        )
        lower_sfa1_smem = rebind[Scalar[accum_type]](
            a_scales_smem[
                0, UInt32(staged_c_row) + bottom_frag_lower_coord[0]
            ].cast[accum_type]()
        )

    syncwarp()
    if lane_id() < Int(CLUSTER_SIZE):
        _ = load_mma_pipeline.consumer_mbar(tma_load_stage_index)[0].arrive()
    syncwarp()

    comptime rep_frag_size = repeats * fragment_size
    var upper_frag: Array[Scalar[accum_type], rep_frag_size]
    var lower_frag = Array[Scalar[accum_type], rep_frag_size](
        uninitialized=True
    )

    comptime for stage in range(num_stages):
        var stage_tmem_addr = tmem_offset + UInt32(stage * stageN)
        upper_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeats,
            dtype=accum_type,
            pack=False,
            width=rep_frag_size,
        ](stage_tmem_addr)

        comptime if is_lower_frag_required:
            lower_frag = tcgen05_ld[
                datapaths=data_paths,
                bits=bits,
                repeat=repeats,
                dtype=accum_type,
                pack=False,
                width=rep_frag_size,
            ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        comptime if stage == num_stages - 1:
            comptime if cta_group == 1:
                _ = mbarrier_arrive(
                    mma_output_pipeline.consumer_mbar(mma_output_stage)
                )
            else:
                umma_arrive_leader_cta(
                    mma_output_pipeline.consumer_mbar(mma_output_stage)
                )

        var b_scale: Scalar[accum_type]

        comptime if MMA_N != BK:
            # check if we cross the border between the two scale_b
            b_scale = (
                b_scale_0 if (stage * stageN + staged_c_col)
                < b_scale_next_n else b_scale_1
            )
        else:
            b_scale = b_scale_0

        comptime for ld_iter in range(repeats):
            comptime for j in range(fragment_size // 2):
                comptime offset = ld_iter * fragment_size + j * 2

                var upper_a_scale = (
                    upper_sfa0_smem if j == 0 else upper_sfa1_smem
                )
                var lower_a_scale = (
                    lower_sfa0_smem if j == 0 else lower_sfa1_smem
                )

                var upper_scale = upper_a_scale * b_scale
                var lower_scale = lower_a_scale * b_scale

                c_upper_main_tile[stage, offset] += rebind[Scalar[accum_type]](
                    upper_frag[offset]
                ) * rebind[Scalar[accum_type]](upper_scale)
                c_upper_main_tile[stage, offset + 1] += rebind[
                    Scalar[accum_type]
                ](upper_frag[offset + 1]) * rebind[Scalar[accum_type]](
                    upper_scale
                )

                comptime if is_lower_frag_required:
                    c_lower_main_tile[stage, offset] += rebind[
                        Scalar[accum_type]
                    ](lower_frag[offset]) * rebind[Scalar[accum_type]](
                        lower_scale
                    )
                    c_lower_main_tile[stage, offset + 1] += rebind[
                        Scalar[accum_type]
                    ](lower_frag[offset + 1]) * rebind[Scalar[accum_type]](
                        lower_scale
                    )


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
@__name(
    t"blackwell_gmm_warp_specialized_blockwise_fp8_{a_type}_{b_type}_{c_type}",
)
def blackwell_gmm_tma_umma_warp_specialized_blockwise_fp8_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    c_tile_rank: Int,
    c_tile_shape_param: IndexList[c_tile_rank],
    c_desc_shape: IndexList[c_tile_rank],
    a_scales_tile_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_tile_rank],
    a_scales_desc_shape: IndexList[a_scales_tile_rank],
    a_scales_type: DType,
    a_offsets_layout: Layout,
    a_gmem_layout: Layout,
    a_scales_gmem_layout: Layout,
    b_scales_type: DType,
    b_scales_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    num_pipeline_stages: Int,
    cluster_shape: StaticTuple[Int32, 3],
    expert_n: Int,
    expert_ids_layout: Layout,
    b_scales_n: Int,
](
    num_active_experts: Int32,
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c_tma_op: TMATensorTile[
        c_type, c_tile_rank, c_tile_shape_param, c_desc_shape
    ],
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    a_scales_tma_op: TMATensorTile[
        a_scales_type,
        a_scales_tile_rank,
        a_scales_tile_shape,
        a_scales_desc_shape,
    ],
    a_offsets: LayoutTensor[.uint32, a_offsets_layout, ImmutAnyOrigin],
    num_iters: Int32,
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, ImmutAnyOrigin],
    expert_ids: LayoutTensor[.int32, expert_ids_layout, ImmutAnyOrigin],
    problem_shape: StaticTuple[Int32, 3],
    a_gmem: LayoutTensor[a_type, a_gmem_layout, ImmutAnyOrigin],
    a_scales_gmem: LayoutTensor[
        a_scales_type, a_scales_gmem_layout, ImmutAnyOrigin
    ],
):
    var _num_active_experts = Int(num_active_experts)
    var _num_iters = Int(num_iters)
    comptime num_output_warps = 4

    comptime accum_type = get_accum_type[a_type]()

    comptime assert (
        b_scales_type == a_scales_type and accum_type == .float32
    ), "a_scales_type must equal b_scales_type, and accum_type must be float32"
    comptime assert transpose_b, "only support k-major B"

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime CLUSTER_SIZE = config.cluster_shape[0] * config.cluster_shape[1]
    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    # For ld from TMEM, use same per-stage stride in column field.
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = NUM_TMEM_COLS // config.num_accum_pipeline_stages

    comptime clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = config.cta_group * EPILOGUE_THREADS

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime assert BK == 128, "Only support BK = 128"
    comptime assert MMA_N <= BK or gcd(MMA_N, BK) == MMA_N - BK, (
        "MMA_N <= BK or gcd(MMA_N, BK) == MMA_N - BK. MMA_N="
        + String(MMA_N)
        + ", GCD="
        + String(gcd(MMA_N, BK))
    )

    comptime num_m_mmas = BM // (config.mma_shape[0] // config.cta_group)
    comptime num_n_mmas = BN // (config.mma_shape[1] // config.cta_group)
    comptime num_k_mmas = BK // config.mma_shape[2]

    comptime CLUSTER_M: Int = config.cluster_shape[0]
    comptime CLUSTER_N: Int = config.cluster_shape[1]

    comptime a_tma_load_size = _idx_product[a_tile_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]
    comptime c_smem_layout = Layout.row_major(BM, MMA_N)

    # keep the physical SMEM buffer BM x MMA_N
    # Use typed layouts as source of truth; bridge to legacy Layout for
    # LayoutTensor and MMA descriptor pipeline.
    comptime a_smem_layout = tile_layout_k_major_typed[
        a_type, BM, BK, swizzle_mode=config.a_swizzle
    ].to_layout()
    comptime b_smem_layout = tile_layout_k_major_typed[
        b_type, BN, BK, swizzle_mode=config.b_swizzle
    ].to_layout() if transpose_b else tile_layout_mn_major_typed[
        b_type, BN, BK, swizzle_mode=config.b_swizzle
    ].to_layout()

    comptime a_scales_smem_layout = Layout.row_major(1, BM)

    var base_ptr_smem = external_memory[
        Scalar[a_type],
        address_space=.SHARED,
        alignment=128,
    ]()

    comptime a_smem_size = a_smem_layout.size() * num_pipeline_stages
    comptime b_smem_size = b_smem_layout.size() * num_pipeline_stages
    comptime c_smem_size = config.output_tile_shape[
        0
    ] * config.output_tile_shape[1] * config.num_output_stages

    comptime a_scales_smem_size = a_scales_smem_layout.size() * num_pipeline_stages

    var a_smem_base = base_ptr_smem
    var b_smem_base = (a_smem_base + a_smem_size).bitcast[Scalar[b_type]]()
    var c_smem_base = (b_smem_base + b_smem_size).bitcast[Scalar[c_type]]()
    var a_scales_smem_base = (c_smem_base + c_smem_size).bitcast[
        Scalar[a_scales_type]
    ]()

    # TileTensor views of the same SMEM for consumer_main_loop (MMA path).
    var a_smem_tt = SMemTileArray2D[
        a_type,
        BM,
        BK,
        num_pipeline_stages,
        swizzle_mode_to_bytes[config.a_swizzle],
    ](a_smem_base)
    var b_smem_tt = SMemTileArray2D[
        b_type,
        BN,
        BK,
        num_pipeline_stages,
        swizzle_mode_to_bytes[config.b_swizzle],
    ](b_smem_base)
    var load_mma_mbar_ptr = (a_scales_smem_base + a_scales_smem_size).bitcast[
        SharedMemBarrier
    ]()

    # Load warp as producer and mma warp as consumer
    var load_mma_pipeline = ProducerConsumerPipeline[num_pipeline_stages](
        load_mma_mbar_ptr.unsafe_origin_cast[MutUntrackedOrigin]()
    )

    var mma_output_mbar_ptr = load_mma_mbar_ptr + 2 * num_pipeline_stages
    var mma_output_pipeline = ProducerConsumerPipeline[
        config.num_accum_pipeline_stages
    ](mma_output_mbar_ptr.unsafe_origin_cast[MutUntrackedOrigin]())

    var clc_full_mbar_ptr = (
        mma_output_mbar_ptr + 2 * config.num_accum_pipeline_stages
    )
    var clc_empty_mbar_ptr = clc_full_mbar_ptr + config.num_clc_pipeline_stages

    # Load warp as producer and scheduler warp as consumer.
    # No data dependence. Introduce dependence to prevent CLC goes too ahead.
    # In the extreme case, all ctas keep querying next work simultaneously,
    # there will be no guarantee they get balanced number of tiles.
    var load_clc_pipeline = ProducerConsumerPipeline[
        config.num_clc_pipeline_stages
    ](
        (
            clc_empty_mbar_ptr + config.num_clc_pipeline_stages
        ).unsafe_origin_cast[MutUntrackedOrigin]()
    )

    var clc_response_ptr = (
        clc_empty_mbar_ptr + 3 * config.num_clc_pipeline_stages
    ).bitcast[Int128]()

    var tmem_dealloc_mbar_ptr = (
        clc_response_ptr + config.num_clc_pipeline_stages
    ).bitcast[Int64]()

    var ptr_tmem_addr = (tmem_dealloc_mbar_ptr + 1).bitcast[UInt32]()

    var clc_response = clc_response_ptr.bitcast[UInt128]()
    var clc_full_mbar = clc_full_mbar_ptr.bitcast[SharedMemBarrier]()
    var clc_empty_mbar = clc_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    var tmem_dealloc_mbar = tmem_dealloc_mbar_ptr.bitcast[SharedMemBarrier]()

    var elect_one_warp = ufloordiv(thread_idx.x, WARP_SIZE) == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = (
        block_rank_in_cluster() % 2 == 0 if config.cta_group == 2 else True
    )
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    comptime max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()
        a_scales_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            Int32(
                config.cluster_shape[0] // config.cta_group
                + config.cluster_shape[1]
                - 1
                + CLUSTER_SIZE * (EPILOGUE_THREADS // 32)
            ),
        )

        mma_output_pipeline.init_mbars(
            accum_pipeline_producer_arv_count,
            Int32(accum_pipeline_consumer_arv_count),
        )
        load_clc_pipeline.init_mbars(
            Int32(clc_throttle_producer_arv_count),
            Int32(clc_throttle_consumer_arv_count),
        )

        tmem_dealloc_mbar[].init(Int32(EPILOGUE_THREADS * config.cta_group))

    comptime for i in range(config.num_clc_pipeline_stages):
        clc_full_mbar[i].init(clc_producer_arv_count)
        clc_empty_mbar[i].init(Int32(clc_consumer_arv_count))

    fence_mbarrier_init()
    cluster_sync()

    var clc_pipe_producer_state = PipelineState[config.num_clc_pipeline_stages](
        0, 1, 0
    )
    var clc_pipe_consumer_state = PipelineState[
        config.num_clc_pipeline_stages
    ]()

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        config.block_tile_shape,
        config.mma_shape,
        accum_type=accum_type,
        cta_group=config.cta_group,
        cluster_shape=config.cluster_shape,
        a_swizzle=config.a_swizzle,
        b_swizzle=config.b_swizzle,
        transpose_b=transpose_b,
    ]()

    # var scheduler = TileScheduler[
    #     num_stages = config.num_clc_pipeline_stages,
    #     cluster_shape = Index[dtype = DType.uint32](
    #         config.cluster_shape[0],
    #         config.cluster_shape[1],
    #         config.cluster_shape[2],
    #     ),
    #     block_swizzle_size = config.block_swizzle_size,
    #     rasterize_order = config.raster_order,
    # ](cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar)

    var scheduler = TileScheduler[
        static_MN=expert_n,
        cluster=Index(
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        cta_group=config.cta_group,
        tile_shape=Index(
            config.block_tile_shape[0],
            config.block_tile_shape[1],
            config.block_tile_shape[2],
        ),
        swapAB=False,
    ](_num_active_experts, a_offsets)

    var work_info = scheduler.fetch_next_work()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        umod(rank_m, config.cta_group),
        ufloordiv(rank_m, config.cta_group),
        rank_n,
    )  # v,m,n

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    # TODO: find a generic way to calculate multicast mask
    comptime for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))
    # they all have the same v and m, but different n,

    comptime for i in range(CLUSTER_M // config.cta_group):
        b_multicast_mask |= UInt16(1 << (i * config.cta_group))

    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * CLUSTER_M)

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    if WarpRole.is_main_load():
        # var required_clc_query = True

        var total_m = Int(a_scales_gmem.dim[1]())

        while not work_info.is_done():
            if (
                not work_info.is_valid()
                or expert_ids[Int(scheduler.current_group_idx)] < 0
            ):
                work_info = scheduler.fetch_next_work()
                continue
            # DO TMA LOAD

            var expert_end_row = Int(
                rebind[UInt32](
                    scheduler.group_offsets[
                        Int(scheduler.current_group_idx + 1)
                    ]
                )
            )
            var m_tile_global_start = Int(work_info.m)
            var use_full_tma = _tile_fits_full_tma[
                a_scales_type, config.block_tile_shape[0]
            ](m_tile_global_start, total_m)

            for i in range(_num_iters):
                if not use_full_tma:
                    load_AB_partial[
                        a_smem_layout=a_smem_layout,
                        b_smem_layout=b_smem_layout,
                        a_scales_smem_layout=a_scales_smem_layout,
                        block_tile_shape=config.block_tile_shape,
                        cta_group=config.cta_group,
                        a_swizzle=config.a_swizzle,
                    ](
                        a_gmem,
                        a_scales_gmem,
                        b_tma_op,
                        a_smem_base,
                        b_smem_base,
                        a_scales_smem_base,
                        load_mma_pipeline,
                        peer_cta_coord,
                        (Int(work_info.m), Int(work_info.n)),
                        b_multicast_mask,
                        i,
                        elect_one_cta,
                        scheduler,
                        expert_ids,
                        expert_end_row,
                        m_tile_global_start,
                    )
                    load_mma_pipeline.producer_step()
                    continue
                load_AB[
                    a_smem_layout=a_smem_layout,
                    b_smem_layout=b_smem_layout,
                    a_scales_smem_layout=a_scales_smem_layout,
                    block_tile_shape=config.block_tile_shape,
                    mma_shape=config.mma_shape,
                    cta_group=config.cta_group,
                ](
                    a_tma_op,
                    b_tma_op,
                    a_scales_tma_op,
                    a_smem_base,
                    b_smem_base,
                    a_scales_smem_base,
                    load_mma_pipeline,
                    peer_cta_coord,
                    (Int(work_info.m), Int(work_info.n)),
                    a_multicast_mask,
                    b_multicast_mask,
                    i,
                    elect_one_cta,
                    scheduler,
                    expert_ids,
                )
                load_mma_pipeline.producer_step()

            syncwarp()
            var next_work_info = scheduler.fetch_next_work()
            work_info = next_work_info

        comptime for i in range(num_pipeline_stages):
            load_mma_pipeline.wait_consumer()
            load_mma_pipeline.producer_step()

    if WarpRole.is_mma():
        tcgen05_alloc[Int32(config.cta_group)](ptr_tmem_addr, max_tmem_cols)
        syncwarp()
        # non blocking, arrives and proceeds
        named_barrier_arrive[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)

        var tmem_addr = ptr_tmem_addr[0]

        while not work_info.is_done():
            if (
                not work_info.is_valid()
                or expert_ids[Int(scheduler.current_group_idx)] < 0
            ):
                work_info = scheduler.fetch_next_work()
                continue
            # scheduler fetch next work
            var next_work_info = scheduler.fetch_next_work()
            # DO MMA
            if elect_one_cta:
                for _ in range(_num_iters):
                    var mma_output_mma_stage = (
                        mma_output_pipeline.producer_stage()
                    )
                    mma_output_pipeline.wait_consumer()
                    var tmem_offset = tmem_addr + (
                        mma_output_mma_stage * UInt32(stage_stride_cols)
                    )

                    consumer_main_loop[
                        block_tile_shape=config.block_tile_shape,
                        mma_shape=config.mma_shape,
                        cta_group=config.cta_group,
                        cluster_shape=config.cluster_shape,
                    ](
                        tmem_offset,
                        a_smem_tt,
                        b_smem_tt,
                        load_mma_pipeline,
                        mma_op,
                        elect_one_warp,
                        0,
                        0,
                    )
                    load_mma_pipeline.consumer_step()

                    # mma arrive multicast will track completion of all mma prior to this barrier.
                    if elect_one_sync():
                        comptime if config.cta_group == 1:
                            mma_arrive[config.cta_group](
                                mma_output_pipeline.producer_mbar(
                                    mma_output_mma_stage
                                )
                            )
                        else:
                            mma_arrive_multicast[config.cta_group](
                                mma_output_pipeline.producer_mbar(
                                    mma_output_mma_stage
                                ),
                                UInt16(mma_complete_mask),
                            )
                    mma_output_pipeline.producer_step()

            work_info = next_work_info

        tcgen05_release_allocation_lock[Int32(config.cta_group)]()

        # wait for epilogue to finish
        tmem_dealloc_mbar[].wait()

        tcgen05_dealloc[Int32(config.cta_group)](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)
        var tmem_addr = ptr_tmem_addr[0]

        while not work_info.is_done():
            if not work_info.is_valid():
                work_info = scheduler.fetch_next_work()
                continue

            # TODO: zero output

            comptime c_smem_layout = Layout.row_major(
                config.output_tile_shape[0], config.output_tile_shape[1]
            )
            comptime reg_info = _get_accumulator_size[
                c_smem_layout=c_smem_layout,
                block_tile_shape=config.block_tile_shape,
                mma_shape=config.mma_shape,
                cta_group=config.cta_group,
            ]()

            comptime is_lower_frag_required = not (
                config.cta_group == 1 and config.block_tile_shape[0] == 64
            )
            # final results accumulator regs for C
            var c_upper_main_tile = LayoutTensor[
                accum_type,
                Layout.row_major(reg_info[0], reg_info[1]),
                MutAnyOrigin,
                address_space=.LOCAL,
            ].stack_allocation()

            var c_lower_main_tile = LayoutTensor[
                accum_type,
                Layout.row_major(reg_info[0], reg_info[1]),
                MutAnyOrigin,
                address_space=.LOCAL,
            ].stack_allocation()

            _ = c_upper_main_tile.fill(0.0)

            comptime if is_lower_frag_required:
                _ = c_lower_main_tile.fill(0.0)

            for k_iter in range(_num_iters):
                promote_accumulators[
                    a_scales_smem_layout=a_scales_smem_layout,
                    block_tile_shape=config.block_tile_shape,
                    mma_shape=config.mma_shape,
                    cta_group=config.cta_group,
                    CLUSTER_SIZE=Int32(CLUSTER_SIZE),
                    is_lower_frag_required=is_lower_frag_required,
                    num_output_warps=num_output_warps,
                ](
                    b_scales,
                    b_scales_n,
                    a_scales_smem_base,
                    c_upper_main_tile,
                    c_lower_main_tile,
                    # accum_pipeline_consumer_state,
                    mma_output_pipeline,
                    tmem_addr,
                    load_mma_pipeline,
                    work_tile_coord=(Int(work_info.m), Int(work_info.n)),
                    elect_one_warp=elect_one_warp,
                    stage_stride_cols=stage_stride_cols,
                    k_iter=k_iter,
                    problem_shape=problem_shape,
                    expert_ids=expert_ids,
                    scheduler=scheduler,
                )
                load_mma_pipeline.consumer_step()
                mma_output_pipeline.consumer_step()

            # TODO (KERN-2081): investigate why this barrier is needed and if we can move/remove it
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()

            # wait for CUDA core promotion to finish and store result
            # scheduler fetch next work
            multi_stage_reg_epilogue[
                c_smem_layout=c_smem_layout,
                c_static_N=expert_n,
                block_tile_shape=config.block_tile_shape,
                mma_shape=config.mma_shape,
                is_lower_frag_required=is_lower_frag_required,
                cta_group=config.cta_group,
                num_output_warps=num_output_warps,
                c_swizzle=config.c_swizzle,
            ](
                c_upper_main_tile,
                c_lower_main_tile,
                c_smem_base.as_unsafe_any_origin(),
                c_tma_op,
                c_ptr,
                c_coord=(Int(work_info.m), Int(work_info.n)),
                elect_one_warp=elect_one_warp,
                group_end_idx=rebind[UInt32](
                    scheduler.group_offsets[
                        Int(scheduler.current_group_idx + 1)
                    ]
                ),
            )

            var next_work_info = scheduler.fetch_next_work()
            work_info = next_work_info

        comptime if config.cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


def grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    transpose_b: Bool,
    //,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    a_offsets: TileTensor[
        mut=False, a_offsets_type, address_space=.GENERIC, ...
    ],
    expert_ids: TileTensor[
        mut=False, expert_ids_type, address_space=.GENERIC, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launches the persistent warp-specialized SM100 blockwise-scaled FP8 grouped GEMM kernel.

    Builds TMA descriptors for A, B, C, and A scales, sizes the
    pipeline stages from available SMEM, and enqueues
    `blackwell_gmm_tma_umma_warp_specialized_blockwise_fp8_kernel` with a
    grid of one block per SM. Falls back to the naive blockwise FP8
    grouped matmul when the A scales K-row stride is not 16-byte aligned.

    Parameters:
        c_type: Element type of the output `C` tensor (inferred).
        a_type: Element type of the input `A` tensor (inferred). Must be
            `float8_e4m3fn`.
        b_type: Element type of the input `B` tensor (inferred). Must be
            `float8_e4m3fn`.
        a_scales_type: Element type of the `a_scales` tensor (inferred).
            Must equal `b_scales_type`.
        b_scales_type: Element type of the `b_scales` tensor (inferred).
            Must equal `a_scales_type`.
        a_offsets_type: Element type of the `a_offsets` tensor (inferred).
        expert_ids_type: Element type of the `expert_ids` tensor (inferred).
        transpose_b: Whether `B` is stored transposed (inferred). Must be
            `True`.
        config: Matmul configuration specifying block tile shape, MMA
            shape, and TMA swizzle modes. Must have `cta_group == 1` and
            `cluster_shape == (1, 1, 1)`.
        elementwise_lambda_fn: Optional epilogue function applied to each
            output element before storing (defaults to `None`).

    Args:
        c: Output tensor of shape `[total_tokens, N]` holding the
            grouped matmul results.
        a: Input activation tensor of shape `[total_tokens, K]` in FP8.
            `K` must be a multiple of `BK`.
        b: Input weight tensor of shape `[num_experts, N, K]` in FP8.
        a_scales: Per-block scales for `A` of shape `[K // BK,
            total_tokens]` where `BK` is the scaling block size. The
            `total_tokens * size_of(a_scales_type)` byte stride must be
            16-byte aligned or this kernel falls back to the naive path.
        b_scales: Per-block scales for `B` of shape `[num_experts, N // BN,
            K // BK]` where `BN` and `BK` are the scaling block sizes.
        a_offsets: Cumulative row offsets per expert, length
            `num_active_experts + 1`. Entry `i + 1` minus entry `i` gives
            the row count for expert `i`.
        expert_ids: Expert index for each active expert slot, mapping the
            grid Z index to the corresponding row offset in `B`.
        max_num_tokens_per_expert: Maximum number of tokens assigned to any
            single expert, used to size the grid M dimension.
        num_active_experts: Number of active experts in this grouped
            matmul, used to size the grid Z dimension.
        ctx: Device context used to enqueue the kernel.
    """
    comptime assert config.cta_group == 1, "Only support cta_group == 1"
    comptime assert (
        config.cluster_shape[0] == 1
        and config.cluster_shape[1] == 1
        and config.cluster_shape[2] == 1
    ), "Only support cluster_shape == (1, 1, 1). Got " + String(
        config.cluster_shape
    )
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn"

    comptime assert (
        a_scales_type == b_scales_type
    ), "a_scales_type must equal b_scales_type"

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert not config.AB_swapped, "Swapped AB is not supported"

    # Extract compile-time shapes from TileTensors.
    comptime num_experts = b.static_shape[0]
    comptime N = c.static_shape[1]
    comptime K = a.static_shape[1]

    # `_copy_partial_a_tile_blockwise_from_gmem` walks K in BK strides without
    # masking the K-tail, so any caller must size K to a BK multiple.
    comptime assert K % BK == 0, "K must be a multiple of BK"

    # Convert TileTensors to LayoutTensors at the kernel boundary.
    var a_tensor = a.to_layout_tensor()
    var c_tensor = c.to_layout_tensor()
    var a_scales_tensor = a_scales.to_layout_tensor()
    var a_offsets_tensor = a_offsets.to_layout_tensor()
    var expert_ids_tensor = expert_ids.to_layout_tensor()

    # `create_tma_tile` rejects an `a_scales` whose K-row byte stride
    # (= total_m * size_of(scales)) is not 16-byte aligned. Detect that
    # at the host and fall back to the naive grouped FP8 kernel — the
    # per-tile `_tile_fits_full_tma` predicate alone is not enough,
    # because descriptor creation fails before any tile runs.
    var total_m = Int(a_scales_tensor.dim(1))
    if total_m * size_of[a_scales_type]() % 16 != 0:
        var b_tensor_n = b.to_layout_tensor()
        var b_scales_tensor_n = b_scales.to_layout_tensor()
        naive_blockwise_scaled_fp8_grouped_matmul[
            BLOCK_DIM_M=16,
            BLOCK_DIM_N=16,
            transpose_b=transpose_b,
            scales_granularity_mnk=Index(1, 128, 128),
        ](
            c_tensor,
            a_tensor,
            b_tensor_n,
            a_scales_tensor,
            b_scales_tensor_n,
            a_offsets_tensor,
            expert_ids_tensor,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )
        return

    var a_tma_op = create_tensor_tile[
        Index(BM // config.cluster_shape[1], BK),
        swizzle_mode=config.a_swizzle,
    ](ctx, a_tensor)

    comptime expert_n = N
    # Reshape 3D weights to 2D for TMA.
    var b_2d = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        address_space=.GENERIC,
    ](b.ptr.as_unsafe_any_origin())
    var b_tma_op = create_tensor_tile[
        Index(
            BN // (config.cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (config.cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode=config.b_swizzle,
    ](ctx, b_2d)

    var a_scales_tma_op = create_tma_tile[1, BM](ctx, a_scales_tensor)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(64, config.output_tile_shape[1])
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    var c_tma_op = create_tensor_tile[
        c_tma_tile_shape,
        swizzle_mode=config.c_swizzle,
    ](ctx, c_tensor)

    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    comptime a_smem_bytes_per_stage = BM * BK * size_of[a_type]()
    comptime b_smem_bytes_per_stage = BN * BK * size_of[b_type]()
    comptime a_scales_smem_bytes_per_stage = BM * size_of[a_scales_type]()
    comptime AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage

    comptime c_smem_bytes = config.output_tile_shape[
        0
    ] * config.output_tile_shape[1] * config.num_output_stages * size_of[
        c_type
    ]()

    comptime MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
    comptime CLC_RESPONSE_BYTES = size_of[Int128]()  # 16 bytes per response
    comptime TMEM_ADDR_BYTES = size_of[
        Int32
    ]()  # 4 bytes or 32 bits for tensor memory address

    comptime accum_full_mbar_bytes = MBAR_BYTES * config.num_accum_pipeline_stages
    comptime accum_empty_mbar_bytes = MBAR_BYTES * config.num_accum_pipeline_stages

    comptime clc_response_bytes = CLC_RESPONSE_BYTES * config.num_clc_pipeline_stages
    comptime clc_full_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_empty_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_throttle_full_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_throttle_empty_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages

    comptime tmem_addr_bytes = TMEM_ADDR_BYTES
    comptime tmem_dealloc_mbar_bytes = MBAR_BYTES

    comptime tmem_writeout_smem = c_smem_bytes + tmem_addr_bytes + tmem_dealloc_mbar_bytes
    comptime accum_smem = accum_full_mbar_bytes + accum_empty_mbar_bytes
    comptime clc_smem = (
        clc_response_bytes
        + clc_full_mbar_bytes
        + clc_empty_mbar_bytes
        + clc_throttle_full_mbar_bytes
        + clc_throttle_empty_mbar_bytes
    )
    comptime smem_leftover = (b200_smem) - (
        clc_smem + accum_smem + tmem_writeout_smem
    )

    comptime tma_mbar_bytes_per_stage = MBAR_BYTES
    comptime mma_mbar_bytes_per_stage = MBAR_BYTES

    comptime producer_consumer_smem_per_stage = (
        AB_smem_per_stage
        + a_scales_smem_bytes_per_stage
        + tma_mbar_bytes_per_stage
        + mma_mbar_bytes_per_stage
    )

    comptime max_pipeline_stages: Int = (
        smem_leftover // producer_consumer_smem_per_stage
    )

    comptime assert (
        max_pipeline_stages >= 1
    ), "not enough smem even for one pipeline stage!"

    comptime producer_consumer_smem = producer_consumer_smem_per_stage * max_pipeline_stages

    comptime smem_size = (
        clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
    )

    # Extract scale shapes from TileTensors and reshape b_scales to 2D.
    comptime b_scales_expert = b_scales.static_shape[0]
    comptime b_scales_n = b_scales.static_shape[1]
    comptime b_scales_k = b_scales.static_shape[2]
    comptime a_scales_k = a_scales.static_shape[0]
    var b_scales_2d = LayoutTensor[
        b_scales_type,
        Layout.row_major(b_scales_expert * b_scales_n, b_scales_k),
        address_space=.GENERIC,
    ](b_scales.ptr.as_unsafe_any_origin())

    comptime kernel = blackwell_gmm_tma_umma_warp_specialized_blockwise_fp8_kernel[
        a_type,
        b_type,
        c_type,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).rank,
        type_of(c_tma_op).tile_shape,
        type_of(c_tma_op).desc_shape,
        type_of(a_scales_tma_op).rank,
        type_of(a_scales_tma_op).tile_shape,
        type_of(a_scales_tma_op).desc_shape,
        a_scales_type,
        type_of(a_offsets_tensor).layout,
        type_of(a_tensor).layout,
        type_of(a_scales_tensor).layout,
        b_scales_type,
        b_scales_2d.layout,
        transpose_b=transpose_b,
        config=config,
        num_pipeline_stages=max_pipeline_stages,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        expert_n=expert_n,
        expert_ids_layout=type_of(expert_ids_tensor).layout,
        b_scales_n=b_scales_n,
    ]

    # TODO
    var grid_dim = (
        B200.sm_count,
        1,
        1,
    )

    comptime cluster_shape = config.cluster_shape

    # TODO
    var problem_shape = StaticTuple[Int32, 3](
        Int32(max_num_tokens_per_expert), Int32(N), Int32(K)
    )

    ctx.enqueue_function[kernel, dump_asm=False](
        Int32(num_active_experts),
        a_tma_op,
        b_tma_op,
        c_tma_op,
        c_tensor.ptr,
        a_scales_tma_op,
        a_offsets_tensor,
        Int32(ceildiv(K, BK)),
        b_scales_2d,
        expert_ids_tensor,
        problem_shape,
        a_tensor,
        a_scales_tensor,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(32 * 7),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )


# ===----------------------------------------------------------------------=== #
# TileTensor primary implementation
# ===----------------------------------------------------------------------=== #


@always_inline
def grouped_matmul_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    //,
    input_scale_granularity: StaticString,
    weight_scale_granularity: StaticString,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    transpose_b: Bool = False,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    a_offsets: TileTensor[
        mut=False, a_offsets_type, address_space=.GENERIC, ...
    ],
    expert_ids: TileTensor[
        mut=False, expert_ids_type, address_space=.GENERIC, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """TileTensor primary implementation of `grouped_matmul_dynamic_scaled_fp8`.
    """
    comptime assert c.rank == 2 and c.flat_rank == 2
    comptime assert a.rank == 2 and a.flat_rank == 2
    comptime assert b.rank == 3 and b.flat_rank == 3
    comptime assert a_scales.rank == 2 and a_scales.flat_rank == 2
    comptime assert b_scales.rank == 3 and b_scales.flat_rank == 3
    comptime assert a_offsets.rank == 1 and a_offsets.flat_rank == 1
    comptime assert expert_ids.rank == 1 and expert_ids.flat_rank == 1

    comptime assert (
        _is_sm10x_gpu(ctx.default_device_info)
        or ctx.default_device_info == H100
    ), "Only support SM100 or SM90"
    comptime assert (
        m_scale_granularity == 1
        and n_scale_granularity == k_scale_granularity == 128
    ), "Only support (1,128,128) scale granularity"
    comptime assert transpose_b, "Only support transpose_b = True"
    comptime assert (
        a_type == b_type == .float8_e4m3fn
    ), "input A and B dtype should be float8_e4m3fn"
    comptime assert a_scales_type == b_scales_type and (
        a_scales_type == .float32 or a_scales_type == .bfloat16
    ), "input A and B scales dtype should be float32 or bfloat16"
    comptime assert (
        input_scale_granularity == "block"
        and weight_scale_granularity == "block"
    ), "Only support block-wise scale granularity"
    comptime assert a_offsets_type == .uint32, (
        "Only uint32 is supported for a_offsets in grouped blockwise scaled"
        " fp8 matmul"
    )
    comptime assert expert_ids_type == .int32, (
        "Only int32 is supported for expert_ids in grouped blockwise scaled"
        " fp8 matmul"
    )

    # Materialize runtime M dimension (may be required for TileTensor dim
    # caching before the value is used inside dispatch functions).
    _ = Int(a.dim[0]())

    if num_active_experts == 0 or max_num_tokens_per_expert == 0:
        return

    @always_inline
    def description_fn() {
        var c, var a, var a_scales, var b_scales, imm
    } -> String:
        # fmt: off
        return String(
            "(gpu",
            ";A=", c.dim(0), "x", a.dim(1), "x", a_type,
            ";C=", c.dim(0), "x", c.dim(1), "x", c_type,
            ";A_scales=[", a_scales.dim(0), ",", a_scales.dim(1), "]",
            ";B_scales=[", b_scales.dim(0), ",", b_scales.dim(1), ",", b_scales.dim(2), "]",
            ";num_experts=", b.static_shape[0],
            ";num_active_experts=", num_active_experts,
            ";max_num_tokens_per_expert=", max_num_tokens_per_expert,
            ";scale_granularity=(", m_scale_granularity, ",", n_scale_granularity, ",", k_scale_granularity, ")",
            ")"
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        get_static_string[
            "grouped_matmul_dynamic_scaled_fp8_",
            String(a_type) + "x" + String(b_type) + "_to_" + String(c_type),
            "_scales_" + String(a_scales_type),
        ](),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(ctx),
    ):
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            # MMA_N=128 halves the number of dispatched (expert, n) tiles vs
            # 64, which dominates runtime for small-M-per-expert decode.
            comptime umma_shape: IndexList[3] = Index(64, 128, 32)

            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                cluster_shape=Index(1, 1, 1),
                mma_shape=umma_shape,
                cta_group=1,
                AB_swapped=False,
                k_group_size=1,
            )
            # Pass TileTensors directly — conversion happens inside.
            grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
                config=config,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                a_offsets,
                expert_ids,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )
            return

        else:
            # Convert to LayoutTensor for naive fallback.
            var a_tensor = a.to_layout_tensor()
            var b_tensor = b.to_layout_tensor()
            var c_tensor = c.to_layout_tensor()
            var a_scales_tensor = a_scales.to_layout_tensor()
            var b_scales_tensor = b_scales.to_layout_tensor()
            var a_offsets_tensor = a_offsets.to_layout_tensor()
            var expert_ids_tensor = expert_ids.to_layout_tensor()
            naive_blockwise_scaled_fp8_grouped_matmul[
                BLOCK_DIM_M=16,
                BLOCK_DIM_N=16,
                transpose_b=transpose_b,
                scales_granularity_mnk=Index(
                    m_scale_granularity,
                    n_scale_granularity,
                    k_scale_granularity,
                ),
            ](
                c_tensor,
                a_tensor,
                b_tensor,
                a_scales_tensor,
                b_scales_tensor,
                a_offsets_tensor,
                expert_ids_tensor,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )
