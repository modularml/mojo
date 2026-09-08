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

"""Provides block-scaled grouped GEMM for 1D×1D scale layouts on SM100 (B200) GPUs."""

from std.math import ceildiv
from std.math.uutils import umod, ufloordiv
from std.sys import align_of, simd_width_of, size_of

from max.gpu import WARP_SIZE
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200, _is_sm10x_gpu
from max.gpu import (
    block_id_in_cluster,
    lane_id,
    thread_idx,
    warp_id as get_warp_id,
)
from max.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    pdl_launch_attributes,
    PDLLevel,
    wait_on_dependent_grids,
)
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from std.collections.string.string_span import get_static_string
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    mma_arrive,
    mma_arrive_multicast,
)
from max.gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
)
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    LTToTTLayout,
    RuntimeLayout,
    RuntimeTuple,
    TileTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
)
from layout.layout import zipped_divide
from layout.layout_tensor import upcast
from layout.runtime_tuple import idx2crd
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
    tile_sf_layout_k_major_typed,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_BlockScaled_SS
from linalg.utils import elementwise_compute_lambda_type
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig, GEMMKind
from structured_kernels.tile_types import (
    SMemTile,
    internal_sf_k_major,
    sf_tile_dim0,
)
from .grouped_matmul_tile_scheduler import TileScheduler
from linalg.matmul.gpu.profiler import (
    MatmulProfileWarp,
    MatmulWarpSpecializationWorkSpaceManager,
)
from linalg.matmul.gpu.sm100.pipeline import ProducerConsumerPipeline
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_K_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from linalg.matmul.gpu.sm100.matmul import (
    WarpRole as _WarpRole,
    stsm_helper,
    shared_memory_epilogue,
    register_epilogue,
    accum_arrive,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind

from std.math.uutils import ufloordiv

# Use WarpRole without scheduler warp for grouped matmul
comptime WarpRole = _WarpRole[has_scheduler=False]


@always_inline
def copy_accum_to_gmem[
    c_type: DType,
    c_tile_rank: Int,
    c_tile_shape: IndexList[c_tile_rank],
    c_desc_shape: IndexList[c_tile_rank],
    num_accum_pipeline_stages: Int,
    c_tensor_layout: Layout,
    /,
    *,
    c_smem_layout: Layout,
    repeat: Int,
    accum_type: DType,
    cta_group: Int,
    epilogue_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    transpose_c: Bool = False,
    scale_c_coord: Bool = True,
](
    c_smem_base: UnsafePointer[
        Scalar[c_type], MutAnyOrigin, address_space=.SHARED
    ],
    c_tma_op: TMATensorTile[c_type, c_tile_rank, c_tile_shape, c_desc_shape],
    c: LayoutTensor[c_type, c_tensor_layout, MutAnyOrigin],
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_pipeline_stages],
    mma_output_stage: UInt32,
    tmem_offset: UInt32,
    c_coord: Tuple[UInt32, UInt32],
    c_shape: Tuple[UInt32, UInt32],
    expert_scale: Float32,
    group_end_idx: UInt32,
):
    """Copies one accumulator stage from TMEM to global memory via shared memory.

    Loads the MMA result from tensor memory, applies the per-expert scale and an
    optional elementwise epilogue, packs the result through `stmatrix` into shared
    memory, and stores it to global memory with TMA (or a CUDA-core fallback for
    unaligned tails). Handles both the transposed (`transpose_c`) and
    non-transposed output layouts.

    Parameters:
        c_type: The data type of the output tensor C.
        c_tile_rank: The rank of the C TMA tile.
        c_tile_shape: The shape of the C TMA tile.
        c_desc_shape: The descriptor shape of the C TMA tile.
        num_accum_pipeline_stages: The number of accumulator pipeline stages.
        c_tensor_layout: The layout of the output LayoutTensor.
        c_smem_layout: The layout of the C shared memory tile.
        repeat: The number of `tcgen05_ld` repetitions per stage.
        accum_type: The accumulator data type held in TMEM.
        cta_group: The number of CTAs cooperating per output tile (1 or 2).
        epilogue_dtype: The dtype used for the cast output before storing.
        block_tile_shape: The (BM, BN, BK) block tile shape.
        mma_shape: The (MMA_M, MMA_N, MMA_K) MMA shape.
        num_output_warps: The number of epilogue warps.
        c_swizzle: The TMA swizzle mode for C.
        elementwise_compute_lambda_fn: Optional fused elementwise epilogue.
        register_based_epilogue: Whether the epilogue runs in registers.
        transpose_c: Whether the output is stored transposed.
        scale_c_coord: Whether to scale tile coordinates by block tile strides.

    Args:
        c_smem_base: Base pointer to the C shared memory buffer.
        c_tma_op: The TMA tensor tile descriptor for C.
        c: The output LayoutTensor in global memory.
        mma_output_pipeline: The producer/consumer pipeline for MMA output.
        mma_output_stage: The accumulator pipeline stage to drain.
        tmem_offset: The column offset into TMEM for this stage.
        c_coord: The (row, column) tile coordinate of the output tile.
        c_shape: The (M, N) extent of the output tensor.
        expert_scale: The per-expert scale factor applied to the accumulator.
        group_end_idx: The exclusive end row index for the current expert group.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime simd_size = simd_width_of[c_type]()

    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()
    comptime stage_contiguous_size = c_smem_layout.shape[1].value()
    comptime data_paths = 16  # same as lanes
    comptime bits = 256
    comptime fragment_size = (data_paths * (bits // 32)) // WARP_SIZE
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    comptime rep_frag_size = repeat * fragment_size
    var upper_frag_partial: Array[Scalar[accum_type], rep_frag_size]
    var lower_frag_partial = Array[Scalar[accum_type], rep_frag_size](
        uninitialized=True
    )
    var upper_frag_casted = Array[Scalar[epilogue_dtype], rep_frag_size](
        uninitialized=True
    )
    var lower_frag_casted = Array[Scalar[epilogue_dtype], rep_frag_size](
        uninitialized=True
    )
    var scale = expert_scale.cast[accum_type]()

    comptime is_lower_frag_required = not (cta_group == 1 and BM == 64)
    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages

    var M = c_shape[0]
    var N = c_shape[1]

    # stmatrix related
    comptime st_matrix_swizzle = c_swizzle
    comptime swizzle_width = c_swizzle.bytes() // size_of[c_type]()
    comptime swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    # lets keep track of the of the starting row and column in GMEM
    var c_row = c_coord[0] * UInt32(BM) if scale_c_coord else c_coord[0]
    var c_col = c_coord[1] * UInt32(MMA_N) if scale_c_coord else c_coord[1]

    comptime for stage in range(num_stages):
        var stage_tmem_addr = tmem_offset + UInt32(stage * stageN)
        upper_frag_partial = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=accum_type,
            pack=False,
            width=rep_frag_size,
        ](stage_tmem_addr)

        comptime if is_lower_frag_required:
            lower_frag_partial = tcgen05_ld[
                datapaths=data_paths,
                bits=bits,
                repeat=repeat,
                dtype=accum_type,
                pack=False,
                width=rep_frag_size,
            ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        comptime if stage == num_stages - 1:
            accum_arrive[cta_group](mma_output_pipeline, mma_output_stage)

        # Fused scale + cast using Blackwell-supported SIMD widths.
        comptime cast_width = 4 // size_of[Scalar[epilogue_dtype]]()
        var scale_vec = SIMD[accum_type, cast_width](scale)

        comptime for _i in range(rep_frag_size // cast_width):
            comptime offset = _i * cast_width
            var src = SIMD[accum_type, cast_width]()
            comptime for _j in range(cast_width):
                src[_j] = upper_frag_partial[offset + _j]
            var scaled = (src * scale_vec).cast[epilogue_dtype]()
            comptime for _j in range(cast_width):
                upper_frag_casted[offset + _j] = scaled[_j]

        comptime if is_lower_frag_required:
            comptime for _i in range(rep_frag_size // cast_width):
                comptime offset = _i * cast_width
                var src = SIMD[accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = lower_frag_partial[offset + _j]
                var scaled = (src * scale_vec).cast[epilogue_dtype]()
                comptime for _j in range(cast_width):
                    lower_frag_casted[offset + _j] = scaled[_j]

        comptime if elementwise_compute_lambda_fn:
            comptime if register_based_epilogue:
                register_epilogue[
                    MMA_M,
                    data_paths,
                    num_stages,
                    bits,
                    stage,
                    stageN,
                    elementwise_compute_lambda_fn.value(),
                    num_output_warps,
                    epilogue_dtype,
                    upper_frag_casted.length,
                    repeat,
                    transpose_c,
                    cta_group=cta_group,
                    is_lower_frag_required=is_lower_frag_required,
                ](upper_frag_casted, lower_frag_casted, c_row, c_col, N)

        # Assume double-buffer for shared memory packing
        comptime c_smem_tile_size = c_smem_layout.size()
        var c_smem_tile = LayoutTensor[
            c_type,
            c_smem_layout,
            MutAnyOrigin,
            address_space=.SHARED,
            alignment=128,
        ](c_smem_base + (stage % 2) * c_smem_tile_size)

        comptime if transpose_c:
            # With SWIZZLE_32B, each swizzle_width chunk (16 bf16 elements)
            # in smem maps to one TMA row. We tile the smem into
            # row_major(stageN, swizzle_width) regions, one per warp frag.
            comptime tiles_per_frag = stageN * swizzle_width // stage_contiguous_size

            comptime if is_lower_frag_required:
                var c_smem_warp_tile_upper = c_smem_tile.tile[
                    tiles_per_frag, stage_contiguous_size
                ](2 * warp_id, 0).reshape[
                    Layout.row_major(stageN, swizzle_width)
                ]()
                var c_smem_warp_tile_lower = c_smem_tile.tile[
                    tiles_per_frag, stage_contiguous_size
                ](2 * warp_id + 1, 0).reshape[
                    Layout.row_major(stageN, swizzle_width)
                ]()

                stsm_helper[swizzle, stageN, transpose_c=transpose_c](
                    upper_frag_casted, lt_to_tt(c_smem_warp_tile_upper)
                )
                stsm_helper[swizzle, stageN, transpose_c=transpose_c](
                    lower_frag_casted, lt_to_tt(c_smem_warp_tile_lower)
                )
            else:
                var c_smem_warp_tile_upper = c_smem_tile.tile[
                    tiles_per_frag, stage_contiguous_size
                ](warp_id, 0).reshape[Layout.row_major(stageN, swizzle_width)]()

                stsm_helper[swizzle, stageN, transpose_c=transpose_c](
                    upper_frag_casted, lt_to_tt(c_smem_warp_tile_upper)
                )

            # Guard the write to shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()

            comptime if elementwise_compute_lambda_fn:
                comptime if not register_based_epilogue:
                    # TODO: Update shared_memory_epilogue_transpose for
                    # SWIZZLE_32B layout if needed.
                    pass
        else:
            comptime c_smem_tile_m = 32 if cta_group == 2 else BM // num_output_warps
            var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, stageN](
                warp_id, 0
            )
            var c_smem_warp_tt = lt_to_tt(c_smem_warp_tile)

            stsm_helper[swizzle, stageN, transpose_c=transpose_c](
                upper_frag_casted,
                c_smem_warp_tt.tile[data_paths, stageN](0, 0),
            )

            comptime if is_lower_frag_required:
                stsm_helper[swizzle, stageN, transpose_c=transpose_c](
                    lower_frag_casted,
                    c_smem_warp_tt.tile[data_paths, stageN](1, 0),
                )

            var c_smem_warp_tile_upper = c_smem_warp_tt.tile[
                data_paths, stageN
            ](0, 0)
            var c_smem_warp_tile_lower = c_smem_warp_tt.tile[
                data_paths, stageN
            ](1, 0)

            # Guard the write to shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()

            comptime if elementwise_compute_lambda_fn:
                comptime if not register_based_epilogue:
                    shared_memory_epilogue[
                        MMA_M,
                        data_paths,
                        num_stages,
                        Int(stage),
                        stageN,
                        c_smem_warp_tile_upper.dtype,
                        c_smem_tile.shape[1](),
                        simd_size,
                        swizzle,
                        elementwise_compute_lambda_fn.value(),
                        num_output_warps,
                    ](
                        M,
                        N,
                        Int(c_col),
                        Int(c_row),
                        c_smem_warp_tile_upper,
                        c_smem_warp_tile_lower,
                    )

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

        var coord_n_mma_m256 = c_col + UInt32(stage * stageN)
        var coord_n_mma_m128 = (
            c_col + UInt32(stage * stageN) + UInt32(BN * ufloordiv(warp_id, 2))
        )

        var cg2_coord_n = coord_n_mma_m256 if MMA_M == 256 else coord_n_mma_m128
        var cg1_coord_n = coord_n_mma_m256
        var coord_n = cg2_coord_n if cta_group == 2 else cg1_coord_n
        var coord_m = c_row

        comptime if transpose_c:
            var n_inbound = Int32(group_end_idx) - Int32(coord_n)
            if n_inbound >= Int32(stageN):
                # Aligned TMA store path for transpose_c with SWIZZLE_32B.
                # Each of the stage_contiguous_size // swizzle_width tiles
                # covers swizzle_width expert-N values across stageN tokens.
                if elect_one_warp and lane == 0:
                    fence_async_view_proxy()

                    comptime for i in range(
                        stage_contiguous_size // swizzle_width
                    ):
                        var c_smem_warp_tile = c_smem_tile.tile[
                            stageN * swizzle_width // stage_contiguous_size,
                            stage_contiguous_size,
                        ](i, 0).reshape[
                            Layout.row_major(stageN, swizzle_width)
                        ]()
                        c_tma_op.async_store(
                            c_smem_warp_tile,
                            (
                                Int(coord_m + UInt32(i * swizzle_width)),
                                Int(coord_n),
                            ),
                        )
                    c_tma_op.commit_group()

                comptime if stage < num_stages - 1:
                    c_tma_op.wait_group[1]()
                else:
                    c_tma_op.wait_group[0]()
            else:
                # Unaligned CUDA core fallback for transpose_c.
                comptime chunkM = swizzle_width
                comptime vec_chunkM = chunkM // simd_size
                comptime chunk_num = stage_contiguous_size // chunkM
                comptime logical_size = (chunk_num * stageN * vec_chunkM)
                comptime output_threads = num_output_warps * WARP_SIZE
                comptime assert (
                    logical_size % output_threads == 0
                ), "logical_size must be divisible by output_threads"
                comptime value_shape = logical_size // output_threads
                comptime cN = c_tensor_layout.shape[1].value()
                comptime smem_alignment = align_of[SIMD[c_type, simd_size]]()

                comptime for v in range(value_shape):
                    comptime thread_offset = v * output_threads
                    var tidx = UInt32(thread_idx.x) + UInt32(thread_offset)
                    var vec_chunkM_idx = tidx % UInt32(vec_chunkM)
                    var rest = tidx // UInt32(vec_chunkM)
                    var n_idx = rest % UInt32(stageN)
                    if Int32(n_idx) >= min(n_inbound, Int32(stageN)):
                        continue
                    var src_idx = UInt32(simd_size) * tidx
                    var c_smem_idx = swizzle(src_idx)
                    var val_vec = (c_smem_tile.ptr + c_smem_idx).load[
                        width=simd_size,
                        alignment=smem_alignment,
                    ]()
                    var chunk_idx = rest // UInt32(stageN)
                    # coord_n = token index, coord_m = expert-N index
                    var global_n = coord_n + n_idx
                    var global_m = coord_m + (
                        chunk_idx * UInt32(vec_chunkM) + vec_chunkM_idx
                    ) * UInt32(simd_size)
                    if global_m < UInt32(cN):
                        (c.ptr + global_n * UInt32(cN) + global_m).store[
                            alignment=smem_alignment
                        ](val_vec)

                # Advance TMA group counter so wait_group properly
                # drains previous stage's in-flight TMA store.
                if elect_one_warp and lane == 0:
                    c_tma_op.commit_group()

                comptime if stage < num_stages - 1:
                    c_tma_op.wait_group[1]()
                else:
                    c_tma_op.wait_group[0]()
        else:
            if (
                size_of[c_type]() != 2
                or coord_m + UInt32(TMA_BM) >= group_end_idx
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
                            IntTuple(UNKNOWN_VALUE, j),
                            element_type=.uint32,
                        ](Int(thread_idx.x), j)
                        var linear_idx = zipped_rt(input_crd) * UInt32(
                            simd_size
                        )
                        var linear_tup = RuntimeTuple[
                            IntTuple(UNKNOWN_VALUE), element_type=.uint32
                        ](Int(linear_idx))
                        var cmem_crd = idx2crd(
                            linear_tup, split_rt.shape, split_rt.stride
                        )
                        var local_i = cmem_crd[0].get_int()
                        var local_j = cmem_crd[1].get_int()
                        var global_i = coord_m + local_i
                        var global_j = coord_n + local_j
                        if (
                            global_i < group_end_idx
                            and global_j + UInt32(simd_size) <= N
                        ):
                            # src_ptr = c_smem_split.ptr + swizzle(linear_idx)
                            var src_ptr = c_smem_split.ptr + (
                                linear_idx if size_of[c_type]()
                                != 2 else swizzle(linear_idx)
                            )
                            var src = src_ptr.load[
                                width=simd_size, alignment=alignment
                            ]()
                            var dst_crd = RuntimeTuple[
                                IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)
                            ](Int(global_i), Int(global_j))
                            var dst_ptr = c.ptr + c.runtime_layout(dst_crd)
                            dst_ptr.store[width=simd_size, alignment=alignment](
                                src
                            )

                # Advance TMA group counter so wait_group properly
                # drains previous stage's in-flight TMA store.
                if elect_one_warp and lane == 0:
                    c_tma_op.commit_group()

                comptime if stage < num_stages - 1:
                    c_tma_op.wait_group[1]()
                else:
                    c_tma_op.wait_group[0]()
            else:
                var cg2_c_smem_coord_m = 0 if MMA_M == 256 else ufloordiv(
                    warp_id, 2
                )
                var cg1_c_smem_coord_m = 0
                var c_smem_coord_m = (
                    cg2_c_smem_coord_m if cta_group == 2 else cg1_c_smem_coord_m
                )
                var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](
                    c_smem_coord_m, 0
                )

                if elect_one_warp and lane == 0:
                    fence_async_view_proxy()
                    c_tma_op.async_store(
                        c_smem_split,
                        (
                            Int(coord_n),
                            Int(coord_m),
                        ),
                    )
                    c_tma_op.commit_group()

                # Keep one tma store in fly
                comptime if stage < num_stages - 1:
                    c_tma_op.wait_group[1]()
                # Last stage guard all tma store to finish
                else:
                    c_tma_op.wait_group[0]()

        comptime if stage > 0 or stage == num_stages - 1:
            # Guard the tma read from shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()


@always_inline
def multi_stage_store_C[
    c_type: DType,
    c_tile_rank: Int,
    c_tile_shape: IndexList[c_tile_rank],
    c_desc_shape: IndexList[c_tile_rank],
    num_accum_pipeline_stages: Int,
    c_tensor_layout: Layout,
    /,
    *,
    c_smem_layout: Layout,
    input_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    stage_stride_cols: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: Int = 4,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,  # if false it will perform epilogue on data in shared memory
    transpose_c: Bool = False,
    scale_c_coord: Bool = True,
](
    c_smem_base: UnsafePointer[
        mut=True, Scalar[c_type], _, address_space=.SHARED
    ],
    c_tma_op: TMATensorTile[c_type, c_tile_rank, c_tile_shape, c_desc_shape],
    c: LayoutTensor[c_type, c_tensor_layout, MutAnyOrigin],
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_pipeline_stages],
    tmem_addr: UInt32,
    work_tile_coord: Tuple[UInt32, UInt32],
    elect_one_warp: Bool,
    expert_scale: Float32,
    M: UInt32,
    N: UInt32,
    group_end_idx: UInt32,
):
    """Drains one MMA output tile from TMEM into global memory in stages.

    Waits for the MMA producer to signal completion, computes the TMEM column
    offset for the current accumulator stage, and delegates to
    `copy_accum_to_gmem` to perform the multi-stage store of the output tile.

    Parameters:
        c_type: The data type of the output tensor C.
        c_tile_rank: The rank of the C TMA tile.
        c_tile_shape: The shape of the C TMA tile.
        c_desc_shape: The descriptor shape of the C TMA tile.
        num_accum_pipeline_stages: The number of accumulator pipeline stages.
        c_tensor_layout: The layout of the output LayoutTensor.
        c_smem_layout: The layout of the C shared memory tile.
        input_type: The data type of the input tensor A (used to select the
            epilogue dtype).
        accum_type: The accumulator data type held in TMEM.
        block_tile_shape: The (BM, BN, BK) block tile shape.
        mma_shape: The (MMA_M, MMA_N, MMA_K) MMA shape.
        stage_stride_cols: The TMEM column stride between accumulator stages.
        c_swizzle: The TMA swizzle mode for C.
        cta_group: The number of CTAs cooperating per output tile (1 or 2).
        num_output_warps: The number of epilogue warps.
        elementwise_compute_lambda_fn: Optional fused elementwise epilogue.
        register_based_epilogue: Whether the epilogue runs in registers.
        transpose_c: Whether the output is stored transposed.
        scale_c_coord: Whether to scale tile coordinates by block tile strides.

    Args:
        c_smem_base: Base pointer to the C shared memory buffer.
        c_tma_op: The TMA tensor tile descriptor for C.
        c: The output LayoutTensor in global memory.
        mma_output_pipeline: The producer/consumer pipeline for MMA output.
        tmem_addr: The base TMEM address for this CTA's accumulator.
        work_tile_coord: The (row, column) coordinate of the output tile.
        elect_one_warp: Whether the calling warp is the elected leader warp.
        expert_scale: The per-expert scale factor applied to the accumulator.
        M: The total number of rows in the output tensor.
        N: The total number of columns in the output tensor.
        group_end_idx: The exclusive end row index for the current expert group.
    """
    # WAIT FOR MMA TO FINISH AND STORE RESULT
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    # assume N dimension is static
    comptime simd_size = simd_width_of[c_type]()

    # TODO (GEX-2630): This is a temporary workaround to support float32 compute epilogue for FP8 models for which we use compute lambda for dequantization.
    # We should remove this once GEX-2630 is fixed.
    comptime epilogue_dtype = c_type if input_type == DType.bfloat16 else DType.float32

    # we break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # stage N is 32
    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()
    comptime stage_contiguous_size = c_smem_layout.shape[1].value()
    # so num stages is usually 256 by 32 is 8
    # MMA Size will be larger than output tile shape. E.G. MMA_MxMMA_N = (128, 256); OUT_MxOUT_N = (128, 32)

    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages

    comptime data_paths = 16  # same as lanes
    comptime bits = 256
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    comptime rep = stageN // (bits // 32)  # repetitions per stage
    # typically repeated 4 times to get the desired 32 elements

    # stageN is how many elements we want to load at once

    # stmatrix related
    comptime st_matrix_swizzle = c_swizzle
    comptime swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    # before i start the process of transferring over num_stages * stageN= MMA_N from tensor memory to global, i should wait
    # on the accum_full_mbar barrier
    var mma_output_stage = mma_output_pipeline.consumer_stage()
    mma_output_pipeline.wait_producer()

    # this is the column offset for all the stages of THIS load, where one load takes (num_stages iterations)
    var tmem_offset = mma_output_stage * UInt32(stage_stride_cols) + tmem_addr

    copy_accum_to_gmem[
        c_smem_layout=c_smem_layout,
        repeat=rep,
        accum_type=accum_type,
        cta_group=cta_group,
        epilogue_dtype=epilogue_dtype,
        block_tile_shape=block_tile_shape,
        mma_shape=mma_shape,
        num_output_warps=num_output_warps,
        c_swizzle=c_swizzle,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        transpose_c=transpose_c,
        scale_c_coord=scale_c_coord,
    ](
        c_smem_base.as_unsafe_any_origin(),
        c_tma_op,
        c,
        mma_output_pipeline,
        mma_output_stage,
        tmem_offset,
        work_tile_coord,
        (M, N),
        expert_scale,
        group_end_idx,
    )


struct B200BlockScaledMatmulSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]:
    """Holds the shared memory buffers and barriers for one block-scaled matmul tile.

    Allocates double-buffered shared memory for A, B, C, and their scale factors,
    plus the TMA-MMA and accumulator pipeline barriers and the TMEM deallocation
    barrier used by the warp-specialized grouped matmul kernel on SM100.

    Parameters:
        a_type: The data type of input tensor A.
        b_type: The data type of input tensor B.
        c_type: The data type of the output tensor C.
        sfa_dtype: The data type of the A scale factors.
        sfb_dtype: The data type of the B scale factors.
        transpose_b: Whether B is stored transposed (K-major).
        config: The block-scaled matmul configuration.
    """

    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime AType = Scalar[Self.a_type]
    comptime BType = Scalar[Self.b_type]
    comptime CType = Scalar[Self.c_type]
    comptime AScalesType = Scalar[Self.sfa_dtype]
    comptime BScalesType = Scalar[Self.sfb_dtype]

    comptime a_smem_size = Self.BM * Self.BK * Self.config.num_pipeline_stages
    comptime b_smem_size = Self.BN * Self.BK * Self.config.num_pipeline_stages
    comptime c_smem_size = Self.OutputM * Self.OutputN * Self.config.num_output_stages

    comptime sfa_smem_size = (
        Self.config.num_sf_k_tiles
        * (Self.BM // SF_MN_GROUP_SIZE)
        * Self.config.sf_block_atom_size
        * Self.config.num_pipeline_stages
    )
    comptime sfb_smem_size = (
        Self.config.num_sf_k_tiles
        * (Self.MMA_N // SF_MN_GROUP_SIZE)
        * Self.config.sf_block_atom_size
        * Self.config.num_pipeline_stages
    )

    comptime num_group_pipeline_stages = Self.config.num_pipeline_stages // Self.config.k_group_size

    # AB pipelines
    var a_smem: Array[Self.AType, Self.a_smem_size]
    var b_smem: Array[Self.BType, Self.b_smem_size]
    var c_smem: Array[Self.CType, Self.c_smem_size]
    var sfa_smem: Array[Self.AScalesType, Self.sfa_smem_size]
    var sfb_smem: Array[Self.BScalesType, Self.sfb_smem_size]

    var tma_mma_mbars: Array[
        SharedMemBarrier, Self.num_group_pipeline_stages * 2
    ]
    # ACCUM
    var accum_mbars: Array[
        SharedMemBarrier, Self.config.num_accum_pipeline_stages * 2
    ]

    # TMEM
    var tmem_dealloc_mbar: Array[SharedMemBarrier, 1]
    var tmem_addr: Array[UInt32, 1]


@always_inline
def load_AB[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    sfa_tile_rank: Int,
    sfa_tile_shape: IndexList[sfa_tile_rank],
    sfa_desc_shape: IndexList[sfa_tile_rank],
    sfb_tile_rank: Int,
    sfb_tile_shape: IndexList[sfb_tile_rank],
    sfb_desc_shape: IndexList[sfb_tile_rank],
    num_pipeline_stages: Int,
    group_scale_offsets_layout: Layout,
    transpose_b: Bool,
    /,
    *,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    sfa_smem_layout: Layout,
    sfb_smem_layout: Layout,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    num_sf_k_tiles: Int,
    cta_group: Int = 1,
    k_group_size: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    sfa_tma_op: TMATensorTile[
        sfa_dtype, sfa_tile_rank, sfa_tile_shape, sfa_desc_shape
    ],
    sfb_tma_op: TMATensorTile[
        sfb_dtype, sfb_tile_rank, sfb_tile_shape, sfb_desc_shape
    ],
    a_smem_base: UnsafePointer[
        mut=True, Scalar[a_type], _, address_space=.SHARED
    ],
    b_smem_base: UnsafePointer[
        mut=True, Scalar[b_type], _, address_space=.SHARED
    ],
    sfa_smem_base: UnsafePointer[
        mut=True, Scalar[sfa_dtype], _, address_space=.SHARED
    ],
    sfb_smem_base: UnsafePointer[
        mut=True, Scalar[sfb_dtype], _, address_space=.SHARED
    ],
    load_mma_pipeline: ProducerConsumerPipeline[num_pipeline_stages],
    peer_cta_coord: Tuple[Int, Int, Int],
    work_tile_coord: Tuple[Int, Int],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: UInt32,
    elect_one_cta: Bool,
    scheduler: TileScheduler,
    expert_id: Int32,
    group_scale_offsets: LayoutTensor[
        .uint32, group_scale_offsets_layout, MutAnyOrigin
    ],
):
    """Issues multicast TMA loads for A, B, and their scale factors into shared memory.

    Waits for the consumer (MMA) to release the current pipeline stage buffer,
    then launches asynchronous multicast TMA copies of the A and B tiles along
    with their corresponding block scale factors into shared memory for one
    pipeline stage, accounting for expert offsets and group scale offsets.

    Parameters:
        a_type: The data type of input tensor A.
        b_type: The data type of input tensor B.
        c_type: The data type of the output tensor C.
        sfa_dtype: The data type of the A scale factors.
        sfb_dtype: The data type of the B scale factors.
        a_tile_rank: The rank of the A TMA tile.
        a_tile_shape: The shape of the A TMA tile.
        a_desc_shape: The descriptor shape of the A TMA tile.
        b_tile_rank: The rank of the B TMA tile.
        b_tile_shape: The shape of the B TMA tile.
        b_desc_shape: The descriptor shape of the B TMA tile.
        sfa_tile_rank: The rank of the A scale factor TMA tile.
        sfa_tile_shape: The shape of the A scale factor TMA tile.
        sfa_desc_shape: The descriptor shape of the A scale factor TMA tile.
        sfb_tile_rank: The rank of the B scale factor TMA tile.
        sfb_tile_shape: The shape of the B scale factor TMA tile.
        sfb_desc_shape: The descriptor shape of the B scale factor TMA tile.
        num_pipeline_stages: The number of load/MMA pipeline stages.
        group_scale_offsets_layout: The layout of the group scale offsets tensor.
        transpose_b: Whether B is stored transposed (K-major).
        a_smem_layout: The shared memory layout for A.
        b_smem_layout: The shared memory layout for B.
        sfa_smem_layout: The shared memory layout for A scale factors.
        sfb_smem_layout: The shared memory layout for B scale factors.
        config: The block-scaled matmul configuration.
        block_tile_shape: The (BM, BN, BK) block tile shape.
        mma_shape: The (MMA_M, MMA_N, MMA_K) MMA shape.
        num_sf_k_tiles: The number of scale factor tiles along K.
        cta_group: The number of CTAs cooperating per output tile (1 or 2).
        k_group_size: The number of K iterations loaded per pipeline stage.

    Args:
        a_tma_op: The TMA tensor tile descriptor for A.
        b_tma_op: The TMA tensor tile descriptor for B.
        sfa_tma_op: The TMA tensor tile descriptor for A scale factors.
        sfb_tma_op: The TMA tensor tile descriptor for B scale factors.
        a_smem_base: Base pointer to the A shared memory buffer.
        b_smem_base: Base pointer to the B shared memory buffer.
        sfa_smem_base: Base pointer to the A scale factor shared memory buffer.
        sfb_smem_base: Base pointer to the B scale factor shared memory buffer.
        load_mma_pipeline: The producer/consumer pipeline for TMA loads.
        peer_cta_coord: The (peer_id, m_coord, n_coord) of the peer CTA.
        work_tile_coord: The (m, n) coordinate of the work tile.
        a_multicast_mask: The multicast mask for A TMA loads.
        b_multicast_mask: The multicast mask for B TMA loads.
        iter_idx: The K iteration index for this load.
        elect_one_cta: Whether this CTA is the elected leader CTA.
        scheduler: The tile scheduler tracking group and expert state.
        expert_id: The expert ID for the current work tile.
        group_scale_offsets: The per-group scale factor offsets tensor.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    comptime sfa_expected_bytes = sfa_smem_layout.size() * size_of[sfa_dtype]()
    comptime sfb_expected_bytes = sfb_smem_layout.size() * size_of[sfb_dtype]()

    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = (
        cta_group
        * (
            a_expected_bytes
            + b_expected_bytes
            + sfa_expected_bytes
            + sfb_expected_bytes
        )
    ) * k_group_size

    comptime a_tma_load_size = _idx_product[a_tile_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    var stage = load_mma_pipeline.producer_stage()
    var tma_mbar = load_mma_pipeline.producer_mbar(stage)
    var expert_offset: Int = Int(expert_id) * scheduler.static_MN
    var a_gmem_slice_coord = (
        peer_cta_coord[2] * a_tma_rows + work_tile_coord[0]
    ) + (expert_offset if config.AB_swapped else 0)
    var b_gmem_slice_coord: Int = (
        peer_cta_coord[1] * b_tma_rows
        + peer_cta_coord[0] * BN
        + work_tile_coord[1]
    ) + (expert_offset if not config.AB_swapped else 0)

    # Wait until MMA (consumer) has used the buffer.
    load_mma_pipeline.wait_consumer()

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

        comptime a_smem_tile_size = a_smem_layout.size()
        comptime b_smem_tile_size = b_smem_layout.size()
        comptime sfa_smem_tile_size = sfa_smem_layout.size()
        comptime sfb_smem_tile_size = sfb_smem_layout.size()

        for j in range(UInt32(k_group_size)):
            var offset = Int(stage * UInt32(k_group_size) + j)
            var a_smem_tile = LayoutTensor[
                a_type,
                a_smem_layout,
                address_space=.SHARED,
                alignment=128,
            ](a_smem_base + offset * a_smem_tile_size)
            var b_smem_tile = LayoutTensor[
                b_type,
                b_smem_layout,
                address_space=.SHARED,
                alignment=128,
            ](b_smem_base + offset * b_smem_tile_size)
            var sfa_smem_tile = LayoutTensor[
                sfa_dtype,
                sfa_smem_layout,
                address_space=.SHARED,
                alignment=128,
            ](sfa_smem_base + offset * sfa_smem_tile_size)
            var sfb_smem_tile = LayoutTensor[
                sfb_dtype,
                sfb_smem_layout,
                address_space=.SHARED,
                alignment=128,
            ](sfb_smem_base + offset * sfb_smem_tile_size)

            var a_smem_slice = type_of(a_smem_tile)(
                a_smem_tile.ptr + peer_cta_coord[2] * a_tma_load_size
            )
            var b_smem_slice = type_of(b_smem_tile)(
                b_smem_tile.ptr + peer_cta_coord[1] * b_tma_load_size
            )

            a_tma_op.async_multicast_load[cta_group](
                a_smem_slice,
                tma_mbar[0],
                (Int(iter_idx + j) * BK, a_gmem_slice_coord),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load[cta_group](
                b_smem_slice,
                tma_mbar[0],
                (Int(iter_idx + j) * BK, b_gmem_slice_coord),
                b_multicast_mask,
            )

            var group_scale_offset_vec = group_scale_offsets[
                Int(scheduler.current_group_idx)
            ]
            comptime assert group_scale_offset_vec.length == 1
            var group_scale_offset = group_scale_offset_vec[0]
            var a_m: Int
            var b_n: Int
            var sf_groups_per_expert = ceildiv(
                scheduler.static_MN, SF_MN_GROUP_SIZE
            )
            if config.AB_swapped:
                a_m = Int(expert_id) * sf_groups_per_expert + ufloordiv(
                    work_tile_coord[0], SF_MN_GROUP_SIZE
                )
                b_n = ufloordiv(work_tile_coord[1], SF_MN_GROUP_SIZE) + Int(
                    group_scale_offset
                )
            else:
                a_m = ufloordiv(work_tile_coord[0], SF_MN_GROUP_SIZE) + Int(
                    group_scale_offset
                )
                b_n = Int(expert_id) * sf_groups_per_expert + ufloordiv(
                    work_tile_coord[1], SF_MN_GROUP_SIZE
                )

            sfa_tma_op.async_copy_4d[cta_group](
                sfa_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    Int(iter_idx + j) * num_sf_k_tiles,
                    a_m,
                ),
            )

            sfb_tma_op.async_copy_4d[cta_group](
                sfb_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    Int(iter_idx + j) * num_sf_k_tiles,
                    b_n,
                ),
            )


@always_inline
def consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    scaling_kind: UMMAKind,
    /,
    *,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    sfa_smem_layout: Layout,
    sfb_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    SFA_NUM_COLS: Int,
    SFB_NUM_COLS: Int,
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    k_group_size: Int = 1,
](
    tmem_addr: UInt32,
    sfa_tmem: UInt32,
    sfb_tmem: UInt32,
    a_smem_base: UnsafePointer[
        mut=True, Scalar[a_type], _, address_space=.SHARED
    ],
    b_smem_base: UnsafePointer[
        mut=True, Scalar[b_type], _, address_space=.SHARED
    ],
    sfa_smem_base: UnsafePointer[
        mut=True, Scalar[sfa_dtype], _, address_space=.SHARED
    ],
    sfb_smem_base: UnsafePointer[
        mut=True, Scalar[sfb_dtype], _, address_space=.SHARED
    ],
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    mma_op: MmaOpSM100_BlockScaled_SS[
        c_type,
        a_type,
        b_type,
        sfa_dtype,
        sfb_dtype,
        scaling_kind,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ],
    elect_one_warp: Bool,
    iter_idx: UInt32,
    k_start: UInt32,
    work_tile_coord: Tuple[Int, Int],
):
    """Consumes one pipeline stage and issues the block-scaled MMA into TMEM.

    Waits for the producer (TMA load) to fill the current stage, then reads A, B,
    and their scale factors from shared memory and launches the block-scaled
    `MmaOpSM100_BlockScaled_SS` into tensor memory, accumulating across K-group
    iterations and committing the result to the accumulator pipeline barrier.

    Parameters:
        accum_type: The accumulator data type held in TMEM.
        c_type: The data type of the output tensor C.
        a_type: The data type of input tensor A.
        b_type: The data type of input tensor B.
        sfa_dtype: The data type of the A scale factors.
        sfb_dtype: The data type of the B scale factors.
        a_swizzle: The TMA swizzle mode for A.
        b_swizzle: The TMA swizzle mode for B.
        transpose_b: Whether B is stored transposed (K-major).
        pipeline_stages: The number of load/MMA pipeline stages.
        scaling_kind: The UMMA scaling kind for block-scaled MMA.
        a_smem_layout: The shared memory layout for A.
        b_smem_layout: The shared memory layout for B.
        sfa_smem_layout: The shared memory layout for A scale factors.
        sfb_smem_layout: The shared memory layout for B scale factors.
        block_tile_shape: The (BM, BN, BK) block tile shape.
        mma_shape: The (MMA_M, MMA_N, MMA_K) MMA shape.
        SFA_NUM_COLS: The number of TMEM columns for A scale factors.
        SFB_NUM_COLS: The number of TMEM columns for B scale factors.
        cta_group: The number of CTAs cooperating per output tile (1 or 2).
        cluster_shape: The cluster shape for multicast MMA.
        k_group_size: The number of K iterations per pipeline stage.

    Args:
        tmem_addr: The base TMEM address for the accumulator.
        sfa_tmem: The base TMEM address for A scale factors.
        sfb_tmem: The base TMEM address for B scale factors.
        a_smem_base: Base pointer to the A shared memory buffer.
        b_smem_base: Base pointer to the B shared memory buffer.
        sfa_smem_base: Base pointer to the A scale factor shared memory buffer.
        sfb_smem_base: Base pointer to the B scale factor shared memory buffer.
        load_mma_pipeline: The producer/consumer pipeline for TMA loads.
        mma_op: The block-scaled SM100 MMA operation object.
        elect_one_warp: Whether the calling warp is the elected leader warp.
        iter_idx: The K iteration index for this MMA.
        k_start: The K iteration at which to initialize the accumulator.
        work_tile_coord: The (m, n) coordinate of the work tile.
    """
    comptime BM = block_tile_shape[0]
    comptime MMA_N = mma_shape[1]

    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    # Compose TMEM address: accum stage encoded in column field with stride in columns.
    comptime a_smem_tile_size = a_smem_layout.size()
    comptime b_smem_tile_size = b_smem_layout.size()
    comptime sfa_smem_tile_size = sfa_smem_layout.size()
    comptime sfb_smem_tile_size = sfb_smem_layout.size()

    comptime sfa_d0 = sf_tile_dim0[BM]
    comptime sfa_d1 = sfa_smem_tile_size // sfa_d0
    comptime sfb_d0 = sf_tile_dim0[MMA_N]
    comptime sfb_d1 = sfb_smem_tile_size // sfb_d0

    if elect_one_sync():
        for j in range(UInt32(k_group_size)):
            var offset = Int(stage * UInt32(k_group_size) + j)
            var a_smem_tile = TileTensor(
                a_smem_base + offset * a_smem_tile_size,
                LTToTTLayout[a_smem_layout](),
            )
            var b_smem_tile = TileTensor(
                b_smem_base + offset * b_smem_tile_size,
                LTToTTLayout[b_smem_layout](),
            )
            var sfa_smem_tile = SMemTile[
                sfa_dtype, internal_sf_k_major[sfa_d0, sfa_d1]
            ](
                sfa_smem_base.as_unsafe_any_origin()
                + offset * sfa_smem_tile_size,
                internal_sf_k_major[sfa_d0, sfa_d1],
            )
            var sfb_smem_tile = SMemTile[
                sfb_dtype, internal_sf_k_major[sfb_d0, sfb_d1]
            ](
                sfb_smem_base.as_unsafe_any_origin()
                + offset * sfb_smem_tile_size,
                internal_sf_k_major[sfb_d0, sfb_d1],
            )

            var sfa_tmem_offset = sfa_tmem + (
                stage * UInt32(k_group_size) + j
            ) * UInt32(SFA_NUM_COLS)
            var sfb_tmem_offset = sfb_tmem + (
                stage * UInt32(k_group_size) + j
            ) * UInt32(SFB_NUM_COLS)

            # When MMA_N doesn't fill a full SF group (128), adjacent
            # N-tiles share one group in TMEM. Odd tiles offset by 2
            # columns to read their half.
            var sfb_tmem_adj: UInt32
            comptime if MMA_N in (64, 192):
                sfb_tmem_adj = UInt32(umod(work_tile_coord[1], 2)) * 2
            else:
                sfb_tmem_adj = UInt32(0)

            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                sfa_smem_tile,
                sfb_smem_tile,
                tmem_addr,
                sfa_tmem_offset,
                sfb_tmem_offset,
                init_c=(
                    (iter_idx + j) == k_start
                ),  # Initialize C on first iteration
                sfb_tmem_adj=sfb_tmem_adj,
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))


def blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[_, _, _, _, _, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel.ON,
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_device: TileTensor[mut=True, config.c_type, address_space=.GENERIC, ...],
    a_device: TileTensor[mut=False, config.a_type, address_space=.GENERIC, ...],
    group_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    group_scale_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    b_device: TileTensor[mut=False, config.b_type, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    a_scales: TileTensor[
        mut=False, config.sfa_dtype, address_space=.GENERIC, ...
    ],
    b_scales: TileTensor[
        mut=False, config.sfb_dtype, address_space=.GENERIC, ...
    ],
    expert_scales: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launch grouped block-scaled matmul kernel on SM100.

    When config.AB_swapped is True, internally swaps A and B operands
    (along with their scale factors) and transposes the output for better
    performance when M is small.

    Accepts TileTensors and converts to LayoutTensors internally.

    Parameters:
        transpose_b: Whether B is stored transposed (K-major).
        config: The block-scaled matmul configuration.
        elementwise_compute_lambda_fn: Optional fused elementwise epilogue.
        pdl_level: The program-dependent launch level for the kernel grid.
        max_profiled_tiles_per_SM: The maximum number of tiles profiled per SM,
            or None to disable profiling.

    Args:
        c_device: The output C tensor in device memory.
        a_device: The input A tensor in device memory.
        group_offsets: The starting token index for each expert group.
        group_scale_offsets: The starting scale index for each expert group.
        b_device: The input B tensor in device memory.
        expert_ids: The expert ID for each group.
        a_scales: The A scale factors tensor in device memory.
        b_scales: The B scale factors tensor in device memory.
        expert_scales: The per-expert scaling factors applied in the epilogue.
        num_active_experts: The number of active experts in this batch.
        ctx: The device context used to launch the kernel.
    """
    comptime c_type = config.c_type
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime sfa_dtype = config.sfa_dtype
    comptime sfb_dtype = config.sfb_dtype

    # Convert TileTensors to LayoutTensors at the boundary.
    # Mutable tensors use to_layout_tensor() directly.
    var c_lt = c_device.to_layout_tensor()
    var a_lt = a_device.to_layout_tensor()
    var b_lt = b_device.to_layout_tensor()

    # The kernel expects all LayoutTensor args with MutAnyOrigin, so rebind
    # immutable TileTensor-derived pointers to MutAnyOrigin.
    comptime _group_offsets_lt_type = type_of(group_offsets.to_layout_tensor())
    var group_offsets_lt = LayoutTensor[
        .uint32, _group_offsets_lt_type.layout, MutAnyOrigin
    ](
        rebind[UnsafePointer[UInt32, MutAnyOrigin]](group_offsets.ptr),
        group_offsets.to_layout_tensor().runtime_layout,
    )
    comptime _group_scale_offsets_lt_type = type_of(
        group_scale_offsets.to_layout_tensor()
    )
    var group_scale_offsets_lt = LayoutTensor[
        .uint32, _group_scale_offsets_lt_type.layout, MutAnyOrigin
    ](
        rebind[UnsafePointer[UInt32, MutAnyOrigin]](group_scale_offsets.ptr),
        group_scale_offsets.to_layout_tensor().runtime_layout,
    )
    comptime _expert_ids_lt_type = type_of(expert_ids.to_layout_tensor())
    var expert_ids_lt = LayoutTensor[
        .int32, _expert_ids_lt_type.layout, MutAnyOrigin
    ](
        rebind[UnsafePointer[Int32, MutAnyOrigin]](expert_ids.ptr),
        expert_ids.to_layout_tensor().runtime_layout,
    )

    # Scales and expert_scales also need MutAnyOrigin rebinding.
    comptime _a_scales_lt_type = type_of(a_scales.to_layout_tensor())
    var a_scales_lt = LayoutTensor[
        sfa_dtype, _a_scales_lt_type.layout, MutAnyOrigin
    ](
        rebind[UnsafePointer[Scalar[sfa_dtype], MutAnyOrigin]](a_scales.ptr),
        a_scales.to_layout_tensor().runtime_layout,
    )
    comptime _b_scales_lt_type = type_of(b_scales.to_layout_tensor())
    var b_scales_lt = LayoutTensor[
        sfb_dtype, _b_scales_lt_type.layout, MutAnyOrigin
    ](
        rebind[UnsafePointer[Scalar[sfb_dtype], MutAnyOrigin]](b_scales.ptr),
        b_scales.to_layout_tensor().runtime_layout,
    )
    comptime _expert_scales_lt_type = type_of(expert_scales.to_layout_tensor())
    var expert_scales_lt = LayoutTensor[
        .float32, _expert_scales_lt_type.layout, MutAnyOrigin
    ](
        rebind[UnsafePointer[Float32, MutAnyOrigin]](expert_scales.ptr),
        expert_scales.to_layout_tensor().runtime_layout,
    )

    comptime c_layout = type_of(c_lt).layout
    comptime a_layout = type_of(a_lt).layout
    comptime b_layout = type_of(b_lt).layout
    comptime group_offsets_layout = type_of(group_offsets_lt).layout
    comptime group_scale_offsets_layout = type_of(group_scale_offsets_lt).layout
    comptime expert_ids_layout = type_of(expert_ids_lt).layout
    comptime sfa_layout = type_of(a_scales_lt).layout
    comptime sfb_layout = type_of(b_scales_lt).layout
    comptime expert_scale_layout = type_of(expert_scales_lt).layout

    # Reshape B weights (3D -> 2D: merge expert dim) and B scales (6D -> 5D)
    # so the private function always receives uniform 2D data / 5D scales.
    comptime num_experts = b_layout.shape[0].value()
    comptime flat_b_layout = Layout.row_major(
        num_experts * c_layout.shape[1].value(),
        a_layout.shape[1].value(),
    )
    var flat_b_device = LayoutTensor[
        b_type,
        flat_b_layout,
        address_space=.GENERIC,
    ](b_device.ptr.as_unsafe_any_origin())

    comptime assert (
        sfb_layout.shape[0].value() == num_experts
    ), "num_experts must be equal to _sfb_layout.shape[0]"
    comptime flat_sfb_layout = Layout.row_major(
        sfb_layout.shape[0].value() * sfb_layout.shape[1].value(),
        sfb_layout.shape[2].value(),
        sfb_layout.shape[3].value(),
        sfb_layout.shape[4].value(),
        sfb_layout.shape[5].value(),
    )
    var flat_b_scales = LayoutTensor[sfb_dtype, flat_sfb_layout, MutAnyOrigin](
        rebind[UnsafePointer[Scalar[sfb_dtype], MutAnyOrigin]](b_scales.ptr)
    )

    comptime if config.AB_swapped:
        # When both A and B are K-major, C = A @ B'.
        # If we swap A and B: D = B @ A', and D' = (B @ A')' = A @ B' = C.
        # So swapping + transposing the output gives the same result.
        # The transpose is handled by transpose_c = config.AB_swapped in the
        # kernel epilogue.
        comptime new_config = config.swap_AB_type()
        _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            c_type,
            c_layout,
            b_type,
            flat_b_layout,
            group_offsets_layout,
            group_scale_offsets_layout,
            a_type,
            a_layout,
            expert_ids_layout,
            sfb_dtype,
            flat_sfb_layout,
            sfa_dtype,
            sfa_layout,
            expert_scale_layout,
            transpose_b,
            config=new_config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            c_lt,
            flat_b_device,
            group_offsets_lt,
            group_scale_offsets_lt,
            a_lt,
            expert_ids_lt,
            flat_b_scales,
            a_scales_lt,
            expert_scales_lt,
            num_active_experts,
            ctx,
        )
    else:
        _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            c_type,
            c_layout,
            a_type,
            a_layout,
            group_offsets_layout,
            group_scale_offsets_layout,
            b_type,
            flat_b_layout,
            expert_ids_layout,
            sfa_dtype,
            sfa_layout,
            sfb_dtype,
            flat_sfb_layout,
            expert_scale_layout,
            transpose_b,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            c_lt,
            a_lt,
            group_offsets_lt,
            group_scale_offsets_lt,
            flat_b_device,
            expert_ids_lt,
            a_scales_lt,
            flat_b_scales,
            expert_scales_lt,
            num_active_experts,
            ctx,
        )


def _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    group_offsets_layout: Layout,
    group_scale_offsets_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    expert_ids_layout: Layout,
    sfa_dtype: DType,
    sfa_layout: Layout,
    sfb_dtype: DType,
    sfb_layout: Layout,
    expert_scale_layout: Layout,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel.ON,
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_device: LayoutTensor[c_type, c_layout, ...],
    a_device: LayoutTensor[mut=False, a_type, a_layout, ...],
    group_offsets: LayoutTensor[.uint32, group_offsets_layout, ...],
    group_scale_offsets: LayoutTensor[.uint32, group_scale_offsets_layout, ...],
    b_device: LayoutTensor[mut=False, b_type, b_layout, ...],
    expert_ids: LayoutTensor[.int32, expert_ids_layout, ...],
    a_scales: LayoutTensor[sfa_dtype, sfa_layout, MutAnyOrigin],
    b_scales: LayoutTensor[sfb_dtype, sfb_layout, MutAnyOrigin],
    expert_scales: LayoutTensor[.float32, expert_scale_layout, MutAnyOrigin],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    # The public function has already:
    #   - Reshaped B data from 3D (experts, N, K) to 2D (experts*N, K)
    #   - Reshaped B scales from 6D to 5D (merging expert dim)
    #   - Swapped A↔B when config.AB_swapped is True
    # So here a_layout and b_layout are both 2D, and sfa_layout and
    # sfb_layout are both 5D.
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        sfa_dtype == sfb_dtype
    ), "Only support same scales dtype for A and B"
    comptime assert sfa_dtype in (MXFP8_SF_DTYPE, NVFP4_SF_DTYPE), (
        "Only support MXFP8_SF_DTYPE (F8-UE8M0) or MXFP4_SF_DTYPE (F8-E4M3) for"
        " scales"
    )

    comptime register_based_epilogue = config.register_based_epilogue

    comptime assert a_scales.rank == 5, "a_scales must be 5D tensors"
    comptime assert b_scales.rank == 5, "b_scales must be 5D tensors"

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    comptime assert config.k_group_size == 1, "Only support k_group_size == 1"

    comptime assert config.num_split_k == 1, "Only support split_k == 1"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    comptime assert (
        sfa_layout.shape[2].value()
        == sfb_layout.shape[2].value()
        == SF_ATOM_M[0]
    ), ""
    comptime assert (
        sfa_layout.shape[3].value()
        == sfb_layout.shape[3].value()
        == SF_ATOM_M[1]
    ), ""
    comptime assert (
        sfa_layout.shape[4].value() == sfb_layout.shape[4].value() == SF_ATOM_K
    ), ""

    comptime if config.cta_group == 2:
        comptime assert MMA_M == 256 and MMA_N in (
            128,
            256,
        ), "Only support cta_group == 2 with MMA_M == 256"

    else:
        comptime assert MMA_M == 128 and MMA_N in (128, 256), (
            "Only support MMA_M == 128 and MMA_N in (128, 256) when"
            " cta_group == 1"
        )

    comptime cluster_shape = config.cluster_shape

    comptime N = c_layout.shape[1].value()
    comptime expert_n = N
    comptime K = a_layout.shape[1].value()
    comptime assert K % 16 == 0, (
        "Due to TMA limitations, K must be a multiple of 16 bytes"
        + " but got K = "
        + String(K)
    )

    var M = c_device.dim[0]()

    comptime assert (
        ceildiv(K, BK) % config.k_group_size == 0
    ), "K iterations must be a multiple of k_group_size"

    var a_tma_op = create_tensor_tile[
        Index(BM // cluster_shape[1], BK), swizzle_mode=config.a_swizzle
    ](ctx, a_device)

    var b_tma_op = create_tensor_tile[
        Index(
            BN // (cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode=config.b_swizzle,
    ](ctx, b_device)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    # Override c_swizzle to SWIZZLE_32B when AB_swapped (transpose_c).
    # With SWIZZLE_32B, TMA's 32-byte row stride matches the 16-element
    # chunks produced by st_matrix[transpose=True], giving a 1:1 mapping
    # between smem rows and gmem token rows.
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_32B if config.AB_swapped else config.c_swizzle
    comptime c_tma_tile_shape_1 = c_swizzle.bytes() // size_of[c_type]()
    var c_tma_op = create_tensor_tile[
        c_tma_tile_shape if not config.AB_swapped else Index(
            c_tma_tile_shape[0], c_tma_tile_shape_1
        ),
        swizzle_mode=c_swizzle,
    ](ctx, c_device)

    comptime scales_4d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    comptime sfa_4d_layout = scales_4d_layout[sfa_layout]
    comptime sfb_4d_layout = scales_4d_layout[sfb_layout]

    var sfa_4d = LayoutTensor[sfa_dtype, sfa_4d_layout, MutAnyOrigin](
        a_scales.ptr,
        RuntimeLayout[sfa_4d_layout].row_major(
            IndexList[4](
                a_scales.dim(0),
                a_scales.dim(1),
                a_scales.dim(2),
                a_scales.dim(3) * a_scales.dim(4),
            ),
        ),
    )
    var sfb_4d = LayoutTensor[sfb_dtype, sfb_4d_layout, MutAnyOrigin](
        b_scales.ptr,
        RuntimeLayout[sfb_4d_layout].row_major(
            IndexList[4](
                b_scales.dim(0),
                b_scales.dim(1),
                b_scales.dim(2),
                b_scales.dim(3) * b_scales.dim(4),
            ),
        ),
    )

    comptime sfa_tma_tile_shape = Index(
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )

    var sfa_tma_op = create_tensor_tile[
        sfa_tma_tile_shape,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, sfa_4d)

    comptime sfb_tma_tile_shape = Index(
        MMA_N // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )

    var sfb_tma_op = create_tensor_tile[
        sfb_tma_tile_shape,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, sfb_4d)

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime SmemType = B200BlockScaledMatmulSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]
    comptime smem_size = size_of[SmemType]()

    comptime max_profiled_tiles = 0 if max_profiled_tiles_per_SM is None else max_profiled_tiles_per_SM.value()
    comptime enable_profiling = max_profiled_tiles > 0

    comptime kernel = blackwell_block_scaled_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).rank,
        type_of(c_tma_op).tile_shape,
        c_device.layout,
        type_of(c_tma_op).desc_shape,
        type_of(sfa_tma_op).rank,
        type_of(sfa_tma_op).tile_shape,
        type_of(sfa_tma_op).desc_shape,
        type_of(sfb_tma_op).rank,
        type_of(sfb_tma_op).tile_shape,
        type_of(sfb_tma_op).desc_shape,
        group_offsets.layout,
        group_scale_offsets.layout,
        expert_ids.layout,
        expert_scales.layout,
        transpose_b,
        config=config,
        expert_n=expert_n,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        pdl_level=pdl_level,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    var grid_dim = (
        B200.sm_count,
        1,
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    # TODO: integrate with existing enums
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime epilogue_warps = 4

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    var workspace: Span[UInt64, MutAnyOrigin]

    comptime if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = {}

    ctx.enqueue_function[kernel](
        Int32(num_active_experts),
        a_tma_op,
        b_tma_op,
        c_tma_op,
        c_device,
        sfa_tma_op,
        sfb_tma_op,
        group_offsets,
        group_scale_offsets,
        expert_ids,
        expert_scales,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 4 EPILOGUE warps
        block_dim=(32 * (load_warps + mma_warps + epilogue_warps)),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )

    comptime if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(sfa_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(sfb_tma_op, `nvvm.grid_constant`)
@__name(
    StaticString(config.get_kernel_name())
    + StaticString(
        "_fused_compute_epi" if elementwise_compute_lambda_fn
        is not None else ""
    ),
)
def blackwell_block_scaled_tma_umma_warp_specialized_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    c_tile_rank: Int,
    c_tile_shape_param: IndexList[c_tile_rank],
    c_tensor_layout: Layout,
    c_desc_shape: IndexList[c_tile_rank],
    sfa_tile_rank: Int,
    sfa_tile_shape: IndexList[sfa_tile_rank],
    sfa_desc_shape: IndexList[sfa_tile_rank],
    sfb_tile_rank: Int,
    sfb_tile_shape: IndexList[sfb_tile_rank],
    sfb_desc_shape: IndexList[sfb_tile_rank],
    group_offsets_layout: Layout,
    group_scale_offsets_layout: Layout,
    expert_ids_layout: Layout,
    expert_scales_layout: Layout,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    expert_n: Int,
    # Need because nvvm.cluster_dim only takes StaticTuple
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel.ON,
    max_profiled_tiles_per_SM: UInt32 = 0,
](
    num_active_experts: Int32,
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c_tma_op: TMATensorTile[
        c_type, c_tile_rank, c_tile_shape_param, c_desc_shape
    ],
    c: LayoutTensor[c_type, c_tensor_layout, MutAnyOrigin],
    sfa_tma_op: TMATensorTile[
        sfa_dtype, sfa_tile_rank, sfa_tile_shape, sfa_desc_shape
    ],
    sfb_tma_op: TMATensorTile[
        sfb_dtype, sfb_tile_rank, sfb_tile_shape, sfb_desc_shape
    ],
    group_offsets: LayoutTensor[.uint32, group_offsets_layout, MutAnyOrigin],
    group_scale_offsets: LayoutTensor[
        .uint32, group_scale_offsets_layout, MutAnyOrigin
    ],
    expert_ids: LayoutTensor[.int32, expert_ids_layout, MutAnyOrigin],
    expert_scales: LayoutTensor[.float32, expert_scales_layout, MutAnyOrigin],
    cluster_dim: StaticTuple[Int32, 3],
    mnk: StaticTuple[UInt32, 3],
    workspace: Span[UInt64, MutAnyOrigin],
):
    var _num_active_experts = Int(num_active_experts)
    comptime register_based_epilogue = config.register_based_epilogue
    comptime assert c_type != .float32, "c_type cannot be float32"
    comptime assert transpose_b, "only support k-major B"

    comptime num_output_warps = 4

    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = config.cta_group * EPILOGUE_THREADS

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    # Override c_swizzle to SWIZZLE_32B when AB_swapped (transpose_c).
    # With SWIZZLE_32B, TMA's 32-byte row stride matches the 16-element
    # chunks produced by st_matrix[transpose=True], giving a 1:1 mapping
    # between smem rows and gmem token rows.
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_32B if config.AB_swapped else config.c_swizzle

    # For ld from TMEM, use same per-stage stride in column field.
    comptime NUM_TMEM_COLS = 512
    comptime SFA_NUM_COLS = config.num_sf_k_tiles * (BM // 32)
    comptime SFB_NUM_COLS = config.num_sf_k_tiles * (MMA_N // 32)
    comptime stage_stride_cols = config.mma_shape[1]

    comptime assert (
        config.num_accum_pipeline_stages * MMA_N
        + (SFA_NUM_COLS + SFB_NUM_COLS) * config.num_pipeline_stages
        <= NUM_TMEM_COLS
    ), "sfa_tmem and sfb_tmem exceed tmem_cols"

    comptime num_m_mmas = BM // (config.mma_shape[0] // config.cta_group)
    comptime num_n_mmas = BN // (config.mma_shape[1] // config.cta_group)
    comptime num_k_mmas = BK // config.mma_shape[2]

    comptime CLUSTER_M: Int = config.cluster_shape[0]
    comptime CLUSTER_N: Int = config.cluster_shape[1]

    comptime a_tma_load_size = _idx_product[a_tile_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    # keep the physical SMEM buffer BM x MMA_N. Use typed layouts as source of
    # truth; bridge to legacy Layout for LayoutTensor and the MMA descriptor
    # pipeline.
    comptime a_smem_layout = tile_layout_k_major_typed[
        a_type, BM, BK, swizzle_mode=config.a_swizzle
    ].to_layout()
    comptime b_smem_layout = tile_layout_k_major_typed[
        b_type, BN, BK, swizzle_mode=config.b_swizzle
    ].to_layout() if transpose_b else tile_layout_mn_major_typed[
        b_type, BN, BK, swizzle_mode=config.b_swizzle
    ].to_layout()

    # The typed form requires whole scale-factor atoms, which keeps it in step
    # with the `BM // SF_MN_GROUP_SIZE` atom counts `sfa_smem_size` and
    # `sfb_smem_size` use to size the buffers these layouts describe.
    comptime sfa_smem_layout = tile_sf_layout_k_major_typed[
        BM,
        SF_K_GROUP_SIZE[config.vec_sf_size] * config.num_sf_k_tiles,
        config.vec_sf_size,
    ].to_layout()
    comptime sfb_smem_layout = tile_sf_layout_k_major_typed[
        MMA_N,
        SF_K_GROUP_SIZE[config.vec_sf_size] * config.num_sf_k_tiles,
        config.vec_sf_size,
    ].to_layout()

    comptime SmemType = B200BlockScaledMatmulSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]

    ref smem_storage = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=128,
    ]().bitcast[SmemType]()[]

    ref a_smem_storage = smem_storage.a_smem
    ref b_smem_storage = smem_storage.b_smem
    ref c_smem_storage = smem_storage.c_smem
    ref sfa_smem_storage = smem_storage.sfa_smem
    ref sfb_smem_storage = smem_storage.sfb_smem
    ref tma_mma_mbars_storage = smem_storage.tma_mma_mbars
    ref accum_mbars_storage = smem_storage.accum_mbars
    ref tmem_addr_storage = smem_storage.tmem_addr
    ref tmem_dealloc_mbar_storage = smem_storage.tmem_dealloc_mbar

    var a_smem_ptr: UnsafePointer[
        Scalar[a_type], origin_of(a_smem_storage), address_space=.SHARED
    ] = a_smem_storage.unsafe_ptr()
    var a_smem_base = a_smem_ptr.bitcast[Scalar[a_type]]()
    var b_smem_ptr: UnsafePointer[
        Scalar[b_type], origin_of(b_smem_storage), address_space=.SHARED
    ] = b_smem_storage.unsafe_ptr()
    var b_smem_base = b_smem_ptr.bitcast[Scalar[b_type]]()
    var c_smem_ptr: UnsafePointer[
        Scalar[c_type], origin_of(c_smem_storage), address_space=.SHARED
    ] = c_smem_storage.unsafe_ptr()
    var c_smem_base = c_smem_ptr.bitcast[Scalar[c_type]]()
    var sfa_smem_ptr: UnsafePointer[
        Scalar[sfa_dtype], origin_of(sfa_smem_storage), address_space=.SHARED
    ] = sfa_smem_storage.unsafe_ptr()
    var sfa_smem_base = sfa_smem_ptr.bitcast[Scalar[sfa_dtype]]()
    var sfb_smem_ptr: UnsafePointer[
        Scalar[sfb_dtype], origin_of(sfb_smem_storage), address_space=.SHARED
    ] = sfb_smem_storage.unsafe_ptr()
    var sfb_smem_base = sfb_smem_ptr.bitcast[Scalar[sfb_dtype]]()

    # Load warp as producer and mma warp as consumer
    # Dependence on MMA input in SMEM.
    # Consumer phase = 1 so that producer's wait on consumer passes trivially
    # at the start when buffer is empty.
    var load_mma_pipeline = ProducerConsumerPipeline[
        config.num_pipeline_stages // config.k_group_size
    ](
        tma_mma_mbars_storage.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ](),
    )

    # MMA warp as producer and Output warp as consumer.
    # Dependence on MMA output in TMEM.
    var mma_output_pipeline = ProducerConsumerPipeline[
        config.num_accum_pipeline_stages
    ](
        accum_mbars_storage.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ](),
    )

    var ptr_tmem_addr: UnsafePointer[
        UInt32, origin_of(tmem_addr_storage), address_space=.SHARED
    ] = tmem_addr_storage.unsafe_ptr()

    var tmem_dealloc_mbar = tmem_dealloc_mbar_storage.unsafe_ptr()

    # hardcode to float32 for now as we only support FP32 accumulation for block scaled matmul
    # TODO: (KERN-2238) replace with get_accum_type[a_type]() when KERN-2238 is fixed and we can return FP32 for FP4-E2M1
    comptime accum_type = DType.float32

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
        sfa_tma_op.prefetch_descriptor()
        sfb_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            Int32(
                config.cluster_shape[0] // config.cta_group
                + config.cluster_shape[1]
                - 1
            ),
        )
        mma_output_pipeline.init_mbars(
            accum_pipeline_producer_arv_count,
            Int32(accum_pipeline_consumer_arv_count),
        )

        tmem_dealloc_mbar[].init(Int32(EPILOGUE_THREADS * config.cta_group))

    fence_mbarrier_init()
    cluster_sync()

    var mma_op = MmaOpSM100_BlockScaled_SS[
        c_type,
        a_type,
        b_type,
        sfa_dtype,
        sfb_dtype,
        config.scaling_kind,
        config.block_tile_shape,
        config.mma_shape,
        accum_type=accum_type,
        cta_group=config.cta_group,
        cluster_shape=config.cluster_shape,
        a_swizzle=config.a_swizzle,
        b_swizzle=config.b_swizzle,
        transpose_b=True,
    ]()

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
        swapAB=config.AB_swapped,
    ](_num_active_experts, group_offsets)

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
        b_multicast_mask |= 1 << UInt16(i * config.cta_group)

    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * CLUSTER_M)

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    var num_iters: UInt32 = ceildiv(mnk[2], UInt32(BK))

    comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
        warp_role, max_profiled_tiles_per_SM
    ]
    var expert_id = Int32(-1)
    if not work_info.is_done():
        expert_id = rebind[Int32](expert_ids[Int(scheduler.current_group_idx)])

    if WarpRole.is_main_load():
        with MatmulProfilerType[0](workspace, 0):
            comptime if pdl_level > PDLLevel.OFF:
                wait_on_dependent_grids()

            while not work_info.is_done():
                if not work_info.is_valid() or expert_id < 0:
                    work_info = scheduler.fetch_next_work()
                    if not work_info.is_done():
                        expert_id = rebind[Int32](
                            expert_ids[Int(scheduler.current_group_idx)]
                        )
                    continue

                # DO TMA LOAD
                for i in range(num_iters // UInt32(config.k_group_size)):
                    load_AB[
                        a_smem_layout=a_smem_layout,
                        b_smem_layout=b_smem_layout,
                        sfa_smem_layout=sfa_smem_layout,
                        sfb_smem_layout=sfb_smem_layout,
                        config=config,
                        block_tile_shape=config.block_tile_shape,
                        mma_shape=config.mma_shape,
                        cta_group=config.cta_group,
                        k_group_size=config.k_group_size,
                        num_sf_k_tiles=config.num_sf_k_tiles,
                    ](
                        a_tma_op,
                        b_tma_op,
                        sfa_tma_op,
                        sfb_tma_op,
                        a_smem_base,
                        b_smem_base,
                        sfa_smem_base,
                        sfb_smem_base,
                        load_mma_pipeline,
                        peer_cta_coord,
                        (Int(work_info.m), Int(work_info.n)),
                        a_multicast_mask,
                        b_multicast_mask,
                        i * UInt32(config.k_group_size),
                        elect_one_cta,
                        scheduler,
                        rebind[Int32](expert_id),
                        group_scale_offsets,
                    )
                    load_mma_pipeline.producer_step()

                syncwarp()
                var next_work_info = scheduler.fetch_next_work()
                work_info = next_work_info
                if not work_info.is_done():
                    expert_id = rebind[Int32](
                        expert_ids[Int(scheduler.current_group_idx)]
                    )

            # Prevent CTA to exit when a peer CTA is still working on mma.
            comptime for i in range(
                config.num_pipeline_stages // config.k_group_size
            ):
                load_mma_pipeline.wait_consumer()
                load_mma_pipeline.producer_step()

    if WarpRole.is_mma():
        with MatmulProfilerType[2](workspace, 0):
            tcgen05_alloc[Int32(config.cta_group)](ptr_tmem_addr, max_tmem_cols)
            syncwarp()
            # non blocking, arrives and proceeds
            named_barrier_arrive[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)

            var tmem_addr = ptr_tmem_addr[0]
            var sfa_tmem = tmem_addr + UInt32(
                config.num_accum_pipeline_stages * MMA_N
            )
            var sfb_tmem = sfa_tmem + UInt32(
                SFA_NUM_COLS * config.num_pipeline_stages
            )

            while not work_info.is_done():
                if not work_info.is_valid() or expert_id < 0:
                    work_info = scheduler.fetch_next_work()
                    if not work_info.is_done():
                        expert_id = rebind[Int32](
                            expert_ids[Int(scheduler.current_group_idx)]
                        )
                    continue
                # scheduler fetch next work
                var next_work_info = scheduler.fetch_next_work()
                # DO MMA
                if elect_one_cta:
                    var mma_output_mma_stage = (
                        mma_output_pipeline.producer_stage()
                    )
                    mma_output_pipeline.wait_consumer()
                    var tmem_offset = tmem_addr + (
                        mma_output_mma_stage * UInt32(stage_stride_cols)
                    )

                    for i in range(num_iters // UInt32(config.k_group_size)):
                        consumer_main_loop[
                            a_smem_layout=a_smem_layout,
                            b_smem_layout=b_smem_layout,
                            sfa_smem_layout=sfa_smem_layout,
                            sfb_smem_layout=sfb_smem_layout,
                            block_tile_shape=config.block_tile_shape,
                            mma_shape=config.mma_shape,
                            SFA_NUM_COLS=SFA_NUM_COLS,
                            SFB_NUM_COLS=SFB_NUM_COLS,
                            cta_group=config.cta_group,
                            cluster_shape=config.cluster_shape,
                            k_group_size=config.k_group_size,
                        ](
                            tmem_offset,
                            sfa_tmem,
                            sfb_tmem,
                            a_smem_base,
                            b_smem_base,
                            sfa_smem_base,
                            sfb_smem_base,
                            load_mma_pipeline,
                            mma_op,
                            elect_one_warp,
                            i * UInt32(config.k_group_size),
                            0,
                            work_tile_coord=(
                                Int(work_info.m),
                                Int(work_info.n),
                            ),
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
                if not work_info.is_done():
                    expert_id = rebind[Int32](
                        expert_ids[Int(scheduler.current_group_idx)]
                    )

            tcgen05_release_allocation_lock[Int32(config.cta_group)]()

            # wait for epilogue to finish
            tmem_dealloc_mbar[].wait()

            comptime if pdl_level > PDLLevel.OFF:
                launch_dependent_grids()

            tcgen05_dealloc[Int32(config.cta_group)](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)
        var tmem_addr = ptr_tmem_addr[0]

        var tile_idx = 0

        while not work_info.is_done():
            if not work_info.is_valid() or expert_id < 0:
                work_info = scheduler.fetch_next_work()
                if not work_info.is_done():
                    expert_id = rebind[Int32](
                        expert_ids[Int(scheduler.current_group_idx)]
                    )
                continue
            with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                var expert_scale = expert_scales[Int(expert_id)]
                var group_end_idx = rebind[UInt32](
                    scheduler.group_offsets[
                        Int(scheduler.current_group_idx + 1)
                    ]
                )
                # WAIT FOR MMA TO FINISH AND STORE RESULT
                # scheduler fetch next work
                multi_stage_store_C[
                    c_smem_layout=Layout.row_major(
                        config.output_tile_shape[0],
                        config.output_tile_shape[1],
                    ),
                    input_type=a_type,
                    accum_type=accum_type,
                    block_tile_shape=config.block_tile_shape,
                    mma_shape=config.mma_shape,
                    stage_stride_cols=stage_stride_cols,
                    c_swizzle=c_swizzle,
                    cta_group=config.cta_group,
                    num_output_warps=num_output_warps,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    register_based_epilogue=register_based_epilogue,
                    transpose_c=config.AB_swapped,
                    scale_c_coord=False,
                ](
                    c_smem_base,
                    c_tma_op,
                    c,
                    mma_output_pipeline,
                    tmem_addr,
                    work_tile_coord=(work_info.m, work_info.n),
                    elect_one_warp=elect_one_warp,
                    expert_scale=rebind[Float32](expert_scale),
                    M=mnk[0],
                    N=mnk[1],
                    group_end_idx=group_end_idx,
                )
                mma_output_pipeline.consumer_step()

                var next_work_info = scheduler.fetch_next_work()
                work_info = next_work_info
                if not work_info.is_done():
                    expert_id = rebind[Int32](
                        expert_ids[Int(scheduler.current_group_idx)]
                    )

            tile_idx += 1

        comptime if config.cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


def grouped_matmul_dynamic_scaled_nvfp4[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_type: DType,
    //,
    transpose_b: Bool = True,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, scales_type, address_space=.GENERIC, ...],
    group_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    group_scale_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    expert_scales: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Performs grouped matrix multiplication with NVFP4 quantization.

    Computes C = A @ B^T for multiple expert groups in a Mixture of Experts
    (MoE) layer. Inputs A and B are NVFP4 quantized (4-bit floating point),
    packed as uint8 (2 values per byte), with float8_e4m3fn scale factors.
    Each group of 16 elements along the K dimension shares a single scale
    factor (1D block scaling).

    Accepts TileTensors and converts to LayoutTensors internally.

    Parameters:
        c_type: The data type of the output tensor C.
        a_type: The data type of input tensor A. Constraints: Must be `uint8`.
        b_type: The data type of input tensor B. Constraints: Must be `uint8`.
        scales_type: The data type of scale factors.
            Constraints: Must be `float8_e4m3fn`.
        transpose_b: Whether B is transposed. Constraints: Must be `True`.
        target: The target device.

    Args:
        c: The output tensor of shape (total_tokens, N).
        a: The input tensor of shape (total_tokens, K // 2), packed NVFP4.
        b: The weight tensor of shape (num_experts, N, K // 2), packed NVFP4.
        a_scales: The scale factors for A in tcgen05 5D layout.
        b_scales: The scale factors for B in tcgen05 6D layout.
        group_offsets: The starting token index for each expert group.
        group_scale_offsets: The starting scale index for each expert group.
        expert_ids: The expert ID for each group.
        expert_scales: The per-expert scaling factors applied in the epilogue.
        num_active_experts: The number of active experts in this batch.
        ctx: The device context for GPU execution.

    Constraints:
        - The target device must be SM100 (B200).
    """
    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "Only support SM100 for grouped NVFP4 matmul"
    comptime assert transpose_b, "Only support transpose_b = True"
    comptime assert (
        a_type == b_type == .uint8
    ), "input A and B dtype should be uint8 for NVFP4"
    comptime assert (
        scales_type == NVFP4_SF_DTYPE
    ), "scales dtype should be NVFP4_SF_DTYPE (float8_e4m3fn)"
    if num_active_experts == 0:
        return

    @always_inline
    def description_fn() {var c, var a, imm} -> String:
        # fmt: off
        return String(
            "(gpu",
            ";A=", c.dim(0), "x", a.dim(1), "x", a_type,
            ";C=", c.dim(0), "x", c.dim(1), "x", c_type,
            ";num_active_experts=", num_active_experts,
            ";transpose_b=", transpose_b,
            ")"
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        get_static_string[
            "grouped_matmul_dynamic_scaled_nvfp4_",
            String(a_type) + "x" + String(b_type) + "_to_" + String(c_type),
            "_scales_" + String(scales_type),
        ](),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(ctx),
    ):
        comptime MMA_K = 32
        comptime bm = 128
        comptime bn = 128
        comptime mma_shape = Index(bm, bn, MMA_K)

        comptime matmul_config = BlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_type, scales_type, transpose_b
        ](
            scaling_kind=UMMAKind.KIND_MXF4NVF4,
            cluster_shape=Index(1, 1, 1),
            mma_shape=mma_shape,
            block_swizzle_size=8,
            cta_group=1,
            AB_swapped=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            gemm_kind=GEMMKind.GMM,
        )

        blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=matmul_config,
        ](
            c,
            a,
            group_offsets,
            group_scale_offsets,
            b,
            expert_ids,
            a_scales,
            b_scales,
            expert_scales,
            num_active_experts,
            ctx,
        )
