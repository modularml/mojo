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
"""SM100 matmul - TileTensor-native compatibility layer for grouped_matmul.

NOTE: This module is maintained for backward compatibility with grouped_matmul
implementations that depend on internal functions (WarpRole, consumer_main_loop,
stsm_helper, shared_memory_epilogue, register_epilogue, accum_arrive).
The helper surface is TileTensor-native; legacy callers should adapt at their
own boundary before entering this module.

For new code, use sm100_structured directly:
- Import configs from: linalg.matmul.gpu.sm100_structured.config
- Import matmul from: linalg.matmul.gpu.sm100_structured.matmul
"""

from std.sys import align_of, simd_width_of, size_of
from std.math.uutils import umod, ufloordiv, udivmod

from max.gpu import WARP_SIZE, lane_id, warp_id
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.mma import st_matrix
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.sync import (
    named_barrier,
    umma_arrive_leader_cta,
    mbarrier_arrive,
)
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    Coord,
    IntTuple,
    Idx,
    Layout,
    RuntimeLayout,
    TileTensor,
    row_major,
    col_major,
)
from layout.swizzle import Swizzle
from layout.tile_layout import (
    TensorLayout,
    ZippedDivideLayout,
    UpcastLayout,
    BlockedProductLayout,
)
from structured_kernels.tile_types import SMemTileArray2D

from std.utils.fast_div import FastDiv
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from ....arch.sm100 import MmaOpSM100_SS
from ....utils import elementwise_compute_lambda_type
from .pipeline import ProducerConsumerPipeline, MbarPtr


@fieldwise_init
struct WarpRole[has_scheduler: Bool = True](TrivialRegisterPassable):
    """Warp-role assignments for the SM100 warp-specialized matmul kernel.

    Each warp is statically assigned to one of four roles: `Mma` computes
    UMMA instructions using TCGEN05; `MainLoad` issues TMA requests to fill
    the A/B shared-memory pipeline; `Scheduler` advances the tile work queue;
    `Epilogue` writes accumulator results to global memory. The role indices
    shift by one when `has_scheduler` is `True`.

    Parameters:
        has_scheduler: Whether a dedicated scheduler warp is present.
    """

    var _role: Int32

    comptime Mma = Self(6) if Self.has_scheduler else Self(5)
    comptime MainLoad = Self(5) if Self.has_scheduler else Self(4)
    comptime Scheduler = Self(4)
    comptime Epilogue = Self(3)

    @always_inline
    def __eq__(self, other: Int) -> Bool:
        return self._role == Int32(other)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    def __ge__(self, other: Int) -> Bool:
        return self._role >= Int32(other)

    @staticmethod
    @always_inline
    def is_main_load() -> Bool:
        return Self.MainLoad == warp_id()

    @staticmethod
    @always_inline
    def is_mma() -> Bool:
        return Self.Mma == warp_id()

    @staticmethod
    @always_inline
    def is_epilogue() -> Bool:
        return Self.Epilogue >= warp_id()

    @staticmethod
    @always_inline
    def is_scheduler() -> Bool:
        comptime assert Self.has_scheduler, "Scheduler warp is not enabled"
        return Self.Scheduler == warp_id()


@always_inline
def consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_dim0: Int,
    a_dim1: Int,
    a_num_tiles: Int,
    a_swizzle_bytes: Int,
    b_dim0: Int,
    b_dim1: Int,
    b_num_tiles: Int,
    b_swizzle_bytes: Int,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    k_group_size: Int = 1,
](
    tmem_addr: UInt32,
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
    ],
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    mma_op: MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
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
):
    """TileTensor overload of `consumer_main_loop`.

    Accepts `SMemTileArray2D` instead of iterator-based shared-memory tiles, indexing directly
    into the tile arrays to get TileTensor tiles for MMA. The tile dimension
    and swizzle parameters (a_dim0, a_dim1, etc.) are explicit because Mojo
    requires them for overload resolution with parametric struct arguments.
    """
    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    # Compose TMEM address: accum stage encoded in column field with stride in columns.
    if elect_one_sync():
        for j in range(UInt32(k_group_size)):
            var offset = stage * UInt32(k_group_size) + j
            var a_smem_tile = a_smem_tiles[offset]
            var b_smem_tile = b_smem_tiles[offset]
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(
                    (iter_idx + j) == k_start
                ),  # Initialize C on first iteration
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))


comptime RLayout32Bits[layout: Layout] = RuntimeLayout[
    layout, element_type=.uint32, linear_idx_type=.uint32
]


@always_inline
def f32_frag_to_smem[
    swizzle_mode: TensorMapSwizzle,
    vec_dtype: DType,
    vec_size: Int,
    DstLayout: TensorLayout,
](
    vec: Array[Scalar[vec_dtype], vec_size],
    dst: TileTensor[_, DstLayout, MutAnyOrigin, address_space=.SHARED],
):
    """Writes an FP32 TCGEN05 accumulator fragment to a swizzled shared-memory tile.

    Implements a manual 8×4 thread-to-element distribution for FP32 fragments,
    since the compiler cannot prove `all_dims_known` through migrated layout types.
    Each lane writes a pair of FP32 values at the correct swizzled row/column offset.

    Parameters:
        swizzle_mode: Shared memory swizzle mode to apply.
        vec_dtype: Element type of the source fragment (must be FP32).
        vec_size: Number of elements in the fragment (= 2 × frag_rows × frag_cols).
        DstLayout: Layout of the destination shared-memory tile.

    Args:
        vec: Source accumulator fragment as a flat inline array.
        dst: Destination tile in shared memory.
    """
    # Manual implementation of dst.vectorize[1, 2]().distribute[row_major(8, 4)]
    # because the compiler can't prove `all_dims_known` through migrated layout types. See MSTDL-2422.
    comptime stride0: Int = dst.static_stride[0]
    comptime shape0 = dst.static_shape[0]
    comptime shape1 = dst.static_shape[1]

    comptime frag_rows = shape0 // 8
    comptime frag_cols = shape1 // 8
    comptime assert (
        2 * frag_rows * frag_cols == vec_size
    ), "2*frag_rows*frag_cols must be equal to vec_size"

    var lane = lane_id()
    var thread_row = (lane >> 2) & 7
    var thread_col = lane & 3
    var base_offset = thread_row * stride0 + thread_col * 2

    comptime for i in range(frag_rows):
        comptime for j in range(frag_cols):
            comptime i_vec = i + j * frag_rows
            var val = SIMD[dst.dtype, 2](
                rebind[Scalar[dst.dtype]](vec[2 * i_vec]),
                rebind[Scalar[dst.dtype]](vec[2 * i_vec + 1]),
            )
            var offset = base_offset + i * 8 * stride0 + j * 8
            (dst.ptr + offset).store(val)


@always_inline
def stsm_helper[
    swizzle: Swizzle,
    stageN: Int,
    vec_dtype: DType,
    vec_size: Int,
    DstLayout: TensorLayout,
    transpose_c: Bool = False,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    vec: Array[Scalar[vec_dtype], vec_size],
    dst: TileTensor[_, DstLayout, MutAnyOrigin, address_space=.SHARED],
    warp_offset: UInt32 = 0,
):
    """Stores a TCGEN05 accumulator fragment to shared memory using st.matrix or scalar stores.

    Routes to `f32_frag_to_smem` for FP32 fragments, or uses `st_matrix` hardware
    instructions (`stsmx4` or `stsmx2`) for narrower types. Applies swizzle addressing
    so the destination layout matches subsequent TMA-store or register-epilogue consumers.

    Parameters:
        swizzle: Swizzle descriptor for shared memory address computation.
        stageN: Column stage dimension used to select stsmx4 vs stsmx2 path.
        vec_dtype: Fragment element type.
        vec_size: Number of elements in the fragment.
        DstLayout: Static layout of the destination tile in shared memory.
        transpose_c: Whether to transpose the C tile before writing (FP32 only).
        swizzle_mode: TMA swizzle mode for FP32 fallback path.

    Args:
        vec: Accumulator fragment to write.
        dst: Destination tile in shared memory.
        warp_offset: Per-warp row offset within the tile (default 0).
    """
    comptime if size_of[dst.dtype]() == 4:
        comptime assert not transpose_c, "transpose_c must be False"
        return f32_frag_to_smem[swizzle_mode](vec, dst)
    # Number of elements in one row is 32B and 16B per stsmx4 and stmtx2 tile, respectively.
    comptime stsmx_row_size = 32 // size_of[
        dst.dtype
    ]() if stageN % 16 == 0 else 16 // size_of[dst.dtype]()
    # Number of elements owned by each lane, each lane has 16B
    comptime stsmx_lane_size = 16 // size_of[dst.dtype]()
    # TODO: constrain the shared memory layout to be 2D row-major.
    # E.g. dst layout can be (16, 16) : (32, 1), which is tiled from
    # row-major(16, 32). The map should use tile's stride to calculate
    # the dst row offset.
    comptime stride0 = dst.static_stride[0]
    comptime stride1 = dst.static_stride[1]
    comptime assert stride1 == 1, (
        "stride1 must be 1. Got: "
        + String(stride1)
        + " for strides ("
        + String(stride0)
        + ", "
        + String(stride1)
        + ")"
    )
    comptime shape0 = dst.static_shape[
        1
    ] if not transpose_c else dst.static_shape[0]
    # the layout looks like
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
    # but transposed and coalesced by 8 elements.
    comptime trans_st_matrix_layout = Layout(
        IntTuple(8, 2, 2), IntTuple(stride0, 8 * stride1, 8 * stride0)
    )
    comptime stsmx_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx_row_size

    var lane = lane_id()
    var stsm_lane_offset = UInt32(
        (lane & 15) * stride0 + (lane >> 4) * 8
    ) if not transpose_c else RLayout32Bits[trans_st_matrix_layout]()(lane)

    # Assume the dst tile has 16 rows and only use stsm in N dim.
    comptime for i in range(shape0 // stsmx_row_size):
        comptime n_offset = i * stsmx_tile_offset
        var offset: UInt32

        comptime if transpose_c:
            offset = (
                swizzle(stsm_lane_offset + UInt32(n_offset) + warp_offset)
                - warp_offset
            )
        else:
            offset = swizzle(stsm_lane_offset + UInt32(n_offset))
        comptime stmtx_simd_width = 4 if stageN % 16 == 0 else 2
        comptime cast_width = 4 // size_of[Scalar[dst.dtype]]()
        var v = SIMD[dst.dtype, stmtx_simd_width * cast_width]()
        comptime for k in range(stmtx_simd_width):
            var src = SIMD[vec_dtype, cast_width]()
            comptime for _j in range(cast_width):
                src[_j] = vec[i * stsmx_lane_size + k * cast_width + _j]
            var casted = src.cast[dst.dtype]()
            comptime for _j in range(cast_width):
                v[k * cast_width + _j] = casted[_j]
        st_matrix[simd_width=stmtx_simd_width, transpose=transpose_c](
            dst.ptr + offset, bitcast[.float32, stmtx_simd_width](v)
        )


@always_inline
def shared_memory_epilogue[
    MMA_M: Int,
    data_paths: Int,
    num_stages: Int,
    stage: Int,
    stageN: Int,
    c_type: DType,
    shared_n: Int,
    simd_size: Int,
    swizzle: Swizzle,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: Int,
    c_row: Int,
    c_smem_warp_tile_upper: TileTensor[mut=True, c_type, ...],
    c_smem_warp_tile_lower: TileTensor[mut=True, c_type, ...],
):
    """Applies a compute epilogue to a C tile stored in shared memory for SM100 matmul.

    Reads the upper and lower warp sub-tiles from shared memory, un-swizzles the
    element addresses to recover (row, col) coordinates in the global C matrix,
    and calls `compute_lambda_fn` with the resulting SIMD values. Supports both
    MMA_M=128 and MMA_M=256 tensor-memory layouts.

    Parameters:
        MMA_M: M dimension of a single MMA tile (128 or 256).
        data_paths: Number of TCGEN05 data paths used in the column layout.
        num_stages: Number of epilogue pipeline stages in the N dimension.
        stage: Current epilogue pipeline stage index.
        stageN: N dimension of one epilogue stage.
        c_type: Element type of the C tile in shared memory.
        shared_n: Column count of the shared-memory C tile, used as the
            row stride in elements.
        simd_size: SIMD vector width used to distribute the shared-memory
            fragment across lanes.
        swizzle: Swizzle descriptor for shared memory address computation.
        compute_lambda_fn: Elementwise lambda applied to each output element.
        num_output_warps: Number of warps participating in the epilogue.

    Args:
        M: Total M dimension of the problem (used for bounds checking).
        N: Total N dimension of the problem (used for bounds checking).
        c_col: Global column offset for this output tile.
        c_row: Global row offset for this output tile.
        c_smem_warp_tile_upper: Upper warp half of the C tile in shared memory.
        c_smem_warp_tile_lower: Lower warp half of the C tile in shared memory.
    """
    # Here we start keeping track of the index / indices this thread is
    # responsible for in shared memory. This is represented with shared_memory_row
    # and shared_memory_column and the children of these values shared_memory_row_upper_half
    # shared_memory_row_lower_half. We also need to update the global memory column c_col by
    # stageN since we are sliding through the overall compute block.

    var staged_c_col = c_col + stage * stageN

    var warp_id = warp_id()
    var shared_memory_row = warp_id * 32

    var shared_memory_row_upper_half = shared_memory_row
    var shared_memory_row_lower_half = shared_memory_row + 16

    # This distribute layout allocates vectors to corresponding threads. If stageN is 32, 8 x 4 is used since each row of
    # 4 threads can access 8 elements (8 x 4 = 32). If stageN is 16 then 16 x 2 is used. Since each fragment contains 16 rows,
    # there will be 2 chunks created when using 8x4.

    comptime distribute_cols = stageN // simd_size
    comptime distribute_rows = WARP_SIZE // distribute_cols

    comptime distribute_layout = row_major[distribute_rows, distribute_cols]()
    var c_smem_upper_frag = c_smem_warp_tile_upper.vectorize[
        1, simd_size
    ]().distribute[distribute_layout, swizzle=swizzle](lane_id())

    var c_smem_lower_frag = c_smem_warp_tile_lower.vectorize[
        1, simd_size
    ]().distribute[distribute_layout, swizzle=swizzle](lane_id())

    comptime fragment_size = c_smem_upper_frag.LayoutType.static_product

    var local_row, local_col = udivmod(lane_id(), distribute_cols)

    var shared_memory_col = local_col * simd_size
    shared_memory_row_lower_half += local_row
    shared_memory_row_upper_half += local_row

    comptime for i in range(fragment_size):
        comptime alignment = align_of[SIMD[c_type, simd_size]]()

        # these offsets are swizzled so to retrieve the corresponding gmem offset we need to remove the swizzle
        # luckily removing the swizzle is as simple as swizzling a second time
        var swz_offset_upper = (
            shared_memory_row_upper_half * shared_n + shared_memory_col
        )
        var swz_offset_lower = (
            shared_memory_row_lower_half * shared_n + shared_memory_col
        )

        var offset_upper = swizzle(swz_offset_upper)
        var offset_lower = swizzle(swz_offset_lower)

        var shared_upper_row: Int64
        var shared_upper_col: Int64
        var shared_lower_row: Int64
        var shared_lower_col: Int64

        # Now that we have the true index we, need to add the global tile index to find the corresponding
        # index, in gmem. However the data will be stored in tensor memory differently depending on
        # MMA_M size, we take that into account here.

        comptime if MMA_M != 256:
            comptime blocked_m_128_layout = BlockedProductLayout[
                type_of(row_major[data_paths * 2, stageN]()),
                type_of(col_major[2, 2]()),
                coalesce_output=True,
            ]()

            var upper_coord = blocked_m_128_layout.idx2crd(
                offset_upper,
            )

            var lower_coord = blocked_m_128_layout.idx2crd(offset_lower)

            shared_upper_row = Int64(upper_coord[0].value())
            shared_lower_row = Int64(lower_coord[0].value())

            var section_offset_upper = Int64(upper_coord[1].tuple()[1].value())
            var col_offset_upper = Int64(upper_coord[1].tuple()[0].value())

            var section_offset_lower = Int64(lower_coord[1].tuple()[1].value())
            var col_offset_lower = Int64(lower_coord[1].tuple()[0].value())

            shared_upper_col = (
                section_offset_upper * Int64(num_stages * stageN)
                + col_offset_upper
            )
            shared_lower_col = (
                section_offset_lower * Int64(num_stages * stageN)
                + col_offset_lower
            )

        else:
            # can't cast to uint64 as it's not supported yet
            # this will cost us slightly in performance
            comptime fast_div = FastDiv[.uint32](shared_n)

            shared_upper_row = (
                Int(offset_upper).cast[fast_div.uint_type]() / fast_div
            ).cast[.int64]()
            shared_upper_col = Int64(offset_upper % shared_n)

            shared_lower_row = (
                Int(offset_lower).cast[fast_div.uint_type]() / fast_div
            ).cast[.int64]()
            shared_lower_col = Int64(offset_lower % shared_n)

        # now we need to add the global tile offset
        var global_upper_row = shared_upper_row + Int64(c_row)
        var global_upper_col = shared_upper_col + Int64(staged_c_col)
        var global_lower_row = shared_lower_row + Int64(c_row)
        var global_lower_col = shared_lower_col + Int64(staged_c_col)

        comptime assert c_smem_upper_frag.flat_rank >= 2
        comptime assert c_smem_lower_frag.flat_rank >= 2

        if global_upper_row < Int64(Int(M)) and global_upper_col < Int64(
            Int(N)
        ):
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_upper_row), Int(global_upper_col)),
                c_smem_upper_frag[Coord(i, Idx[0])],
            )
            c_smem_upper_frag[Coord(i, Idx[0])] = reg_val

        if global_lower_row < Int64(Int(M)) and global_lower_col < Int64(
            Int(N)
        ):
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_lower_row), Int(global_lower_col)),
                c_smem_lower_frag[i, Idx[0]],
            )
            c_smem_lower_frag[Coord(i, Idx[0])] = reg_val

        # If more than one chunk is created (happens when 8x4 is used)
        # they will be spaced 8 rows away from each other

        shared_memory_row_upper_half += distribute_rows
        shared_memory_row_lower_half += distribute_rows

    named_barrier[Int32(num_output_warps * WARP_SIZE)]()


@always_inline
def _compute_register_lambda_fn[
    epilogue_dtype: DType,
    frag_size: Int,
    inc: Int,
    offset: Int,
    compute_lambda_fn: elementwise_compute_lambda_type,
    transpose_c: Bool,
](
    top_coord: StaticTuple[UInt32, 2],
    bottom_coord: StaticTuple[UInt32, 2],
    mut frag: Array[Scalar[epilogue_dtype], frag_size],
    staged_c_row: UInt32,
    staged_c_col: UInt32,
):
    # update local coordinates w/ global memory offsets
    var top_frag_upper_coord = StaticTuple[UInt32, 2](
        staged_c_row + top_coord[0], staged_c_col + top_coord[1] + UInt32(inc)
    )

    var bottom_frag_upper_coord = StaticTuple[UInt32, 2](
        staged_c_row + bottom_coord[0],
        staged_c_col + bottom_coord[1] + UInt32(inc),
    )

    # In normal case, top and bottom are elements on the M dimension
    # when transpose_c is true, they are on the N dimension. We change the index order
    # when we do the transpose and pass the elements one-by-one to the lambda function.
    comptime for i in range(2):
        comptime if not transpose_c:
            frag[offset + i] = compute_lambda_fn(
                IndexList[2](
                    Int(top_frag_upper_coord[0]),
                    Int(top_frag_upper_coord[1] + UInt32(i)),
                ),
                frag[offset + i],
            )
            frag[offset + 2 + i] = compute_lambda_fn(
                IndexList[2](
                    Int(bottom_frag_upper_coord[0]),
                    Int(bottom_frag_upper_coord[1] + UInt32(i)),
                ),
                frag[offset + 2 + i],
            )
        else:
            frag[offset + i] = compute_lambda_fn(
                IndexList[2](
                    Int(top_frag_upper_coord[1] + UInt32(i)),
                    Int(top_frag_upper_coord[0]),
                ),
                frag[offset + i],
            )
            frag[offset + 2 + i] = compute_lambda_fn(
                IndexList[2](
                    Int(bottom_frag_upper_coord[1] + UInt32(i)),
                    Int(bottom_frag_upper_coord[0]),
                ),
                frag[offset + 2 + i],
            )


@always_inline
def register_epilogue[
    MMA_M: Int,
    data_paths: Int,
    num_stages: Int,
    bits: Int,
    stage: Int,
    stageN: Int,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: Int,
    epilogue_dtype: DType,
    frag_size: Int,
    repeats: Int,
    transpose_c: Bool,
    cta_group: Int,
    is_lower_frag_required: Bool,
](
    mut upper_frag_casted: Array[Scalar[epilogue_dtype], frag_size],
    mut lower_frag_casted: Array[Scalar[epilogue_dtype], frag_size],
    c_row: UInt32,
    c_col: UInt32,
    N: UInt32,
):
    """Applies an elementwise compute epilogue to accumulator fragments held in tensor memory.

    Computes the global (row, col) coordinate of each element in the upper and
    lower accumulator fragments based on the TCGEN05 tensor-memory fragment
    layout, the MMA_M tile shape, and the CTA group configuration, then invokes
    `compute_lambda_fn` to transform each element in place. The lower fragment
    is processed only when `is_lower_frag_required` is `True`.

    Parameters:
        MMA_M: M dimension of a single MMA tile (64, 128, or 256).
        data_paths: Number of TCGEN05 data paths (must be 16).
        num_stages: Number of epilogue pipeline stages in the N dimension.
        bits: Tensor-memory load width in bits (must be 256).
        stage: Current epilogue pipeline stage index.
        stageN: N dimension of one epilogue stage.
        compute_lambda_fn: Elementwise lambda applied to each output element.
        num_output_warps: Number of warps participating in the epilogue.
        epilogue_dtype: Element type of the output fragments.
        frag_size: Number of elements per fragment.
        repeats: Number of 16x256b loads repeated per fragment.
        transpose_c: Whether the C tile is transposed.
        cta_group: Number of CTAs in the UMMA group.
        is_lower_frag_required: Whether the lower fragment needs processing.

    Args:
        upper_frag_casted: Upper accumulator fragment to transform in place.
        lower_frag_casted: Lower accumulator fragment to transform in place.
        c_row: Global row offset of the output tile.
        c_col: Global column offset of the output tile.
        N: Total N dimension of the problem (used for bounds checking).
    """
    comptime assert (
        bits == 256 and data_paths == 16
    ), "Only 16x256b tensor memory load is supported"

    comptime load_width = 2

    var warp_id = warp_id()

    # get global memory offset based on tile coordinates

    # we update the column offset to include the current stage
    var staged_c_col = c_col + UInt32(stage * stageN)
    var staged_c_row = c_row

    comptime if MMA_M == 256 or (MMA_M == 128 and cta_group == 1):
        # based on layout A/D (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-a)
        staged_c_row += UInt32(warp_id * 32)
    elif MMA_M == 64 and cta_group == 1:
        # based on layout F (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-f)
        staged_c_row += UInt32(warp_id * 16)
    else:
        # based on layout B (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-b)
        staged_c_row += UInt32(umod(warp_id, 2) * 32)
        staged_c_col += UInt32(ufloordiv(warp_id, 2) * num_stages * stageN)

    # this is the tensor memory layout
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-fragments-shape-16256b
    # we use it to figure out the starting coordinate
    comptime threads_per_row = stageN // repeats // load_width
    var top_frag_upper_coord_left = StaticTuple[UInt32, 2](
        UInt32(ufloordiv(lane_id(), threads_per_row)),
        UInt32(umod(lane_id(), threads_per_row) * load_width),
    )

    # getting the other 3 coordinates is straightforward. Each fragment is spaced out by 16 rows
    # and within each fragment the elements are spaced out by 8 rows(this can be seen by the tv layout).
    var bottom_frag_upper_coord_left = StaticTuple[UInt32, 2](
        top_frag_upper_coord_left[0] + 8, top_frag_upper_coord_left[1]
    )

    var top_frag_lower_coord_left = StaticTuple[UInt32, 2](
        top_frag_upper_coord_left[0] + 16, top_frag_upper_coord_left[1]
    )

    var bottom_frag_lower_coord_left = StaticTuple[UInt32, 2](
        top_frag_lower_coord_left[0] + 8, top_frag_lower_coord_left[1]
    )

    comptime for i in range(repeats):
        # each tensor memory load (16x256b) may be repeated based on our desired size.
        # if that's the case our fragment will be repeated as well. So process it in chunks i.e
        # one 16x256b at a time.
        # inc represents the shift in global memory offset for each chunk, based on the repeat, and
        # offset represents the offset into the fragment for each chunk.

        comptime inc = i * 8
        comptime offset = i * 4

        comptime helper = _compute_register_lambda_fn[
            epilogue_dtype=epilogue_dtype,
            frag_size=frag_size,
            compute_lambda_fn=compute_lambda_fn,
            inc=inc,
            offset=offset,
            transpose_c=transpose_c,
        ]

        helper(
            top_frag_upper_coord_left,
            bottom_frag_upper_coord_left,
            upper_frag_casted,
            staged_c_row,
            staged_c_col,
        )

        comptime if is_lower_frag_required:
            helper(
                top_frag_lower_coord_left,
                bottom_frag_lower_coord_left,
                lower_frag_casted,
                staged_c_row,
                staged_c_col,
            )


@always_inline
def accum_arrive[
    cta_group: Int
](mma_output_pipeline: ProducerConsumerPipeline, mma_output_stage: UInt32):
    """Signals arrival at the MMA output pipeline barrier for a given stage.

    Dispatches between a plain `mbarrier_arrive` for single-CTA groups and
    `umma_arrive_leader_cta` for multi-CTA UMMA groups, where only the leader
    CTA signals arrival.

    Parameters:
        cta_group: Number of CTAs in the UMMA group.

    Args:
        mma_output_pipeline: Producer-consumer pipeline tracking MMA completion.
        mma_output_stage: Pipeline stage index to arrive at.
    """
    comptime if cta_group == 1:
        _ = mbarrier_arrive(
            rebind[MbarPtr](mma_output_pipeline.consumer_mbar(mma_output_stage))
        )
    else:
        umma_arrive_leader_cta(
            rebind[MbarPtr](mma_output_pipeline.consumer_mbar(mma_output_stage))
        )
