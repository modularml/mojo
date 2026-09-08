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

from std.memory import UnsafePointer
from std.math import align_down, align_up, ceildiv, clamp, rsqrt
from std.math.uutils import umod, ufloordiv, uceildiv
from std.sys.info import align_of, simd_width_of, size_of

import max.gpu.primitives.warp as warp
from std.algorithm import vectorize
from max.algorithm import map_reduce, mean, variance
from max.algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    sync_parallelize,
)
from max.algorithm.reduction import _simd_sum, _simd_sum_elementwise
from std.bit import log2_floor
from max.gpu import (
    WARP_SIZE,
    thread_idx,
    block_dim,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import (
    syncwarp,
    barrier,
)
from max.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from max.gpu.host.info import is_cpu, is_gpu
from max.gpu.memory import external_memory
from std.sys.info import is_apple_gpu
from max.gpu.primitives import block
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from layout import (
    Coord,
    CoordLike,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from layout.tile_layout import Layout
from std.memory import ThinAllocation, dealloc, unsafe_stack_allocation
from std.memory.alloc import Layout as AllocLayout
from max.runtime.asyncrt import parallelism_level
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.coord import ComptimeInt, CoordLike
from std.utils.index import Index, IndexList
from algorithm.rowwise import strided_load
from std.utils.static_tuple import StaticTuple
from std.utils.numerics import get_accum_type, max_finite, min_finite
from comm.rms_norm_fp8 import rms_norm_fused_fp8
from internal_utils.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize

# Free-form row-wise scaffolder (Row) + monoids.
from algorithm import rowwise
from algorithm.rowwise_types import RowCoord
from algorithm.reduce_op import ReduceMax, ReduceSum, Welford
from ._ragged_utils import get_batch_from_row_offsets
from .reshape import reshape
from .rope import _rope
from .shapes import _get_start_indices_of_nth_subvolume_static

comptime _APPLE_STATIC_SHMEM_MAX_BYTES = 32 * 1024
"""Maximum number of bytes that can be used on Apple GPUs (32K)."""

comptime _APPLE_STATIC_SHMEM_MAX_COUNT[
    T: AnyType
] = _APPLE_STATIC_SHMEM_MAX_BYTES // size_of[T]()
"""Maximum number of elements of type T that can fit in Apple's
static shared memory which is 32k."""


@always_inline
def block_reduce[
    dtype: DType, max_warps_per_block: Int
](val: Scalar[dtype]) -> Scalar[dtype]:
    var m2_shared = unsafe_stack_allocation[
        max_warps_per_block, dtype, address_space=.SHARED
    ]()
    var m2_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()

    var warp_m2 = warp.sum(val)

    var warp_id = warp_id[broadcast=True]()
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0:
        var block_m2 = Scalar[dtype](0)

        # Only read lanes corresponding to active warps to avoid
        # reading uninitialized shared memory.
        if lane_idx < ufloordiv(block_dim.x, WARP_SIZE):
            block_m2 = m2_shared[lane_idx]

        # On some GPUs, the warp-level reduction implicitly requires all lanes
        # to participate in the reduction. Otherwise, we would get deadlocks.
        block_m2 = warp.lane_group_sum[num_lanes=max_warps_per_block](block_m2)

        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


@always_inline
def block_reduce_dual_sum[
    dtype: DType, max_warps_per_block: Int
](val0: Scalar[dtype], val1: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype]
]:
    """Combined block reduction for two sums using only 2 barriers."""
    var shared0 = unsafe_stack_allocation[
        max_warps_per_block, dtype, address_space=.SHARED
    ]()
    var shared1 = unsafe_stack_allocation[
        max_warps_per_block, dtype, address_space=.SHARED
    ]()
    var broadcast0 = unsafe_stack_allocation[1, dtype, address_space=.SHARED]()
    var broadcast1 = unsafe_stack_allocation[1, dtype, address_space=.SHARED]()

    var warp_sum0 = warp.sum(val0)
    var warp_sum1 = warp.sum(val1)

    var warp_id = warp_id()
    var lane_idx = lane_id()

    if lane_idx == 0:
        shared0[warp_id] = warp_sum0
        shared1[warp_id] = warp_sum1
    barrier()

    if warp_id == 0:
        var block_sum0 = Scalar[dtype](0)
        var block_sum1 = Scalar[dtype](0)

        if lane_idx < ufloordiv(block_dim.x, WARP_SIZE):
            block_sum0 = shared0[lane_idx]
            block_sum1 = shared1[lane_idx]

        block_sum0 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            block_sum0
        )
        block_sum1 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            block_sum1
        )

        if lane_idx == 0:
            broadcast0[0] = block_sum0
            broadcast1[0] = block_sum1
    barrier()
    return (broadcast0[0], broadcast1[0])


# using numerically stable Welford online algorithm to compute single pass mean and variance
def welford_update[
    dtype: DType, //
](
    val: Scalar[dtype],
    mut mean: Scalar[dtype],
    mut m2: Scalar[dtype],
    mut count: Scalar[dtype],
):
    count += 1
    var d1 = val - mean
    mean += d1 / count
    var d2 = val - mean
    m2 += d1 * d2


def welford_combine[
    dtype: DType, //
](
    mean: Scalar[dtype],
    m2: Scalar[dtype],
    count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
):
    if count == 0:
        return
    var x_count = count + res_count
    var m = count / x_count
    var delta = mean - res_mean
    res_mean += delta * m
    res_m2 += m2 + delta * delta * res_count * m
    res_count = x_count


def welford_warp_reduce[
    dtype: DType, //
](
    thread_mean: Scalar[dtype],
    thread_m2: Scalar[dtype],
    thread_count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
):
    res_mean = thread_mean
    res_m2 = thread_m2
    res_count = thread_count

    comptime limit = log2_floor(WARP_SIZE)

    comptime for mask in reversed(range(limit)):
        var mean = warp.shuffle_down(res_mean, UInt32(1 << mask))
        var m2 = warp.shuffle_down(res_m2, UInt32(1 << mask))
        var count = warp.shuffle_down(res_count, UInt32(1 << mask))
        welford_combine(mean, m2, count, res_mean, res_m2, res_count)


def welford_block_all_reduce[
    dtype: DType, //
](
    thread_mean: Scalar[dtype],
    thread_m2: Scalar[dtype],
    thread_count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
):
    var mean_shared = unsafe_stack_allocation[
        WARP_SIZE, dtype, address_space=.SHARED
    ]()
    var m2_shared = unsafe_stack_allocation[
        WARP_SIZE, dtype, address_space=.SHARED
    ]()
    var count_shared = unsafe_stack_allocation[
        WARP_SIZE, dtype, address_space=.SHARED
    ]()
    var mean_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()
    var m2_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()
    var count_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()

    var warp_idx = warp_id()
    var lane_idx = lane_id()
    var warp_mean = Scalar[dtype]()
    var warp_m2 = Scalar[dtype]()
    var warp_count = Scalar[dtype]()
    welford_warp_reduce(
        thread_mean, thread_m2, thread_count, warp_mean, warp_m2, warp_count
    )
    barrier()

    if lane_idx == 0:
        mean_shared[warp_idx] = warp_mean
        m2_shared[warp_idx] = warp_m2
        count_shared[warp_idx] = warp_count
    barrier()

    if warp_idx == 0:
        if thread_idx.x < ufloordiv(block_dim.x, WARP_SIZE):
            warp_mean = mean_shared[lane_idx]
            warp_m2 = m2_shared[lane_idx]
            warp_count = count_shared[lane_idx]
        else:
            warp_mean = Scalar[dtype](0)
            warp_m2 = Scalar[dtype](0)
            warp_count = Scalar[dtype](0)
        syncwarp()
        var block_mean = Scalar[dtype](0)
        var block_m2 = Scalar[dtype](0)
        var block_count = Scalar[dtype](0)
        welford_warp_reduce(
            warp_mean, warp_m2, warp_count, block_mean, block_m2, block_count
        )
        if lane_idx == 0:
            mean_broadcast[0] = block_mean
            m2_broadcast[0] = block_m2
            count_broadcast[0] = block_count

    barrier()

    welford_combine(
        mean_broadcast[0],
        m2_broadcast[0],
        count_broadcast[0],
        res_mean,
        res_m2,
        res_count,
    )


@always_inline
def _rms_norm_warp_tiling_subkernel[
    dtype: DType,
    simd_width: SIMDLength,
    accum_type: DType,
    //,
    max_warps_per_block: Int,
    multiply_before_cast: Bool,
    rows_per_warp: Int = 1,
](
    row: Int,
    idx: Int,
    vec_data: SIMD[accum_type, simd_width],
    gamma_val: SIMD[dtype, simd_width],
    epsilon: Float32,
    weight_offset: Scalar[accum_type],
    num_cols: Int,
) -> SIMD[dtype, simd_width]:
    # To utilize simd vector load.
    var thread_m2: Scalar[accum_type] = (vec_data**2).reduce_add()

    var row_m2: Scalar[accum_type]
    comptime if rows_per_warp == 2:
        # Each half warp handles reduction for one row.
        row_m2 = warp.lane_group_sum[num_lanes=WARP_SIZE // 2](thread_m2)
    else:
        row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )

    var norm_factor = rsqrt(
        (row_m2 / Scalar[accum_type](num_cols)) + epsilon.cast[accum_type]()
    )
    var norm_val: SIMD[dtype, simd_width] = 0
    if idx < num_cols:
        comptime if multiply_before_cast:
            var gamma_accum = gamma_val.cast[accum_type]() + weight_offset
            norm_val = (vec_data * norm_factor * gamma_accum).cast[dtype]()
        else:
            norm_val = (vec_data * norm_factor).cast[dtype]() * (
                gamma_val + weight_offset.cast[dtype]()
            )

    return norm_val


@__name(t"rms_norm_gpu_warp_tiling_128_{dtype}_{multiply_before_cast}")
def rms_norm_gpu_warp_tiling_128[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    Engine: TensorEngine,
    //,
    simd_width: Int,
    warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDLength, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    gamma: TileTensor[dtype, LayoutType, origin, Engine=Engine],
    epsilon: Float32,
    weight_offset: Float32,
    num_rows: Int32,
    num_cols: Int32,
):
    var _num_rows = Int(num_rows)
    var _num_cols = Int(num_cols)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1
    comptime half_warp_size = WARP_SIZE // 2
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var weight_offset_accum = weight_offset.cast[accum_type]()

    var vec_data = SIMD[accum_type, simd_width](0)
    var tid = thread_idx.x
    # Each warp handles 2 rows, so total rows per block is warps_per_block * 2
    var block_row = block_idx.x * warps_per_block * 2
    var warp_id = ufloordiv(tid, WARP_SIZE)
    var sub_warp_id = ufloordiv(umod(tid, WARP_SIZE), half_warp_size)
    # Each warp handles 2 rows, offset by the block's base row
    var row = block_row + warp_id * 2 + Int(sub_warp_id)
    var local_tid = umod(tid, half_warp_size)
    var idx = local_tid * simd_width

    with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
        var gamma_val = SIMD[dtype, simd_width](0)
        if row < _num_rows and idx < _num_cols:
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()
            # Prefetch gamma before reduction to overlap load with compute.
            gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(idx)
            )

        var norm_val = _rms_norm_warp_tiling_subkernel[
            warps_per_block, multiply_before_cast, rows_per_warp=2
        ](
            row,
            idx,
            vec_data,
            gamma_val,
            epsilon,
            weight_offset_accum,
            _num_cols,
        )
        if row < _num_rows and idx < _num_cols:
            output_fn[simd_width, align](row, idx, norm_val)


# Barrier-free, SMEM-free warp-per-row RMSNorm: one warp owns a full row,
# `rows_per_block` warps per block, `warp.sum` reduction (no block barrier).
#
# `single_pass` selects the load strategy:
#   * single_pass=True: cache each lane's `chunks` exact-fit SIMD vectors in
#     registers across the reduction and normalize from registers (input read
#     ONCE). Requires exact fit (`chunks * WARP_SIZE * simd_width == num_cols`).
#     Fastest for narrow f32/bf16 rows of 1..4 vectors/lane (1.2-1.4x over
#     two-pass); beyond that the register cache spills.
#   * single_pass=False: two passes -- accumulate mean-of-squares, then reload
#     from L2 and normalize. Handles ragged tails and wider rows. `chunks` is
#     unused (pass any value, e.g. 1).
@__name(
    t"rms_norm_gpu_warp_per_row_{dtype}_{single_pass}_{chunks}_{multiply_before_cast}"
)
def rms_norm_gpu_warp_per_row[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    Engine: TensorEngine,
    //,
    simd_width: Int,
    rows_per_block: Int,
    single_pass: Bool,
    chunks: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDLength, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    gamma: TileTensor[dtype, LayoutType, origin, Engine=Engine],
    epsilon: Float32,
    weight_offset: Float32,
    num_rows: Int32,
    num_cols: Int32,
):
    var _num_rows = Int(num_rows)
    var _num_cols = Int(num_cols)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var tid = thread_idx.x
    var warp_in_block = ufloordiv(tid, WARP_SIZE)
    var lane = umod(tid, WARP_SIZE)
    var row = block_idx.x * rows_per_block + Int(warp_in_block)

    comptime stride = WARP_SIZE * simd_width

    comptime if single_pass:
        # Register-cached input chunks (accum precision), carried across the warp
        # reduction so the normalize pass needs no re-read.
        var vec_data = Array[SIMD[accum_type, simd_width], chunks](
            fill=SIMD[accum_type, simd_width](0)
        )

        with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
            # Single load pass: cache each chunk and accumulate mean-of-squares.
            var thread_m2 = Scalar[accum_type](0)
            if row < _num_rows:
                comptime for c in range(chunks):
                    var col = Int(lane) * simd_width + c * stride
                    vec_data[c] = input_fn[simd_width](row, col).cast[
                        accum_type
                    ]()
                    thread_m2 += (vec_data[c] ** 2).reduce_add()

            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](_num_cols)) + eps_accum
            )

            # Normalize from the cached registers (no reload).
            if row < _num_rows:
                comptime for c in range(chunks):
                    var col = Int(lane) * simd_width + c * stride
                    var gamma_val = gamma.load[
                        width=simd_width, alignment=align
                    ](Coord(col))
                    var norm_val: SIMD[dtype, simd_width]
                    comptime if multiply_before_cast:
                        var gamma_accum = (
                            gamma_val.cast[accum_type]() + weight_offset_accum
                        )
                        norm_val = (
                            vec_data[c] * norm_factor * gamma_accum
                        ).cast[dtype]()
                    else:
                        norm_val = (vec_data[c] * norm_factor).cast[dtype]() * (
                            gamma_val + weight_offset.cast[dtype]()
                        )
                    output_fn[simd_width, align](row, col, norm_val)
    else:
        with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
            # Pass 1: accumulate the per-thread mean-of-squares scalar only.
            var thread_m2 = Scalar[accum_type](0)
            if row < _num_rows:
                var col = Int(lane) * simd_width
                while col < _num_cols:
                    var v = input_fn[simd_width](row, col).cast[accum_type]()
                    thread_m2 += (v**2).reduce_add()
                    col += stride

            # Barrier-free, SMEM-free warp reduction (shuffle butterfly).
            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](_num_cols)) + eps_accum
            )

            # Pass 2: reload from L2 and normalize.
            if row < _num_rows:
                var col = Int(lane) * simd_width
                while col < _num_cols:
                    var v = input_fn[simd_width](row, col).cast[accum_type]()
                    var gamma_val = gamma.load[
                        width=simd_width, alignment=align
                    ](Coord(col))
                    var norm_val: SIMD[dtype, simd_width]
                    comptime if multiply_before_cast:
                        var gamma_accum = (
                            gamma_val.cast[accum_type]() + weight_offset_accum
                        )
                        norm_val = (v * norm_factor * gamma_accum).cast[dtype]()
                    else:
                        norm_val = (v * norm_factor).cast[dtype]() * (
                            gamma_val + weight_offset.cast[dtype]()
                        )
                    output_fn[simd_width, align](row, col, norm_val)
                    col += stride


# Rebuild a statically-typed `Coord` from a runtime `IndexList`, preserving the
# `Coord`'s static dims (`ComptimeInt`) and filling its dynamic leaves from the
# `IndexList`. Needed at the rms_norm/layer_norm call sites: the static-divisor
# `divmod` fold needs the `Coord` *type*, but a `Coord` does
# not survive the trip to a device, and captured into a `capturing` closure it
# fails the launch the same way. So the device closures capture the `IndexList`
# and rebuild the typed `Coord` in-kernel here.
@always_inline
def _index_list_to_typed_coord[
    element_types: TypeList[Trait=CoordLike, ...]
](witness: Coord[*element_types], il: IndexList[witness.rank]) -> Coord[
    *element_types
]:
    # Default-construct sets every static dim to its `ComptimeInt` literal.
    var res = Coord[*element_types]()

    comptime for i in range(witness.rank):
        comptime ElemT = element_types[i]
        comptime if not ElemT.is_static_value:
            res[i] = rebind[ElemT](Scalar[ElemT.DTYPE](il[i]))

    return res


# SM100 (B200) primary target; portable (only uses `block_reduce`, warp
# shuffle, and a per-thread grid-stride column loop — no arch intrinsics).
#
# Warp-tiling RMSNorm tuned to cut instruction overhead. Three structural
# choices vs the old single-chunk-per-thread form (which was instruction-issue
# bound on B200, not bandwidth bound -- high SM issue but low L2/HBM
# throughput):
#
#   1. The rank-N row -> base-coords translation
#      (`_get_start_indices_of_nth_subvolume`) is hoisted and run ONCE per
#      thread, then reused for every chunk's load AND store. The old 2D
#      wrappers ran that divmod chain twice per thread (once in `input_fn_2d`,
#      once in `output_fn_2d`); we now take the rank-N `input_fn`/`output_fn`
#      directly and only mutate `base[rank - 1]` per chunk.
#   2. Each thread processes `chunks_per_thread` independent vector chunks
#      (unrolled LDGs cached in registers), so the launch uses fewer threads
#      per block -> fewer warps -> a cheaper two-barrier `block_reduce`, and
#      the loads pipeline (ILP). Data stays in registers between the reduction
#      and the normalize pass (still a single global read).
#   3. When `exact_fit` (block_dim * simd_width * chunks_per_thread == cols),
#      every thread is fully active, so the per-chunk `col < num_cols` guards
#      (the ISETP/SEL bloat) are dropped at comptime. A guarded variant
#      (`exact_fit=False`) handles ragged tails.
@__name(
    t"rms_norm_gpu_warp_tiling_{dtype}_{chunks_per_thread}_{exact_fit}_{multiply_before_cast}"
)
def rms_norm_gpu_warp_tiling[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    rank: Int,
    Engine: TensorEngine,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    chunks_per_thread: Int,
    exact_fit: Bool,
    input_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDLength, alignment: Int](
        Coord, SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    shape: IndexList[rank],
    gamma: TileTensor[dtype, LayoutType, origin, Engine=Engine],
    epsilon: Float32,
    weight_offset: Float32,
    num_cols: Int32,
):
    var _num_cols = Int(num_cols)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var bdim = Int(block_dim.x)

    # Hoist the rank-N row translation ONCE; reuse the base for load and store.
    var base = _get_start_indices_of_nth_subvolume(row, shape)

    # Per-chunk register-cached input (in accum precision) and gamma weights,
    # carried across the reduction so the normalize pass needs no re-read.
    var vec_data = Array[SIMD[accum_type, simd_width], chunks_per_thread](
        fill=SIMD[accum_type, simd_width](0)
    )
    var gamma_val = Array[SIMD[dtype, simd_width], chunks_per_thread](
        fill=SIMD[dtype, simd_width](0)
    )

    with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
        var thread_m2 = Scalar[accum_type](0)

        comptime for c in range(chunks_per_thread):
            var col = (c * bdim + tid) * simd_width
            comptime if exact_fit:
                base[rank - 1] = col
                vec_data[c] = input_fn[simd_width](Coord(base)).cast[
                    accum_type
                ]()
                gamma_val[c] = gamma.load[width=simd_width, alignment=align](
                    Coord(col)
                )
                thread_m2 += (vec_data[c] ** 2).reduce_add()
            else:
                if col < _num_cols:
                    base[rank - 1] = col
                    vec_data[c] = input_fn[simd_width](Coord(base)).cast[
                        accum_type
                    ]()
                    gamma_val[c] = gamma.load[
                        width=simd_width, alignment=align
                    ](Coord(col))
                    thread_m2 += (vec_data[c] ** 2).reduce_add()

        var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](num_cols)) + eps_accum
        )

        comptime for c in range(chunks_per_thread):
            var col = (c * bdim + tid) * simd_width

            @always_inline
            @__parameter
            def _normalize() -> SIMD[dtype, simd_width]:
                comptime if multiply_before_cast:
                    var gamma_accum = (
                        gamma_val[c].cast[accum_type]() + weight_offset_accum
                    )
                    return (vec_data[c] * norm_factor * gamma_accum).cast[
                        dtype
                    ]()
                else:
                    return (vec_data[c] * norm_factor).cast[dtype]() * (
                        gamma_val[c] + weight_offset.cast[dtype]()
                    )

            comptime if exact_fit:
                base[rank - 1] = col
                output_fn[simd_width, align](Coord(base), _normalize())
            else:
                if col < _num_cols:
                    base[rank - 1] = col
                    output_fn[simd_width, align](Coord(base), _normalize())


@always_inline
def _rms_norm_gpu_block_subkernel[
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDLength, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    num_cols: Int,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    var thread_m2 = Scalar[accum_type](0)
    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width
        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            thread_m2 += (vec_data**2).reduce_add()

    var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_m2
    )
    var norm_factor = rsqrt((row_m2 / Scalar[accum_type](num_cols)) + eps_accum)

    # Need a pass again to perform in place normalization.
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width

        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            var norm_val: SIMD[dtype, simd_width]
            var gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(offset)
            )

            if multiply_before_cast:
                var gamma_accum = (
                    gamma_val.cast[accum_type]() + weight_offset_accum
                )
                norm_val = (vec_data * norm_factor * gamma_accum).cast[dtype]()
            else:
                norm_val = (vec_data * norm_factor).cast[dtype]() * (
                    gamma_val + weight_offset.cast[dtype]()
                )

            output_fn[simd_width, align](row, offset, norm_val)


@__name(t"rms_norm_gpu_block_{dtype}_{multiply_before_cast}")
def rms_norm_gpu_block[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    Engine: TensorEngine,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDLength, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    gamma: TileTensor[dtype, LayoutType, origin, Engine=Engine],
    epsilon: Float32,
    weight_offset: Float32,
    num_cols: Int32,
):
    var _num_cols = Int(num_cols)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
        _rms_norm_gpu_block_subkernel[
            simd_width,
            max_warps_per_block,
            input_fn,
            output_fn,
            multiply_before_cast,
        ](gamma, epsilon, weight_offset.cast[dtype](), _num_cols)


def rms_norm_gpu[
    dtype: DType,
    //,
    rank: Int,
    input_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDLength, alignment: Int](
        Coord, SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.OVERLAP_AT_BEGINNING,
](
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    # `shape` arrives as a `Coord`, with statically-known outer dims encoded in
    # its type, and the lambdas are `Coord`-form all the way down to the
    # kernels. `shape_il` materializes the runtime `IndexList` once, for the
    # two things that cannot take a coord: the warp-tiling kernel's shape
    # argument, and the row-translation wrappers' capture (see
    # `_index_list_to_typed_coord` for why).
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    if rank == 0:
        return

    var shape_il = rebind[IndexList[rank]](coord_to_index_list(shape))

    # Derive the number of columns from the `gamma` input as this value may be
    # statically known.
    var cols = Int(gamma.dim[0]())

    if cols == 0:
        return

    var rows = Int(shape.product()) // cols

    # A rank owns an empty shard when rows < TP group size (bs=1 decode at TP4)
    # and `enqueue_function` rejects the zero `grid_dim` every launch derives.
    if rows == 0:
        return

    # The 2D wrappers translate each flattened `(row, col)` back to the original
    # n-D coordinate. The row -> n-D decomposition divides by the outer dims; on
    # the static-shape path those divisors are the `ComptimeInt` dims carried in
    # `type_of(shape)` (a `Coord`), so the per-row `divmod` strength-reduces to
    # magic-multiply + shift instead of the runtime Newton-reciprocal `IDIV` that
    # a plain `IndexList` divisor forces. Dynamic dims fall back to the runtime
    # value in `shape_il`, so this path is behavior-identical to the pre-migration
    # `_get_start_indices_of_nth_subvolume` form for non-static shapes.
    #
    # `@__copy_capture(shape_il)` is required: these wrappers are embedded into
    # GPU kernels as `capturing` closures, and a captured *local* `var` (unlike
    # a function parameter, which the pre-migration code captured directly) is
    # not carried to the device without an explicit copy-capture. Without it the
    # rank-N `_get_start_indices_of_nth_subvolume` divmod reads garbage outer
    # dims on device (rank-2 is unaffected since its outer translation is
    # trivial; rank>=3 produces wrong results / launch failures).
    #
    # The fold needs the `Coord` *type*, but a `Coord` captured here fails the
    # launch (see `_index_list_to_typed_coord`), so the capture is the
    # `IndexList` and the typed coord is rebuilt in-kernel: `type_of(shape)()`
    # supplies the static dims at comptime, `shape_il` the dynamic leaves.
    @__copy_capture(shape_il)
    @__parameter
    @always_inline
    def output_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        var shape_witness = type_of(shape)()
        var shape_coord = _index_list_to_typed_coord(
            shape_witness,
            rebind[IndexList[shape_witness.rank]](shape_il),
        )
        var indices = _get_start_indices_of_nth_subvolume_static(
            row, shape_coord
        )
        indices[rank - 1] = col
        output_fn[simd_width, alignment](Coord(indices), val)

    @__copy_capture(shape_il)
    @__parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        var shape_witness = type_of(shape)()
        var shape_coord = _index_list_to_typed_coord(
            shape_witness,
            rebind[IndexList[shape_witness.rank]](shape_il),
        )
        var indices = _get_start_indices_of_nth_subvolume_static(
            row, shape_coord
        )
        indices[rank - 1] = col
        return input_fn[simd_width](Coord(indices))

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime sm_version = ctx.default_device_info.version
    comptime sm_count = ctx.default_device_info.sm_count
    comptime warp_per_row_rows_per_block = 8
    var warp_per_row_min_grid = 3 * sm_count
    var warp_per_row_region = (
        ceildiv(rows, warp_per_row_rows_per_block) >= warp_per_row_min_grid
    )
    # Conservative floor: below one block per SM, chunking can't raise
    # occupancy and only reassociates the mean-of-squares reduction (the real
    # crossover is higher and width-dependent; this sits safely below it).
    var enough_rows_to_chunk = rows >= sm_count

    var grid_dim = rows
    var block_dim = min(
        align_up(ceildiv(cols, simd_width), WARP_SIZE),
        WARP_SIZE * max_warps_per_block,
    )

    # Warp-tiling launch geometry for a given per-thread (`eff_simd`, `chunks`):
    # one block per row, `_wt_threads_per_block` threads (rounded to whole
    # warps, capped at the device max), each owning `eff_simd * chunks` columns.
    # Single source of truth for both the launcher and the warp-per-row gate
    # below.
    @__parameter
    @always_inline
    def _wt_threads_per_block[eff_simd: Int, chunks: Int]() -> Int:
        var threads = ceildiv(ceildiv(cols, eff_simd), chunks)
        return min(
            align_up(threads, WARP_SIZE),
            WARP_SIZE * max_warps_per_block,
        )

    # `exact` means every thread is fully active (the block tiles the row with
    # no ragged tail), so the unguarded kernel can be used.
    @__parameter
    @always_inline
    def _wt_exact[eff_simd: Int, chunks: Int]() -> Bool:
        return (
            _wt_threads_per_block[eff_simd, chunks]() * eff_simd * chunks
        ) == cols

    # Within the warp-tiling column range, warp-tiling (register-cached chunks)
    # beats the barrier-free warp-per-row kernel on SM100 (B200) whenever it
    # tiles the row exactly *and* the row is wide enough to amortize the
    # inter-warp block reduction: a measured 1.1-1.4x at 2048..8192 cols.
    # Warp-per-row wins in two cases, both of which must stay on it:
    #   1. ragged tail (e.g. 8192x2880, cols not a clean multiple of the
    #      per-thread tile -> wasted threads + per-chunk bounds guards), and
    #   2. narrow rows (cols <= 1024), where the one-warp-per-row kernel has no
    #      inter-warp barrier to pay and warp-tiling's block reduce dominates
    #      (measured: 8192x256 1.9x, 8192x512 1.7x, 8192x1024 1.3x slower under
    #      warp-tiling).
    # So prefer warp-tiling only for exact-fit rows past the narrow-row floor;
    # everything else keeps warp-per-row. Uses the native-width chunk count the
    # dispatch below would pick (1 up to one warp-row, 2 up to two, else 4) and
    # the shared geometry.
    #
    # This is only consulted to gate warp-per-row, which itself only runs for
    # `cols <= WARP_SIZE * simd_width * max_warps_per_block`. Within that range
    # the dispatch always launches at native `simd_width` with these same chunk
    # counts, so the gate matches the kernel that runs. The wider-row branch
    # below (`simd_width * 2`) lies entirely past the warp-per-row bound, so the
    # native-width value computed here is never read against it.
    #
    # `warp_tiling_min_cols` is the narrow-row floor: warp-tiling is preferred
    # only above it. 1024 = WARP_SIZE * simd_width * 4 (= one max-width
    # per-thread tile across a single warp) is the measured crossover on B200
    # bf16 -- the largest exact-fit width that still loses to warp-per-row.
    var warp_tiling_min_cols = WARP_SIZE * simd_width * 4
    var warp_tiling_exact_fit: Bool
    if cols <= (WARP_SIZE * simd_width):
        warp_tiling_exact_fit = _wt_exact[simd_width, 1]()
    elif cols <= (WARP_SIZE * simd_width * 2):
        warp_tiling_exact_fit = _wt_exact[simd_width, 2]()
    else:
        warp_tiling_exact_fit = _wt_exact[simd_width, 4]()
    var warp_tiling_exact = warp_tiling_exact_fit and (
        cols > warp_tiling_min_cols
    )

    # Launch the multi-chunk warp-tiling kernel. `exact_fit` (every thread
    # fully active, no ragged tail) is decided at runtime and selects the
    # unguarded instantiation.
    @__parameter
    @always_inline
    def _launch_warp_tiling[eff_simd: Int, chunks: Int]() raises:
        var threads_per_block = _wt_threads_per_block[eff_simd, chunks]()
        var exact = _wt_exact[eff_simd, chunks]()

        @__parameter
        @always_inline
        def _enqueue[exact_fit: Bool]() raises:
            comptime kernel = rms_norm_gpu_warp_tiling[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                Engine=gamma.Engine,
                rank=rank,
                eff_simd,
                max_warps_per_block,
                chunks,
                exact_fit,
                input_fn,
                output_fn,
                multiply_before_cast=multiply_before_cast,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel](
                shape_il.canonicalize(),
                gamma,
                epsilon.cast[.float32](),
                weight_offset.cast[.float32](),
                Int32(cols),
                grid_dim=rows,
                block_dim=threads_per_block,
                attributes=pdl_launch_attributes(pdl_level),
            )

        if exact:
            _enqueue[True]()
        else:
            _enqueue[False]()

    # _rms_norm_input_alignment trusts gates like this one. Loosen or
    # remove it and update that function too.
    if cols % simd_width == 0:
        # When the number of columns are less enough that they can be placed in
        # registers we do warp tiling which is a single pass to do mean/var
        # computation and normalization.
        if cols <= 128 and dtype == .bfloat16:
            # Experimentally determined to be the best - tapers off at 2.
            comptime warps_per_block = 2
            # Each warp handles 2 rows, so total rows per block is warps_per_block * 2.
            block_dim = warps_per_block * WARP_SIZE
            grid_dim = ceildiv(rows, warps_per_block * 2)

            comptime kernel = rms_norm_gpu_warp_tiling_128[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                Engine=gamma.Engine,
                simd_width,
                warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel](
                gamma,
                epsilon.cast[.float32](),
                weight_offset.cast[.float32](),
                Int32(rows),
                Int32(cols),
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
        elif (
            cols >= 128
            and cols <= (WARP_SIZE * simd_width * max_warps_per_block)
            and warp_per_row_region
            and not warp_tiling_exact
        ):
            comptime rows_per_block = warp_per_row_rows_per_block
            block_dim = rows_per_block * WARP_SIZE
            grid_dim = ceildiv(rows, rows_per_block)

            # Single-pass register-cached warp-per-row for narrow, exact-fit
            # rows. Keeps the 8-rows/block high-occupancy geometry of the
            # two-pass kernel but loads each lane's row chunks ONCE (cached in
            # registers across the `warp.sum`) instead of reloading + resquaring
            # in pass 2.
            comptime sp_stride = WARP_SIZE * simd_width
            var sp_chunks = cols // sp_stride
            if (
                dtype in (DType.float32, DType.bfloat16)
                and (cols % sp_stride == 0)
                and (sp_chunks >= 1)
                and (sp_chunks <= 4)
            ):
                comptime for cc in range(1, 5):
                    if sp_chunks == cc:
                        comptime kernel = rms_norm_gpu_warp_per_row[
                            mut=gamma.mut,
                            LayoutType=gamma.LayoutType,
                            origin=gamma.origin,
                            Engine=gamma.Engine,
                            simd_width,
                            rows_per_block,
                            True,
                            cc,
                            input_fn_2d,
                            output_fn_2d,
                            multiply_before_cast=multiply_before_cast,
                            pdl_level=pdl_level,
                        ]
                        ctx.enqueue_function[kernel](
                            gamma,
                            epsilon.cast[.float32](),
                            weight_offset.cast[.float32](),
                            Int32(rows),
                            Int32(cols),
                            grid_dim=grid_dim,
                            block_dim=block_dim,
                            attributes=pdl_launch_attributes(pdl_level),
                        )
            else:
                comptime kernel = rms_norm_gpu_warp_per_row[
                    mut=gamma.mut,
                    LayoutType=gamma.LayoutType,
                    origin=gamma.origin,
                    Engine=gamma.Engine,
                    simd_width,
                    rows_per_block,
                    False,
                    1,
                    input_fn_2d,
                    output_fn_2d,
                    multiply_before_cast=multiply_before_cast,
                    pdl_level=pdl_level,
                ]
                ctx.enqueue_function[kernel](
                    gamma,
                    epsilon.cast[.float32](),
                    weight_offset.cast[.float32](),
                    Int32(rows),
                    Int32(cols),
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(pdl_level),
                )
        elif cols <= (WARP_SIZE * simd_width * max_warps_per_block):
            # CDNA4 (MI355X): when there are enough rows to keep the GPU busy,
            # use a 2x-wider per-thread SIMD so each row's block needs half the
            # warps. This halves the inter-warp block reduction cost and
            # doubles blocks-per-CU, lifting achieved HBM bandwidth by ~15-30%
            # on prefill-sized shapes. It is gated on row count because the
            # smaller block lowers total occupancy when rows are few (a net loss
            # below ~8x the CU count).
            comptime sw_wide = simd_width * 2
            comptime widen_ok = sm_version == "CDNA4"
            var enough_rows = rows >= 8 * sm_count
            if widen_ok and enough_rows and cols % sw_wide == 0:
                _launch_warp_tiling[sw_wide, 1]()
            elif not enough_rows_to_chunk:
                _launch_warp_tiling[simd_width, 1]()
            else:
                # Narrow rows: a single full-width pass keeps the block small;
                # split into independent chunks (ILP + smaller block_reduce)
                # once there are enough columns to fill ~2+ chunks/thread.
                if cols <= (WARP_SIZE * simd_width):
                    _launch_warp_tiling[simd_width, 1]()
                elif cols <= (WARP_SIZE * simd_width * 2):
                    _launch_warp_tiling[simd_width, 2]()
                else:
                    _launch_warp_tiling[simd_width, 4]()
        elif (
            cols <= (WARP_SIZE * (simd_width * 2) * max_warps_per_block)
            and cols % (simd_width * 2) == 0
        ):
            # Wider rows: double the vector width and, for high-row launches,
            # split into chunks.
            if not enough_rows_to_chunk:
                _launch_warp_tiling[simd_width * 2, 1]()
            elif cols <= (WARP_SIZE * simd_width * 2 * 2):
                _launch_warp_tiling[simd_width * 2, 2]()
            else:
                _launch_warp_tiling[simd_width * 2, 4]()
        else:
            comptime kernel = rms_norm_gpu_block[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                Engine=gamma.Engine,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel](
                gamma,
                epsilon.cast[.float32](),
                weight_offset.cast[.float32](),
                Int32(cols),
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
    else:
        comptime kernel = rms_norm_gpu_block[
            mut=gamma.mut,
            LayoutType=gamma.LayoutType,
            origin=gamma.origin,
            Engine=gamma.Engine,
            1,
            max_warps_per_block,
            input_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
            pdl_level=pdl_level,
        ]
        ctx.enqueue_function[kernel](
            gamma,
            epsilon.cast[.float32](),
            weight_offset.cast[.float32](),
            Int32(cols),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


def _sum_to_mean[
    dtype: DType, //
](sum_val: Scalar[dtype], n: Int) -> Scalar[dtype]:
    comptime if dtype.is_integral():
        return sum_val // Scalar[dtype](n)
    return sum_val / Scalar[dtype](n)


def rms_norm_cpu[
    dtype: DType,
    //,
    input_fn: def[width: Int](Int, Int) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    out_shape: Coord,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1
    comptime assert out_shape.rank == 2, "out_shape must be rank 2"

    comptime simd_width = simd_width_of[dtype]()

    var num_rows = Int(out_shape[0].value())
    var num_cols = Int(out_shape[1].value())

    var simd_loop_end = align_down(num_cols, simd_width)
    comptime intermediate_type = get_accum_type[dtype]()

    # PyTorch converts the input to float32 before computing the RMS norm
    # https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L76
    for var row in range(num_rows):
        var sum_simd = SIMD[intermediate_type, simd_width]()
        for col in range(0, simd_loop_end, simd_width):
            sum_simd += (
                input_fn[simd_width](row, col).cast[intermediate_type]() ** 2
            )

        var sum_val = sum_simd.reduce_add()
        for col in range(simd_loop_end, num_cols):
            sum_val += input_fn[1](row, col).cast[intermediate_type]() ** 2

        var mean_val = _sum_to_mean(sum_val, num_cols)
        var norm_factor = rsqrt(mean_val + epsilon.cast[intermediate_type]())

        def _normalize[simd_width: Int](col: Int) {gamma, weight_offset, mut}:
            var input_val = input_fn[simd_width](row, col).cast[
                intermediate_type
            ]()
            var gamma_val = gamma.load[width=simd_width, alignment=1](
                Coord(col)
            )
            var norm_val: SIMD[dtype, simd_width]

            if multiply_before_cast:
                var gamma_offset = gamma_val + weight_offset.cast[dtype]()
                norm_val = (input_val * norm_factor).cast[
                    dtype
                ]() * gamma_offset
            else:
                norm_val = (input_val * norm_factor).cast[dtype]() * (
                    gamma_val + weight_offset.cast[dtype]()
                )

            output_fn[simd_width, 1](row, col, norm_val)

        vectorize[simd_width](num_cols, _normalize)


def rms_norm_cpu[
    dtype: DType,
    //,
    input_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDLength, alignment: Int](
        Coord, SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: Optional[DeviceContext] = None,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime rank = shape.rank

    var last_dim = Int(shape[rank - 1].value())
    var prod_all_but_last_dim = Int(shape.product()) // last_dim

    var num_workers = min(parallelism_level(ctx), prod_all_but_last_dim)
    var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

    def task_func(
        thread_id: Int,
    ) {
        var chunk_size,
        var prod_all_but_last_dim,
        var last_dim,
        var epsilon,
        var weight_offset,
        imm,
    }:
        var num_rows = min(
            chunk_size, prod_all_but_last_dim - thread_id * chunk_size
        )
        var row_idx = thread_id * chunk_size

        @__copy_capture(row_idx)
        @__parameter
        @always_inline
        def output_fn_2d[
            simd_width: SIMDLength, alignment: Int
        ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
            # Translate a given 2D index back to the original n-D tensor.
            var indices = _get_start_indices_of_nth_subvolume_static(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            output_fn[simd_width, alignment](Coord(indices), val)

        @__copy_capture(row_idx)
        @__parameter
        @always_inline
        def input_fn_2d[
            simd_width: Int
        ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
            # Translate a given 2D index back to the original n-D tensor.
            var indices = _get_start_indices_of_nth_subvolume_static(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            return input_fn[simd_width](Coord(indices))

        rms_norm_cpu[
            input_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ](
            gamma,
            epsilon,
            weight_offset,
            out_shape=Coord(num_rows, last_dim),
        )

    sync_parallelize(task_func, num_workers, ctx)


@always_inline
def _rms_norm_input_alignment[
    dtype: DType, width: Int, target: StaticString
]() -> Int:
    """The alignment an rms_norm input_fn can claim for a load of `width`.

    Sound only on GPU, whose dispatchers prove `cols % width == 0` before
    requesting `width > 1`. CPU always claims 1: its vectorized loop runs
    regardless of `num_cols % simd_width`.
    """
    comptime if is_cpu[target]():
        return 1
    else:
        return align_of[SIMD[dtype, width]]()


@always_inline
def _rms_norm_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: def[width: Int, alignment: Int](Coord) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDLength, alignment: Int](
        Coord, SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    # Everything here is `Coord`-form: the rank check, the empty guard and both
    # target paths read the coord directly. Callers whose lambdas need runtime
    # index subscripts (`kv_cache.mojo`) wrap theirs to `Coord`-form at the call
    # site (see `coord_to_index_list`).
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"

    var reduce_dim_size = Int(shape[rank - 1].value())

    # Note: we only support reduction along the last dimension
    if Int(gamma.layout.shape[0]().value()) != reduce_dim_size:
        raise Error(
            "Gamma size "
            + String(gamma.layout.shape[0]().value())
            + " does not match dimension of reduction "
            + String(reduce_dim_size)
            + "."
        )

    if Int(shape.product()) == 0:
        # Nothing to do.
        return

    @__parameter
    @always_inline
    def input_fn_target[width: Int](coords: Coord) -> SIMD[dtype, width]:
        comptime align = _rms_norm_input_alignment[dtype, width, target]()
        return input_0_fn[width, align](coords)

    comptime if is_cpu[target]():
        rms_norm_cpu[
            input_fn_target,
            output_fn,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma,
            epsilon,
            weight_offset,
            Optional[DeviceContext](ctx),
        )
    elif is_gpu[target]():
        rms_norm_gpu[
            rank,
            input_fn_target,
            output_fn,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma,
            epsilon,
            weight_offset,
            ctx,
        )
    else:
        comptime assert False, "unsupported target " + target


# ===----------------------------------------------------------------------=== #
# Fused Q/K RMSNorm apply: kernel + CPU/GPU entry points + dispatcher
# ===----------------------------------------------------------------------=== #
#
# SM100 (B200) primary target; portable to SM90 / CDNA4 / Apple (pure
# elementwise apply, no arch-specific intrinsics, no block reduction).
#
# Given the already-reduced per-row statistics `qk_var` of shape `[M, 2]`
# (col 0 = mean(q^2), col 1 = mean(k^2), float32), and per-column float32
# scales `gamma_q [Nq]` / `gamma_k [Nk]`, this applies the QK-RMSNorm scale to
# Q `[M, Nq]` and K `[M, Nk]` in a SINGLE launch:
#
#     rs_q = rsqrt(qk_var[m, 0] + epsilon)            # float32
#     q_out[m, c] = cast( (cast(q[m,c], f32) * rs_q) * gamma_q[c], out_dtype )
#     rs_k = rsqrt(qk_var[m, 1] + epsilon)
#     k_out[m, c] = cast( (cast(k[m,c], f32) * rs_k) * gamma_k[c], out_dtype )
#
# The grouping is `((x_f32 * rs) * gamma)` then cast (NOT `x_f32 * (rs*gamma)`)
# to bit-match the unfused graph it replaces. This fuses the ~7 tiny
# elementwise/View kernels of the QK-norm apply chain into one launch, keeping
# the grid tiny for small-M decode (grid = (rows, 2); e.g. M=16 -> 32 blocks).


@__name(t"apply_qk_rms_norm_gpu_block_{in_dtype}_{out_dtype}")
def apply_qk_rms_norm_gpu_block[
    in_dtype: DType,
    out_dtype: DType,
    q_out_mut: Bool,
    q_out_layout: TensorLayout,
    q_out_origin: Origin[mut=q_out_mut],
    q_out_engine: TensorEngine,
    k_out_mut: Bool,
    k_out_layout: TensorLayout,
    k_out_origin: Origin[mut=k_out_mut],
    k_out_engine: TensorEngine,
    gamma_q_mut: Bool,
    gamma_q_layout: TensorLayout,
    gamma_q_origin: Origin[mut=gamma_q_mut],
    gamma_q_engine: TensorEngine,
    gamma_k_mut: Bool,
    gamma_k_layout: TensorLayout,
    gamma_k_origin: Origin[mut=gamma_k_mut],
    gamma_k_engine: TensorEngine,
    var_mut: Bool,
    var_layout: TensorLayout,
    var_origin: Origin[mut=var_mut],
    var_engine: TensorEngine,
    q_layout: TensorLayout,
    q_origin: Origin,
    q_engine: TensorEngine,
    k_layout: TensorLayout,
    k_origin: Origin,
    k_engine: TensorEngine,
    //,
    simd_width: Int,
](
    q_out: TileTensor[
        out_dtype, q_out_layout, q_out_origin, Engine=q_out_engine
    ],
    k_out: TileTensor[
        out_dtype, k_out_layout, k_out_origin, Engine=k_out_engine
    ],
    gamma_q: TileTensor[
        .float32, gamma_q_layout, gamma_q_origin, Engine=gamma_q_engine
    ],
    gamma_k: TileTensor[
        .float32, gamma_k_layout, gamma_k_origin, Engine=gamma_k_engine
    ],
    qk_var: TileTensor[.float32, var_layout, var_origin, Engine=var_engine],
    q: TileTensor[in_dtype, q_layout, q_origin, Engine=q_engine],
    k: TileTensor[in_dtype, k_layout, k_origin, Engine=k_engine],
    epsilon: Float32,
    q_cols: Int32,
    k_cols: Int32,
) where (q_out_mut and k_out_mut):
    """Fused per-element QK-RMSNorm apply for Q and K in a single launch.

    The grid is 2D: `block_idx.x` selects the row and `block_idx.y` selects the
    operand (0 = Q, 1 = K). Each block owns one (row, operand) and threads
    grid-stride across that operand's columns, applying `((x * rs) * gamma)`.
    All operands (`q [M, Nq]`, `k [M, Nk]`, `gamma_q [Nq]`, `gamma_k [Nk]`,
    `qk_var [M, 2]`, and the outputs `q_out [M, Nq]` / `k_out [M, Nk]`) are
    loaded/stored directly from their `TileTensor`s, matching the in-file
    rms_norm `gamma.load[...]` idiom.
    """
    var _q_cols = Int(q_cols)
    var _k_cols = Int(k_cols)
    comptime assert q.flat_rank == 2, "q must have rank 2"
    comptime assert k.flat_rank == 2, "k must have rank 2"
    comptime align = align_of[SIMD[.float32, simd_width]]()

    var tid = thread_idx.x
    var row = block_idx.x
    # block_idx.y is uniform across the block, so this branch never diverges.
    var is_k = block_idx.y == 1
    var num_cols = _k_cols if is_k else _q_cols

    with PDL():
        # rsqrt of the (already cross-rank reduced) per-row mean of squares.
        var rs = rsqrt(
            qk_var.load[width=1](Coord(Index(Int(row), Int(block_idx.y))))
            + epsilon
        )

        # Each block owns a single (row, operand); threads grid-stride the cols.
        for x in range(ceildiv(ceildiv(num_cols, simd_width), block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                if is_k:
                    var xf = k.load[width=simd_width](
                        Coord(Index(Int(row), offset))
                    ).cast[.float32]()
                    var g = gamma_k.load[width=simd_width, alignment=align](
                        Coord(offset)
                    )
                    k_out.store[width=simd_width](
                        Coord(Index(Int(row), offset)),
                        ((xf * rs) * g).cast[out_dtype](),
                    )
                else:
                    var xf = q.load[width=simd_width](
                        Coord(Index(Int(row), offset))
                    ).cast[.float32]()
                    var g = gamma_q.load[width=simd_width, alignment=align](
                        Coord(offset)
                    )
                    q_out.store[width=simd_width](
                        Coord(Index(Int(row), offset)),
                        ((xf * rs) * g).cast[out_dtype](),
                    )


def apply_qk_rms_norm_gpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, .float32, ...],
    gamma_k: TileTensor[mut=False, .float32, ...],
    qk_var: TileTensor[mut=False, .float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Float32,
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    """Launches the fused Q/K RMSNorm apply: one launch, grid (rows, 2).

    `block_idx.y` selects Q (0) or K (1). Block dim is sized for the wider of
    the two operands; the narrower operand simply leaves trailing threads idle.
    """
    if rows == 0 or (q_cols == 0 and k_cols == 0):
        return

    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    # 2D grid: x = row, y = operand (0 = Q, 1 = K). One block per (row, operand)
    # keeps the grid tiny for small-M decode (e.g. M=16 -> 32 blocks).
    var grid_dim = (rows, 2)
    var max_cols = max(q_cols, k_cols)

    if q_cols % simd_width == 0 and k_cols % simd_width == 0:
        # Vectorized loads/stores; size threads for the wider operand.
        var block_dim = min(
            align_up(ceildiv(max_cols, simd_width), WARP_SIZE),
            WARP_SIZE * max_warps_per_block,
        )
        comptime kernel = apply_qk_rms_norm_gpu_block[
            in_dtype=q.dtype,
            out_dtype=out_dtype,
            q_out_mut=q_out.mut,
            q_out_layout=q_out.LayoutType,
            q_out_origin=q_out.origin,
            q_out_engine=q_out.Engine,
            k_out_mut=k_out.mut,
            k_out_layout=k_out.LayoutType,
            k_out_origin=k_out.origin,
            k_out_engine=k_out.Engine,
            gamma_q_mut=gamma_q.mut,
            gamma_q_layout=gamma_q.LayoutType,
            gamma_q_origin=gamma_q.origin,
            gamma_q_engine=gamma_q.Engine,
            gamma_k_mut=gamma_k.mut,
            gamma_k_layout=gamma_k.LayoutType,
            gamma_k_origin=gamma_k.origin,
            gamma_k_engine=gamma_k.Engine,
            var_mut=qk_var.mut,
            var_layout=qk_var.LayoutType,
            var_origin=qk_var.origin,
            var_engine=qk_var.Engine,
            q_layout=q.LayoutType,
            q_origin=q.origin,
            q_engine=q.Engine,
            k_layout=k.LayoutType,
            k_origin=k.origin,
            k_engine=k.Engine,
            simd_width=simd_width,
        ]
        ctx.enqueue_function[kernel](
            q_out,
            k_out,
            gamma_q,
            gamma_k,
            qk_var,
            q,
            k,
            epsilon,
            Int32(q_cols),
            Int32(k_cols),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        # General N (incl. non-multiple of vector width): scalar loads/stores.
        var block_dim = min(
            align_up(max_cols, WARP_SIZE),
            WARP_SIZE * max_warps_per_block,
        )
        comptime kernel = apply_qk_rms_norm_gpu_block[
            in_dtype=q.dtype,
            out_dtype=out_dtype,
            q_out_mut=q_out.mut,
            q_out_layout=q_out.LayoutType,
            q_out_origin=q_out.origin,
            q_out_engine=q_out.Engine,
            k_out_mut=k_out.mut,
            k_out_layout=k_out.LayoutType,
            k_out_origin=k_out.origin,
            k_out_engine=k_out.Engine,
            gamma_q_mut=gamma_q.mut,
            gamma_q_layout=gamma_q.LayoutType,
            gamma_q_origin=gamma_q.origin,
            gamma_q_engine=gamma_q.Engine,
            gamma_k_mut=gamma_k.mut,
            gamma_k_layout=gamma_k.LayoutType,
            gamma_k_origin=gamma_k.origin,
            gamma_k_engine=gamma_k.Engine,
            var_mut=qk_var.mut,
            var_layout=qk_var.LayoutType,
            var_origin=qk_var.origin,
            var_engine=qk_var.Engine,
            q_layout=q.LayoutType,
            q_origin=q.origin,
            q_engine=q.Engine,
            k_layout=k.LayoutType,
            k_origin=k.origin,
            k_engine=k.Engine,
            simd_width=1,
        ]
        ctx.enqueue_function[kernel](
            q_out,
            k_out,
            gamma_q,
            gamma_k,
            qk_var,
            q,
            k,
            epsilon,
            Int32(q_cols),
            Int32(k_cols),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


def apply_qk_rms_norm_cpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, .float32, ...],
    gamma_k: TileTensor[mut=False, .float32, ...],
    qk_var: TileTensor[mut=False, .float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Float32,
    rows: Int,
    q_cols: Int,
    k_cols: Int,
):
    """Naive CPU reference path (also used as a correctness oracle)."""
    comptime assert q.flat_rank == 2, "q must have rank 2"
    comptime assert k.flat_rank == 2, "k must have rank 2"
    for r in range(rows):
        var rs_q = rsqrt(qk_var.load[width=1](Coord(Index(r, 0))) + epsilon)
        for c in range(q_cols):
            var xf = q.load[width=1](Coord(Index(r, c)))[0].cast[
                DType.float32
            ]()
            var g = gamma_q.load[width=1](Coord(c))
            q_out.store[width=1](
                Coord(Index(r, c)), ((xf * rs_q) * g).cast[out_dtype]()
            )

        var rs_k = rsqrt(qk_var.load[width=1](Coord(Index(r, 1))) + epsilon)
        for c in range(k_cols):
            var xf = k.load[width=1](Coord(Index(r, c)))[0].cast[
                DType.float32
            ]()
            var g = gamma_k.load[width=1](Coord(c))
            k_out.store[width=1](
                Coord(Index(r, c)), ((xf * rs_k) * g).cast[out_dtype]()
            )


def apply_qk_rms_norm[
    in_dtype: DType,
    out_dtype: DType,
    //,
    target: StaticString = "cpu",
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, .float32, ...],
    gamma_k: TileTensor[mut=False, .float32, ...],
    qk_var: TileTensor[mut=False, .float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Float32,
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    """Fused per-element QK-RMSNorm apply for two operands Q and K.

    Given the already cross-rank-reduced per-row statistics `qk_var [M, 2]`
    (col 0 = mean(q^2), col 1 = mean(k^2), float32) and per-column float32
    scales `gamma_q [Nq]` / `gamma_k [Nk]`, applies in a single launch:

    `q_out[m,c] = cast((cast(q[m,c], f32) * rsqrt(qk_var[m,0] + eps)) * gamma_q[c], out_dtype)`
    and likewise for K with column 1. The grouping `((x * rs) * gamma)` then
    cast matches the unfused graph this replaces for bit-accuracy. This fuses
    the QK-RMSNorm apply chain (~7 tiny elementwise/View kernels) into one
    launch, used for cross-head QK-RMSNorm under tensor parallelism.

    All operands (`q` / `k` activations, the outputs `q_out` / `k_out`, and the
    `gamma_q` / `gamma_k` / `qk_var` inputs) are passed directly as
    `TileTensor`s and loaded/stored in-kernel, matching the in-file rms_norm
    `gamma.load[...]` idiom.

    Parameters:
        in_dtype: Element type of both activation inputs (`bfloat16` or
            `float32`).
        out_dtype: Element type of the outputs (typically equal to `in_dtype`).
        target: `"cpu"` or a GPU target string.

    Args:
        q_out: Scaled Q output, shape `[M, Nq]`.
        k_out: Scaled K output, shape `[M, Nk]`.
        gamma_q: Per-column float32 Q scales, shape `[Nq]`.
        gamma_k: Per-column float32 K scales, shape `[Nk]`.
        qk_var: Per-row float32 statistics, shape `[M, 2]` (col 0 = mean(q^2),
            col 1 = mean(k^2)).
        q: Q activations, shape `[M, Nq]`.
        k: K activations, shape `[M, Nk]`.
        epsilon: RMSNorm epsilon, added to the variance before `rsqrt`.
        rows: Shared leading dimension of Q and K.
        q_cols: Number of columns of Q.
        k_cols: Number of columns of K.
        ctx: Device context (ignored on CPU).
    """

    @always_inline
    def description_fn() {imm} -> String:
        return trace_arg("qk", Coord(rows, q_cols + k_cols), in_dtype)

    with Trace[TraceLevel.OP, target=target](
        "apply_qk_rms_norm",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        if rows == 0:
            return

        comptime if is_cpu[target]():
            apply_qk_rms_norm_cpu(
                q_out,
                k_out,
                gamma_q,
                gamma_k,
                qk_var,
                q,
                k,
                epsilon,
                rows,
                q_cols,
                k_cols,
            )
        elif is_gpu[target]():
            apply_qk_rms_norm_gpu(
                q_out,
                k_out,
                gamma_q,
                gamma_k,
                qk_var,
                q,
                k,
                epsilon,
                rows,
                q_cols,
                k_cols,
                ctx,
            )
        else:
            comptime assert False, "unsupported target " + target


def group_norm_reshape[
    dtype: DType,
    rank: Int,
](
    shape: Coord,
    buf: TileTensor[dtype, ...],
    channels_per_group: Int,
    spatial: Int,
    out result: TileTensor[
        dtype,
        Layout[
            shape_types=DynamicCoord[.int64, 2].element_types,
            stride_types=DynamicCoord[.int64, 2].element_types,
        ],
        buf.origin,
        address_space=buf.address_space,
    ],
):
    """
    Reshapes an input buffer for group normalization by flattening all
    dimensions except the group dimension. Returns a 2D buffer of shape
    (num_groups * N, group_size), where group_size is the product of
    channels_per_group and spatial.
    """
    comptime assert buf.rank == rank, "buf.rank must equal rank"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    var group_size = channels_per_group * spatial
    var prod_all_but_group_dim = Int(shape.product()) // group_size
    var new_shape = IndexList[2](prod_all_but_group_dim, group_size)
    var reshaped = reshape[2](buf, new_shape)
    result = {
        reshaped.ptr,
        reshaped.layout,
    }


@__name(t"group_norm_gpu_warp_tiling_{dtype}")
def group_norm_gpu_warp_tiling[
    LayoutType: TensorLayout,
    origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
](
    output: TileTensor[dtype, LayoutType, origin],
    epsilon: Float32,
    num_groups: Int32,
    channels_per_group: Int32,
    spatial: Int32,
):
    var _num_groups = Int(num_groups)
    var _channels_per_group = Int(channels_per_group)
    var _spatial = Int(spatial)
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var idx = thread_idx.x * simd_width

    var vec_data = SIMD[accum_type, simd_width]()
    var group_size = _channels_per_group * _spatial

    var row = block_idx.x
    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    var thread_mean = Scalar[accum_type]()
    var thread_m2 = Scalar[accum_type]()
    var thread_count = Scalar[accum_type]()

    with PDL():
        if idx + simd_width <= group_size:
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()

            comptime for i in range(simd_width):
                welford_update(
                    vec_data[i], thread_mean, thread_m2, thread_count
                )

        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        if idx + simd_width <= group_size:
            var g = umod(row, _num_groups)
            var c_base = g * _channels_per_group
            var norm_val = SIMD[accum_type, simd_width]()
            for i in range(simd_width):
                var offset = (idx + i) // _spatial
                var c = c_base + offset
                var gamma_val = gamma_fn[1](Coord(c))
                var beta_val = beta_fn[1](Coord(c))
                norm_val[i] = (
                    vec_data[i] - row_mean
                ) * norm_factor * gamma_val.cast[accum_type]() + beta_val.cast[
                    accum_type
                ]()

            var output_idx = output.layout(Coord(row, idx))
            output.raw_store[alignment=align](
                output_idx, norm_val.cast[dtype]()
            )


@__name(t"group_norm_gpu_block_{dtype}")
def group_norm_gpu_block[
    LayoutType: TensorLayout,
    origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
](
    output: TileTensor[dtype, LayoutType, origin],
    epsilon: Float32,
    num_groups: Int32,
    channels_per_group: Int32,
    spatial: Int32,
):
    var _num_groups = Int(num_groups)
    var _channels_per_group = Int(channels_per_group)
    var _spatial = Int(spatial)
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    var group_size = _channels_per_group * _spatial

    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    with PDL():
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        for x in range(ceildiv(group_size // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < group_size:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                comptime for i in range(simd_width):
                    welford_update(
                        vec_data[i], thread_mean, thread_m2, thread_count
                    )

        welford_block_all_reduce(
            thread_mean,
            thread_m2,
            thread_count,
            row_mean,
            row_m2,
            row_count,
        )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        for x in range(ceildiv(group_size // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < group_size:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                var g = umod(row, _num_groups)
                var c_base = g * _channels_per_group

                var norm_val = SIMD[accum_type, simd_width]()
                for i in range(simd_width):
                    var offset_c = (offset + i) // _spatial
                    var c = c_base + offset_c
                    var gamma_val = gamma_fn[1](Coord(c))
                    var beta_val = beta_fn[1](Coord(c))
                    norm_val[i] = (
                        vec_data[i] - row_mean
                    ) * norm_factor * gamma_val.cast[
                        accum_type
                    ]() + beta_val.cast[
                        accum_type
                    ]()

                var output_row_offset = output.layout(Coord(row, offset))
                output.raw_store[alignment=align](
                    output_row_offset, norm_val.cast[dtype]()
                )


@__name(t"group_norm_gpu_multi_block_stats_{dtype}")
def group_norm_gpu_multi_block_stats[
    StatsLayoutType: TensorLayout,
    stats_origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
](
    stats: TileTensor[get_accum_type[dtype](), StatsLayoutType, stats_origin],
    num_splits: Int32,
    group_size: Int32,
):
    """Multi-block stats kernel: computes partial Welford statistics per split.

    Grid: num_rows * _num_splits blocks. Each block handles one split of one
    group and writes partial (mean, m2, count) to the stats buffer.
    Stats layout: stats[block_idx * 3 + {0,1,2}] = {mean, m2, count}.
    """
    var _num_splits = Int(num_splits)
    var _group_size = Int(group_size)
    comptime accum_type = get_accum_type[dtype]()

    var block_id = block_idx.x
    var row, split_id = divmod(block_id, _num_splits)
    var tid = thread_idx.x

    # Compute chunk boundaries (each split handles a contiguous chunk,
    # aligned to simd_width).
    var total_simd_elems = _group_size // simd_width
    var chunk_simd_size = ceildiv(total_simd_elems, _num_splits)
    var chunk_start = split_id * chunk_simd_size * simd_width
    var chunk_end = min(chunk_start + chunk_simd_size * simd_width, _group_size)
    var chunk_iters = ceildiv(chunk_simd_size, block_dim.x)

    with PDL():
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        for x in range(chunk_iters):
            var offset = (
                chunk_start + x * block_dim.x * simd_width + tid * simd_width
            )
            if offset < chunk_end:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()
                comptime for i in range(simd_width):
                    welford_update(
                        vec_data[i],
                        thread_mean,
                        thread_m2,
                        thread_count,
                    )

        var row_mean = Scalar[accum_type]()
        var row_m2 = Scalar[accum_type]()
        var row_count = Scalar[accum_type]()
        welford_block_all_reduce(
            thread_mean,
            thread_m2,
            thread_count,
            row_mean,
            row_m2,
            row_count,
        )

        # Thread 0 writes partial stats to the global stats buffer.
        if tid == 0:
            var base_idx = block_id * 3
            stats.store(Coord(base_idx), row_mean)
            stats.store(Coord(base_idx + 1), row_m2)
            stats.store(Coord(base_idx + 2), row_count)


@__name(t"group_norm_gpu_multi_block_norm_{dtype}")
def group_norm_gpu_multi_block_norm[
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    StatsLayoutType: TensorLayout,
    stats_origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
](
    output: TileTensor[dtype, OutputLayoutType, output_origin],
    stats: TileTensor[get_accum_type[dtype](), StatsLayoutType, stats_origin],
    epsilon: Float32,
    num_groups: Int32,
    channels_per_group: Int32,
    spatial: Int32,
    num_splits: Int32,
    group_size: Int32,
):
    """Multi-block normalize kernel: reduces partial stats and normalizes.

    Grid: num_rows * num_splits blocks. Each block reads all partial stats
    for its group, reduces to final mean/variance, then normalizes its
    chunk of elements.
    """
    var _num_groups = Int(num_groups)
    var _channels_per_group = Int(channels_per_group)
    var _spatial = Int(spatial)
    var _num_splits = Int(num_splits)
    var _group_size = Int(group_size)
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var block_id = block_idx.x
    var row, split_id = divmod(block_id, _num_splits)
    var tid = thread_idx.x

    # Same chunk boundaries as stats kernel.
    var total_simd_elems = _group_size // simd_width
    var chunk_simd_size = ceildiv(total_simd_elems, _num_splits)
    var chunk_start = split_id * chunk_simd_size * simd_width
    var chunk_end = min(chunk_start + chunk_simd_size * simd_width, _group_size)
    var chunk_iters = ceildiv(chunk_simd_size, block_dim.x)

    with PDL():
        # Reduce all partial stats for this group (_num_splits is small,
        # typically 4-16, so this loop is cheap).
        var row_mean = Scalar[accum_type]()
        var row_m2 = Scalar[accum_type]()
        var row_count = Scalar[accum_type]()
        var stats_row_base = row * _num_splits * 3
        for s in range(_num_splits):
            var base_idx = stats_row_base + s * 3
            welford_combine(
                stats.load[width=1](Coord(base_idx)),
                stats.load[width=1](Coord(base_idx + 1)),
                stats.load[width=1](Coord(base_idx + 2)),
                row_mean,
                row_m2,
                row_count,
            )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        var g = row % _num_groups
        var c_base = g * _channels_per_group

        for x in range(chunk_iters):
            var offset = (
                chunk_start + x * block_dim.x * simd_width + tid * simd_width
            )
            if offset < chunk_end:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                var norm_val = SIMD[accum_type, simd_width]()

                # Vectorized gamma/beta: when all SIMD elements share the
                # same channel (common case for large _spatial dims), load
                # gamma/beta once and broadcast.
                var c_first = c_base + offset // _spatial
                var c_last = c_base + ceildiv(offset + simd_width - 1, _spatial)
                if c_first == c_last:
                    var gamma_val = gamma_fn[1](Coord(c_first)).cast[
                        accum_type
                    ]()
                    var beta_val = beta_fn[1](Coord(c_first)).cast[accum_type]()
                    norm_val = (
                        vec_data - row_mean
                    ) * norm_factor * gamma_val + beta_val
                else:
                    for i in range(simd_width):
                        var c = c_base + (offset + i) // _spatial
                        var gamma_val = gamma_fn[1](Coord(c))
                        var beta_val = beta_fn[1](Coord(c))
                        norm_val[i] = (
                            vec_data[i] - row_mean
                        ) * norm_factor * gamma_val.cast[
                            accum_type
                        ]() + beta_val.cast[
                            accum_type
                        ]()

                var output_row_offset = output.layout(Coord(row, offset))
                output.raw_store[alignment=align](
                    output_row_offset, norm_val.cast[dtype]()
                )


def group_norm_gpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    gamma_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
](
    shape: Coord,
    epsilon: Float32,
    output: TileTensor[mut=True, dtype, ...],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime accum_type = get_accum_type[dtype]()
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"

    var N = Int(shape[0].value())
    var C = Int(shape[1].value())

    var spatial = Int(shape.product()) // (N * C)

    # The device closure reads only the innermost dim; a typed `Coord` capture
    # fails the launch (CUDA_ERROR_INVALID_VALUE), so unwrap it here and let the
    # closure capture plain `Int`s.
    var last_dim = Int(shape[rank - 1].value())
    var channels_per_group = C // num_groups

    var output_rs = group_norm_reshape[dtype, rank](
        shape, output, channels_per_group, spatial
    )

    comptime OutputLinearIdxType = Scalar[output_rs.linear_idx_type]

    var num_rows = output_rs.dim[0]()
    var num_cols = output_rs.dim[1]()

    # Zero-sized input (e.g. a ``(B, C, 0, 0)`` tensor flowing through a
    # diffusion VAE encoder for the text-to-image placeholder): nothing
    # to normalize.  The output buffer is pre-allocated zero-element by
    # the caller and the kernel's ``num_cols < simd_width`` misalignment
    # check below would otherwise abort.  Early-return is correct because
    # mean/var of an empty group has no defined value and the downstream
    # readers also have zero spatial dims.
    if num_rows == OutputLinearIdxType(0) or num_cols == OutputLinearIdxType(0):
        return

    @__parameter
    @always_inline
    @__copy_capture(spatial, last_dim, num_groups, channels_per_group)
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) capturing -> SIMD[dtype, simd_width]:
        var n, g = divmod(row, num_groups)
        var c = g * channels_per_group

        comptime if rank == 4:
            var inner_volume = spatial
            var c_offset, hw = divmod(col, inner_volume)
            c += c_offset
            var h, w = divmod(hw, last_dim)

            # Guard against c_offset boundary straddling.  A view-fused
            # NHWC→NCHW transpose generates a strided_load(stride=C) for
            # the W-dimension.  That load is correct within a single c_offset
            # region (consecutive-w with stride C maps to adjacent NHWC
            # addresses), but it reads wrong elements when the simd_width
            # window crosses a c_offset boundary at a multiple of
            # inner_volume=H*W.  This happens when H*W % simd_width != 0 and
            # the thread's starting column lands near a boundary.  Fall back
            # to element-wise scalar loads in that case so each element's
            # full (n, c, h, w) index is recomputed independently.
            if ceildiv(col + simd_width - 1, inner_volume) != c_offset:
                var result = SIMD[dtype, simd_width]()
                for i in range(simd_width):
                    var cur_col = col + i
                    var c_off, hw_i = divmod(cur_col, inner_volume)
                    var h_i, w_i = divmod(hw_i, last_dim)
                    result[i] = input_fn[1](
                        Coord(n, g * channels_per_group + c_off, h_i, w_i)
                    )[0]
                return result

            return input_fn[simd_width](Coord(n, c, h, w))

        elif rank == 3:
            var inner_volume = spatial
            var c_offset, l = divmod(col, inner_volume)
            c += c_offset

            if ceildiv(col + simd_width - 1, inner_volume) != c_offset:
                var result = SIMD[dtype, simd_width]()
                for i in range(simd_width):
                    var cur_col = col + i
                    var c_off, l_i = divmod(cur_col, inner_volume)
                    result[i] = input_fn[1](
                        Coord(n, g * channels_per_group + c_off, l_i)
                    )[0]
                return result

            return input_fn[simd_width](Coord(n, c, l))

        else:
            comptime assert False, "group_norm_gpu requires a rank of 3 or 4"

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    if num_cols < OutputLinearIdxType(simd_width):
        raise Error(
            "group_norm_gpu requires num_cols >= simd_width; got num_cols="
            + String(num_cols)
            + " and simd_width="
            + String(simd_width)
        )

    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = num_rows
    var block_dim = min(
        ceildiv(
            ceildiv(num_cols, OutputLinearIdxType(simd_width)),
            OutputLinearIdxType(WARP_SIZE),
        )
        * OutputLinearIdxType(WARP_SIZE),
        OutputLinearIdxType(WARP_SIZE * max_warps_per_block),
    )

    if num_cols % OutputLinearIdxType(simd_width) == 0:
        # When the number of columns is small enough that they can be placed in
        # registers, we do warp tiling, which is a single pass to do mean/var
        # computation and normalization.
        if num_cols <= OutputLinearIdxType(
            WARP_SIZE * simd_width * max_warps_per_block
        ):
            comptime kernel = group_norm_gpu_warp_tiling[
                LayoutType=output_rs.LayoutType,
                origin=output_rs.origin,
                dtype=dtype,
                simd_width=simd_width,
                input_fn=input_fn_2d,
                gamma_fn=gamma_fn,
                beta_fn=beta_fn,
            ]
            ctx.enqueue_function[kernel](
                output_rs,
                epsilon.cast[.float32](),
                Int32(num_groups),
                Int32(channels_per_group),
                Int32(spatial),
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
        else:
            # Use multi-block reduction when the grid is too small for
            # good GPU occupancy.  Each group is split across num_splits
            # blocks so that more SMs are active.
            comptime desired_min_grid = 256
            var num_splits = 1
            if Int(num_rows) < desired_min_grid:
                num_splits = min(ceildiv(desired_min_grid, Int(num_rows)), 32)
                # Ensure each split has enough work (≥ 1 SIMD iter per
                # thread at block_dim threads).
                var group_size = Int(num_cols)
                var max_useful_splits = max(
                    1,
                    group_size
                    // (Int(simd_width) * WARP_SIZE * max_warps_per_block),
                )
                num_splits = min(num_splits, max_useful_splits)

            if num_splits > 1:
                var group_size = Int(num_cols)

                # Allocate a small buffer for partial Welford statistics:
                # 3 values (mean, m2, count) per (row, split).
                var stats_size = Int(num_rows) * num_splits * 3
                var stats_buf = ctx.enqueue_create_buffer[accum_type](
                    stats_size
                )
                var stats = TileTensor(
                    stats_buf,
                    row_major(stats_size),
                )

                # Compute block_dim based on per-split chunk size.
                # Cap at 256 threads: both kernels capture closures
                # (input_fn_2d with its coordinate computation chain,
                # gamma_fn, beta_fn) that cause high register pressure,
                # especially for bfloat16 (simd_width=8).  256 threads
                # keeps total register usage within GPU limits while
                # each thread processes more elements per iteration.
                comptime mb_max_block_dim = min(
                    256, WARP_SIZE * max_warps_per_block
                )
                var total_simd_elems = group_size // simd_width
                var chunk_simd_size = ceildiv(total_simd_elems, num_splits)
                var mb_block_dim = min(
                    align_up(chunk_simd_size, WARP_SIZE),
                    mb_max_block_dim,
                )
                var mb_grid_dim = Int(num_rows) * num_splits

                # Kernel 1: compute partial Welford stats per split.
                comptime stats_kernel = group_norm_gpu_multi_block_stats[
                    StatsLayoutType=stats.LayoutType,
                    stats_origin=stats.origin,
                    dtype=dtype,
                    simd_width=simd_width,
                    input_fn=input_fn_2d,
                ]
                ctx.enqueue_function[stats_kernel](
                    stats,
                    Int32(num_splits),
                    Int32(group_size),
                    grid_dim=mb_grid_dim,
                    block_dim=mb_block_dim,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )

                # Kernel 2: reduce stats and normalize each chunk.
                comptime norm_kernel = group_norm_gpu_multi_block_norm[
                    OutputLayoutType=output_rs.LayoutType,
                    output_origin=output_rs.origin,
                    StatsLayoutType=stats.LayoutType,
                    stats_origin=stats.origin,
                    dtype=dtype,
                    simd_width=simd_width,
                    input_fn=input_fn_2d,
                    gamma_fn=gamma_fn,
                    beta_fn=beta_fn,
                ]
                ctx.enqueue_function[norm_kernel](
                    output_rs,
                    stats,
                    epsilon.cast[.float32](),
                    Int32(num_groups),
                    Int32(channels_per_group),
                    Int32(spatial),
                    Int32(num_splits),
                    Int32(group_size),
                    grid_dim=mb_grid_dim,
                    block_dim=mb_block_dim,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )

                _ = stats_buf^
            else:
                comptime kernel = group_norm_gpu_block[
                    LayoutType=output_rs.LayoutType,
                    origin=output_rs.origin,
                    dtype=dtype,
                    simd_width=simd_width,
                    input_fn=input_fn_2d,
                    gamma_fn=gamma_fn,
                    beta_fn=beta_fn,
                ]
                ctx.enqueue_function[kernel](
                    output_rs,
                    epsilon.cast[.float32](),
                    Int32(num_groups),
                    Int32(channels_per_group),
                    Int32(spatial),
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )
    else:
        comptime kernel = group_norm_gpu_block[
            LayoutType=output_rs.LayoutType,
            origin=output_rs.origin,
            dtype=dtype,
            simd_width=1,
            input_fn=input_fn_2d,
            gamma_fn=gamma_fn,
            beta_fn=beta_fn,
        ]
        ctx.enqueue_function[kernel](
            output_rs,
            epsilon.cast[.float32](),
            Int32(num_groups),
            Int32(channels_per_group),
            Int32(spatial),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


def group_norm_cpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    gamma_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
](
    shape: Coord,
    epsilon: Float32,
    output: TileTensor[mut=True, dtype, ...],
    num_groups: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Computes group normalization on CPU.

    Reduces a single-pass Welford mean/variance over each `(batch, group)`
    block of `channels_per_group * spatial` elements, then applies the
    per-channel `gamma`/`beta` affine transform. Parallelizes across
    `N * num_groups` blocks.

    Parameters:
        dtype: Element type of the input and output tensors.
        rank: Tensor rank of the input and output tensors (3 or 4).
        input_fn: Function called to generate an input value.
        gamma_fn: Function called to generate a gamma value.
        beta_fn: Function called to generate a beta value.

    Args:
        shape: The shape of the input/output tensor.
        epsilon: Small constant for numerical stability.
        output: Output tensor receiving the normalized result.
        num_groups: Number of groups the channel dimension is split into.
        ctx: Optional device context used to size the CPU thread pool.
    """
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime accum_type = get_accum_type[dtype]()
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"

    var N = Int(shape[0].value())
    var C = Int(shape[1].value())
    var spatial = Int(shape.product()) // (N * C)
    var channels_per_group = C // num_groups
    var group_size = channels_per_group * spatial
    var num_rows = N * num_groups

    if num_rows == 0 or group_size == 0:
        return

    var num_workers = min(parallelism_level(ctx), num_rows)
    var chunk_size = ceildiv(num_rows, num_workers)

    def task_func(
        thread_id: Int,
    ) raises {
        var shape,
        var num_groups,
        var channels_per_group,
        var spatial,
        var epsilon,
        imm,
    }:
        var row_start = thread_id * chunk_size
        var row_end = min(row_start + chunk_size, num_rows)

        for row in range(row_start, row_end):
            var n, g = divmod(row, num_groups)
            var c_base = g * channels_per_group

            @__copy_capture(shape, n, c_base, spatial)
            @__parameter
            @always_inline
            def indices_for(col: Int) -> DynamicCoord[.int64, rank]:
                var c_offset, s = divmod(col, spatial)
                comptime if rank == 4:
                    var h, w = divmod(s, Int(shape[3].value()))
                    return rebind[DynamicCoord[.int64, rank]](
                        Coord(
                            Int64(n),
                            Int64(c_base + c_offset),
                            Int64(h),
                            Int64(w),
                        )
                    )
                else:
                    return rebind[DynamicCoord[.int64, rank]](
                        Coord(Int64(n), Int64(c_base + c_offset), Int64(s))
                    )

            # Single-pass Welford mean/variance over the group.
            var mean = Scalar[accum_type]()
            var m2 = Scalar[accum_type]()
            var count = Scalar[accum_type]()
            for col in range(group_size):
                var val = input_fn[1](indices_for(col))[0].cast[accum_type]()
                welford_update(val, mean, m2, count)

            var norm_factor = rsqrt(m2 / count + epsilon.cast[accum_type]())

            for col in range(group_size):
                var idx = indices_for(col)
                var val = input_fn[1](idx)[0].cast[accum_type]()
                var gamma_val = gamma_fn[1](Coord(idx[1]))[0].cast[accum_type]()
                var beta_val = beta_fn[1](Coord(idx[1]))[0].cast[accum_type]()
                var norm_val = (val - mean) * norm_factor * gamma_val + beta_val
                output.store(idx, norm_val.cast[dtype]())

    sync_parallelize(task_func, num_workers, ctx)


@always_inline
def group_norm[
    dtype: DType,
    rank: Int,
    input_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    gamma_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    /,
    target: StaticString = "cpu",
](
    shape: Coord,
    epsilon: Float32,
    groups: Int32,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime assert (
        rank > 2 and rank < 5
    ), "group_norm requires input rank of 3 or 4"

    var shape_il = coord_to_index_list(shape)

    if shape != output.layout.shape_coord():
        raise Error(
            "Input/output shape mismatch: input = {shape}, output ="
            " {output.dynamic_shape}"
        )

    var num_groups: Int = Int(groups[0])

    var C = shape_il[1]
    if C % num_groups != 0:
        raise Error(
            "Invalid num_groups: channels (C = {C}) must be divisible by"
            " num_groups = {num_groups}"
        )

    @always_inline
    def description_fn() {imm} -> String:
        return trace_arg("input", shape_il, dtype)

    with Trace[TraceLevel.OP, target=target](
        "group_norm",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        comptime if is_cpu[target]():
            group_norm_cpu[
                dtype=dtype,
                rank=rank,
                input_fn=input_fn,
                gamma_fn=gamma_fn,
                beta_fn=beta_fn,
            ](
                shape,
                epsilon,
                output,
                num_groups,
                Optional[DeviceContext](ctx),
            )
        elif is_gpu[target]():
            group_norm_gpu[
                dtype=dtype,
                rank=rank,
                input_fn=input_fn,
                gamma_fn=gamma_fn,
                beta_fn=beta_fn,
            ](
                shape,
                epsilon,
                output,
                num_groups,
                ctx=ctx,
            )
        else:
            comptime assert False, "unsupported target " + target


# ===----------------------------------------------------------------------=== #
# Row-based rms_norm + layer_norm (free-form body layer).
#
# rms_norm: one ReduceSum-of-squares phase -> per-element map
# (normalize * gamma). layer_norm: one Welford phase (mean + variance in a
# single pass) -> map (normalize * gamma + beta). gamma/beta are broadcast
# side-loads inside the map, like the GC IR's `%weight` load. The row is
# cached once and replayed across the reduce + map on the
# cache-eligible tier.
# ===----------------------------------------------------------------------=== #


def rms_norm[
    dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    target: StaticString,
    multiply_before_cast: Bool = True,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    context: Optional[DeviceContext] = None,
) raises:
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "rms_norm requires fp accum"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    comptime assert shape.is_flat, "shape must be flat"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], target, 64, dtype, accum
    ]()
    var axis_size_accum = Scalar[accum](Int(axis_size.value()))

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var epsilon,
        var weight_offset,
        var axis_size_accum,
        var input_fn,
        var output_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # Reduce: sum of squares -> inv_rms.
        @always_inline
        def square[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # `inv_rms` held W-wide (one value broadcast to every lane, or one per
        # output column, depending on how the launch maps threads to
        # outputs); read via `.slice[width]` uniformly either way.
        var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms = rsqrt(ssq / axis_size_accum + epsilon.cast[accum]())

        # gamma indexed by the reduced axis. Its inner (lane) stride is 1 when
        # the reduced axis is inner (lanes are consecutive reduced positions),
        # else 0 (lanes are columns at one position — one gamma, splatted). A
        # strided load handles both via a compile-time stride, no branching.
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Emit: per-element normalize, scale by gamma, and store.
        @always_inline
        def write[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {
            var gamma,
            var inv_rms,
            var weight_offset,
            var output_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var col = Int(idx.coord[reduce_dim].value())
            var g_raw = strided_load[width, g_stride, alignment=alignment](
                gamma.ptr_at_offset(Coord(col))
            )
            var normed = tile.cast[accum]() * inv_rms.slice[width]()

            # `multiply_before_cast` matches legacy: True multiplies gamma in
            # `accum` then casts; False casts the normed value first, then
            # multiplies gamma in the input dtype.
            comptime if multiply_before_cast:
                var gamma_with_offset = (
                    g_raw.cast[accum]() + weight_offset.cast[accum]()
                )
                var result = (normed * gamma_with_offset).cast[dtype]()
                output_fn[width, alignment](idx.coord, result)
            else:
                var gamma_with_offset = g_raw + weight_offset
                var result = normed.cast[dtype]() * gamma_with_offset
                output_fn[width, alignment](idx.coord, result)

        # `gamma`/`inv_rms`/`weight_offset`/`output_fn` ride into `elementwise`.
        row.elementwise(write, load)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based rms_norm_rope (the IR litmus test): multiple inputs of
# differing dtypes (primary input; cos/sin side-loads in cos_sin_dtype; gamma),
# one ReduceSum-of-squares phase, then a map that loads the rotate-half partner
# (a second axis position) — the IR's `imap.concat.select` / `imap.slice.index`.
# The primary input is cached and reused by the reduce + the map's normed term;
# the partner is an additional load (inherent to rope, not a re-read).
# ===----------------------------------------------------------------------=== #


def rms_norm_rope[
    input_dtype: DType,
    output_dtype: DType,
    cos_sin_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[input_dtype, width]),
    CosFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[cos_sin_dtype, width]),
    SinFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[cos_sin_dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[output_dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    target: StaticString,
    multiply_before_cast: Bool = True,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    cos_fn: CosFn,
    sin_fn: SinFn,
    output_fn: OutputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, input_dtype, ...],
    epsilon: Scalar[input_dtype],
    weight_offset: Scalar[input_dtype],
    context: Optional[DeviceContext] = None,
) raises:
    comptime accum = get_accum_type[input_dtype]()
    comptime assert accum.is_floating_point(), "rope requires fp accum"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    comptime assert shape.is_flat, "shape must be flat"

    comptime if AxisSizeT.is_static_value:
        comptime assert (
            AxisSizeT.static_value % 2 == 0
        ), "rope requires an even row width"
    comptime simd_width_cand = rowwise.pick_simd_width[
        ReduceSum[accum, 1], target, 64, input_dtype, accum
    ]()
    # Rotate-half needs each SIMD tile to sit wholly in one half and align to
    # the partner offset, i.e. `half` must be a multiple of the SIMD width. That
    # only holds when the row width is statically known and half-divisible;
    # otherwise (dynamic / non-divisible cols) fall back to scalar.
    comptime simd_width = simd_width_cand if (
        AxisSizeT.is_static_value
        and (AxisSizeT.static_value // 2) % simd_width_cand == 0
    ) else 1
    var axis_size_int = Int(axis_size.value())
    if axis_size_int % 2 != 0:
        raise Error("rope requires an even row width")
    # `half` is the rotate-half split point. `axis_size_int` is the (possibly
    # dynamic) row width, so this holds whether or not `AxisSizeT` is static.
    var half = axis_size_int // 2
    var axis_size_accum = Scalar[accum](axis_size_int)

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var half,
        var gamma,
        var epsilon,
        var weight_offset,
        var axis_size_accum,
        var input_fn,
        var cos_fn,
        var sin_fn,
        var output_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_fn} -> SIMD[input_dtype, width]:
            return input_fn[width, alignment](idx.coord)

        var row = rowwise.Row[
            params, accum, input_dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # Reduce: sum of squares -> inv_rms.
        @always_inline
        def square[
            width: Int
        ](tile: SIMD[input_dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # `inv_rms` held W-wide (one value broadcast to every lane, or one per
        # output column, depending on how the launch maps threads to
        # outputs); read via `.slice[width]` uniformly either way.
        var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms = rsqrt(ssq / axis_size_accum + epsilon.cast[accum]())
        var woff = weight_offset.cast[accum]()

        # gamma indexed by the reduced axis: inner stride 1 when that axis is
        # inner (consecutive), else 0 (splat). Strided load, compile-time
        # stride, no branching.
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Cache: normalize, scale by gamma, and round to `output_dtype` before
        # RoPE to match the unfused (separate rms_norm + rope) rounding, then
        # stage the result into shared memory so the rotate-half partner —
        # which lives on a *different* participant — is read from shmem
        # instead of re-loaded from global and re-normalized.
        # `multiply_before_cast` selects accum-multiply (True) vs
        # output-dtype-multiply (False), matching legacy.
        @always_inline
        def normed_tile[
            width: Int
        ](tile: SIMD[input_dtype, width], idx: RowCoord[row_rank]) {
            var gamma,
            var inv_rms,
            var woff,
            var weight_offset,
        } -> SIMD[output_dtype, width]:
            var col = Int(idx.coord[reduce_dim].value())
            var g_raw = strided_load[
                width, g_stride, alignment=align_of[SIMD[input_dtype, width]]()
            ](gamma.ptr_at_offset(Coord(col)))
            var scaled = tile.cast[accum]() * inv_rms.slice[width]()

            comptime if multiply_before_cast:
                var gamma_with_offset = g_raw.cast[accum]() + woff
                return (scaled * gamma_with_offset).cast[output_dtype]()
            else:
                var gamma_with_offset = (g_raw + weight_offset).cast[
                    output_dtype
                ]()
                return scaled.cast[output_dtype]() * gamma_with_offset

        var normed = row.cache[output_dtype, shared=True](normed_tile)

        # Emit: reads other columns of the staged row via `normed.load`
        # (the rotate-half partner). `load`'s fallback path needs the
        # primary loader + the producer that built `normed` — `RowCache`
        # doesn't carry them as fields, so `write` captures them too and
        # re-supplies them to `normed.load`.
        @always_inline
        def write[
            width: Int
        ](nc: SIMD[output_dtype, width], idx: RowCoord[row_rank]) {
            var normed,
            var load,
            var normed_tile,
            var cos_fn,
            var sin_fn,
            var half,
            var output_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[input_dtype, width]()
            var col = Int(idx.coord[reduce_dim].value())
            var normed_c = nc.cast[accum]()

            # Rotate-half partner: concat(-x2, x1). A W-aligned tile sits
            # wholly in one half, so the sign is uniform. The partner's normed
            # value is read straight from the staged shmem row.
            var partner: RowCoord[row_rank]
            var sign: Scalar[accum]
            if col < half:
                partner = idx.at_axis[reduce_dim](col + half)
                sign = Scalar[accum](-1)
            else:
                partner = idx.at_axis[reduce_dim](col - half)
                sign = Scalar[accum](1)
            var rotated = (
                sign
                * normed.load[width](partner, load, normed_tile).cast[accum]()
            )

            var cos_c = cos_fn[width, alignment](idx.coord).cast[accum]()
            var sin_c = sin_fn[width, alignment](idx.coord).cast[accum]()
            var out = normed_c * cos_c + rotated * sin_c
            comptime output_alignment = ctx_p.element_alignment[
                output_dtype, width
            ]()
            var result = out.cast[output_dtype]()
            output_fn[width, output_alignment](
                idx.coord,
                result,
            )

        row.elementwise(normed, write, load, normed_tile)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based layer_norm: mean + variance via Welford (or a two-pass
# sum), then a normalizing map scaled by gamma and shifted by beta.
# ===----------------------------------------------------------------------=== #


def layer_norm[
    dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    target: StaticString,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, dtype, ...],
    beta: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    context: Optional[DeviceContext] = None,
) raises:
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "layer_norm requires fp accum"
    # Welford's triple-field state is 12B at W=1, so the default 64B
    # per-thread budget halves the SIMD width to 4 for a bf16/fp16 input
    # (natural width 8), giving layer_norm 2x the tiles/ALU of rms_norm
    # (4B `ReduceSum` state keeps W=8). Bump the budget to 96B here so
    # Welford keeps W=8 (12B * 8 = 96); scoped to this call site rather
    # than raising the global default. fp32/fp64 inputs are unaffected
    # (natural width already <= 4).
    comptime simd_width = rowwise.pick_simd_width[
        Welford[accum, 1], target, 96, dtype, accum
    ]()

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var beta,
        var epsilon,
        var input_fn,
        var output_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # Reduce: mean and variance in one Welford pass.
        @always_inline
        def cast_to_accum[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            return tile.cast[accum]()

        # Stats held W-wide (one value broadcast to every lane, or one per
        # output column, depending on how the launch maps threads to
        # outputs); read via `.slice[width]` uniformly either way.
        # `cast_to_accum` crosses into `reduce` as a value arg (value-closure
        # inner-callback form).
        var stats = row.reduce[Welford[accum, params.simd_width]](
            cast_to_accum, load
        )
        var mean = stats.mean
        var inv = rsqrt(stats.M2 / stats.count + epsilon.cast[accum]())

        # gamma/beta indexed by the reduced axis: inner stride 1 when that axis
        # is inner (consecutive), else 0 (splat). Strided load, compile-time
        # stride, no branching.
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Emit: per-element normalize, scale by gamma, shift by beta, store.
        @always_inline
        def write[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {
            var mean,
            var inv,
            var gamma,
            var beta,
            var output_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var col = Int(idx.coord[reduce_dim].value())
            var gamma_val = strided_load[width, g_stride, alignment=alignment](
                gamma.ptr_at_offset(Coord(col))
            ).cast[accum]()
            var beta_val = strided_load[width, g_stride, alignment=alignment](
                beta.ptr_at_offset(Coord(col))
            ).cast[accum]()
            var out = (tile.cast[accum]() - mean.slice[width]()) * inv.slice[
                width
            ]() * gamma_val + beta_val
            var result = out.cast[dtype]()
            output_fn[width, alignment](idx.coord, result)

        # `write` crosses into `elementwise` as a value arg; `mean`/`inv`/
        # `gamma`/`beta`/`output_fn` ride the value via its capture list.
        row.elementwise(write, load)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based layer_norm_rope_ragged: mean + variance via Welford (layer_norm),
# then a map scaled by gamma/beta that additionally applies ragged-position
# RoPE (the same absolute-position lookup as rope_ragged, via the interleaved
# complex multiply from rope.mojo's `_rope`) to the row's leading `rope_dim`
# columns; the remaining columns pass through unrotated. `rope_dim` comes
# from `freqs_cis`'s own width, matching rope_ragged's own convention.
# ===----------------------------------------------------------------------=== #


def layer_norm_rope_ragged[
    input_dtype: DType,
    output_dtype: DType,
    freq_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[input_dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[output_dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    target: StaticString,
    interleaved: Bool,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, input_dtype, ...],
    beta: TileTensor[mut=False, input_dtype, ...],
    epsilon: Scalar[input_dtype],
    input_row_offsets: TileTensor[.uint32, ...],
    start_pos: TileTensor[.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    context: Optional[DeviceContext] = None,
) raises:
    # TODO(feras): non-interleaved RoPE needs the far-partner row-cache
    # lookup rms_norm_rope uses for its rotate-half, not yet implemented here.
    comptime assert (
        interleaved
    ), "layer_norm_rope_ragged: non-interleaved RoPE is not implemented"
    comptime accum = get_accum_type[input_dtype]()
    comptime assert accum.is_floating_point(), "layer_norm requires fp accum"
    comptime assert freqs_cis.LayoutType._shape_types[
        1
    ].is_static_value, "Need static rope_dim for freqs_cis"
    comptime rope_dim = Int(freqs_cis.static_shape[1])
    comptime simd_width_cand = rowwise.pick_simd_width[
        Welford[accum, 1], target, 96, input_dtype, accum
    ]()
    # A SIMD tile must sit wholly within the roped region or wholly within
    # the passthrough region -- never straddle the boundary -- so `rope_dim`
    # must be simd-width-divisible; otherwise fall back to scalar.
    comptime simd_width = simd_width_cand if (
        rope_dim % simd_width_cand == 0
    ) else 1

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var beta,
        var epsilon,
        var input_fn,
        var output_fn,
        var input_row_offsets,
        var start_pos,
        var freqs_cis,
    }:
        comptime row_rank = row_coords.rank

        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_fn} -> SIMD[input_dtype, width]:
            return input_fn[width, alignment](idx.coord)

        var row = rowwise.Row[
            params, accum, input_dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        @always_inline
        def cast_to_accum[
            width: Int
        ](tile: SIMD[input_dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            return tile.cast[accum]()

        var stats = row.reduce[Welford[accum, params.simd_width]](
            cast_to_accum, load
        )
        var mean = stats.mean
        var inv = rsqrt(stats.M2 / stats.count + epsilon.cast[accum]())

        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Ragged absolute position, resolved once per row (shared by every
        # column) -- same lookup rope_ragged's own kernel does per element.
        var row_idx = coord_to_index_list(row_coords)
        var global_token_idx = row_idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )
        var position_idx = Int(start_pos[batch_idx]) + token_idx

        # Cache: normalize and scale/shift by gamma/beta, staged into shared
        # memory so `write` can be a pure per-tile map (no partner lookups
        # needed -- interleaved RoPE pairs adjacent lanes within one tile).
        @always_inline
        def normalize[
            width: Int
        ](tile: SIMD[input_dtype, width], idx: RowCoord[row_rank]) {
            var gamma,
            var beta,
            var mean,
            var inv,
        } -> SIMD[output_dtype, width]:
            comptime alignment = align_of[SIMD[input_dtype, width]]()
            var col = Int(idx.coord[reduce_dim].value())
            var gamma_val = strided_load[width, g_stride, alignment=alignment](
                gamma.ptr_at_offset(Coord(col))
            ).cast[accum]()
            var beta_val = strided_load[width, g_stride, alignment=alignment](
                beta.ptr_at_offset(Coord(col))
            ).cast[accum]()
            var out = (tile.cast[accum]() - mean.slice[width]()) * inv.slice[
                width
            ]() * gamma_val + beta_val
            return out.cast[output_dtype]()

        var normed = row.cache[output_dtype, shared=True](normalize)

        # Emit: passthrough for columns >= rope_dim; interleaved complex-
        # multiply RoPE (rope.mojo's `_rope`) for columns < rope_dim. The
        # block/warp-tier reduce can still dispatch a scalar (width=1) tail
        # tile even when `simd_width` divides `rope_dim`, since the tail is
        # governed by the row's total length, not `rope_dim` alone -- that
        # tile holds one column, not a full (re, im) pair, so `_rope`'s
        # in-register deinterleave doesn't apply. Read the missing partner
        # straight from the staged row cache instead (mirrors
        # row_rms_norm_rope's rotate-half partner load).
        @always_inline
        def write[
            width: Int
        ](nc: SIMD[output_dtype, width], idx: RowCoord[row_rank]) {
            var normed,
            var load,
            var normalize,
            var freqs_cis,
            var position_idx,
            var output_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[output_dtype, width]()
            var col = Int(idx.coord[reduce_dim].value())
            var result: SIMD[output_dtype, width]
            if col < rope_dim:
                comptime if width == 1:
                    var pair_base = (col // 2) * 2
                    var im_c = pair_base + 1
                    var is_re = col == pair_base
                    var partner_idx = idx.at_axis[reduce_dim](
                        im_c if is_re else pair_base
                    )
                    var partner = normed.load[1](
                        partner_idx,
                        load,
                        normalize,
                    ).cast[freq_dtype]()
                    var self_val = nc.cast[freq_dtype]()
                    var x_re = self_val if is_re else partner
                    var x_im = partner if is_re else self_val
                    var f_re = freqs_cis.load[width=1, alignment=1](
                        (
                            Scalar[freqs_cis.linear_idx_type](position_idx),
                            Scalar[freqs_cis.linear_idx_type](pair_base),
                        )
                    )
                    var f_im = freqs_cis.load[width=1, alignment=1](
                        (
                            Scalar[freqs_cis.linear_idx_type](position_idx),
                            Scalar[freqs_cis.linear_idx_type](im_c),
                        )
                    )
                    var result_re = x_re * f_re - x_im * f_im
                    var result_im = x_re * f_im + x_im * f_re
                    var out_val = result_re if is_re else result_im
                    result = out_val.cast[output_dtype]()
                else:
                    var freq_val = freqs_cis.load[width=width, alignment=1](
                        (
                            Scalar[freqs_cis.linear_idx_type](position_idx),
                            Scalar[freqs_cis.linear_idx_type](col),
                        )
                    )
                    result = _rope(nc, freq_val)
            else:
                result = nc
            output_fn[width, alignment](
                idx.coord,
                result,
            )

        row.elementwise(normed, write, load, normalize)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based row_mean_of_squares: the rms_norm reduction core on its own.
# One ReduceSum-of-squares phase, then a scalar `emit` of `sum(x^2) / axis_size`
# (a true reduction — the reduced axis collapses to one value per row). Splits
# input and output dtypes; the square and sum run in the input's accum type
# (fp32 for bf16/fp16). Drop-in signature for the legacy `row_mean_of_squares`
# it replaces (2-arg `input_fn`, `(row, value)` `output_fn`); `reduce_dim`
# defaults to the last axis (what the op needs) but any axis is supported.
# ===----------------------------------------------------------------------=== #


def row_mean_of_squares[
    in_dtype: DType,
    out_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[in_dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[out_dtype, width]) -> None),
    /,
    target: StaticString = "cpu",
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    shape: Coord,
    ctx: DeviceContext,
) raises:
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    comptime assert shape.is_flat, "shape must be flat"
    comptime assert 0 <= reduce_dim < rank, "reduce_dim must be in [0, rank)"
    comptime accum = get_accum_type[in_dtype]()
    comptime assert (
        accum.is_floating_point()
    ), "row_mean_of_squares requires fp accum"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], target, 64, in_dtype, accum
    ]()
    var axis_size = Int(shape[reduce_dim].value())
    var axis_size_accum = Scalar[accum](axis_size)

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var axis_size_accum,
        var input_fn,
        var output_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_fn} -> SIMD[in_dtype, width]:
            return input_fn[width](idx.coord)

        # Prepare Row: this is a true reduction (no fuse-eligible cache), so
        # the axis size is always the dynamic form.
        var row = rowwise.Row[
            params, accum, in_dtype, reduce_dim, row_rank, is_cached=False
        ](row_coords, Int(axis_size), ctx_p, load)

        # Reduce: sum of squares -> mean of squares.
        @always_inline
        def square[
            width: Int
        ](tile: SIMD[in_dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # Mean of squares held W-wide (one value broadcast to every lane);
        # read via `.slice[width]` uniformly in the terminal.
        var mean_sq = (
            row.reduce[ReduceSum[accum, params.simd_width]](square, load).acc
            / axis_size_accum
        )

        # Emit: one value per row, at `oc` (reduced axis pinned to 0).
        @always_inline
        def write(
            oc: RowCoord[row_rank],
        ) {var mean_sq, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                mean_sq.slice[params.emit_tile_width]().cast[out_dtype](),
            )

        # `mean_sq`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
    ](body, shape, ctx)


# ===----------------------------------------------------------------------=== #
# Row-based fused Q/K row_mean_of_squares: composes the single-input
# reduction above twice (Q -> output column 0, K -> column 1). The scaffolder is
# single-primary-input, so the two reductions run as two launches rather than
# legacy's single fused launch; each is an optimal Row reduction. This
# mirrors the op's own definition — "two `row_mean_of_squares` ops plus a
# concat" — and matches the GC IR (two `iter.reduce` phases).
# ===----------------------------------------------------------------------=== #


def row_mean_of_squares_qk[
    in_dtype: DType,
    out_dtype: DType,
    //,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, out_dtype, ...],
    query: TileTensor[mut=False, in_dtype, ...],
    key: TileTensor[mut=False, in_dtype, ...],
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    if rows == 0:
        return

    @always_inline
    def q_in[width: Int](idx: Coord) {var query} -> SIMD[in_dtype, width]:
        return query.load[width=width, alignment=1](idx)

    # Column 0 (q) / column 1 (k) sit stride-2 apart in `output`'s `[rows, 2]`
    # layout, so a `width`-wide batch of adjacent rows can't land in one
    # contiguous vector store; write it back lane by lane instead (`width`
    # is comptime, so this unrolls to `width` scalar stores).
    @always_inline
    def q_out[
        width: SIMDLength
    ](oc: Coord, val: SIMD[out_dtype, width]) {var output}:
        comptime for i in range(width):
            output.store[width=1](Coord(Int(oc[0].value()) + i, 0), val[i])

    row_mean_of_squares[in_dtype, out_dtype, 2, target=target](
        q_in, q_out, Coord(rows, q_cols), ctx
    )

    @always_inline
    def k_in[width: Int](idx: Coord) {var key} -> SIMD[in_dtype, width]:
        return key.load[width=width, alignment=1](idx)

    @always_inline
    def k_out[
        width: SIMDLength
    ](oc: Coord, val: SIMD[out_dtype, width]) {var output}:
        comptime for i in range(width):
            output.store[width=1](Coord(Int(oc[0].value()) + i, 1), val[i])

    row_mean_of_squares[in_dtype, out_dtype, 2, target=target](
        k_in, k_out, Coord(rows, k_cols), ctx
    )


# ===----------------------------------------------------------------------=== #
# Row-based rms_norm + residual add (composite of two fused rms_norms). In
# one launch: reduce sum(input^2) -> inv_rms1; a second phase reduces
# sum(intermediate^2) where `intermediate = rms_norm(input, gamma1) + residual`
# (and emits `intermediate` as the residual output); then a map applies the
# second norm `rms_norm(intermediate, gamma2)` to the output. The intermediate
# is recomputed from the register-cached input + a residual reload rather than
# held in shared memory (legacy's single-kernel trick), so it costs one extra
# pass of memory traffic on large rows.
# ===----------------------------------------------------------------------=== #


def rms_norm_fused_residual_add[
    dtype: DType,
    rank: Int,
    Input0Fn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    Input1Fn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    Output0Fn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    OutputResidualFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
    reduce_dim: Int = rank - 1,
](
    input_0_fn: Input0Fn,
    input_1_fn: Input1Fn,
    output_0_fn: Output0Fn,
    output_residual_fn: OutputResidualFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma1: TileTensor[mut=False, dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[mut=False, dtype, ...],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    context: Optional[DeviceContext] = None,
) raises:
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "rms_norm requires fp accum"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], target, 64, dtype, accum
    ]()
    var axis_size_int = Int(axis_size.value())
    var axis_size_accum = Scalar[accum](axis_size_int)

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma1,
        var epsilon1,
        var weight_offset1,
        var gamma2,
        var epsilon2,
        var weight_offset2,
        var axis_size_accum,
        var input_0_fn,
        var input_1_fn,
        var output_0_fn,
        var output_residual_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_0_fn} -> SIMD[dtype, width]:
            return input_0_fn[width](idx.coord)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # gamma inner (lane) stride: 1 when the reduced axis is inner
        # (consecutive), else 0 (one gamma, splatted). Strided load, no branch.
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        @always_inline
        def square[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # Reduce (phase 1): sum of squares of the input -> inv_rms1.
        var ssq1 = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms1 = rsqrt(ssq1 / axis_size_accum + epsilon1.cast[accum]())

        # Cache: intermediate = rms_norm(input, gamma1) + residual, in `dtype`
        # (matching legacy: the first norm is cast to dtype before the
        # residual add). Emits the residual output on the way through, and is
        # cached once so phase 2 and the terminal read it back from registers
        # instead of recomputing it (and re-loading the residual) twice.
        @always_inline
        def intermediate[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {
            var gamma1,
            var inv_rms1,
            var input_1_fn,
            var weight_offset1,
            var output_residual_fn,
            var ctx_p,
        } -> SIMD[dtype, width]:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var col = Int(idx.coord[reduce_dim].value())
            var gamma1_val = strided_load[width, g_stride, alignment=alignment](
                gamma1.ptr_at_offset(Coord(col))
            )
            var normed = tile.cast[accum]() * inv_rms1.slice[width]()
            var residual = input_1_fn[width](idx.coord)

            var inter: SIMD[dtype, width]
            comptime if multiply_before_cast:
                var gamma1_with_offset = (
                    gamma1_val.cast[accum]() + weight_offset1.cast[accum]()
                )
                inter = (normed * gamma1_with_offset).cast[dtype]() + residual
            else:
                var gamma1_with_offset = gamma1_val + weight_offset1
                inter = normed.cast[dtype]() * gamma1_with_offset + residual
            output_residual_fn[width, alignment](
                idx.coord,
                inter,
            )
            return inter

        var inter = row.cache[dtype](intermediate)

        @always_inline
        def square_intermediate[
            width: Int
        ](staged_tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            var staged_accum = staged_tile.cast[accum]()
            return staged_accum * staged_accum

        # Reduce (phase 2): sum of squares of the (staged) intermediate.
        # `square_intermediate` crosses into the cached `reduce` overload as a
        # trailing value arg. `load`/`intermediate` (the producer that built
        # `inter`) ride along for the non-fuse recompute fallback.
        var ssq2 = row.reduce[ReduceSum[accum, params.simd_width]](
            inter, square_intermediate, load, intermediate
        ).acc
        var inv_rms2 = rsqrt(ssq2 / axis_size_accum + epsilon2.cast[accum]())

        # Emit: apply the second norm to the (staged) intermediate.
        @always_inline
        def write[
            width: Int
        ](staged_tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {
            var inv_rms2,
            var gamma2,
            var weight_offset2,
            var output_0_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var col = Int(idx.coord[reduce_dim].value())
            var gamma2_val = strided_load[width, g_stride, alignment=alignment](
                gamma2.ptr_at_offset(Coord(col))
            )
            var normed = staged_tile.cast[accum]() * inv_rms2.slice[width]()

            comptime if multiply_before_cast:
                var gamma2_with_offset = (
                    gamma2_val.cast[accum]() + weight_offset2.cast[accum]()
                )
                var result = (normed * gamma2_with_offset).cast[dtype]()
                output_0_fn[width, alignment](
                    idx.coord,
                    result,
                )
            else:
                var gamma2_with_offset = gamma2_val + weight_offset2
                var result = normed.cast[dtype]() * gamma2_with_offset
                output_0_fn[width, alignment](
                    idx.coord,
                    result,
                )

        # `write` crosses into the cached `elementwise` overload as a
        # trailing value arg; `inv_rms2`/`gamma2`/`weight_offset2`/
        # `output_0_fn` ride its captures. `load`/`intermediate` ride along
        # for the non-fuse recompute fallback.
        row.elementwise(inter, write, load, intermediate)
        # `inv_rms1` read only inside the comptime `intermediate` cache
        # closure; discard-read marks it used.
        _ = inv_rms1

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based rms_norm + dynamic-scaled FP8 quantization: sum-of-squares and
# max(|gamma*x|) reduced in one cached pass, then a per-row FP8 scale is emitted
# and the normalized row is quantized to FP8.
# ===----------------------------------------------------------------------=== #


def rms_norm_fused_quantize_dynamic_scaled_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scale_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[in_dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[out_dtype, width]) -> None
    ),
    ScaleFn: ImplicitlyCopyable
    & RegisterPassable
    & (def(Coord, Scalar[scale_dtype]) -> None),
    AxisSizeT: CoordLike,
    /,
    target: StaticString,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    scale_fn: ScaleFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    context: Optional[DeviceContext] = None,
) raises:
    comptime accum = get_accum_type[in_dtype]()
    comptime assert accum.is_floating_point(), "rms_norm requires fp accum"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], target, 64, in_dtype, accum
    ]()
    var axis_size_accum = Scalar[accum](Int(axis_size.value()))

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var epsilon,
        var weight_offset,
        var scale_ub,
        var axis_size_accum,
        var input_fn,
        var output_fn,
        var scale_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_fn} -> SIMD[in_dtype, width]:
            return input_fn[width, alignment](idx.coord)

        var row = rowwise.Row[
            params, accum, in_dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # gamma indexed by the reduced axis: inner stride 1 when that axis is
        # inner (consecutive), else 0 (splat).
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Plain non-capturing helper: receives the gamma tensor + weight offset
        # as arguments (captures no runtime value), so the value-closure
        # callbacks below can call it by name while capturing only plain values.
        @always_inline
        def gamma_load[
            width: Int
        ](
            gamma_tensor: TileTensor[mut=False, in_dtype, ...],
            col: Int,
            weight_offset: Scalar[in_dtype],
        ) -> SIMD[accum, width]:
            var gamma_raw = strided_load[
                width, g_stride, alignment=align_of[SIMD[in_dtype, width]]()
            ](gamma_tensor.ptr_at_offset(Coord(col)))
            return gamma_raw.cast[accum]() + weight_offset.cast[accum]()

        @always_inline
        def square[
            width: Int
        ](tile: SIMD[in_dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # Reduce (phase 1): mean of squares of the input -> inv_rms.
        var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms = rsqrt(ssq / axis_size_accum + epsilon.cast[accum]())

        # Reduce (phase 2): max(|gamma*x|) over the same cached row. The norm
        # factor is a positive scalar, so max(|gamma*x*inv_rms|) ==
        # max(|gamma*x|)*inv_rms; the FP8 scale is derived from the raw input
        # without staging the normalized row.
        @always_inline
        def abs_gamma_x[
            width: Int
        ](tile: SIMD[in_dtype, width], idx: RowCoord[row_rank]) {
            var gamma, var weight_offset
        } -> SIMD[accum, width]:
            return abs(
                tile.cast[accum]()
                * gamma_load[width](
                    gamma, Int(idx.coord[reduce_dim].value()), weight_offset
                )
            )

        # `gamma`/`weight_offset` ride `abs_gamma_x`'s captures into `reduce`;
        # `gamma_load` is called by name (captures nothing).
        var row_abs_max = row.reduce[ReduceMax[accum, params.simd_width]](
            abs_gamma_x, load
        ).acc
        var row_max = (row_abs_max * inv_rms).reduce_max()
        var scale_result = compute_dynamic_fp8_scale[out_dtype](
            row_max, scale_ub.cast[scale_dtype]()
        )
        var scale_factor = scale_result[0]
        var scale_recip = scale_result[1]

        # Emit (scale): per-row output, the dynamic FP8 scale (reduced axis
        # pinned to 0).
        @always_inline
        def write_scale(
            oc: RowCoord[row_rank],
        ) {var scale_factor, var scale_fn}:
            scale_fn(
                oc.coord,
                scale_factor,
            )

        # `scale_factor`/`scale_fn` ride `write_scale`'s captures into `emit`.
        row.emit(write_scale)

        # Emit (per-element): normalize, scale by gamma, quantize to FP8.
        @always_inline
        def write[
            width: Int
        ](tile: SIMD[in_dtype, width], idx: RowCoord[row_rank]) {
            var gamma,
            var weight_offset,
            var inv_rms,
            var scale_recip,
            var output_fn,
            var ctx_p,
        }:
            var normed = (
                tile.cast[accum]() * inv_rms.slice[width]()
            ) * gamma_load[width](
                gamma, Int(idx.coord[reduce_dim].value()), weight_offset
            )
            var out_fp8 = fp8_quantize[out_dtype](normed, scale_recip)
            comptime alignment = ctx_p.element_alignment[out_dtype, width]()
            output_fn[width, alignment](
                idx.coord,
                out_fp8,
            )

        # `gamma`/`weight_offset`/`inv_rms`/`scale_recip`/`output_fn` ride
        # `write`'s captures into `elementwise`.
        row.elementwise(write, load)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)
