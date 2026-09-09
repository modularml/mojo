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
"""Cluster-launched top-k/top-p kernels.

One block per row uses only as many SMs as there are rows, which leaves most
of the GPU idle at decode batch sizes. The kernels here spread each row over
the CTAs of one thread-block cluster and combine the per-row reductions over
distributed shared memory. The launchers fall back to the single-block kernels
in `topk_fi` when a CTA's slice does not fit in shared memory or when the
target has no clusters.
"""

from std.bit import next_power_of_two
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.primitives import warp
from max.gpu.primitives.id import cluster_dim
from max.gpu.primitives import block
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_allreduce,
    cluster_allgather,
    cluster_sync,
)
from max.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
    PDLLevel,
)
from max.gpu.host import DeviceContext, Dim, FuncAttribute
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from std.sys.info import is_apple_gpu
from layout import (
    ComptimeInt,
    Coord,
    Idx,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.tile_layout import Layout
from std.math import ceildiv, exp, gcd
from std.memory import bitcast, unsafe_stack_allocation
from max.runtime.tracing import Trace, TraceLevel, trace_arg
from std.sys import size_of
from std.utils.static_tuple import StaticTuple
from std.random import Random
from .topk_fi import (
    TopKTopPMaskedProbsKernel,
    TopKTopPSamplingFromProbKernel,
    ValueCount,
    device_sampling_from_prob,
    _block_reduce_value_count,
    _topp_budget,
)

# The largest cluster that these kernels use. All CTAs of a cluster must run
# at the same time on one GPC. Above four CTAs the scheduler gives back more
# than the added parallelism supplies.
comptime _MAX_CUTOFF_CLUSTER = 4

# The largest shared-memory request that a CTA makes to stage its part of a
# row. B200 supplies 228 KiB per SM. Keep some for the reduction scratch and
# the publish slots.
comptime _MAX_STAGE_SMEM_BYTES = 216 * 1024


def _stage_smem_bytes(d: Int, vec_size: Int, cluster_size: Int) -> Int:
    """Bytes one CTA needs to stage its contiguous slice of the row."""
    return (
        ceildiv(ceildiv(d, vec_size), cluster_size)
        * vec_size
        * size_of[DType.float32]()
    )


@always_inline
@__parameter
def _max(x: SIMD, y: type_of(x)) -> type_of(x):
    return max(x, y)


@always_inline
@__parameter
def _sum(x: SIMD, y: type_of(x)) -> type_of(x):
    return x + y


comptime _CUTOFF_SEARCH_MAX_ITERS = 64

# A publish slot holds the widest vector that the CTAs of a cluster combine,
# twice over: once for the peers to read, and once for the combined result of
# this CTA.
comptime _CLUSTER_SLOT_FLOATS = 16


@always_inline
def _block_reduce_cutoff_stats[
    block_size: Int, n: Int, broadcast: Bool = True
](vals: StaticTuple[Float32, n]) -> StaticTuple[Float32, n]:
    """Reduces one round of cutoff statistics across the block in one pass.

    Lane 0 holds a minimum, lane 1 a maximum, and the remaining lanes hold
    sums. Counts are carried as floats, which is exact below 2^24.
    """

    @always_inline
    @__parameter
    def _reduce_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx == 0:
            return warp.min(v)
        elif reduction_idx == 1:
            return warp.max(v)
        else:
            return warp.sum(v)

    var initial = StaticTuple[Float32, n](0)
    initial[0] = Float32.MAX_FINITE
    initial[1] = Float32.MIN_FINITE

    return block._block_reduce[
        block_size, warp_reduce_fn=_reduce_fn, broadcast=broadcast
    ](vals, initial_vals=initial)


@always_inline
def _cluster_cutoff_search[
    vec_size: Int,
    block_size: Int,
    cluster_size: Int,
](
    d: Int,
    k: Int32,
    p_eff: Float32,
    low_init: Float32,
    high_init: Float32,
    mass_above_low_init: Float32,
    staged: UnsafePointer[mut=True, Float32, _, address_space=.SHARED],
    vec_begin: Int,
    vec_end: Int,
) -> Tuple[Float32, Float32]:
    """`_topk_topp_cutoff_search` with the row spread over a cluster.

    Same constraint, contract and value-snapping termination as the
    single-block search in `topk_fi`: a token survives iff `count(> t) < k`
    and `mass(> t) <= p_eff`, the returned cutoff is exact, and callers must
    guarantee the predicate fails at `low_init`, holds at `high_init`, and
    `mass_above_low_init == mass(> low_init)`.

    Three things differ from the single-block form, all tied to the cluster
    launch:

    - Each CTA scans only its half-open vector range `[vec_begin, vec_end)`,
      reading the slice it staged in shared memory directly -- there is no
      load callback, because a closure that captures a dynamic shared-memory
      pointer returns the wrong data.
    - The per-round statistics combine across the cluster in rank order, so
      every CTA holds bit-identical values, takes the same branch, and stops
      on the same round -- a disagreeing CTA would run a different number of
      cluster barriers. The block reduction does not broadcast: thread 0
      publishes to the peers, and the cluster combine supplies the result to
      the rest of the block. Consecutive rounds alternate between two publish
      slots, so neither needs a sync to retire it.
    - Pivots bisect in the float bit domain rather than the value domain,
      which roughly halves the rounds to converge over a distribution that
      spans many orders of magnitude.

    Returns:
        ``(cutoff, kept_mass)`` where ``kept_mass == mass(> cutoff)``. On the
        (defensive) iteration cap, returns the current bracket state, which
        keeps a superset of the constraint set but stays self-consistent.
    """
    var low = low_init
    var high = high_init
    var mass_above_low = mass_above_low_init

    var g_begin = vec_begin + Int(thread_idx.x)

    var cluster_slot = unsafe_stack_allocation[
        2 * _CLUSTER_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=.SHARED,
    ]()
    var phase = 0

    @always_inline
    @__parameter
    def _cutoff_stats_combine(x: SIMD, y: type_of(x)) -> type_of(x):
        # Same lane layout as `_block_reduce_cutoff_stats`, padded to a
        # power of two.
        var r = x + y
        r[0] = min(x[0], y[0])
        r[1] = max(x[1], y[1])
        return r

    for _ in range(_CUTOFF_SEARCH_MAX_ITERS):
        # Pivots split the bracket in the float bit domain: non-negative
        # floats order the same in value and bit space, and the working
        # distribution spans many orders of magnitude, so thirds of the bit
        # range cross an exponent per step where thirds of the value range
        # walk `high` down one exponent at a time. `high > low` here, so the
        # span cannot underflow.
        var lo_bits = max(low, Float32(0)).to_bits[.uint32]()
        var hi_bits = max(high, Float32(0)).to_bits[.uint32]()
        var span = hi_bits - lo_bits
        # A span below three bits can round both pivots down to `low` and
        # stall the search.
        var third = span // 3
        var off_0 = max(third, UInt32(1))
        var off_1 = max(2 * third, off_0 + 1)
        var pivot_0 = bitcast[.float32](lo_bits + off_0)
        var pivot_1 = bitcast[.float32](lo_bits + off_1)

        # Accumulate thread-local counts/masses across the slice. The
        # accumulators stay scalar on purpose: at block_size 1024 only 64
        # registers per thread are available, and SIMD-wide accumulators
        # spill.
        var thread_count_0 = Float32(0)
        var thread_count_1 = Float32(0)
        var thread_mass_0 = Float32(0)
        var thread_mass_1 = Float32(0)
        var min_gt_low = high
        var max_le_high = low

        for g in range(g_begin, vec_end, block_size):
            var v = staged.load[width=vec_size]((g - vec_begin) * vec_size)
            comptime for j in range(vec_size):
                var e = v[j]
                if e > pivot_0:
                    thread_count_0 += 1
                    thread_mass_0 += e
                if e > pivot_1:
                    thread_count_1 += 1
                    thread_mass_1 += e
                if e > low:
                    min_gt_low = min(min_gt_low, e)
                if e <= high:
                    max_le_high = max(max_le_high, e)

        var stats = _block_reduce_cutoff_stats[block_size, 6, broadcast=False](
            StaticTuple[Float32, 6](
                min_gt_low,
                max_le_high,
                thread_mass_0,
                thread_mass_1,
                thread_count_0,
                thread_count_1,
            )
        )
        var packed = SIMD[.float32, 8](0)
        comptime for i in range(6):
            packed[i] = stats[i]
        var combined = cluster_allreduce[
            _cutoff_stats_combine, cluster_size, need_tail_sync=False
        ](cluster_slot + phase * _CLUSTER_SLOT_FLOATS, packed)
        phase ^= 1

        min_gt_low = combined[0]
        max_le_high = combined[1]
        var mass_0 = combined[2]
        var mass_1 = combined[3]
        var count_0 = Int32(combined[4])
        var count_1 = Int32(combined[5])

        # pivot_1 > pivot_0: if the constraint still fails above the higher
        # pivot it also fails above the lower one, so test high-to-low.
        if count_1 >= k or mass_1 > p_eff:
            low = pivot_1
            mass_above_low = mass_1
        elif count_0 >= k or mass_0 > p_eff:
            low = pivot_0
            mass_above_low = mass_0
            high = min(pivot_1, max_le_high)
        else:
            high = min(pivot_0, max_le_high)

        # Exactly one distinct data value remains in (low, high]: every token
        # above `low` passes the predicate, every token at or below fails.
        if min_gt_low == max_le_high or high <= low:
            break

    return Tuple[Float32, Float32](low, mass_above_low)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(t"topk_topp_masked_probs_cluster_{dtype}_{cluster_size}")
def TopKTopPMaskedProbsClusterKernel[
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmOrigin,
    cluster_size: Int,
    LogitsEngine: TensorEngine,
](
    logits: TileTensor[
        dtype, LogitsLayoutType, logits_origin, Engine=LogitsEngine
    ],
    probs_ptr: UnsafePointer[Float32, MutAnyOrigin],
    top_k_arr: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
    top_k_val: Int32,
    top_p_arr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    top_p_val: Float32,
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    d: Int32,
):
    """`TopKTopPMaskedProbsKernel` with one row spread over a cluster.

    Works in the unnormalized domain `e_i = exp((logit_i - row_max) / temp)`:
    a token survives the joint constraint iff `e > cutoff` (recovered by the
    same dual-pivot search the sampler uses) and its masked probability is
    `e / kept_mass`. The output row is that masked renormalized distribution
    -- the same tensor `TopKTopPSamplingFromProbKernel` emits under
    `emit_dist`, so a verifier's target-side probabilities and a draft's
    proposal distribution are described identically.

    One block per row uses only as many SMs as there are rows, which leaves
    most of the GPU idle at decode batch sizes. Here the CTAs of one cluster
    share the row. Each CTA reduces its own contiguous slice, and the cluster
    combines the whole-row reductions that the cutoff needs.

    The launch must set `cluster_dim` to `cluster_size`.
    """
    comptime assert (
        not is_apple_gpu()
    ), "TopKTopPMaskedProbsClusterKernel is not supported on Apple GPUs"
    var _d = Int(d)
    var tx = Int(thread_idx.x)

    debug_assert(
        Int(cluster_dim.x) == cluster_size,
        "launch cluster_dim must match the kernel's cluster_size",
    )

    # A cluster covers one row, and each CTA takes a contiguous range of
    # vectors: staging in shared memory needs each CTA's elements to be a
    # compact range it can address as `g - vec_begin`, and contiguity keeps
    # the allocation at exactly `ceil(d / cluster_size)` elements.
    var rank = Int(block_rank_in_cluster())
    var bx = Int(block_idx.x) // cluster_size
    var n_vec = _d // vec_size
    var slice_vec = ceildiv(n_vec, cluster_size)
    var vec_begin = min(rank * slice_vec, n_vec)
    var vec_end = min(vec_begin + slice_vec, n_vec)

    # The softmax weights are staged in dynamic shared memory, so the search
    # re-reads them from the SM rather than from L2 on every iteration. The
    # host guarantees the slice fits before choosing this kernel; a row too
    # wide for the budget goes to the single-block kernel instead.
    var smem = external_memory[Float32, address_space=.SHARED, alignment=16]()

    # Publish slots for this kernel's own cross-CTA combines, used in turn so
    # neither needs a sync to retire it. The cutoff search brings its own
    # pair.
    var cluster_slot = unsafe_stack_allocation[
        2 * _CLUSTER_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=.SHARED,
    ]()

    wait_on_dependent_grids()
    launch_dependent_grids()

    var k = Int(top_k_val)
    if top_k_arr:
        k = Int(top_k_arr.unsafe_value().load(bx))
    if k <= 0 or k > _d:
        k = _d

    var p = top_p_val
    if top_p_arr:
        p = top_p_arr.unsafe_value()[bx]
    p = p.clamp(Float32(0.0), Float32(1.0))

    var temp_val = Float32(1.0)
    if temperature:
        temp_val = temperature.unsafe_value()[bx]
    # Clamp so a greedy (T=0) row cannot divide by zero.
    var inv_temp = 1.0 / max(temp_val, Float32(1e-6))

    var logits_row = TileTensor(logits.ptr + bx * _d, row_major(Idx[1], _d))

    # Row max, combined across the cluster in rank order so every CTA holds
    # the same bits. Every value below derives from it, so one disagreeing
    # CTA would branch differently through the search. The block reduction
    # does not broadcast: thread 0 publishes to the peers, and the cluster
    # combine supplies the result to the rest of the block.
    var thread_max = Float32.MIN
    for i in range(vec_begin + tx, vec_end, block_size):
        var v = logits_row.load[width=vec_size]((Idx[0], i * vec_size)).cast[
            DType.float32
        ]()
        thread_max = max(thread_max, v.reduce_max())

    var m = cluster_allreduce[_max, cluster_size, need_tail_sync=False](
        cluster_slot,
        Float32(block.max[block_size=block_size, broadcast=False](thread_max)),
    )[0]

    @__parameter
    @always_inline
    def load_e(offset: Int) -> SIMD[.float32, vec_size]:
        var v = logits_row.load[width=vec_size]((Idx[0], offset)).cast[
            DType.float32
        ]()
        return exp((v - m) * inv_temp)

    var probs_row = TileTensor(probs_ptr + bx * _d, row_major(Idx[1], _d))

    # Total mass, plus how many tokens carry any: a row whose every positive
    # token already satisfies the constraint has no boundary to find, and the
    # search's precondition (the predicate fails at 0) would not hold.
    #
    # The same pass stages `e` into this CTA's slice of shared memory. The
    # cutoff search reads the working distribution once per iteration and
    # would otherwise rebuild it each time -- a load, an FMA and a
    # transcendental per element per pass -- so staging pays the exp once.
    var thread_sum = Float32(0)
    var thread_pos: Int32 = 0
    for i in range(vec_begin + tx, vec_end, block_size):
        var e = load_e(i * vec_size)
        smem.store[width=vec_size]((i - vec_begin) * vec_size, e)
        thread_sum += e.reduce_add()
        comptime for j in range(vec_size):
            if e[j] > 0:
                thread_pos += 1
    var block_total = _block_reduce_value_count[.float32, broadcast=False](
        ValueCount[.float32](thread_sum, thread_pos)
    )

    var totals = cluster_allreduce[_sum, cluster_size, need_tail_sync=False](
        cluster_slot + _CLUSTER_SLOT_FLOATS,
        SIMD[.float32, 2](block_total.value, Float32(block_total.count)),
    )
    var z = totals[0]
    var total_count = Int32(totals[1])
    var p_eff = _topp_budget(p, z)

    var cut = Float32(0)
    var mass_s = z
    if total_count >= Int32(k) or z > p_eff:
        # The search reads elements other lanes staged, so order the writes
        # against it -- cluster-wide, since the search's own combines assume
        # every CTA has arrived.
        cluster_sync()
        var refined = _cluster_cutoff_search[
            vec_size, block_size, cluster_size
        ](
            _d,
            Int32(k),
            p_eff,
            0.0,
            1.0,
            z,
            smem,
            vec_begin,
            vec_end,
        )
        cut = refined[0]
        mass_s = refined[1]

    # Masks the staged weights; each lane owns the elements it wrote (the
    # same partition as the staging pass), so this needs no barrier against
    # the search.
    for i in range(vec_begin + tx, vec_end, block_size):
        var e = smem.load[width=vec_size]((i - vec_begin) * vec_size)
        var masked = (e.gt(cut)).select(e / mass_s, SIMD[.float32, vec_size](0))
        probs_row.store[width=vec_size]((Idx[0], i * vec_size), masked)

    # Terminal keep-alive. Peers read this CTA's shared memory through `mapa`,
    # so it may not retire until every CTA of the cluster is done reading --
    # letting one exit early is an illegal access, not just a stale read.
    cluster_sync()


def topk_topp_masked_probs_cluster[
    dtype: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopPArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    ProbsLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64, Int64].element_types,
        stride_types=Coord[Int64, ComptimeInt[1]].element_types,
    ],
    TopKArrEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPArrEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    logits: TileTensor[mut=False, dtype, ...],
    probs: TileTensor[.float32, ProbsLayoutType, MutAnyOrigin],
    top_k_val: Int,
    top_p_val: Float32 = 1.0,
    top_k_arr: Optional[
        TileTensor[
            .int64, TopKArrLayoutType, ImmutAnyOrigin, Engine=TopKArrEngine
        ]
    ] = None,
    top_p_arr: Optional[
        TileTensor[
            .float32, TopPArrLayoutType, ImmutAnyOrigin, Engine=TopPArrEngine
        ]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
) raises:
    """Computes per-row top-k/top-p masked softmax on a cluster device.

    See `TopKTopPMaskedProbsKernel` for what the output means. The package's
    `topk_topp_masked_probs` dispatcher routes here on NVIDIA SM90+ devices;
    a batch that fills the machine or a slice too wide for shared memory
    still falls back to the single-block kernel at runtime.

    Parameters:
        dtype: Element type of `logits`.
        block_size: Threads per block.
        TopKArrLayoutType: Memory layout of `top_k_arr`.
        TopPArrLayoutType: Memory layout of `top_p_arr`.
        TemperatureLayoutType: Memory layout of `temperature`.
        ProbsLayoutType: Memory layout of `probs`.
        TopKArrEngine: Engine policy of `top_k_arr`.
        TopPArrEngine: Engine policy of `top_p_arr`.
        TemperatureEngine: Engine policy of `temperature`.

    Args:
        ctx: Device context.
        logits: Input logits [batch_size, d].
        probs: Output masked renormalized distribution [batch_size, d].
        top_k_val: Default top-k; `<= 0` or `> d` keeps every token.
        top_p_val: Default top-p threshold.
        top_k_arr: Optional per-row top-k [batch_size].
        top_p_arr: Optional per-row top-p [batch_size].
        temperature: Optional per-row temperature [batch_size]; 0 is clamped.

    Raises:
        Error: If the tensor shapes disagree.
    """
    comptime assert logits.rank == 2, "logits rank must be 2"

    var shape = coord_to_index_list(logits.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("logits", shape, dtype),
                    "top_k=" + String(top_k_val),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_topp_masked_probs",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var probs_shape = coord_to_index_list(probs.layout.shape_coord())
        if probs_shape[0] != batch_size or probs_shape[1] != d:
            raise Error("probs shape must match the logits shape")

        # Speculative decoding runs this with zero rows on every step that has
        # no drafts to verify, and a grid of 0 is not a legal launch.
        if batch_size == 0:
            return

        var vec_size = gcd(8, d)

        var top_k_ptr: Optional[UnsafePointer[Int64, ImmutAnyOrigin]] = None
        if top_k_arr:
            top_k_ptr = top_k_arr.unsafe_value().ptr

        var top_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
        if top_p_arr:
            top_p_ptr = top_p_arr.unsafe_value().ptr

        var temperature_ptr: Optional[
            UnsafePointer[Float32, ImmutAnyOrigin]
        ] = None
        if temperature:
            temperature_ptr = temperature.unsafe_value().ptr

        # A row takes the narrowest cluster whose slice fits the staging
        # budget: one CTA for small vocabularies (staged, no cluster
        # traffic), two when the row must split, four for the widest. A row
        # too wide for four slices takes the single-block kernel with no
        # staging.
        var cluster_size = 0
        comptime for c in [1, 2, _MAX_CUTOFF_CLUSTER]:
            if (
                cluster_size == 0
                and _stage_smem_bytes(d, vec_size, c) <= _MAX_STAGE_SMEM_BYTES
            ):
                cluster_size = c

        @__parameter
        def launch_cluster[vec_size: Int, cluster_size: Int]() raises:
            comptime kernel = TopKTopPMaskedProbsClusterKernel[
                block_size,
                vec_size,
                dtype,
                logits.LayoutType,
                ImmOrigin(logits.origin),
                cluster_size,
                LogitsEngine=logits.Engine,
            ]
            var smem_bytes = _stage_smem_bytes(d, vec_size, cluster_size)
            ctx.enqueue_function[kernel](
                logits.as_immut(),
                probs.ptr,
                top_k_ptr,
                Int32(top_k_val),
                top_p_ptr,
                top_p_val,
                temperature_ptr,
                Int32(d),
                grid_dim=batch_size * cluster_size,
                block_dim=block_size,
                cluster_dim=Dim(cluster_size, 1, 1),
                shared_mem_bytes=smem_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem_bytes)
                ),
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        @__parameter
        def launch_single[vec_size: Int]() raises:
            comptime kernel = TopKTopPMaskedProbsKernel[
                block_size,
                vec_size,
                dtype,
                logits.LayoutType,
                ImmOrigin(logits.origin),
            ]
            ctx.enqueue_function[kernel](
                logits.as_immut(),
                probs.ptr,
                top_k_ptr,
                Int32(top_k_val),
                top_p_ptr,
                top_p_val,
                temperature_ptr,
                Int32(d),
                Optional[UnsafePointer[Int32, MutAnyOrigin]](None),
                grid_dim=batch_size,
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        # `vec_size = gcd(8, d)`, so 8 is the widest case.
        comptime for param_vec_size in [8, 4, 2, 1]:
            if vec_size == param_vec_size:
                comptime for param_cluster in [1, 2, _MAX_CUTOFF_CLUSTER]:
                    if cluster_size == param_cluster:
                        return launch_cluster[param_vec_size, param_cluster]()
                return launch_single[param_vec_size]()


@always_inline
def _block_reduce_sums[
    block_size: Int, n: Int, broadcast: Bool = True
](vals: StaticTuple[Float32, n]) -> StaticTuple[Float32, n]:
    """Reduces `n` independent sums across the block in one pass.

    Not `block.sum`: that reduces a vector's lanes into one total before the
    block reduction, while this keeps each lane a separate sum.
    """

    @always_inline
    @__parameter
    def _reduce_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        return warp.sum(v)

    return block._block_reduce[
        block_size, warp_reduce_fn=_reduce_fn, broadcast=broadcast
    ](vals, initial_vals=StaticTuple[Float32, n](0))


@always_inline
def _sampling_rejection_loop_cluster[
    vec_size: Int,
    block_size: Int,
    dtype: DType,
    deterministic: Bool,
    cluster_size: Int,
](
    d: Int,
    k: Int,
    p_eff: Float32,
    z: Float32,
    mut generator: Random,
    smem: UnsafePointer[mut=True, Float32, _, address_space=.SHARED],
    rank: Int,
    vec_begin: Int,
    vec_end: Int,
) -> Tuple[Int, Float32, Float32, Float32]:
    """Draws one token under the joint top-k/top-p constraint, per cluster.

    Each trial draws a target `u` inside the remaining mass and walks the CDF
    of the weights above `low` to find the crossing token. That prefix sum
    decomposes across the cluster: every CTA sums its staged slice, a gather
    gives each CTA every rank's sum -- rank order is row order, because the
    slices are contiguous -- and only the CTA whose prefix interval contains
    `u` scans for the crossing, over its own slice in shared memory. The
    pivot masses that accept or shrink the bracket combine across the cluster
    the same way the cutoff search's statistics do.

    Exactly one CTA reports a crossing, so the per-lane max combine below
    cannot pair one rank's index with another rank's weight; the accept
    invariant `low < high <= accepted_e` depends on that pairing staying
    exact. Every branch decision reads cluster-combined values, so all CTAs
    run the same trial count and the same number of cluster barriers.

    Returns `(sampled_id, low, accepted_e, q)`: the sampled index, the floor
    of the final bracket, the weight of the accepted token (-1 if the row is
    degenerate), and the mass above `low`. A cutoff search that starts from
    `(low, accepted_e]` with budget `q` refines the bracket to the exact
    constraint boundary, and the sampled token stays above any cutoff that
    search returns.

    Parameters:
        vec_size: Number of elements each thread loads per access.
        block_size: Number of threads per block.
        dtype: Element type of the underlying distribution.
        deterministic: If True, take the deterministic sampling path.
        cluster_size: Number of CTAs sharing the row.

    Args:
        d: Row width.
        k: Top-k bound; pass `d` when top-k does not constrain the row.
        p_eff: Top-p budget, scaled by the total mass of the row.
        z: Total mass of the row; the first draw's budget.
        generator: Per-row random generator. Every CTA holds an identical
            copy and advances it in lockstep, once per trial.
        smem: This CTA's staged slice of the working distribution.
        rank: This CTA's rank within the cluster.
        vec_begin: First vector of this CTA's contiguous slice.
        vec_end: One past the last vector of the slice.
    """
    var tx = Int(thread_idx.x)
    var sampled_id_sram = unsafe_stack_allocation[
        1, Int, address_space=.SHARED
    ]()

    # Publish slots for this loop's combines, sized for the widest (the
    # rank gather) and used in turn: three combines per trial, so consecutive
    # trials keep alternating.
    comptime slot_floats = 4 * (1 + next_power_of_two(cluster_size))
    var cluster_slot = unsafe_stack_allocation[
        2 * slot_floats,
        Float32,
        alignment=16,
        address_space=.SHARED,
    ]()
    var phase = 0

    var sampled_id = 0
    var low = Float32(0)
    var high = Float32(1)
    var q = z
    var accepted_e = Float32(-1)

    while low < high:
        var u = generator.step_uniform()[0] * q

        # Slice statistics over the staged weights: the mass above `low`,
        # the slice's last index above it, and that element's weight -- the
        # fallback token when no crossing is found. Indices travel as floats,
        # which is exact below 2^24.
        var thread_sum = Float32(0)
        var thread_last = Float32(-1)
        for g in range(vec_begin + tx, vec_end, block_size):
            var v = smem.load[width=vec_size]((g - vec_begin) * vec_size)
            comptime for j in range(vec_size):
                if v[j] > low:
                    thread_sum += v[j]
                    thread_last = Float32(g * vec_size + j)
        # Lane 0 is the reduce's min slot, unused here; feeding its identity
        # lets the cutoff-stats reduce serve unchanged.
        var slice_stats = _block_reduce_cutoff_stats[
            block_size, 3, broadcast=False
        ](StaticTuple[Float32, 3](Float32.MAX_FINITE, thread_last, thread_sum))
        var slice_last = slice_stats[1]
        var slice_last_val = Float32(0)
        if tx == 0 and slice_last >= 0:
            slice_last_val = smem[Int(slice_last) - vec_begin * vec_size]

        var published = SIMD[.float32, 4](
            slice_stats[2], slice_last, slice_last_val, 0
        )
        var table = cluster_allgather[cluster_size, need_tail_sync=False](
            cluster_slot + phase * slot_floats, published
        )
        phase ^= 1

        # Rank prefixes in rank order are row-order prefixes. The owner is
        # the first rank whose interval contains `u`; the fallback token is
        # the last one above `low` anywhere in the row.
        var prefix = Float32(0)
        var owner = -1
        var owner_base = Float32(0)
        var global_last = Float32(-1)
        var global_last_val = Float32(0)
        comptime for r in range(cluster_size):
            var r_sum = table[r * 4 + 0]
            var r_last = table[r * 4 + 1]
            if owner < 0 and u < prefix + r_sum:
                owner = r
                owner_base = prefix
            prefix += r_sum
            if r_last > global_last:
                global_last = r_last
                global_last_val = table[r * 4 + 2]

        # The owner CTA scans its slice for the crossing; the others wait at
        # the combine below. Exactly one CTA contributes, which keeps the
        # (index, weight) pairing exact across the per-lane max combine.
        if tx == 0:
            sampled_id_sram[0] = d
        barrier()
        var found = SIMD[.float32, 2](-1, 0)
        if rank == owner:
            var aggregate = owner_base
            var slice_vecs = vec_end - vec_begin
            for i in range(ceildiv(slice_vecs, block_size)):
                var probs_vec = SIMD[.float32, vec_size](0)
                var g = i * block_size + tx
                if g < slice_vecs:
                    probs_vec = smem.load[width=vec_size](g * vec_size)
                # The scan runs in slice-local indices: lanes past the slice
                # carry zeros and fail the `> low` predicate, so the row-wide
                # `d` bound is harmless, and the found index converts to
                # row-global below.
                var result = device_sampling_from_prob[
                    vec_size, block_size, dtype, deterministic
                ](i, d, low, u, probs_vec, aggregate, sampled_id_sram)
                aggregate = result[0]
                if aggregate > u:
                    break
            barrier()
            var sid = sampled_id_sram[0]
            if sid < d:
                found[0] = Float32(sid + vec_begin * vec_size)
                found[1] = smem[sid]

        var comb = cluster_allreduce[_max, cluster_size, need_tail_sync=False](
            cluster_slot + phase * slot_floats, found
        )
        phase ^= 1

        var pivot_0: Float32
        if comb[0] >= 0:
            sampled_id = Int(comb[0])
            pivot_0 = comb[1]
        elif global_last >= 0:
            # The scan missed, or rounding put `u` beyond the summed mass:
            # fall back to the last token above `low`, as the single-block
            # loop does.
            sampled_id = Int(global_last)
            pivot_0 = global_last_val
        else:
            # Degenerate row: nothing above `low` anywhere. From logits, one
            # non-finite value does it -- `row_max` goes +inf, so every
            # weight is exp(inf-inf)=NaN or exp(-inf)=0 and no comparison can
            # be true. Emit an in-range index and stop.
            sampled_id = 0
            break

        var pivot_1 = (pivot_0 + high) / 2.0

        # Masses and counts above both pivots in one pass and one combine. A
        # cross-CTA combine costs too much to defer pivot_1 the way the
        # single-block loop defers its second block reduce.
        var thread_mass_0 = Float32(0)
        var thread_count_0 = Float32(0)
        var thread_mass_1 = Float32(0)
        var thread_count_1 = Float32(0)
        for g in range(vec_begin + tx, vec_end, block_size):
            var v = smem.load[width=vec_size]((g - vec_begin) * vec_size)
            comptime for j in range(vec_size):
                var e = v[j]
                if e > pivot_0:
                    thread_mass_0 += e
                    thread_count_0 += 1
                if e > pivot_1:
                    thread_mass_1 += e
                    thread_count_1 += 1
        var sums = _block_reduce_sums[block_size, 4, broadcast=False](
            StaticTuple[Float32, 4](
                thread_mass_0, thread_mass_1, thread_count_0, thread_count_1
            )
        )
        var packed = SIMD[.float32, 4](sums[0], sums[1], sums[2], sums[3])

        var combined = cluster_allreduce[
            _sum, cluster_size, need_tail_sync=False
        ](cluster_slot + phase * slot_floats, packed)
        phase ^= 1
        var mass_0 = combined[0]
        var mass_1 = combined[1]
        var count_0 = Int32(combined[2])
        var count_1 = Int32(combined[3])

        if count_0 < Int32(k) and mass_0 <= p_eff:
            # Accepted: count below k AND mass within the budget. `<=` so
            # that p=0 correctly accepts the argmax.
            accepted_e = pivot_0
            break
        if count_1 < Int32(k) and mass_1 <= p_eff:
            # pivot_0 rejected, pivot_1 accepted.
            low = pivot_0
            high = pivot_1
            q = mass_0
        else:
            # Both pivots rejected.
            low = pivot_1
            q = mass_1

    return Tuple[Int, Float32, Float32, Float32](sampled_id, low, accepted_e, q)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(
    t"topk_topp_sampling_emit_dist_cluster_{dtype}_{out_idx_type}_{deterministic}_{dist_dtype}_{cluster_size}",
)
def TopKTopPSamplingEmitDistClusterKernel[
    ProbsLayoutType: TensorLayout,
    probs_origin: ImmOrigin,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    deterministic: Bool,
    cluster_size: Int,
    dist_dtype: DType = .float32,
    ProbsEngine: TensorEngine = DefaultEngine[element_width=1],
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    probs: TileTensor[dtype, ProbsLayoutType, probs_origin, Engine=ProbsEngine],
    output: TileTensor[
        out_idx_type, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    out_dist: UnsafePointer[Scalar[dist_dtype], MutAnyOrigin],
    indices: Optional[UnsafePointer[Scalar[out_idx_type], ImmutAnyOrigin]],
    top_k_arr: Optional[UnsafePointer[Scalar[out_idx_type], ImmutAnyOrigin]],
    top_k_val: Int32,
    top_p_arr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    top_p_val: Float32,
    d: Int32,
    rng_seed: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]],
    rng_offset: UInt64,
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    min_p: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
):
    """`TopKTopPSamplingFromProbKernel` with `from_logits` and `emit_dist`,
    spread over a cluster.

    Every phase shares the row across the cluster. Each CTA stages its
    min-p-masked slice of the softmax weights in shared memory once, and the
    rejection loop, the cutoff search and the mask all read that copy: the
    loop's per-trial CDF prefix decomposes over the contiguous slices (see
    `_sampling_rejection_loop_cluster`), and its pivot masses combine across
    the cluster exactly like the search's statistics.

    Everything downstream of the loop consumes the loop's own combined
    values -- `row_max`, `p_eff`, the bracket and its mass budget -- on every
    CTA. Rebuilding any of them independently risks a one-ulp disagreement
    that puts the sampled token outside the emitted nucleus, which is the
    hazard the fused single-block kernel exists to avoid.

    The launch must set `cluster_dim` to `cluster_size`.

    Parameters:
        ProbsLayoutType: Memory layout of the input `probs` tile.
        probs_origin: Origin tag for the immutable input `probs` tile.
        OutputLayoutType: Memory layout of the output `output` tile.
        output_origin: Origin tag for the mutable output `output` tile.
        block_size: Number of threads per block.
        vec_size: Number of elements each thread loads per vectorized access.
        dtype: Element type of the `probs` tensor.
        out_idx_type: Index type used for the sampled output indices.
        deterministic: If True, use deterministic sampling.
        cluster_size: Number of CTAs sharing each row.
        dist_dtype: Element type of `out_dist`.
        ProbsEngine: Engine of the input `probs` tile.
        OutputEngine: Engine of the output `output` tile.

    Args:
        probs: Input logits [batch_size, d].
        output: Output sampled indices [batch_size].
        out_dist: Output masked renormalized distribution [batch_size, d].
        indices: Optional row indices for batch indexing [batch_size].
        top_k_arr: Optional per-row top_k values [batch_size].
        top_k_val: Default top_k value if top_k_arr is null.
        top_p_arr: Optional per-row top_p values [batch_size].
        top_p_val: Default top_p value if top_p_arr is null.
        d: Vocabulary size.
        rng_seed: Optional per-row seed array [batch_size], indexed by
            row_idx. If null, defaults to 0.
        rng_offset: Random offset for Random number generator.
        temperature: Optional per-row temperature [batch_size]; 0 is clamped.
        min_p: Optional per-row min-p thresholds [batch_size].
    """
    comptime assert (
        not is_apple_gpu()
    ), "TopKTopPSamplingEmitDistClusterKernel is not supported on Apple GPUs"
    comptime assert output.flat_rank == 1
    var _d = Int(d)
    var tx = Int(thread_idx.x)

    debug_assert(
        Int(cluster_dim.x) == cluster_size,
        "launch cluster_dim must match the kernel's cluster_size",
    )

    # A cluster covers one row; each CTA owns a contiguous range of vectors
    # so it can address its staged slice as `g - vec_begin`.
    var rank = Int(block_rank_in_cluster())
    var bx = Int(block_idx.x) // cluster_size
    var n_vec = _d // vec_size
    var slice_vec = ceildiv(n_vec, cluster_size)
    var vec_begin = min(rank * slice_vec, n_vec)
    var vec_end = min(vec_begin + slice_vec, n_vec)

    # Each CTA stages its slice of the softmax weights in dynamic shared
    # memory, so the sampling trials and the search re-read them from the SM
    # rather than from L2. The host guarantees the slice fits before choosing
    # this kernel.
    var smem = external_memory[Float32, address_space=.SHARED, alignment=16]()

    # Publish slots for this kernel's own cross-CTA combines; the loop and
    # the search bring their own pairs.
    var cluster_slot = unsafe_stack_allocation[
        2 * _CLUSTER_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=.SHARED,
    ]()

    wait_on_dependent_grids()
    launch_dependent_grids()

    var row_idx = bx
    if indices:
        row_idx = Int(indices.unsafe_value().load(bx))

    var seed_val = UInt64(0)
    if rng_seed:
        seed_val = rng_seed.unsafe_value()[row_idx]

    # Offset is keyed on row_idx (the request's logical row), not bx (the
    # physical batch slot), so a request samples identically regardless of
    # where it lands in the batch. The per-row seed already decorrelates rows.
    var generator = Random(seed=seed_val, offset=UInt64(row_idx) + rng_offset)

    var k = Int(top_k_val)
    if top_k_arr:
        k = Int(top_k_arr.unsafe_value().load(row_idx))
    if k == -1:
        k = Int(top_k_val)
    if k <= 0 or k > _d:
        k = _d

    var p = top_p_val
    if top_p_arr:
        p = top_p_arr.unsafe_value()[row_idx]

    var temp_val = Float32(1.0)
    if temperature:
        temp_val = temperature.unsafe_value()[row_idx]
    # Clamp so a greedy (T=0) row cannot divide by zero.
    var inv_temp = 1.0 / max(temp_val, Float32(1e-6))

    var min_p_thresh = Float32(0.0)
    if min_p:
        min_p_thresh = min_p.unsafe_value()[row_idx]

    var probs_row = TileTensor(probs.ptr + row_idx * _d, row_major(Idx[1], _d))

    # Row max, combined across the cluster in rank order so every CTA holds
    # the same bits. Every value below derives from it, so one disagreeing
    # CTA would branch differently through the loop and the search.
    var thread_max = Float32.MIN
    for i in range(vec_begin + tx, vec_end, block_size):
        var v = probs_row.load[width=vec_size]((Idx[0], i * vec_size)).cast[
            DType.float32
        ]()
        thread_max = max(thread_max, v.reduce_max())

    var row_max = cluster_allreduce[_max, cluster_size, need_tail_sync=False](
        cluster_slot,
        Float32(block.max[block_size=block_size, broadcast=False](thread_max)),
    )[0]

    # One pass stages each CTA's min-p-masked weights in shared memory --
    # the same working distribution the single-block kernel's `load_dist`
    # serves -- and sums two masses: the unmasked `z` (the sampling budget,
    # matching the separate-softmax path where probabilities are normalized
    # before min-p masking) and the masked total that the emitted
    # distribution normalizes by.
    var thread_sum = Float32(0)
    var thread_masked_sum = Float32(0)
    for i in range(vec_begin + tx, vec_end, block_size):
        var v = probs_row.load[width=vec_size]((Idx[0], i * vec_size)).cast[
            DType.float32
        ]()
        var e = exp((v - row_max) * inv_temp)
        thread_sum += e.reduce_add()
        if min_p_thresh > 0:
            comptime for j in range(vec_size):
                if e[j] < min_p_thresh:
                    e[j] = 0
            thread_masked_sum += e.reduce_add()
        smem.store[width=vec_size]((i - vec_begin) * vec_size, e)

    var z = cluster_allreduce[_sum, cluster_size, need_tail_sync=False](
        cluster_slot + _CLUSTER_SLOT_FLOATS,
        Float32(block.sum[block_size=block_size, broadcast=False](thread_sum)),
    )[0]
    # `min_p_thresh` is cluster-uniform (one row per cluster), so every CTA
    # takes the same branch and the collective stays legal. Without a mask
    # the masked total is `z`.
    var masked_z = z
    if min_p_thresh > 0:
        masked_z = cluster_allreduce[_sum, cluster_size, need_tail_sync=False](
            cluster_slot,
            Float32(
                block.sum[block_size=block_size, broadcast=False](
                    thread_masked_sum
                )
            ),
        )[0]

    # Top-p budget in the unnormalized domain. Identical on every CTA because
    # z came out of the rank-ordered cluster fold.
    var p_eff = _topp_budget(p, z)

    # The loop reads staged elements that other threads of this block wrote;
    # it never touches a peer CTA's slice, so a block barrier is enough.
    barrier()
    var drawn = _sampling_rejection_loop_cluster[
        vec_size, block_size, dtype, deterministic, cluster_size
    ](_d, k, p_eff, z, generator, smem, rank, vec_begin, vec_end)
    var sampled_id = drawn[0]
    var low = drawn[1]
    var accepted_e = drawn[2]
    var q = drawn[3]

    # A row that never accepted (non-finite logits, or the bracket collapsing)
    # has no constraint set to report, so its distribution stays zero --
    # callers read that as "no distribution for this row".
    var cutoff = Float32.MAX_FINITE
    var kept_mass = Float32(1)
    if accepted_e >= 0:
        # The search reads elements other lanes staged, so order the staging
        # writes against it -- cluster-wide, since the search's own combines
        # assume every CTA has arrived.
        cluster_sync()

        # The search needs `mass(> low)` over the staged (masked) weights. A
        # refined `q` came from staged sums and is exactly that; the initial
        # budget is the unmasked `z`, which overstates it whenever min-p
        # zeroed weight.
        var mass_above_low = q if low > 0 else masked_z
        var refined = _cluster_cutoff_search[
            vec_size, block_size, cluster_size
        ](
            _d,
            Int32(k),
            p_eff,
            low,
            accepted_e,
            mass_above_low,
            smem,
            vec_begin,
            vec_end,
        )
        cutoff = refined[0]
        kept_mass = refined[1]

    # Masks the staged weights; each lane owns the elements it wrote (the
    # same partition as the staging pass), so this needs no barrier against
    # the search.
    var dist_row = TileTensor(out_dist + bx * _d, row_major(Idx[1], _d))
    for i in range(vec_begin + tx, vec_end, block_size):
        var e = smem.load[width=vec_size]((i - vec_begin) * vec_size)
        var masked = (e.gt(cutoff)).select(
            e / kept_mass, SIMD[.float32, vec_size](0)
        )
        dist_row.store[width=vec_size](
            (Idx[0], i * vec_size), masked.cast[dist_dtype]()
        )

    if rank == 0 and tx == 0:
        output[bx] = Scalar[out_idx_type](sampled_id)

    # Terminal keep-alive. Peers read this CTA's shared memory through `mapa`,
    # so it may not retire until every CTA of the cluster is done reading.
    cluster_sync()


def topk_topp_sampling_from_prob_cluster[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    from_logits: Bool = False,
    emit_dist: Bool = False,
    dist_dtype: DType = .float32,
    DistLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64, Int64].element_types,
        stride_types=Coord[Int64, ComptimeInt[1]].element_types,
    ],
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    IndicesLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopPArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    SeedLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    MinPLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopKArrEngine: TensorEngine = DefaultEngine[element_width=1],
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPArrEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    MinPEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    probs: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, out_idx_type, ...],
    top_k_val: Int,
    top_p_val: Float32 = 1.0,
    deterministic: Bool = False,
    rng_seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
    rng_offset: UInt64 = 0,
    indices: Optional[
        TileTensor[
            out_idx_type,
            IndicesLayoutType,
            ImmutAnyOrigin,
            Engine=IndicesEngine,
        ]
    ] = None,
    top_k_arr: Optional[
        TileTensor[
            out_idx_type,
            TopKArrLayoutType,
            ImmutAnyOrigin,
            Engine=TopKArrEngine,
        ]
    ] = None,
    top_p_arr: Optional[
        TileTensor[
            .float32,
            TopPArrLayoutType,
            ImmutAnyOrigin,
            Engine=TopPArrEngine,
        ]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    min_p: Optional[
        TileTensor[.float32, MinPLayoutType, ImmutAnyOrigin, Engine=MinPEngine]
    ] = None,
    out_dist: Optional[
        TileTensor[dist_dtype, DistLayoutType, MutAnyOrigin]
    ] = None,
) raises:
    """Joint top-k + top-p sampling from probability distribution.

    Performs stochastic sampling considering only tokens that satisfy both the
    top-k count constraint AND the top-p nucleus constraint. When top_p_val is
    1.0 (default) this behaves identically to topk_sampling_from_prob.

    When `emit_dist` is set, the masked renormalized distribution is written
    to `out_dist` as well; see the kernel docstring. A batch too small to
    fill the GPU takes `TopKTopPSamplingEmitDistClusterKernel`, which spreads
    each row's sampling trials, cutoff search and mask over a cluster. The
    package's `topk_topp_sampling_from_prob` dispatcher routes here on NVIDIA
    SM90+ devices when the distribution is requested.

    When `from_logits` is True, `probs` contains raw logits: softmax with
    per-row temperature scaling and the optional min-p mask are fused into
    the sampling kernel, avoiding the [batch_size, d] probability round-trip
    through global memory and the separate softmax / mask kernel launches.

    Parameters:
        dtype: Element type of the `probs` tensor.
        out_idx_type: Index type used for the sampled output indices.
        block_size: Number of threads per block (defaults to 1024).
        from_logits: If True, `probs` holds raw logits and softmax with
            per-row temperature scaling and min-p masking is fused into
            the kernel (defaults to False).
        emit_dist: If True, also write the masked renormalized distribution
            to `out_dist` (defaults to False).
        dist_dtype: Element type of `out_dist`.
        DistLayoutType: Memory layout of the optional `out_dist` tensor.
        TopKArrLayoutType: Memory layout of the optional `top_k_arr` tensor.
        IndicesLayoutType: Memory layout of the optional `indices` tensor.
        TopPArrLayoutType: Memory layout of the optional `top_p_arr` tensor.
        SeedLayoutType: Memory layout of the optional `rng_seed` tensor.
        TemperatureLayoutType: Memory layout of the optional `temperature`
            tensor.
        MinPLayoutType: Memory layout of the optional `min_p` tensor.
        TopKArrEngine: Engine of the optional `top_k_arr` tensor.
        IndicesEngine: Engine of the optional `indices` tensor.
        TopPArrEngine: Engine of the optional `top_p_arr` tensor.
        SeedEngine: Engine of the optional `rng_seed` tensor.
        TemperatureEngine: Engine of the optional `temperature`
            tensor.
        MinPEngine: Engine of the optional `min_p` tensor.

    Args:
        ctx: Device context for kernel execution.
        probs: Input probability distribution [batch_size, d], or raw logits
            when `from_logits` is True.
        output: Output sampled indices [batch_size].
        top_k_val: Default top-k value (number of top tokens to consider).
        top_p_val: Default top-p value (nucleus probability threshold).
        deterministic: Whether to use deterministic sampling.
        rng_seed: Optional per-row seed tensor [batch_size], indexed by the
            request's logical row (see `indices`). If None, defaults to 0.
        rng_offset: Random offset for Random number generator.
        indices: Optional row indices for batch indexing [batch_size].
        top_k_arr: Optional per-row top-k values [batch_size].
        top_p_arr: Optional per-row top-p values [batch_size].
        temperature: Optional per-row temperature values [batch_size]. Only
            used when `from_logits` is True; defaults to 1.0 per row.
        min_p: Optional per-row min-p thresholds [batch_size]. Only used
            when `from_logits` is True.
        out_dist: Output masked distribution [batch_size, d]. Required when
            `emit_dist` is set.

    Raises:
        Error: If tensor ranks or shapes are invalid.
    """

    comptime assert probs.rank == 2, "probs rank must be 2"
    comptime assert output.rank == 1, "output rank must be 1"

    var shape = coord_to_index_list(probs.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("probs", shape, dtype),
                    "top_k_val=" + String(top_k_val),
                    "top_p_val=" + String(top_p_val),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_topp_sampling_from_prob",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var out_shape = coord_to_index_list(output.layout.shape_coord())
        if out_shape[0] != batch_size:
            raise Error("output batch size must match probs batch size")

        # Use up to 8 elements per vector to minimize the number of chunks
        # (and therefore the number of block-level reductions in inner loops).
        # GPU vector loads handle wider-than-native SIMD efficiently, and the
        # per-element idx < d guard handles non-aligned tails correctly.
        var vec_size = gcd(8, d)

        var indices_ptr: Optional[
            UnsafePointer[Scalar[out_idx_type], ImmutAnyOrigin]
        ] = None
        if indices:
            indices_ptr = indices.unsafe_value().ptr

        var top_k_ptr: Optional[
            UnsafePointer[Scalar[out_idx_type], ImmutAnyOrigin]
        ] = None
        if top_k_arr:
            top_k_ptr = top_k_arr.unsafe_value().ptr

        var top_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
        if top_p_arr:
            top_p_ptr = top_p_arr.unsafe_value().ptr

        var seed_ptr: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]] = None
        if rng_seed:
            seed_ptr = rng_seed.unsafe_value().ptr

        var temperature_ptr: Optional[
            UnsafePointer[Float32, ImmutAnyOrigin]
        ] = None
        if temperature:
            temperature_ptr = temperature.unsafe_value().ptr

        var min_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
        if min_p:
            min_p_ptr = min_p.unsafe_value().ptr

        var dist_ptr: Optional[
            UnsafePointer[Scalar[dist_dtype], MutAnyOrigin]
        ] = None

        comptime if emit_dist:
            if not out_dist:
                raise Error("out_dist is required when emit_dist is set")
            var dist_shape = coord_to_index_list(
                out_dist.unsafe_value().layout.shape_coord()
            )
            if dist_shape[0] != batch_size or dist_shape[1] != d:
                raise Error("out_dist shape must match probs shape")
            dist_ptr = out_dist.unsafe_value().ptr

        # Only the `emit_dist` variant spreads over a cluster: sampling alone
        # is one pass and does not repay the cluster's combines. A row takes
        # the narrowest cluster whose slice fits the staging budget, exactly
        # as in the masked-probs launcher; a row too wide for four slices
        # keeps the single-block kernel with no staging.
        var cluster_size = 0
        comptime if emit_dist and from_logits:
            comptime for c in [1, 2, _MAX_CUTOFF_CLUSTER]:
                if (
                    cluster_size == 0
                    and _stage_smem_bytes(d, vec_size, c)
                    <= _MAX_STAGE_SMEM_BYTES
                ):
                    cluster_size = c

        @__parameter
        def launch_cluster[
            vec_size: Int, deterministic: Bool, cluster_size: Int
        ]() raises:
            comptime if emit_dist and from_logits:
                comptime kernel = TopKTopPSamplingEmitDistClusterKernel[
                    probs.LayoutType,
                    ImmOrigin(probs.origin),
                    output.LayoutType,
                    output.origin,
                    block_size,
                    vec_size,
                    dtype,
                    out_idx_type,
                    deterministic,
                    cluster_size,
                    dist_dtype,
                    ProbsEngine=probs.Engine,
                    OutputEngine=output.Engine,
                ]
                var smem_bytes = _stage_smem_bytes(d, vec_size, cluster_size)
                ctx.enqueue_function[kernel](
                    probs.as_immut(),
                    output,
                    dist_ptr.unsafe_value(),
                    indices_ptr,
                    top_k_ptr,
                    Int32(top_k_val),
                    top_p_ptr,
                    top_p_val,
                    Int32(d),
                    seed_ptr,
                    rng_offset,
                    temperature_ptr,
                    min_p_ptr,
                    grid_dim=batch_size * cluster_size,
                    block_dim=block_size,
                    cluster_dim=Dim(cluster_size, 1, 1),
                    shared_mem_bytes=smem_bytes,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(smem_bytes)
                    ),
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )

        @__parameter
        def launch_single[vec_size: Int, deterministic: Bool]() raises:
            comptime kernel = TopKTopPSamplingFromProbKernel[
                probs.LayoutType,
                ImmOrigin(probs.origin),
                output.LayoutType,
                output.origin,
                block_size,
                vec_size,
                dtype,
                out_idx_type,
                deterministic,
                from_logits,
                emit_dist,
                dist_dtype,
                ProbsEngine=probs.Engine,
                OutputEngine=output.Engine,
            ]
            ctx.enqueue_function[kernel](
                probs.as_immut(),
                output,
                dist_ptr,
                indices_ptr,
                top_k_ptr,
                Int32(top_k_val),
                top_p_ptr,
                top_p_val,
                Int32(d),
                seed_ptr,
                rng_offset,
                temperature_ptr,
                min_p_ptr,
                Optional[UnsafePointer[Int32, MutAnyOrigin]](None),
                grid_dim=batch_size,
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        # `vec_size = gcd(8, d)`, so 8 is the widest case.
        @__parameter
        def dispatch_vec_size[deterministic: Bool]() raises:
            comptime for param_vec_size in [8, 4, 2, 1]:
                if vec_size == param_vec_size:
                    comptime for param_cluster in [1, 2, _MAX_CUTOFF_CLUSTER]:
                        if cluster_size == param_cluster:
                            return launch_cluster[
                                param_vec_size, deterministic, param_cluster
                            ]()
                    return launch_single[param_vec_size, deterministic]()

        if deterministic:
            dispatch_vec_size[True]()
        else:
            dispatch_vec_size[False]()
