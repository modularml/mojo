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

from std.algorithm.functional import unswitch
from std.collections import OptionalReg
from std.math import ceildiv, clamp, gcd
from std.sys import get_defined_int, size_of
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout import (
    ComptimeInt,
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RowMajorLayout,
    TileTensor,
    row_major,
)
from layout.tma_async import (
    TMATensorTile,
    _gather4_box_width,
)
from max.gpu.host.nvidia.tma import TensorMapL2Promotion, TensorMapSwizzle
from std.logger import Logger
from std.memory import bitcast

from nn.attention.gpu.nvidia.common import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand, KVCacheMHAOperand
from nn.attention.mha_utils import (
    MHAConfig,
)
from nn.attention.gpu.nvidia.common import KVTMATile
from std.utils.numerics import get_accum_type, nan
from std.utils.index import Index, IndexList

comptime logger = Logger()

# Q-sequence fold bounds for speculative decoding dispatch.
# When q_max_seq_len is in [MIN_FOLD_Q, MAX_FOLD_Q], the kernel folds q-tokens
# into the head dimension. MAX_FOLD_Q is the largest supported q_seq for fold
# (1 actual + 7 spec tokens). The fold loop iterates over
# range(MIN_FOLD_Q, MAX_FOLD_Q + 1).
comptime MIN_FOLD_Q = 2
comptime MAX_FOLD_Q = 8


# TODO: Remove once stdlib's SwitchedFunction2 supports `raises`.
# The stdlib 2-predicate unswitch uses SwitchedFunction2 which is
# `def[sw0: Bool, sw1: Bool]() -> None` (no raises). Sparse dispatch needs
# raises, so this local helper takes a raising unified closure value.
@always_inline
def _unswitch_raises[
    FuncType: def[sw0: Bool, sw1: Bool]() raises -> None
](
    dynamic_switch_a: Bool,
    dynamic_switch_b: Bool,
    switched_func: FuncType,
) raises:
    if dynamic_switch_a:

        @always_inline
        def switched_a_true[static_switch: Bool]() raises {imm}:
            switched_func[True, static_switch]()

        unswitch(dynamic_switch_b, switched_a_true)
    else:

        @always_inline
        def switched_a_false[static_switch: Bool]() raises {imm}:
            switched_func[False, static_switch]()

        unswitch(dynamic_switch_b, switched_a_false)


@always_inline
def _get_partition_bucket[half_sms: Int, i: Int]() -> Int:
    """Return the i-th partition bucket value.

    The bucket list uses half_sms (sm_count // 2) as its last entry so it
    adapts to any GPU: B200 (148 SMs) -> 74, B300 (160 SMs) -> 80, etc.
    """
    # Fixed bucket values for num_partitions to reduce CUDA graph captures.
    # Instead of many distinct values (each requiring a separate capture),
    # we map to a small set of fixed buckets.  The last bucket is half_sms
    # so the cap adapts to the target GPU.
    comptime _NUM_PARTITIONS = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 8,
        5: 9,
        6: 16,
        7: 18,
        8: 20,
        9: 32,
        10: 37,
        11: 64,
        12: 72,
        13: half_sms,
    }
    comptime _default = _NUM_PARTITIONS.get(len(_NUM_PARTITIONS) - 1, 0)
    comptime res = _NUM_PARTITIONS.get(i, _default)
    return res


def _bucket_num_partitions[half_sms: Int](num_partitions: Int) -> Int:
    """Map num_partitions to the smallest bucket value >= num_partitions.

    The bucket list uses half_sms (sm_count // 2) as its last entry so it
    adapts to any GPU: B200 (148 SMs) -> 74, B300 (160 SMs) -> 80, etc.
    """
    comptime _NUM_PARTITIONS = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 8,
        5: 9,
        6: 16,
        7: 18,
        8: 20,
        9: 32,
        10: 37,
        11: 64,
        12: 72,
        13: half_sms,
    }
    comptime _default = _NUM_PARTITIONS.get(len(_NUM_PARTITIONS) - 1, 0)
    comptime for kv in _NUM_PARTITIONS.items():
        comptime v = kv.value
        if num_partitions <= v:
            return v
    return _default


# Number of bucket entries in the partition table (fixed at 14).
comptime _NUM_PARTITION_BUCKETS = 14


def _unbucketed_split_error(num_partitions: Int) -> Error:
    """Builds the error for a split count with no compiled combine kernel.

    Falling through instead would skip the split-K reduction and leave the
    output at whatever it held, which is indistinguishable from a correct
    result and faster than one.

    Args:
        num_partitions: The split count that matched no bucket.

    Returns:
        The error to raise.
    """
    return Error(
        "MLA decode split-K: num_partitions=",
        num_partitions,
        (
            " has no compiled combine kernel (not in the bucket table), so the"
            " partial outputs would never be reduced. Route the split count"
            " through `_bucket_num_partitions` or add the value to the table."
        ),
    )


from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    QOTMATile,
    ORaggedTMATile,
    ScalesTMATile,
    tma_tile_qo,
    tma_tile_o,
    tma_tile_scales,
    MLA_Decode_Pack,
    num_matrix_view_rows_decode,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_kv_bf16 import (
    MLA_SM100_Decode_KV_BF16,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_kv_fp8 import (
    MLA_SM100_Decode_KV_FP8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8 import (
    MLA_SM100_Decode_QKV_FP8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8_layout_g import (
    MLA_SM100_Decode_QKV_FP8_Layout_G,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8_per_token_scale_rope_aware import (
    MLA_SM100_Decode_QKV_FP8_PerTokenScale_RopeAware,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_combine import (
    mla_decode_combine_partial_outputs,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse import (
    MLA_SM100_Decode_Sparse,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse_kv_fp8 import (
    MLA_SM100_Decode_Sparse_KV_FP8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse_qkv_fp8 import (
    MLA_SM100_Decode_Sparse_QKV_FP8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse_kv_bf16 import (
    MLA_SM100_Decode_Sparse_KV_BF16,
)


# ------------------------------------------------------------------------------
# Compute num_partitions heuristic (shared by dispatch and pre-compute op)
# ------------------------------------------------------------------------------
#
# Two separate functions: _compute_num_partitions_64 for single head group
# (64 heads, Kimi K2.5) and _compute_num_partitions_128 for multiple head
# groups (128 heads, DeepSeek V3/R1). The routing function
# _compute_num_partitions dispatches via comptime if on the head group count.
# ------------------------------------------------------------------------------


# The part of a CTA's cost that does not scale with the tokens it gathers
# (prologue, pipeline fill, drain, epilogue), in gathered-token equivalents so
# that only its ratio to the per-token cost enters the choice below. Fitted;
# the selected split is insensitive to the exact value.
comptime _CTA_FIXED_COST_IN_TOKENS = 187

# The model charges a partial wave as a whole one, so a difference below this
# margin is noise. Splits it cannot tell apart go to the tie-break below
# rather than the cost.
comptime _COST_TIE_MARGIN_PERCENT = 2


def _cost_optimal_partition_bucket[
    half_sms: Int
](
    ctas_per_partition: Int,
    split_len: Int,
    sm_count: Int,
    max_partitions: Int,
) -> Int:
    """Returns the cheapest split count the combine kernel can be launched for.

    Wall time for a decode launch is
    `ceil(total_ctas / sm_count) * (fixed_cost + work_per_split)`: splitting
    divides the work per split but multiplies the CTA count, and every extra
    wave re-pays the fixed cost. Sizing the split to fill one wave models
    neither term — measured at batch 16, the largest one-wave split is the
    worst reachable choice — so minimise the product directly over the (small)
    candidate set.

    The search runs over the partition buckets rather than over every integer
    because `dispatch_combine` matches `num_partitions` against that set
    exactly, with no fallback: a count outside it launches the decode and never
    reduces its partial outputs. Rounding an unconstrained optimum up to the
    next bucket is not the optimum of the reachable set — at the spec-decode
    verify shape the unconstrained optimum is 3, which rounds to the value the
    shipped policy already picks — so the constraint belongs inside the search.

    Parameters:
        half_sms: sm_count // 2 — the largest bucket value (compile-time).

    Args:
        ctas_per_partition: CTAs launched per split, i.e. `grid.x * grid.y *
            batch_size`.
        split_len: KV tokens each (batch, position) attends before splitting.
        sm_count: Number of SMs on the target GPU.
        max_partitions: Largest admissible split count.

    Returns:
        The cost-minimising number of split-K partitions.
    """
    var best_np = 1
    var best_cost = -1
    # Ascending, so taking a candidate only when it is materially cheaper keeps
    # the smallest split among those the model cannot distinguish.
    comptime for b in range(_NUM_PARTITION_BUCKETS):
        var np = _get_partition_bucket[half_sms, b]()
        if np <= max_partitions:
            var waves = ceildiv(ctas_per_partition * np, sm_count)
            var cost = waves * (
                _CTA_FIXED_COST_IN_TOKENS + ceildiv(split_len, np)
            )
            if best_cost < 0 or cost * 100 < best_cost * (
                100 - _COST_TIE_MARGIN_PERCENT
            ):
                best_cost = cost
                best_np = np
    return best_np


def _compute_num_partitions_64[
    num_heads: Int,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
    # Read-once shared-index fold (KERN-3141): grid.y collapses to 1, so the
    # decode relies on split-K ALONE to fill the SMs. Relax the per-split page
    # floor for the folded shapes so the wave-aligned target_partitions
    # survives. Default False -> byte-identical to the production path.
    fold_shared_index: Bool = False,
](
    batch_size: Int,
    effective_max_cache_len: Int,
    q_max_seq_len: Int,
    split_page_size: Int,
    sm_count: Int,
    relax_split_floor: Bool = False,
    cost_optimal_split: Bool = False,
) -> Int:
    """Wave-aligned split count for single head group (e.g. Kimi K2.5, 64 heads).

    This is the EXACT logic from the original unified _compute_num_partitions
    when _head_groups == 1.  Do NOT modify without verifying byte-for-byte
    equivalence with the 64-head production path.

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        is_fp8_kv: Whether the KV cache is FP8 (compile-time).
        half_sms: sm_count // 2 — maximum split-K partitions (compile-time).
        fold_shared_index: Read-once shared-index fold (KERN-3141); relaxes the
            per-split page floor to 1 for folded shapes (compile-time).

    Args:
        batch_size: Current batch size.
        effective_max_cache_len: Max KV cache length.
        q_max_seq_len: Max query sequence length (1 for decode).
        split_page_size: Page granularity for split-K (64 or 128).
        sm_count: Number of SMs on the target GPU.
        relax_split_floor: When True, drop the per-split page floor to 1 so np
            tracks the effective KV page count (~one page per split). Set by the
            q_len=1 sparse FP8 split-K tuning (KERN-3217); the gating predicate
            lives at the dispatch call site. Default False retains the previous
            partition-selection logic (the production floor of 4).
        cost_optimal_split: When True, choose the split minimising modelled wall
            time (`_cost_optimal_partition_bucket`) instead of the wave target,
            page floor and bucket rounding below. Set only for the unfolded
            sparse spec-decode verify; the gating predicate lives at the
            dispatch call site, which alone sees whether the launch folds.

    Returns:
        The number of split-K partitions.
    """
    var num_kv_cache_pages = ceildiv(effective_max_cache_len, split_page_size)

    if cost_optimal_split:
        # The gate at the call site fires only for a launch that does not fold
        # q into the M tile, so grid.y is q_max_seq_len and every q position
        # costs its own CTA — which is exactly what the shape-keyed fold below
        # gets wrong for this launch.
        return _cost_optimal_partition_bucket[half_sms](
            ceildiv(num_heads, 64) * q_max_seq_len * batch_size,
            effective_max_cache_len,
            sm_count,
            min(half_sms, num_kv_cache_pages),
        )

    # When fold is active (spec decoding with num_heads * q_max_seq_len <= 64),
    # the kernel packs all q_tokens into the M tile of a single CTA, so
    # ctas_per_partition should NOT multiply by q_max_seq_len. q > 1 implies
    # spec decoding; the fold path requires the M tile to fit ≤ 64.
    var fold_active = (q_max_seq_len > 1) and (num_heads * q_max_seq_len <= 64)
    var q_len_factor = 1 if fold_active else q_max_seq_len
    var ctas_per_partition = ceildiv(num_heads, 64) * q_len_factor * batch_size

    # Single head group: 85% fill threshold (floor * ctas * 20 >= sm * 17).
    var floor_target = sm_count // ctas_per_partition
    var ceil_target = ceildiv(sm_count, ctas_per_partition)
    var target_partitions: Int
    if (
        floor_target >= 1
        and floor_target * ctas_per_partition * 20 >= sm_count * 17
    ):
        target_partitions = floor_target
    else:
        target_partitions = ceil_target

    target_partitions = min(target_partitions, half_sms)

    # Pages-per-CTA lower bound to avoid under-utilizing CTAs (per-CTA setup
    # + combine-grid overhead would dominate if each CTA processed too few
    # pages).
    comptime _min_pages_per_split = 4
    # Read-once shared-index fold (KERN-3141): grid.y=1 leaves split-K as the
    # only SM-fill lever, so drop the per-split page floor to 1 for the folded
    # shapes (matches FlashInfer's ~18-way split of the topk domain). Off ->
    # the production floor of 4, byte-identical.
    # Intentionally coarser than the launch-selection gate (_fold_ok also
    # excludes extra_kv/variable_topk/attn_sink): num_partitions is a tuning
    # knob, so a relaxed floor on a fallen-back unfolded launch is
    # correctness-neutral.
    # relax_split_floor additionally relaxes the floor for the unfolded q_len=1
    # sparse FP8 decode (KERN-3217), which is SM-underfilled at the floor of 4;
    # the caller has already clamped effective_max_cache_len to the kernel's
    # min(topk, cache+q) bound so np tracks the real page count (one per split).
    var _min_pages = 1 if (
        (fold_shared_index and fold_active) or relax_split_floor
    ) else _min_pages_per_split
    var max_np_for_min_pages = num_kv_cache_pages // _min_pages

    # Policy: honor wave-aligned target_partitions, but cap np DOWN if it
    # would leave too few pages per CTA (under-utilization). Long K-loops
    # per CTA are FINE — they amortize fixed costs better than splitting.
    var num_partitions: Int = min(target_partitions, max_np_for_min_pages)

    # Clamp: allow np=1 for very short cache + large batch, or when single
    # head group fills >= 80% of SMs.
    comptime _np1_cache_threshold = 512 if is_fp8_kv else 256
    var min_partitions: Int
    if effective_max_cache_len <= _np1_cache_threshold and batch_size >= 64:
        min_partitions = 1
    elif ctas_per_partition * 5 >= sm_count * 4:
        min_partitions = 1
    elif num_kv_cache_pages >= 2:
        min_partitions = 2
    else:
        min_partitions = 1
    num_partitions = clamp(
        num_partitions, min_partitions, min(half_sms, num_kv_cache_pages)
    )

    num_partitions = _bucket_num_partitions[half_sms](num_partitions)
    return num_partitions


def _compute_num_partitions_128[
    num_heads: Int,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
](
    batch_size: Int,
    effective_max_cache_len: Int,
    q_max_seq_len: Int,
    split_page_size: Int,
    sm_count: Int,
) -> Int:
    """Wave-aligned split count for multiple head groups (e.g. DeepSeek V3/R1,
    128 heads).

    Optimized to match FlashInfer's split counts for DeepSeek configurations:
    - max_pages_per_split = 32 (vs 18) to reduce min_partitions_for_work
    - Doubled min_pages_per_split to compensate for 2x CTA multiplier
    - np=1 allowed for bs >= 64 with short cache (eliminates combine kernel)
    - wave_quantum capped at target_partitions for bs=4-8

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        is_fp8_kv: Whether the KV cache is FP8 (compile-time).
        half_sms: sm_count // 2 — maximum split-K partitions (compile-time).

    Args:
        batch_size: Current batch size.
        effective_max_cache_len: Max KV cache length.
        q_max_seq_len: Max query sequence length (1 for decode).
        split_page_size: Page granularity for split-K (64 or 128).
        sm_count: Number of SMs on the target GPU.

    Returns:
        The number of split-K partitions.
    """
    var num_kv_cache_pages = ceildiv(effective_max_cache_len, split_page_size)

    var ctas_per_partition = ceildiv(num_heads, 64) * q_max_seq_len * batch_size
    var wave_quantum = min(
        sm_count // gcd(ctas_per_partition, sm_count), half_sms
    )

    # Multiple head groups: 90% fill threshold.
    var floor_target = sm_count // ctas_per_partition
    var ceil_target = ceildiv(sm_count, ctas_per_partition)
    var target_partitions: Int
    if (
        floor_target >= 1
        and floor_target * ctas_per_partition * 10 >= sm_count * 9
    ):
        target_partitions = floor_target
    else:
        target_partitions = ceil_target

    target_partitions = min(target_partitions, half_sms)

    if batch_size >= 16:
        wave_quantum = min(wave_quantum, max(target_partitions, 1))

    # Change 4: Cap wave_quantum for bs=4-8 to prevent over-splitting.
    # At these batch sizes, wave_quantum=37 can cause excessive splits
    # (e.g., bs=8/cl=65K: 37 splits vs FlashInfer's 18). Capping at
    # target_partitions keeps splits aligned with SM fill needs.
    if batch_size >= 4 and batch_size <= 8:
        wave_quantum = min(wave_quantum, target_partitions)

    # Change 1: max_pages_per_split = 32 (vs 18 in the 64-head path).
    # For DeepSeek bs=8/cl=65K: pages=512, min_partitions = ceil(512/32)=16
    # (was ceil(512/18)=29 with the old value).
    comptime _head_groups = ceildiv(num_heads, 64)
    comptime max_pages_per_split = 32
    var min_partitions_for_work = ceildiv(
        num_kv_cache_pages, max_pages_per_split
    )

    comptime _min_pps_large_batch = 16 if is_fp8_kv else 8
    comptime _min_pps_small_batch = 8 if is_fp8_kv else 4
    var min_pages_per_split: Int
    if batch_size >= 64:
        min_pages_per_split = _min_pps_large_batch
    elif batch_size <= 8 and is_fp8_kv:
        min_pages_per_split = 4
    else:
        min_pages_per_split = _min_pps_small_batch

    # Base thresholds are tuned for 2 head groups; apply 2/head_groups.
    min_pages_per_split = min_pages_per_split * 2 // _head_groups

    # Change 2: Double min_pages_per_split to compensate for 2x CTA
    # multiplier with 2 head groups. This halves the page_constrained
    # value, reducing np.
    min_pages_per_split = min_pages_per_split * 2

    var num_waves = max(1, ceildiv(min_partitions_for_work, wave_quantum))
    var wave_aligned = num_waves * wave_quantum

    # Multiple head groups: pick smaller of wave-aligned and
    # page-constrained, ensure at least target_partitions.
    var num_partitions = max(
        target_partitions,
        min(wave_aligned, num_kv_cache_pages // min_pages_per_split),
    )

    # Change 3: Allow np=1 at high batch with short cache.
    # For bs >= 64 and effective_max_cache_len <= 2176, the combine kernel
    # overhead (16K-32K CTAs) is catastrophic. Allowing np=1 eliminates it.
    comptime _np1_cache_threshold = 512 if is_fp8_kv else 256
    var min_partitions: Int
    if effective_max_cache_len <= _np1_cache_threshold and batch_size >= 64:
        min_partitions = 1
        # R2: Override target_partitions so it doesn't act as a floor that
        # prevents np=1. Recompute num_partitions with the lowered floor.
        target_partitions = 1
        num_partitions = max(
            target_partitions,
            min(wave_aligned, num_kv_cache_pages // min_pages_per_split),
        )
    elif batch_size >= 64 and effective_max_cache_len <= 2176:
        # DeepSeek-specific: eliminate combine kernel for high-batch
        # short-cache.
        min_partitions = 1
        # R2: Same fix — recompute with lowered floor.
        target_partitions = 1
        num_partitions = max(
            target_partitions,
            min(wave_aligned, num_kv_cache_pages // min_pages_per_split),
        )
    elif num_kv_cache_pages >= 2:
        min_partitions = 2
    else:
        min_partitions = 1

    # R1: When a single partition already fills the GPU
    # (ctas_per_partition >= sm_count), splitting just adds combine kernel
    # overhead without improving decode parallelism. Force np=1.
    #
    # For DeepSeek 128 heads: ctas_per_partition = 2 * batch_size.
    # ctas_per_partition >= sm_count when bs >= sm_count/2 (e.g., bs >= 74
    # on B200 with 148 SMs).
    #
    # The combine kernel grid at these batch sizes is catastrophic:
    #   bs=512/cl=4096 with np=2: 32768 combine CTAs (221 waves at wph=4)
    #   bs=256/cl=4096 with np=2: 16384 combine CTAs (110 waves at wph=4)
    # Forcing np=1 eliminates combine entirely. Each decode CTA processes
    # more pages but that cost is less than the combine overhead.
    if ctas_per_partition >= sm_count:
        num_partitions = 1
        min_partitions = 1

    num_partitions = clamp(
        num_partitions, min_partitions, min(half_sms, num_kv_cache_pages)
    )

    num_partitions = _bucket_num_partitions[half_sms](num_partitions)
    return num_partitions


def _compute_num_partitions[
    num_heads: Int,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
    # Read-once shared-index fold (KERN-3141); relaxes the 64-head split floor.
    fold_shared_index: Bool = False,
](
    batch_size: Int,
    effective_max_cache_len: Int,
    q_max_seq_len: Int,
    split_page_size: Int,
    sm_count: Int,
    relax_split_floor: Bool = False,
    cost_optimal_split: Bool = False,
) -> Int:
    """Routing function that dispatches to head-count-specific heuristics.

    Single head group (num_heads <= 64, e.g. Kimi K2.5) calls
    _compute_num_partitions_64.  Multiple head groups (num_heads > 64,
    e.g. DeepSeek V3/R1) calls _compute_num_partitions_128.

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        is_fp8_kv: Whether the KV cache is FP8 (compile-time).
        half_sms: sm_count // 2 — maximum split-K partitions (compile-time).
        fold_shared_index: Read-once shared-index fold (KERN-3141); relaxes the
            per-split page floor for folded shapes (compile-time).

    Args:
        batch_size: Current batch size.
        effective_max_cache_len: Max KV cache length.
        q_max_seq_len: Max query sequence length (1 for decode).
        split_page_size: Page granularity for split-K (64 or 128).
        sm_count: Number of SMs on the target GPU.
        relax_split_floor: When True, drop the per-split page floor to 1 in the
            single-head-group heuristic so np tracks the effective KV page count
            (KERN-3217 q_len=1 sparse FP8 split-K tuning; the gating predicate
            lives at the dispatch call site). Default False retains the previous
            partition-selection logic (the production floor); ignored by the
            multi-head-group (128-head) path.
        cost_optimal_split: Choose the split minimising modelled wall time; see
            `_compute_num_partitions_64`. Ignored by the multi-head-group
            (128-head) path.

    Returns:
        The number of split-K partitions.
    """
    comptime _head_groups = ceildiv(num_heads, 64)

    comptime if _head_groups == 1:
        return _compute_num_partitions_64[
            num_heads, is_fp8_kv, half_sms, fold_shared_index
        ](
            batch_size,
            effective_max_cache_len,
            q_max_seq_len,
            split_page_size,
            sm_count,
            relax_split_floor=relax_split_floor,
            cost_optimal_split=cost_optimal_split,
        )
    else:
        return _compute_num_partitions_128[num_heads, is_fp8_kv, half_sms](
            batch_size,
            effective_max_cache_len,
            q_max_seq_len,
            split_page_size,
            sm_count,
        )


# ------------------------------------------------------------------------------
# Public pre-compute function for MOGG ops
# ------------------------------------------------------------------------------
def compute_mla_dispatch_scalars[
    num_heads: Int,
    _is_cache_length_accurate: Bool = False,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
    # Read-once shared-index fold (KERN-3141); relaxes the split-K page floor.
    fold_shared_index: Bool = False,
](
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    sm_count: Int,
) -> Tuple[Int, Int, Int]:
    """Pure computation of the packed MLA dispatch metadata.

    Returns ``(batch_size, q_max_seq_len, num_partitions)``.
    These three values are baked into the size-3 GPU buffer.
    ``effective_split_len`` is computed directly inside the MoGG ops from
    ``max_cache_length`` (``max_cache_valid_length + q_max_seq_len`` when
    ``_is_cache_length_accurate=False``, else ``max_cache_valid_length``),
    and no longer needs to be returned here.
    """
    var effective = max_cache_valid_length

    comptime if not _is_cache_length_accurate:
        effective += q_max_seq_len

    var split_page_size = 64 if (effective <= 512 and batch_size >= 32) else 128
    var num_partitions = _compute_num_partitions[
        num_heads, is_fp8_kv, half_sms, fold_shared_index
    ](batch_size, effective, q_max_seq_len, split_page_size, sm_count)

    return (batch_size, q_max_seq_len, num_partitions)


def _dispatch_scalars_for_heads[
    num_heads: Int
](
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    is_fp8_kv: Bool,
    sm_count: Int,
) -> Tuple[Int, Int, Int]:
    """Binds `is_fp8_kv` at comptime for one already-bound head count."""
    if is_fp8_kv:
        return compute_mla_dispatch_scalars[num_heads, is_fp8_kv=True](
            batch_size, max_cache_valid_length, q_max_seq_len, sm_count
        )
    return compute_mla_dispatch_scalars[num_heads](
        batch_size, max_cache_valid_length, q_max_seq_len, sm_count
    )


def compute_mla_dispatch_scalars_runtime(
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    num_heads: Int,
    is_fp8_kv: Bool,
    sm_count: Int,
) raises -> Tuple[Int, Int, Int]:
    """Binds a runtime Q-head count to a comptime dispatch instantiation.

    Every count here is one the kernel is measured correct at; 12, 24 and 48
    are Kimi K3 at TP8/TP4/TP2. The list is not exhaustive — a count the kernel
    supports but that no model asks for yet is left out rather than paying its
    instantiation, so add an arm when a model needs one.
    """
    if num_heads == 8:
        return _dispatch_scalars_for_heads[8](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 12:
        return _dispatch_scalars_for_heads[12](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 16:
        return _dispatch_scalars_for_heads[16](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 24:
        return _dispatch_scalars_for_heads[24](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 32:
        return _dispatch_scalars_for_heads[32](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 48:
        return _dispatch_scalars_for_heads[48](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 64:
        return _dispatch_scalars_for_heads[64](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 128:
        return _dispatch_scalars_for_heads[128](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    raise Error(
        "Unsupported MLA num_heads for direct dispatch metadata binding: "
        + String(num_heads)
    )


struct MLADispatchScalarArgs[
    num_heads: Int,
    _is_cache_length_accurate: Bool = False,
    is_fp8_kv: Bool = False,
    # Read-once shared-index fold (KERN-3141); relaxes the split-K page floor so
    # the pre-computed num_partitions matches what the folded launch expects.
    fold_shared_index: Bool = False,
]:
    """Pre-computed MLA decode args for the legacy (non-capturable) path.

    Owns a GPU buffer containing ``[batch_size, q_max_seq_len, num_partitions]``
    and caches the host-side ``batch_size``/``q_max_seq_len`` pair needed by
    ``mla_decode_sm100_dispatch``.

    Usage::

        var args = MLADispatchScalarArgs[num_heads=128](
            batch_size, max_cache_len, q_max_seq_len, ctx,
        )
        var gpu_lt = args.gpu_layout_tensor()
        mla_decode_sm100_dispatch[...](
            ..., gpu_lt,
            args.batch_size, args.q_max_seq_len, max_cache_len,
            ctx,
        )
        _ = args  # keepalive
    """

    comptime MLAScalarArgsLT = LayoutTensor[
        .int64, Layout.row_major(3), MutAnyOrigin
    ]

    var gpu_buf: DeviceBuffer[.int64]
    var batch_size: Int
    var q_max_seq_len: Int

    def __init__(
        out self,
        batch_size: Int,
        max_cache_len: Int,
        q_max_seq_len: Int,
        ctx: DeviceContext,
    ) raises:
        self.gpu_buf = ctx.enqueue_create_buffer[.int64](3)
        self.batch_size = batch_size
        self.q_max_seq_len = q_max_seq_len

        comptime sm_count = ctx.default_device_info.sm_count
        comptime _half_sms = sm_count // 2
        var scalars = compute_mla_dispatch_scalars[
            num_heads=Self.num_heads,
            _is_cache_length_accurate=Self._is_cache_length_accurate,
            is_fp8_kv=Self.is_fp8_kv,
            half_sms=_half_sms,
            fold_shared_index=Self.fold_shared_index,
        ](batch_size, max_cache_len, q_max_seq_len, sm_count)

        # Note: scalars[3] (effective_split_len) is only consumed by the
        # capturable-graph dispatcher path, not by the legacy GPU buffer.
        var host_args = Array[Int64, 3](uninitialized=True)
        host_args[0] = Int64(scalars[0])
        host_args[1] = Int64(scalars[1])
        host_args[2] = Int64(scalars[2])
        var output_buf = DeviceBuffer[.int64](
            ctx, self.gpu_buf.unsafe_ptr(), 3, owning=False
        )
        output_buf.enqueue_copy_from(
            UnsafePointer(to=host_args).bitcast[Int64]()
        )

    def gpu_layout_tensor(
        self,
    ) -> Self.MLAScalarArgsLT:
        return Self.MLAScalarArgsLT(
            rebind[UnsafePointer[Int64, origin=MutAnyOrigin]](
                self.gpu_buf.unsafe_ptr()
            ),
        )

    def gpu_tile_tensor(
        self,
    ) -> TileTensor[.int64, RowMajorLayout[ComptimeInt[3]], MutAnyOrigin]:
        return TileTensor(
            rebind[UnsafePointer[Int64, MutAnyOrigin]](
                self.gpu_buf.unsafe_ptr()
            ),
            row_major((Idx[3],)),
        )


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------
def mla_decode_sm100_dispatch[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    *,
    ragged: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
    per_token_scale_rope_aware: Bool = False,
    sparse: Bool = False,
    # Sparse-only routing flag: when True, route to the BF16-rope sparse
    # kernel (split FP8 nope + BF16 rope, two TMAs). When False (default),
    # route to the all-FP8 sparse kernel (single 576-byte gather4 TMA).
    rope_aware_kv_sparse: Bool = False,
    # Read-once shared-index fold (KERN-3141): when the MTP-folded positions
    # share one identical topk list, gather it ONCE. Drives the sparse fp8
    # kernel's fold + the split-K floor relax. False -> unchanged baseline.
    fold_shared_index: Bool = False,
](
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    k: k_t,
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    scale: Float32,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    mask: mask_t,
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    batch_size: Int,
    q_max_seq_len: Int,
    max_cache_valid_length: Int,
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Extra KV parameters (forwarded to mla_decode_sm100_sink_split_k).
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Pre-computed grid-time scalar from the dispatcher input list (capturable
    # graph path). When provided, the value bypasses the local recompute so
    # the host-side grid sizing matches the device-side divmod on
    # scalar_args_buf[2].
    num_partitions_in: Optional[Int] = None,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    var scales_ptr = k.scales_raw_ptr()

    var effective_max_cache_len = max_cache_valid_length

    comptime if not _is_cache_length_accurate:
        effective_max_cache_len += q_max_seq_len

    # For sparse decode, the split-K factor must be driven by the actual
    # number of tokens each batch attends to (max_topk + max_extra_topk),
    # NOT the full cache length.  With topk << cache_len (e.g. topk=256
    # vs cache_len=163840), using the full cache length creates far too
    # many splits — each excess CTA just early-exits but still wastes
    # launch overhead and combine kernel grid size.
    #
    # indices_stride is the max topk across all batches: for fixed topk
    # it equals topk; for variable topk it equals max(topk_per_batch)
    # (the allocation stride).  Similarly extra_indices_stride is
    # max_extra_topk.
    var effective_split_len = effective_max_cache_len
    comptime if sparse:
        var max_topk = indices_stride
        var max_extra_topk = extra_indices_stride
        effective_split_len = max_topk + max_extra_topk

    # Sliding-window split-K shrink : cap the effective cache length at
    # window_size + q_max_seq_len so partition heuristics don't
    # over-split a region the kernel will skip anyway.
    comptime if mask_t.get_type_name() == "SlidingWindowCausalMask":
        comptime _sw_window_size: Int = mask_t.sliding_window_size()
        var sw_cap: Int = _sw_window_size + q_max_seq_len
        # at high batch (bs>=64) with a small SW cap
        # (<=2048), shrinking the effective_split_len to sw_cap forces the
        # partition heuristic into np=1.
        if batch_size >= 64 and sw_cap <= 2048:
            sw_cap = 2048
        effective_split_len = min(effective_split_len, sw_cap)

    var use_small_split_pages = effective_split_len <= 512 and batch_size >= 32
    var split_page_size = 64 if use_small_split_pages else 128
    comptime sm_count = ctx.default_device_info.sm_count
    comptime _half_sms = sm_count // 2
    comptime _is_fp8_kv = (k_t.dtype == DType.float8_e4m3fn)

    # q_len=1 sparse FP8 split-K tuning (KERN-3217): the single-token sparse
    # FP8 decode with no extra_kv / variable_topk / attn_sink is SM-underfilled
    # under the default per-split page floor of 4 (e.g. bs8 launches 4*8 = 32
    # CTAs on 148 SMs, ~0.2 waves). Scoped to the validated GLM-5.2 B200 regime:
    # q_len == 1, batch_size <= 8, configured top-k == 2048 (indices_stride is
    # the fixed top-k at this site; `not topk_lengths` confirms it is fixed, not
    # variable), and RAW cache length max_cache_valid_length in [1024, 2048]
    # (the raw dispatch arg, NOT effective_max_cache_len, so the guard bounds
    # the scope, not the clamp). In scope: relax the floor to 1 so np tracks the
    # effective KV page count (~one page per split), and clamp the split length
    # to the kernel's own bound min(topk, cache+q) so np never over-splits when
    # topk exceeds the live cache (the kernel clamps topk to actual_tokens =
    # cache_length + seq_len; see OffsetPosition in mla_decode_utils.mojo). This
    # is the unfolded-q1 analog of the fold_shared_index floor relax above.
    # Out of scope -- batch_size > 8, q_len > 1, top-k != 2048, cache outside
    # [1024, 2048], dense, BF16, and every unsupported sparse feature -- retains
    # the previous partition-selection logic: the guard is false, so BOTH the
    # floor (relax_split_floor=False) and the split-len clamp (_np_split_len =
    # effective_split_len) use the prior computation.
    var _relax_q1_sparse_floor = (
        sparse
        and _is_fp8_kv
        and q_max_seq_len == 1
        and batch_size <= 8
        and indices_stride == 2048
        and max_cache_valid_length >= 1024
        and max_cache_valid_length <= 2048
        and extra_indices_stride == 0
        and not topk_lengths
        and not attn_sink_ptr
    )
    var _np_split_len = min(
        effective_split_len, effective_max_cache_len
    ) if _relax_q1_sparse_floor else effective_split_len
    # Unfolded sparse spec-decode verify, where the shape-keyed fold below
    # reports a fold that only `fold_shared_index` actually performs, so the
    # split is sized for a grid `q_max_seq_len` times smaller than the one
    # launched. Sound here because sparse np comes from the top-k stride
    # rather than the cache length, so it stays a pure function of the shape
    # the caller already specialises on; the rest is scope.
    var _cost_optimal_split = (
        sparse
        and _is_fp8_kv
        and not fold_shared_index
        and not rope_aware_kv_sparse
        and q_max_seq_len > 1
        and q_max_seq_len <= MAX_FOLD_Q
        and num_heads * q_max_seq_len <= 64
        and extra_indices_stride == 0
    )
    var num_partitions = _compute_num_partitions[
        num_heads, _is_fp8_kv, _half_sms, fold_shared_index
    ](
        batch_size,
        _np_split_len,
        q_max_seq_len,
        split_page_size,
        sm_count,
        relax_split_floor=_relax_q1_sparse_floor,
        cost_optimal_split=_cost_optimal_split,
    )

    if num_partitions_in:
        num_partitions = num_partitions_in.value()

    # =========================================================================
    # split_page_size routing: use finer split granularity for short cache
    # with moderate-to-large batch to improve split balance.
    #
    # When cache is short (effective_max_cache_len <= 512, i.e. <=4 pages at
    # page_size=128) and batch is large enough (>=32), splitting with
    # page_size=64 gives twice the page count, enabling better work
    # distribution across splits.
    # For example, bs=64/cl=256 gets 5 pages at page_size=64 (vs 3 at 128),
    # allowing np=2 with 2-3 pages per split instead of 1-2.
    # =========================================================================
    @__parameter
    @always_inline
    def launch_impl[split_page_size_param: Int]() raises:
        _mla_decode_sm100_dispatch_impl[
            q_type=q_type,
            k_t=k_t,
            output_type=output_type,
            mask_t=mask_t,
            config=config,
            depth=depth,
            num_heads=num_heads,
            group=group,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=decoding_warp_split_k,
            split_page_size=split_page_size_param,
            per_token_scale_rope_aware=per_token_scale_rope_aware,
            sparse=sparse,
            rope_aware_kv_sparse=rope_aware_kv_sparse,
            fold_shared_index=fold_shared_index,
        ](
            q,
            k,
            output,
            scale,
            valid_length,
            mask,
            scales_ptr,
            scalar_args_buf,
            batch_size,
            q_max_seq_len,
            num_partitions,
            effective_max_cache_len,
            ctx,
            q_scale_ptr,
            d_indices,
            indices_stride,
            topk_lengths,
            attn_sink_ptr,
            extra_k=extra_k,
            extra_d_indices=extra_d_indices,
            extra_indices_stride=extra_indices_stride,
            extra_topk_lengths=extra_topk_lengths,
            extra_scales_ptr=extra_scales_ptr,
            logical_indices=logical_indices,
        )

    comptime if k_t.page_size == 0 or k_t.page_size >= 128:
        # Non-paged (page_size=0) and page_size>=128 both satisfy the
        # `page_size % split_page_size == 0` assertion for both 64 and
        # 128 split granularities, so the full routing is emittable.
        if use_small_split_pages:
            launch_impl[64]()
        else:
            launch_impl[128]()
    elif k_t.page_size == 64:
        # page_size=64: only split_page_size=64 satisfies the divisibility
        # assertion. The 128 branch would fail instantiation.
        launch_impl[64]()
    else:
        # page_size < 64 (e.g. 16, 32): split_page_size = page_size so each
        # split aligns with a whole number of physical pages. Dispatch the
        # split_page_size at comptime so only the valid instantiation is
        # emitted.
        launch_impl[k_t.page_size]()


# ------------------------------------------------------------------------------
# Inner dispatch implementation parameterized on split_page_size
# ------------------------------------------------------------------------------
# Mirrors `FA4_WS_POISON` in `sm100/dispatch.mojo`: the split-K workspaces
# below are deliberately allocated without a fill, because an empty partition
# skips its O store while still writing `-inf` to its LSE slot, and
# `mla_decode_combine`'s scale check is what keeps the unwritten O slots from
# contributing. `mla_decode_combine.mojo`'s prefetch load therefore reads
# uninitialized memory by design and initcheck reports it. NaN-filling both
# workspaces initializes them (silencing the false report) and grades that
# contract: a dropped scale check or a missing `-inf` write propagates NaN into
# the output and trips the test's own assert. Off by default and
# comptime-pruned, so production codegen is untouched. Shares the FA4 flag so
# one `-D` covers every attention split-K workspace. See KERN-3535.
comptime MLA_WS_POISON: Bool = get_defined_int["FA4_WS_POISON", 0]() != 0


def _mla_decode_sm100_dispatch_impl[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    *,
    ragged: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
    split_page_size: Int = 128,
    per_token_scale_rope_aware: Bool = False,
    sparse: Bool = False,
    rope_aware_kv_sparse: Bool = False,
    # Read-once shared-index fold (KERN-3141); see mla_decode_sm100_dispatch.
    fold_shared_index: Bool = False,
](
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    k: k_t,
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    scale: Float32,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    mask: mask_t,
    scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    batch_size: Int,
    q_max_seq_len: Int,
    num_partitions: Int,
    effective_max_cache_len: Int,
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Extra KV parameters (forwarded to mla_decode_sm100_sink_split_k).
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    comptime hw_info = ctx.default_device_info
    comptime sm_count = hw_info.sm_count
    comptime _half_sms = sm_count // 2

    comptime AccumType = get_accum_type[output.dtype]()
    # The split-K output is the 512 latent (shared by both rows), not the row
    # width the model stores, so it stays 512 under NoPE.
    comptime v_depth = 512
    comptime _is_fp8_kv = (k_t.dtype == DType.float8_e4m3fn)

    # Ensure KV cache page_size is evenly divisible by split_page_size.
    # If the KV cache page_size shrinks in the future, splits must not
    # straddle physical page boundaries.
    comptime assert (
        k_t.page_size % split_page_size == 0
    ), "KV cache page_size must be divisible by split_page_size"

    var block_z = batch_size * num_partitions

    if num_partitions > 1:
        comptime SplitAccumType = NonNullPointer[AccumType]
        # Create partial output buffer (same type as output - bfloat16)
        # Each split writes its partial attention result here
        # Note: Output dimension is v_depth (512), not depth (576)
        var o_accum_split_data = ctx.enqueue_create_buffer[output_type](
            Int(
                num_partitions
                * batch_size
                * q_max_seq_len
                * num_heads
                * v_depth
            )
        )
        var o_accum_split = TileTensor(
            o_accum_split_data,
            row_major(
                Coord(
                    Int(num_partitions),
                    Int(batch_size),
                    Int(q_max_seq_len),
                    Int(num_heads),
                    Int(v_depth),
                )
            ),
        )
        # Create LSE accumulator buffer (AccumType = float32 for numerical stability)
        var lse_accum_data = ctx.enqueue_create_buffer[AccumType](
            Int(num_partitions * batch_size * q_max_seq_len * num_heads)
        )
        comptime if MLA_WS_POISON:
            ctx.enqueue_memset(o_accum_split_data, nan[output_type]())
            ctx.enqueue_memset(lse_accum_data, nan[AccumType]())
        var lse_accum_split = TileTensor(
            lse_accum_data,
            row_major(
                Coord(
                    Int(num_partitions),
                    Int(batch_size),
                    Int(q_max_seq_len),
                    Int(num_heads),
                )
            ),
        )
        var lse_accum_split_ptr: SplitAccumType = {
            lse_accum_split.to_device_buffer(ctx)
            .unsafe_ptr()
            .as_imm()
            .as_unsafe_any_origin()
        }

        # Get input_row_offsets pointer for combine kernel's ragged output writes.
        var input_row_offsets_ptr = rebind[
            UnsafePointer[UInt32, origin=MutAnyOrigin]
        ](valid_length.ptr)

        # Inner function parameterized on has_attn_sink to specialize both
        # the decode kernel and combine kernel at compile time. The runtime
        # branch on attn_sink_ptr happens once (below) to select the right
        # compile-time specialization.
        @__parameter
        def _launch_split_k_path[_has_attn_sink: Bool]() raises:
            # Launch main MLA decode kernel (writes partial results to accumulators)
            mla_decode_sm100_sink_split_k[
                q_type=q_type,
                k_t=k_t,
                output_type=output_type,
                mask_t=mask_t,
                config=config,
                depth=depth,
                num_heads=num_heads,
                SplitAccumType=SplitAccumType,
                group=group,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=True,
                split_page_size=split_page_size,
                per_token_scale_rope_aware=per_token_scale_rope_aware,
                has_attn_sink=_has_attn_sink,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
                fold_shared_index=fold_shared_index,
            ](
                q,
                k,
                o_accum_split,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_length,
                mask,
                scales_ptr,
                scalar_args_buf,
                ctx,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
                effective_max_cache_len=effective_max_cache_len,
                logical_indices=logical_indices,
            )

            # Dispatch to specialized kernel based on num_partitions for compile-time unrolling.
            # Supports up to sm_count//2 splits to allow higher SM utilization.
            @__parameter
            def launch_combine[n_splits: Int, wph: Int]() raises:
                mla_decode_combine_partial_outputs[
                    output_type=output_type,
                    accum_type=AccumType,
                    head_dim=v_depth,
                    num_splits=n_splits,
                    ragged=ragged,
                    warps_per_head=wph,
                    has_attn_sink=_has_attn_sink,
                ](
                    o_accum_split,
                    lse_accum_split,
                    output,
                    input_row_offsets_ptr,
                    attn_sink_ptr,
                    batch_size,
                    q_max_seq_len,
                    Int(num_heads),
                    ctx,
                )

            @__parameter
            def launch_combine_split_parallel[n_splits: Int]() raises:
                mla_decode_combine_partial_outputs[
                    output_type=output_type,
                    accum_type=AccumType,
                    head_dim=v_depth,
                    num_splits=n_splits,
                    ragged=ragged,
                    warps_per_head=1,
                    has_attn_sink=_has_attn_sink,
                    split_parallel=True,
                ](
                    o_accum_split,
                    lse_accum_split,
                    output,
                    input_row_offsets_ptr,
                    attn_sink_ptr,
                    batch_size,
                    q_max_seq_len,
                    Int(num_heads),
                    ctx,
                )

            @__parameter
            def dispatch_combine[wph: Int]() raises:
                """Dispatch the combine kernel with the given warps_per_head,
                matching num_partitions to the correct compile-time bucket.

                Raises:
                    If the kernel dispatch fails, or if `num_partitions` is not
                    a compiled bucket.
                """
                var launched = False
                comptime for _b in range(_NUM_PARTITION_BUCKETS):
                    comptime if _get_partition_bucket[_half_sms, _b]() >= 2:
                        if (
                            num_partitions
                            == _get_partition_bucket[_half_sms, _b]()
                        ):
                            launch_combine[
                                _get_partition_bucket[_half_sms, _b](), wph
                            ]()
                            launched = True
                if not launched:
                    raise _unbucketed_split_error(num_partitions)

            @__parameter
            def dispatch_combine_split_parallel() raises:
                """Dispatch the split-parallel combine kernel, matching
                num_partitions to the correct compile-time bucket.

                Raises:
                    If the kernel dispatch fails, or if `num_partitions` is not
                    a compiled bucket.
                """
                var launched = False
                comptime for _b in range(_NUM_PARTITION_BUCKETS):
                    comptime if _get_partition_bucket[_half_sms, _b]() >= 2:
                        if (
                            num_partitions
                            == _get_partition_bucket[_half_sms, _b]()
                        ):
                            launch_combine_split_parallel[
                                _get_partition_bucket[_half_sms, _b]()
                            ]()
                            launched = True
                if not launched:
                    raise _unbucketed_split_error(num_partitions)

            # Choose combine strategy based on split count and batch size.
            #
            # Split-parallel combine: 8 warps per CTA each process a range
            # of splits independently, then tree-reduce in shared memory.
            # This gives 8x memory stream parallelism and avoids the massive
            # compile-time unrolled loop of the original kernel. Best for
            # long KV cache (>= 16384 tokens) where the many splits benefit
            # from 8x parallel memory streams.
            #
            # Original combine: warps cooperate on head_dim within each split.
            # Better for moderate cache lengths where the per-split overhead
            # of split-parallel dominates.
            #
            # Decision: use split-parallel when cache_length >= 16384.
            # For shorter cache, use the original kernel with wph tuning.
            #
            # The original kernel's wph selection logic follows (unchanged):
            #
            # The combine grid is (batch_size, seq_len, ceildiv(num_heads, hpb))
            # where hpb = heads_per_block = 8 // wph. Each CTA processes hpb
            # heads, using wph warps per head. The total combine CTA count is:
            #   batch_size * seq_len * ceildiv(num_heads, 8 // wph)
            #
            # This is a heuristic based on the following observations and empirical
            # tuning for B200 with 148 SMs:
            #
            # For DeepSeek V3/R1 (num_heads=128, seq_len=1):
            #   wph=2: hpb=4, grid_z=32,  combine CTAs = bs * 32
            #   wph=4: hpb=2, grid_z=64,  combine CTAs = bs * 64
            #   wph=8: hpb=1, grid_z=128, combine CTAs = bs * 128
            #
            # The optimal wph depends on two factors:
            #
            # 1. Batch size (controls combine CTA count): large batch means more
            #    CTAs launched, so lower wph (fewer CTAs) reduces combine overhead.
            #
            # 2. Number of splits (work per CTA): with few splits (np <= 4), each
            #    CTA only reduces 2-4 partial results -- the work per CTA is tiny
            #    regardless of wph. In this case, wph=4 beats wph=2 because the
            #    extra warps reduce per-CTA latency via more parallel vector loads,
            #    and the CTA count difference (e.g., bs*64 vs bs*32) is secondary
            #    since each CTA finishes very quickly.
            #    With many splits (np > 4), the combine work per CTA is non-trivial
            #    and CTA count dominates, so lower wph is preferred.
            #
            # We use combine_ctas_base (the combine CTA count at wph=2, where
            # hpb=4) as the decision metric. This adapts to models with different
            # num_heads, unlike raw batch_size thresholds.
            #
            # For DeepSeek V3/R1 (num_heads=128, q_max_seq_len=1):
            #   combine_ctas_base = bs * 32
            #   combine_ctas_base >= 2048 <==> bs >= 64
            #   combine_ctas_base >= 512  <==> bs >= 16
            #
            # Decision matrix (empirically tuned for B200 with 148 SMs):
            #   cache_len >= 16384:                                 split-parallel
            #   BF16: ctas >= 4096 AND np <= 4 AND cache <= 1280:   wph=1
            #   ctas >= 2048 AND np > 4:                            wph=2
            #   ctas >= 512:                                        wph=4
            #   ctas < 512 (small grid):                            wph=8
            #
            # The wph=1 path is BF16-only. FP8 decode finishes ~2x faster
            # (half the KV bytes), leaving less PDL overlap for the combine
            # kernel. With wph=1 the combine kernel has too few warps per
            # head to sustain memory throughput. FP8 falls through to the
            # wph=4 path which provides better per-head parallelism.
            #   bs=128, cl=1024 FP8: wph=1 -> 37.3us, wph=4 -> 34.8us
            var combine_ctas_base = (
                batch_size * q_max_seq_len * ceildiv(num_heads, 4)
            )

            # Actual combine CTA count at wph=8 (highest CTA count candidate).
            # Used as a secondary guard to prevent excessive wave counts when
            # combine_ctas_base (computed at wph=2) falls below the primary
            # thresholds. This is especially important for models with fewer
            # heads (e.g., Kimi K2.5 with 64 heads) where ctas_base is half
            # of DeepSeek's but the actual wph=8 CTA count can still be large.
            #
            # For Kimi K2.5 (num_heads=64):
            #   wph=8: hpb=1, grid_z=64,  CTAs = bs * 64
            #   wph=4: hpb=2, grid_z=32,  CTAs = bs * 32
            #   wph=2: hpb=4, grid_z=16,  CTAs = bs * 16
            #
            #   bs=8: ctas_base=128 < 512 (old path -> wph=8, 512 CTAs, 3.5
            #         waves). With this guard: 8*64=512 > 296, -> wph=4,
            #         256 CTAs (1.7 waves).
            comptime _ctas_wph8 = ceildiv(num_heads, 1)  # hpb=1 at wph=8

            if effective_max_cache_len >= 16384 and batch_size <= 2:
                dispatch_combine_split_parallel()
            elif (
                combine_ctas_base >= 4096
                and num_partitions <= 4
                and effective_max_cache_len <= 1280
                and not _is_fp8_kv
            ):
                dispatch_combine[1]()
            elif combine_ctas_base >= 2048 and num_partitions > 4:
                dispatch_combine[2]()
            elif combine_ctas_base >= 512:
                dispatch_combine[4]()
            elif batch_size * _ctas_wph8 > sm_count * 2:
                dispatch_combine[4]()
            else:
                dispatch_combine[8]()

        # Runtime branch: specialize on has_attn_sink for both the decode
        # kernel and the combine kernel. When attn_sink_ptr is null, the
        # has_attn_sink=False path generates zero overhead.
        if attn_sink_ptr:
            _launch_split_k_path[True]()
        else:
            _launch_split_k_path[False]()
    else:
        comptime SplitAccumType = NullPointer[AccumType]
        var lse_accum_split_ptr: SplitAccumType = {}

        @__parameter
        def _launch_no_split_path[_has_attn_sink: Bool]() raises:
            mla_decode_sm100_sink_split_k[
                q_type=q_type,
                k_t=k_t,
                output_type=output_type,
                mask_t=mask_t,
                config=config,
                depth=depth,
                num_heads=num_heads,
                SplitAccumType=SplitAccumType,
                group=group,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=False,
                split_page_size=split_page_size,
                per_token_scale_rope_aware=per_token_scale_rope_aware,
                has_attn_sink=_has_attn_sink,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
                fold_shared_index=fold_shared_index,
            ](
                q,
                k,
                output,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_length,
                mask,
                scales_ptr,
                scalar_args_buf,
                ctx,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
                effective_max_cache_len=effective_max_cache_len,
                logical_indices=logical_indices,
            )

        if attn_sink_ptr:
            _launch_no_split_path[True]()
        else:
            _launch_no_split_path[False]()


def mla_decode_sm100_sink_split_k[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    SplitAccumType: OptionalPointer,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
    decoding_warp_split_k: Bool,
    split_page_size: Int = 128,
    per_token_scale_rope_aware: Bool = False,
    has_attn_sink: Bool = False,
    sparse: Bool = False,
    # Sparse-only routing flag: when True, route to the BF16-rope sparse
    # kernel (split FP8 nope + BF16 rope, two TMAs). When False (default),
    # route to the all-FP8 sparse kernel (single 576-byte gather4 TMA).
    # Only meaningful when `sparse=True`. Ignored for dense paths.
    rope_aware_kv_sparse: Bool = False,
    # Read-once shared-index MTP fold (KERN-3141); see mla_decode_sm100_dispatch.
    fold_shared_index: Bool = False,
](
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    k: k_t,
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    mask: mask_t,
    scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Extra KV: separate always-attend cache. When extra_k is provided
    # (non-default), the sparse kernel appends extra_topk tokens after
    # the original topk tokens in a unified loop.
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Effective max cache length.  Layout G structural eligibility uses
    # `num_heads * q_len <= BM_G(32)`.  Defaults to 0 so unrelated callers
    # (BF16, sparse, etc.) pass through to the BN_QK=64 branch unchanged.
    effective_max_cache_len: Int = 0,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    comptime _scale_block_size = k_t.quantization_granularity if k_t.quantization_enabled else 0
    # Use native FP8 path when:
    # 1. KV is FP8 tensorwise (scale_block_size == 0)
    # 2. Q is also FP8 (q_type must match kv_type) — the pipeline provides FP8 Q
    # When Q is BF16, fall through to the old FP8 converter or BF16 path.
    comptime _native_fp8 = (
        k_t.dtype == .float8_e4m3fn
        and _scale_block_size == 0
        and q_type == .float8_e4m3fn
    )
    # Per-tensor rope-aware: split content (FP8 tensorwise) + rope (BF16) path
    comptime _per_token_scale_rope_aware = per_token_scale_rope_aware

    # For native FP8: Q is FP8 (1 byte) but swizzle_mode is the output
    # swizzle (SWIZZLE_128B for BF16). Using size_of[q_type]()=1 with
    # SWIZZLE_128B gives swizzle_elems=128, causing padded_q_depth=640
    # instead of 576. Use output_type size (2 for BF16) so
    # swizzle_elems=64 and padded dims are correct.
    comptime _dtype_size = size_of[output_type]() if _native_fp8 else size_of[
        q_type
    ]()
    # Comptime-local aliases so the nested `_launch_sparse_kv_fp8_fold_sel`
    # closure can read these in its `comptime if` (nested closures capture
    # comptime locals, not the enclosing function's comptime params by bare
    # name).
    comptime _fold_shared_index = fold_shared_index
    comptime _has_attn_sink = has_attn_sink
    # Scoped to the sparse native-FP8 kernel ONLY: this same mla_config
    # object is also read by the dense native_fp8 branch below (`elif
    # _native_fp8:`) and by mla_config_g (Layout G); both keep their
    # existing single-buffer SW64-gather kernel bodies, so
    # native_fp8_unified_gather must be False there.
    comptime _native_fp8_unified_gather = sparse and _native_fp8
    # V/output is the 512 latent. SMEM stays 576 so the trailing block P
    # aliases survives a NoPE row. Neither tracks the row the model stores.
    comptime _latent_depth = 512
    comptime _smem_q_depth = 576
    comptime mla_config = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,  # num_q_heads/h_k(1)
        depth=_latent_depth,
        q_depth=_smem_q_depth,
        input_q_depth=depth,  # 576 with rotary, 512 NoPE
        dtype_size=_dtype_size,
        kv_type_size=size_of[k_t.dtype](),
        swizzle_mode=config.swizzle_mode,
        kv_mma_swizzle_mode=config.swizzle_mode,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
        split_page_size=split_page_size,
        scale_block_size=_scale_block_size,
        native_fp8=_native_fp8,
        per_token_scale_rope_aware=_per_token_scale_rope_aware,
        native_fp8_unified_gather=_native_fp8_unified_gather,
    )
    var num_rows_q = num_matrix_view_rows_decode(q)

    # `depth` and `BK` are the row as the model stores it (NoPE narrows both).
    var k_tma_op = k.create_tma_tile[
        BN=mla_config.BK_PV,  # tile_m =64
        depth=mla_config.input_q_depth,
        BK=mla_config.input_q_depth,
        swizzle_mode=mla_config.kv_tma_swizzle_mode,
    ](ctx)
    var o_ptr = rebind[UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]](
        output.ptr
    )
    var num_rows_o = num_matrix_view_rows_decode(output)
    var o_tma_op = tma_tile_o[
        swizzle_mode=mla_config.swizzle_mode,
        BM=mla_config.out_rows,
        BK=mla_config.BN_PV // 4,
        depth=mla_config.depth,
    ](ctx, o_ptr, num_rows_o)

    # =========================================================================
    # Sparse routing: when sparse=True (comptime), use the gather4 sparse
    # kernel instead of the standard page-table path.
    # =========================================================================
    comptime if sparse:
        var q_ptr = rebind[UnsafePointer[Scalar[q_type], origin=MutAnyOrigin]](
            q.ptr
        )

        # Gate the all-FP8 KV sparse variant on the explicit caller flag.
        # `rope_aware_kv_sparse=True` (test-only) routes to the BF16-rope
        # sparse kernel; default `False` routes to the all-FP8 path here
        # (single 576-byte gather4 TMA).  Production callers never set
        # the flag and so always get the all-FP8 kernel.
        comptime if not rope_aware_kv_sparse:
            # ---------- BF16 KV sparse dispatch ----------
            # BF16 KV cache: single BF16 + SWIZZLE_128B gather4 TMA
            # descriptor covering the full 576-element row (1152 bytes).
            comptime if k_t.dtype == .bfloat16:
                comptime _kv_bf16_tile_width = mla_config.input_q_depth
                var k_gather4_tma_bf16 = k.create_gather4_tma_tile[
                    tile_width=_kv_bf16_tile_width,
                    tile_stride=_kv_bf16_tile_width,
                    swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
                    tile_height=mla_config.BK_PV,
                    tma_dtype=DType.bfloat16,
                    l2_promotion=TensorMapL2Promotion.L2_128B,
                ](ctx)

                var extra_k_val_bf16 = extra_k.or_else(k)
                var extra_k_gather4_tma_bf16 = (
                    extra_k_val_bf16.create_gather4_tma_tile[
                        tile_width=_kv_bf16_tile_width,
                        tile_stride=_kv_bf16_tile_width,
                        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
                        tile_height=mla_config.BK_PV,
                        tma_dtype=DType.bfloat16,
                        l2_promotion=TensorMapL2Promotion.L2_128B,
                    ](ctx)
                )
                var extra_kv_lut_val_bf16 = extra_k_val_bf16
                var q_tma_sparse = tma_tile_qo[
                    swizzle_mode=mla_config.swizzle_mode,
                    BM=mla_config.BM,
                    BK=mla_config.input_q_depth,
                    depth=mla_config.input_q_depth,
                ](ctx, q_ptr, num_rows_q)

                @always_inline
                def _launch_sparse_kv_bf16[
                    _has_extra_kv: Bool, _has_variable_topk: Bool
                ]() raises {imm}:
                    if ragged:
                        comptime ValidLengthType = NonNullPointer[.uint32]
                        var valid_len: ValidLengthType = {
                            valid_length.ptr.as_imm().as_unsafe_any_origin()
                        }
                        launch_mla_sm100_decode_sparse_kv_bf16[
                            q_type=q_type,
                            KVLUTType=k_t,
                            output_type=output_type,
                            SplitAccumType=SplitAccumType,
                            MaskType=mask_t,
                            config=mla_config,
                            ValidLengthType=ValidLengthType,
                            ragged=True,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            has_attn_sink=has_attn_sink,
                            has_extra_kv=_has_extra_kv,
                            has_variable_topk=_has_variable_topk,
                        ](
                            q_tma_sparse,
                            k_gather4_tma_bf16,
                            o_tma_op,
                            k,
                            lse_accum_split_ptr,
                            scale,
                            batch_size,
                            block_z,
                            num_partitions,
                            q_max_seq_len,
                            valid_len,
                            mask,
                            d_indices,
                            indices_stride,
                            topk_lengths,
                            attn_sink_ptr,
                            extra_k_gather4_tma_bf16,
                            extra_kv_lut_val_bf16,
                            extra_d_indices,
                            extra_topk_lengths,
                            extra_indices_stride,
                            scalar_args_buf,
                            ctx,
                        )
                    else:
                        comptime ValidLengthType = NullPointer[.uint32]
                        var valid_len: ValidLengthType = {}
                        launch_mla_sm100_decode_sparse_kv_bf16[
                            q_type=q_type,
                            KVLUTType=k_t,
                            output_type=output_type,
                            SplitAccumType=SplitAccumType,
                            MaskType=mask_t,
                            config=mla_config,
                            ValidLengthType=ValidLengthType,
                            ragged=False,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            has_attn_sink=has_attn_sink,
                            has_extra_kv=_has_extra_kv,
                            has_variable_topk=_has_variable_topk,
                        ](
                            q_tma_sparse,
                            k_gather4_tma_bf16,
                            o_tma_op,
                            k,
                            lse_accum_split_ptr,
                            scale,
                            batch_size,
                            block_z,
                            num_partitions,
                            q_max_seq_len,
                            valid_len,
                            mask,
                            d_indices,
                            indices_stride,
                            topk_lengths,
                            attn_sink_ptr,
                            extra_k_gather4_tma_bf16,
                            extra_kv_lut_val_bf16,
                            extra_d_indices,
                            extra_topk_lengths,
                            extra_indices_stride,
                            scalar_args_buf,
                            ctx,
                        )

                _unswitch_raises(
                    extra_k is not None,
                    Bool(topk_lengths),
                    _launch_sparse_kv_bf16,
                )
                return

            # ---------- FP8 KV sparse dispatch (default) ----------
            # Single K gather4 TMA covering full 576-byte row
            # (INT64, SWIZZLE_NONE, tile_width=72 INT64 = 576 B).
            comptime _kv_tile_width = mla_config.input_q_depth // 8
            var k_gather4_tma = k.create_gather4_tma_tile[
                tile_width=_kv_tile_width,
                tile_stride=_kv_tile_width,
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                tile_height=mla_config.BK_PV,
                tma_dtype=DType.int64,
                l2_promotion=TensorMapL2Promotion.L2_128B,
            ](ctx)

            var extra_k_val_fp8 = extra_k.or_else(k)
            var extra_k_gather4_tma = extra_k_val_fp8.create_gather4_tma_tile[
                tile_width=_kv_tile_width,
                tile_stride=_kv_tile_width,
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                tile_height=mla_config.BK_PV,
                tma_dtype=DType.int64,
                l2_promotion=TensorMapL2Promotion.L2_128B,
            ](ctx)
            var extra_kv_lut_val_fp8 = extra_k_val_fp8

            comptime if _native_fp8:
                var q_tma_sparse_fp8 = tma_tile_qo[
                    swizzle_mode=mla_config.kv_tma_swizzle_mode,
                    BM=mla_config.BM,
                    BK=mla_config.input_q_depth,
                    depth=mla_config.input_q_depth,
                ](ctx, q_ptr, num_rows_q)

                # Unified gather: reuse the SAME contiguous INT64/
                # SWIZZLE_NONE gather4 descriptor the old FP8 KV path built
                # above (k_gather4_tma / extra_k_gather4_tma) -- the native
                # kernel's own re-swizzle warpgroup now reproduces the SW64
                # layout in SMEM instead of TMA hardware doing it directly,
                # so no separate SW64/FP8 gather4 descriptor is needed here.

                @__parameter
                @always_inline
                def _launch_sparse_qkv_fp8[
                    _has_extra_kv: Bool,
                    _has_variable_topk: Bool,
                    _fold_shared_index_val: Bool = False,
                    _q_len_fold_val: Int = 1,
                ]() raises:
                    if ragged:
                        comptime ValidLengthType = NonNullPointer[.uint32]
                        var valid_len: ValidLengthType = {
                            valid_length.ptr.as_imm().as_unsafe_any_origin()
                        }
                        launch_mla_sm100_decode_sparse_qkv_fp8[
                            q_type=q_type,
                            KVLUTType=k_t,
                            output_type=output_type,
                            SplitAccumType=SplitAccumType,
                            MaskType=mask_t,
                            config=mla_config,
                            ValidLengthType=ValidLengthType,
                            ragged=True,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            has_attn_sink=has_attn_sink,
                            has_extra_kv=_has_extra_kv,
                            has_variable_topk=_has_variable_topk,
                            fold_shared_index=_fold_shared_index_val,
                            q_len_fold=_q_len_fold_val,
                        ](
                            q_tma_sparse_fp8,
                            k_gather4_tma,
                            o_tma_op,
                            k,
                            lse_accum_split_ptr,
                            scale,
                            batch_size,
                            block_z,
                            num_partitions,
                            q_max_seq_len,
                            valid_len,
                            mask,
                            d_indices,
                            indices_stride,
                            topk_lengths,
                            scales_ptr,
                            attn_sink_ptr,
                            extra_k_gather4_tma,
                            extra_kv_lut_val_fp8,
                            extra_d_indices,
                            extra_topk_lengths,
                            extra_indices_stride,
                            extra_scales_ptr,
                            scalar_args_buf,
                            ctx,
                            logical_indices=logical_indices,
                        )
                    else:
                        comptime ValidLengthType = NullPointer[.uint32]
                        var valid_len: ValidLengthType = {}
                        launch_mla_sm100_decode_sparse_qkv_fp8[
                            q_type=q_type,
                            KVLUTType=k_t,
                            output_type=output_type,
                            SplitAccumType=SplitAccumType,
                            MaskType=mask_t,
                            config=mla_config,
                            ValidLengthType=ValidLengthType,
                            ragged=False,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            has_attn_sink=has_attn_sink,
                            has_extra_kv=_has_extra_kv,
                            has_variable_topk=_has_variable_topk,
                            fold_shared_index=_fold_shared_index_val,
                            q_len_fold=_q_len_fold_val,
                        ](
                            q_tma_sparse_fp8,
                            k_gather4_tma,
                            o_tma_op,
                            k,
                            lse_accum_split_ptr,
                            scale,
                            batch_size,
                            block_z,
                            num_partitions,
                            q_max_seq_len,
                            valid_len,
                            mask,
                            d_indices,
                            indices_stride,
                            topk_lengths,
                            scales_ptr,
                            attn_sink_ptr,
                            extra_k_gather4_tma,
                            extra_kv_lut_val_fp8,
                            extra_d_indices,
                            extra_topk_lengths,
                            extra_indices_stride,
                            extra_scales_ptr,
                            scalar_args_buf,
                            ctx,
                            logical_indices=logical_indices,
                        )

                @always_inline
                def _launch_sparse_qkv_fp8_fold_sel[
                    _has_extra_kv: Bool, _has_variable_topk: Bool
                ]() raises {imm}:
                    comptime _fold_ok = (
                        _fold_shared_index
                        and not _has_extra_kv
                        and not _has_variable_topk
                        and not _has_attn_sink
                    )
                    comptime if _fold_ok:
                        comptime for n in range(MIN_FOLD_Q, MAX_FOLD_Q + 1):
                            comptime if mla_config.num_q_heads * n <= mla_config.BM:
                                if q_max_seq_len == n:
                                    _launch_sparse_qkv_fp8[
                                        _has_extra_kv,
                                        _has_variable_topk,
                                        True,
                                        n,
                                    ]()
                                    return
                    _launch_sparse_qkv_fp8[_has_extra_kv, _has_variable_topk]()

                _unswitch_raises(
                    extra_k is not None,
                    Bool(topk_lengths),
                    _launch_sparse_qkv_fp8_fold_sel,
                )
                return

            # Mojo elaborates code after `comptime if _native_fp8: return` even
            # for the _native_fp8=True specialization. Without this guard,
            # Sparse_KV_FP8 (q_type=bfloat16 only) and the SWIZZLE_128B Q TMA
            # get instantiated for FP8-Q and ICE.
            comptime if not _native_fp8:
                # SWIZZLE_128B is safe: 576 BF16 bytes = 1152B, divisible by 128B.
                var q_tma_sparse = tma_tile_qo[
                    swizzle_mode=mla_config.swizzle_mode,
                    BM=mla_config.BM,
                    BK=mla_config.input_q_depth,
                    depth=mla_config.input_q_depth,
                ](ctx, q_ptr, num_rows_q)

                @__parameter
                @always_inline
                def _launch_sparse_kv_fp8[
                    _has_extra_kv: Bool,
                    _has_variable_topk: Bool,
                    _fold_shared_index: Bool = False,
                    _q_len_fold: Int = 1,
                ]() raises:
                    if ragged:
                        comptime ValidLengthType = NonNullPointer[.uint32]
                        var valid_len: ValidLengthType = {
                            valid_length.ptr.as_imm().as_unsafe_any_origin()
                        }
                        launch_mla_sm100_decode_sparse_kv_fp8[
                            q_type=q_type,
                            KVLUTType=k_t,
                            output_type=output_type,
                            SplitAccumType=SplitAccumType,
                            MaskType=mask_t,
                            config=mla_config,
                            ValidLengthType=ValidLengthType,
                            ragged=True,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            has_attn_sink=has_attn_sink,
                            has_extra_kv=_has_extra_kv,
                            has_variable_topk=_has_variable_topk,
                            fold_shared_index=_fold_shared_index,
                            q_len_fold=_q_len_fold,
                        ](
                            q_tma_sparse,
                            k_gather4_tma,
                            o_tma_op,
                            k,
                            lse_accum_split_ptr,
                            scale,
                            batch_size,
                            block_z,
                            num_partitions,
                            q_max_seq_len,
                            valid_len,
                            mask,
                            d_indices,
                            indices_stride,
                            topk_lengths,
                            scales_ptr,
                            attn_sink_ptr,
                            extra_k_gather4_tma,
                            extra_kv_lut_val_fp8,
                            extra_d_indices,
                            extra_topk_lengths,
                            extra_indices_stride,
                            extra_scales_ptr,
                            scalar_args_buf,
                            ctx,
                        )
                    else:
                        comptime ValidLengthType = NullPointer[.uint32]
                        var valid_len: ValidLengthType = {}
                        launch_mla_sm100_decode_sparse_kv_fp8[
                            q_type=q_type,
                            KVLUTType=k_t,
                            output_type=output_type,
                            SplitAccumType=SplitAccumType,
                            MaskType=mask_t,
                            config=mla_config,
                            ValidLengthType=ValidLengthType,
                            ragged=False,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            has_attn_sink=has_attn_sink,
                            has_extra_kv=_has_extra_kv,
                            has_variable_topk=_has_variable_topk,
                            fold_shared_index=_fold_shared_index,
                            q_len_fold=_q_len_fold,
                        ](
                            q_tma_sparse,
                            k_gather4_tma,
                            o_tma_op,
                            k,
                            lse_accum_split_ptr,
                            scale,
                            batch_size,
                            block_z,
                            num_partitions,
                            q_max_seq_len,
                            valid_len,
                            mask,
                            d_indices,
                            indices_stride,
                            topk_lengths,
                            scales_ptr,
                            attn_sink_ptr,
                            extra_k_gather4_tma,
                            extra_kv_lut_val_fp8,
                            extra_d_indices,
                            extra_topk_lengths,
                            extra_indices_stride,
                            extra_scales_ptr,
                            scalar_args_buf,
                            ctx,
                        )

                @always_inline
                def _launch_sparse_kv_fp8_fold_sel[
                    _has_extra_kv: Bool, _has_variable_topk: Bool
                ]() raises {imm}:
                    comptime _fold_ok = (
                        _fold_shared_index
                        and not _has_extra_kv
                        and not _has_variable_topk
                        and not _has_attn_sink
                    )
                    comptime if _fold_ok:
                        comptime for n in range(MIN_FOLD_Q, MAX_FOLD_Q + 1):
                            comptime if mla_config.num_q_heads * n <= mla_config.BM:
                                if q_max_seq_len == n:
                                    _launch_sparse_kv_fp8[
                                        _has_extra_kv,
                                        _has_variable_topk,
                                        True,
                                        n,
                                    ]()
                                    return
                    _launch_sparse_kv_fp8[_has_extra_kv, _has_variable_topk]()

                _unswitch_raises(
                    extra_k is not None,
                    Bool(topk_lengths),
                    _launch_sparse_kv_fp8_fold_sel,
                )
                return

        comptime if rope_aware_kv_sparse:
            # K_nope gather4 TMA: INT64, SWIZZLE_NONE (linear SMEM layout).
            # tile_width = nope only (padded_depth / 8 = 64 INT64 elements).
            # tile_stride = full row (nope + rope) / 8 = 80 INT64 elements.
            comptime _nope_tile_width = mla_config.padded_depth // 8
            comptime _nope_tile_stride = (
                mla_config.padded_depth + mla_config.rope_depth * 2
            ) // 8
            var k_nope_gather4_tma = k.create_gather4_tma_tile[
                tile_width=_nope_tile_width,
                tile_stride=_nope_tile_stride,
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                tile_height=mla_config.BK_PV,
                tma_dtype=DType.int64,
                l2_promotion=TensorMapL2Promotion.L2_128B,
            ](ctx)

            # K_rope gather4 TMA: BF16, SWIZZLE_128B.
            # Row stride in BF16 elements = total_row_bytes / sizeof(bf16).
            comptime _rope_gather4_tile_width = (
                mla_config.padded_depth + mla_config.rope_depth * 2
            ) // 2
            var k_rope_gather4_tma = k.create_rope_gather4_tma_tile[
                tile_width=_rope_gather4_tile_width,
                padded_depth=mla_config.padded_depth,
                swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
                tile_height=mla_config.BK_PV,
                l2_promotion=TensorMapL2Promotion.L2_128B,
            ](ctx)

            # Extra KV: create separate TMA descriptors from extra_k when provided.
            # When extra_k is None, we create dummy descriptors from k (they won't
            # be used since has_extra_kv=False eliminates all extra code paths).
            var extra_k_val = extra_k.or_else(k)
            var extra_k_nope_gather4_tma = extra_k_val.create_gather4_tma_tile[
                tile_width=_nope_tile_width,
                tile_stride=_nope_tile_stride,
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                tile_height=mla_config.BK_PV,
                tma_dtype=DType.int64,
                l2_promotion=TensorMapL2Promotion.L2_128B,
            ](ctx)
            var extra_k_rope_gather4_tma = (
                extra_k_val.create_rope_gather4_tma_tile[
                    tile_width=_rope_gather4_tile_width,
                    padded_depth=mla_config.padded_depth,
                    swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
                    tile_height=mla_config.BK_PV,
                    l2_promotion=TensorMapL2Promotion.L2_128B,
                ](ctx)
            )
            var extra_kv_lut_val = extra_k_val
            var q_tma_sparse = tma_tile_qo[
                swizzle_mode=mla_config.swizzle_mode,
                BM=mla_config.BM,
                BK=mla_config.input_q_depth,
                depth=mla_config.input_q_depth,
            ](ctx, q_ptr, num_rows_q)

            @always_inline
            def _launch_sparse[
                _has_extra_kv: Bool, _has_variable_topk: Bool
            ]() raises {imm}:
                if ragged:
                    comptime ValidLengthType = NonNullPointer[.uint32]
                    var valid_len: ValidLengthType = {
                        valid_length.ptr.as_imm().as_unsafe_any_origin()
                    }
                    launch_mla_sm100_decode_sparse[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        ValidLengthType=ValidLengthType,
                        ragged=True,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        has_attn_sink=has_attn_sink,
                        has_extra_kv=_has_extra_kv,
                        has_variable_topk=_has_variable_topk,
                    ](
                        q_tma_sparse,
                        k_nope_gather4_tma,
                        k_rope_gather4_tma,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        d_indices,
                        indices_stride,
                        topk_lengths,
                        scales_ptr,
                        attn_sink_ptr,
                        extra_k_nope_gather4_tma,
                        extra_k_rope_gather4_tma,
                        extra_kv_lut_val,
                        extra_d_indices,
                        extra_topk_lengths,
                        extra_indices_stride,
                        extra_scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )
                else:
                    comptime ValidLengthType = NullPointer[.uint32]
                    var valid_len: ValidLengthType = {}
                    launch_mla_sm100_decode_sparse[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        ValidLengthType=ValidLengthType,
                        ragged=False,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        has_attn_sink=has_attn_sink,
                        has_extra_kv=_has_extra_kv,
                        has_variable_topk=_has_variable_topk,
                    ](
                        q_tma_sparse,
                        k_nope_gather4_tma,
                        k_rope_gather4_tma,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        d_indices,
                        indices_stride,
                        topk_lengths,
                        scales_ptr,
                        attn_sink_ptr,
                        extra_k_nope_gather4_tma,
                        extra_k_rope_gather4_tma,
                        extra_kv_lut_val,
                        extra_d_indices,
                        extra_topk_lengths,
                        extra_indices_stride,
                        extra_scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )

            _unswitch_raises(
                extra_k is not None, Bool(topk_lengths), _launch_sparse
            )
            return

    # Per-token-scale rope-aware: split content (FP8) + rope (BF16) with separate TMAs.
    # Q buffer layout: FP8 content (512 bytes) | BF16 rope (128 bytes) per row = 640 bytes/row.
    # K cache layout: FP8 content (512 bytes) | BF16 rope (128 bytes) per row = 640 bytes/row.
    # The KV cache 640 bytes/row layout is enforced by create_rope_tma_tile in kv_cache/types.mojo.
    comptime if _per_token_scale_rope_aware:
        # Q row stride in FP8 bytes: 512 FP8 content + 64 BF16 rope = 640 bytes.
        # The `depth` parameter in tma_tile_qo sets the row stride of the
        # LayoutTensor, which the TMA descriptor uses as the global memory
        # stride.  It must equal the full row width so that consecutive
        # rows (heads/tokens) are read correctly.
        comptime _q_row_bytes = mla_config.padded_depth + mla_config.rope_depth * 2  # 640
        # Same stride in BF16 units for the rope TMA.
        comptime _q_row_bf16 = _q_row_bytes // 2  # 320

        # Q_nope TMA: FP8 content, SWIZZLE_64B, BM x padded_depth (512)
        var q_ptr_fp8_content = rebind[
            UnsafePointer[Float8_e4m3fn, origin=MutAnyOrigin]
        ](q.ptr)
        var q_nope_tma = tma_tile_qo[
            swizzle_mode=mla_config.content_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config.BM,
            BK=mla_config.padded_depth,  # 512
            depth=_q_row_bytes,  # 640 (full row stride in FP8 bytes)
        ](ctx, q_ptr_fp8_content, num_rows_q)

        # Q_rope TMA: BF16 rope, SWIZZLE_128B, BM x rope_depth (64)
        # Rope starts at byte offset padded_depth (512) from Q row start.
        var q_ptr_bf16_rope = rebind[
            UnsafePointer[BFloat16, origin=MutAnyOrigin]
        ](q.ptr + mla_config.padded_depth)
        var q_rope_tma = tma_tile_qo[
            swizzle_mode=mla_config.rope_swizzle_mode,  # SWIZZLE_128B
            BM=mla_config.BM,
            BK=mla_config.rope_depth,  # 64
            depth=_q_row_bf16,  # 320 (full row stride in BF16 elements)
        ](ctx, q_ptr_bf16_rope, num_rows_q)

        # K_content TMA: FP8 content from KV cache, SWIZZLE_64B, BK_PV x padded_depth (512).
        # The KV cache has 640 bytes/row layout (512 FP8 content + 128 BF16 rope).
        # create_tma_tile reads only the first 512 bytes (FP8 content) per row.
        var k_content_tma = k.create_tma_tile[
            BN=mla_config.BK_PV,  # 64
            depth=mla_config.padded_depth,  # 512
            BK=mla_config.padded_depth,  # 512
            swizzle_mode=mla_config.content_swizzle_mode,  # SWIZZLE_64B
        ](ctx)

        # K_rope TMA: BF16 rope from KV cache, SWIZZLE_128B, BK_PV x rope_depth (64).
        # The KV cache row layout is padded_depth FP8 bytes followed by
        # rope_depth BF16 elements.  create_rope_tma_tile offsets the base
        # pointer by padded_depth bytes and reinterprets as BF16.
        var k_rope_tma = k.create_rope_tma_tile[
            BN=mla_config.BK_PV,  # 64
            BK=mla_config.rope_depth,  # 64
            padded_depth=mla_config.padded_depth,  # 512
            swizzle_mode=mla_config.rope_swizzle_mode,  # SWIZZLE_128B
        ](ctx)

        # Scales TMA: per-token float32 scales loaded via TMA.
        # The scales tensor is [total_blocks, page_size, 1, 1] in row-major,
        # indexed by row_idx (same paging as KV cache blocks).
        # We treat it as a flat [1, total_elements] 2D tensor for TMA.
        var _total_scale_elements = k.num_kv_rows()
        var scale_tma = tma_tile_scales[BN_QK=mla_config.BN_QK](
            ctx, scales_ptr, _total_scale_elements
        )

        if ragged:
            comptime ValidLengthType = NonNullPointer[.uint32]
            var valid_len: ValidLengthType = {
                valid_length.ptr.as_imm().as_unsafe_any_origin()
            }
            launch_mla_sm100_decode_fp8_per_token_scale_rope_aware[
                q_type=q_type,
                KVLUTType=k_t,
                output_type=output_type,
                SplitAccumType=SplitAccumType,
                MaskType=mask_t,
                config=mla_config,
                ValidLengthType=ValidLengthType,
                ragged=True,
                _is_cache_length_accurate=_is_cache_length_accurate,
                has_per_token_scales=True,
            ](
                q_nope_tma,
                q_rope_tma,
                k_content_tma,
                k_rope_tma,
                scale_tma,
                o_tma_op,
                k,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_len,
                mask,
                q_scale_ptr,
                scalar_args_buf,
                ctx,
            )
        else:
            comptime ValidLengthType = NullPointer[.uint32]
            var valid_len: ValidLengthType = {}
            launch_mla_sm100_decode_fp8_per_token_scale_rope_aware[
                q_type=q_type,
                KVLUTType=k_t,
                output_type=output_type,
                SplitAccumType=SplitAccumType,
                MaskType=mask_t,
                config=mla_config,
                ValidLengthType=ValidLengthType,
                ragged=False,
                _is_cache_length_accurate=_is_cache_length_accurate,
                has_per_token_scales=True,
            ](
                q_nope_tma,
                q_rope_tma,
                k_content_tma,
                k_rope_tma,
                scale_tma,
                o_tma_op,
                k,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_len,
                mask,
                q_scale_ptr,
                scalar_args_buf,
                ctx,
            )
    elif _native_fp8:
        var q_ptr_fp8 = rebind[
            UnsafePointer[Scalar[k_t.dtype], origin=MutAnyOrigin]
        ](q.ptr)
        var q_tma_fp8 = tma_tile_qo[
            swizzle_mode=mla_config.kv_tma_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config.BM,
            BK=mla_config.input_q_depth,
            depth=mla_config.input_q_depth,
        ](ctx, q_ptr_fp8, num_rows_q)

        # Layout G config + BM=32 Q TMA tile. Built unconditionally so the
        # comptime branch inside `_launch_r` / `_launch_n` can reference
        # them; when the runtime gate misses, the unused branch is pruned.
        comptime mla_config_g = MLA_SM100_Decode_Config(
            num_q_heads=mla_config.num_q_heads,
            group=mla_config.group,
            depth=mla_config.depth,
            q_depth=mla_config.q_depth,
            input_q_depth=mla_config.input_q_depth,
            dtype_size=mla_config.dtype_size,
            kv_type_size=size_of[k_t.dtype](),
            swizzle_mode=mla_config.swizzle_mode,
            kv_mma_swizzle_mode=mla_config.kv_mma_swizzle_mode,
            page_size=mla_config.page_size,
            decoding_warp_split_k=mla_config.decoding_warp_split_k,
            split_page_size=mla_config.split_page_size,
            scale_block_size=mla_config.scale_block_size,
            native_fp8=True,
            per_token_scale_rope_aware=mla_config.per_token_scale_rope_aware,
            decode_layout_g=True,
        )
        var q_tma_fp8_g = tma_tile_qo[
            swizzle_mode=mla_config_g.kv_tma_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config_g.BM,  # 32
            BK=mla_config_g.input_q_depth,
            depth=mla_config_g.input_q_depth,
        ](ctx, q_ptr_fp8, num_rows_q)

        # Dispatch is routed at comptime by num_q_heads × q_max_seq_len.
        # q_max_seq_len > 1 implies spec decoding (1 actual + N spec ahead),
        # capped at MAX_FOLD_Q (1 actual + (MAX_FOLD_Q - 1) spec). For each q
        # in {MIN_FOLD_Q..MAX_FOLD_Q}: num_heads × q ≤ 32 → Layout-G fold
        # (BM=32), ≤ 64 → Layout-E fold (BM=64), > 64 → non-fold (kernel
        # handles q in grid dim). For q=1 (regular decode), num_heads ≤ 32 →
        # Layout-G non-fold, else → Layout-E non-fold.

        if ragged:
            comptime ValidLengthType = NonNullPointer[.uint32]
            var valid_len: ValidLengthType = {
                valid_length.ptr.as_imm().as_unsafe_any_origin()
            }

            @__parameter
            @always_inline
            def _launch_r[
                _fold_q: Bool,
                _q_len_fold: Int,
                _layout_g: Bool = False,
            ]() raises:
                comptime if _layout_g:
                    launch_mla_sm100_decode_native_fp8_layout_g[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config_e=mla_config,
                        config_g=mla_config_g,
                        ValidLengthType=ValidLengthType,
                        ragged=True,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        fold_q=_fold_q,
                        q_len_fold=_q_len_fold,
                    ](
                        q_tma_fp8_g,
                        k_tma_op,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )
                else:
                    launch_mla_sm100_decode_native_fp8[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        ValidLengthType=ValidLengthType,
                        ragged=True,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        fold_q=_fold_q,
                        q_len_fold=_q_len_fold,
                    ](
                        q_tma_fp8,
                        k_tma_op,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )

            # Spec decoding implied by q > 1; cap at MAX_FOLD_Q (1 actual +
            # (MAX_FOLD_Q - 1) spec).
            comptime for n in range(MIN_FOLD_Q, MAX_FOLD_Q + 1):
                comptime if mla_config.num_q_heads * n <= 32:
                    if q_max_seq_len == n:
                        _launch_r[True, n, True]()  # Layout-G fold (BM=32)
                        return
                elif mla_config.num_q_heads * n <= 64:
                    if q_max_seq_len == n:
                        _launch_r[True, n, False]()  # Layout-E fold (BM=64)
                        return
                else:
                    # num_heads * n > 64: kernel can't fold; fall through to
                    # non-fold (kernel handles q in grid dim).
                    if q_max_seq_len == n:
                        _launch_r[False, 1, False]()  # non-fold
                        return

            # q_max_seq_len == 1 (regular decode) or > MAX_FOLD_Q (shouldn't
            # happen, fallback).
            comptime if mla_config.num_q_heads <= 32:
                _launch_r[False, 1, True]()  # Layout-G non-fold
            else:
                _launch_r[False, 1, False]()  # Layout-E non-fold
            return
        else:
            comptime ValidLengthType = NullPointer[.uint32]
            var valid_len: ValidLengthType = {}

            @__parameter
            @always_inline
            def _launch_n[
                _fold_q: Bool,
                _q_len_fold: Int,
                _layout_g: Bool = False,
            ]() raises:
                comptime if _layout_g:
                    launch_mla_sm100_decode_native_fp8_layout_g[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config_e=mla_config,
                        config_g=mla_config_g,
                        ValidLengthType=ValidLengthType,
                        ragged=False,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        fold_q=_fold_q,
                        q_len_fold=_q_len_fold,
                    ](
                        q_tma_fp8_g,
                        k_tma_op,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )
                else:
                    launch_mla_sm100_decode_native_fp8[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        ValidLengthType=ValidLengthType,
                        ragged=False,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        fold_q=_fold_q,
                        q_len_fold=_q_len_fold,
                    ](
                        q_tma_fp8,
                        k_tma_op,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )

            # Spec decoding implied by q > 1; cap at MAX_FOLD_Q (1 actual +
            # (MAX_FOLD_Q - 1) spec).
            comptime for n in range(MIN_FOLD_Q, MAX_FOLD_Q + 1):
                comptime if mla_config.num_q_heads * n <= 32:
                    if q_max_seq_len == n:
                        _launch_n[True, n, True]()  # Layout-G fold (BM=32)
                        return
                elif mla_config.num_q_heads * n <= 64:
                    if q_max_seq_len == n:
                        _launch_n[True, n, False]()  # Layout-E fold (BM=64)
                        return
                else:
                    # num_heads * n > 64: kernel can't fold; fall through to
                    # non-fold (kernel handles q in grid dim).
                    if q_max_seq_len == n:
                        _launch_n[False, 1, False]()  # non-fold
                        return

            # q_max_seq_len == 1 (regular decode) or > MAX_FOLD_Q (shouldn't
            # happen, fallback).
            comptime if mla_config.num_q_heads <= 32:
                _launch_n[False, 1, True]()  # Layout-G non-fold
            else:
                _launch_n[False, 1, False]()  # Layout-E non-fold
            return
    else:
        # BF16 / old FP8 converter path: Q is BF16, create BF16 Q TMA.
        var q_ptr = rebind[UnsafePointer[Scalar[q_type], origin=MutAnyOrigin]](
            q.ptr
        )
        var q_tma_op = tma_tile_qo[
            swizzle_mode=mla_config.swizzle_mode,
            BM=mla_config.BM,
            BK=mla_config.input_q_depth,
            depth=mla_config.input_q_depth,
        ](ctx, q_ptr, num_rows_q)

        if ragged:
            comptime ValidLengthType = NonNullPointer[.uint32]
            var valid_len: ValidLengthType = {
                valid_length.ptr.as_imm().as_unsafe_any_origin()
            }
            launch_mla_sm100_decode_enqueue_kernel[
                q_type=q_type,
                KVLUTType=k_t,
                output_type=output_type,
                SplitAccumType=SplitAccumType,
                MaskType=mask_t,
                config=mla_config,
                ValidLengthType=ValidLengthType,
                ragged=True,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q_tma_op,
                k_tma_op,
                o_tma_op,
                k,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_len,
                mask,
                scales_ptr,
                scalar_args_buf,
                ctx,
            )
        else:
            comptime ValidLengthType = NullPointer[.uint32]
            var valid_len: ValidLengthType = {}
            launch_mla_sm100_decode_enqueue_kernel[
                q_type=q_type,
                KVLUTType=k_t,
                output_type=output_type,
                SplitAccumType=SplitAccumType,
                MaskType=mask_t,
                config=mla_config,
                ValidLengthType=ValidLengthType,
                ragged=False,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q_tma_op,
                k_tma_op,
                o_tma_op,
                k,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_len,
                mask,
                scales_ptr,
                scalar_args_buf,
                ctx,
            )


@always_inline
def launch_mla_sm100_decode_enqueue_kernel[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config.BM,  # tile_m =64
        BK=config.input_q_depth,
        swizzle_mode=config.swizzle_mode,
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config.kv_tma_swizzle_mode,
        BN=config.BK_PV,  # tile_m =64
        BK=config.input_q_depth,
    ],
    o_tma: ORaggedTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)

    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    # bf16: 3 warp groups; fp8: 4 warp groups (adds fp8-to-bf16 convert WG)
    # - one for load/store/2xMMA
    # - one for compute softmax
    # - one for compute correction
    # - (fp8 only) one for fp8-to-bf16 conversion
    var block_dim = (config.num_threads, 1, 1)
    logger.info(
        "block_dim:",
        block_dim[0],
        block_dim[1],
        block_dim[2],
        "grid_dim:",
        grid_dim[0],
        grid_dim[1],
        grid_dim[2],
        "config.smem_used:",
        config.smem_used,
        "config.num_q_heads:",
        config.num_q_heads,
        "config.num_kv_heads:",
        config.num_kv_heads,
        "config.num_threads:",
        config.num_threads,
        "config.num_kv_stages:",
        config.num_kv_stages,
        "config.BM:",
        config.BM,
        "config.BN_QK:",
        config.BN_QK,
        "config.BK_QK:",
        config.BK_QK,
        "config.BK_PV:",
        config.BK_PV,
        "config.q_depth:",
        config.q_depth,
        "config.depth:",
        config.depth,
        "config.padded_depth:",
        config.padded_depth,
        "config.padded_q_depth:",
        config.padded_q_depth,
        "config.rope_depth:",
        config.rope_depth,
        "config.swizzle_mode:",
        config.swizzle_mode,
        "output_tile_width:",
        (config.BN_PV // 8) * (4 // size_of[output_type]()),
    )

    logger.info("------ Dispatching to SM100 MLA-SM100-DECODE ------")
    logger.info(
        "QK Type:",
        KVLUTType.dtype,
        "Q Depth:",
        config.q_depth,
        "Number of Q // KV Heads:",
        config.num_q_heads,
        "//",
        config.num_kv_heads,
        "Batch Size:",
        block_z,
        "Num Partitions:",
        num_partitions,
    )

    # Dispatch to BF16 or old FP8 converter kernel (not native FP8 — that has
    # its own launch function with FP8 Q TMA).
    # Route ALL FP8 KV (both tensorwise and blockwise) to the FP8 converter
    # kernel. When we reach this function, native FP8 has already been ruled
    # out (Q is BF16), so the converter kernel handles FP8->BF16 conversion.
    comptime _is_old_fp8 = KVLUTType.dtype == DType.float8_e4m3fn
    comptime kernel = MLA_SM100_Decode_KV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        Engine=scalar_args_buf.Engine,
    ].kernel if _is_old_fp8 else MLA_SM100_Decode_KV_BF16[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        Engine=scalar_args_buf.Engine,
    ].kernel
    # Enable PDL (Programmatic Dependent Launch) for split-K mode to chain
    # the MLA decode kernel with the combine kernel, reducing host synchronization.
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF

    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_native_fp8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    # when True, the kernel packs
    # q_len_fold q_tokens into the BM=64 M tile and grid.y collapses to 1.
    fold_q: Bool = False,
    # comptime number of q_tokens to fold.  Must satisfy
    # `num_q_heads * q_len_fold <= BM` (caller-enforced) and `q_len_fold > 1`.
    q_len_fold: Int = 1,
](
    q_tma: QOTMATile[
        dtype=KVLUTType.dtype,  # FP8 Q TMA
        BM=config.BM,
        BK=config.input_q_depth,
        swizzle_mode=config.kv_tma_swizzle_mode,  # SWIZZLE_64B
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config.kv_tma_swizzle_mode,
        BN=config.BK_PV,
        BK=config.input_q_depth,
    ],
    o_tma: ORaggedTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Launch the native FP8 MLA decode kernel with FP8 Q TMA.

    This is a dedicated launch function for the native FP8 path because
    the Q TMA has FP8 dtype (SWIZZLE_64B) instead of BF16 (SWIZZLE_128B).

    Under `fold_q=True`, BM=64 packs `q_len_fold * num_q_heads` M-rows of
    the same batch element and grid.y collapses to 1 (all q_tokens live in
    one CTA).  This avoids spawning `q_max_seq_len` CTAs and lets the
    softmax/correction WGs amortize the QK setup.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    # fold collapses grid.y to 1 since BM packs all q_tokens.
    var grid_y = 1 if fold_q else q_max_seq_len
    var grid_dim = (block_x, grid_y, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info("------ Dispatching to SM100 Native FP8 MLA-DECODE ------")

    comptime kernel = MLA_SM100_Decode_QKV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        fold_q=fold_q,
        q_len_fold=q_len_fold,
        Engine=scalar_args_buf.Engine,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


# Layout G launcher for the qkv_fp8 native-FP8 kernel (BM=32, MMA_M=32,
# 5-stage). Takes two configs because k_tma / o_tma are identical between
# Layout E and Layout G — `config_e` types those, while `config_g`
# (`decode_layout_g=True`) types the kernel struct and BM=32 Q TMA tile.
@always_inline
def launch_mla_sm100_decode_native_fp8_layout_g[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config_e: MLA_SM100_Decode_Config,
    config_g: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    fold_q: Bool = False,
    q_len_fold: Int = 1,
](
    q_tma: QOTMATile[
        dtype=KVLUTType.dtype,
        BM=config_g.BM,
        BK=config_g.input_q_depth,
        swizzle_mode=config_g.kv_tma_swizzle_mode,  # SWIZZLE_64B
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config_e.kv_tma_swizzle_mode,
        BN=config_e.BK_PV,
        BK=config_e.input_q_depth,
    ],
    o_tma: ORaggedTMATile[
        dtype=output_type,
        BM=config_e.out_rows,
        BK=config_e.BN_PV // 4,
        swizzle_mode=config_e.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Launch the Layout G native FP8 MLA decode kernel (BM=32, 5-stage)."""
    comptime assert config_g.decode_layout_g, (
        "launch_mla_sm100_decode_native_fp8_layout_g requires"
        " config_g.decode_layout_g==True"
    )
    # Layout E and Layout G must agree on BN_QK/BK_QK/BK_PV/swizzle so the shared
    # k_tma / o_tma tiles flow through unchanged.
    comptime assert (
        config_e.BN_QK == config_g.BN_QK
    ), "Layout E/G BN_QK mismatch (k_tma reuse invariant)"
    comptime assert (
        config_e.BK_QK == config_g.BK_QK
    ), "Layout E/G BK_QK mismatch (k_tma reuse invariant)"
    comptime assert (
        config_e.BK_PV == config_g.BK_PV
    ), "Layout E/G BK_PV mismatch (k_tma reuse invariant)"
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config_g.num_q_heads, config_g.BM)
    # fold collapses grid.y to 1 since BM packs all q_tokens.
    var grid_y = 1 if fold_q else q_max_seq_len
    var grid_dim = (block_x, grid_y, block_z)
    var block_dim = (config_g.num_threads, 1, 1)

    logger.info(
        "------ Dispatching to SM100 Native FP8 MLA-DECODE Layout G ------"
    )

    comptime kernel = MLA_SM100_Decode_QKV_FP8_Layout_G[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config_g,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        fold_q=fold_q,
        q_len_fold=q_len_fold,
        Engine=scalar_args_buf.Engine,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config_g.decoding_warp_split_k else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config_g.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config_g.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_fp8_per_token_scale_rope_aware[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_per_token_scales: Bool = False,
](
    q_nope_tma: QOTMATile[
        dtype=DType.float8_e4m3fn,
        BM=config.BM,
        BK=config.padded_depth,  # 512
        swizzle_mode=config.content_swizzle_mode,  # SWIZZLE_64B
    ],
    q_rope_tma: QOTMATile[
        dtype=DType.bfloat16,
        BM=config.BM,
        BK=config.rope_depth,  # 64
        swizzle_mode=config.rope_swizzle_mode,  # SWIZZLE_128B
    ],
    k_content_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config.content_swizzle_mode,  # SWIZZLE_64B
        BN=config.BK_PV,  # 64
        BK=config.padded_depth,  # 512
    ],
    k_rope_tma: KVTMATile[
        dtype=DType.bfloat16,
        swizzle_mode=config.rope_swizzle_mode,  # SWIZZLE_128B
        BN=config.BK_PV,  # 64
        BK=config.rope_depth,  # 64
    ],
    scale_tma: ScalesTMATile[BN_QK=config.BN_QK],
    o_tma: ORaggedTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    scalar_args_buf: TileTensor[DType.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Launch the FP8 per-token-scale rope-aware MLA decode kernel with split content/rope TMAs.

    This is a dedicated launch function for the SnapMLA FP8 per-token-scale rope-aware path.
    Q and K are split into FP8 content (512 dims, SWIZZLE_64B) and BF16 rope
    (64 dims, SWIZZLE_128B), requiring 5 TMA descriptors (content, rope, scales, Q_nope, Q_rope).
    Per-token scales are loaded via TMA alongside content and rope.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info(
        "------ Dispatching to SM100 FP8 PerTensor RopeAware MLA-DECODE ------"
    )

    comptime kernel = MLA_SM100_Decode_QKV_FP8_PerTokenScale_RopeAware[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        has_per_token_scales=has_per_token_scales,
        Engine=scalar_args_buf.Engine,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        q_nope_tma,
        q_rope_tma,
        k_content_tma,
        k_rope_tma,
        scale_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        q_scale_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_sparse[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config.BM,
        BK=config.input_q_depth,
        swizzle_mode=config.swizzle_mode,
    ],
    # K_nope gather4 TMA: INT64, SWIZZLE_NONE (linear SMEM layout).
    # tile_width = padded_depth / 8 = 64 INT64 elements (nope only).
    k_nope_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.padded_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.padded_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    # K_rope gather4 TMA: BF16, SWIZZLE_128B.
    k_rope_tma: TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.bfloat16,
                (config.padded_depth + config.rope_depth * 2) // 2,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.bfloat16,
                (config.padded_depth + config.rope_depth * 2) // 2,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
    ],
    o_tma: ORaggedTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    # Extra KV parameters (separate always-attend cache).
    extra_k_nope_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.padded_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.padded_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    extra_k_rope_tma: TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.bfloat16,
                (config.padded_depth + config.rope_depth * 2) // 2,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.bfloat16,
                (config.padded_depth + config.rope_depth * 2) // 2,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
    ],
    extra_kv_lut: KVLUTType,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_indices_stride: Int,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Launch the sparse MLA decode kernel with gather4 TMA descriptors.

    d_indices stores encoded values: physical_block * page_size + offset.
    The kernel uses kv_lut.get_tma_row() to convert each encoded index
    into a physical TMA row.

    topk_lengths: per-batch array of actual topk counts. When non-null,
    topk_lengths[batch_idx] gives the number of valid entries for that
    batch. indices_stride is the stride between batches in d_indices
    (i.e., the max topk / allocation size).

    attn_sink_ptr: per-head correction values [num_heads_q], float32.
    When non-null, adjusts softmax denominator to account for
    non-selected tokens in sparse attention.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info("------ Dispatching to SM100 Sparse MLA-DECODE ------")

    comptime kernel = MLA_SM100_Decode_Sparse[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        has_attn_sink=has_attn_sink,
        has_extra_kv=has_extra_kv,
        has_variable_topk=has_variable_topk,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    # Sparse kernel needs extra SMEM beyond config.smem_used:
    # - 4 idx_bars barriers (4 * 8 = 32 bytes)
    # - ptr_tmem_addr (4 bytes, UInt32)
    # - idx_smem double-buffered (2 * BN_QK * sizeof(Int32) = 512 bytes)
    # Total extra: 548 bytes.
    comptime sparse_extra_smem = 4 * config.mbar_size + 4 + 2 * config.BN_QK * 4
    comptime sparse_smem_used = config.smem_used + sparse_extra_smem
    ctx.enqueue_function[kernel](
        q_tma,
        k_nope_tma,
        k_rope_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        d_indices,
        Int32(indices_stride),
        topk_lengths,
        scales_ptr,
        attn_sink_ptr,
        extra_k_nope_tma,
        extra_k_rope_tma,
        extra_kv_lut,
        extra_d_indices,
        extra_topk_lengths,
        Int32(extra_indices_stride),
        extra_scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=sparse_smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(sparse_smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_sparse_kv_fp8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
    # Read-once shared-index fold (KERN-3141): pack q_len_fold * num_q_heads
    # rows into the BM tile so grid.y collapses to 1 and the ONE shared topk
    # list is gathered once. Default False -> the unfolded baseline launch.
    fold_shared_index: Bool = False,
    q_len_fold: Int = 1,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config.BM,
        BK=config.input_q_depth,
        swizzle_mode=config.swizzle_mode,
    ],
    # Single K gather4 TMA: INT64, SWIZZLE_NONE, tile_width=72 INT64 (576 B).
    k_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.input_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.input_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    o_tma: ORaggedTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    # Extra KV parameters (separate always-attend cache).
    extra_k_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.input_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.input_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    extra_kv_lut: KVLUTType,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_indices_stride: Int,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Launches the all-FP8 sparse MLA decode kernel.

    This sibling of `launch_mla_sm100_decode_sparse` uses a single 576-byte
    gather4 TMA covering the full nope+rope row as FP8 (vs the BF16-rope
    parent which uses two separate descriptors).
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    # Shared-index fold packs all q positions into one BM tile -> grid.y = 1.
    var grid_y = 1 if fold_shared_index else q_max_seq_len
    var grid_dim = (block_x, grid_y, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info(
        "------ Dispatching to SM100 Sparse MLA-DECODE (all-FP8 KV) ------"
    )

    comptime kernel = MLA_SM100_Decode_Sparse_KV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        has_attn_sink=has_attn_sink,
        has_extra_kv=has_extra_kv,
        has_variable_topk=has_variable_topk,
        fold_shared_index=fold_shared_index,
        q_len_fold=q_len_fold,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    # Extra SMEM beyond the BF16-rope sparse kernel's budget:
    # - 4 idx_bars barriers (4 * mbar_size bytes)
    # - per-block cvt→QK handoff bars: 9 blocks x num_kv_stages (144 bytes)
    # - ptr_tmem_addr (4 bytes, UInt32)
    # - idx_smem double-buffered (2 * BN_QK * sizeof(Int32) = 512 bytes)
    comptime cvt_blk_bars_smem = (
        config.padded_q_depth // config.BN_QK
    ) * config.num_kv_stages * config.mbar_size
    comptime sparse_extra_smem = (
        4 * config.mbar_size + cvt_blk_bars_smem + 4 + 2 * config.BN_QK * 4
    )
    comptime sparse_smem_used = config.smem_used + sparse_extra_smem

    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        d_indices,
        Int32(indices_stride),
        topk_lengths,
        scales_ptr,
        attn_sink_ptr,
        extra_k_tma,
        extra_kv_lut,
        extra_d_indices,
        extra_topk_lengths,
        Int32(extra_indices_stride),
        extra_scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=sparse_smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(sparse_smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_sparse_kv_bf16[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config.BM,
        BK=config.input_q_depth,
        swizzle_mode=config.swizzle_mode,
    ],
    # Single BF16 gather4 TMA: SWIZZLE_128B, BN_QK rows, tile_width =
    # `input_q_depth` (the row as stored). The box width is swizzle-derived
    # and so is the same at either row width. Only the column-group count moves.
    k_tma: TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.bfloat16,
                config.input_q_depth,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.bfloat16,
                config.input_q_depth,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
    ],
    o_tma: ORaggedTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    # Extra KV TMA (separate always-attend cache): BF16, SWIZZLE_128B,
    # same descriptor shape as the main K TMA.
    extra_k_tma: TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.bfloat16,
                config.input_q_depth,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.bfloat16,
                config.input_q_depth,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
    ],
    extra_kv_lut: KVLUTType,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_indices_stride: Int,
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Launches the all-BF16 sparse MLA decode kernel.

    The K TMA is a single BF16 + SWIZZLE_128B gather4 descriptor covering
    the full `padded_q_depth` (576) row.  No `scales_ptr` /
    `extra_scales_ptr` because BF16 KV requires no FP8 dequantization.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    comptime KernelStruct = MLA_SM100_Decode_Sparse_KV_BF16[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        has_attn_sink=has_attn_sink,
        has_extra_kv=has_extra_kv,
        has_variable_topk=has_variable_topk,
    ]
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    # config's num_threads is the shared 3-WG default; the 4-WG sparse
    # kernel supplies its own block size.
    var block_dim = (KernelStruct.num_threads, 1, 1)

    logger.info(
        "------ Dispatching to SM100 Sparse MLA-DECODE (all-BF16 KV) ------"
    )

    comptime kernel = KernelStruct.kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    # Extra SMEM beyond the dense config:
    #   - ptr_tmem_addr (4 bytes, UInt32)
    #   - idx_smem double-buffered (2 * BN_QK * sizeof(Int32) = 512 bytes)
    comptime sparse_extra_smem = 4 + 2 * config.BN_QK * 4
    comptime sparse_smem_used = config.smem_used + sparse_extra_smem

    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        d_indices,
        Int32(indices_stride),
        topk_lengths,
        attn_sink_ptr,
        extra_k_tma,
        extra_kv_lut,
        extra_d_indices,
        extra_topk_lengths,
        Int32(extra_indices_stride),
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=sparse_smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(sparse_smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_sparse_qkv_fp8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
    fold_shared_index: Bool = False,
    q_len_fold: Int = 1,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config.BM,
        BK=config.input_q_depth,
        swizzle_mode=config.kv_tma_swizzle_mode,
    ],
    # Single K gather4 TMA covering the full 576-byte row: INT64,
    # SWIZZLE_NONE, tile_width=72 INT64 (matches the old FP8-KV kernel's
    # efficient contiguous gather -- 16 gather4 instructions/tile instead
    # of the SW64-fragmented 144/tile). The re-swizzle warpgroup reproduces
    # the SW64 layout the native FP8 MMA operand expects from this in SMEM.
    k_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.input_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.input_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    o_tma: ORaggedTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    extra_k_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.input_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.input_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    extra_kv_lut: KVLUTType,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_indices_stride: Int,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    """Launches the native FP8 sparse MLA decode kernel (3 warpgroups).

    Sibling of `launch_mla_sm100_decode_sparse_kv_fp8` but without the
    FP8→BF16 convert warpgroup. Q arrives as FP8 (SWIZZLE_64B TMA). Both
    QK and PV MMAs run natively in FP8 (tcgen05.mma.kind::f8f6f4), going
    from 4 warpgroups back to 3 and improving occupancy.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_y = 1 if fold_shared_index else q_max_seq_len
    var grid_dim = (block_x, grid_y, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info(
        "------ Dispatching to SM100 Sparse MLA-DECODE (native FP8 Q+KV) ------"
    )

    comptime kernel = MLA_SM100_Decode_Sparse_QKV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        has_attn_sink=has_attn_sink,
        has_extra_kv=has_extra_kv,
        has_variable_topk=has_variable_topk,
        fold_shared_index=fold_shared_index,
        q_len_fold=q_len_fold,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    # Extra SMEM beyond config.smem_used (unified-gather native FP8, no
    # cvt_blk_bars -- the second KV buffer + its pipeline are already
    # folded into config.smem_used via native_fp8_unified_gather):
    #   - idx_bars: 2*N barriers (depth matches config.num_kv_stages, N)
    #   - ptr_tmem_addr (4 bytes, UInt32)
    #   - idx_smem, N-deep (N * BN_QK * sizeof(Int32))
    comptime sparse_extra_smem = (
        2 * config.num_kv_stages * config.mbar_size
        + 4
        + config.num_kv_stages * config.BN_QK * 4
    )
    comptime sparse_smem_used = config.smem_used + sparse_extra_smem

    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        d_indices,
        Int32(indices_stride),
        logical_indices,
        topk_lengths,
        scales_ptr,
        attn_sink_ptr,
        extra_k_tma,
        extra_kv_lut,
        extra_d_indices,
        extra_topk_lengths,
        Int32(extra_indices_stride),
        extra_scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=sparse_smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(sparse_smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )
