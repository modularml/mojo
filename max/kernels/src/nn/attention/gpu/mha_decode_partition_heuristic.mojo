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

"""Split-K partition-count heuristics for MHA and MLA decode kernels.

Exposes per-backend functions that decide how many partitions the decode
split-K loop should run across, plus num_keys-independent upper bounds used
to launch graph-stable decode grids whose extra partitions early-return in
the kernel. CUDA uses a simple SM-fill target clamped to the single-warp
reducer limit; HIP (MI355X) uses a wave-aligned formula tuned for both MHA
and MLA decode shapes.
"""

from std.bit import next_power_of_two
from max.gpu.host import DeviceAttribute, DeviceContext
from std.math import ceildiv, clamp


@always_inline
def _bucket_partitions(n: Int) -> Int:
    """Bucket a partition count up to a fixed grid-shape ladder.

    Power-of-two bucketing leaves a 2x gap between 64 and 128, meaningful
    for h=64 / h=128 shapes where the target landing point is often in the
    65..127 range and 128 is wasteful. We insert mid-points {48, 96} to
    halve the worst-case over-partitioning above 32 while keeping the
    bucket count small enough for HIP graph capture.

    Ladder: 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256.

    The SM100 path has an analogous helper, `_bucket_num_partitions` in
    `nvidia/sm100/mla_decode_dispatch.mojo`, with a different ladder
    (top = `half_sms`, parameterized on GPU SM count), driven by SM100's
    wave-fill target instead of an AMD-reducer hard cap. They aren't
    shared because the constraints differ; the pattern is the same.
    """
    if n <= 32:
        return next_power_of_two(n)
    if n <= 48:
        return 48
    if n <= 64:
        return 64
    if n <= 96:
        return 96
    if n <= 128:
        return 128
    if n <= 192:
        return 192
    return 256


def cuda_mha_decoding_max_num_partitions(
    batch_size: Int,
    heads_per_group: Int,
    sm_count: Int,
) -> Int:
    """Returns the num_keys-independent upper bound on CUDA decode split-K partitions.

    Targets one partition per idle SM, clamped to [1, 32]: the 32 ceiling is
    the MHA split-K reducer's single-warp WARP_SIZE limit and the lower bound
    guards the large-batch case where `batch_size * heads_per_group > sm_count`
    would otherwise drive the SM term to zero. Not rounded to a power of two
    so an exact target avoids over-partitioning.

    Args:
        batch_size: Number of decode requests in the batch.
        heads_per_group: Query heads sharing each kv-head group (kv_num_heads for MHA).
        sm_count: Device multiprocessor count used to size the SM-fill target.

    Returns:
        Partition count clamped to [1, 32], used as the upper bound for every num_keys.
    """
    # num_keys-independent partition target: fill one partition per idle SM,
    # clamped to [1, 32]. The 32 ceiling is the MHA split-K reducer's
    # single-warp WARP_SIZE limit; the lower bound of 1 guards the case where
    # batch_size * heads_per_group > sm_count drives the SM term to 0
    # (large-batch decode), which would otherwise return 0 partitions and
    # divide by zero downstream in get_start_and_end_for_partitions. Not rounded
    # to a power of two: the reducer handles any count in [1, 32], so an exact
    # target avoids over-partitioning (e.g. 17 -> 32). The actual count
    # (cuda_mha_decoding_num_partitions) only further mins this with
    # num_keys // 512, so this is the upper bound for every num_keys -- used to
    # launch a num_keys-independent (graph-stable) decode grid whose extra
    # partitions early-return in the kernel.
    return clamp(sm_count // (batch_size * heads_per_group), 1, 32)


def cuda_mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    sm_count: Int,
) -> Int:
    """Returns the CUDA decode split-K partition count for the given shape.

    Takes the num_keys-independent upper bound from
    `cuda_mha_decoding_max_num_partitions` and further limits it so each
    partition spans at least 512 keys. The `max(1, ...)` floor preserves the
    >= 1 guard when `num_keys < 512`.

    Args:
        batch_size: Number of decode requests in the batch.
        num_keys: Number of key cache entries to scan.
        heads_per_group: Query heads sharing each kv-head group (kv_num_heads for MHA).
        sm_count: Device multiprocessor count used to size the SM-fill target.

    Returns:
        Partition count in [1, 32] further limited by `num_keys // 512`.
    """
    # The num_keys-independent upper bound, further limited so each partition
    # spans at least 512 keys. Deriving from the max keeps the SM-fill target
    # and the [1, 32] clamp in one place, so max >= actual holds by
    # construction. The max(1, ...) floor preserves the >= 1 guard when
    # num_keys < 512 (a 0 here divides by zero downstream).
    return min(
        cuda_mha_decoding_max_num_partitions(
            batch_size, heads_per_group, sm_count
        ),
        max(1, num_keys // 512),
    )


def hip_mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    sm_count: Int,
    is_mla: Bool = False,
) -> Int:
    """Wave-aligned split-K target for MI355X MHA + MLA decode.

    Two regimes, distinguished by whether the kernel packs queries
    by BM (MLA) or spawns one CTA per kv-head (MHA):

    - **MHA-style** (`heads_per_group < BM`): the call comes from
      `get_mha_decoding_num_partitions` passing
      `heads_per_group = num_heads // group = kv_num_heads`, which is
      typically small (≤ 8). Each (kv_head, batch) is its own CTA in
      grid_y, so `actual_ctas_per_partition = heads_per_group ×
      batch_size`. When `work_items = heads_per_group × batch_size ≥
      sm_count`, one partition already fills the GPU; use few
      partitions, just enough to amortize key reads. Derived from the
      original heuristic's HIGH_OCC branch, with two changes: its cap
      narrows to 32 at exactly 16 kv heads from batch 8 up, and the
      16-kv-head key-stream rule below pre-empts the branch entirely
      once a full unsplit wave and 8 pages are reached.

    - **MLA-style** (`heads_per_group ≥ BM`): the call comes from
      `mla.mojo` passing `heads_per_group = num_heads` (≥ BM=32 for
      h ∈ {32, 64, 128}). MLA packs BM queries into one CTA, so
      `actual_ctas_per_partition = ceildiv(num_heads, BM) ×
      batch_size`. Even when `work_items` looks large (e.g. bs=8
      h=64 → 512), actual CTAs are only 16; needs many partitions.
      Apply the 2-wave wave-aligned formula:
          one_wave    = sm_count // ctas_per_partition
          two_wave    = 2 × sm_count // ctas_per_partition
          work_floor  = ceildiv(pages, MAX_PAGES_PER_SPLIT)
          np_target   = clamp(work_floor, one_wave, two_wave)

      EXCEPTION (MLA num_heads <= 16, e.g. Kimi-K2.5 TP=4, and single-kv-head
      MHA): the one_wave floor underfills. With num_blocks_y=1,
      ctas_per_partition = batch_size, so one wave (np = sm/bs) gives each CU
      exactly one CTA: no second block to overlap HBM-read latency. These
      shapes are latency-bound, so target two full waves instead:
          np_target   = min(two_wave, pages)
      Measured on MI355: two-wave np is 5-10% faster than one-wave across
      bs=4 (32K-128K) and bs=8/16 short context; past two waves regresses on
      reduce cost. MLA bs=1 is unchanged (two_wave=512 clamps to the 256 cap);
      single-kv-head MHA does move, since its finer split floor doubles pages.

    Phase 0 sweep (PARTITIONING_PLAN.md) validated MLA-style at h=64:
        bs=1  ctx=131K → np=128 (capped, fills GPU at 1-wave + cap)
        bs=2  ctx=65K  → np=64  (one_wave=64 dominates)
        bs=2  ctx=80K  → np=64  (one_wave=64 dominates; work_floor 64 capped)
        bs=2  ctx=131K → np=128 (work_floor=103 → bucket to 128)
        bs=8  ctx=80K  → np=32  (work_floor 64 capped by two_wave=32)
        bs=8  ctx=131K → np=32  (work_floor 103 capped by two_wave=32)
        bs=16 ctx=131K → np=16  (work_floor 103 capped by two_wave=16)

    AMD reducer constraint: `mla_splitk_reduce` supports MAX_PARTITIONS
    up to `parts_per_lane × WARP_SIZE`; the 256-partition specialization
    (parts_per_lane=4) lifts the MLA-style cap to 256. Only nk >= 64K
    (pages >= 256) actually reaches np=256; smaller nk is page-limited.

    Tunables (MLA-style):
        BM                   = 32   (MLA decode block-M on MI355)
        SPLIT_PAGE_SIZE      = 256  (min keys per partition; 128 for
                                     single-kv-head MHA)
        MAX_PAGES_PER_SPLIT  = 5    (= 1280 keys per partition cap)
        MAX_HIP_PARTITIONS   = 256  (reducer's MAX_PARTITIONS limit; the
                                     MHA-style branch above stays pinned ≤64)

    Args:
        batch_size: Number of decode requests in the batch.
        num_keys: Number of key cache entries to scan.
        heads_per_group: Not a group size: the kv-head count for MHA
            (`num_heads // group`), `num_heads` for MLA.
        sm_count: Device multiprocessor count used to size the wave-fill target.
        is_mla: Whether the caller is the MLA decode path (defaults to `False`).
    """
    comptime BM = 32
    comptime SPLIT_PAGE_SIZE = 256
    comptime MAX_PAGES_PER_SPLIT = 5
    comptime MAX_HIP_PARTITIONS = 256
    # Empirically-tuned divisor used in the MHA HIGH_OCC branch to scale
    # partitions inversely with work_items. NOT WARP_SIZE — it's a
    # workload-shaping constant inherited from the pre-Phase-1 heuristic
    # (happens to equal 64 on gfx950 by coincidence).
    comptime MHA_OCC_SCALE_DIVISOR = 64

    # No partitioning for very short caches — split-K overhead exceeds win.
    if num_keys <= SPLIT_PAGE_SIZE:
        return 1

    # On the MHA side `heads_per_group` is the kv-head count (`num_heads //
    # group`); on the MLA side it is `num_heads`.
    var kv_num_heads = heads_per_group

    # At exactly 16 kv heads the optimum is a fixed key-stream length per
    # partition rather than a wave-fill target, because 16 x batch_size CTAs
    # already fill a wave before any split. So take one partition per split
    # page — worth 4-12% on the EAGLE3 draft shape (depth 128, bf16) at batch
    # 16-128 across 2048-32768 keys, neutral beyond — under three measured
    # bounds: past 16 partitions the reduce and
    # launch cost of the 16 x batch_size x partitions grid outgrows the shorter
    # key stream (a reducer bound, so it stays a literal); below one unsplit
    # wave, partitions are the only way to fill the device (batch 4 loses 12% at
    # 32K keys), gated on `sm_count` so it tracks the device rather than MI355's
    # 256 CUs; and below 8 pages splitting at all costs 8-12%. The ladder rounds
    # up, so 9 pages launches 16 partitions with 7 key-empty -- safe, an empty
    # partition resets its own softmax stats before storing them
    # (`mha_decode.mojo`).
    var coarse_pages = ceildiv(num_keys, SPLIT_PAGE_SIZE)
    if (
        (not is_mla)
        and kv_num_heads == 16
        and kv_num_heads * batch_size >= sm_count
        and coarse_pages >= 8
    ):
        return _bucket_partitions(min(coarse_pages, 16))

    var work_items = heads_per_group * batch_size

    # MHA-style cheap-reducer cap; never above 64 so the MLA-only
    # MAX_HIP_PARTITIONS bump (128->256) does not change MHA grids.
    #
    # 32 at exactly 16 kv heads from batch 8 up: partitions past 32 only
    # shorten an already-short key stream while adding reduce work. Only the
    # band the rule above leaves behind reaches here -- batch 8 to 15, or
    # fewer than 8 pages.
    #
    # 64 elsewhere: at 8 or fewer kv heads 48 and 64 are still the optimum, and
    # below batch 8 the optimum flips with `num_keys` in opposite directions for
    # 16 and 32 kv heads, so one bound cannot model it. 32 kv heads (plain MHA,
    # no GQA) wants 16-32 at 64K keys too, but is the noisiest shape measured
    # here (up to 15.9% between identical batches) and unmeasured past 64K.
    # TODO: tighten 32 kv heads once a stable protocol can resolve it.
    var mha_partition_cap = 32 if (
        kv_num_heads == 16 and batch_size >= 8
    ) else 64

    # MHA-style: kv_num_heads spawns CTAs directly in grid_y.
    if (not is_mla) and heads_per_group < BM:
        if work_items >= sm_count:
            # High occupancy: 1 partition already fills the GPU. Scale
            # partition size up as work_items grows so we don't
            # over-partition (more concurrent CTAs → fewer per-CTA pages
            # is fine). The divisor is empirical, not WARP_SIZE.
            var occupancy_scale = max(1, work_items // MHA_OCC_SCALE_DIVISOR)
            var np_mha = min(
                ceildiv(num_keys, SPLIT_PAGE_SIZE * occupancy_scale),
                mha_partition_cap,
            )
            return min(_bucket_partitions(np_mha), MAX_HIP_PARTITIONS)
        # Low occupancy MHA: rare. Fall through to MLA-style formula
        # since it handles the wave-fill case correctly with
        # ctas_per_partition = work_items (BM packing is a no-op when
        # heads_per_group < BM).

    # MLA-style: also every MHA shape with 32+ kv heads, at any occupancy.
    var ctas_per_partition = max(1, ceildiv(heads_per_group, BM) * batch_size)

    # Single-kv-head MHA takes a finer split floor: grid_y is 1, so partitions
    # are the only thing filling the GPU, and a 256-key floor allows just 8
    # partitions at 2048 keys where two waves want 512 // batch_size; halving is
    # worth 19-23% at 2048 keys and 10-20% at 4096 across batch 1-8. 64 keys
    # per partition measured slower again, so stop at 128. A finer floor stays
    # exact because the split-K range helper aligns each partition up to a BN
    # tile and empties the leftovers.
    var mha_single_kv_head = (not is_mla) and kv_num_heads == 1
    var split_floor = (
        SPLIT_PAGE_SIZE // 2 if mha_single_kv_head else SPLIT_PAGE_SIZE
    )
    var pages = ceildiv(num_keys, split_floor)
    var one_wave = max(1, sm_count // ctas_per_partition)
    var two_wave = max(1, (2 * sm_count) // ctas_per_partition)
    var work_floor = ceildiv(pages, MAX_PAGES_PER_SPLIT)

    var np_target: Int
    # Single-kv-head MHA underfills at `one_wave` for the same reason MLA at
    # `heads_per_group <= 16` does: `ctas_per_partition` is then exactly
    # `batch_size`. Measured separately for MHA (bs=32, 16 q-heads over 1
    # kv-head, depth 128): one_wave leaves 8-22% at nk 4096-8192, for plain
    # decode and the S=4 token fold alike. Held to one kv head because more kv
    # heads multiply grid_y, which `ctas_per_partition` does not account for.
    var one_wave_underfills = (
        heads_per_group <= 16 if is_mla else mha_single_kv_head
    )
    if one_wave_underfills:
        # num_heads <= 16 (Kimi-K2.5 TP=4) packs all heads into one block
        # (num_blocks_y=1), so ctas_per_partition = batch_size — tiny. Decode
        # is latency-bound: each CTA stalls on HBM K-reads, so fill TWO waves
        # — a second CTA per CU hides the first's stalls. Bounded by available
        # pages (cannot split below one page) and the 256-partition cap. The
        # one_wave floor used below underfills here: measured on MI355, the
        # two-wave np is 5-10% faster than one-wave across bs=4 (32K-128K) and
        # bs=8/16 short context, while going *past* two waves regresses (split-K
        # reduce cost). MLA bs=1 is unchanged: two_wave=512 clamps to the
        # 256-partition cap = one wave, the most CTAs it can reach. Single-kv-head
        # MHA is capped at 64 instead and does move, because its finer split
        # floor doubles `pages`.
        np_target = min(two_wave, pages)
    else:
        # MLA num_heads >= 32, MHA with 32+ kv heads (which skips the MHA-style
        # branch at any occupancy, since `heads_per_group < BM` is false), and
        # multi-kv-head low-occupancy MHA: keep the tuned
        # wave-aligned target — clamp work_floor to [one_wave, two_wave]
        # (one_wave floor, two_wave cap; validated for num_heads=64/128 in the
        # Phase-0/1 sweeps).
        np_target = clamp(work_floor, one_wave, two_wave)

    # The MHA split-K reducer runs in a single warp and only handles up to
    # WARP_SIZE partitions, hence the 64 ceiling on `mha_partition_cap`. MLA
    # uses a partition-aware reducer and keeps the full 256.
    var partition_cap = MAX_HIP_PARTITIONS if is_mla else mha_partition_cap
    var num_partitions = min(np_target, pages, partition_cap)

    # Bucket to a fixed ladder (1, 2, ..., 64, 96, 128, 192, 256) so
    # HIP graph capture sees a small number of decode grid shapes.
    return min(_bucket_partitions(num_partitions), partition_cap)


def mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    ctx: DeviceContext,
    is_mla: Bool = False,
) raises -> Int:
    """Dispatches to the backend-specific decode split-K partition heuristic.

    Queries the device multiprocessor count lazily and routes CUDA and HIP to
    their tuned heuristics; every other backend (Metal, accelerators with no
    split-K decode path) returns 1 so the decode kernel runs unsplit.

    Args:
        batch_size: Number of decode requests in the batch.
        num_keys: Number of key cache entries to scan.
        heads_per_group: Query heads sharing each kv-head group (kv_num_heads for MHA, num_heads for MLA).
        ctx: Device context used to query the backend API and SM count.
        is_mla: Whether the caller is the MLA decode path (HIP only).

    Returns:
        Number of split-K partitions for the decode kernel's split-K loop.
    """
    var api = ctx.api()
    if api == "hip" or api == "cuda":
        # MULTIPROCESSOR_COUNT is only meaningful for the split-K heuristics,
        # so query it lazily here rather than for every backend.
        var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        if api == "hip":
            return hip_mha_decoding_num_partitions(
                batch_size,
                num_keys,
                heads_per_group,
                sm_count,
                is_mla=is_mla,
            )
        return cuda_mha_decoding_num_partitions(
            batch_size,
            num_keys,
            heads_per_group,
            sm_count,
        )
    # CUDA and HIP have tuned split-K decode heuristics above. Every other
    # backend (Metal, plus accelerators with no split-K decode path) runs the
    # decode unsplit; a single partition is always valid — the decode kernel
    # reads this count only to bound its split-K loop.
    return 1


def mha_decoding_max_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    ctx: DeviceContext,
) raises -> Int:
    """Returns the num_keys-independent upper bound on `mha_decoding_num_partitions`.

    Used to launch a graph-stable decode grid. Only the CUDA decode path
    over-launches and early-returns the extra partitions; every other backend
    keeps `max == actual` so the `max >= actual` invariant holds and no
    over-launch path is taken.

    Args:
        batch_size: Number of decode requests in the batch.
        num_keys: Number of key cache entries to scan.
        heads_per_group: Query heads sharing each kv-head group (kv_num_heads for MHA).
        ctx: Device context used to query the backend API and SM count.

    Returns:
        Upper bound on the partition count for any num_keys on this backend.
    """
    # num_keys-independent upper bound on mha_decoding_num_partitions, used to
    # launch a graph-stable decode grid. Only the CUDA decode path over-launches
    # and early-returns the extra partitions; every other backend keeps
    # max == actual so the (max >= actual) invariant holds and no over-launch
    # path is taken.
    if ctx.api() == "cuda":
        var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        return cuda_mha_decoding_max_num_partitions(
            batch_size, heads_per_group, sm_count
        )
    return mha_decoding_num_partitions(
        batch_size, num_keys, heads_per_group, ctx
    )
