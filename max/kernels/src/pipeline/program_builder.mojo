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
"""Program builder: constructs PipelineProgram from a loop body.

Contains:
  - _construct_mma_blocks, _redistribute_globals
  - build_double_buffer_program
  - derive_waits_from_blocks, derive_safe_max_globals, derive_drain_mask
  - dump_program_blocks
  - _verify_* functions, verify_schedule
  - build_kernel_program
  - default_kernel_deps_double_buffer
  - single_buffer_reorder, optimize_within_barriers
  - mma_block_interleave, mma_block_interleave_list, double_buffer_reorder
"""

from std.collections import Array, List

from .config import PipelineConfig, ScheduleConfig, SchedulingStrategy
from .dependency_graph import LoopBody, OpNode
from .phase_derivation import derive_edges_from_ops
from .pipeline_dsl import Pipe, ScheduleEntry
from .program import MMABlockSpec, PipelineProgram
from .schedulers import (
    greedy_schedule,
    optimal_schedule,
    optimal_schedule_with_halves,
)
from .types import DepEdge, OpDesc, OpRole, _Ops, _is_prefetch


def _derive_main_wait_vm(
    body: LoopBody, order: List[Int], config: PipelineConfig
) -> Int:
    """Drain target for the top-of-half / first-cross-stage `entry_wait`.

    Walks the body's prefetch loads, sums their per-channel vm_cost,
    then returns `total - per_half`, i.e. drain to leave one half's
    worth of prefetches in flight. The semantic: when this wait fires,
    the previous iter's writes targeting *this* half have committed,
    while future iters' prefetches can stay outstanding.

    For 2-half ping-pong / 4-wave (8 prefetches × VMCNT, 4 per half):
    `(8 - 4) * VMCNT = 4 * VMCNT`, same value as the hardcoded
    `4 * config.vm_per_load_a` constant we previously used.
    Generalises correctly for kernels with non-uniform per-half
    prefetch counts and for `num_partitions != 2`.
    """
    var num_ops = len(body.ops)
    var total_vm = 0
    for i in range(num_ops):
        var op = body.ops[order[i]].op
        if op.role == OpRole.GLOBAL_LOAD and _is_prefetch(op):
            total_vm += config.vm_per_channel(op.channel)
    var halves = config.num_partitions
    if halves <= 0:
        return total_vm
    return total_vm - (total_vm // halves)


def _construct_mma_blocks(
    body: LoopBody,
    order: List[Int],
    config: PipelineConfig,
    sched: ScheduleConfig,
) -> PipelineProgram:
    """Group ordered ops into MMA-centered blocks.

    Walks the op list in execution order, collecting fragment loads and
    global loads until an MMA op delimits the block. Structural flags
    (wait counts, schedule barriers, drain masks) come from ScheduleConfig.
    """
    var num_ops = len(body.ops)
    var num_blocks = config.total_blocks()
    var bph = config.blocks_per_partition()
    var p = PipelineProgram(num_blocks, trailing_barrier=False)
    var block_idx = 0
    var op_idx = 0
    # Auto-derive the MAIN_WAIT drain target from the body's prefetch
    # distribution. Replaces the previous hardcoded `4 * vm_per_load_a`
    # constant — that "4" was a magic number matching hand-tuned 4-wave
    # but didn't generalise to other prefetch counts.
    var main_wait_vm = _derive_main_wait_vm(body, order, config)

    for _block in range(num_blocks):
        var pre_ops = List[OpDesc]()
        var global_loads = List[OpDesc]()
        var mma_op = OpDesc.none()

        while op_idx < num_ops:
            var op = body.ops[order[op_idx]].op
            if op.role == OpRole.COMPUTE:
                mma_op = op
                op_idx += 1
                break
            elif op.role == OpRole.FRAGMENT_LOAD:
                pre_ops.append(op)
            elif op.role == OpRole.GLOBAL_LOAD:
                global_loads.append(op)
            op_idx += 1

        var pos_in_partition = block_idx % bph
        var partition_idx = block_idx // bph
        var pre_op_0 = pre_ops[0] if len(pre_ops) >= 1 else OpDesc.none()
        var pre_op_1 = pre_ops[1] if len(pre_ops) >= 2 else OpDesc.none()
        var gl0 = global_loads[0] if len(global_loads) >= 1 else OpDesc.none()
        var gl1 = global_loads[1] if len(global_loads) >= 2 else OpDesc.none()

        var pre_sync = OpDesc.none()
        var post_barrier_lgkm = True
        var trailing_sched = (sched.sched_barrier_mask >> block_idx) & 1 == 1

        # Cross-stage frag detection: a fragment-load whose `stage` field
        # doesn't equal its block's half is reading from the cross SMEM
        # stage (the stage the *other* half consumes). Such reads need an
        # explicit vm drain so the cross stage's last writer (this iter's
        # other half, or the previous iter) has finished before this
        # block's LDS read. Without this drain the framework's per-half
        # `wait_vm_last` (only at the last block of the half) fires too
        # late and the cross-stage frag races against in-flight loads.
        var pre_op_0_cross = (
            pre_op_0.is_present()
            and pre_op_0.role == OpRole.FRAGMENT_LOAD
            and pre_op_0.stage != partition_idx
        )
        var pre_op_1_cross = (
            pre_op_1.is_present()
            and pre_op_1.role == OpRole.FRAGMENT_LOAD
            and pre_op_1.stage != partition_idx
        )
        var pre_op_is_cross = pre_op_0_cross or pre_op_1_cross

        # Only emit the mid-half drain at the FIRST cross-stage block in
        # each half — subsequent cross-stage blocks are already covered by
        # the same drain.
        var seen_cross_in_partition = False
        var partition_start = partition_idx * bph
        for prior_b in range(partition_start, block_idx):
            var prior = p.blocks[prior_b]
            if (
                prior.pre_op_0.is_present()
                and prior.pre_op_0.role == OpRole.FRAGMENT_LOAD
                and prior.pre_op_0.stage != partition_idx
            ):
                seen_cross_in_partition = True
                break
            if (
                prior.pre_op_1.is_present()
                and prior.pre_op_1.role == OpRole.FRAGMENT_LOAD
                and prior.pre_op_1.stage != partition_idx
            ):
                seen_cross_in_partition = True
                break
        var first_cross_in_partition = (
            pre_op_is_cross and not seen_cross_in_partition
        )

        # Entry wait: vm drain that fires BEFORE pre_op_0 (= before any
        # frag-load this block issues). Two situations need it:
        #
        #  - **Top-of-half** (pos_in_partition == 0) when this half contains
        #    any cross-stage frag: drain to "4 newest in flight" so the
        #    half's first same-stage frag-load sees committed LDS from
        #    the previous half's tail loads. Mirrors inline's TOP
        #    `vmcnt(MAIN_WAIT)`.
        #
        #  - **First cross-stage block in this half**: drain so the
        #    cross-stage frag-load reads valid LDS (the cross stage's
        #    last writer, in the *other* half or the previous iter,
        #    must have committed). Mirrors inline's MID
        #    `vmcnt(MAIN_WAIT)`.
        #
        # The `pre_sync` slot is after globals, used for end-of-half
        # drain — different position, different purpose, kept as-is.

        # For pos_in_partition == 0, look ahead in the body to detect whether
        # this half contains cross-stage frags in remaining blocks.
        var top_drain_needed = False
        if pos_in_partition == 0:
            top_drain_needed = pre_op_is_cross
            if not top_drain_needed:
                var scan_idx = op_idx
                var scanned_mmas = 0
                while scan_idx < num_ops and scanned_mmas < bph - 1:
                    var sop = body.ops[order[scan_idx]].op
                    if sop.role == OpRole.COMPUTE:
                        scanned_mmas += 1
                    elif (
                        sop.role == OpRole.FRAGMENT_LOAD
                        and sop.stage != partition_idx
                    ):
                        top_drain_needed = True
                        break
                    scan_idx += 1

        var entry_wait = OpDesc.none()
        var entry_wait_lgkm = OpDesc.none()
        if first_cross_in_partition or top_drain_needed:
            # Drain target derived from the body's prefetch distribution
            # (`_derive_main_wait_vm`): leave one half's worth of
            # prefetches in flight. For 4-wave (8 prefetches × VMCNT,
            # 4 per half) this works out to `4 * VMCNT`, matching the
            # hand-tuned `MAIN_WAIT` constant.
            entry_wait = OpDesc.wait_vm_n(main_wait_vm)

        if pos_in_partition == 0 and sched.wait_lgkm_first < 255:
            # When `entry_wait` is set on this same block, the lgkm
            # drain pairs with the vm drain in the entry section —
            # which fires BEFORE any of this block's `pre_op` /
            # `global_load` ops. At that point, the only outstanding
            # lgkm is from the previous iteration (whose ds_reads feed
            # this block's MMA inputs via the cross-stage rotation
            # registers). Drain to 0 so the MMA's input registers are
            # ready by the time the cross-warp `s_barrier` returns —
            # `sched.wait_lgkm_first` was computed for the original
            # post-globals `pre_sync` position where leaving "globals
            # in flight" was meaningful.
            if entry_wait.is_present() and sched.minimal_barriers:
                entry_wait_lgkm = OpDesc.wait_lgkm_n(0)
            else:
                pre_sync = OpDesc.wait_lgkm_n(sched.wait_lgkm_first)
        elif first_cross_in_partition:
            # MID barrier point: pair `entry_wait`'s vm drain with a
            # full lgkm drain so the cross-stage frag's pre-MMA register
            # state matches what inline's hand-written MID emits
            # (`s_waitcnt vmcnt + lgkmcnt(0) + s_barrier`).
            if entry_wait.is_present() and sched.minimal_barriers:
                entry_wait_lgkm = OpDesc.wait_lgkm_n(0)
            else:
                pre_sync = OpDesc.wait_lgkm_n(0)
        elif pos_in_partition == bph - 1:
            pre_sync = OpDesc.wait_vm_n(sched.wait_vm_last)
            post_barrier_lgkm = sched.lgkm_after_last
        elif sched.inter_block_lgkm_drain:
            # Interior block (not top, not first cross-stage, not last)
            # — inject `wait_lgkm(0)` at block entry so the previous
            # block's frag/load LDS ops drain before this block's
            # frag-load reads (avoids LDS port contention).
            entry_wait_lgkm = OpDesc.wait_lgkm_n(0)

        # Drain LDS reads before DRAM→LDS writes to avoid LDS port
        # contention. Only useful when a block has both fragment loads
        # (LDS reads) and global loads (DRAM→LDS writes).
        var has_frag = pre_op_0.is_present() or pre_op_1.is_present()
        var has_dram = gl0.is_present() or gl1.is_present()
        var drain_bit = (sched.drain_lgkm_mask >> block_idx) & 1 == 1
        var drain = drain_bit and has_frag and has_dram

        # Barrier strategy: when `sched.minimal_barriers` is set, the
        # cross-warp `s_barrier` lives in the entry section (paired
        # with the `entry_wait` vm drain at TOP-of-half and first-
        # cross-stage blocks — see `expand_to_list` / `emit_minimal_
        # barrier_block` for the emit shape). We do NOT also emit a
        # `pre_mma_barrier` at those blocks — that's a second redundant
        # `s_barrier` covering the same cross-warp invariant. Dropping
        # it cuts the per-iter barrier count from 8 to 4, matching the
        # hand-tuned inline body. For non-minimal kernels (ping-pong,
        # simple) keep the full pre-MMA + post-MMA layout.
        var is_top_of_half = pos_in_partition == 0
        var emit_pre_mma_barrier = True
        var emit_pre_mma_set_prio = True
        var emit_post_mma_barrier = True
        var emit_post_mma_set_prio = True
        var emit_post_barrier_lgkm = post_barrier_lgkm
        if sched.minimal_barriers:
            # No pre-MMA `s_barrier` under minimal_barriers — the entry
            # barrier (paired with `entry_wait`) handles cross-warp
            # visibility. The MMA's input registers are loaded by
            # earlier blocks' `pre_op_*` and drained by either this
            # block's `entry_wait_lgkm` (interior + TOP + CROSS) or the
            # final block's `pre_sync` wait — explicit lgkm waits are
            # always sufficient without an additional `s_barrier` here.
            emit_pre_mma_barrier = False
            # `set_prio[1]` is an LLVM scheduling barrier that blocks
            # register-allocator reuse across it (observed: dropping it
            # entirely cuts VGPR pressure 102 → 86). But it's also a
            # warp-scheduler priority hint that can be load-bearing for
            # MMA throughput in some pipelines (e.g. the original
            # 8-warp ping-pong). For schedules that opt into
            # `minimal_barriers` AND `omit_mma_set_prio` we drop it
            # entirely and rely on `rocdl.waves_per_eu=1` for priority.
            # Without `omit_mma_set_prio` we keep it at TOP + MID
            # boundaries (4 per iter) as the safer default.
            emit_pre_mma_set_prio = False if sched.omit_mma_set_prio else (
                is_top_of_half or first_cross_in_partition
            )
            emit_post_mma_barrier = False
            emit_post_mma_set_prio = False
            emit_post_barrier_lgkm = False

        p.blocks[block_idx] = MMABlockSpec(
            mma=mma_op,
            pre_op_0=pre_op_0,
            pre_op_1=pre_op_1,
            global_load=gl0,
            global_load_1=gl1,
            entry_wait=entry_wait,
            entry_wait_lgkm=entry_wait_lgkm,
            pre_sync=pre_sync,
            pre_mma_barrier=emit_pre_mma_barrier,
            pre_mma_set_prio=emit_pre_mma_set_prio,
            post_mma_barrier=emit_post_mma_barrier,
            post_mma_set_prio=emit_post_mma_set_prio,
            post_barrier_lgkm=emit_post_barrier_lgkm,
            trailing_sched_barrier=trailing_sched,
            global_load_prefetch=_is_prefetch(gl0),
            global_load_1_prefetch=_is_prefetch(gl1),
            drain_lgkm_before_loads=drain,
            global_before_frag=sched.global_before_frag,
            barrier_before_pre_ops=sched.barrier_before_pre_ops,
            wrap_waits_with_sched_barrier=(sched.wrap_waits_with_sched_barrier),
        )
        block_idx += 1

    debug_assert(block_idx == num_blocks, "expected num_blocks blocks")
    debug_assert(op_idx == num_ops, "expected all ops consumed")
    return p^


def _redistribute_globals(
    mut program: PipelineProgram,
    config: PipelineConfig,
    sched: ScheduleConfig,
):
    """Rebalance global loads across blocks for uniformity.

    When max_globals is set, moves excess globals from over-full blocks
    to under-full blocks so each block gets exactly max_globals loads
    (if enough are available). Recomputes drain flags for affected blocks.
    """
    var max_g = config.block_sizing.max_globals
    if max_g <= 0:
        return

    var num_blocks = config.total_blocks()

    # Collect excess globals from over-full blocks.
    var pool = List[OpDesc]()
    for i in range(num_blocks):
        var b = program.blocks[i]
        var count = (1 if b.global_load.is_present() else 0) + (
            1 if b.global_load_1.is_present() else 0
        )
        while count > max_g:
            if b.global_load_1.is_present():
                pool.append(b.global_load_1)
                b.global_load_1 = OpDesc.none()
                b.global_load_1_prefetch = False
            elif b.global_load.is_present():
                pool.append(b.global_load)
                b.global_load = OpDesc.none()
                b.global_load_prefetch = False
            count -= 1
        program.blocks[i] = b

    # Assign pool globals to under-full blocks.
    var pool_idx = 0
    for i in range(num_blocks):
        if pool_idx >= len(pool):
            break
        var b = program.blocks[i]
        var count = (1 if b.global_load.is_present() else 0) + (
            1 if b.global_load_1.is_present() else 0
        )
        while count < max_g and pool_idx < len(pool):
            if not b.global_load.is_present():
                b.global_load = pool[pool_idx]
                b.global_load_prefetch = _is_prefetch(pool[pool_idx])
            elif not b.global_load_1.is_present():
                b.global_load_1 = pool[pool_idx]
                b.global_load_1_prefetch = _is_prefetch(pool[pool_idx])
            pool_idx += 1
            count += 1
        # Recompute drain flag for affected blocks.
        var has_frag = b.pre_op_0.is_present() or b.pre_op_1.is_present()
        var has_dram = (
            b.global_load.is_present() or b.global_load_1.is_present()
        )
        var redist_drain_bit = (sched.drain_lgkm_mask >> i) & 1 == 1
        b.drain_lgkm_before_loads = redist_drain_bit and has_frag and has_dram
        program.blocks[i] = b


def build_double_buffer_program(
    body: LoopBody,
    order: List[Int],
    config: PipelineConfig,
    sched: ScheduleConfig = ScheduleConfig(),
) -> PipelineProgram:
    """Build PipelineProgram from LDG ops in the given execution order.

    Two-phase construction:
      1. Group ops into MMA-centered blocks with structural flags.
      2. Redistribute global loads for uniform distribution.
    """
    var program = _construct_mma_blocks(body, order, config, sched)
    _redistribute_globals(program, config, sched)
    return program^


def derive_waits_from_blocks(
    program: PipelineProgram,
    config: PipelineConfig,
    lgkm_per_a: Int = 0,
    lgkm_per_b: Int = 0,
) -> Tuple[Int, Int]:
    """Derive wait counts from the finalized block structure.

    Unlike the old derive_wait_counts (which operated on the flat LDG
    ordering before block construction), this works on the final
    PipelineProgram after CSP ordering AND post-construction redistribution.
    The counts always reflect the actual block layout.

    Per-channel lgkm cost is read from the config first
    (`config.lgkm_per_channel(channel)`); the legacy `lgkm_per_a/b`
    parameters are only consulted when the config has them unset
    (= 0), preserving backward compat for callers that still thread
    them through `ScheduleConfig`.

    wait_lgkm_first: lgkm ops in block 0's pre_ops (fragment loads issued
        before the first barrier/MMA). Ensures fragment loads complete
        before the MMA consumes their register values.

    wait_vm_last: at the last block's pre_sync, all global loads from
        all blocks in this half have been issued (globals come before
        pre_sync in the block layout). Completion loads must have
        finished; prefetch loads may remain outstanding.
        wait_vm = total_vm_in_half - completion_vm.

    Completion detection uses STAGE-BASED logic (not the k_offset-based
    global_load_prefetch flag, which serves the prologue). A load is
    completion if its stage matches the OTHER half's read stage
    (stage != half), because the other half's fragment loads will read
    from that LDS stage after the half-boundary barrier. A load to the
    SAME half's stage (stage == half) is prefetch; it won't be read
    until the next iteration of this half, so it can remain outstanding.

    Returns (wait_lgkm_first, wait_vm_last).
    """
    var bph = config.blocks_per_partition()
    var num_partitions = config.num_partitions
    var best_lgkm = 0
    var best_vm = 255  # Start high, take min (strictest wait wins).

    # Per-channel lgkm: prefer the config (auto-derived from
    # KernelGeometry); fall back to the explicit params if unset.
    var ch0_lgkm = config.lgkm_per_channel(0)
    if ch0_lgkm == 0:
        ch0_lgkm = lgkm_per_a
    var ch1_lgkm = config.lgkm_per_channel(1)
    if ch1_lgkm == 0:
        ch1_lgkm = lgkm_per_b

    for half in range(num_partitions):
        var partition_start = half * bph

        # wait_lgkm_first: the pre_sync goes after global loads and before
        # the barrier. At that point, frag lgkm + global lgkm are in flight.
        # We want frag loads to complete, so wait until only global lgkm
        # remains outstanding: lgkmcnt = global_lgkm_in_block0.
        var block0 = program.blocks[partition_start]
        var gl_lgkm = 0
        if block0.global_load.is_present():
            gl_lgkm += ch0_lgkm if block0.global_load.channel == 0 else ch1_lgkm
        if block0.global_load_1.is_present():
            gl_lgkm += (
                ch0_lgkm if block0.global_load_1.channel == 0 else ch1_lgkm
            )
        best_lgkm = max(best_lgkm, gl_lgkm)

        # wait_vm_last: walk all blocks in this half.
        # A load is COMPLETION if its stage targets the OTHER half's read
        # stage (stage != half). The other half's fragment loads will read
        # from that stage after the barrier — the data MUST be in LDS.
        # A load to the SAME stage (stage == half) is prefetch — not read
        # until next iteration, can remain outstanding through the barrier.
        var total_vm = 0
        var completion_vm = 0
        for b in range(bph):
            var block = program.blocks[partition_start + b]
            if block.global_load.is_present():
                var vm = config.vm_per_channel(block.global_load.channel)
                total_vm += vm
                if block.global_load.stage != half:
                    completion_vm += vm
            if block.global_load_1.is_present():
                var vm = config.vm_per_channel(block.global_load_1.channel)
                total_vm += vm
                if block.global_load_1.stage != half:
                    completion_vm += vm
        # Allow same-stage prefetch loads to remain outstanding; require
        # other-stage completion loads to finish before the half-boundary
        # barrier. Use min across halves (stricter wait wins).
        best_vm = min(best_vm, total_vm - completion_vm)

    return (best_lgkm, best_vm)


def derive_safe_max_globals(num_k_mmas: Int) -> Int:
    """Derive safe max_globals for uniform global load distribution.

    Returns 1 if uniform distribution is safe under warp stagger, 0 otherwise.

    The safety condition depends on the number of K-dimension MMA tiles
    (num_k_mmas). With warp stagger, WG0 runs 1 MMA phase ahead of WG1.
    When globals are uniformly distributed, a prefetch buffer_load_*_lds in
    block b writes to LDS stage h asynchronously. If block b+1's fragment
    loads read from the same stage, the async LDS write must complete before
    the ds_read; the MMA compute between them must provide enough cycles.

    With num_k_mmas >= 2, each MMA block has 2+ MMAs (~32 cycles on MI355X),
    providing sufficient latency for async LDS writes (~20 cycles).
    With num_k_mmas == 1, the single MMA (~16 cycles) is insufficient.
    """
    if num_k_mmas >= 2:
        return 1  # Safe: enough MMA latency covers async LDS writes.
    return 0  # Unsafe: bunch all globals at block boundaries.


def derive_drain_mask(
    program: PipelineProgram,
    config: PipelineConfig,
) -> Int:
    """Derive per-block lgkm drain mask from block content analysis.

    Drains LDS reads (wait_lgkm(0)) before DRAM→LDS writes when a block's
    fragment loads and global loads target different channels. This avoids
    LDS port contention between concurrent reads and writes to different
    LDS regions.

    Skips the drain when fragment loads and the global load share the same
    channel: sequential access to the same LDS region has less contention.

    Returns a bitmask where bit i=1 means block i should drain lgkm before
    its global loads.
    """
    var mask = 0
    var bph = config.blocks_per_partition()
    var num_blocks = bph * config.num_partitions

    for i in range(num_blocks):
        var b = program.blocks[i]
        var has_frag = b.pre_op_0.is_present() or b.pre_op_1.is_present()
        var has_global = b.global_load.is_present()
        if not has_frag or not has_global:
            continue

        # Collect fragment load channels.
        var frag_ch_0 = b.pre_op_0.channel if b.pre_op_0.is_present() else -1
        var frag_ch_1 = b.pre_op_1.channel if b.pre_op_1.is_present() else -1
        var global_ch = b.global_load.channel

        # Drain when any fragment load targets a different channel than
        # the global load — different LDS regions contend on the port.
        var cross_channel = False
        if frag_ch_0 >= 0 and frag_ch_0 != global_ch:
            cross_channel = True
        if frag_ch_1 >= 0 and frag_ch_1 != global_ch:
            cross_channel = True

        if cross_channel:
            mask |= 1 << i

    return mask


def dump_program_blocks(program: PipelineProgram, config: PipelineConfig):
    """Print the MMA block layout for diagnostic analysis.

    Shows each block's fragment loads, global loads, and MMA with stage/subtile
    annotations. Useful for debugging schedule construction and verifying that
    prefetch loads don't land in unsafe positions relative to fragment loads.
    """
    var bph = config.blocks_per_partition()

    for half in range(config.num_partitions):
        print("=== Half", half, "===")
        for b in range(bph):
            var idx = half * bph + b
            var block = program.blocks[idx]
            print("  Block", idx, ":")
            if block.pre_op_0.is_present():
                print(
                    "    pre_op_0: tag=",
                    block.pre_op_0.tag,
                    "stage=",
                    block.pre_op_0.stage,
                    "ch=",
                    block.pre_op_0.channel,
                )
            if block.pre_op_1.is_present():
                print(
                    "    pre_op_1: tag=",
                    block.pre_op_1.tag,
                    "stage=",
                    block.pre_op_1.stage,
                    "ch=",
                    block.pre_op_1.channel,
                )
            if block.global_load.is_present():
                var pfx = (
                    " [prefetch]" if block.global_load.stage
                    == half else " [completion]"
                )
                print(
                    "    global_load: tag=",
                    block.global_load.tag,
                    "stage=",
                    block.global_load.stage,
                    "ch=",
                    block.global_load.channel,
                    pfx,
                )
            if block.global_load_1.is_present():
                var pfx = (
                    " [prefetch]" if block.global_load_1.stage
                    == half else " [completion]"
                )
                print(
                    "    global_load_1: tag=",
                    block.global_load_1.tag,
                    "stage=",
                    block.global_load_1.stage,
                    "ch=",
                    block.global_load_1.channel,
                    pfx,
                )
            print(
                "    mma: tag=",
                block.mma.tag,
                "stage=",
                block.mma.stage,
                "sub=",
                block.mma.subtile,
            )
            if block.drain_lgkm_before_loads:
                print("    [drain_lgkm_before_loads]")


def _verify_block_completeness(program: PipelineProgram, num_blocks: Int):
    """Every block must have exactly one MMA op with COMPUTE role."""
    for i in range(num_blocks):
        debug_assert(
            program.blocks[i].mma.is_present(),
            "verify_schedule: block " + String(i) + " has no MMA op",
        )
        debug_assert(
            program.blocks[i].mma.role == OpRole.COMPUTE,
            "verify_schedule: block " + String(i) + " MMA op has wrong role",
        )


def _verify_wait_count_bounds(
    program: PipelineProgram,
    config: PipelineConfig,
    lgkm_per_a: Int,
    lgkm_per_b: Int,
    wait_lgkm_first: Int,
    wait_vm_last: Int,
):
    """Wait counts must be non-negative and bounded by actual lgkm costs."""
    var bph = config.blocks_per_partition()
    var num_partitions = config.num_partitions

    if wait_lgkm_first < 255:
        debug_assert(
            wait_lgkm_first >= 0,
            "verify_schedule: wait_lgkm_first is negative",
        )
        # wait_lgkm_first should not exceed total lgkm in any block 0's
        # globals (it represents outstanding lgkm when frags must be done).
        # Only check when lgkm costs are known (auto_waits provides them).
        if lgkm_per_a > 0 or lgkm_per_b > 0:
            for half in range(num_partitions):
                var b0 = program.blocks[half * bph]
                var gl_lgkm = 0
                if b0.global_load.is_present():
                    gl_lgkm += (
                        lgkm_per_a if b0.global_load.channel
                        == 0 else lgkm_per_b
                    )
                if b0.global_load_1.is_present():
                    gl_lgkm += (
                        lgkm_per_a if b0.global_load_1.channel
                        == 0 else lgkm_per_b
                    )
                debug_assert(
                    wait_lgkm_first <= gl_lgkm,
                    "verify_schedule: wait_lgkm_first ("
                    + String(wait_lgkm_first)
                    + ") > block 0 global lgkm ("
                    + String(gl_lgkm)
                    + ") in half "
                    + String(half),
                )

    debug_assert(
        wait_vm_last >= 0,
        "verify_schedule: wait_vm_last is negative",
    )


def _verify_distribution_invariant(
    program: PipelineProgram, num_blocks: Int, max_globals: Int
):
    """When max_globals > 0, every block should have uniform global loads."""
    if max_globals <= 0:
        return

    var total_globals = 0
    for i in range(num_blocks):
        var b = program.blocks[i]
        var count = (1 if b.global_load.is_present() else 0) + (
            1 if b.global_load_1.is_present() else 0
        )
        total_globals += count

    # Only check uniformity if we have enough globals to fill all blocks.
    if total_globals >= num_blocks * max_globals:
        for i in range(num_blocks):
            var b = program.blocks[i]
            var count = (1 if b.global_load.is_present() else 0) + (
                1 if b.global_load_1.is_present() else 0
            )
            debug_assert(
                count == max_globals,
                "verify_schedule: block "
                + String(i)
                + " has "
                + String(count)
                + " globals, expected "
                + String(max_globals),
            )


def _verify_stage_consistency(program: PipelineProgram, config: PipelineConfig):
    """Fragment loads in half h should use stage h (depth=2 double buffer).

    Skipped when `config.cross_stage_rotation` is set: schedules that
    intentionally pre-load the next K-partition's leading-quadrant
    fragments from the cross stage (4-wave's mini-3/4 rotation) have
    `frag.stage != partition_index` *by design*, and a separate
    correctness path (`derive_cross_stage_rotation_edges` +
    `filter_spurious_cross_stage_flow` in `phase_derivation`) handles
    the dependency edges those frags introduce.
    """
    if config.depth != 2 or config.cross_stage_rotation:
        return

    var bph = config.blocks_per_partition()
    var num_partitions = config.num_partitions

    for half in range(num_partitions):
        var expected_stage = half
        for b in range(bph):
            var block = program.blocks[half * bph + b]
            if block.pre_op_0.is_present():
                debug_assert(
                    block.pre_op_0.stage == expected_stage,
                    "verify_schedule: block "
                    + String(half * bph + b)
                    + " pre_op_0 stage="
                    + String(block.pre_op_0.stage)
                    + " expected="
                    + String(expected_stage),
                )
            if block.pre_op_1.is_present():
                debug_assert(
                    block.pre_op_1.stage == expected_stage,
                    "verify_schedule: block "
                    + String(half * bph + b)
                    + " pre_op_1 stage="
                    + String(block.pre_op_1.stage)
                    + " expected="
                    + String(expected_stage),
                )


def _verify_prefetch_coverage(program: PipelineProgram, num_blocks: Int):
    """At least one global load must be marked as prefetch."""
    var has_prefetch = False
    for i in range(num_blocks):
        if program.blocks[i].global_load_prefetch:
            has_prefetch = True
            break
        if program.blocks[i].global_load_1_prefetch:
            has_prefetch = True
            break
    debug_assert(
        has_prefetch,
        "verify_schedule: no global load is marked as prefetch",
    )


def _verify_completion_coverage(
    program: PipelineProgram, config: PipelineConfig
):
    """Each half must have at least one completion load (stage != half).

    Without this, the other half's fragment loads would read stale LDS
    because wait_vm at the half boundary would drain nothing.

    Skipped when `config.cross_stage_rotation` is set: 4-wave's
    rotation doesn't rely on a cross-stage *global* load for drain
    (its cross-stage *fragment* loads + per-mini `s_waitcnt vmcnt(N)`
    handle the dependency directly), so the absence of a stage-h+1
    global load in half h is intentional.
    """
    if config.depth != 2 or config.cross_stage_rotation:
        return

    var bph = config.blocks_per_partition()
    var num_partitions = config.num_partitions

    for half in range(num_partitions):
        var has_completion = False
        for b in range(bph):
            var block = program.blocks[half * bph + b]
            if (
                block.global_load.is_present()
                and block.global_load.stage != half
            ):
                has_completion = True
            if (
                block.global_load_1.is_present()
                and block.global_load_1.stage != half
            ):
                has_completion = True
        debug_assert(
            has_completion,
            "verify_schedule: half "
            + String(half)
            + " has no completion load (stage != half)"
            + " — wait_vm would drain nothing, race condition",
        )


def _verify_warp_stagger_lds_safety(
    program: PipelineProgram, config: PipelineConfig
):
    """Verify no cross-block LDS write/read race under warp stagger.

    With warp stagger, WG0 runs 1 MMA phase ahead of WG1. If block b has a
    prefetch global load (buffer_load_*_lds) writing to LDS stage h, and the
    next block b+1 has fragment loads (ds_read) from the same stage h, the
    async LDS write may not have completed before the read fires.

    Safety depends on MMA latency within each block: if the kernel has enough
    K-dimension MMA tiles per block (num_k_mmas >= 2), the MMA compute provides
    sufficient cycles (~32 on MI355X) to cover the async LDS write (~20 cycles).
    With only 1 MMA per block (~16 cycles), the latency is insufficient.

    The safe max_globals value is derived by derive_safe_max_globals() at
    schedule construction time:
      - max_globals=0 bunches loads at block boundaries, avoiding the race
      - max_globals=1 spreads loads uniformly (safe only with num_k_mmas >= 2)

    This verification checks that when max_globals == 0 (bunched), no prefetch
    loads appear in mid-half blocks where they could race with the next block's
    fragment loads from the same stage.
    """
    if config.depth != 2 or not config.warp_stagger.enabled:
        return

    # When max_globals > 0, derive_safe_max_globals() has already validated
    # that there is enough MMA latency to cover the async LDS write. The
    # uniform distribution is safe by construction.
    if config.block_sizing.max_globals > 0:
        return

    # When max_globals == 0, global loads should be bunched at block
    # boundaries (first/last blocks of each half). Verify no mid-half
    # block has prefetch loads targeting the same stage as the next
    # block's fragment loads.
    var bph = config.blocks_per_partition()

    for half in range(config.num_partitions):
        var base = half * bph
        for i in range(1, bph - 1):
            var block = program.blocks[base + i]

            for gl_idx in range(2):
                var gl = (
                    block.global_load if gl_idx == 0 else block.global_load_1
                )
                if not gl.is_present():
                    continue
                # Prefetch = load targets same half's stage (stage == half).
                if gl.stage != half:
                    continue  # Completion load — safe.

                var next_block = program.blocks[base + i + 1]
                for fl_idx in range(2):
                    var fl = (
                        next_block.pre_op_0 if fl_idx
                        == 0 else next_block.pre_op_1
                    )
                    if not fl.is_present():
                        continue
                    debug_assert(
                        fl.stage != gl.stage,
                        "verify_schedule: cross-block LDS race — block "
                        + String(base + i)
                        + " prefetch writes stage "
                        + String(gl.stage)
                        + " which block "
                        + String(base + i + 1)
                        + " reads via fragment load. Use max_globals=0"
                        + " or increase num_k_mmas to add latency.",
                    )


def verify_schedule(
    program: PipelineProgram,
    config: PipelineConfig,
    lgkm_per_a: Int,
    lgkm_per_b: Int,
    wait_lgkm_first: Int,
    wait_vm_last: Int,
):
    """Verify structural invariants of a finalized pipeline schedule.

    Runs at compile time, zero runtime cost. Catches bugs in schedule
    construction that would otherwise surface as silent GPU miscomputes.

    Checks:
      1. Block completeness: every block has exactly 1 MMA op
      2. Wait count bounds: wait values are non-negative and bounded
      3. Distribution invariant: uniform global loads when max_globals > 0
      4. Stage consistency: fragment/global ops use correct stage for half
      5. Prefetch coverage: at least one global is marked prefetch
      6. Completion coverage: each half has a completion load
      7. Warp stagger LDS safety: no cross-block prefetch/fragment race

    Note: these are structural checks on the generated program. They do
    not verify that the execution ordering respects all LDG dependency
    edges; that is enforced by the scheduler (greedy_schedule /
    optimal_schedule) which only places ops whose d=0 predecessors are
    already scheduled.

    Args:
        program: The finalized `PipelineProgram` to verify.
        config: Pipeline configuration providing block structure geometry
            and partition count.
        lgkm_per_a: Per-channel lgkm cost for channel 0 loads, used to
            bound `wait_lgkm_first`. 0 means unknown (skips the bound
            check).
        lgkm_per_b: Per-channel lgkm cost for channel 1 loads, used to
            bound `wait_lgkm_first`. 0 means unknown (skips the bound
            check).
        wait_lgkm_first: The lgkm wait count applied at the first block of
            each half. 255 means no wait is emitted.
        wait_vm_last: The vm wait count applied at the last block of each
            half.
    """
    var num_blocks = config.blocks_per_partition() * config.num_partitions
    var max_g = config.block_sizing.max_globals

    debug_assert(
        len(program.blocks) == num_blocks,
        "verify_schedule: expected "
        + String(num_blocks)
        + " blocks, got "
        + String(len(program.blocks)),
    )

    _verify_block_completeness(program, num_blocks)
    _verify_wait_count_bounds(
        program,
        config,
        lgkm_per_a,
        lgkm_per_b,
        wait_lgkm_first,
        wait_vm_last,
    )
    _verify_distribution_invariant(program, num_blocks, max_g)
    _verify_stage_consistency(program, config)
    _verify_prefetch_coverage(program, num_blocks)
    _verify_completion_coverage(program, config)
    _verify_warp_stagger_lds_safety(program, config)


def build_kernel_program(
    body: List[OpDesc],
    config: PipelineConfig,
    sched: ScheduleConfig = ScheduleConfig(),
    edges: List[DepEdge] = List[DepEdge](),
) -> PipelineProgram:
    """Build the finalized PipelineProgram for a double-buffer kernel.

    Runs the full pipeline: LDG construction → CSP/greedy scheduling →
    program building → redistribution → auto-waits → auto-drain →
    verification. Returns the PipelineProgram before expansion to entries.

    Used by ScheduleCompiler.compile() for double-buffer kernel derivation
    and by derive_prologue_from_program() for prologue extraction.

    Args:
        body: Loop body op list (typically `PipelineSchedule.build_body()`).
        config: Pipeline configuration (depth, MMA grid, etc.).
        sched: Scheduling configuration (CSP knobs, wait overrides).
        edges: Optional dependency edges to drive scheduling. If empty
            (default), falls back to `derive_edges_from_ops(body, config)`
            (preserves existing behavior for callers that don't override
            `PipelineSchedule.derive_edges`). Schedules that need extra
            edges (e.g., cross-half FRAG→MMA for cross-stage rotation)
            should override `derive_edges` and pass them through here.

    Returns:
        Finalized `PipelineProgram` before expansion to entries.
    """
    # Build LDG from body ops + (caller-provided or auto-derived) edges.
    var ldg = LoopBody()
    for i in range(len(body)):
        ldg.ops.append(OpNode(op=body[i]))
    if len(edges) > 0:
        ldg.edges = edges.copy()
    else:
        ldg.edges = derive_edges_from_ops(body, config)

    # Schedule.
    var order = List[Int]()
    if sched.scheduling == SchedulingStrategy.CSP:
        order = optimal_schedule_with_halves(
            ldg,
            max_globals_per_block=config.block_sizing.max_globals,
            lds_contention_penalty=sched.lds_contention_penalty,
            max_vgpr=sched.max_vgpr,
        )
    elif sched.scheduling == SchedulingStrategy.GREEDY:
        order = greedy_schedule(ldg)
    else:
        for i in range(len(body)):
            order.append(i)

    # Build program (with placeholder waits — will be patched if auto_waits).
    var effective_sched = sched
    var program = build_double_buffer_program(
        ldg, order, config, effective_sched
    )

    # Auto-derive wait counts and drain mask from the FINAL block structure
    # (after CSP ordering and post-construction redistribution).
    var final_lgkm = 0
    var final_vm = 0
    if sched.auto_waits:
        var waits = derive_waits_from_blocks(
            program,
            config,
            sched.lgkm_per_load_a,
            sched.lgkm_per_load_b,
        )
        final_lgkm = waits[0]
        final_vm = waits[1]

        # Auto-derive per-block drain mask from channel analysis (opt-in).
        if sched.auto_drain:
            var auto_mask = derive_drain_mask(program, config)
            var num_blocks = (
                config.blocks_per_partition() * config.num_partitions
            )
            for i in range(num_blocks):
                var b = program.blocks[i]
                var has_frag = (
                    b.pre_op_0.is_present() or b.pre_op_1.is_present()
                )
                var has_dram = (
                    b.global_load.is_present() or b.global_load_1.is_present()
                )
                b.drain_lgkm_before_loads = (
                    (auto_mask >> i) & 1 == 1 and has_frag and has_dram
                )
                program.blocks[i] = b

        var bph = config.blocks_per_partition()
        for half in range(config.num_partitions):
            var b0 = half * bph
            var bN = b0 + bph - 1
            # Patch block 0: wait_lgkm.
            # When `minimal_barriers` AND block 0 already has an
            # `entry_wait` (vm drain), put the lgkm drain in
            # `entry_wait_lgkm` so the two waits are emitted back-to-back
            # and LLVM merges them into one `vmcnt(N) lgkmcnt(M)` instr.
            if waits[0] < 255:
                var blk = program.blocks[b0]
                # Under barrier_before_pre_ops the wait fires BEFORE
                # any of this block's frags/globals issue, so it must
                # drain all prior LDS ops (lgkmcnt=0). Otherwise the
                # wait fires after this block's frag+load issued, and
                # only frags need draining (waits[0] = block0.global_lgkm).
                var lgkm_val = 0 if sched.barrier_before_pre_ops else waits[0]
                if sched.minimal_barriers and blk.entry_wait.is_present():
                    blk.entry_wait_lgkm = OpDesc.wait_lgkm_n(lgkm_val)
                    blk.pre_sync = OpDesc.none()
                else:
                    blk.pre_sync = OpDesc.wait_lgkm_n(lgkm_val)
                program.blocks[b0] = blk
            # Patch last block: wait_vm.
            var blk = program.blocks[bN]
            blk.pre_sync = OpDesc.wait_vm_n(waits[1])
            program.blocks[bN] = blk

    # Verify schedule invariants (comptime only — zero runtime cost).
    verify_schedule(
        program,
        config,
        sched.lgkm_per_load_a,
        sched.lgkm_per_load_b,
        final_lgkm,
        final_vm,
    )

    return program^


def default_kernel_deps_double_buffer(
    kernel: List[ScheduleEntry],
    config: PipelineConfig,
) -> List[DepEdge]:
    """Derive kernel-phase deps for double-buffer by scanning MMA positions.

    For each half: barrier→MMA[0], wait_lgkm→MMA[0], MMA chain.
    """
    var half = len(kernel) // config.num_partitions
    var result = List[DepEdge]()

    for h in range(config.num_partitions):
        var offset = h * half
        var mma_positions = List[Int]()
        var block_starts = List[Int]()
        var lgkm_position = -1

        for i in range(half):
            var op = kernel[offset + i].op
            if op.role == OpRole.COMPUTE:
                mma_positions.append(offset + i)
            if op.role == OpRole.SYNC and len(mma_positions) == len(
                block_starts
            ):
                block_starts.append(offset + i)
            if op.tag == _Ops.WAIT_LGKM.value and lgkm_position < 0:
                lgkm_position = offset + i

        # Block 0: start→mma, lgkm→mma.
        if len(mma_positions) >= 1 and len(block_starts) >= 1:
            result.append(
                DepEdge.flow(
                    producer=block_starts[0],
                    consumer=mma_positions[0],
                )
            )
        if lgkm_position >= 0 and len(mma_positions) >= 1:
            result.append(
                DepEdge.flow(
                    producer=lgkm_position,
                    consumer=mma_positions[0],
                )
            )
        # MMA chain: mma[0]→mma[1]→mma[2].
        if len(mma_positions) >= 3:
            result.append(
                DepEdge.flow(
                    producer=mma_positions[0],
                    consumer=mma_positions[1],
                )
            )
            result.append(
                DepEdge.flow(
                    producer=mma_positions[1],
                    consumer=mma_positions[2],
                )
            )

    return result^


# =============================================================================
# Pipeline Transforms (logical body → pipelined execution order)
# =============================================================================


def single_buffer_reorder(
    logical: List[OpDesc],
    config: PipelineConfig,
) -> List[OpDesc]:
    """Single-buffer pipeline reorder: logical iteration → pipelined steady-state.

    Reorders ops to overlap consecutive iterations, hiding memory latency
    behind compute. Adds one extra barrier (sync) for the pipelined
    steady-state, so the output has len(logical) + 1 ops.

    Output order:

    ```text
    frag[1..T-1], compute[0], sync, store_shared, load_global,
    compute[1..T-1], sync, frag[0]
    ```

    Args:
        logical: The logical iteration's ops in causal order.
        config: Pipeline configuration providing the loop-carried op
            selector used to split fragment loads across the iteration
            boundary.
    """
    var lc = config.loop_carried
    var result = List[OpDesc]()
    var n = len(logical)

    # 1. Fragment loads except loop-carried.
    for i in range(n):
        if logical[i].role == OpRole.FRAGMENT_LOAD:
            if logical[i].subtile != lc.selector:
                result.append(logical[i])

    # 2. Compute[0]: uses loop-carried fragment from previous iteration.
    for i in range(n):
        if logical[i].role == OpRole.COMPUTE:
            if logical[i].subtile == lc.selector:
                result.append(logical[i])

    # 3. Sync: barrier before writing SMEM.
    result.append(OpDesc.barrier())

    # 4. Store shared.
    for i in range(n):
        if logical[i].role == OpRole.SHARED_STORE:
            result.append(logical[i])

    # 5. Load global (prefetch).
    for i in range(n):
        if logical[i].role == OpRole.GLOBAL_LOAD:
            result.append(logical[i])

    # 6. Remaining compute.
    for i in range(n):
        if logical[i].role == OpRole.COMPUTE:
            if logical[i].subtile != lc.selector:
                result.append(logical[i])

    # 7. Sync: barrier before reading new SMEM.
    result.append(OpDesc.barrier())

    # 8. Loop-carried fragment.
    for i in range(n):
        if logical[i].role == lc.role:
            if logical[i].subtile == lc.selector:
                result.append(logical[i])

    return result^


def optimize_within_barriers(
    body: List[OpDesc],
    config: PipelineConfig,
) -> List[OpDesc]:
    """CSP-optimize op ordering within barrier-delimited segments.

    Barriers (OpRole.SYNC) are fixed ordering points. Non-barrier ops
    between consecutive barriers form segments. Each segment is independently
    optimized via the CSP backtracking solver to minimize makespan.

    Edges are derived from the full body, then filtered per segment.
    Only d=0 (intra-iteration) edges are relevant for segment-local ordering.

    Args:
        body: The op list with fixed barriers delimiting segments to
            optimize.
        config: Pipeline configuration used to derive dependency edges for
            segment scheduling.
    """
    var n = len(body)

    # Find barrier positions.
    var barrier_indices = List[Int]()
    for i in range(n):
        if body[i].role == OpRole.SYNC:
            barrier_indices.append(i)

    # Build segment boundaries: [seg_start, barrier) per segment,
    # plus a final segment after the last barrier.
    var seg_starts = List[Int]()
    var seg_ends = List[Int]()
    var start = 0
    for b in range(len(barrier_indices)):
        seg_starts.append(start)
        seg_ends.append(barrier_indices[b])
        start = barrier_indices[b] + 1
    seg_starts.append(start)
    seg_ends.append(n)

    # Derive edges from the full body.
    var all_edges = derive_edges_from_ops(body, config)

    # Optimize each segment independently, emitting barriers between.
    var result = List[OpDesc]()
    for s in range(len(seg_starts)):
        var seg_lo = seg_starts[s]
        var seg_hi = seg_ends[s]
        var seg_len = seg_hi - seg_lo

        if seg_len <= 1:
            # Trivial segment — emit in original order.
            for i in range(seg_lo, seg_hi):
                result.append(body[i])
        else:
            # Build sub-LoopBody for this segment.
            var ldg = LoopBody()
            for i in range(seg_lo, seg_hi):
                ldg.ops.append(OpNode(op=body[i]))

            # Filter edges: both endpoints in this segment, d=0 only.
            for e in range(len(all_edges)):
                var edge = all_edges[e]
                if edge.loop_distance != 0:
                    continue
                if (
                    edge.producer_idx >= seg_lo
                    and edge.producer_idx < seg_hi
                    and edge.consumer_idx >= seg_lo
                    and edge.consumer_idx < seg_hi
                ):
                    ldg.edges.append(
                        DepEdge(
                            producer_idx=edge.producer_idx - seg_lo,
                            consumer_idx=edge.consumer_idx - seg_lo,
                            dep_kind=edge.dep_kind,
                            loop_distance=0,
                        )
                    )

            var order = optimal_schedule(ldg)
            for i in range(len(order)):
                result.append(body[seg_lo + order[i]])

        # Emit barrier after segment (except after the last segment).
        if s < len(barrier_indices):
            result.append(body[barrier_indices[s]])

    return result^


def mma_block_interleave[
    N: Int
](logical: Pipe[N], config: PipelineConfig) -> Pipe[N]:
    """Interleave ops across MMA blocks for latency hiding.

    Takes the logical iteration in causal order: what one ping-pong half
    computes:

    ```text
    global_loads → fragment_loads → MMAs
    ```

    Distributes them across MMA blocks so fragment loads and global loads
    execute during MMA stalls. Fragment loads are placed just before their
    first consumer MMA. Global loads fill remaining slots.

    Block sizing from config: heavy blocks (new M-tile row) vs light blocks
    (continuation). Fragment ordering from config: B-before-A or A-before-B.
    """
    var sizing = config.block_sizing
    var b_first = config.frag_order.b_before_a
    var result = Pipe[N]()
    var pos = 0

    # Classify ops by type.
    var globals = Array[OpDesc, N](uninitialized=True)
    var frag_a = Array[OpDesc, N](uninitialized=True)
    var frag_b = Array[OpDesc, N](uninitialized=True)
    var mmas = Array[OpDesc, N](uninitialized=True)
    var n_g = 0
    var n_fa = 0
    var n_fb = 0
    var n_m = 0

    for i in range(N):
        var op = logical.ops[i]
        if op.role == OpRole.GLOBAL_LOAD:
            globals[n_g] = op
            n_g += 1
        elif op.role == OpRole.FRAGMENT_LOAD and op.channel == 0:
            frag_a[n_fa] = op
            n_fa += 1
        elif op.role == OpRole.FRAGMENT_LOAD and op.channel == 1:
            frag_b[n_fb] = op
            n_fb += 1
        elif op.role == OpRole.COMPUTE:
            mmas[n_m] = op
            n_m += 1

    # Track which M-tile rows and N-tile cols have been seen.
    var seen_m = Array[Int, N](fill=0)
    var seen_n = Array[Int, N](fill=0)

    var g_idx = 0  # next global load to place

    # Build blocks in MMA execution order.
    for b in range(n_m):
        var m_tile = mmas[b].stage  # M-tile (row) index
        var n_tile = mmas[b].subtile  # N-tile (col) index

        var new_m = seen_m[m_tile] == 0
        var new_n = seen_n[n_tile] == 0

        # Emit fragment loads in configured order (B-first or A-first).
        if b_first:
            if new_n:
                for j in range(n_fb):
                    if frag_b[j].subtile == n_tile:
                        result.ops[pos] = frag_b[j]
                        pos += 1
                        break
            if new_m:
                for j in range(n_fa):
                    if frag_a[j].subtile == m_tile:
                        result.ops[pos] = frag_a[j]
                        pos += 1
                        break
        else:
            if new_m:
                for j in range(n_fa):
                    if frag_a[j].subtile == m_tile:
                        result.ops[pos] = frag_a[j]
                        pos += 1
                        break
            if new_n:
                for j in range(n_fb):
                    if frag_b[j].subtile == n_tile:
                        result.ops[pos] = frag_b[j]
                        pos += 1
                        break

        # Fill with global loads using config block sizing.
        var frag_count = 0
        if new_n:
            frag_count += 1
        if new_m:
            frag_count += 1
        var target = sizing.heavy if new_m else sizing.light
        var available = max(target - frag_count - 1, 0)
        var n_globals_here = min(available, n_g - g_idx)

        for _ in range(n_globals_here):
            result.ops[pos] = globals[g_idx]
            pos += 1
            g_idx += 1

        # Emit MMA.
        result.ops[pos] = mmas[b]
        pos += 1

        seen_m[m_tile] = 1
        seen_n[n_tile] = 1

    return result^


def mma_block_interleave_list(
    logical: List[OpDesc],
    config: PipelineConfig,
) -> List[OpDesc]:
    """List-based MMA block interleave (equivalent to mma_block_interleave).

    Distributes ops across MMA blocks for latency hiding. Fragment loads
    are placed just before their first consumer MMA; global loads fill
    remaining slots. Output order matches the Pipe[N] version exactly.

    Args:
        logical: The logical iteration's ops in causal order: global
            loads, fragment loads, then MMAs.
        config: Pipeline configuration providing block sizing and
            fragment ordering.
    """
    var sizing = config.block_sizing
    var b_first = config.frag_order.b_before_a
    var n = len(logical)
    var result = List[OpDesc]()

    # Classify ops.
    var globals = List[OpDesc]()
    var frag_a = List[OpDesc]()
    var frag_b = List[OpDesc]()
    var mmas = List[OpDesc]()

    for i in range(n):
        var op = logical[i]
        if op.role == OpRole.GLOBAL_LOAD:
            globals.append(op)
        elif op.role == OpRole.FRAGMENT_LOAD and op.channel == 0:
            frag_a.append(op)
        elif op.role == OpRole.FRAGMENT_LOAD and op.channel == 1:
            frag_b.append(op)
        elif op.role == OpRole.COMPUTE:
            mmas.append(op)

    # Track which M-tile rows and N-tile cols have been seen.
    var seen_m = List[Bool]()
    var seen_n = List[Bool]()
    for _ in range(n):
        seen_m.append(False)
        seen_n.append(False)

    var g_idx = 0

    for b in range(len(mmas)):
        var m_tile = mmas[b].stage
        var n_tile = mmas[b].subtile

        var new_m = not seen_m[m_tile]
        var new_n = not seen_n[n_tile]

        if b_first:
            if new_n:
                for j in range(len(frag_b)):
                    if frag_b[j].subtile == n_tile:
                        result.append(frag_b[j])
                        break
            if new_m:
                for j in range(len(frag_a)):
                    if frag_a[j].subtile == m_tile:
                        result.append(frag_a[j])
                        break
        else:
            if new_m:
                for j in range(len(frag_a)):
                    if frag_a[j].subtile == m_tile:
                        result.append(frag_a[j])
                        break
            if new_n:
                for j in range(len(frag_b)):
                    if frag_b[j].subtile == n_tile:
                        result.append(frag_b[j])
                        break

        # Fill with global loads.
        var frag_count = 0
        if new_n:
            frag_count += 1
        if new_m:
            frag_count += 1
        var target = sizing.heavy if new_m else sizing.light
        var available = max(target - frag_count - 1, 0)
        if sizing.max_globals > 0:
            # Floor: ensure at least 1 global per block (if any remain).
            # Cap: don't exceed max_globals.
            available = min(max(available, 1), sizing.max_globals)
        var n_globals_here = min(available, len(globals) - g_idx)

        for _ in range(n_globals_here):
            result.append(globals[g_idx])
            g_idx += 1

        result.append(mmas[b])

        seen_m[m_tile] = True
        seen_n[n_tile] = True

    return result^


def double_buffer_reorder(
    logical: List[OpDesc],
    config: PipelineConfig,
) -> List[OpDesc]:
    """Reorder a double-buffer spec's logical ops into interleaved execution order.

    The spec provides ops in logical order: half0 ops followed by half1 ops.
    This function splits at the midpoint, applies mma_block_interleave to
    each half independently, and concatenates the results.
    """
    var partition_size = len(logical) // 2
    var result = List[OpDesc]()

    # Split into halves.
    var half0 = List[OpDesc]()
    var half1 = List[OpDesc]()
    for i in range(len(logical)):
        if i < partition_size:
            half0.append(logical[i])
        else:
            half1.append(logical[i])

    # Interleave each half.
    var interleaved0 = mma_block_interleave_list(half0, config)
    var interleaved1 = mma_block_interleave_list(half1, config)

    for i in range(len(interleaved0)):
        result.append(interleaved0[i])
    for i in range(len(interleaved1)):
        result.append(interleaved1[i])

    return result^
