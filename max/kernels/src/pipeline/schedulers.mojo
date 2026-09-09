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
"""Scheduling algorithms: greedy and within-iteration optimal schedulers.

These are free functions that consume a LoopBody and return a permutation.
The graph (LoopBody) never schedules itself; scheduling is an external
concern. This follows the LLVM pattern where ScheduleDAG is pure data and
MachineSchedStrategy is a separate pluggable component.

Note: the "optimal" scheduler minimizes within-iteration makespan (single
iteration time span), not the initiation interval (II) used in modulo
scheduling. Inter-iteration overlap is handled by the prologue/epilogue
structure in program_builder, not by the scheduler. See DESIGN.md for
scope and limitations.
"""

from std.collections import List

from .dependency_graph import LoopBody, OpNode
from .types import OpRole, ResourceKind


def _is_ready(
    body: LoopBody,
    op_idx: Int,
    scheduled: List[Bool],
) -> Bool:
    """Check if all d=0 FLOW and ANTI predecessors are scheduled."""
    for e in range(len(body.edges)):
        var edge = body.edges[e]
        if edge.loop_distance >= 1:
            continue
        # Enforce both FLOW (RAW) and ANTI (WAR) d=0 edges.
        # ANTI edges model LDS read-before-write: mma_load must
        # complete before a prefetch global_load overwrites the
        # same LDS buffer. Without enforcing ANTI edges, the
        # scheduler can move prefetch writes before reads, causing
        # LDS corruption.
        if edge.consumer_idx == op_idx and not scheduled[edge.producer_idx]:
            return False
    return True


@fieldwise_init
struct ScheduleMetrics(ImplicitlyCopyable, Movable):
    """Combined evaluation metrics for a schedule ordering."""

    var makespan: Int
    """Total execution time (max finish time across all ops)."""
    var vgpr_peak: Int
    """Peak VGPR pressure (max simultaneous live VGPRs)."""


def _evaluate_makespan(
    body: LoopBody, order: List[Int], lds_contention_penalty: Int = 0
) -> Int:
    """Simulate execution and return makespan only (backward-compatible)."""
    return _evaluate_schedule(body, order, lds_contention_penalty).makespan


def _evaluate_schedule(
    body: LoopBody, order: List[Int], lds_contention_penalty: Int = 0
) -> ScheduleMetrics:
    """Simulate execution of an ordering on a simple resource model.

    Models hardware with independent MMA and VALU execution units, plus
    optional LDS port contention and VGPR pressure tracking:
      - MMA_UNIT: serial (capacity 1), MFMA ops serialize with each other
      - VALU: serial (capacity 1), VALU ops serialize with each other
      - MMA and VALU are *independent*; they can issue simultaneously.
      - LDS port (when lds_contention_penalty > 0): shared between LDS reads
        and LDS writes. Overlapping ops incur a penalty.
      - VGPR pressure: tracked via op.vgpr_def (registers allocated) and
        op.vgpr_kill (registers freed). Peak is the high-water mark.

    Returns ScheduleMetrics with makespan and peak VGPR pressure.
    """
    var finish_time = List[Int]()
    for _ in range(len(body.ops)):
        finish_time.append(0)
    var mma_free = 0
    var valu_free = 0
    var lds_port_free = 0
    var vgpr_live = 0
    var vgpr_peak = 0

    for pos in range(len(body.ops)):
        var op_idx = order[pos]
        var op = body.ops[op_idx]
        var lat = op.latency
        var is_mma = op.resource == ResourceKind.MMA_UNIT
        var is_valu = op.resource == ResourceKind.VALU
        var uses_lds_port = lds_contention_penalty > 0 and (
            op.resource == ResourceKind.LDS
            or op.resource == ResourceKind.GLOBAL_MEM
        )

        # Earliest start: max of d=0 predecessor finish times.
        var start = 0
        for e in range(len(body.edges)):
            var edge = body.edges[e]
            if edge.loop_distance >= 1:
                continue
            if edge.consumer_idx != op_idx:
                continue
            var pred_finish = finish_time[edge.producer_idx]
            if pred_finish > start:
                start = pred_finish

        # Serial MMA resource constraint.
        if is_mma and mma_free > start:
            start = mma_free

        # Serial VALU resource constraint (independent from MMA).
        if is_valu and valu_free > start:
            start = valu_free

        # LDS port contention.
        if uses_lds_port and lds_port_free > start:
            start = lds_port_free

        var end = start + lat
        finish_time[op_idx] = end
        if is_mma:
            mma_free = end
        if is_valu:
            valu_free = end
        if uses_lds_port:
            lds_port_free = start + lds_contention_penalty

        # VGPR pressure: new registers allocated, then dead ones freed.
        vgpr_live += op.op.vgpr_def
        if vgpr_live > vgpr_peak:
            vgpr_peak = vgpr_live
        vgpr_live -= op.op.vgpr_kill

    # Makespan = max finish time.
    var makespan = 0
    for i in range(len(body.ops)):
        if finish_time[i] > makespan:
            makespan = finish_time[i]
    return ScheduleMetrics(makespan=makespan, vgpr_peak=vgpr_peak)


def _asap_lower_bound(body: LoopBody, lds_contention_penalty: Int = 0) -> Int:
    """Compute a lower bound on the achievable within-iteration makespan.

    Returns max(critical_path, compute_bound, lds_serial_bound) where:
      critical_path   = max(ASAP[i] + latency[i]): longest dep chain
      compute_bound   = max(mma_serial_bound, valu_serial_bound)
      mma_serial_bound = min_mma_asap + sum(latency[i]) for MMA ops
      valu_serial_bound = min_valu_asap + sum(latency[i]) for VALU ops
      lds_serial_bound = sum(penalty) for LDS-port ops (when penalty > 0)

    The min_asap offset accounts for dependency-forced delays: if no MMA
    op can start before time T (e.g., because FRAG_K takes T cycles),
    then MMA serial time is T + sum(mma_latencies), not just the sum.

    Note: MMA_UNIT and VALU are independent execution units on CDNA, so
    their bounds are taken as max (not sum), the smaller one runs in the
    shadow of the larger.

    Note: this is a within-iteration bound, not the MII (minimum initiation
    interval) from modulo scheduling theory. If a schedule's makespan
    equals this bound, no better within-iteration ordering exists.
    """
    var asap = body.compute_asap()

    # Critical path bound: longest ASAP completion time.
    var critical_path = 0
    for i in range(len(body.ops)):
        var completion = asap[i] + body.ops[i].latency
        if completion > critical_path:
            critical_path = completion

    # MMA serial bound: earliest possible start + total latency.
    # No MMA op can execute before its ASAP time, so the chain can't
    # start before the minimum ASAP across all MMA ops.
    var mma_total = 0
    var mma_min_asap = 999999
    for i in range(len(body.ops)):
        if body.ops[i].resource == ResourceKind.MMA_UNIT:
            mma_total += body.ops[i].latency
            if asap[i] < mma_min_asap:
                mma_min_asap = asap[i]
    var mma_bound = mma_total
    if mma_total > 0:
        mma_bound = mma_min_asap + mma_total

    # VALU serial bound: same logic — earliest VALU start + total.
    var valu_total = 0
    var valu_min_asap = 999999
    for i in range(len(body.ops)):
        if body.ops[i].resource == ResourceKind.VALU:
            valu_total += body.ops[i].latency
            if asap[i] < valu_min_asap:
                valu_min_asap = asap[i]
    var valu_bound = valu_total
    if valu_total > 0:
        valu_bound = valu_min_asap + valu_total

    # MMA and VALU are independent — take the max, not the sum.
    var compute_bound = mma_bound if mma_bound > valu_bound else valu_bound
    var bound = (
        critical_path if critical_path > compute_bound else compute_bound
    )

    # LDS port serial bound: total penalty for ops that use the LDS port.
    if lds_contention_penalty > 0:
        var lds_count = 0
        for i in range(len(body.ops)):
            var res = body.ops[i].resource
            if res == ResourceKind.LDS or res == ResourceKind.GLOBAL_MEM:
                lds_count += 1
        var lds_total = lds_count * lds_contention_penalty
        if lds_total > bound:
            bound = lds_total

    return bound


def greedy_schedule(body: LoopBody) -> List[Int]:
    """Constrained list scheduler for MMA-centered block structure.

    Derives a valid execution order from the dependency graph,
    respecting structural constraints of the interleaved ping-pong kernel:

      1. Half isolation: ops 0..num_ops/2 are scheduled first (blocks 0-3),
         then ops num_ops/2..num_ops (blocks 4-7). This preserves the
         two-half warp-group structure that build_program_from_ldg_ordered()
         requires (first 12 ops → blocks 0-3, last 12 → blocks 4-7).
      2. Data dependencies: all d=0 predecessors (FLOW and ANTI) must be
         scheduled before the consumer. FLOW edges enforce RAW (register
         and accumulator) deps. ANTI edges enforce WAR (LDS buffer) deps:
         all mma_loads reading from an LDS buffer must complete before any
         prefetch global_load writes to that buffer.
      3. MMA-centered blocks: each block terminates with exactly 1 MMA.
      4. Priority: lowest op index wins among ready ops. This reproduces
         the declaration order from define_interleaved_loop_body(), which
         is the correct execution order. The ScheduleConfig wait counts
         (vmcnt, lgkmcnt) are calibrated for this specific ordering, so
         any reordering requires recalculating those counts.

    Returns a permutation of [0, num_ops) suitable for
    build_program_from_ldg_ordered(). When ops are defined in execution
    order, this produces the identity permutation, validating that the
    dependency graph is sufficient to derive the schedule.
    """
    var scheduled = List[Bool]()
    for _ in range(len(body.ops)):
        scheduled.append(False)
    var order = List[Int]()
    var pos = 0

    # Schedule each half independently.
    var partition_size = len(body.ops) // 2
    for half in range(2):
        var half_lo = half * partition_size
        var half_hi = half_lo + partition_size

        # Process 4 MMA-centered blocks per half.
        for _block in range(4):
            while pos < len(body.ops):
                # Pick lowest-index ready op in this half.
                var best = -1
                for i in range(half_lo, half_hi):
                    if scheduled[i]:
                        continue
                    if not _is_ready(body, i, scheduled):
                        continue
                    best = i
                    break  # Lowest index wins.

                debug_assert(best >= 0, "no ready op found for block")

                order.append(best)
                pos += 1
                scheduled[best] = True
                if body.ops[best].op.role == OpRole.COMPUTE:
                    break  # Block terminated by MMA/compute.

    debug_assert(pos == len(body.ops), "all ops must be scheduled")
    return order^


def simple_greedy_schedule(body: LoopBody) -> List[Int]:
    """Generic dependency-respecting greedy scheduler.

    At each step, picks the lowest-index ready op. Unlike greedy_schedule(),
    this has no structural assumptions (no half-isolation, no MMA-terminated
    blocks) and works with any LoopBody.

    Used as a seed for optimal_schedule() to provide a strong initial upper
    bound, enabling early pruning of the backtracking search.

    Args:
        body: The loop body whose ops are to be scheduled.

    Returns:
        Permutation of [0, num_ops): a valid execution ordering.
    """
    var scheduled = List[Bool]()
    for _ in range(len(body.ops)):
        scheduled.append(False)
    var order = List[Int]()

    for _ in range(len(body.ops)):
        var found = False
        for i in range(len(body.ops)):
            if scheduled[i]:
                continue
            if not _is_ready(body, i, scheduled):
                continue
            order.append(i)
            scheduled[i] = True
            found = True
            break
        debug_assert(found, "no ready op found in simple_greedy_schedule")

    return order^


def _partial_lower_bound(
    body: LoopBody,
    order: List[Int],
    placed: List[Bool],
    pos: Int,
) -> Int:
    """Compute a lower bound on the final makespan from a partial placement.

    Simulates the first `pos` ops to get current MMA/VALU finish times,
    then adds the total remaining latency for each resource. The final
    makespan can't be less than resource_finish + remaining_resource_total
    for any resource.

    This enables branch-and-bound pruning in the CSP solver: if
    partial_lb >= best_cost, the branch can be pruned.
    """
    # Simulate the partial schedule to get resource finish times.
    var finish_time = List[Int]()
    for _ in range(len(body.ops)):
        finish_time.append(0)
    var mma_free = 0
    var valu_free = 0

    for p in range(pos):
        var op_idx = order[p]
        var op = body.ops[op_idx]
        var lat = op.latency
        var is_mma = op.resource == ResourceKind.MMA_UNIT
        var is_valu = op.resource == ResourceKind.VALU

        var start = 0
        for e in range(len(body.edges)):
            var edge = body.edges[e]
            if edge.loop_distance >= 1:
                continue
            if edge.consumer_idx != op_idx:
                continue
            var pred_finish = finish_time[edge.producer_idx]
            if pred_finish > start:
                start = pred_finish

        if is_mma and mma_free > start:
            start = mma_free
        if is_valu and valu_free > start:
            start = valu_free

        var end = start + lat
        finish_time[op_idx] = end
        if is_mma:
            mma_free = end
        if is_valu:
            valu_free = end

    # Sum remaining resource latencies.
    var remaining_mma = 0
    var remaining_valu = 0
    for i in range(len(body.ops)):
        if placed[i]:
            continue
        if body.ops[i].resource == ResourceKind.MMA_UNIT:
            remaining_mma += body.ops[i].latency
        elif body.ops[i].resource == ResourceKind.VALU:
            remaining_valu += body.ops[i].latency

    # Lower bound: each resource must serialize its remaining ops.
    var mma_bound = mma_free + remaining_mma
    var valu_bound = valu_free + remaining_valu
    var bound = mma_bound if mma_bound > valu_bound else valu_bound

    # Also consider current max finish time (for non-MMA/VALU ops).
    var current_max = 0
    for p in range(pos):
        var ft = finish_time[order[p]]
        if ft > current_max:
            current_max = ft
    if current_max > bound:
        bound = current_max

    return bound


def optimal_schedule(body: LoopBody, max_vgpr: Int = 999999) -> List[Int]:
    """Find the minimum-makespan execution ordering via backtracking search.

    Explores all valid orderings (respecting d=0 dependency edges) and
    returns the one with minimum simulated within-iteration makespan.
    Among equal-makespan solutions, prefers lower peak VGPR pressure.

    When max_vgpr is set (default: no limit), rejects orderings whose
    peak VGPR pressure exceeds the budget. This enables occupancy
    exploration: "What is the fastest schedule that fits in N VGPRs?"

    Uses iterative backtracking with explicit state (no recursion).

    Returns:
        Permutation of [0, num_ops) with minimum makespan (and
        lowest pressure among equal-makespan solutions).
    """
    # Seed from generic greedy — provides a strong initial upper bound.
    var seed = simple_greedy_schedule(body)
    var seed_metrics = _evaluate_schedule(body, seed)
    var best_cost = seed_metrics.makespan
    var best_pressure = seed_metrics.vgpr_peak
    var best_order = seed^

    # Early exit: if greedy already achieves the lower bound, it's optimal.
    var lb = _asap_lower_bound(body)
    if best_cost == lb and (max_vgpr == 999999 or best_pressure <= max_vgpr):
        return best_order^

    var placed = List[Bool]()
    for _ in range(len(body.ops)):
        placed.append(False)
    var order = List[Int]()
    for _ in range(len(body.ops)):
        order.append(0)
    var pos = 0

    # last_tried[d] = last op index tried at depth d (-1 = none yet).
    var last_tried = List[Int]()
    for _ in range(len(body.ops)):
        last_tried.append(-1)

    # Partial VGPR pressure at each depth for incremental tracking.
    var vgpr_at_depth = List[Int]()
    for _ in range(len(body.ops) + 1):
        vgpr_at_depth.append(0)

    while pos >= 0:
        if pos == len(body.ops):
            # Complete ordering — evaluate and potentially update best.
            var metrics = _evaluate_schedule(body, order)
            var cost = metrics.makespan
            var pressure = metrics.vgpr_peak

            # Skip if pressure exceeds budget.
            var dominated = pressure > max_vgpr
            # Accept if: better makespan, or same makespan + lower pressure.
            if not dominated and (
                cost < best_cost
                or (cost == best_cost and pressure < best_pressure)
            ):
                best_cost = cost
                best_pressure = pressure
                for i in range(len(body.ops)):
                    best_order[i] = order[i]
                # Early termination: provably optimal makespan + zero pressure.
                if best_cost == lb and best_pressure == 0:
                    return best_order^
            # Backtrack.
            pos -= 1
            placed[order[pos]] = False
            continue

        # Find next ready op with index > last_tried[pos].
        var found = False
        for i in range(last_tried[pos] + 1, len(body.ops)):
            if placed[i]:
                continue
            if not _is_ready(body, i, placed):
                continue
            # Tentatively place it.
            order[pos] = i
            placed[i] = True

            # Incremental VGPR pressure check.
            var prev_vgpr = vgpr_at_depth[pos]
            var new_vgpr = prev_vgpr + body.ops[i].op.vgpr_def
            if new_vgpr > max_vgpr:
                placed[i] = False
                last_tried[pos] = i
                continue
            vgpr_at_depth[pos + 1] = new_vgpr - body.ops[i].op.vgpr_kill

            # Branch-and-bound: prune if partial lower bound >= best.
            var plb = _partial_lower_bound(body, order, placed, pos + 1)
            if plb >= best_cost:
                placed[i] = False
                last_tried[pos] = i
                continue

            last_tried[pos] = i
            pos += 1
            found = True
            break

        if not found:
            # No valid candidate at this depth — backtrack.
            last_tried[pos] = -1
            if pos > 0:
                pos -= 1
                placed[order[pos]] = False
            else:
                break  # Exhausted all possibilities.

    return best_order^


def optimal_schedule_with_halves(
    body: LoopBody,
    max_globals_per_block: Int = 0,
    lds_contention_penalty: Int = 0,
    max_vgpr: Int = 999999,
) -> List[Int]:
    """Optimal scheduler with half-isolation constraint.

    Positions [0, N/2) draw from ops [0, N/2), positions [N/2, N)
    draw from ops [N/2, N). This mirrors the greedy_schedule()
    half-isolation for the ping-pong kernel but uses exhaustive
    backtracking instead of greedy selection.

    Half isolation preserves the warp-group structure that
    build_program_from_ldg_ordered() requires: first N/2 ops map to
    blocks 0-3 (warp group 0), last N/2 to blocks 4-7 (warp group 1).

    Pruning optimizations for compile-time feasibility:
      1. Seed best_cost from greedy greedy_schedule(): provides
         a strong upper bound that prunes most branches immediately.
      2. Early termination when best_cost == lower_bound and pressure
         is at the floor: provably optimal, stop searching.
      3. Incremental VGPR pressure tracking: branches that exceed
         max_vgpr are pruned the moment they cross the budget.

    When max_globals_per_block > 0, an additional structural constraint
    limits how many GLOBAL_LOAD ops can appear between consecutive
    COMPUTE (MMA) ops. This ensures uniform global load distribution
    across MMA blocks after build_double_buffer_program groups by MMA
    delimiters. For example, max_globals_per_block=1 yields a [1,1,1,1]
    distribution instead of the unconstrained [1,0,2,1].

    When lds_contention_penalty > 0, the resource model adds a shared
    LDS port constraint: fragment loads (LDS reads) and global loads
    (LDS writes) contend on the same port, with the given penalty per
    overlap. This biases the solver toward separating LDS reads from
    LDS writes in the schedule.

    When max_vgpr < 999999, orderings whose peak VGPR pressure exceeds
    the budget are rejected. Among equal-makespan solutions, prefers
    the one with lower peak pressure (occupancy-aware scheduling).
    """
    # Seed from greedy schedule — provides a strong initial upper bound.
    var greedy = greedy_schedule(body)
    var seed_metrics = _evaluate_schedule(body, greedy, lds_contention_penalty)
    var best_cost = seed_metrics.makespan
    var best_pressure = seed_metrics.vgpr_peak
    var best_order = greedy^

    # Early exit: if greedy already achieves the lower bound and the
    # seed pressure is at the floor (=0 means no VGPRs hinted), stop.
    var lb = _asap_lower_bound(body, lds_contention_penalty)
    if best_cost == lb and (max_vgpr == 999999 or best_pressure <= max_vgpr):
        return best_order^

    var placed = List[Bool]()
    for _ in range(len(body.ops)):
        placed.append(False)
    var order = List[Int]()
    for _ in range(len(body.ops)):
        order.append(0)
    var pos = 0

    var last_tried = List[Int]()
    for _ in range(len(body.ops)):
        last_tried.append(-1)

    # Incremental VGPR pressure at each search depth.
    var vgpr_at_depth = List[Int]()
    for _ in range(len(body.ops) + 1):
        vgpr_at_depth.append(0)

    var partition_size = len(body.ops) // 2

    while pos >= 0:
        if pos == len(body.ops):
            # Both halves complete — evaluate.
            var metrics = _evaluate_schedule(
                body, order, lds_contention_penalty
            )
            var cost = metrics.makespan
            var pressure = metrics.vgpr_peak

            # Reject if pressure exceeds budget; otherwise accept iff
            # better makespan, or equal makespan + lower pressure.
            var dominated = pressure > max_vgpr
            if not dominated and (
                cost < best_cost
                or (cost == best_cost and pressure < best_pressure)
            ):
                best_cost = cost
                best_pressure = pressure
                for i in range(len(body.ops)):
                    best_order[i] = order[i]
                # Early termination: provably optimal makespan + zero
                # pressure (no VGPR hints to optimize against).
                if best_cost == lb and best_pressure == 0:
                    return best_order^
            # Backtrack.
            pos -= 1
            placed[order[pos]] = False
            continue

        # Determine which half based on current position.
        var half_lo = 0 if pos < partition_size else partition_size
        var half_hi = partition_size if pos < partition_size else len(body.ops)

        # Ensure search starts at half_lo.
        var start_from = last_tried[pos] + 1
        if start_from < half_lo:
            start_from = half_lo

        # Count globals in current block (since last COMPUTE) for the
        # max_globals_per_block constraint.
        var globals_in_block = 0
        if max_globals_per_block > 0:
            var half_start_pos = 0 if pos < partition_size else partition_size
            for j in range(pos - 1, half_start_pos - 1, -1):
                var placed_role = body.ops[order[j]].op.role
                if placed_role == OpRole.COMPUTE:
                    break
                if placed_role == OpRole.GLOBAL_LOAD:
                    globals_in_block += 1

        # Find next ready op in [half_lo, half_hi).
        var found = False
        for i in range(start_from, half_hi):
            if placed[i]:
                continue
            if not _is_ready(body, i, placed):
                continue
            # Enforce max globals per block: skip GLOBAL_LOAD if at limit.
            if (
                max_globals_per_block > 0
                and body.ops[i].op.role == OpRole.GLOBAL_LOAD
                and globals_in_block >= max_globals_per_block
            ):
                continue
            # Incremental VGPR pressure check.
            var prev_vgpr = vgpr_at_depth[pos]
            var new_vgpr = prev_vgpr + body.ops[i].op.vgpr_def
            if new_vgpr > max_vgpr:
                continue
            vgpr_at_depth[pos + 1] = new_vgpr - body.ops[i].op.vgpr_kill

            order[pos] = i
            placed[i] = True
            last_tried[pos] = i
            pos += 1
            found = True
            break

        if not found:
            last_tried[pos] = -1
            if pos > 0:
                pos -= 1
                placed[order[pos]] = False
            else:
                break  # Exhausted all possibilities.

    return best_order^
