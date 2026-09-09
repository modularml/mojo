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
"""Phase derivation: recipes, default prologue/kernel/epilogue, and edge rules.

Contains:
  - PhaseAction, PhaseStep: declarative recipe primitives
  - apply_phase_recipe: recipe evaluator
  - single_buffer_prologue_recipe, single_buffer_epilogue_recipe
  - pipe_to_list, default_prologue, default_kernel, default_epilogue
  - _strip_drain_fuse_blocks, _expand_epilogue_blocks
  - derive_epilogue_from_program
  - double_buffer_edge_rules, single_buffer_edge_rules, apply_edge_rules
  - derive_edges_from_ops
  - default_warp_stagger, default_warp_stagger_double_buffer
  - derive_prologue_from_program
"""

from std.collections import List

from .config import PipelineConfig, ScheduleConfig
from .pipeline_dsl import Pipe, ScheduleEntry
from .program import MMABlockSpec, PipelineProgram
from .types import (
    DepEdge,
    DepKind,
    EdgeRule,
    KOffsetKind,
    OpDesc,
    OpRole,
    Phase,
)

# =============================================================================
# PhaseRecipe — declarative prologue/epilogue generation
# =============================================================================


@fieldwise_init
struct PhaseAction(ImplicitlyCopyable, Movable):
    """Action type for a recipe step."""

    var value: Int

    comptime EMIT = Self(0)  # Match body ops and emit them
    comptime BARRIER = Self(1)  # Emit s_barrier
    comptime FENCE = Self(2)  # Emit schedule_barrier fence

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value


@fieldwise_init
struct PhaseStep(ImplicitlyCopyable, Movable):
    """One step in a phase recipe.

    Specifies what to emit: either infrastructure ops (BARRIER, FENCE) or
    data ops matched from the body by role and predicates (EMIT).
    """

    var action: PhaseAction
    """Action type for this step (`EMIT`, `BARRIER`, or `FENCE`)."""
    var match_role: OpRole
    """For `EMIT` steps: required `OpRole` of the body op to match."""
    var match_subtile: Int
    """For `EMIT` steps: `-1` = any, else a specific subtile value."""
    var exclude_lc: Bool
    """For `EMIT` steps: skip loop-carried ops (`subtile == lc.selector`)."""
    var match_lc_only: Bool
    """For `EMIT` steps: emit only loop-carried ops."""
    var match_all: Bool
    """For `EMIT` steps: emit all matching ops (`True`) or first match only
    (`False`)."""

    @staticmethod
    def emit(
        role: OpRole,
        *,
        subtile: Int = -1,
        exclude_lc: Bool = False,
        match_lc_only: Bool = False,
        match_all: Bool = False,
    ) -> Self:
        """Emit body ops matching the given role and predicates."""
        return Self(
            action=PhaseAction.EMIT,
            match_role=role,
            match_subtile=subtile,
            exclude_lc=exclude_lc,
            match_lc_only=match_lc_only,
            match_all=match_all,
        )

    @staticmethod
    def barrier() -> Self:
        """Emit an s_barrier infrastructure op."""
        return Self(
            action=PhaseAction.BARRIER,
            match_role=OpRole.NONE,
            match_subtile=-1,
            exclude_lc=False,
            match_lc_only=False,
            match_all=False,
        )

    @staticmethod
    def fence() -> Self:
        """Emit a schedule_barrier fence infrastructure op."""
        return Self(
            action=PhaseAction.FENCE,
            match_role=OpRole.NONE,
            match_subtile=-1,
            exclude_lc=False,
            match_lc_only=False,
            match_all=False,
        )


def apply_phase_recipe(
    body: List[OpDesc],
    steps: List[PhaseStep],
    config: PipelineConfig,
    phase: Phase,
) -> List[ScheduleEntry]:
    """Evaluate a phase recipe against a body, producing ScheduleEntries.

    Walks the recipe steps in order. For each EMIT step, scans the body
    for ops matching the step's role and predicates, then emits them as
    ScheduleEntry values. BARRIER and FENCE steps emit infrastructure ops.
    """
    var lc_sel = config.loop_carried.selector
    var result = List[ScheduleEntry]()
    var slot = 0

    for s in range(len(steps)):
        var step = steps[s]

        if step.action == PhaseAction.BARRIER:
            result.append(
                ScheduleEntry(
                    op=OpDesc.barrier(),
                    time_slot=slot,
                    phase=phase,
                    is_prefetch=False,
                )
            )
            slot += 1
            continue

        if step.action == PhaseAction.FENCE:
            result.append(
                ScheduleEntry(
                    op=OpDesc.schedule_barrier(),
                    time_slot=slot,
                    phase=phase,
                    is_prefetch=False,
                )
            )
            slot += 1
            continue

        # EMIT: match body ops by role and predicates.
        for i in range(len(body)):
            var op = body[i]
            if op.role != step.match_role:
                continue
            if step.match_subtile >= 0 and op.subtile != step.match_subtile:
                continue
            if step.exclude_lc and op.subtile == lc_sel:
                continue
            if step.match_lc_only and op.subtile != lc_sel:
                continue

            result.append(
                ScheduleEntry(
                    op=op,
                    time_slot=slot,
                    phase=phase,
                    is_prefetch=False,
                )
            )
            slot += 1

            if not step.match_all:
                break

    return result^


def single_buffer_prologue_recipe() -> List[PhaseStep]:
    """Recipe for single-buffer prologue.

    Sequence: load → store → barrier → load (prefetch) → LC frag → fence.
    """
    var steps = List[PhaseStep]()
    steps.append(PhaseStep.emit(OpRole.GLOBAL_LOAD))
    steps.append(PhaseStep.emit(OpRole.SHARED_STORE))
    steps.append(PhaseStep.barrier())
    steps.append(PhaseStep.emit(OpRole.GLOBAL_LOAD))
    steps.append(PhaseStep.emit(OpRole.FRAGMENT_LOAD, match_lc_only=True))
    steps.append(PhaseStep.fence())
    return steps^


def single_buffer_epilogue_recipe() -> List[PhaseStep]:
    """Recipe for single-buffer epilogue (2 drain passes).

    Drain 1: fence, non-LC frags, barrier, store, all computes, fence.
    Drain 2: barrier, all frags, all computes, fence.
    """
    var steps = List[PhaseStep]()
    # Drain 1
    steps.append(PhaseStep.fence())
    steps.append(
        PhaseStep.emit(OpRole.FRAGMENT_LOAD, exclude_lc=True, match_all=True)
    )
    steps.append(PhaseStep.barrier())
    steps.append(PhaseStep.emit(OpRole.SHARED_STORE))
    steps.append(PhaseStep.emit(OpRole.COMPUTE, match_all=True))
    steps.append(PhaseStep.fence())
    # Drain 2
    steps.append(PhaseStep.barrier())
    steps.append(PhaseStep.emit(OpRole.FRAGMENT_LOAD, match_all=True))
    steps.append(PhaseStep.emit(OpRole.COMPUTE, match_all=True))
    steps.append(PhaseStep.fence())
    return steps^


# =============================================================================
# Framework Default Derivations (reusable by PipelineSchedule implementations)
# =============================================================================


def pipe_to_list[N: Int](p: Pipe[N]) -> List[OpDesc]:
    """Convert a Pipe to a List[OpDesc]."""
    return List(length=N, fill_with=lambda (i: Int) -> OpDesc: p.ops[i])


def default_prologue(
    body: List[OpDesc],
    config: PipelineConfig,
) -> List[ScheduleEntry]:
    """Derive prologue for single-buffer pipeline from body + config.

    Delegates to single_buffer_prologue_recipe() via apply_phase_recipe().
    Sequence: fill (load→store→barrier), prefetch (load),
    loop-carried fragment, schedule fence.
    """
    return apply_phase_recipe(
        body, single_buffer_prologue_recipe(), config, Phase.PROLOGUE
    )


def default_kernel(body: List[OpDesc]) -> List[ScheduleEntry]:
    """Direct 1:1 mapping of body ops to kernel phase entries."""
    var ker = List[ScheduleEntry]()
    for i in range(len(body)):
        ker.append(
            ScheduleEntry(
                op=body[i],
                time_slot=i,
                phase=Phase.KERNEL,
                is_prefetch=False,
            )
        )
    return ker^


def default_epilogue(
    body: List[OpDesc],
    config: PipelineConfig,
) -> List[ScheduleEntry]:
    """Derive epilogue for single-buffer pipeline (2 drain passes).

    Delegates to single_buffer_epilogue_recipe() via apply_phase_recipe().
    Drain 1: fence, non-LC frags, barrier, store, all computes, fence.
    Drain 2: barrier, all frags, all computes, fence.
    """
    return apply_phase_recipe(
        body, single_buffer_epilogue_recipe(), config, Phase.EPILOGUE
    )


def _strip_drain_fuse_blocks(
    blocks: List[MMABlockSpec],
    config: PipelineConfig,
) -> List[MMABlockSpec]:
    """Strip prefetch loads, add wait_vm(0) drain, fuse trailing blocks.

    Shared epilogue transform used by derive_epilogue_from_program().
    For each half:
      1. Find first block with a prefetch load → drain start
      2. Strip prefetch loads, keep completion loads
      3. Add wait_vm(0) from drain_start onward
      4. Fuse adjacent blocks when the second has no pre-ops or loads
      5. Drop drain on the final output block (redundant)
    """
    var blocks_per_partition = config.blocks_per_partition()
    var out = List[MMABlockSpec]()

    for half in range(config.num_partitions):
        var base = half * blocks_per_partition

        # Find drain start: first block with any prefetch load.
        var drain_start = blocks_per_partition  # sentinel: no drain
        for i in range(blocks_per_partition):
            var bi = blocks[base + i]
            if bi.global_load_prefetch or bi.global_load_1_prefetch:
                drain_start = i
                break

        # Build stripped blocks with drain and fusing.
        var i = 0
        while i < blocks_per_partition:
            var b = blocks[base + i]
            var block = MMABlockSpec(
                mma=b.mma,
                pre_op_0=b.pre_op_0,
                pre_op_1=b.pre_op_1,
                global_load=(
                    b.global_load if not b.global_load_prefetch else OpDesc.none()
                ),
                global_load_1=(
                    b.global_load_1 if not b.global_load_1_prefetch else OpDesc.none()
                ),
                # Drain lgkm before each epilogue block's `pre_op` so the
                # MMA's input registers (loaded by prior block's frag-
                # loads) are committed before this block's MMA reads
                # them. The main-loop equivalent rides on
                # `inter_block_lgkm_drain` and `entry_wait_lgkm` paired
                # with `entry_wait`; the epilogue derivation strips
                # entry_wait, so emit the lgkm drain explicitly here.
                # Without it, MMA fires with stale registers on multi-
                # K-iter shapes (caught by logit verification on Olmo /
                # phi-4 / DeepSeek-V2-Lite / FLUX.2 BF16).
                entry_wait_lgkm=OpDesc.wait_lgkm[0](),
                # Carry minimal_barriers / omit_mma_set_prio flags forward
                # from the source block so the epilogue inherits the same
                # barrier+setprio layout as the main loop. Without this,
                # the epilogue blows up with one full barrier+setprio pair
                # per MMA (~5 sync ops/MMA × 8 MMAs = 40 sync ops vs
                # ~12 in the hand-tuned epilogue).
                pre_mma_barrier=b.pre_mma_barrier,
                pre_mma_set_prio=b.pre_mma_set_prio,
                post_mma_barrier=b.post_mma_barrier,
                post_mma_set_prio=b.post_mma_set_prio,
                post_barrier_lgkm=b.post_barrier_lgkm,
            )

            # Add drain in the active zone.
            if i >= drain_start:
                block.pre_sync = OpDesc.wait_vm[0]()

            # Check if next block can be fused (no pre-ops, no loads).
            if i + 1 < blocks_per_partition:
                var nx = blocks[base + i + 1]
                var nx_has_ops = nx.pre_op_0.is_present()
                var nx_has_load = (
                    nx.global_load.is_present() and not nx.global_load_prefetch
                ) or (
                    nx.global_load_1.is_present()
                    and not nx.global_load_1_prefetch
                )
                if not nx_has_ops and not nx_has_load:
                    block.fused_mma = nx.mma
                    out.append(block)
                    i += 2
                    continue

            out.append(block)
            i += 1

    # Drop drain on the last output block (redundant — already drained).
    if len(out) > 0:
        out[len(out) - 1].pre_sync = OpDesc.none()

    return out^


def _expand_epilogue_blocks(
    blocks: List[MMABlockSpec],
) -> List[ScheduleEntry]:
    """Expand stripped epilogue blocks to entries with trailing barrier."""
    var result = List[ScheduleEntry]()
    for i in range(len(blocks)):
        blocks[i].expand_to_list(result, Phase.EPILOGUE)
    result.append(
        ScheduleEntry(
            op=OpDesc.barrier(),
            time_slot=len(result),
            phase=Phase.EPILOGUE,
            is_prefetch=False,
        )
    )
    return result^


def derive_epilogue_from_program(
    program: PipelineProgram,
    config: PipelineConfig,
) -> List[ScheduleEntry]:
    """Auto-derive epilogue from a kernel program.

    Generic transformation for double-buffer schedules:
      1. Strip prefetch global loads from each block
      2. Add wait_vm(0) drain starting at the first stripped block per half
      3. Fuse adjacent blocks where the second has no pre-ops or loads
      4. Append trailing s_barrier
    """
    var stripped = _strip_drain_fuse_blocks(program.blocks, config)
    return _expand_epilogue_blocks(stripped)


# =============================================================================
# Declarative Edge Rule Tables
# =============================================================================


def _default_edge_rule() -> EdgeRule:
    """Return an EdgeRule with all predicates at their "don't care" defaults."""
    return EdgeRule(
        producer_role=OpRole.NONE,
        consumer_role=OpRole.NONE,
        dep_kind=DepKind.FLOW,
        loop_distance=0,
        match_channel=False,
        match_stage=False,
        match_subtile=False,
        use_config_match=False,
        same_half=False,
        cross_half=False,
        producer_half=-1,
        k_offset_filter=0,
        lc_producer=-1,
        lc_consumer=-1,
        producer_ordinal=-1,
        consumer_ordinal=-1,
        first_match_only=False,
    )


def double_buffer_edge_rules() -> List[EdgeRule]:
    """5 rules encoding the 4 phases of double-buffer edge derivation.

    Phase 1: Register FLOW: fragment_load → compute (same-stage half,
             config match key).
    Phase 2a: Accumulator forward: compute half 0 → compute half 1 (d=0).
    Phase 2b: Accumulator backward: compute half 1 → compute half 0 (d=1).
    Phase 3: LDS FLOW: global_load → fragment_load (same channel+stage,
             distance derived from k_offset).
    Phase 4: LDS ANTI: fragment_load → global_load (same channel+stage,
             consumer non-K_PREV only).
    """
    var rules = List[EdgeRule]()

    # Phase 1: Register FLOW — fragment_load → compute, same-stage half.
    var phase1 = _default_edge_rule()
    phase1.producer_role = OpRole.FRAGMENT_LOAD
    phase1.consumer_role = OpRole.COMPUTE
    phase1.dep_kind = DepKind.FLOW
    phase1.loop_distance = 0
    phase1.use_config_match = True
    phase1.same_half = True
    rules.append(phase1)

    # Phase 2a: Accumulator forward — compute[half 0] → compute[half 1], d=0.
    var phase2a = _default_edge_rule()
    phase2a.producer_role = OpRole.COMPUTE
    phase2a.consumer_role = OpRole.COMPUTE
    phase2a.dep_kind = DepKind.FLOW
    phase2a.loop_distance = 0
    phase2a.match_stage = True
    phase2a.match_subtile = True
    phase2a.cross_half = True
    phase2a.producer_half = 0
    rules.append(phase2a)

    # Phase 2b: Accumulator backward — compute[half 1] → compute[half 0], d=1.
    var phase2b = _default_edge_rule()
    phase2b.producer_role = OpRole.COMPUTE
    phase2b.consumer_role = OpRole.COMPUTE
    phase2b.dep_kind = DepKind.FLOW
    phase2b.loop_distance = 1
    phase2b.match_stage = True
    phase2b.match_subtile = True
    phase2b.cross_half = True
    phase2b.producer_half = 1
    rules.append(phase2b)

    # Phase 3: LDS FLOW — global_load → fragment_load, same channel+stage.
    # Loop distance is derived at rule-application time from the producer's
    # k_offset (see apply_edge_rules): K_PREV loads feed the current
    # iteration (d=0), all others feed the next iteration (d=1).
    # The -1 sentinel means "auto-derive from k_offset".
    # NOTE: This convention is specific to the AMD ping-pong kernel's
    # K-dimension tiling. A different kernel structure would need different
    # distance derivation logic.
    var phase3 = _default_edge_rule()
    phase3.producer_role = OpRole.GLOBAL_LOAD
    phase3.consumer_role = OpRole.FRAGMENT_LOAD
    phase3.dep_kind = DepKind.FLOW
    phase3.loop_distance = -1  # K_PREV → d=0, else d=1
    phase3.match_channel = True
    phase3.match_stage = True
    rules.append(phase3)

    # Phase 4: LDS ANTI — fragment_load → global_load, same channel+stage.
    # Only for non-K_PREV global loads (prefetch writes).
    # d=0 means "fragment_load must finish before the same-iteration
    # global_load overwrites its LDS slot." Cross-iteration protection
    # comes from double-buffer stage alternation (even/odd iterations use
    # different LDS regions), not from the edge distance.
    # NOTE: In general modulo scheduling theory, WAR distance would be
    # buffer_depth (d=2 for double-buffer). The d=0 formulation here is
    # correct for the AMD ping-pong pattern where stage alternation
    # provides the cross-iteration guarantee.
    var phase4 = _default_edge_rule()
    phase4.producer_role = OpRole.FRAGMENT_LOAD
    phase4.consumer_role = OpRole.GLOBAL_LOAD
    phase4.dep_kind = DepKind.ANTI
    phase4.loop_distance = 0
    phase4.match_channel = True
    phase4.match_stage = True
    phase4.k_offset_filter = 2  # consumer must be non-K_PREV
    rules.append(phase4)

    return rules^


def single_buffer_edge_rules(config: PipelineConfig) -> List[EdgeRule]:
    """8 rules encoding single-buffer structural dependencies.

    Rule 1: frag → compute (same subtile, non-lc, first match only)
    Rule 2: sync[0] → store
    Rule 3: store → sync[1]
    Rule 4: sync[1] → lc_frag
    Rule 5: frag → sync[0] (ANTI, non-lc frags)
    Rule 6: lc_frag → lc_comp (d=1)
    Rule 7: load → store (d=1)
    Rule 8: lc_frag → store (ANTI, d=1)
    """
    var rules = List[EdgeRule]()

    # Rule 1: frag → compute (same subtile, non-loop-carried, first match).
    var r1 = _default_edge_rule()
    r1.producer_role = OpRole.FRAGMENT_LOAD
    r1.consumer_role = OpRole.COMPUTE
    r1.dep_kind = DepKind.FLOW
    r1.match_subtile = True
    r1.lc_producer = 0
    r1.lc_consumer = 0
    r1.first_match_only = True
    rules.append(r1)

    # Rule 2: sync[0] → shared_store.
    var r2 = _default_edge_rule()
    r2.producer_role = OpRole.SYNC
    r2.consumer_role = OpRole.SHARED_STORE
    r2.dep_kind = DepKind.FLOW
    r2.producer_ordinal = 0
    rules.append(r2)

    # Rule 3: shared_store → sync[1].
    var r3 = _default_edge_rule()
    r3.producer_role = OpRole.SHARED_STORE
    r3.consumer_role = OpRole.SYNC
    r3.dep_kind = DepKind.FLOW
    r3.consumer_ordinal = 1
    rules.append(r3)

    # Rule 4: sync[1] → lc_frag.
    var r4 = _default_edge_rule()
    r4.producer_role = OpRole.SYNC
    r4.consumer_role = OpRole.FRAGMENT_LOAD
    r4.dep_kind = DepKind.FLOW
    r4.producer_ordinal = 1
    r4.lc_consumer = 1
    rules.append(r4)

    # Rule 5: non-lc frag → sync[0] (ANTI).
    var r5 = _default_edge_rule()
    r5.producer_role = OpRole.FRAGMENT_LOAD
    r5.consumer_role = OpRole.SYNC
    r5.dep_kind = DepKind.ANTI
    r5.lc_producer = 0
    r5.consumer_ordinal = 0
    rules.append(r5)

    # Rule 6: lc_frag → lc_comp (loop-carried, d=1).
    var r6 = _default_edge_rule()
    r6.producer_role = OpRole.FRAGMENT_LOAD
    r6.consumer_role = OpRole.COMPUTE
    r6.dep_kind = DepKind.FLOW
    r6.loop_distance = 1
    r6.lc_producer = 1
    r6.lc_consumer = 1
    rules.append(r6)

    # Rule 7: global_load → shared_store (loop-carried, d=1).
    var r7 = _default_edge_rule()
    r7.producer_role = OpRole.GLOBAL_LOAD
    r7.consumer_role = OpRole.SHARED_STORE
    r7.dep_kind = DepKind.FLOW
    r7.loop_distance = 1
    rules.append(r7)

    # Rule 8: lc_frag → shared_store (ANTI, loop-carried, d=1).
    var r8 = _default_edge_rule()
    r8.producer_role = OpRole.FRAGMENT_LOAD
    r8.consumer_role = OpRole.SHARED_STORE
    r8.dep_kind = DepKind.ANTI
    r8.loop_distance = 1
    r8.lc_producer = 1
    rules.append(r8)

    return rules^


def apply_edge_rules(
    body: List[OpDesc],
    config: PipelineConfig,
    rules: List[EdgeRule],
) -> List[DepEdge]:
    """Apply declarative edge rules to a loop body, producing DepEdge list.

    Pre-classifies ops by role, ordinal, half, and loop-carried status, then
    evaluates each rule against all (producer, consumer) pairs that match the
    rule's role requirements.  This is the generic evaluator that replaces
    the hand-coded 4-phase and 8-rule logic in derive_edges_from_ops.
    """
    var n = len(body)
    var half = n // 2

    # Pre-classify: per-role index lists.
    var by_role = List[List[Int]]()
    for _ in range(8):  # OpRole values 0-6 + NONE(255), use 8 slots
        by_role.append(List[Int]())

    @always_inline
    def _role_idx(role: OpRole) -> Int:
        return role._value if role._value < 8 else 7

    # Per-op ordinal within its role (Nth occurrence).
    var ordinals = List[Int]()
    var role_counts = List[Int]()
    for _ in range(8):
        role_counts.append(0)
    for _ in range(n):
        ordinals.append(0)

    # Loop-carried status per op.
    var is_lc = List[Bool]()
    var lc_sel = config.loop_carried.selector

    for i in range(n):
        var ri = _role_idx(body[i].role)
        by_role[ri].append(i)
        ordinals[i] = role_counts[ri]
        role_counts[ri] += 1
        is_lc.append(body[i].subtile == lc_sel)

    # Half assignment per op.
    @always_inline
    def _op_half(idx: Int) {half} -> Int:
        return 0 if idx < half else 1

    # _in_half for Phase 1: op at idx is in the half that processes stage.
    @always_inline
    def _in_half(idx: Int, stage: Int) {half} -> Bool:
        return (stage == 0) == (idx < half)

    var edges = List[DepEdge]()

    for r in range(len(rules)):
        var rule = rules[r]
        var prod_ri = _role_idx(rule.producer_role)
        var cons_ri = _role_idx(rule.consumer_role)

        for pi in range(len(by_role[prod_ri])):
            var p = by_role[prod_ri][pi]
            var p_op = body[p]

            # Producer-side filters.
            if (
                rule.producer_ordinal >= 0
                and ordinals[p] != rule.producer_ordinal
            ):
                continue
            if rule.lc_producer == 0 and is_lc[p]:
                continue
            if rule.lc_producer == 1 and not is_lc[p]:
                continue
            if rule.producer_half >= 0 and _op_half(p) != rule.producer_half:
                continue

            for ci in range(len(by_role[cons_ri])):
                var c = by_role[cons_ri][ci]
                var c_op = body[c]

                # Skip self-edges.
                if p == c:
                    continue

                # Consumer-side filters.
                if (
                    rule.consumer_ordinal >= 0
                    and ordinals[c] != rule.consumer_ordinal
                ):
                    continue
                if rule.lc_consumer == 0 and is_lc[c]:
                    continue
                if rule.lc_consumer == 1 and not is_lc[c]:
                    continue

                # Field matching predicates.
                if rule.match_channel and p_op.channel != c_op.channel:
                    continue
                if rule.match_stage and p_op.stage != c_op.stage:
                    continue
                if rule.match_subtile and p_op.subtile != c_op.subtile:
                    continue

                # Half predicates.
                if rule.same_half and _op_half(p) != _op_half(c):
                    continue
                if rule.cross_half and _op_half(p) == _op_half(c):
                    continue

                # Config match key (Phase 1 register FLOW special case).
                if rule.use_config_match:
                    if not _in_half(c, p_op.stage):
                        continue
                    if (
                        config.compute_match_key(c_op, p_op.channel)
                        != p_op.subtile
                    ):
                        continue

                # K-offset filter on consumer.
                if rule.k_offset_filter == 1:
                    if c_op.k_offset != KOffsetKind.K_PREV:
                        continue
                elif rule.k_offset_filter == 2:
                    if c_op.k_offset == KOffsetKind.K_PREV:
                        continue

                # Compute loop distance.
                var d = rule.loop_distance
                if d == -1:
                    d = 0 if p_op.k_offset == KOffsetKind.K_PREV else 1

                edges.append(DepEdge(p, c, rule.dep_kind, d))

                if rule.first_match_only:
                    break

    return edges^


def derive_edges_from_ops(
    body: List[OpDesc],
    config: PipelineConfig,
) -> List[DepEdge]:
    """Derive all dependency edges from op metadata and pipeline config.

    Delegates to `apply_edge_rules()` with the appropriate declarative rule
    table.  Edge predicates are derived from role, channel, and stage
    metadata (no kernel-specific tag knowledge).

      depth=1 (single-buffer): 8 structural rules via role-based matching
        (sync chains, loop-carried fragments, subtile-matched frag→compute)
      depth>=2 (double-buffer): 5 rules encoding 4 phases via channel-based
        matching (register FLOW, accumulator, LDS FLOW, LDS ANTI)

    This enables the PipelineSchedule trait to derive edges automatically
    from config() + build_body(), eliminating derive_edges as a required
    method.
    """
    if config.depth == 1:
        return apply_edge_rules(body, config, single_buffer_edge_rules(config))
    return apply_edge_rules(body, config, double_buffer_edge_rules())


def derive_cross_stage_rotation_edges(body: List[OpDesc]) -> List[DepEdge]:
    """Returns cross-partition FRAG→MMA + same-partition MMA→FRAG ANTI
    edges for cross-stage rotation patterns.

    The default `Phase 1` edge rule is `same_half=True`: it models
    only same-partition frag→MMA flows. For schedules that use cross-
    stage rotation (the body's sub=0 frag-loads read from the *other*
    K-partition's SMEM stage to preload that partition's leading
    quadrants), two extra edge classes are needed:

    1. **Cross-partition FLOW** (frag → MMA across partitions) so
       wait derivation and CSP scheduling see the register hand-off:
         - Partition 0 frag(ch, sub=0) → Partition 1 MMA,
           `loop_distance=0` (same outer iter).
         - Partition 1 frag(ch, sub=0) → Partition 0 MMA,
           `loop_distance=1` (loop-carried for next outer iter).

    2. **Same-partition ANTI** (MMA → FRAG within the producer's
       partition) so CSP scheduling never moves the cross-stage frag
       *before* the same-partition MMAs that still read the register
       it's about to overwrite.

    Config match key for FLOW (sub=0 only):
      - ch=0 (A): frag.subtile matches MMA.stage (m_quad).
      - ch=1 (B): frag.subtile matches MMA.subtile (n_quad).

    A frag is "cross-stage" iff `frag.stage != frag's partition index`
    (= reads from the other partition's stage).

    Args:
        body: Loop body op list to analyze.

    Returns:
        Cross-partition FLOW + same-partition ANTI edges for the
        rotation pattern.
    """
    var n = len(body)
    var partition_size = n // 2
    var edges = List[DepEdge]()

    for i in range(n):
        var f = body[i]
        if f.role != OpRole.FRAGMENT_LOAD or f.subtile != 0:
            continue
        var f_part = 0 if i < partition_size else 1
        # Cross-stage frag has stage != its own partition.
        if f.stage == f_part:
            continue

        for j in range(n):
            var m = body[j]
            if m.role != OpRole.COMPUTE:
                continue
            var m_part = 0 if j < partition_size else 1
            # config match: ch=0 → frag.sub matches MMA.stage; ch=1 →
            # frag.sub matches MMA.subtile. Both at sub=0 here, so
            # ch=0 matches MMA.stage==0; ch=1 matches MMA.subtile==0.
            var matches = (
                f.channel == 0
                and m.stage == 0
                or f.channel == 1
                and m.subtile == 0
            )
            if not matches:
                continue

            if m_part != f_part:
                # Cross-partition FLOW (forward d=0 / loop-carried d=1).
                var dist = 0 if f_part == 0 else 1
                edges.append(DepEdge.flow(i, j, dist))
            else:
                # Same-partition ANTI: cross-stage FRAG must fire
                # *after* the same-partition MMA that reads the
                # register the FRAG is about to overwrite.
                edges.append(DepEdge.anti(j, i, 0))

    return edges^


def filter_spurious_cross_stage_flow(
    raw_edges: List[DepEdge], body: List[OpDesc]
) -> List[DepEdge]:
    """Drops spurious same-partition FLOW edges that the default Phase 1
    rule emits for cross-stage frags.

    Default Phase 1 (`same_half=True`, `use_config_match=True`) says
    "same-partition MMA depends on cross-stage FRAG", but in the
    rotation pattern that same-partition MMA actually consumes the
    *previous iter's* cross-stage frag (via the loop-carried d=1
    cross-partition FLOW from `derive_cross_stage_rotation_edges`),
    not this iter's. The bogus FLOW edge combined with the same-
    partition ANTI edge creates a circular dep that deadlocks
    greedy/CSP. Drop it.

    Pairs with `derive_cross_stage_rotation_edges`: schedules that
    use cross-stage rotation should run their default-derived edges
    through this filter, then append the cross-stage edges.

    Args:
        raw_edges: Default-derived edges to filter.
        body: Loop body op list, used to look up roles and stages.

    Returns:
        `raw_edges` with the spurious cross-stage FLOW edges removed.
    """
    var n = len(body)
    var partition_size = n // 2
    var edges = List[DepEdge]()
    for e in range(len(raw_edges)):
        var edge = raw_edges[e]
        if edge.dep_kind == DepKind.FLOW:
            var p = body[edge.producer_idx]
            var c = body[edge.consumer_idx]
            if (
                p.role == OpRole.FRAGMENT_LOAD
                and p.subtile == 0
                and c.role == OpRole.COMPUTE
            ):
                var p_part = 0 if edge.producer_idx < partition_size else 1
                var c_part = 0 if edge.consumer_idx < partition_size else 1
                if p_part == c_part and p.stage != p_part:
                    continue  # spurious — drop
        edges.append(edge)
    return edges^


def default_warp_stagger(prologue_len: Int) -> Int:
    """Default warp stagger: right after prologue."""
    return prologue_len + 1


def default_warp_stagger_double_buffer(body: List[OpDesc]) -> Int:
    """Count stage-0 prefetch loads in body for double-buffer warp stagger."""
    var count = 0
    for i in range(len(body)):
        var op = body[i]
        if (
            op.role == OpRole.GLOBAL_LOAD
            and op.k_offset != KOffsetKind.K_PREV
            and op.k_offset != KOffsetKind.NONE
            and op.stage == 0
        ):
            count += 1
    return count


def derive_prologue_from_program(
    program: PipelineProgram,
    config: PipelineConfig,
    sched: ScheduleConfig = ScheduleConfig(),
    bootstrap: List[OpDesc] = List[OpDesc](),
) -> List[ScheduleEntry]:
    """Derive prologue from a finalized PipelineProgram.

    Extracts global loads from the program's first-half blocks, groups them
    by buffer stage (0 vs 1), and emits the standard prologue sequence:
      stage-0 loads at K0 → wait_vm(0) → barrier → stage-1 loads at K1 → wait_vm(0)

    When `sched.partial_prologue_drain` is True, the inter-stage
    `wait_vm(0)` + barrier and the trailing `wait_vm(0)` are skipped:
    all prefetches issue continuously, and the framework instead
    appends the schedule's `bootstrap_frags()` paired with partial
    `wait_vm(N) + barrier` drains so each bootstrap frag-load fires
    after exactly the prefetch it depends on has completed (the rest
    stay in flight for the first main-loop iter).

    Per-frag wait values are derived from cumulative prefetch
    `vm_cost`: the i-th bootstrap frag drains down to leave
    `total_vm - sum(prefetch_vm_costs[:i+1])` outstanding.

    This replaces default_prologue_double_buffer() for schedules that build
    a PipelineProgram. The advantage is that the prologue is always consistent
    with the kernel body's actual load distribution (after CSP reordering
    and redistribution), rather than scanning raw body ops independently.
    """
    var result = List[ScheduleEntry]()
    var slot = 0

    @always_inline
    def _emit(
        mut result: List[ScheduleEntry],
        op: OpDesc,
        mut slot: Int,
    ):
        result.append(
            ScheduleEntry(
                op=op,
                time_slot=slot,
                phase=Phase.PROLOGUE,
                is_prefetch=False,
            )
        )
        slot += 1

    def _vm_cost(op: OpDesc, config: PipelineConfig) -> Int:
        return config.vm_per_channel(op.channel)

    var num_blocks = len(program.blocks)

    # Helper to emit a global load as a prologue entry with the given k_offset.
    @always_inline
    def _emit_load(
        mut result: List[ScheduleEntry],
        gl: OpDesc,
        k_off: KOffsetKind,
        config: PipelineConfig,
        mut slot: Int,
    ):
        var vm = _vm_cost(gl, config)
        _emit(
            result,
            OpDesc(
                tag=gl.tag,
                stage=gl.stage,
                subtile=gl.subtile,
                k_offset=k_off,
                vm_cost=vm,
                resource=gl.resource,
                latency=gl.latency,
                role=gl.role,
            ),
            slot,
        )

    # Stage 0: collect prefetch loads with stage == 0 from ALL blocks.
    # K_PREV loads are NOT prefetches (they reference the previous iteration).
    for bi in range(num_blocks):
        var b = program.blocks[bi]
        if b.global_load_prefetch and b.global_load.stage == 0:
            _emit_load(result, b.global_load, KOffsetKind.K0, config, slot)
        if b.global_load_1_prefetch and b.global_load_1.stage == 0:
            _emit_load(result, b.global_load_1, KOffsetKind.K0, config, slot)

    if not sched.partial_prologue_drain:
        # Drain stage 0: wait for ALL stage-0 loads to land in LDS.
        _emit(result, OpDesc.wait_vm_n(0), slot)
        # Barrier (kernel inserts warp stagger before this).
        _emit(result, OpDesc.barrier(), slot)

    # Stage 1: collect prefetch loads with stage == 1 from ALL blocks.
    for bi in range(num_blocks):
        var b = program.blocks[bi]
        if b.global_load_prefetch and b.global_load.stage == 1:
            _emit_load(result, b.global_load, KOffsetKind.K1, config, slot)
        if b.global_load_1_prefetch and b.global_load_1.stage == 1:
            _emit_load(result, b.global_load_1, KOffsetKind.K1, config, slot)

    if not sched.partial_prologue_drain:
        # Drain all outstanding loads before the kernel body begins.
        _emit(result, OpDesc.wait_vm_n(0), slot)
    elif len(bootstrap) > 0:
        # Bootstrap frags + partial drains: walk the prefetches in the
        # order they were emitted above, draining one's worth of vmcnt
        # before each bootstrap frag-load. The 1st bootstrap frag
        # depends on the 1st prefetch's data, the 2nd on the 2nd, etc.
        # Barriers carry the implicit lgkm-fence so no separate
        # wait_lgkm is needed.
        var prefetch_vm_costs = List[Int]()
        for i in range(len(result)):
            if result[i].op.role == OpRole.GLOBAL_LOAD:
                prefetch_vm_costs.append(result[i].op.vm_cost)
        var total_vm = 0
        for i in range(len(prefetch_vm_costs)):
            total_vm += prefetch_vm_costs[i]
        var n_bootstrap = len(bootstrap)
        debug_assert(
            n_bootstrap <= len(prefetch_vm_costs),
            "more bootstrap frags than prefetches",
        )
        var vm_remaining = total_vm
        for i in range(n_bootstrap):
            vm_remaining -= prefetch_vm_costs[i]
            _emit(result, OpDesc.wait_vm_n(vm_remaining), slot)
            _emit(result, OpDesc.barrier(), slot)
            _emit(result, bootstrap[i], slot)

    return result^
