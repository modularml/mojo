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
"""Inline 4-wave schedule for AMD GPU matmul / implicit-GEMM conv kernels.

Per half, four mini-iters of `(LOAD, FRAG, MMA)` with **cross-stage
MMA fragment rotation** in mini-iters 3-4: those frag-loads read the
cross SMEM stage to pre-load the *next* half's leading-quadrant
fragments while this half's last MMAs are still computing, a
register-rotation trick that hides LDS-load latency between halves
(saves one frag-load worth of LDS latency on each half's first MMA).

Three load-bearing structural choices vs the default ping-pong-shaped
matmul schedule:

1. **Mini-iter declaration order**: body is `(LOAD, FRAG, MMA) * 4` per
   K-partition (12 ops × 2 partitions = 24). After
   `_construct_mma_blocks` packs into MMA-centered blocks, each block
   holds exactly 1 frag + 1 global + 1 MMA, matching `_run_iter`'s
   mini-iter shape.

2. **Cross-stage MMA fragment rotation**: minis 3-4 emit
   `MMA_LOAD_A[stage=os, sub=0]` and `MMA_LOAD_B[stage=os, sub=0]`,
   reading from the cross stage. Same-partition MMAs no longer touch
   A_quad[0]/B_quad[0] after mini-iter 2, so the cross-stage frag
   harmlessly overwrites them with data the *next* partition's leading
   MMA will consume.

3. **`SchedulingStrategy.IDENTITY` and no `double_buffer_reorder`**:
   the body order is the final order. The framework's
   `mma_block_interleave_list` matches frags to MMAs by `subtile`
   ignoring `stage`; running it on a cross-stage body would place a
   wrong-stage frag before the same-partition MMA. Bypassing the
   reorder preserves the per-mini-iter grouping.

The framework's default `double_buffer_edge_rules` include `Phase 1`
(`FRAGMENT_LOAD → COMPUTE`, `same_half=True`, `use_config_match=True`).
Cross-stage rotation is a *cross-half* register flow: half 0's mini-3-4
frags feed half 1's MMA(0,*), and half 1's feed next-iter half 0's
MMA(0,*). `derive_edges` overrides the default to append the
cross-half rules so wait derivation knows to drain frag lgkm at the
half boundary.

The framework's auto-prologue (`derive_prologue_from_program`) infers
initial frag-loads from the body's same-stage `b.fan[2]` ops; with this
body's cross-stage sub=0 frags it would read the *cross* stage at
k_base=0 (=k=BK data) instead of the same stage (=k=0 data) needed for
the first main iter's MMA[0,0]+MMA[0,1] (`A_quad[0]`+`B_quad[0]`). The
calling kernel inserts an explicit bootstrap pair after the framework
prologue to overwrite those quadrants with same-stage sub=0 data.
"""

from std.collections import List

from .pipeline_body import PipelineBody
from pipeline.types import (
    DepEdge,
    KOffsetKind,
    OpDesc,
    OpRole,
    ScheduleOps,
    annotate_ops,
)


# =============================================================================
# 4-wave op tags
# =============================================================================
# Op tags 0-5 are kernel-specific (LOAD_A/B, COMPUTE, MMA_LOAD_A/B, MMA);
# framework-infrastructure tags (BARRIER, WAIT_VM, etc.) are inherited
# from `ScheduleOps` at 128+.


@fieldwise_init
struct FourWaveOps(ScheduleOps):
    """Op tags for the 4-wave matmul kernel."""

    var value: Int
    """Integer tag identifying the op kind."""

    comptime LOAD_A = Self(0)
    """DRAM->LDS prefetch of an A sub-tile."""
    comptime LOAD_B = Self(1)
    """DRAM->LDS prefetch of a B sub-tile."""
    comptime COMPUTE = Self(2)
    """Generic compute op (unused by 4-wave; reserved tag)."""
    comptime MMA_LOAD_A = Self(3)
    """LDS->register fragment load of an A quadrant."""
    comptime MMA_LOAD_B = Self(4)
    """LDS->register fragment load of a B quadrant."""
    comptime MMA = Self(5)
    """MFMA quadrant compute op."""


# Convenience aliases for use in `_bind` dispatch and op constructors.
comptime LOAD_A = FourWaveOps.LOAD_A.value
"""Integer tag for A DRAM->LDS prefetch."""
comptime LOAD_B = FourWaveOps.LOAD_B.value
"""Integer tag for B DRAM->LDS prefetch."""
comptime COMPUTE = FourWaveOps.COMPUTE.value
"""Integer tag for the reserved generic compute op."""
comptime MMA_LOAD_A = FourWaveOps.MMA_LOAD_A.value
"""Integer tag for A LDS->register frag-load."""
comptime MMA_LOAD_B = FourWaveOps.MMA_LOAD_B.value
"""Integer tag for B LDS->register frag-load."""
comptime MMA = FourWaveOps.MMA.value
"""Integer tag for the MFMA quadrant compute op."""
from pipeline.config import (
    PipelineConfig,
    ScheduleConfig,
    SchedulingStrategy,
    TargetProfile,
    WarpStaggerRule,
)
from pipeline.geometry import KernelGeometry
from pipeline.phase_derivation import (
    derive_cross_stage_rotation_edges,
    derive_edges_from_ops,
    filter_spurious_cross_stage_flow,
)
from pipeline.program import PipelineProgram, emit_minimal_barrier_block
from pipeline.compiler import (
    PipelineSchedule,
    ScheduleCompiler,
    compile_schedule,
)
from .amd_target import mi355x_target


# =============================================================================
# Logical body — table-driven mini-iter declaration
# =============================================================================
#
# Note on terminology: the framework's `num_partitions=2` config field is
# named for the ping-pong "two warp groups" pattern, where each "half"
# is one warp group's slice of the work. 4-wave has only 4 warps total
# — there are no warp-group halves here. The "halves" in this kernel
# are actually two source K-tile *partitions*: the main loop processes
# two consecutive BK-sized K-tiles per outer iter, and each partition
# uses a different SMEM stage (so DRAM prefetches for the next iter's
# K-tile pair can overlap with this iter's MMAs). The local
# `_logical_k_partition` name reflects that; the framework's
# `num_partitions` field stays as-is (would need cross-cutting rename).


@fieldwise_init
struct MiniIterSpec(Copyable, Movable):
    """One mini-iter of the 4-wave body: (DRAM prefetch, frag-load, MMA)."""

    var load_tag: Int
    """`LOAD_A` or `LOAD_B`: channel-A vs channel-B prefetch."""
    var load_channel: Int
    """0 (A) or 1 (B). Redundant with `load_tag` but needed by the
    framework's edge derivation."""
    var load_subtile: Int
    """Which sub-tile of the source SMEM half this prefetch writes (0 or 1)."""
    var frag_tag: Int
    """`MMA_LOAD_A` or `MMA_LOAD_B`: register frag-load."""
    var frag_channel: Int
    """0 (A) or 1 (B) for the frag-load."""
    var frag_subtile: Int
    """Which sub-tile of the SMEM stage to read into the fragment register."""
    var frag_cross_stage: Bool
    """True iff the frag-load reads from the *cross* K-partition's stage
    (cross-stage rotation pre-loads the next partition's leading
    quadrants while this partition's MMAs are still issuing). False =
    read from the same stage as the prefetches in this partition."""
    var mma_m_quad: Int
    """Which `m_quad` of the warp's 2x2 quadrant grid this MMA computes."""
    var mma_n_quad: Int
    """Which `n_quad` of the warp's 2x2 quadrant grid this MMA computes."""


# 4-wave's 4 mini-iters per K-partition. Same shape across both
# partitions — only the SMEM stage flips.
#
# Cross-stage rotation: mini-3/4's frag-loads read from the *cross*
# stage to pre-load the next partition's leading quadrants
# (A_quad[0] / B_quad[0]) while this partition's last MMAs still
# issue. The next partition's MMA(0, *) then fires immediately
# without waiting on its own LDS reads.
comptime FOUR_WAVE_MINI_ITERS = [
    # Mini-1: prefetch A[curr,0], frag B[curr,1], MMA(0,0).
    MiniIterSpec(
        load_tag=LOAD_A,
        load_channel=0,
        load_subtile=0,
        frag_tag=MMA_LOAD_B,
        frag_channel=1,
        frag_subtile=1,
        frag_cross_stage=False,
        mma_m_quad=0,
        mma_n_quad=0,
    ),
    # Mini-2: prefetch B[curr,0], frag A[curr,1], MMA(0,1).
    MiniIterSpec(
        load_tag=LOAD_B,
        load_channel=1,
        load_subtile=0,
        frag_tag=MMA_LOAD_A,
        frag_channel=0,
        frag_subtile=1,
        frag_cross_stage=False,
        mma_m_quad=0,
        mma_n_quad=1,
    ),
    # Mini-3: prefetch B[curr,1], frag A[CROSS,0] (rotation), MMA(1,0).
    MiniIterSpec(
        load_tag=LOAD_B,
        load_channel=1,
        load_subtile=1,
        frag_tag=MMA_LOAD_A,
        frag_channel=0,
        frag_subtile=0,
        frag_cross_stage=True,
        mma_m_quad=1,
        mma_n_quad=0,
    ),
    # Mini-4: prefetch A[curr,1], frag B[CROSS,0] (rotation), MMA(1,1).
    MiniIterSpec(
        load_tag=LOAD_A,
        load_channel=0,
        load_subtile=1,
        frag_tag=MMA_LOAD_B,
        frag_channel=1,
        frag_subtile=0,
        frag_cross_stage=True,
        mma_m_quad=1,
        mma_n_quad=1,
    ),
]
"""4-wave's 4 mini-iters per K-partition.

Same shape across both partitions; only the SMEM stage flips.
Mini-3/4's frag-loads read from the cross stage to pre-load the
next partition's leading quadrants (A_quad[0] / B_quad[0]) while
this partition's last MMAs still issue.
"""


def _logical_k_partition[partition_idx: Int]() -> List[OpDesc]:
    """Build one K-partition (12 ops: 4 mini-iters × LOAD+FRAG+MMA) by
    stamping out the `FOUR_WAVE_MINI_ITERS` table with the right SMEM
    stage and K-offset.

    `partition_idx` ∈ {0, 1}: which K-tile of the source pair this is.
    The current SMEM stage is `partition_idx`; the cross stage is
    `1 - partition_idx`.
    """
    comptime curr_stage = partition_idx
    comptime cross_stage = 1 - partition_idx
    comptime k_offset = (
        KOffsetKind.K1 if partition_idx == 1 else KOffsetKind.K0
    )

    with PipelineBody() as b:
        comptime for i in range(len(FOUR_WAVE_MINI_ITERS)):
            comptime mini = FOUR_WAVE_MINI_ITERS[i]
            comptime frag_stage = (
                cross_stage if mini.frag_cross_stage else curr_stage
            )
            b.load(
                mini.load_tag,
                ch=mini.load_channel,
                stage=curr_stage,
                sub=mini.load_subtile,
                k=k_offset,
            )
            b.frag(
                mini.frag_tag,
                ch=mini.frag_channel,
                stage=frag_stage,
                sub=mini.frag_subtile,
            )
            b.compute(MMA, stage=mini.mma_m_quad, sub=mini.mma_n_quad)
        return b.done()


# =============================================================================
# Pipeline4Wave — preserve declaration order, no `double_buffer_reorder`
# =============================================================================


struct Pipeline4Wave[geometry: KernelGeometry](PipelineSchedule):
    """4-wave pipeline schedule with cross-stage register rotation.

    Returns the 24-op body in mini-iter order.  Framework consumes that
    order verbatim under `SchedulingStrategy.IDENTITY`, so the final
    kernel emission matches the hand-written `_run_iter` body op-for-op
    (modulo wait-count derivation, which the framework handles via
    `derive_waits_from_blocks`).

    Takes a `KernelGeometry` (kernel-shape-derived constants) as its
    only template parameter; replaces the previous `[is_fp8, lgkm_a,
    lgkm_b]` triple. `lgkm_per_load_*` is read directly from geometry,
    not threaded through `ScheduleConfig`.

    Parameters:
        geometry: Kernel-shape-derived constants (lgkm/vm costs, etc.).
    """

    var _schedule_config: ScheduleConfig
    var _target: TargetProfile

    @staticmethod
    def _default_schedule_config() -> ScheduleConfig:
        """4-wave default: minimal barriers + cross-stage rotation
        knobs. See `pipeline.strategies` for what each strategy contributes.
        """
        return ScheduleConfig.from_strategies(
            scheduling=SchedulingStrategy.IDENTITY,
            # Minimal barriers + omit set_prio: cross-stage rotation
            # provides natural inter-block sync via register flow + lgkm
            # waits; per-block s_barrier/set_prio are overhead.
            minimal_barriers=True,
            omit_mma_set_prio=True,
            # Inter-block lgkm drain: matches hand-tuned `wait_lgkm(0)`
            # between mini-1 and mini-2 of each half.
            inter_block_lgkm_drain=True,
            # Partial prologue drain: kernel emits its own partial
            # vm-drains via `bootstrap_frags`; framework prologue
            # leaves 6 of 8 prefetches in flight on entry.
            partial_prologue_drain=True,
            # Auto-derived lgkm wait values come from KernelGeometry —
            # `schedule_config()` populates these per-instance.
            auto_waits=True,
        )

    def __init__(
        out self,
        config: ScheduleConfig = Self._default_schedule_config(),
        target: TargetProfile = mi355x_target(),
    ):
        """Constructs a `Pipeline4Wave` schedule with optional overrides.

        Args:
            config: Schedule-level knobs (wait counts, barrier policy).
                Cross-stage-rotation invariants are re-applied even if
                the caller mutates them.
            target: Target hardware profile (defaults to MI355X).
        """
        var sc = config
        # Force the cross-stage-rotation invariants regardless of
        # caller-provided config (for safety; the strategy-struct path
        # already sets these but accept overrides here too).
        sc.scheduling = SchedulingStrategy.IDENTITY
        sc.minimal_barriers = True
        sc.omit_mma_set_prio = True
        sc.inter_block_lgkm_drain = True
        sc.partial_prologue_drain = True
        # 4-wave only has 4 warps, so warp-group staggering (an 8-warp
        # ping-pong optimisation) is meaningless. Override the target's
        # default to none so `warp_stagger_index` lands past-the-end of
        # the prologue — the kernel's single-loop emission is correct
        # by construction (no split needed).
        var tgt = target
        tgt.pipeline.warp_stagger = WarpStaggerRule.none()
        # Plumb lgkm_per_load_* from KernelGeometry into PipelineConfig
        # so `derive_waits_from_blocks` can read them directly without
        # going through ScheduleConfig. (`vm_per_load_*` already lives
        # on PipelineConfig — keeps the kernel-shape-derived counts
        # together.)
        tgt.pipeline.lgkm_per_load_a = Self.geometry.lgkm_per_load_a
        tgt.pipeline.lgkm_per_load_b = Self.geometry.lgkm_per_load_b
        # 4-wave's mini-3/4 frag-loads read from the cross K-partition's
        # SMEM stage (register rotation pre-loads the next partition's
        # leading quadrants while this partition's MMAs still issue),
        # so the body has frags with `stage != partition_index` by
        # design. Tell the framework's verifier to skip the strict
        # "fragment loads in half h must use stage h" invariant
        # (`_verify_stage_consistency`) — the cross-stage edges are
        # already enforced separately via
        # `derive_cross_stage_rotation_edges` +
        # `filter_spurious_cross_stage_flow`.
        tgt.pipeline.cross_stage_rotation = True
        self._schedule_config = sc
        self._target = tgt

    def config(self) -> PipelineConfig:
        """Returns the underlying target `PipelineConfig`.

        Returns:
            The pipeline config from the target profile.
        """
        return self._target.pipeline

    def declare_ops(self) -> List[OpDesc]:
        """Declares the logical 24-op body across both K-partitions.

        Returns:
            The full list of `OpDesc`s in mini-iter order.
        """
        var ops = _logical_k_partition[0]()
        ops.extend(_logical_k_partition[1]())
        return ops^

    def build_body(self) -> List[OpDesc]:
        """Annotates logical ops with target cost model.

        Skips `double_buffer_reorder`: the body is already in mini-iter
        order and `mma_block_interleave_list` would break cross-stage
        frag placement (it matches frags to MMAs by `subtile` only,
        ignoring the frag's `stage` field, so it cannot distinguish a
        same-stage sub=0 frag from a cross-stage sub=0 frag).

        Returns:
            The annotated list of `OpDesc`s ready for compilation.
        """
        var logical = self.declare_ops()
        return annotate_ops(logical, self._target.cost_model)

    def bootstrap_frags(self) -> List[OpDesc]:
        """Bootstraps A_quad[0] + B_quad[0] for the first main-loop iter.

        The body's sub=0 frags read the *cross* stage as part of the
        cross-stage rotation pattern. For the very first main iter
        there's no previous half to have populated those quadrants, so
        we explicitly emit two same-stage sub=0 frag-loads here. The
        framework pairs each with a partial `wait_vm` drain (and a
        barrier) so each fires after exactly the prefetch it depends
        on completes; the remaining 6 prefetches stay in flight.

        Returns:
            A 2-element list of A/B sub=0 frag-load `OpDesc`s.
        """
        var ops = List[OpDesc]()
        ops.append(
            OpDesc(
                tag=MMA_LOAD_A,
                stage=0,
                subtile=0,
                channel=0,
                role=OpRole.FRAGMENT_LOAD,
            )
        )
        ops.append(
            OpDesc(
                tag=MMA_LOAD_B,
                stage=0,
                subtile=0,
                channel=1,
                role=OpRole.FRAGMENT_LOAD,
            )
        )
        return ops^

    def derive_edges(self, body: List[OpDesc]) -> List[DepEdge]:
        """Derives dependency edges with cross-stage rotation fixups.

        Runs the framework's default edge derivation, filters out the
        spurious same-partition FLOW edges that Phase 1 emits for
        cross-stage frags, then appends the cross-partition FLOW + same-
        partition ANTI edges. Both helpers live in
        `pipeline.phase_derivation` and are reusable across cross-stage
        rotation schedules.

        Args:
            body: The annotated op list returned by `build_body()`.

        Returns:
            The complete list of dependency edges for wait derivation.
        """
        var raw = derive_edges_from_ops(body, self.config())
        var edges = filter_spurious_cross_stage_flow(raw, body)
        var cross = derive_cross_stage_rotation_edges(body)
        for i in range(len(cross)):
            edges.append(cross[i])
        return edges^

    def schedule_config(self) -> ScheduleConfig:
        """Returns the schedule-level configuration for this pipeline.

        Returns:
            The `ScheduleConfig` set up in `__init__`.
        """
        # `lgkm_per_load_*` no longer flows through `ScheduleConfig` —
        # `derive_waits_from_blocks` reads it directly from
        # `PipelineConfig.lgkm_per_load_a/b`, populated in `__init__`
        # from `KernelGeometry`.
        return self._schedule_config

    def build_explicit_blocks(
        self,
        body: List[OpDesc],
        program: PipelineProgram,
    ) -> List[List[OpDesc]]:
        """Emits each block via `emit_minimal_barrier_block`.

        Same shape as the hand-tuned `_run_iter`'s mini-iters: optional
        sched_barrier wrap + entry waits, frag/load section, optional
        sync-group wrap + pre_sync/barrier/post-barrier-lgkm, then the
        MMA.

        Wait values, frag/load assignments, and barrier flags come
        from `program.blocks[i]` (populated by
        `_construct_mma_blocks` + auto-wait derivation). The schedule's
        only contribution is choosing the helper: the per-block ops
        are entirely framework-derived.

        Args:
            body: The annotated op list (unused here; ops come from
                `program.blocks`).
            program: The compiled pipeline program containing per-block
                wait counts and barrier flags.

        Returns:
            One inner list per block, each holding the ops emitted by
            `emit_minimal_barrier_block`.
        """
        var explicit = List[List[OpDesc]]()
        for i in range(len(program.blocks)):
            var block = program.blocks[i]
            explicit.append(
                emit_minimal_barrier_block(
                    block, wrap_waits=block.wrap_waits_with_sched_barrier
                )
            )
        return explicit^


def build_schedule[
    geometry: KernelGeometry,
](
    config: ScheduleConfig = Pipeline4Wave[geometry]._default_schedule_config(),
    target: TargetProfile = mi355x_target(),
) -> ScheduleCompiler:
    """Compiles the 4-wave pipeline schedule.

    Takes a `KernelGeometry` (kernel-shape-derived constants) as its
    only template parameter; the framework reads `lgkm_per_load_*`
    from it directly. Use as a comptime value; the kernel reads
    entries via comptime for.

    Parameters:
        geometry: Kernel-shape-derived constants (lgkm/vm costs, etc.).

    Args:
        config: Schedule-level overrides; defaults to the cross-stage
            rotation invariants.
        target: Target hardware profile; defaults to MI355X.

    Returns:
        A `ScheduleCompiler` with prologue, kernel, and epilogue lists.
    """
    var sc = compile_schedule(Pipeline4Wave[geometry](config, target))
    sc.lgkm_per_load_a = geometry.lgkm_per_load_a
    sc.lgkm_per_load_b = geometry.lgkm_per_load_b
    return sc^
