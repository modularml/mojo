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
"""Pipeline configuration structs: LoopCarriedSpec, BlockSizing, FragOrder,
SchedulingStrategy, ScheduleConfig, WarpStaggerRule, PipelineConfig, and
TargetProfile.
"""

from .types import OpDesc, OpRole, TargetCostModel

# =============================================================================
# Pipeline Configuration
# =============================================================================


@fieldwise_init
struct LoopCarriedSpec(ImplicitlyCopyable, Movable):
    """Which ops are loop-carried (loaded at end of iter, consumed at start).

    For single-buffer pipelines, fragment[0] is typically loop-carried:
    it's loaded at the end of iteration N and consumed by compute[0] at
    the start of iteration N+1.
    """

    var role: OpRole  # Which role is loop-carried (e.g., FRAGMENT_LOAD)
    var selector: Int  # Which subtile (e.g., 0 for fragment[0])

    @staticmethod
    def fragment_zero() -> Self:
        """Fragment[0] is loop-carried (default matmul pattern)."""
        return Self(role=OpRole.FRAGMENT_LOAD, selector=0)

    @staticmethod
    def none() -> Self:
        """No loop-carried ops (double-buffered pipelines)."""
        return Self(role=OpRole.NONE, selector=0)


@fieldwise_init
struct BlockSizing(ImplicitlyCopyable, Movable):
    """Target op count per MMA block for latency hiding.

    Heavy blocks (introducing a new M-tile row) get more ops to fill the
    MMA latency window. Light blocks (continuation of same M-tile) get
    fewer ops since only one new fragment is needed.
    """

    var heavy: Int  # Blocks introducing a new M-tile row
    var light: Int  # Continuation blocks (same M-tile)
    var max_globals: Int  # Max global loads per block (0=unlimited)

    @staticmethod
    def default_2x2() -> Self:
        """2x2 MMA grid default: 4 ops for new M-tile, 2 for continuation."""
        return Self(heavy=4, light=2, max_globals=0)

    @staticmethod
    def uniform(n: Int) -> Self:
        """All blocks get the same target size."""
        return Self(heavy=n, light=n, max_globals=0)


@fieldwise_init
struct FragOrder(ImplicitlyCopyable, Movable):
    """Fragment load ordering within an MMA block."""

    var b_before_a: Bool  # True: B-frag first. False: A-frag first.

    @staticmethod
    def b_first() -> Self:
        """B-fragment before A-fragment."""
        return Self(b_before_a=True)

    @staticmethod
    def a_first() -> Self:
        """A-fragment before B-fragment."""
        return Self(b_before_a=False)


@fieldwise_init
struct SchedulingStrategy(Equatable, ImplicitlyCopyable, Movable):
    """Scheduling strategy for pipeline op ordering.

    IDENTITY: Declaration order (no scheduling).
    GREEDY: Greedy greedy_schedule() heuristic.
    CSP: Backtracking CSP solver (provably optimal).
    """

    var _value: Int

    comptime IDENTITY = Self(0)
    comptime GREEDY = Self(1)
    comptime CSP = Self(2)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value


struct ScheduleConfig(ImplicitlyCopyable, Movable):
    """Tunable parameters for schedule generation.

    Controls the structural decisions in program builders:
    scheduling strategy, barrier placement, and drain behavior.

    The default configuration auto-derives wait counts from the program
    structure (Halide-inspired: declare intent, derive consequences).
    Manual wait overrides are available for testing and experimentation
    but should not be needed for correct operation.
    """

    var scheduling: SchedulingStrategy
    """The strategy for op ordering (`IDENTITY`, `GREEDY`, or `CSP`)."""
    var sched_barrier_mask: Int
    """The bitmask of blocks that get trailing `schedule_barrier` fences.
    Default: `0b01010101` (blocks 0, 2, 4, 6)."""
    var auto_waits: Bool
    """Whether to auto-derive wait counts from schedule order."""
    var drain_lgkm_mask: Int
    """The per-block bitmask for selective LDS drains."""
    var auto_drain: Bool
    """Whether to auto-derive the drain mask from channel analysis."""
    var lds_contention_penalty: Int
    """The CSP solver penalty for LDS port overlap."""
    # Manual wait overrides (only used when auto_waits=False).
    var wait_lgkm_first: Int
    """The manual `wait_lgkm(N)` override (used when `auto_waits=False`)."""
    var wait_vm_last: Int
    """The manual `wait_vm(N)` override (used when `auto_waits=False`)."""
    var lgkm_per_load_a: Int
    """The number of `lgkmcnt` ops per channel-A load (for wait derivation)."""
    var lgkm_per_load_b: Int
    """The number of `lgkmcnt` ops per channel-B load (for wait derivation)."""
    var lgkm_after_last: Bool
    """Whether to insert `wait_lgkm(0)` after the last block barrier."""
    var minimal_barriers: Bool
    """Whether to suppress per-block `s_barrier`s and `set_prio` pairs and
    emit `s_barrier` only at top-of-half and the first cross-stage block."""
    var omit_mma_set_prio: Bool
    """When `minimal_barriers=True`, whether to drop the pre-MMA
    `s_setprio[1]` entirely so the LLVM register allocator can reuse VGPRs
    across it."""
    var max_vgpr: Int
    """The hint for the cost model on the kernel's VGPR budget. Default is
    effectively unlimited."""
    # Per-block emission-shape knobs (default off → ping-pong layout).
    var global_before_frag: Bool
    """Whether to swap the in-block emission order of global loads and
    fragment loads. Default emits frags first then prefetches."""
    var barrier_before_pre_ops: Bool
    """Whether to move the per-block `pre_sync` + barrier section before the
    frag/prefetch section instead of between prefetch and MMA."""
    var inter_block_lgkm_drain: Bool
    """Whether to populate `entry_wait_lgkm` on non-top, non-cross-stage
    blocks with `wait_lgkm(0)` so an inter-mini LDS drain fires between
    consecutive same-half MMAs."""
    # When True, every contiguous wait/barrier group inside a block is
    # wrapped with `schedule_barrier()` on both sides — matching the
    # density of `_sched_barrier()` calls in hand-tuned 4-wave inline.
    # `schedule_barrier` doesn't emit GPU instructions; it's an LLVM
    # machine-scheduler fence that prevents reordering across the wrap
    # points, preserving the (load + frag + mma) cluster shape around
    # the wait/barrier boundaries. Default False.
    var wrap_waits_with_sched_barrier: Bool
    """Whether to wrap each contiguous wait/barrier group with
    `schedule_barrier()` on both sides to fence the LLVM machine scheduler."""
    # Prologue emission strategy. Default False uses
    # `derive_prologue_from_program`'s standard layout: stage-0 prefetches
    # → wait_vm(0) → barrier → stage-1 prefetches → wait_vm(0). True drops
    # both `wait_vm(0)` and the inter-stage barrier, leaving the prefetches
    # in flight on entry to the kernel — the kernel is then responsible
    # for emitting its own partial drains (e.g. `wait_vm(N1)` + barrier +
    # bootstrap frag_a + `wait_vm(N2)` + barrier + bootstrap frag_b) so
    # the main loop's first iter starts with most prefetches still in
    # flight rather than fully drained.
    var partial_prologue_drain: Bool
    """Whether to skip the framework prologue's `wait_vm(0)` drains and
    inter-stage barrier so prefetches stay in flight on entry to the kernel."""

    @always_inline
    def __init__(
        out self,
        *,
        scheduling: SchedulingStrategy = SchedulingStrategy.IDENTITY,
        sched_barrier_mask: Int = 0b01010101,
        auto_waits: Bool = True,
        drain_lgkm_mask: Int = 0,
        auto_drain: Bool = False,
        lds_contention_penalty: Int = 0,
        wait_lgkm_first: Int = 8,
        wait_vm_last: Int = 6,
        lgkm_per_load_a: Int = 0,
        lgkm_per_load_b: Int = 0,
        lgkm_after_last: Bool = False,
        minimal_barriers: Bool = False,
        omit_mma_set_prio: Bool = False,
        max_vgpr: Int = 999999,
        global_before_frag: Bool = False,
        barrier_before_pre_ops: Bool = False,
        inter_block_lgkm_drain: Bool = False,
        partial_prologue_drain: Bool = False,
        wrap_waits_with_sched_barrier: Bool = False,
    ):
        self.scheduling = scheduling
        self.sched_barrier_mask = sched_barrier_mask
        self.auto_waits = auto_waits
        self.drain_lgkm_mask = drain_lgkm_mask
        self.auto_drain = auto_drain
        self.lds_contention_penalty = lds_contention_penalty
        self.wait_lgkm_first = wait_lgkm_first
        self.wait_vm_last = wait_vm_last
        self.lgkm_per_load_a = lgkm_per_load_a
        self.lgkm_per_load_b = lgkm_per_load_b
        self.lgkm_after_last = lgkm_after_last
        self.minimal_barriers = minimal_barriers
        self.omit_mma_set_prio = omit_mma_set_prio
        self.max_vgpr = max_vgpr
        self.global_before_frag = global_before_frag
        self.barrier_before_pre_ops = barrier_before_pre_ops
        self.inter_block_lgkm_drain = inter_block_lgkm_drain
        self.partial_prologue_drain = partial_prologue_drain
        self.wrap_waits_with_sched_barrier = wrap_waits_with_sched_barrier

    @staticmethod
    def from_strategies(
        *,
        scheduling: SchedulingStrategy = SchedulingStrategy.IDENTITY,
        max_vgpr: Int = 999999,
        lds_contention_penalty: Int = 0,
        # The strategy structs from `pipeline.strategies` are passed
        # as `Bool`-shaped views to avoid a circular import from
        # `pipeline.config`. Callers pass the strategy struct fields
        # directly; see `pipeline.strategies` for the named factories.
        # Barrier strategy:
        minimal_barriers: Bool = False,
        omit_mma_set_prio: Bool = False,
        sched_barrier_mask: Int = 0b01010101,
        wrap_waits_with_sched_barrier: Bool = False,
        barrier_before_pre_ops: Bool = False,
        # Wait strategy:
        auto_waits: Bool = True,
        drain_lgkm_mask: Int = 0,
        auto_drain: Bool = False,
        wait_lgkm_first: Int = 8,
        wait_vm_last: Int = 6,
        lgkm_after_last: Bool = False,
        inter_block_lgkm_drain: Bool = False,
        partial_prologue_drain: Bool = False,
        # Load strategy:
        global_before_frag: Bool = False,
        lgkm_per_load_a: Int = 0,
        lgkm_per_load_b: Int = 0,
    ) -> Self:
        """Constructs a `ScheduleConfig` from grouped strategy values.

        Equivalent to the flat-field constructor but groups related
        flags by phase (barrier / wait / load). `pipeline.strategies`
        provides named factories (`BarrierStrategy.minimal_no_set_prio`
        etc.) that callers can spread into this constructor.

        Existing flat-field callers continue to work unchanged.

        Args:
            scheduling: CSP solver scheduling strategy.
            max_vgpr: VGPR budget hint for the cost model.
            lds_contention_penalty: Penalty for LDS port overlap.
            minimal_barriers: Suppress per-block `s_barrier`s and
                `set_prio` pairs.
            omit_mma_set_prio: Drop the pre-MMA `s_setprio[1]` when
                `minimal_barriers=True`.
            sched_barrier_mask: Bitmask of which blocks get trailing
                `schedule_barrier` fences.
            wrap_waits_with_sched_barrier: Wrap each contiguous
                wait/barrier group with `schedule_barrier`.
            barrier_before_pre_ops: Move pre_sync + barrier ahead of
                the frag/global section.
            auto_waits: Auto-derive wait counts from program structure.
            drain_lgkm_mask: Per-block bitmask for selective LDS drains.
            auto_drain: Auto-derive `drain_lgkm_mask` from channel
                analysis.
            wait_lgkm_first: Manual `wait_lgkm` override.
            wait_vm_last: Manual `wait_vm` override for the last block.
            lgkm_after_last: Insert `wait_lgkm(0)` after the last
                block's barrier.
            inter_block_lgkm_drain: Emit `wait_lgkm(0)` at non-top,
                non-cross interior block starts.
            partial_prologue_drain: Skip `wait_vm(0)` drains in the
                framework prologue.
            global_before_frag: Emit globals before frags in each block.
            lgkm_per_load_a: `lgkmcnt` entries per channel-A frag-load.
            lgkm_per_load_b: `lgkmcnt` entries per channel-B frag-load.

        Returns:
            A fully populated `ScheduleConfig`.
        """
        return Self(
            scheduling=scheduling,
            sched_barrier_mask=sched_barrier_mask,
            auto_waits=auto_waits,
            drain_lgkm_mask=drain_lgkm_mask,
            auto_drain=auto_drain,
            lds_contention_penalty=lds_contention_penalty,
            wait_lgkm_first=wait_lgkm_first,
            wait_vm_last=wait_vm_last,
            lgkm_per_load_a=lgkm_per_load_a,
            lgkm_per_load_b=lgkm_per_load_b,
            lgkm_after_last=lgkm_after_last,
            minimal_barriers=minimal_barriers,
            omit_mma_set_prio=omit_mma_set_prio,
            max_vgpr=max_vgpr,
            global_before_frag=global_before_frag,
            barrier_before_pre_ops=barrier_before_pre_ops,
            inter_block_lgkm_drain=inter_block_lgkm_drain,
            partial_prologue_drain=partial_prologue_drain,
            wrap_waits_with_sched_barrier=wrap_waits_with_sched_barrier,
        )


@fieldwise_init
struct WarpStaggerRule(ImplicitlyCopyable, Movable):
    """Declarative warp stagger configuration.

    Controls how warp groups are staggered in the prologue to hide
    latency. The stagger index determines how many prologue ops G0
    executes before G1 starts.

    Modes:
      - enabled=False: no stagger (single-buffer default)
      - enabled=True, compute_from_body=True: derive stagger by counting
        stage-0 prefetch loads in the body (double-buffer default)
      - enabled=True, compute_from_body=False: use fixed_amount directly
    """

    var enabled: Bool  # Whether to apply warp stagger
    var compute_from_body: Bool  # Derive amount from body analysis
    var fixed_amount: Int  # Used when compute_from_body is False

    @staticmethod
    def none() -> Self:
        """No stagger (single-buffer default)."""
        return Self(enabled=False, compute_from_body=False, fixed_amount=0)

    @staticmethod
    def auto() -> Self:
        """Auto-derive stagger from body (double-buffer default)."""
        return Self(enabled=True, compute_from_body=True, fixed_amount=0)

    @staticmethod
    def fixed(amount: Int) -> Self:
        """Fixed stagger amount."""
        return Self(enabled=True, compute_from_body=False, fixed_amount=amount)


struct PipelineConfig(ImplicitlyCopyable, Movable):
    """Declarative pipeline strategy.

    Captures all the knowledge needed to transform a logical loop body
    into a pipelined schedule: buffer depth, prefetch distance, loop-carried
    edges, MMA block sizing, and hardware model.

    Platform-specific factories (e.g., mi355x_double_buffer() in
    amd_target.mojo) provide tuned configurations.
    """

    # --- Buffer strategy ---
    var depth: Int  # 1 = single-buffer, 2 = double-buffer
    var prefetch: Int  # Iterations ahead for DRAM loads (typically 1)

    # --- Phase derivation ---
    var drain_passes: Int  # Epilogue drain iterations
    var prologue_fill: Int  # Extra load iterations in prologue (depth - 1)

    # --- Execution ordering ---
    var loop_carried: LoopCarriedSpec  # Ops crossing iteration boundaries
    var block_sizing: BlockSizing  # MMA block op targets
    var frag_order: FragOrder  # Fragment ordering within a block

    # --- MMA grid ---
    var m_mmas: Int  # M-dimension MMA tiles (rows in spatial grid)
    var n_mmas: Int  # N-dimension MMA tiles (cols in spatial grid)
    var num_partitions: Int  # Number of warp groups (1 for single, 2 for ping-pong)

    # --- Hardware model ---
    var mma_serial: Bool  # MMA unit is serial (capacity 1)
    var mma_latency: Int  # MMA latency in cycles
    var vm_per_load_a: Int  # vmcnt ops per global load (channel 0 / A)
    var vm_per_load_b: Int  # vmcnt ops per global load (channel 1 / B)
    # Kernel-geometry-derived lgkm counts (channels 0 / 1). Lives here
    # alongside `vm_per_load_*` because both fall out of kernel
    # geometry, not scheduling tuning. Schedules populate these from
    # `KernelGeometry.lgkm_per_load_a/b` at construction time —
    # `derive_waits_from_blocks` reads from here directly so the values
    # don't need to flow through `ScheduleConfig`. Default 0 = unset
    # (legacy callers fall back to `ScheduleConfig.lgkm_per_load_*`).
    var lgkm_per_load_a: Int
    """Kernel-geometry-derived `lgkmcnt` entries per channel-A frag-load.
    `0` falls back to `ScheduleConfig.lgkm_per_load_a`."""
    var lgkm_per_load_b: Int
    """Kernel-geometry-derived `lgkmcnt` entries per channel-B frag-load.
    `0` falls back to `ScheduleConfig.lgkm_per_load_b`."""

    # --- Channel configuration ---
    # For register FLOW edges: which compute field does each channel's
    # fragment subtile match against?
    #   0 = match compute.stage (row), 1 = match compute.subtile (col)
    # channel 0 (A): frag.subtile matches compute.stage (row)
    # channel 1 (B): frag.subtile matches compute.subtile (col)
    var ch0_match_field: Int  # 0 = stage, 1 = subtile
    var ch1_match_field: Int  # 0 = stage, 1 = subtile

    # --- Warp stagger ---
    var warp_stagger: WarpStaggerRule  # Warp group stagger configuration

    # --- Cross-stage rotation opt-out ---
    var cross_stage_rotation: Bool
    """True when the schedule intentionally pre-loads the next
    K-partition's leading-quadrant fragments from the *other* SMEM
    stage (4-wave's mini-3/4 register rotation). Relaxes the
    "fragment loads in half h must use stage h" invariant in
    `program_builder._verify_stage_consistency`: same-stage and
    cross-stage frags coexist by design when this is True. Default
    False keeps the strict check active for ping-pong and other
    schedules that don't rotate."""

    @always_inline
    def __init__(
        out self,
        *,
        depth: Int,
        prefetch: Int,
        drain_passes: Int,
        prologue_fill: Int,
        loop_carried: LoopCarriedSpec,
        block_sizing: BlockSizing,
        frag_order: FragOrder,
        m_mmas: Int,
        n_mmas: Int,
        num_partitions: Int,
        mma_serial: Bool,
        mma_latency: Int,
        vm_per_load_a: Int,
        vm_per_load_b: Int,
        ch0_match_field: Int,
        ch1_match_field: Int,
        warp_stagger: WarpStaggerRule,
        lgkm_per_load_a: Int = 0,
        lgkm_per_load_b: Int = 0,
        cross_stage_rotation: Bool = False,
    ):
        """Constructs a `PipelineConfig` from individual fields.

        `lgkm_per_load_a` / `lgkm_per_load_b` are optional kernel-geometry
        defaults; pass `0` to fall back to `ScheduleConfig.lgkm_per_load_*`.
        See the field-level docstrings on `PipelineConfig` for per-field
        meanings.

        Args:
            depth: Pipeline buffer depth (1 = single, 2 = double).
            prefetch: DRAM-prefetch distance, typically 1.
            drain_passes: Epilogue drain iteration count.
            prologue_fill: Extra load iterations in the prologue.
            loop_carried: Ops crossing loop iteration boundaries.
            block_sizing: MMA block op targets.
            frag_order: Fragment ordering within a block.
            m_mmas: M-dimension MMA tile count.
            n_mmas: N-dimension MMA tile count.
            num_partitions: Number of warp groups.
            mma_serial: Whether the MMA unit is serial.
            mma_latency: MMA latency in cycles.
            vm_per_load_a: `vmcnt` ops per channel-A global load.
            vm_per_load_b: `vmcnt` ops per channel-B global load.
            ch0_match_field: Channel-0 register-flow match field.
            ch1_match_field: Channel-1 register-flow match field.
            warp_stagger: Warp-group stagger configuration.
            lgkm_per_load_a: `lgkmcnt` ops per channel-A frag-load
                (`0` = fall back to `ScheduleConfig`).
            lgkm_per_load_b: `lgkmcnt` ops per channel-B frag-load
                (`0` = fall back to `ScheduleConfig`).
            cross_stage_rotation: Set to True for schedules that
                intentionally pre-load the next K-partition's
                leading-quadrant fragments from the cross stage
                (4-wave's mini-3/4 rotation). Relaxes the strict
                stage-consistency invariant in
                `_verify_stage_consistency`.
        """
        self.depth = depth
        self.prefetch = prefetch
        self.drain_passes = drain_passes
        self.prologue_fill = prologue_fill
        self.loop_carried = loop_carried
        self.block_sizing = block_sizing
        self.frag_order = frag_order
        self.m_mmas = m_mmas
        self.n_mmas = n_mmas
        self.num_partitions = num_partitions
        self.mma_serial = mma_serial
        self.mma_latency = mma_latency
        self.vm_per_load_a = vm_per_load_a
        self.vm_per_load_b = vm_per_load_b
        self.ch0_match_field = ch0_match_field
        self.ch1_match_field = ch1_match_field
        self.warp_stagger = warp_stagger
        self.lgkm_per_load_a = lgkm_per_load_a
        self.lgkm_per_load_b = lgkm_per_load_b
        self.cross_stage_rotation = cross_stage_rotation

    # --- Derived counts (no magic numbers) ---

    def mmas_per_partition(self) -> Int:
        """MMA ops per warp group: m_mmas × n_mmas."""
        return self.m_mmas * self.n_mmas

    def globals_per_partition(self) -> Int:
        """Global loads per warp group: m_mmas + n_mmas (A + B tiles)."""
        return self.m_mmas + self.n_mmas

    def frags_per_partition(self) -> Int:
        """Fragment loads per warp group: m_mmas + n_mmas (A + B frags)."""
        return self.m_mmas + self.n_mmas

    def ops_per_partition(self) -> Int:
        """Total ops per warp group."""
        return (
            self.globals_per_partition()
            + self.frags_per_partition()
            + self.mmas_per_partition()
        )

    def total_ops(self) -> Int:
        """Total ops across all warp groups."""
        return self.num_partitions * self.ops_per_partition()

    def blocks_per_partition(self) -> Int:
        """MMA blocks per warp group (one block per MMA)."""
        return self.mmas_per_partition()

    def total_blocks(self) -> Int:
        """Total MMA blocks."""
        return self.num_partitions * self.blocks_per_partition()

    def compute_match_key(self, compute_op: OpDesc, channel: Int) -> Int:
        """Extract the compute field that a fragment on `channel` matches.

        For channel 0 (A): returns compute.stage (row).
        For channel 1 (B): returns compute.subtile (col).
        """
        var field = (
            self.ch0_match_field if channel == 0 else self.ch1_match_field
        )
        return compute_op.stage if field == 0 else compute_op.subtile

    def vm_per_channel(self, channel: Int) -> Int:
        """Return vmcnt cost for a global load on the given channel."""
        return self.vm_per_load_a if channel == 0 else self.vm_per_load_b

    def lgkm_per_channel(self, channel: Int) -> Int:
        """Returns the `lgkmcnt` cost for a fragment load on the given
        channel.

        Reads from `lgkm_per_load_a/b` set on the config (typically
        populated from `KernelGeometry`). Returns 0 if unset; callers
        should fall back to `ScheduleConfig.lgkm_per_load_*` for
        legacy schedules.

        Args:
            channel: 0 for channel A, anything else for channel B.

        Returns:
            `lgkmcnt` entries per fragment load on `channel`, or 0 if
            unset.
        """
        return self.lgkm_per_load_a if channel == 0 else self.lgkm_per_load_b

    def total_edges(self) -> Int:
        """Total dependency edges for double-buffer pipeline.

        Four phases of edges connect ops within and across iterations:

        - reg_flow: fragment_load → compute (register FLOW).
          Each half has 2 channels (A, B), each channel's frag feeds
          m*n compute ops.
        - accum: compute → compute (accumulator forwarding).
          m*n accumulator tiles forwarded between halves, twice
          (half0→half1 at d=0, half1→half0 at d=1).
        - lds_flow: global_load → fragment_load (LDS FLOW).
          Each half has g global loads, each feeds one frag per buffer
          stage (×2 for double-buffering).
        - lds_anti: fragment_load → global_load (LDS ANTI).
          Prevents a prefetch write from overwriting data a frag still
          needs. 2*g frags total minus 1 (last frag has no successor
          load), doubled for both channels.
        """
        var m = self.m_mmas
        var n = self.n_mmas
        var h = self.num_partitions
        var g = self.globals_per_partition()
        var reg_flow = h * 2 * m * n
        var accum = m * n * 2
        var lds_flow = h * g * 2
        var lds_anti = (2 * g - 1) * 2
        return reg_flow + accum + lds_flow + lds_anti


# =============================================================================
# Target Profile — unified hardware description
# =============================================================================


struct TargetProfile(ImplicitlyCopyable, Movable):
    """Unified hardware target description.

    The algorithm declares WHAT ops exist (logical op table). The
    `TargetProfile` describes HOW the hardware executes them. Platform-
    specific factories (e.g., `mi355x_target()` in `amd_target.mojo`) provide
    both from a single call (no redundant configuration).

    Usage:

    ```mojo
    # Platform-specific factory (from amd_target):
    comptime target = mi355x_target()
    var annotated = annotate_ops(logical_ops, target.cost_model)
    var config = target.pipeline
    ```
    """

    var cost_model: TargetCostModel
    """Per-op costs (resource, latency, role)."""
    var pipeline: PipelineConfig
    """Pipeline structure (depth, MMA grid, buffer strategy)."""

    def __init__(
        out self,
        cost_model: TargetCostModel,
        pipeline: PipelineConfig,
    ):
        self.cost_model = cost_model
        self.pipeline = pipeline
