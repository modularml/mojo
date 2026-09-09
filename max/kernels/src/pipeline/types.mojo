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
"""Core types for the pipeline scheduling framework.

Contains ResourceKind, ScheduleOps, _Ops, KOffsetKind, Phase, OpRole,
OpCost, TargetCostModel, OpDesc, DepKind, DepEdge, EdgeRule, and
the annotate_ops helper.
"""

from std.collections import Array
from std.collections import List

# =============================================================================
# Resource Kind (maps to hardware execution units)
# =============================================================================


@fieldwise_init
struct ResourceKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Hardware resource that an operation occupies.

    Each operation is assigned to a resource unit with a finite capacity.
    Two operations on the same resource at the same time slot contend.
    The resource also determines which wait counter tracks the operation:

      GLOBAL_MEM → vmcnt    (buffer_load instructions)
      LDS        → lgkmcnt  (ds_read / LDS instructions)
      MMA_UNIT   → MFMA     (matrix fused multiply-add)
      VALU       → VALU     (vector ALU: softmax, exp2, reductions)
      SCALAR     → SALU     (barriers, priority hints, fences)
      NONE       → sentinel (no-op / not applicable)

    On CDNA architectures, MMA_UNIT and VALU are independent execution
    units that can issue simultaneously. The scheduler models this by
    tracking separate free times for each, allowing MMA and VALU ops
    to overlap.
    """

    var _value: Int

    comptime GLOBAL_MEM = Self(0)  # Global→LDS loads (tracked by vmcnt)
    comptime LDS = Self(1)  # LDS→register loads (tracked by lgkmcnt)
    comptime MMA_UNIT = Self(2)  # MFMA execution unit
    comptime SCALAR = Self(3)  # SALU: barriers, setprio, schedule_barrier
    comptime VALU = Self(4)  # Vector ALU (parallel with MMA on CDNA)
    """Vector ALU: parallel with MMA on CDNA architectures."""
    comptime NONE = Self(255)  # No-op sentinel

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.GLOBAL_MEM:
            writer.write("GLOBAL_MEM")
        elif self == Self.LDS:
            writer.write("LDS")
        elif self == Self.MMA_UNIT:
            writer.write("MMA_UNIT")
        elif self == Self.VALU:
            writer.write("VALU")
        elif self == Self.SCALAR:
            writer.write("SCALAR")
        else:
            writer.write("NONE")


# =============================================================================
# Schedule Ops — extensible enum via trait with comptime defaults
# =============================================================================
# The ScheduleOps trait defines the framework's infrastructure ops with fixed
# default values in the 128+ range. Each kernel defines a concrete struct
# conforming to ScheduleOps (or a trait extending it), adding kernel-specific
# data ops in the 0-127 range. This prevents tag collisions by construction:
# framework ops are always >= 128, kernel ops are always < 128.
#
# Usage:
#   trait ScheduleOps defines base infrastructure ops as Self-typed constants.
#   Each kernel struct conforms to ScheduleOps, inheriting framework ops and
#   adding kernel-specific ops. The .value field gives the Int tag for OpDesc.
#
#   Example:
#     struct PingPongOps(ScheduleOps, TrivialRegisterPassable):
#         var value: Int
#         ...
#         comptime LOAD_A = Self(0)     # kernel-specific
#         # PingPongOps.BARRIER inherited from trait as Self(128)


trait ScheduleOps(Equatable):
    """Base 'enum' for pipeline schedule operations.

    Framework infrastructure ops have fixed defaults in the 128+ range.
    Kernel-specific ops should be defined in the 0-127 range by conforming
    structs. Each constant is a Self-typed value wrapping an Int tag.
    """

    def __init__(out self, value: Int):
        ...

    comptime BARRIER = Self(128)
    """Workgroup-wide barrier (e.g. `s_barrier` on AMD)."""
    comptime WAIT_VM = Self(129)
    """Wait on outstanding global-memory loads (e.g. `s_waitcnt vmcnt`)."""
    comptime WAIT_LGKM = Self(130)
    """Wait on outstanding LDS / scalar-memory ops (e.g. `s_waitcnt lgkmcnt`)."""
    comptime SET_PRIO = Self(131)
    """Wave priority hint (e.g. `s_setprio`)."""
    comptime SCHEDULE_BARRIER = Self(132)
    """LLVM scheduling-barrier hint (`schedule_barrier`)."""
    comptime SCHED_GROUP_BARRIER = Self(133)
    """LLVM scheduling-group barrier hint (`s_sched_group_barrier`)."""
    comptime NONE = Self(255)
    """Sentinel value for an absent or unspecified op."""


@fieldwise_init
struct _Ops(ScheduleOps):
    """Framework-internal concrete ScheduleOps for accessing default tag values.

    Kernels should not use this; define your own conforming struct instead.
    """

    var value: Int


# =============================================================================
# K-Offset Descriptor
# =============================================================================


@fieldwise_init
struct KOffsetKind(Equatable, ImplicitlyCopyable, Movable):
    """Describes how to compute the K offset for a load operation.

    K offsets are relative to the loop variable `k` in the main loop,
    expressed as multiples of BK.
    """

    var bk_multiple: Int

    comptime K0 = Self(0)  # k + 0*BK (current stage 0)
    comptime K1 = Self(1)  # k + 1*BK (current stage 1)
    comptime K_NEXT = Self(2)  # k + 2*BK (next iteration's stage 0)
    comptime K_PREV = Self(3)  # k - 1*BK (previous iteration's stage 1)
    comptime NONE = Self(255)  # Not applicable (non-load ops)

    @always_inline
    def signed_bk_multiple(self) -> Int:
        """Return the signed BK multiplier (K_PREV=3 maps to -1)."""
        return -1 if self.bk_multiple == 3 else self.bk_multiple


# =============================================================================
# Schedule Phase
# =============================================================================


@fieldwise_init
struct Phase(Equatable, ImplicitlyCopyable, Movable):
    """Which phase of the pipeline an entry belongs to."""

    var _value: Int

    comptime PROLOGUE = Self(0)
    comptime KERNEL = Self(1)
    comptime EPILOGUE = Self(2)


# =============================================================================
# Operation Role
# =============================================================================


@fieldwise_init
struct OpRole(Equatable, ImplicitlyCopyable, Movable):
    """Role of an operation in the pipeline data flow.

    Classifies ops for automatic phase derivation (prologue, epilogue, etc.).
    Set at OpDesc construction time by factory methods.
    """

    var _value: Int

    comptime GLOBAL_LOAD = Self(0)  # GMEM → registers (prefetchable)
    comptime SHARED_STORE = Self(1)  # registers → SMEM
    comptime FRAGMENT_LOAD = Self(2)  # SMEM → registers
    comptime COMPUTE = Self(3)  # MMA compute
    comptime SYNC = Self(4)  # barrier
    comptime FENCE = Self(5)  # schedule barrier (compiler hint)
    comptime VALU_COMPUTE = Self(6)  # VALU compute (softmax, exp2, reductions)
    """Vector ALU compute (softmax, exp2, reductions)."""
    comptime NONE = Self(255)  # sentinel


# =============================================================================
# Target Cost Model
# =============================================================================


struct OpCost(ImplicitlyCopyable, Movable):
    """Hardware cost annotation for a single operation kind.

    Maps an operation tag to the hardware resource it occupies, its latency
    in cycles, its data-flow role, and (optionally) the VGPR liveness it
    induces. Provided by a TargetCostModel.
    """

    var resource: ResourceKind
    """Hardware execution unit."""
    var latency: Int
    """Latency in cycles."""
    var role: OpRole
    """Pipeline data-flow role."""
    var vgpr_def: Int
    """VGPRs this op brings into scope (new live register values)."""
    var vgpr_kill: Int
    """VGPRs this op releases (last use of some register buffer)."""

    @always_inline
    def __init__(
        out self,
        resource: ResourceKind,
        latency: Int,
        role: OpRole,
        *,
        vgpr_def: Int = 0,
        vgpr_kill: Int = 0,
    ):
        """Constructs an `OpCost`.

        Args:
            resource: Hardware execution unit.
            latency: Latency in cycles.
            role: Pipeline data-flow role.
            vgpr_def: VGPRs the op brings into scope.
            vgpr_kill: VGPRs the op releases.
        """
        self.resource = resource
        self.latency = latency
        self.role = role
        self.vgpr_def = vgpr_def
        self.vgpr_kill = vgpr_kill

    @staticmethod
    def none() -> OpCost:
        """Sentinel for unregistered op tags."""
        return OpCost(ResourceKind.NONE, 0, OpRole.NONE)


struct TargetCostModel(ImplicitlyCopyable, Movable):
    """Maps op tags to hardware cost annotations.

    Separates the algorithm (what ops exist) from the target (how much they
    cost). The algorithm declares logical ops with buffer metadata (tag,
    stage, subtile, channel, k_offset). The cost model supplies resource
    kind, latency, and role: the constraints that drive scheduling.

    Op tags 0-127 are kernel-specific (registered via set_cost).
    Op tags 128+ are framework infrastructure ops and are not looked up
    in the cost model; they carry their own annotations.

    Usage:

    ```mojo
    var model = TargetCostModel()
    model.set_cost(0, OpCost(ResourceKind.GLOBAL_MEM, 200, OpRole.GLOBAL_LOAD))
    var annotated = annotate_ops(logical_ops, model)
    ```
    """

    var _costs: Array[OpCost, 128]

    def __init__(out self):
        self._costs = Array[OpCost, 128](fill=OpCost.none())

    def __init__(out self, *, copy: Self):
        self._costs = copy._costs.copy()

    def set_cost(mut self, tag: Int, cost: OpCost):
        """Register a cost annotation for a kernel op tag (0-127).

        Args:
            tag: Kernel op tag to register the cost for (0-127).
            cost: Hardware cost annotation to associate with the tag.
        """
        debug_assert(tag < 128, "only kernel op tags (0-127) can be registered")
        self._costs[tag] = cost

    def get_cost(self, tag: Int) -> OpCost:
        """Look up the cost for a kernel op tag.

        Args:
            tag: Kernel op tag to look up (0-127).
        """
        debug_assert(tag < 128, "only kernel op tags (0-127) have costs")
        return self._costs[tag]


# =============================================================================
# Operation Descriptor
# =============================================================================


struct OpDesc(ImplicitlyCopyable, Movable):
    """Describes a single operation in the pipeline schedule."""

    var tag: Int
    """The type of operation (kernel-specific, used by _emit dispatch)."""
    var stage: Int
    """Buffer stage index (0 or 1 for double-buffering)."""
    var subtile: Int
    """Subtile index within the stage (0 or 1)."""
    var k_offset: KOffsetKind
    """How to compute the K dimension offset for loads."""
    var vm_cost: Int
    """Number of vmcnt (global load) ops this produces."""
    var lgkm_cost: Int
    """Number of lgkmcnt (LDS) ops this produces."""
    var wait_value: Int
    """For WAIT_VM/WAIT_LGKM ops, the count to wait for."""
    var resource: ResourceKind
    """Hardware execution unit (GLOBAL_MEM, LDS, MMA_UNIT, SCALAR)."""
    var latency: Int
    """Estimated execution latency in cycles."""
    var role: OpRole
    """Pipeline data-flow role (GLOBAL_LOAD, FRAGMENT_LOAD, etc.)."""
    var channel: Int
    """Data path identifier for edge derivation. Ops on the same channel
    share a buffer (e.g., 0=A matrix, 1=B matrix). -1 = none."""
    var vgpr_def: Int
    """VGPRs this op brings into scope (new live register values)."""
    var vgpr_kill: Int
    """VGPRs this op releases (last use of some register buffer)."""

    @always_inline
    def __init__(
        out self,
        *,
        tag: Int,
        stage: Int = 0,
        subtile: Int = 0,
        k_offset: KOffsetKind = KOffsetKind.NONE,
        vm_cost: Int = 0,
        lgkm_cost: Int = 0,
        wait_value: Int = 0,
        resource: ResourceKind = ResourceKind.NONE,
        latency: Int = 0,
        role: OpRole = OpRole.NONE,
        channel: Int = -1,
        vgpr_def: Int = 0,
        vgpr_kill: Int = 0,
    ):
        self.tag = tag
        self.stage = stage
        self.subtile = subtile
        self.k_offset = k_offset
        self.vm_cost = vm_cost
        self.lgkm_cost = lgkm_cost
        self.wait_value = wait_value
        self.resource = resource
        self.latency = latency
        self.role = role
        self.channel = channel
        self.vgpr_def = vgpr_def
        self.vgpr_kill = vgpr_kill

    @always_inline
    def is_present(self) -> Bool:
        """True if this is a real op (not the NONE sentinel)."""
        return self.tag != _Ops.NONE.value

    # --- Kernel op factory ---

    @staticmethod
    @always_inline
    def op(
        tag: Int,
        resource: ResourceKind,
        latency: Int,
        role: OpRole,
        *,
        channel: Int = -1,
        stage: Int = 0,
        subtile: Int = 0,
        k_offset: KOffsetKind = KOffsetKind.NONE,
        vm_cost: Int = 0,
        lgkm_cost: Int = 0,
        wait_value: Int = 0,
        vgpr_def: Int = 0,
        vgpr_kill: Int = 0,
    ) -> OpDesc:
        """Construct an OpDesc with all metadata inline.

        All scheduling metadata (tag, resource, latency, role, channel) is
        specified directly. Per-instance fields (stage, subtile, k_offset)
        are keyword arguments.

        Args:
            tag: Kernel-specific op tag (0-127) used by `_emit` dispatch.
            resource: Hardware execution unit (`GLOBAL_MEM`, `LDS`,
                `MMA_UNIT`, or `SCALAR`).
            latency: Estimated execution latency in cycles.
            role: Pipeline data-flow role (`GLOBAL_LOAD`,
                `FRAGMENT_LOAD`, etc.).
            channel: Data path identifier for edge derivation. Ops on the
                same channel share a buffer (defaults to -1).
            stage: Buffer stage index (0 or 1 for double-buffering)
                (defaults to 0).
            subtile: Subtile index within the stage (defaults to 0).
            k_offset: How to compute the K dimension offset for loads
                (defaults to `KOffsetKind.NONE`).
            vm_cost: Number of `vmcnt` (global load) ops this produces
                (defaults to 0).
            lgkm_cost: Number of `lgkmcnt` (LDS) ops this produces
                (defaults to 0).
            wait_value: For `WAIT_VM`/`WAIT_LGKM` ops, the count to wait
                for (defaults to 0).
            vgpr_def: VGPRs this op brings into scope as new live register
                values (defaults to 0).
            vgpr_kill: VGPRs this op releases as last use of some register
                buffer (defaults to 0).
        """
        return OpDesc(
            tag=tag,
            stage=stage,
            subtile=subtile,
            k_offset=k_offset,
            vm_cost=vm_cost,
            lgkm_cost=lgkm_cost,
            wait_value=wait_value,
            resource=resource,
            latency=latency,
            role=role,
            channel=channel,
            vgpr_def=vgpr_def,
            vgpr_kill=vgpr_kill,
        )

    # --- Logical op factory (for use with TargetCostModel) ---

    @staticmethod
    @always_inline
    def logical(
        tag: Int,
        *,
        channel: Int = -1,
        stage: Int = 0,
        subtile: Int = 0,
        k_offset: KOffsetKind = KOffsetKind.NONE,
    ) -> OpDesc:
        """Declare a logical op: buffer metadata only, no hardware costs.

        Resource, latency, and role are left as sentinel values (NONE/0).
        Call annotate_ops() with a TargetCostModel to fill them in before
        scheduling. This separates the algorithm (what ops exist) from the
        target (how expensive they are).

        Args:
            tag: Kernel-specific op tag (0-127) used by `_emit` dispatch.
            channel: Data path identifier for edge derivation. Ops on the
                same channel share a buffer (defaults to -1).
            stage: Buffer stage index (0 or 1 for double-buffering)
                (defaults to 0).
            subtile: Subtile index within the stage (defaults to 0).
            k_offset: How to compute the K dimension offset for loads
                (defaults to `KOffsetKind.NONE`).
        """
        return OpDesc(
            tag=tag,
            stage=stage,
            subtile=subtile,
            k_offset=k_offset,
            channel=channel,
        )

    # --- Infrastructure factory methods (used by framework phase derivation) ---

    @staticmethod
    def barrier() -> OpDesc:
        return OpDesc(
            tag=_Ops.BARRIER.value,
            resource=ResourceKind.SCALAR,
            role=OpRole.SYNC,
        )

    @staticmethod
    def wait_vm[count: Int]() -> OpDesc:
        return OpDesc(
            tag=_Ops.WAIT_VM.value,
            wait_value=count,
            resource=ResourceKind.SCALAR,
            role=OpRole.SYNC,
        )

    @staticmethod
    def wait_vm_n(count: Int) -> OpDesc:
        """Runtime-parameterized wait_vm (for ScheduleConfig-driven waits).

        Args:
            count: Number of outstanding `vmcnt` (global load) ops to wait
                for.
        """
        return OpDesc(
            tag=_Ops.WAIT_VM.value,
            wait_value=count,
            resource=ResourceKind.SCALAR,
            role=OpRole.SYNC,
        )

    @staticmethod
    def wait_lgkm[count: Int]() -> OpDesc:
        return OpDesc(
            tag=_Ops.WAIT_LGKM.value,
            wait_value=count,
            resource=ResourceKind.SCALAR,
            role=OpRole.SYNC,
        )

    @staticmethod
    def wait_lgkm_n(count: Int) -> OpDesc:
        """Runtime-parameterized wait_lgkm (for ScheduleConfig-driven waits).

        Args:
            count: Number of outstanding `lgkmcnt` (LDS / scalar-memory) ops
                to wait for.
        """
        return OpDesc(
            tag=_Ops.WAIT_LGKM.value,
            wait_value=count,
            resource=ResourceKind.SCALAR,
            role=OpRole.SYNC,
        )

    @staticmethod
    def set_prio[priority: Int]() -> OpDesc:
        """Priority hint: s_setprio[priority]().

        Parameters:
            priority: Wave priority value passed to the `s_setprio`
                instruction.
        """
        return OpDesc(
            tag=_Ops.SET_PRIO.value,
            wait_value=priority,
            resource=ResourceKind.SCALAR,
            role=OpRole.FENCE,
        )

    @staticmethod
    def schedule_barrier() -> OpDesc:
        """Compiler scheduling fence."""
        return OpDesc(
            tag=_Ops.SCHEDULE_BARRIER.value,
            resource=ResourceKind.SCALAR,
            role=OpRole.FENCE,
        )

    @staticmethod
    def none() -> OpDesc:
        """No-op sentinel (used for optional MMA block fields)."""
        return OpDesc(tag=_Ops.NONE.value)

    @staticmethod
    def sched_group_barrier[mask: Int, count: Int]() -> OpDesc:
        """Schedule_group_barrier hint. Mask encoded in subtile, count in
        wait_value.

        Parameters:
            mask: Scheduling-group mask, encoded into the `subtile` field.
            count: Barrier count, encoded into the `wait_value` field.
        """
        return OpDesc(
            tag=_Ops.SCHED_GROUP_BARRIER.value,
            subtile=mask,
            wait_value=count,
            resource=ResourceKind.SCALAR,
            role=OpRole.FENCE,
        )


def annotate_ops(
    ops: List[OpDesc],
    model: TargetCostModel,
) -> List[OpDesc]:
    """Apply a target cost model to logical ops.

    For each op with a kernel-specific tag (< 128), looks up the cost model
    and fills in resource, latency, and role. Infrastructure ops (tag >= 128)
    are passed through unchanged; they carry their own annotations.

    This is the bridge between the algorithm (which declares logical ops via
    OpDesc.logical()) and the scheduler (which needs resource/latency data).

    Args:
        ops: List of logical ops declared via `OpDesc.logical()`.
        model: Target cost model supplying resource, latency, and role for
            kernel op tags.
    """
    var result = List[OpDesc]()
    for i in range(len(ops)):
        var op = ops[i]
        if op.tag < 128:
            var cost = model.get_cost(op.tag)
            op.resource = cost.resource
            op.latency = cost.latency
            op.role = cost.role
            op.vgpr_def = cost.vgpr_def
            op.vgpr_kill = cost.vgpr_kill
        result.append(op)
    return result^


# =============================================================================
# Dependency Edge
# =============================================================================


@fieldwise_init
struct DepKind(Equatable, ImplicitlyCopyable, Movable):
    """Type of dependency between operations."""

    var _value: Int

    comptime FLOW = Self(0)  # RAW: producer writes, consumer reads
    comptime ANTI = Self(1)  # WAR: consumer writes, producer reads
    comptime OUTPUT = Self(2)  # WAW: both write same resource


struct DepEdge(ImplicitlyCopyable, Movable):
    """A dependency between two schedule entries.

    Used for C2 (data dependence) validation: the producer entry must
    complete before the consumer entry can execute.

    From the modulo scheduling paper, each edge in the Loop Dependency Graph
    has a loop distance `d`:
      - d=0: same-iteration dependency (producer and consumer in same iter)
      - d>=1: loop-carried dependency (consumer reads data from `d` iters ago)

    The C2 constraint with loop distance is:

    ```text
    τ(consumer) - τ(producer) >= latency(producer) - T * d
    ```

    For d=0 (same iteration), this simplifies to the existing check:

    ```text
    time_slot(consumer) > time_slot(producer)
    ```
    """

    var producer_idx: Int
    """Index of the producing entry in its phase's entry list."""
    var consumer_idx: Int
    """Index of the consuming entry in its phase's entry list."""
    var dep_kind: DepKind
    """Type of dependency (`FLOW`, `ANTI`, or `OUTPUT`)."""
    var loop_distance: Int
    """Loop iterations between producer and consumer (`0` = same iteration)."""

    @always_inline
    def __init__(
        out self,
        producer_idx: Int,
        consumer_idx: Int,
        dep_kind: DepKind,
        loop_distance: Int = 0,
    ):
        self.producer_idx = producer_idx
        self.consumer_idx = consumer_idx
        self.dep_kind = dep_kind
        self.loop_distance = loop_distance

    @staticmethod
    def flow(producer: Int, consumer: Int, loop_distance: Int = 0) -> DepEdge:
        return DepEdge(producer, consumer, DepKind.FLOW, loop_distance)

    @staticmethod
    def anti(producer: Int, consumer: Int, loop_distance: Int = 0) -> DepEdge:
        return DepEdge(producer, consumer, DepKind.ANTI, loop_distance)


# =============================================================================
# Declarative Edge Rules
# =============================================================================


@fieldwise_init
struct EdgeRule(ImplicitlyCopyable, Movable):
    """Declarative edge derivation rule.

    Each rule describes a class of dependency edges: for every (producer,
    consumer) pair whose `OpDesc` fields satisfy the predicates, emit a
    `DepEdge` with the given kind and loop distance.

    The evaluator (`apply_edge_rules`) pre-classifies ops by role, then for
    each rule scans only relevant (producer_role, consumer_role) pairs and
    checks the predicate fields. This replaces the hand-coded 4-phase
    double-buffer logic and 8-rule single-buffer logic in
    `derive_edges_from_ops` with inspectable data.

    Fields fall into five groups:

    1. **Core**: producer/consumer roles, dependency kind, loop distance.
       `loop_distance = -1` means "derive from producer.k_offset":
       K_PREV → d=0 (current-iteration load), otherwise d=1 (prefetch).

    2. **Field matching**: require same channel / stage / subtile between
       the producer and consumer ops. `use_config_match` activates the
       `PipelineConfig.compute_match_key()` logic (Phase 1 register-FLOW).

    3. **Half predicates** (double-buffer): `same_half` / `cross_half` /
       `producer_half` constrain which half each op lives in.

    4. **K-offset filter**: 0=any, 1=K_PREV only, 2=non-K_PREV only.
       Applied to the *consumer* for LDS-ANTI rules, to the *producer* for
       LDS-FLOW distance derivation.

    5. **Single-buffer predicates**: `lc_producer`/`lc_consumer` (loop-
       carried status), `producer_ordinal`/`consumer_ordinal` (Nth op of
       that role), `first_match_only` (break after first consumer match).
    """

    # --- Core ---
    var producer_role: OpRole
    """Producer op role for this rule."""
    var consumer_role: OpRole
    """Consumer op role for this rule."""
    var dep_kind: DepKind
    """Dependency kind (`FLOW`, `ANTI`, or `OUTPUT`)."""
    var loop_distance: Int
    """Loop iterations between producer and consumer (`0`, `1`, or `-1`).
    `-1` derives from `producer.k_offset`: `K_PREV` → `d=0`
    (current-iteration load), otherwise `d=1` (prefetch)."""

    # --- Field matching predicates ---
    var match_channel: Bool
    """Require the producer and consumer to share the same channel."""
    var match_stage: Bool
    """Require the producer and consumer to share the same stage."""
    var match_subtile: Bool
    """Require the producer and consumer to share the same subtile."""
    var use_config_match: Bool
    """Use `PipelineConfig.compute_match_key()` (Phase 1 register-FLOW)."""

    # --- Positional predicates (double-buffer halves) ---
    var same_half: Bool
    """Require both ops to live in the same half."""
    var cross_half: Bool
    """Require the ops to live in different halves."""
    var producer_half: Int
    """Producer half filter (`-1` = any, `0` = first half, `1` = second half)."""

    # --- K-offset filter ---
    var k_offset_filter: Int
    """K-offset filter (`0` = any, `1` = `K_PREV` only, `2` = non-`K_PREV`
    only). Applied to the consumer for LDS-ANTI rules and to the producer
    for LDS-FLOW distance derivation."""

    # --- Loop-carried filter (single-buffer) ---
    var lc_producer: Int
    """Loop-carried filter for the producer (`-1` = any, `0` = non-lc,
    `1` = lc)."""
    var lc_consumer: Int
    """Loop-carried filter for the consumer (`-1` = any, `0` = non-lc,
    `1` = lc)."""

    # --- Ordinal filter (single-buffer sync ordering) ---
    var producer_ordinal: Int
    """Producer ordinal filter (`-1` = any, `N` = Nth occurrence of
    `producer_role`)."""
    var consumer_ordinal: Int
    """Consumer ordinal filter (`-1` = any, `N` = Nth occurrence of
    `consumer_role`)."""

    # --- Matching behavior ---
    var first_match_only: Bool
    """Break after the first consumer match per producer."""


# =============================================================================
# Prefetch predicate
# =============================================================================


def _is_prefetch(op: OpDesc) -> Bool:
    """K-offset-based prefetch: K0/K1 loads are initial-fill data (used by
    the prologue to identify which loads pre-fill LDS). K_PREV/NONE are
    not prefetch; they reference the previous iteration.

    Note: this is the PROLOGUE meaning of prefetch. For kernel wait_vm
    derivation, derive_waits_from_blocks uses stage-based completion
    detection (a load is completion if stage != half).
    """
    return (
        op.is_present()
        and op.k_offset != KOffsetKind.K_PREV
        and op.k_offset != KOffsetKind.NONE
    )
