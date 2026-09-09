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
"""Ping-pong schedule for AMD GPU matmul kernels.

Kernel-specific schedule definition that builds on the generic framework
in schedule_framework.mojo. The DeclarativeSchedule struct encapsulates the
complete constraint-based pipeline: logical op declaration, cost model
annotation, automatic edge derivation, and CSP-based optimal scheduling.

The schedule has 2 halves of 4 MMA blocks each, with double-buffered
global→LDS→register data flow and warp staggering for latency hiding.
"""

from std.collections import List

from .pipeline_body import PipelineBody
from pipeline.types import (
    KOffsetKind,
    OpDesc,
    OpRole,
    ResourceKind,
    ScheduleOps,
    TargetCostModel,
    annotate_ops,
)
from pipeline.config import (
    PipelineConfig,
    ScheduleConfig,
    SchedulingStrategy,
    TargetProfile,
)
from pipeline.program_builder import double_buffer_reorder
from pipeline.compiler import (
    PipelineSchedule,
    ScheduleCompiler,
    compile_schedule,
)
from .amd_target import mi355x_target


# =============================================================================
# Ping-pong op tags — extends ScheduleOps with kernel-specific data ops (0-5)
# =============================================================================


@fieldwise_init
struct PingPongOps(ScheduleOps):
    """Op tags for the ping-pong double-buffered matmul kernel.

    Kernel-specific data ops in the 0-127 range; framework infrastructure ops
    inherited from ScheduleOps at 128+.
    """

    var value: Int

    comptime LOAD_A = Self(0)
    comptime LOAD_B = Self(1)
    comptime COMPUTE = Self(2)
    comptime MMA_LOAD_A = Self(3)
    comptime MMA_LOAD_B = Self(4)
    comptime MMA = Self(5)


# Convenience aliases (Int values) for use in op factories and external imports.
comptime LOAD_A = PingPongOps.LOAD_A.value
comptime LOAD_B = PingPongOps.LOAD_B.value
comptime COMPUTE = PingPongOps.COMPUTE.value
comptime MMA_LOAD_A = PingPongOps.MMA_LOAD_A.value
comptime MMA_LOAD_B = PingPongOps.MMA_LOAD_B.value
comptime MMA = PingPongOps.MMA.value

# =============================================================================
# Ping-pong op factories
# =============================================================================

comptime _GMEM = ResourceKind.GLOBAL_MEM
comptime _LDS = ResourceKind.LDS
comptime _MMA_RES = ResourceKind.MMA_UNIT
comptime _GL = OpRole.GLOBAL_LOAD
comptime _FL = OpRole.FRAGMENT_LOAD
comptime _C = OpRole.COMPUTE


def _load_a(
    *,
    stage: Int = 0,
    subtile: Int = 0,
    k_offset: KOffsetKind = KOffsetKind.NONE,
    vm_cost: Int = 0,
) -> OpDesc:
    return OpDesc.op(
        LOAD_A,
        _GMEM,
        200,
        _GL,
        channel=0,
        stage=stage,
        subtile=subtile,
        k_offset=k_offset,
        vm_cost=vm_cost,
    )


def _load_b(
    *,
    stage: Int = 0,
    subtile: Int = 0,
    k_offset: KOffsetKind = KOffsetKind.NONE,
    vm_cost: Int = 0,
) -> OpDesc:
    return OpDesc.op(
        LOAD_B,
        _GMEM,
        200,
        _GL,
        channel=1,
        stage=stage,
        subtile=subtile,
        k_offset=k_offset,
        vm_cost=vm_cost,
    )


def _mma_load_a(*, stage: Int = 0, subtile: Int = 0) -> OpDesc:
    return OpDesc.op(
        MMA_LOAD_A, _LDS, 20, _FL, channel=0, stage=stage, subtile=subtile
    )


def _mma_load_b(*, stage: Int = 0, subtile: Int = 0) -> OpDesc:
    return OpDesc.op(
        MMA_LOAD_B, _LDS, 20, _FL, channel=1, stage=stage, subtile=subtile
    )


def _mma(*, stage: Int = 0, subtile: Int = 0) -> OpDesc:
    return OpDesc.op(MMA, _MMA_RES, 16, _C, stage=stage, subtile=subtile)


# =============================================================================
# DeclarativeSchedule — constraint-based automatic scheduling
# =============================================================================


def _logical_half[h: Int]() -> List[OpDesc]:
    """Logical op table for one half of the ping-pong algorithm.

    Pure data: declares WHAT ops exist with buffer metadata only.
    No resource kinds, no latencies, no roles: those come from the
    TargetCostModel.
    """
    comptime s = h
    comptime os = 1 - h
    comptime k_off = KOffsetKind.K1 if h == 1 else KOffsetKind.K0
    comptime k_special = KOffsetKind.K_PREV if h == 0 else KOffsetKind.K0

    with PipelineBody() as b:
        # Global loads: DRAM → LDS (4 per half)
        b.load(LOAD_A, ch=0, stage=os, sub=1, k=k_special)  # completion
        b.load(LOAD_A, ch=0, stage=s, sub=0, k=k_off)
        b.load(LOAD_B, ch=1, stage=s, sub=0, k=k_off)
        b.load(LOAD_B, ch=1, stage=s, sub=1, k=k_off)
        # Fragment loads: LDS → registers (2×A + 2×B per half)
        b.fan[2](MMA_LOAD_A, ch=0, stage=s)
        b.fan[2](MMA_LOAD_B, ch=1, stage=s)
        # MMA compute: 2×2 tile grid (4 per half)
        b.grid[2, 2](MMA)
        return b.done()


struct DeclarativeSchedule[
    is_fp8: Bool,
    lgkm_a: Int,
    lgkm_b: Int,
](PipelineSchedule):
    """Constraint-based pipeline: algorithm declares ops, target supplies costs.

    The algorithm specifies WHAT ops exist: just the tag and buffer metadata
    (stage, subtile, channel, k_offset). No resource kinds, no latencies,
    no roles.

    The TargetProfile specifies HOW the hardware executes them: per-op costs
    (resource, latency, role) via the cost model, and pipeline structure
    (depth, MMA grid, buffer strategy) via the pipeline config. One factory
    call (e.g., mi355x_target()) provides everything.

    The framework then:
      1. Annotates logical ops with the cost model (annotate_ops)
      2. Reorders into MMA-block-interleaved execution order (double_buffer_reorder)
      3. Derives optimal scheduling via CSP backtracking (optimal_schedule_with_halves)
      4. Derives wait counts from schedule order (derive_wait_counts)

    Usage:
        comptime schedule = build_schedule[is_fp8, lgkm_a, lgkm_b]()
    """

    var _schedule_config: ScheduleConfig
    var _target: TargetProfile

    def __init__(
        out self,
        config: ScheduleConfig = ScheduleConfig(
            scheduling=SchedulingStrategy.CSP, auto_waits=True
        ),
        target: TargetProfile = mi355x_target(),
    ):
        self._schedule_config = config
        self._target = target

    # Backward-compatible constructor: accepts separate PipelineConfig + TargetCostModel.
    def __init__(
        out self,
        config: ScheduleConfig,
        hw_config: PipelineConfig,
        cost_model: TargetCostModel,
    ):
        self._schedule_config = config
        self._target = TargetProfile(cost_model=cost_model, pipeline=hw_config)

    # -----------------------------------------------------------------
    # PipelineSchedule trait implementation
    # -----------------------------------------------------------------

    def config(self) -> PipelineConfig:
        return self._target.pipeline

    def declare_ops(self) -> List[OpDesc]:
        """Algorithm description: WHAT ops exist, with buffer metadata only.

        Returns 24 logical ops (2 halves x 12 ops) from the ping-pong op
        table. No resource kinds, no latencies, no roles: those come from
        the TargetProfile's cost model. See _logical_half() for the table.
        """
        var ops = _logical_half[0]()
        ops.extend(_logical_half[1]())
        return ops^

    def build_body(self) -> List[OpDesc]:
        """Apply cost model to logical ops, then reorder for execution.

        1. declare_ops() → logical ops (no hardware costs)
        2. annotate_ops() → ops with resource/latency/role from cost model
        3. double_buffer_reorder() → MMA-block-interleaved execution order
        """
        var logical = self.declare_ops()
        var annotated = annotate_ops(logical, self._target.cost_model)
        return double_buffer_reorder(annotated, self.config())

    def schedule_config(self) -> ScheduleConfig:
        """Return tuning knobs with lgkm counts from type params."""
        var sched = self._schedule_config
        sched.lgkm_per_load_a = Self.lgkm_a
        sched.lgkm_per_load_b = Self.lgkm_b
        return sched


def build_schedule[
    is_fp8: Bool,
    lgkm_a: Int,
    lgkm_b: Int,
](
    config: ScheduleConfig = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP, auto_waits=True
    ),
    target: TargetProfile = mi355x_target(),
) -> ScheduleCompiler:
    """Compile the declarative constraint-based schedule.

    The caller provides:
      - ScheduleConfig: tuning knobs (scheduling strategy, auto_waits)
      - TargetProfile: unified hardware target (cost model + pipeline config)

    The framework automatically derives the execution ordering and wait
    counts from op constraints. This is the constraint-based alternative
    to build_interleaved_schedule().
    """
    var sc = compile_schedule(
        DeclarativeSchedule[is_fp8, lgkm_a, lgkm_b](config, target)
    )
    sc.lgkm_per_load_a = lgkm_a
    sc.lgkm_per_load_b = lgkm_b
    return sc^
