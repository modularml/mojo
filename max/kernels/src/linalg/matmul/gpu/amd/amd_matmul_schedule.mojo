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
"""Declarative software pipeline schedule for the default AMD matmul kernel.

This module defines the Loop Dependency Graph (LDG), schedule builder, and
schedule hint derivation for the single-buffer matmul pipeline in matmul.mojo.

Architecture (single-buffer, barrier-gated pipeline):

  Prologue:
    load_dram → store_smem → barrier → load_dram(prefetch) → load_frag[0]

  Kernel body (per K-loop iteration, num_k_tiles=T):
    load_frag[1..T-1], compute[0], barrier,
    store_smem, load_dram(prefetch), compute[1..T-1],
    barrier, load_frag[0],
    schedule_group_barrier hints

  Epilogue (2 drain iterations):
    Drain 1: load_frag[1..T-1], barrier, store_smem, compute[0..T-1]
    Drain 2: barrier, load_frag[0..T-1], compute[0..T-1]

Key differences from ping-pong matmul:
  - Single SMEM buffer (barriers gate read/write phases, no double-buffering)
  - All warps identical (no warp groups or stagger)
  - Bundled ops: load_dram=A+B, load_frag=A+B, store_smem=A+B
  - Iterator-based K advancement (no KOffsetKind)
  - schedule_group_barrier hints instead of schedule_barrier fences
"""

from std.collections import List

from .pipeline_body import PipelineBody
from pipeline.types import (
    OpDesc,
    OpRole,
    ResourceKind,
    ScheduleOps,
    TargetCostModel,
    annotate_ops,
)
from pipeline.config import (
    PipelineConfig,
    TargetProfile,
)
from pipeline.pipeline_dsl import Pipe, ScheduleEntry
from pipeline.program_builder import (
    optimize_within_barriers,
    single_buffer_reorder,
)
from pipeline.compiler import (
    PipelineSchedule,
    ScheduleCompiler,
)
from .amd_target import (
    AMDScheduleHints,
    append_amd_hints,
    mi355x_single_buffer_target,
)


# =============================================================================
# Default matmul op tags — extends ScheduleOps with kernel-specific data ops
# =============================================================================


@fieldwise_init
struct DefaultMatmulOps(ScheduleOps):
    """Op tags for the default single-buffer matmul kernel.

    Kernel-specific data ops in the 0-127 range; framework infrastructure ops
    inherited from ScheduleOps at 128+.
    """

    var value: Int

    comptime LOAD_DRAM = Self(0)
    comptime STORE_SMEM = Self(1)
    comptime LOAD_FRAG = Self(2)
    comptime COMPUTE = Self(3)


# Convenience aliases (Int values) for use in op factories and external imports.
comptime LOAD_DRAM = DefaultMatmulOps.LOAD_DRAM.value
comptime STORE_SMEM = DefaultMatmulOps.STORE_SMEM.value
comptime LOAD_FRAG = DefaultMatmulOps.LOAD_FRAG.value
comptime COMPUTE = DefaultMatmulOps.COMPUTE.value

# =============================================================================
# Default matmul op factories (used by tests and legacy code)
# =============================================================================

comptime _GMEM = ResourceKind.GLOBAL_MEM
comptime _LDS = ResourceKind.LDS
comptime _MMA = ResourceKind.MMA_UNIT


def _load_dram() -> OpDesc:
    return OpDesc.op(LOAD_DRAM, _GMEM, 200, OpRole.GLOBAL_LOAD, channel=0)


def _store_smem() -> OpDesc:
    return OpDesc.op(STORE_SMEM, _LDS, 20, OpRole.SHARED_STORE, channel=0)


def _load_frag(*, subtile: Int = 0) -> OpDesc:
    return OpDesc.op(
        LOAD_FRAG, _LDS, 20, OpRole.FRAGMENT_LOAD, channel=0, subtile=subtile
    )


def _compute(*, subtile: Int = 0) -> OpDesc:
    return OpDesc.op(COMPUTE, _MMA, 64, OpRole.COMPUTE, subtile=subtile)


def load_frags[start: Int, end: Int]() -> Pipe[end - start]:
    """Build a Pipe of load_frag ops for k-tiles start..end-1.

    Parameters:
        start: Inclusive starting K-tile index for the load_frag op range.
        end: Exclusive ending K-tile index for the load_frag op range.
    """
    var p = Pipe[end - start]()
    comptime for k in range(start, end):
        p.ops[k - start] = _load_frag(subtile=k)
    return p^


def compute_range[start: Int, end: Int]() -> Pipe[end - start]:
    """Build a Pipe of compute ops for k-tiles start..end-1.

    Parameters:
        start: Inclusive starting K-tile index for the compute op range.
        end: Exclusive ending K-tile index for the compute op range.
    """
    var p = Pipe[end - start]()
    comptime for k in range(start, end):
        p.ops[k - start] = _compute(subtile=k)
    return p^


# =============================================================================
# Logical op table (declarative — no costs)
# =============================================================================


def _logical_body[num_k_tiles: Int]() -> List[OpDesc]:
    """Logical op table for single-buffer matmul iteration.

    Pure data: declares WHAT ops exist with buffer metadata only.
    No resource kinds, no latencies, no roles; those come from the
    TargetCostModel.
    """
    with PipelineBody() as b:
        b.load(LOAD_DRAM, ch=0)  # DRAM → LDS (A+B bundled)
        b.store(STORE_SMEM, ch=0)  # LDS write (A+B bundled)
        b.barrier()  # read/write gate
        b.fan[num_k_tiles](LOAD_FRAG, ch=0)  # LDS → registers per k-tile
        b.fan[num_k_tiles](COMPUTE)  # MMA per k-tile
        return b.done()


# =============================================================================
# SingleBufferSchedule — PipelineSchedule implementation for default matmul
# =============================================================================


struct SingleBufferSchedule[num_k_tiles: Int](PipelineSchedule):
    """Declarative schedule for the default single-buffer matmul.

    The algorithm declares logical ops (tag + buffer metadata only).
    The target cost model supplies resource, latency, and role.
    The framework derives the pipeline ordering and dependency edges.

    Parameters:
        num_k_tiles: Number of K-tiles along the contraction dimension,
            each producing one fragment load and one MMA compute op.
    """

    var hints: AMDScheduleHints
    var _target: TargetProfile

    def __init__(
        out self,
        hints: AMDScheduleHints,
        target: TargetProfile = mi355x_single_buffer_target(),
    ):
        self.hints = hints
        self._target = target

    def config(self) -> PipelineConfig:
        return self._target.pipeline

    def build_body(self) -> List[OpDesc]:
        """Declare logical ops, annotate with cost model, reorder for pipeline.
        """
        # Table-driven: logical ops are pure data (tag + buffer metadata).
        var logical = _logical_body[Self.num_k_tiles]()

        # Annotate with cost model: stamps resource/latency/role.
        var annotated = annotate_ops(logical, self._target.cost_model)

        # Structural reorder: assigns ops to barrier-delimited segments.
        var body = single_buffer_reorder(annotated, self._target.pipeline)

        # CSP optimization: optimal ordering within each segment.
        return optimize_within_barriers(body, self._target.pipeline)

    def transform_kernel(
        self, ker: List[ScheduleEntry], body: List[OpDesc]
    ) -> List[ScheduleEntry]:
        """Append AMD schedule_group_barrier hints to kernel entries.

        Args:
            ker: The kernel schedule entries to append AMD
                schedule_group_barrier hints to.
            body: The annotated body ops used to count fragment loads and
                split loop-carried vs non-loop-carried frags.
        """
        var result = ker.copy()
        append_amd_hints(result, body, self.config(), self.hints)
        return result^


# =============================================================================
# Top-Level Schedule Builder
# =============================================================================


def build_default_matmul_schedule[
    num_k_tiles: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_k_mmas: Int,
    MMA_M: Int,
    MMA_N: Int,
    a_loads_per_thread: Int,
    b_loads_per_thread: Int,
]() -> ScheduleCompiler:
    """Build the complete software pipeline schedule for the default matmul.

    Uses ScheduleCompiler with SingleBufferSchedule trait implementation.
    All phases derived from the logical body; no magic numbers in derivation.

    Returns a ScheduleCompiler with all phases as Lists. Use as a
    comptime value; the kernel reads entries via comptime for.

    Parameters:
        num_k_tiles: Number of K-tiles along the contraction dimension,
            each producing one fragment load and one MMA compute op.
        num_m_mmas: Number of MMA tiles along the M dimension of the
            output tile.
        num_n_mmas: Number of MMA tiles along the N dimension of the
            output tile.
        num_k_mmas: Number of MMA tiles along the K dimension per
            COMPUTE body op.
        MMA_M: M dimension of a single MMA instruction (for example 16
            or 32).
        MMA_N: N dimension of a single MMA instruction (for example 16
            or 32).
        a_loads_per_thread: Number of DS_WRITE ops per STORE_SMEM for
            the A operand tile.
        b_loads_per_thread: Number of DS_WRITE ops per STORE_SMEM for
            the B operand tile.
    """
    var schedule = SingleBufferSchedule[num_k_tiles](
        AMDScheduleHints(
            m_mmas=num_m_mmas,
            n_mmas=num_n_mmas,
            k_mmas=num_k_mmas,
            mma_m=MMA_M,
            mma_n=MMA_N,
            a_loads_per_thread=a_loads_per_thread,
            b_loads_per_thread=b_loads_per_thread,
        )
    )
    var sc = ScheduleCompiler()
    sc.compile(schedule)
    return sc^
