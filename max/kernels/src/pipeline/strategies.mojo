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
"""Strategy structs that group related `ScheduleConfig` flags.

`ScheduleConfig`'s flat field list grew to ~20 flags as new emission
shapes landed. Many of those flags are tightly correlated (e.g. you
never set `omit_mma_set_prio=True` without also setting
`minimal_barriers=True`); grouping them into named strategy structs
documents the relationships and reduces the constructor's argument
count at the call site.

These structs are an *opt-in* alternative to the flat-field
`ScheduleConfig` constructor; existing callers passing flat kwargs
continue to work unchanged. Use via:

```mojo
var sc = ScheduleConfig.from_strategies(
    barrier=BarrierStrategy.minimal_no_set_prio(
        sched_barrier_mask=0xFF,
        wrap_waits=True,
    ),
    wait=WaitStrategy.auto_with_inter_block_drain(),
    load=LoadStrategy.default(),
    scheduling=SchedulingStrategy.IDENTITY,
)
```

Strategy methods named `default()` return the framework's default
(ping-pong) behaviour, passing only `scheduling` to
`from_strategies` reproduces `ScheduleConfig()` exactly.
"""

from .config import SchedulingStrategy


# =============================================================================
# Barrier strategy — minimal_barriers, omit_set_prio, sched_barrier_mask,
# wrap_waits_with_sched_barrier, barrier_before_pre_ops
# =============================================================================


@fieldwise_init
struct BarrierStrategy(Copyable, Movable):
    """How the schedule emits barriers and barrier-adjacent fences."""

    var minimal: Bool
    """Suppresses per-block `s_barrier`s and `set_prio` pairs; emits
    `s_barrier` only at top-of-half and the first cross-stage block.
    See `ScheduleConfig.minimal_barriers`."""
    var omit_set_prio: Bool
    """When `minimal=True`, drops the pre-MMA `s_setprio[1]` entirely.
    See `ScheduleConfig.omit_mma_set_prio`."""
    var sched_barrier_mask: Int
    """Bitmask of which blocks get trailing `schedule_barrier` fences.
    Default: `0b01010101`."""
    var wrap_waits_sched_barrier: Bool
    """Wraps each contiguous wait/barrier group with `schedule_barrier`
    on both sides. See `ScheduleConfig.wrap_waits_with_sched_barrier`."""
    var barrier_before_pre_ops: Bool
    """Moves the pre_sync + barrier section ahead of the frag/global
    section in each block. See `ScheduleConfig.barrier_before_pre_ops`."""

    @staticmethod
    def default() -> Self:
        """Returns the ping-pong default: full barriers, no minimal mode.

        Returns:
            A `BarrierStrategy` matching the framework default.
        """
        return Self(
            minimal=False,
            omit_set_prio=False,
            sched_barrier_mask=0b01010101,
            wrap_waits_sched_barrier=False,
            barrier_before_pre_ops=False,
        )

    @staticmethod
    def minimal_no_set_prio(
        sched_barrier_mask: Int = 0,
        wrap_waits: Bool = False,
        barrier_before_pre_ops: Bool = False,
    ) -> Self:
        """Returns the cross-stage-rotation default: minimal barriers,
        no `set_prio`.

        Args:
            sched_barrier_mask: Bitmask of which blocks get trailing
                `schedule_barrier` fences.
            wrap_waits: Wrap wait/barrier groups with `schedule_barrier`
                on both sides.
            barrier_before_pre_ops: Move pre_sync + barrier ahead of the
                frag/global section in each block.

        Returns:
            A `BarrierStrategy` configured for the cross-stage rotation
            kernel pattern.
        """
        return Self(
            minimal=True,
            omit_set_prio=True,
            sched_barrier_mask=sched_barrier_mask,
            wrap_waits_sched_barrier=wrap_waits,
            barrier_before_pre_ops=barrier_before_pre_ops,
        )


# =============================================================================
# Wait strategy — auto_waits, drain_lgkm_mask, auto_drain, manual overrides,
# inter_block_lgkm_drain, partial_prologue_drain, lgkm_after_last
# =============================================================================


@fieldwise_init
struct WaitStrategy(Copyable, Movable):
    """How the schedule derives and places vmcnt/lgkmcnt waits."""

    var auto_waits: Bool
    """Auto-derives wait counts from program structure."""
    var drain_lgkm_mask: Int
    """Per-block bitmask for selective LDS drains."""
    var auto_drain: Bool
    """Auto-derives `drain_lgkm_mask` from channel analysis."""
    var wait_lgkm_first: Int
    """Manual `wait_lgkm` override (used when `auto_waits=False`).
    `255` means unset."""
    var wait_vm_last: Int
    """Manual `wait_vm` override for the last block."""
    var lgkm_after_last: Bool
    """Inserts `wait_lgkm(0)` after the last block's barrier."""
    var inter_block_lgkm_drain: Bool
    """Emits `wait_lgkm(0)` at non-top, non-cross interior block starts."""
    var partial_prologue_drain: Bool
    """Skips standard `wait_vm(0)` drains in the framework prologue
    (kernel emits its own partial drains via `bootstrap_frags`)."""

    @staticmethod
    def default() -> Self:
        """Returns the ping-pong default: auto waits, no inter-block drain.

        Returns:
            A `WaitStrategy` matching the framework default.
        """
        return Self(
            auto_waits=True,
            drain_lgkm_mask=0,
            auto_drain=False,
            wait_lgkm_first=8,
            wait_vm_last=6,
            lgkm_after_last=False,
            inter_block_lgkm_drain=False,
            partial_prologue_drain=False,
        )

    @staticmethod
    def auto_with_inter_block_drain(
        partial_prologue_drain: Bool = False,
    ) -> Self:
        """Returns auto-derived waits plus inter-block lgkm drain (4-wave
        pattern).

        Args:
            partial_prologue_drain: Skip standard `wait_vm(0)` drains in
                the framework prologue.

        Returns:
            A `WaitStrategy` configured for the 4-wave kernel pattern.
        """
        return Self(
            auto_waits=True,
            drain_lgkm_mask=0,
            auto_drain=False,
            wait_lgkm_first=8,
            wait_vm_last=6,
            lgkm_after_last=False,
            inter_block_lgkm_drain=True,
            partial_prologue_drain=partial_prologue_drain,
        )


# =============================================================================
# Load strategy — global_before_frag, lgkm_per_load_a/b
# =============================================================================


@fieldwise_init
struct LoadStrategy(Copyable, Movable):
    """In-block ordering of fragment loads vs global prefetches."""

    var global_before_frag: Bool
    """Emits global prefetches before fragment loads in each block.
    See `ScheduleConfig.global_before_frag`."""
    var lgkm_per_load_a: Int
    """`lgkmcnt` entries per channel-A frag-load (for wait derivation;
    auto-derived from kernel geometry; see
    `pipeline.geometry.KernelGeometry`)."""
    var lgkm_per_load_b: Int
    """`lgkmcnt` entries per channel-B frag-load."""

    @staticmethod
    def default() -> Self:
        """Returns the ping-pong default: frags before globals, manual lgkm.

        Returns:
            A `LoadStrategy` matching the framework default.
        """
        return Self(
            global_before_frag=False, lgkm_per_load_a=0, lgkm_per_load_b=0
        )
