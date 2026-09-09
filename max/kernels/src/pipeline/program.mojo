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
"""MMABlockSpec and PipelineProgram: declarative pipeline program representation.

These structures hold the compiled schedule as a sequence of MMA blocks,
separating schedule definition from schedule expansion.
"""

from std.collections import List

from .pipeline_dsl import EntryBuilder, ScheduleEntry
from .types import OpDesc, OpRole, Phase, _Ops

# =============================================================================
# MMA Block Specification & Pipeline Program
# =============================================================================


struct MMABlockSpec(ImplicitlyCopyable, Movable):
    """Declarative specification for one MMA block in a pipeline schedule.

    Each MMA block follows a fixed pattern with optional elements:
      [pre_op_0?], [pre_op_1?], [global_load?], [global_load_1?],
      [pre_sync?], barrier, [schedule_barrier?], [wait_lgkm(0)?],
      set_prio(1), mma, [fused_mma?], set_prio(0),
      barrier, [schedule_barrier?]

    Optional elements are controlled by sentinel values:
      - OpDesc fields: _Ops.NONE.value means skip
      - Bool flags: False means skip the corresponding element

    All operations are stored as pre-built OpDesc values, avoiding any
    runtime-to-comptime conversion.
    """

    var mma: OpDesc
    var pre_op_0: OpDesc
    var pre_op_1: OpDesc
    var global_load: OpDesc
    var global_load_1: OpDesc
    var entry_wait: OpDesc
    """`wait_vm` op emitted at the start of the block (or `OpDesc.none()`)."""
    var entry_wait_lgkm: OpDesc
    """`wait_lgkm` op emitted at the start of the block (or `OpDesc.none()`)."""
    var pre_sync: OpDesc
    var fused_mma: OpDesc
    var pre_mma_barrier: Bool
    """Emit an `s_barrier` before the MMA op."""
    var pre_mma_set_prio: Bool
    """Emit `s_setprio[1]` before the MMA op."""
    var post_mma_barrier: Bool
    """Emit an `s_barrier` after the MMA op."""
    var post_mma_set_prio: Bool
    """Emit `s_setprio[0]` after the MMA op."""
    var post_barrier_lgkm: Bool
    var post_barrier_sched: Bool
    var trailing_sched_barrier: Bool
    var global_load_prefetch: Bool
    var global_load_1_prefetch: Bool
    var drain_lgkm_before_loads: Bool
    # Per-block emission-shape knobs (defaults preserve ping-pong layout).
    # `global_before_frag` swaps the in-block order of global loads and
    # fragment loads. `barrier_before_pre_ops` moves the pre_sync+barrier
    # section ahead of the frag/global section.
    # `wrap_waits_with_sched_barrier` wraps each contiguous wait/barrier
    # group with `schedule_barrier` on both sides as an LLVM machine-
    # scheduler fence. See `ScheduleConfig` for rationale.
    var global_before_frag: Bool
    """Swaps the in-block order of global loads and fragment loads."""
    var barrier_before_pre_ops: Bool
    """Moves the pre_sync + barrier section ahead of the frag/global section."""
    var wrap_waits_with_sched_barrier: Bool
    """Wraps each contiguous wait/barrier group with `schedule_barrier`
    on both sides as an LLVM machine-scheduler fence."""

    @always_inline
    def __init__(
        out self,
        *,
        mma: OpDesc,
        pre_op_0: OpDesc = OpDesc.none(),
        pre_op_1: OpDesc = OpDesc.none(),
        global_load: OpDesc = OpDesc.none(),
        global_load_1: OpDesc = OpDesc.none(),
        entry_wait: OpDesc = OpDesc.none(),
        entry_wait_lgkm: OpDesc = OpDesc.none(),
        pre_sync: OpDesc = OpDesc.none(),
        fused_mma: OpDesc = OpDesc.none(),
        pre_mma_barrier: Bool = True,
        pre_mma_set_prio: Bool = True,
        post_mma_barrier: Bool = True,
        post_mma_set_prio: Bool = True,
        post_barrier_lgkm: Bool = True,
        post_barrier_sched: Bool = False,
        trailing_sched_barrier: Bool = False,
        global_load_prefetch: Bool = False,
        global_load_1_prefetch: Bool = False,
        drain_lgkm_before_loads: Bool = False,
        global_before_frag: Bool = False,
        barrier_before_pre_ops: Bool = False,
        wrap_waits_with_sched_barrier: Bool = False,
    ):
        self.mma = mma
        self.pre_op_0 = pre_op_0
        self.pre_op_1 = pre_op_1
        self.global_load = global_load
        self.global_load_1 = global_load_1
        self.entry_wait = entry_wait
        self.entry_wait_lgkm = entry_wait_lgkm
        self.pre_sync = pre_sync
        self.fused_mma = fused_mma
        self.pre_mma_barrier = pre_mma_barrier
        self.pre_mma_set_prio = pre_mma_set_prio
        self.post_mma_barrier = post_mma_barrier
        self.post_mma_set_prio = post_mma_set_prio
        self.post_barrier_lgkm = post_barrier_lgkm
        self.post_barrier_sched = post_barrier_sched
        self.trailing_sched_barrier = trailing_sched_barrier
        self.global_load_prefetch = global_load_prefetch
        self.global_load_1_prefetch = global_load_1_prefetch
        self.drain_lgkm_before_loads = drain_lgkm_before_loads
        self.global_before_frag = global_before_frag
        self.barrier_before_pre_ops = barrier_before_pre_ops
        self.wrap_waits_with_sched_barrier = wrap_waits_with_sched_barrier

    @always_inline
    def _has_entry_wait_group(self) -> Bool:
        """True if entry_wait or entry_wait_lgkm is present."""
        return self.entry_wait.is_present() or self.entry_wait_lgkm.is_present()

    @always_inline
    def _has_pre_mma_sync_group(self) -> Bool:
        """True if a `pre_sync` + barrier section will fire."""
        return self.pre_sync.is_present() or self.pre_mma_barrier

    @always_inline
    def _entry_wait_wrap(self) -> Int:
        """0 or 2 schedule_barrier ops wrapping the entry_wait group."""
        if self.wrap_waits_with_sched_barrier and self._has_entry_wait_group():
            return 2
        return 0

    @always_inline
    def _pre_mma_sync_wrap(self) -> Int:
        """0 or 2 schedule_barrier ops wrapping the pre_sync+barrier group."""
        if (
            self.wrap_waits_with_sched_barrier
            and self._has_pre_mma_sync_group()
        ):
            return 2
        return 0

    @always_inline
    def _n_pre_barrier_ops(self) -> Int:
        """Count optional ops before the barrier (entry_wait, pre_op_0/1, drain, global loads, pre_sync).
        """
        var n = 0
        if self.entry_wait.is_present():
            # entry_wait is followed by an s_barrier in the emit
            # (cross-warp visibility fence after the vm drain). Count
            # both ops together.
            n += 2
        if self.entry_wait_lgkm.is_present():
            n += 1
        if self.pre_op_0.is_present():
            n += 1
        if self.pre_op_1.is_present():
            n += 1
        if self.drain_lgkm_before_loads:
            n += 1
        if self.global_load.is_present():
            n += 1
        if self.global_load_1.is_present():
            n += 1
        if self.pre_sync.is_present():
            n += 1
        # schedule_barrier wraps around the entry_wait group and the
        # pre_sync+barrier group (when enabled).
        n += self._entry_wait_wrap()
        n += self._pre_mma_sync_wrap()
        return n

    @always_inline
    def entry_count(self) -> Int:
        """Count the number of schedule entries this block will expand to.

        With all flags True (the default ping-pong layout), 5 mandatory
        entries (barrier, set_prio(1), mma, set_prio(0), barrier) plus
        one for each optional field present. The `pre_mma_*` /
        `post_mma_*` Bool flags let `minimal_barriers` schedules
        suppress unneeded sync ops.
        """
        var n = self._n_pre_barrier_ops() + 1  # mma
        if self.pre_mma_barrier:
            n += 1
            if self.post_barrier_sched:
                n += 1
            if self.post_barrier_lgkm:
                n += 1
        if self.pre_mma_set_prio:
            n += 1
        if self.fused_mma.is_present():
            n += 1
        if self.post_mma_set_prio:
            n += 1
        if self.post_mma_barrier:
            n += 1
            if self.trailing_sched_barrier:
                n += 1
        return n

    @always_inline
    def mma_position(self) -> Int:
        """Return the offset of the MMA op within this block's entries."""
        var n = self._n_pre_barrier_ops()
        if self.pre_mma_barrier:
            n += 1
            if self.post_barrier_sched:
                n += 1
            if self.post_barrier_lgkm:
                n += 1
        if self.pre_mma_set_prio:
            n += 1
        return n

    @always_inline
    def _n_ops_before_barrier_section(self) -> Int:
        """Count ops emitted before the barrier section.

        With `barrier_before_pre_ops=False` (default) the barrier sits
        after frags + globals, so this equals `_n_pre_barrier_ops`.
        With `barrier_before_pre_ops=True` the barrier sits right after
        `entry_wait`/`entry_wait_lgkm`/`pre_sync`; fragments and
        globals come *after* the barrier, so they don't count here.
        """
        if not self.barrier_before_pre_ops:
            return self._n_pre_barrier_ops()
        var n = 0
        if self.entry_wait.is_present():
            n += 1
        if self.entry_wait_lgkm.is_present():
            n += 1
        if self.pre_sync.is_present():
            n += 1
        # schedule_barrier wraps around entry_wait group + sync group.
        n += self._entry_wait_wrap()
        # The leading half of the sync wrap fires before the barrier.
        if self._pre_mma_sync_wrap() > 0:
            n += 1
        return n

    @always_inline
    def post_barrier_lgkm_position(self) -> Int:
        """Return the offset of the post-barrier wait_lgkm(0) within entries.

        Returns -1 if `post_barrier_lgkm` is False or the pre-MMA
        barrier is suppressed.
        """
        if not self.post_barrier_lgkm or not self.pre_mma_barrier:
            return -1
        var n = self._n_ops_before_barrier_section() + 1  # +1 for barrier
        if self.post_barrier_sched:
            n += 1
        return n

    @always_inline
    def expand[N: Int, phase: Phase](self, mut b: EntryBuilder[N, phase]):
        """Expand this block spec into schedule entries via an EntryBuilder.

        Emission shape is controlled by two per-block flags:
          - `global_before_frag` swaps the order of `pre_op_*` (frags)
            and `global_load*` (DRAM→LDS prefetches).
          - `barrier_before_pre_ops` moves the `pre_sync` + barrier
            section ahead of the frag/global section.
        """
        # Entry wait group, optionally wrapped with schedule_barrier
        # fences when `wrap_waits_with_sched_barrier` is set.
        if self._entry_wait_wrap() > 0:
            b.emit(OpDesc.schedule_barrier())
        b.emit_if(self.entry_wait)
        # `entry_wait_lgkm` is emitted IMMEDIATELY after `entry_wait`
        # (when both present) so LLVM's waitcnt-merge pass can coalesce
        # them into one `s_waitcnt vmcnt(N) lgkmcnt(M)` instruction.
        b.emit_if(self.entry_wait_lgkm)
        # Cross-warp visibility fence — see `emit_minimal_barrier_block`
        # for full rationale. Without this `s_barrier`, the subsequent
        # frag-loads can race against other warps' still-pending
        # `buffer_load_lds → LDS` writes from the previous iter.
        if self.entry_wait.is_present():
            b.emit(OpDesc.barrier())
        if self._entry_wait_wrap() > 0:
            b.emit(OpDesc.schedule_barrier())

        @__parameter
        @always_inline
        def emit_sync_section():
            if self._pre_mma_sync_wrap() > 0:
                b.emit(OpDesc.schedule_barrier())
            b.emit_if(self.pre_sync)
            if self.pre_mma_barrier:
                b.emit(OpDesc.barrier())
                b.emit_flag(OpDesc.schedule_barrier(), self.post_barrier_sched)
                b.emit_flag(OpDesc.wait_lgkm[0](), self.post_barrier_lgkm)
            if self._pre_mma_sync_wrap() > 0:
                b.emit(OpDesc.schedule_barrier())

        @__parameter
        @always_inline
        def emit_load_section():
            if self.global_before_frag:
                b.emit_if(self.global_load, self.global_load_prefetch)
                b.emit_if(self.global_load_1, self.global_load_1_prefetch)
                b.emit_flag(OpDesc.wait_lgkm[0](), self.drain_lgkm_before_loads)
                b.emit_if(self.pre_op_0)
                b.emit_if(self.pre_op_1)
            else:
                b.emit_if(self.pre_op_0)
                b.emit_if(self.pre_op_1)
                b.emit_flag(OpDesc.wait_lgkm[0](), self.drain_lgkm_before_loads)
                b.emit_if(self.global_load, self.global_load_prefetch)
                b.emit_if(self.global_load_1, self.global_load_1_prefetch)

        if self.barrier_before_pre_ops:
            emit_sync_section()
            emit_load_section()
        else:
            emit_load_section()
            emit_sync_section()

        if self.pre_mma_set_prio:
            b.emit(OpDesc.set_prio[1]())
        b.emit(self.mma)
        b.emit_if(self.fused_mma)
        if self.post_mma_set_prio:
            b.emit(OpDesc.set_prio[0]())
        if self.post_mma_barrier:
            b.emit(OpDesc.barrier())
            b.emit_flag(OpDesc.schedule_barrier(), self.trailing_sched_barrier)

    @always_inline
    def expand_to_list(self, mut out: List[ScheduleEntry], phase: Phase):
        """Expand this block spec by appending to a List.

        Emission shape mirrors `expand`; see that method for the per-block
        ordering flags.
        """

        @always_inline
        def _e(
            mut out: List[ScheduleEntry],
            op: OpDesc,
            phase: Phase,
            prefetch: Bool = False,
        ):
            out.append(
                ScheduleEntry(
                    op=op,
                    time_slot=len(out),
                    phase=phase,
                    is_prefetch=prefetch,
                )
            )

        if self._entry_wait_wrap() > 0:
            _e(out, OpDesc.schedule_barrier(), phase)
        if self.entry_wait.is_present():
            _e(out, self.entry_wait, phase)
        if self.entry_wait_lgkm.is_present():
            _e(out, self.entry_wait_lgkm, phase)
        # Cross-warp visibility fence — see the matching block in
        # `emit_minimal_barrier_block` for the full rationale. The
        # `entry_wait` (`wait_vm[N]`) is a per-warp drain; the
        # subsequent frag-loads can race against other warps' still-
        # pending `buffer_load_lds → LDS` writes without an explicit
        # `s_barrier` between them.
        if self.entry_wait.is_present():
            _e(out, OpDesc.barrier(), phase)
        if self._entry_wait_wrap() > 0:
            _e(out, OpDesc.schedule_barrier(), phase)

        # Sync section (pre_sync + barrier + post-barrier ops).
        if self.barrier_before_pre_ops:
            if self._pre_mma_sync_wrap() > 0:
                _e(out, OpDesc.schedule_barrier(), phase)
            if self.pre_sync.is_present():
                _e(out, self.pre_sync, phase)
            if self.pre_mma_barrier:
                _e(out, OpDesc.barrier(), phase)
                if self.post_barrier_sched:
                    _e(out, OpDesc.schedule_barrier(), phase)
                if self.post_barrier_lgkm:
                    _e(out, OpDesc.wait_lgkm[0](), phase)
            if self._pre_mma_sync_wrap() > 0:
                _e(out, OpDesc.schedule_barrier(), phase)

        # Load section (frags + globals, ordered by global_before_frag).
        if self.global_before_frag:
            if self.global_load.is_present():
                _e(out, self.global_load, phase, self.global_load_prefetch)
            if self.global_load_1.is_present():
                _e(
                    out,
                    self.global_load_1,
                    phase,
                    self.global_load_1_prefetch,
                )
            if self.drain_lgkm_before_loads:
                _e(out, OpDesc.wait_lgkm[0](), phase)
            if self.pre_op_0.is_present():
                _e(out, self.pre_op_0, phase)
            if self.pre_op_1.is_present():
                _e(out, self.pre_op_1, phase)
        else:
            if self.pre_op_0.is_present():
                _e(out, self.pre_op_0, phase)
            if self.pre_op_1.is_present():
                _e(out, self.pre_op_1, phase)
            if self.drain_lgkm_before_loads:
                _e(out, OpDesc.wait_lgkm[0](), phase)
            if self.global_load.is_present():
                _e(out, self.global_load, phase, self.global_load_prefetch)
            if self.global_load_1.is_present():
                _e(
                    out,
                    self.global_load_1,
                    phase,
                    self.global_load_1_prefetch,
                )

        # Sync section after load section (ping-pong default).
        if not self.barrier_before_pre_ops:
            if self._pre_mma_sync_wrap() > 0:
                _e(out, OpDesc.schedule_barrier(), phase)
            if self.pre_sync.is_present():
                _e(out, self.pre_sync, phase)
            if self.pre_mma_barrier:
                _e(out, OpDesc.barrier(), phase)
                if self.post_barrier_sched:
                    _e(out, OpDesc.schedule_barrier(), phase)
                if self.post_barrier_lgkm:
                    _e(out, OpDesc.wait_lgkm[0](), phase)
            if self._pre_mma_sync_wrap() > 0:
                _e(out, OpDesc.schedule_barrier(), phase)

        if self.pre_mma_set_prio:
            _e(out, OpDesc.set_prio[1](), phase)
        _e(out, self.mma, phase)
        if self.fused_mma.is_present():
            _e(out, self.fused_mma, phase)
        if self.post_mma_set_prio:
            _e(out, OpDesc.set_prio[0](), phase)
        if self.post_mma_barrier:
            _e(out, OpDesc.barrier(), phase)
            if self.trailing_sched_barrier:
                _e(out, OpDesc.schedule_barrier(), phase)


# =============================================================================
# Block-emission helpers — for schedules using `build_explicit_blocks`
# =============================================================================


def emit_minimal_barrier_block(
    block: MMABlockSpec, wrap_waits: Bool, global_before_frag: Bool = False
) -> List[OpDesc]:
    """Emit one block in the "minimal-barrier + cross-stage rotation"
    shape: for schedules that override `build_explicit_blocks`.

    Layout (per block):
      - Sync-group A: `[sched_barrier]` `entry_wait` `entry_wait_lgkm`
        `[sched_barrier]`. Fences emitted iff `wrap_waits=True` and
        either entry-wait field is present.
      - Load section: frags + globals, in order controlled by
        `global_before_frag` (False = ping-pong default; True =
        load-before-frag for kernels like 4-wave inline that benefit).
      - Sync-group B: `[sched_barrier]` `pre_sync` `[barrier
        post_barrier_lgkm]` `[sched_barrier]`. Fences emitted iff
        `wrap_waits=True` and either pre_sync or pre_mma_barrier is
        present.
      - Final `mma`.

    Reads wait values, frag/global ops, barrier flags from
    `block`, typically populated by `_construct_mma_blocks` and
    patched by `derive_waits_from_blocks`. Schedules consume the
    derived structure and emit it in their preferred order, without
    the conditional template branching of `MMABlockSpec.expand`.

    Bypasses `pre_mma_set_prio` / `post_mma_*` / `fused_mma` / drain
    flags; those don't apply under the minimal-barriers pattern.
    Schedules with different needs should write their own emitter.

    Args:
        block: The block spec to emit.
        wrap_waits: Wrap each contiguous wait/barrier group with
            `schedule_barrier` on both sides.
        global_before_frag: Load globals before frags inside the block.

    Returns:
        Ordered op list for the block, ready to be appended to a
        per-block emission override.
    """
    var ops = List[OpDesc]()

    # Sync-group A
    var has_a = block._has_entry_wait_group()
    if wrap_waits and has_a:
        ops.append(OpDesc.schedule_barrier())
    if block.entry_wait.is_present():
        ops.append(block.entry_wait)
    if block.entry_wait_lgkm.is_present():
        ops.append(block.entry_wait_lgkm)
    # Cross-warp visibility fence on top-of-half / first-cross-stage
    # entries. The `entry_wait` (`wait_vm[N]`) only drains *this warp's*
    # in-flight `buffer_load_lds`; it does NOT guarantee that the LDS
    # writes from those loads are visible across the workgroup. The
    # subsequent frag-loads (`pre_op_0/1`) read LDS regions that may
    # have been written by other warps in the previous iter. Without an
    # explicit `s_barrier` here, those `ds_read`s race against the
    # other warps' still-pending LDS-write commit, producing intermittent
    # wrong values (observed at FP8 BM=64 on a small subset of conv
    # shapes — ~10% per-run flake in CI). Mirrors the handwritten
    # body's top-of-iter `wait_vm + wait_lgkm + s_barrier` pattern.
    if block.entry_wait.is_present():
        ops.append(OpDesc.barrier())
    if wrap_waits and has_a:
        ops.append(OpDesc.schedule_barrier())

    # Load section. We always emit a `schedule_barrier` between the
    # global-load (buffer_load_lds, vmcnt-tracked) and the frag-load
    # (ds_read, lgkmcnt-tracked) — without it LLVM's machine scheduler
    # is free to hoist the ds_read past the buffer_load_lds, which
    # races against in-flight prior-iter LDS writes to the same region.
    # The fence is *load-bearing for correctness* on cross-stage
    # rotation kernels and harmless on others.
    var has_globals = (
        block.global_load.is_present() or block.global_load_1.is_present()
    )
    var has_frags = block.pre_op_0.is_present() or block.pre_op_1.is_present()
    var needs_load_frag_fence = has_globals and has_frags
    if global_before_frag:
        if block.global_load.is_present():
            ops.append(block.global_load)
        if block.global_load_1.is_present():
            ops.append(block.global_load_1)
        if needs_load_frag_fence:
            ops.append(OpDesc.schedule_barrier())
        if block.pre_op_0.is_present():
            ops.append(block.pre_op_0)
        if block.pre_op_1.is_present():
            ops.append(block.pre_op_1)
    else:
        if block.pre_op_0.is_present():
            ops.append(block.pre_op_0)
        if block.pre_op_1.is_present():
            ops.append(block.pre_op_1)
        if needs_load_frag_fence:
            ops.append(OpDesc.schedule_barrier())
        if block.global_load.is_present():
            ops.append(block.global_load)
        if block.global_load_1.is_present():
            ops.append(block.global_load_1)

    # Sync-group B
    var has_b = block._has_pre_mma_sync_group()
    if wrap_waits and has_b:
        ops.append(OpDesc.schedule_barrier())
    if block.pre_sync.is_present():
        ops.append(block.pre_sync)
    if block.pre_mma_barrier:
        ops.append(OpDesc.barrier())
        if block.post_barrier_lgkm:
            ops.append(OpDesc.wait_lgkm[0]())
    if wrap_waits and has_b:
        ops.append(OpDesc.schedule_barrier())

    ops.append(block.mma)
    return ops^


struct PipelineProgram(Copyable, Movable):
    """A pipeline schedule phase as a sequence of MMA block specifications.

    Separates schedule *definition* (what blocks to emit) from schedule
    *expansion* (writing ScheduleEntry values into a List). This makes
    schedules declarative data rather than imperative code.

    Per-block emission has two paths:
      - Default: each `MMABlockSpec` is expanded via the flag-driven
        template in `MMABlockSpec.expand_to_list`. The `*_strategy`
        flags on `ScheduleConfig` control op order, barrier placement,
        wrap fences, etc.
      - Override: schedules can supply a parallel `explicit_blocks`
        list (one `List[OpDesc]` per block) that bypasses the template
        entirely. When `explicit_blocks[i]` is non-empty the framework
        emits those ops verbatim, giving schedules full control over
        per-block emission shape without needing new flags.

    `explicit_blocks` defaults to empty (every block uses the
    template). Schedules opt in by overriding `build_explicit_blocks`
    on the `PipelineSchedule` trait.
    """

    var blocks: List[MMABlockSpec]
    var explicit_blocks: List[List[OpDesc]]
    """Per-block override op lists. Empty entries fall back to the
    template; non-empty entries are emitted verbatim. Defaults to empty
    so every block uses the template."""
    var trailing_barrier: Bool

    @always_inline
    def __init__(
        out self, num_blocks: Int = 0, *, trailing_barrier: Bool = False
    ):
        self.blocks = List[MMABlockSpec]()
        self.explicit_blocks = List[List[OpDesc]]()
        for _ in range(num_blocks):
            self.blocks.append(MMABlockSpec(mma=OpDesc.none()))
            self.explicit_blocks.append(List[OpDesc]())
        self.trailing_barrier = trailing_barrier

    @always_inline
    def _block_entry_count(self, block_idx: Int) -> Int:
        """Entry count for one block: explicit override if non-empty,
        else the flag-driven template count.
        """
        if (
            block_idx < len(self.explicit_blocks)
            and len(self.explicit_blocks[block_idx]) > 0
        ):
            return len(self.explicit_blocks[block_idx])
        return self.blocks[block_idx].entry_count()

    @always_inline
    def total_entries(self) -> Int:
        """Count total schedule entries this program will expand to."""
        var n = 0
        for i in range(len(self.blocks)):
            n += self._block_entry_count(i)
        if self.trailing_barrier:
            n += 1
        return n

    @always_inline
    def block_start(self, block_idx: Int) -> Int:
        """Return the starting entry index for the given block."""
        var n = 0
        for i in range(block_idx):
            n += self._block_entry_count(i)
        return n

    @always_inline
    def mma_entry(self, block_idx: Int) -> Int:
        """Return the entry index of the MMA op in the given block.

        For explicit-override blocks: scans the op list for the first
        COMPUTE-role op. For template blocks: uses `mma_position`.
        """
        var start = self.block_start(block_idx)
        if (
            block_idx < len(self.explicit_blocks)
            and len(self.explicit_blocks[block_idx]) > 0
        ):
            var n = len(self.explicit_blocks[block_idx])
            for j in range(n):
                if self.explicit_blocks[block_idx][j].role == OpRole.COMPUTE:
                    return start + j
            # Fall through if no MMA found (shouldn't happen for
            # well-formed blocks); return block_start as a sentinel.
            return start
        return start + self.blocks[block_idx].mma_position()

    @always_inline
    def expand_to_list(self, phase: Phase) -> List[ScheduleEntry]:
        """Expand all blocks into a List of schedule entries.

        For each block: emit `explicit_blocks[i]` verbatim if non-empty,
        otherwise call the block's flag-driven `expand_to_list`.
        """
        var out = List[ScheduleEntry]()
        for i in range(len(self.blocks)):
            if (
                i < len(self.explicit_blocks)
                and len(self.explicit_blocks[i]) > 0
            ):
                var n_ops = len(self.explicit_blocks[i])
                for j in range(n_ops):
                    out.append(
                        ScheduleEntry(
                            op=self.explicit_blocks[i][j],
                            time_slot=len(out),
                            phase=phase,
                            is_prefetch=False,
                        )
                    )
            else:
                self.blocks[i].expand_to_list(out, phase)
        if self.trailing_barrier:
            out.append(
                ScheduleEntry(
                    op=OpDesc.barrier(),
                    time_slot=len(out),
                    phase=phase,
                    is_prefetch=False,
                )
            )
        return out^
