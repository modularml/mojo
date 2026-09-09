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
"""Diagnostic tooling for compiled pipeline schedules.

Compiled schedules are comptime values (`List[ScheduleEntry]`), and
`List` is not `ImplicitlyCopyable`, so the only way to walk the entries
in a printer is via a comptime `for i in range(N)` loop with
`materialize[entries[i]]()` per access. The helpers below provide the
ingredients for that pattern; callers wire up the comptime loops.

Usage pattern:

```mojo
comptime n_krn = len(schedule.kernel)
print_phase_header("KERNEL", n_krn)
comptime for i in range(n_krn):
    print_entry[my_op_name_fn]("KRN", i, materialize[schedule.kernel[i]]())

var counts = WaitCounts()
comptime for i in range(n_krn):
    counts.add(materialize[schedule.kernel[i]](), i, n_krn)
print_wait_counts("kernel  ", counts)
```
"""

from std.collections import List

from .pipeline_dsl import ScheduleEntry
from .types import KOffsetKind, OpDesc, _Ops


# =============================================================================
# Tag and k-offset name lookup
# =============================================================================


def default_op_name(tag: Int) -> StaticString:
    """Returns the default name for a framework op tag.

    Kernels can override by writing their own `def(Int) -> StaticString`
    that maps their op tags to display names; pass it as the
    `op_name_fn` parameter on `print_entry`.

    Args:
        tag: The op tag value to look up.

    Returns:
        Fixed-width display string for the tag, padded so that columns
        align across rows.
    """
    if tag == _Ops.BARRIER.value:
        return "BARRIER        "
    elif tag == _Ops.WAIT_VM.value:
        return "WAIT_VM        "
    elif tag == _Ops.WAIT_LGKM.value:
        return "WAIT_LGKM      "
    elif tag == _Ops.SET_PRIO.value:
        return "SET_PRIO       "
    elif tag == _Ops.SCHEDULE_BARRIER.value:
        return "SCHED_BARRIER  "
    elif tag == _Ops.NONE.value:
        return "NONE           "
    elif tag < 128:
        # Kernel-specific tag — caller didn't supply a name function.
        return "OP             "
    else:
        return "?              "


def _koff_name(k: KOffsetKind) -> StaticString:
    if k == KOffsetKind.K0:
        return "K0   "
    elif k == KOffsetKind.K1:
        return "K1   "
    elif k == KOffsetKind.K_NEXT:
        return "KNEXT"
    elif k == KOffsetKind.K_PREV:
        return "KPREV"
    else:
        return "-    "


# =============================================================================
# Single-entry print
# =============================================================================


def print_entry(
    prefix: StaticString, idx: Int, name: StaticString, e: ScheduleEntry
):
    """Prints one `ScheduleEntry` on a single line, with the caller's
    pre-resolved tag name.

    Format:

    ```text
    <prefix> [<idx>] <name>  s=<stage> sub=<subtile> k=<k_offset>
                             wait=<wait_value> pf=<is_prefetch>
    ```

    Callers resolve the tag name themselves (kernel tags are
    schedule-specific):

    ```mojo
    var name = (
        my_op_name(e.op.tag) if e.op.tag < 128
        else default_op_name(e.op.tag)
    )
    print_entry("KRN ", i, name, e)
    ```

    Args:
        prefix: Short label printed before the index (e.g. "KRN").
        idx: Index of the entry within its phase.
        name: Display name for `e.op.tag`, resolved by the caller.
        e: The schedule entry to print.
    """
    var op = e.op
    print(
        prefix,
        " [",
        idx,
        "] ",
        name,
        " s=",
        op.stage,
        " sub=",
        op.subtile,
        " k=",
        _koff_name(op.k_offset),
        " wait=",
        op.wait_value,
        " pf=",
        e.is_prefetch,
        sep="",
    )


# =============================================================================
# Phase header
# =============================================================================


def print_phase_header(label: StaticString, n_entries: Int):
    """Prints a phase section header with separator rule.

    Args:
        label: Phase name to display (e.g. "KERNEL").
        n_entries: Number of entries that follow under this phase.
    """
    print(label, " (", n_entries, " entries)", sep="")
    print("-" * 80)


# =============================================================================
# Wait-count summary
# =============================================================================


@fieldwise_init
struct WaitCounts(Copyable, Movable):
    """Per-phase wait/barrier counts, populated by repeated `add` calls.

    Walk a phase with a comptime for and call `counts.add(entry, i, n)`
    per entry; `n_combined` tracks vm waits whose immediate successor
    is an lgkm wait (i.e., would coalesce into one
    `s_waitcnt vmcnt(N) lgkmcnt(M)` instruction at codegen time).
    """

    var n_vm: Int
    """Number of `s_waitcnt vmcnt` waits seen in the phase."""
    var n_lgkm: Int
    """Number of `s_waitcnt lgkmcnt` waits seen in the phase."""
    var n_combined: Int
    """Number of vm waits whose successor is an lgkm wait (codegen
    coalesces these into a single `s_waitcnt vmcnt(N) lgkmcnt(M)`)."""
    var n_barrier: Int
    """Number of `s_barrier` instructions in the phase."""
    var n_sched_barrier: Int
    """Number of `s_sched_group_barrier` hints in the phase."""

    @staticmethod
    def empty() -> Self:
        """Returns a `WaitCounts` with every counter set to zero.

        Returns:
            A zero-initialized `WaitCounts`.
        """
        return Self(
            n_vm=0, n_lgkm=0, n_combined=0, n_barrier=0, n_sched_barrier=0
        )


def update_wait_counts(mut counts: WaitCounts, e: ScheduleEntry, next_tag: Int):
    """Updates counts for one entry; `next_tag` is the next entry's tag
    (or `_Ops.NONE.value` for the last entry).

    Args:
        counts: Counter aggregate updated in place.
        e: The schedule entry being inspected.
        next_tag: Tag of the next entry, used to detect coalesced
            vm+lgkm pairs. Pass `_Ops.NONE.value` for the last entry.
    """
    var t = e.op.tag
    if t == _Ops.WAIT_VM.value:
        counts.n_vm += 1
        if next_tag == _Ops.WAIT_LGKM.value:
            counts.n_combined += 1
    elif t == _Ops.WAIT_LGKM.value:
        counts.n_lgkm += 1
    elif t == _Ops.BARRIER.value:
        counts.n_barrier += 1
    elif t == _Ops.SCHEDULE_BARRIER.value:
        counts.n_sched_barrier += 1


def print_wait_counts(label: StaticString, c: WaitCounts):
    """Prints a one-line summary of a `WaitCounts` value.

    Args:
        label: Phase label printed in front of the counts.
        c: Counter aggregate to summarize.
    """
    print(
        "  ",
        label,
        ": ",
        c.n_vm,
        " vmcnt, ",
        c.n_lgkm,
        " lgkmcnt (",
        c.n_combined,
        " coalesced vm+lgkm), ",
        c.n_barrier,
        " s_barrier, ",
        c.n_sched_barrier,
        " sched_barrier",
        sep="",
    )
