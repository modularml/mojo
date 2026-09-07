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
"""Implements UAX #29 grapheme cluster boundary detection.

This module provides the core algorithm for determining grapheme cluster
boundaries in Unicode text, following the Unicode Standard Annex #29
"Unicode Text Segmentation" specification.

Reference: https://unicode.org/reports/tr29/#Grapheme_Cluster_Boundary_Rules

Performance TODOs:
- TODO: Replace binary search with a two-stage trie for O(1) property
  lookup (~11 comparisons per codepoint → 2 indexed reads, similar data
  size).
- TODO: Merge GBP and InCB into a single lookup table (4 bits + 2 bits
  fit in one byte) to halve lookup cost for non-ASCII codepoints.
"""

from std.builtin.globals import global_constant
from std.collections.string._grapheme_break_lookups import (
    _GBP_RANGE_STARTS,
    _GBP_RANGE_VALUES,
    _INCB_RANGE_STARTS,
    _INCB_RANGE_VALUES,
)
from std.collections.string._utf8 import _is_utf8_continuation_byte
from std.collections import Span


# ===----------------------------------------------------------------------=== #
# ASCII fast-path helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _is_safe_ascii_for_grapheme(b: Byte) -> Bool:
    """Return `True` if byte `b` is in the safe-ASCII range for grapheme
    iteration.

    Safe-ASCII bytes (U+0020..U+007E, printable ASCII excluding DEL) all
    have `GBP_OTHER`. A safe-ASCII byte is always its own grapheme
    cluster when the previous codepoint's GBP is not `PREPEND` and the
    next codepoint (if any) is also safe-ASCII — GB9/GB9a/GB9b only
    trigger with non-ASCII GBPs.
    """
    return b >= 0x20 and b <= 0x7E


# ===----------------------------------------------------------------------=== #
# Grapheme Break Property enum constants
# ===----------------------------------------------------------------------=== #

comptime GBP_OTHER: UInt8 = 0
comptime GBP_CR: UInt8 = 1
comptime GBP_LF: UInt8 = 2
comptime GBP_CONTROL: UInt8 = 3
comptime GBP_EXTEND: UInt8 = 4
comptime GBP_ZWJ: UInt8 = 5
comptime GBP_REGIONAL_INDICATOR: UInt8 = 6
comptime GBP_PREPEND: UInt8 = 7
comptime GBP_SPACINGMARK: UInt8 = 8
comptime GBP_L: UInt8 = 9
comptime GBP_V: UInt8 = 10
comptime GBP_T: UInt8 = 11
comptime GBP_LV: UInt8 = 12
comptime GBP_LVT: UInt8 = 13
comptime GBP_EXTENDED_PICTOGRAPHIC: UInt8 = 14

# ===----------------------------------------------------------------------=== #
# Indic_Conjunct_Break property constants (for GB9c)
# ===----------------------------------------------------------------------=== #

comptime INCB_NONE: UInt8 = 0
comptime INCB_CONSONANT: UInt8 = 1
comptime INCB_EXTEND: UInt8 = 2
comptime INCB_LINKER: UInt8 = 3


# ===----------------------------------------------------------------------=== #
# Lookup functions
# ===----------------------------------------------------------------------=== #


@always_inline
def _range_lookup[
    starts_table: Array[UInt32, _],
    values_table: Array[UInt8, _],
    default: UInt8,
](cp: UInt32) -> UInt8:
    """Look up a property value for a codepoint using binary search over a
    sorted range table.

    Parameters:
        starts_table: Comptime sorted array of range start codepoints.
        values_table: Comptime array of property values parallel to starts.
        default: Value to return if `cp` falls before all ranges.

    Args:
        cp: The Unicode codepoint value.

    Returns:
        The property value for this codepoint.
    """
    var starts = Span(global_constant[starts_table]())
    var values = Span(global_constant[values_table]())

    # Note: Span.binary_search_by finds an *exact* match, but we need
    # the largest start <= cp (upper_bound - 1). A partition_point /
    # upper_bound on Span would be a better fit here if one existed.
    # Binary search for the largest start <= cp
    var lo = 0
    var hi = len(starts)
    while lo < hi:
        var mid = (lo + hi) // 2
        if starts[mid] <= cp:
            lo = mid + 1
        else:
            hi = mid

    if lo == 0:
        return default
    return values[lo - 1]


@always_inline
def _gbp_lookup(cp: UInt32) -> UInt8:
    """Look up the Grapheme Break Property for a codepoint.

    Args:
        cp: The Unicode codepoint value.

    Returns:
        The GBP enum value for this codepoint.
    """
    # Fast path: avoid the global_constant access + binary search for the
    # common case of ASCII/Latin-1 printable characters, which are all Other.
    # U+00A9 (©) is the first Extended_Pictographic codepoint in the table.
    if cp < 0x00A9:
        if cp >= 0x20 and cp < 0x7F:
            return GBP_OTHER
        if cp >= 0xA0:
            return GBP_OTHER
    return _range_lookup[_GBP_RANGE_STARTS, _GBP_RANGE_VALUES, GBP_OTHER](cp)


@always_inline
def _incb_lookup(cp: UInt32) -> UInt8:
    """Look up the Indic_Conjunct_Break property for a codepoint.

    Args:
        cp: The Unicode codepoint value.

    Returns:
        The InCB enum value for this codepoint.
    """
    # Fast path: avoid the global_constant access + binary search for
    # non-Indic codepoints. InCB=Consonant/Linker only appear at U+0915+
    # (Devanagari) and later. The table does contain InCB=Extend entries
    # below U+0900 (e.g. U+0300 combining marks), but those only reach the
    # `prev_in_incb && incb == INCB_EXTEND` branch in `_is_grapheme_break`,
    # and `prev_in_incb` is only set by an InCB=Consonant -- which, by the
    # same codepoint bound, can never precede one of these low codepoints.
    # Returning INCB_NONE is therefore equivalent to the real lookup value.
    if cp < 0x0900:
        return INCB_NONE
    return _range_lookup[_INCB_RANGE_STARTS, _INCB_RANGE_VALUES, INCB_NONE](cp)


# ===----------------------------------------------------------------------=== #
# Grapheme break state machine
# ===----------------------------------------------------------------------=== #


struct _GraphemeBreakState(ImplicitlyCopyable):
    """Tracks state for UAX #29 grapheme cluster boundary detection.

    The state machine processes codepoints sequentially and determines
    whether a grapheme cluster boundary exists between consecutive
    codepoints.
    """

    var prev_gbp: UInt8
    """GBP of the previous codepoint."""

    var ri_count_is_odd: Bool
    """Whether we have seen an odd number of consecutive Regional Indicators."""

    var in_ext_pict_seq: Bool
    """Whether we are in an Extended_Pictographic Extend* sequence (for GB11)."""

    var seen_incb_linker: Bool
    """Whether we have seen an InCB Linker in the current conjunct (for GB9c)."""

    var in_incb_conjunct: Bool
    """Whether we are in an InCB Consonant [{Extend|InCB_Linker}*] seq."""

    def __init__(out self):
        """Initialize with start-of-text state."""
        # GB4 breaks after Control, so using GBP_CONTROL here ensures the
        # first codepoint always starts a new grapheme cluster (SOT = break).
        self.prev_gbp = GBP_CONTROL
        self.ri_count_is_odd = False
        self.in_ext_pict_seq = False
        self.seen_incb_linker = False
        self.in_incb_conjunct = False


@always_inline
def _reset_grapheme_state_to_other(mut state: _GraphemeBreakState):
    """Mutate `state` to the configuration after consuming an `Other`
    codepoint.

    Shared by the ASCII fast paths. After consuming a safe-ASCII byte,
    `prev_gbp` is `Other`, there is no in-progress RI/ZWJ/InCB sequence,
    and the state is ready to decide the next grapheme boundary.
    """
    state.prev_gbp = GBP_OTHER
    state.ri_count_is_odd = False
    state.in_ext_pict_seq = False
    state.in_incb_conjunct = False
    state.seen_incb_linker = False


@always_inline
def _is_grapheme_break(mut state: _GraphemeBreakState, cp: UInt32) -> Bool:
    """Determine if there is a grapheme cluster boundary before `cp`.

    Implements the full set of UAX #29 grapheme cluster boundary rules.
    Updates `state` for the next call.

    Args:
        state: The current segmentation state (modified in place).
        cp: The codepoint value to test.

    Returns:
        True if there is a grapheme cluster boundary before `cp`.
    """
    var gbp = _gbp_lookup(cp)
    var prev = state.prev_gbp

    # Update state for the next iteration (done early so we can return)
    var prev_in_ext_pict_seq = state.in_ext_pict_seq
    var prev_ri_odd = state.ri_count_is_odd
    var prev_in_incb = state.in_incb_conjunct
    var prev_seen_linker = state.seen_incb_linker

    # Update RI parity tracking
    if gbp == GBP_REGIONAL_INDICATOR:
        state.ri_count_is_odd = not prev_ri_odd
    else:
        state.ri_count_is_odd = False

    # Update Extended_Pictographic sequence tracking (for GB11)
    if gbp == GBP_EXTENDED_PICTOGRAPHIC:
        state.in_ext_pict_seq = True
    elif gbp == GBP_EXTEND:
        pass  # Keep in_ext_pict_seq unchanged
    elif gbp == GBP_ZWJ:
        pass  # Keep in_ext_pict_seq unchanged
    else:
        state.in_ext_pict_seq = False

    # Update InCB conjunct tracking (for GB9c)
    var incb = _incb_lookup(cp)
    if incb == INCB_CONSONANT:
        state.in_incb_conjunct = True
        state.seen_incb_linker = False
    elif prev_in_incb and incb == INCB_EXTEND:
        pass  # Stay in conjunct
    elif prev_in_incb and incb == INCB_LINKER:
        state.seen_incb_linker = True
    elif prev_in_incb and gbp == GBP_EXTEND:
        # Extend characters also continue the conjunct
        pass
    elif prev_in_incb and gbp == GBP_ZWJ:
        pass
    else:
        state.in_incb_conjunct = False
        state.seen_incb_linker = False

    state.prev_gbp = gbp

    # === Apply break rules ===

    # GB3: Do not break between a CR and LF.
    if prev == GBP_CR and gbp == GBP_LF:
        return False

    # GB4: Break after Controls (CR, LF, Control).
    if prev == GBP_CR or prev == GBP_LF or prev == GBP_CONTROL:
        return True

    # GB5: Break before Controls (CR, LF, Control).
    if gbp == GBP_CR or gbp == GBP_LF or gbp == GBP_CONTROL:
        return True

    # GB6: Do not break Hangul syllable sequences.
    # L x (L | V | LV | LVT)
    if prev == GBP_L and (
        gbp == GBP_L or gbp == GBP_V or gbp == GBP_LV or gbp == GBP_LVT
    ):
        return False

    # GB7: (LV | V) x (V | T)
    if (prev == GBP_LV or prev == GBP_V) and (gbp == GBP_V or gbp == GBP_T):
        return False

    # GB8: (LVT | T) x T
    if (prev == GBP_LVT or prev == GBP_T) and gbp == GBP_T:
        return False

    # GB9: Do not break before Extend or ZWJ.
    if gbp == GBP_EXTEND or gbp == GBP_ZWJ:
        return False

    # GB9a: Do not break before SpacingMarks.
    if gbp == GBP_SPACINGMARK:
        return False

    # GB9b: Do not break after Prepend.
    if prev == GBP_PREPEND:
        return False

    # GB9c: Do not break within Indic conjunct clusters.
    # \p{InCB=Consonant} [{Extend|InCB_Linker}* InCB_Linker {Extend|InCB_Linker}*] x \p{InCB=Consonant}
    if prev_in_incb and prev_seen_linker and incb == INCB_CONSONANT:
        return False

    # GB11: Do not break within emoji modifier sequences or emoji ZWJ sequences.
    # \p{Extended_Pictographic} Extend* ZWJ x \p{Extended_Pictographic}
    if (
        prev_in_ext_pict_seq
        and prev == GBP_ZWJ
        and gbp == GBP_EXTENDED_PICTOGRAPHIC
    ):
        return False

    # GB12/GB13: Do not break within emoji flag sequences.
    # sot (RI RI)* RI x RI
    # [^RI] (RI RI)* RI x RI
    if gbp == GBP_REGIONAL_INDICATOR and prev == GBP_REGIONAL_INDICATOR:
        # Don't break if an odd number of RIs precede this one (pair incomplete).
        # Break if an even number precede (all pairs complete, start new pair).
        return not prev_ri_odd

    # GB999: Otherwise, break everywhere.
    return True


# ===----------------------------------------------------------------------=== #
# Reverse-scan helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _decode_previous_codepoint(
    span: ImmSpan[Byte, _], end: Int
) -> Tuple[UInt32, Int]:
    """Decode the codepoint whose last byte is at `end - 1`.

    Walks backward from `end - 1` over continuation bytes to find the
    codepoint's lead byte, then decodes it forward.

    Args:
        span: The byte span containing valid UTF-8.
        end: One past the last byte of the codepoint to decode. Must be
            in `(0, len(span)]`.

    Returns:
        A tuple `(codepoint_value, byte_length)` for the decoded codepoint.
    """
    var pos = end - 1
    while pos > 0 and _is_utf8_continuation_byte(span[pos]):
        pos -= 1
    var byte_len = end - pos
    var cp, _ = Codepoint.unsafe_decode_utf8_codepoint(span[pos:end])
    return (cp.to_u32(), byte_len)


def _find_safe_grapheme_start(span: ImmSpan[Byte, _], end: Int) -> Int:
    """Find a byte offset `<= end` that is a guaranteed grapheme boundary.

    Used by reverse grapheme iteration: given the end of a range, walk
    backward until we find a position where the UAX #29 state machine can
    safely restart and produce the same grapheme boundaries it would from
    a full-text forward scan.

    A position `p` is a guaranteed boundary when either:
      - `p == 0` (start-of-text, GB1).
      - The codepoint at `p` has GBP ∈ {Control, CR, LF} (GB5), except the
        CR × LF case (GB3).
      - The codepoint immediately before `p` has GBP ∈ {Control, CR, LF}
        (GB4), again excluding CR × LF.

    In pathological inputs (e.g., a long run with no Control/CR/LF), this
    falls back to `0`.

    Args:
        span: The byte span, valid UTF-8.
        end: The byte offset to back up from. Must be in `[0, len(span)]`.

    Returns:
        A byte offset `safe` with `0 <= safe <= end` that is a guaranteed
        grapheme-cluster boundary in `span`.
    """
    var safe = end
    var cur_gbp = GBP_OTHER
    var has_cur = False
    while safe > 0:
        var cp: UInt32
        var n: Int
        cp, n = _decode_previous_codepoint(span, safe)
        var prev_gbp = _gbp_lookup(cp)
        var candidate = safe - n

        # A boundary exists between `prev_gbp` (the codepoint just before
        # `safe`, i.e. at byte offset `candidate`) and `cur_gbp` (the
        # codepoint at byte offset `safe`, if we've already seen one) iff
        # either gbp is Control/CR/LF and we're not in the CR × LF
        # no-break case.
        if has_cur:
            var is_crlf = prev_gbp == GBP_CR and cur_gbp == GBP_LF
            if not is_crlf and (
                prev_gbp == GBP_CONTROL
                or prev_gbp == GBP_CR
                or prev_gbp == GBP_LF
                or cur_gbp == GBP_CONTROL
                or cur_gbp == GBP_CR
                or cur_gbp == GBP_LF
            ):
                return safe

        cur_gbp = prev_gbp
        has_cur = True
        safe = candidate

    return 0
