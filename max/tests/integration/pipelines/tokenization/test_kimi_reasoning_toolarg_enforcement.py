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
"""Offline CPU test that the Kimi ``tool_choice=auto`` grammar enforces tool-call
argument schemas even when the tool section is opened straight from the reasoning
block, with no closing ``</think>``.

The ``auto`` grammar (``get_builtin_structural_tag("kimi", ...,
tool_choice="auto", reasoning=True)``) is a reasoning prefix followed by a
tool-call section triggered by ``<|tool_calls_section_begin|>``. Kimi may open
that section directly from thinking without emitting a closing ``</think>`` (see
kimik2_5/reasoning.py: a reasoning span ends on ``</think>`` OR
``<|tool_calls_section_begin|>``), so argument enforcement must arm on the section
marker whether or not a ``</think>`` precedes it. The reasoning prefix is
therefore optional and excludes the tool markers, so it never absorbs the section
marker as free reasoning text. ``tool_choice=required`` has no reasoning prefix
and enforces from the first token.

The test compiles the real StructuralTag over a synthetic byte + Kimi-marker RAW
vocab (no torch/server/GPU) and asserts the argument-value mask is TIGHT -- and
schema-violating tokens are rejected -- in three situations:

* straight ``</think>``-first driving,
* spec-decode-style fork / rollback of the matcher, and
* the section marker opened without a preceding ``</think>``,

across the schema constructs that matter for tool args (enum, integer,
$ref/$defs, anyOf), for both ``auto`` and ``required``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max._core import xgrammar as xgr
from max._xgrammar import get_builtin_structural_tag

# Kimi structural markers. In the real tiktoken vocab each is a single special
# token id; we mirror that by adding each as one token in the synthetic vocab.
SECTION_BEGIN = "<|tool_calls_section_begin|>"
SECTION_END = "<|tool_calls_section_end|>"
CALL_BEGIN = "<|tool_call_begin|>"
CALL_END = "<|tool_call_end|>"
ARG_BEGIN = "<|tool_call_argument_begin|>"
THINK_START = "<think>"
THINK_END = "</think>"
IM_END = "<|im_end|>"

_MARKERS = [
    SECTION_BEGIN,
    SECTION_END,
    CALL_BEGIN,
    CALL_END,
    ARG_BEGIN,
    THINK_START,
    THINK_END,
    IM_END,
]


def _vocab() -> tuple[list[bytes], dict[str, int]]:
    """256 single-byte tokens + one token per Kimi marker + a stop token."""
    toks: list[bytes] = [bytes([i]) for i in range(256)]
    marker_ids: dict[str, int] = {}
    for m in _MARKERS:
        marker_ids[m] = len(toks)
        toks.append(m.encode("utf-8"))
    marker_ids["<STOP>"] = len(toks)
    toks.append(b"<STOP>")
    return toks, marker_ids


def _tool(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": "f", "parameters": schema},
    }


def _compiler() -> tuple[Any, list[bytes], dict[str, int]]:
    vocab, marker_ids = _vocab()
    ti = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[marker_ids["<STOP>"]],
    )
    return xgr.GrammarCompiler(ti), vocab, marker_ids


def _compile(compiler: Any, schema: dict[str, Any], tool_choice: str) -> Any:
    reasoning = tool_choice == "auto"
    tag = get_builtin_structural_tag(
        "kimi",
        tools=[_tool(schema)],
        tool_choice=tool_choice,
        reasoning=reasoning,
    )
    return compiler.compile_structural_tag(tag.model_dump_json())


def _allowed(matcher: Any, vocab_size: int) -> np.ndarray:
    size = xgr.get_bitmask_size(vocab_size)
    bm = np.full((size,), -1, dtype=np.int32)
    matcher.fill_next_token_bitmask(bm)
    bits = np.zeros(vocab_size, dtype=bool)
    for t in range(vocab_size):
        bits[t] = bool((int(bm[t >> 5]) >> (t & 31)) & 1)
    return bits


def _allowed_count(matcher: Any, vocab_size: int) -> int:
    return int(_allowed(matcher, vocab_size).sum())


# A token stream is a list of ints (byte id or marker id). Helper to build one.
def _bytes_ids(s: str) -> list[int]:
    return list(s.encode("utf-8"))


def _accept_all(matcher: Any, ids: list[int]) -> tuple[bool, int]:
    """Accept every id; returns (all_ok, num_accepted)."""
    n = 0
    for t in ids:
        if not matcher.accept_token(t):
            return False, n
        n += 1
    return True, n


_ENUM = {
    "type": "object",
    "properties": {"u": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    "required": ["u"],
    "additionalProperties": False,
}
_INT = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
    "additionalProperties": False,
}
# $ref/$defs resolving to a string-enum leaf: reaches the same tight enum value
# via a $ref, exercising ref resolution at the value position.
_REF_ENUM = {
    "type": "object",
    "properties": {"u": {"$ref": "#/$defs/Unit"}},
    "$defs": {"Unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    "required": ["u"],
    "additionalProperties": False,
}
# anyOf(string-enum | null): once the value opens with a quote, only the
# string-enum branch is live, so the value must still be tight.
_ANYOF_ENUM = {
    "type": "object",
    "properties": {
        "u": {
            "anyOf": [
                {"type": "string", "enum": ["celsius", "fahrenheit"]},
                {"type": "null"},
            ]
        }
    },
    "required": ["u"],
    "additionalProperties": False,
}


def _envelope_prefix_ids(
    marker_ids: dict[str, int], value_open: str, *, with_reasoning: bool
) -> list[int]:
    """Token stream from turn start up to (not incl.) the arg VALUE position.

    ``value_open`` is the JSON typed so far, e.g. ``{"u":"`` (enum) or ``{"x":``
    (integer). ``with_reasoning`` prepends ``</think>`` (the reasoning-prefix
    terminator) required by the auto grammar; ``required`` has no such prefix.
    """
    ids: list[int] = []
    if with_reasoning:
        ids.append(marker_ids[THINK_END])  # close empty reasoning prefix
    ids.append(marker_ids[SECTION_BEGIN])  # trigger
    ids.append(marker_ids[CALL_BEGIN])
    ids += _bytes_ids("functions.f:")
    ids += _bytes_ids("0")  # call index (\d+)
    ids.append(marker_ids[ARG_BEGIN])
    ids += _bytes_ids(value_open)
    return ids


def _tool_call_ids(marker_ids: dict[str, int], value_open: str) -> list[int]:
    """The tool-call token stream *without* the reasoning prefix / trigger arming
    -- section marker, call marker, header, index, arg marker, then ``value_open``."""
    ids = [marker_ids[SECTION_BEGIN], marker_ids[CALL_BEGIN]]
    ids += _bytes_ids("functions.f:0")
    ids.append(marker_ids[ARG_BEGIN])
    ids += _bytes_ids(value_open)
    return ids


# TIGHT: the enum value admits only its two first bytes ('c','f'); anything
# materially larger is LOOSE (the AnyText/free region admits ~the whole vocab).
_TIGHT_ENUM = 2


# ===----------------------------------------------------------------------=== #
# Straight driving: token-by-token through the envelope, the value mask is TIGHT
# for both auto (after ``</think>``) and required.
# ===----------------------------------------------------------------------=== #
def test_control_straight_drive_is_tight() -> None:
    compiler, vocab, mids = _compiler()
    vsz = len(vocab)

    # auto + enum: at the value position only 'c'/'f' are admitted.
    m = xgr.GrammarMatcher(_compile(compiler, _ENUM, "auto"))
    ok, _ = _accept_all(
        m, _envelope_prefix_ids(mids, '{"u":"', with_reasoning=True)
    )
    auto_enum = _allowed_count(m, vsz)

    # required + enum: same, but with no reasoning prefix.
    mr = xgr.GrammarMatcher(_compile(compiler, _ENUM, "required"))
    okr, _ = _accept_all(
        mr, _envelope_prefix_ids(mids, '{"u":"', with_reasoning=False)
    )
    req_enum = _allowed_count(mr, vsz)

    # auto + integer: after '3' the mask must forbid '.' (mask == consume).
    mi = xgr.GrammarMatcher(_compile(compiler, _INT, "auto"))
    _accept_all(mi, _envelope_prefix_ids(mids, '{"x":3', with_reasoning=True))
    dot_masked = bool(_allowed(mi, vsz)[ord(".")])
    dot_consume = mi.fork().accept_token(ord("."))

    detail = (
        f"auto_enum_allowed={auto_enum} required_enum_allowed={req_enum} "
        f"int_after_3 dot_masked={dot_masked} dot_consume={dot_consume}"
    )
    assert ok and okr, f"control envelope driving must be accepted; {detail}"
    assert auto_enum == _TIGHT_ENUM, detail
    assert req_enum == _TIGHT_ENUM, detail
    assert dot_masked is False and dot_consume is False, detail


# ===----------------------------------------------------------------------=== #
# Fork / rollback stability: spec-decode-style fork, snapshot, and rollback of the
# matcher must leave the value mask TIGHT -- these matcher operations do not
# loosen enforcement.
# ===----------------------------------------------------------------------=== #
def test_fork_snapshot_rollback_stay_tight() -> None:
    compiler, vocab, mids = _compiler()
    vsz = len(vocab)
    prefix = _envelope_prefix_ids(mids, '{"u":"', with_reasoning=True)

    # (a) fork+advance-through-drafts+discard at every step, then accept real.
    m = xgr.GrammarMatcher(_compile(compiler, _ENUM, "auto"))
    for i, t in enumerate(prefix):
        fk = m.fork()
        _accept_all(fk, prefix[i:])  # walk remaining as speculative drafts
        del fk
        assert m.accept_token(t), f"real accept rejected at {i}"
    fork_discard = _allowed_count(m, vsz)

    # (b) marker-as-bonus: a fork validates the whole draft batch (trigger +
    # call + args); the persistent matcher then accepts the same batch, with the
    # section-begin trigger landing mid-batch (as a bonus token would).
    m2 = xgr.GrammarMatcher(_compile(compiler, _ENUM, "auto"))
    _accept_all(m2, [mids[THINK_END]])
    batch = _tool_call_ids(mids, '{"u":"')
    fk = m2.fork()
    _accept_all(fk, batch + _bytes_ids("c"))
    del fk
    ok_b, _ = _accept_all(m2, batch)
    marker_bonus = _allowed_count(m2, vsz)

    # (c) rollback across the trigger boundary, then re-drive.
    m3 = xgr.GrammarMatcher(
        _compile(compiler, _ENUM, "auto"), max_rollback_tokens=64
    )
    _accept_all(m3, [mids[THINK_END]])
    mid = [mids[SECTION_BEGIN], mids[CALL_BEGIN]] + _bytes_ids("fun")
    _accept_all(m3, mid)
    m3.rollback(len(mid))  # undo across the trigger
    ok_c, _ = _accept_all(m3, _tool_call_ids(mids, '{"u":"'))
    rollback_redrive = _allowed_count(m3, vsz)

    detail = (
        f"fork_discard_allowed={fork_discard} "
        f"marker_bonus_allowed={marker_bonus} "
        f"rollback_redrive_allowed={rollback_redrive} (TIGHT={_TIGHT_ENUM})"
    )
    assert fork_discard == _TIGHT_ENUM, detail
    assert ok_b and marker_bonus == _TIGHT_ENUM, detail
    assert ok_c and rollback_redrive == _TIGHT_ENUM, detail


# ===----------------------------------------------------------------------=== #
# Section marker without ``</think>``: opening the tool call directly from the
# reasoning block (no closing ``</think>``) must still arm enforcement -- the
# value mask is TIGHT and schema-violating tokens are rejected. Covered across
# enum, integer, $ref/$defs, and anyOf, for both auto and required (which has no
# reasoning prefix and enforces from the first token).
# ===----------------------------------------------------------------------=== #
def test_section_marker_without_think_is_enforced() -> None:
    compiler, vocab, mids = _compiler()
    vsz = len(vocab)

    # (label, schema, value_open, after, bad_byte, tight_max): drive the tool
    # call to a constrained value position, accept ``after`` extra value bytes,
    # then require the value mask is tight (<= tight_max) and ``bad_byte`` (a
    # schema violation) is rejected.
    flip_cases = [
        ("enum", _ENUM, '{"u":"', "", ord("s"), _TIGHT_ENUM),  # 'symlink'
        ("integer", _INT, '{"x":', "3", ord("."), 32),  # 3.7
        ("ref_defs_enum", _REF_ENUM, '{"u":"', "", ord("s"), _TIGHT_ENUM),
        ("anyof_enum", _ANYOF_ENUM, '{"u":"', "", ord("s"), _TIGHT_ENUM),
    ]

    def drive_no_think(
        schema: dict[str, Any], value_open: str, after: str, tool_choice: str
    ) -> tuple[int, Any]:
        """Drive SECTION_BEGIN .. value with NO preceding </think>."""
        m = xgr.GrammarMatcher(_compile(compiler, schema, tool_choice))
        ok, n = _accept_all(m, _tool_call_ids(mids, value_open))
        assert ok, f"{tool_choice} driving rejected ({n} accepted)"
        if after:
            ok2, _ = _accept_all(m, _bytes_ids(after))
            assert ok2, f"{tool_choice} value drive rejected"
        return _allowed_count(m, vsz), m

    # Grammar-shape property (check once): at the very start the section marker is
    # admitted as reasoning *text* and </think> is admitted -- trigger not armed.
    start = _allowed(xgr.GrammarMatcher(_compile(compiler, _ENUM, "auto")), vsz)
    assert bool(start[mids[SECTION_BEGIN]]), (
        "section marker not admitted as text"
    )
    assert bool(start[mids[THINK_END]]), "</think> not admitted at start"

    for label, schema, value_open, after, bad, tight_max in flip_cases:
        auto_allowed, m_auto = drive_no_think(schema, value_open, after, "auto")
        auto_admits_bad = m_auto.fork().accept_token(bad)
        req_allowed, m_req = drive_no_think(
            schema, value_open, after, "required"
        )
        req_admits_bad = m_req.fork().accept_token(bad)

        # Both modes arm enforcement -> tight value mask + violating byte
        # rejected (a loose mask would admit ~the whole vocab).
        detail = (
            f"{label}: auto_allowed={auto_allowed} "
            f"auto_rejects_bad={not auto_admits_bad} "
            f"required_allowed={req_allowed} "
            f"required_rejects_bad={not req_admits_bad}"
        )
        assert auto_allowed <= tight_max, detail
        assert auto_admits_bad is False, detail
        assert req_allowed <= tight_max, detail
        assert req_admits_bad is False, detail
