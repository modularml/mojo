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
"""Deterministic, model-independent grammar/mask tests for response_format.

Regression for the runaway-output incident: a
``response_format.json_schema.schema`` that omits the root ``type`` compiles
(via llguidance) to a grammar whose START state permits a bare, unbounded
top-level value -- including a JSON string. A model that degenerates into a
repetition loop inside that string can never emit the only terminator (the
closing quote), so EOS is never unmasked and generation runs to
``max_length`` with ``finish_reason="length"``.

These tests assert the *token bitmask* directly with a tiny byte+special
tokenizer (the same pattern as the gemma4 ``test_tool_parser.py`` grammar
tests), so they need no GPU and no model: the broken mask is purely a
function of the schema and the grammar backend.
"""

from __future__ import annotations

from typing import Any

import llguidance.numpy as ln
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.serve.parser.tool_call_normalization import (
    normalize_response_format_schema,
)

_N_VOCAB = 256
_EOS = 0


class _ByteTokenizer:
    """Raw byte tokenizer: token IDs 0-255 map to single bytes.

    Sufficient to exercise JSON-schema grammars, which operate over the raw
    UTF-8 byte stream. ID 0 doubles as EOS.
    """

    eos_token_id: int = _EOS
    bos_token_id: int | None = None
    unk_token_id: int | None = None

    def __init__(self) -> None:
        self.tokens: list[bytes] = [bytes([i]) for i in range(256)]

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return None

    def __call__(self, s: bytes | str) -> list[int]:
        if isinstance(s, str):
            s = s.encode("utf-8")
        return list(s)


def _ll_tokenizer() -> LLTokenizer:
    return LLTokenizer(TokenizerWrapper(_ByteTokenizer()), n_vocab=_N_VOCAB)


def _allowed_tokens(matcher: LLMatcher) -> set[int]:
    """Return the set of token IDs the matcher permits as the next token."""
    bitmask = ln.allocate_token_bitmask(1, _N_VOCAB)
    ln.fill_next_token_bitmask(matcher, bitmask, index=0)
    return {
        t for t in range(_N_VOCAB) if (int(bitmask[0, t // 32]) >> (t % 32)) & 1
    }


def _matcher_for(schema: dict[str, Any]) -> LLMatcher:
    grammar = LLMatcher.grammar_from_json_schema(
        schema, overrides={"whitespace_pattern": ""}
    )
    matcher = LLMatcher(_ll_tokenizer(), grammar)
    assert not matcher.is_error(), matcher.get_error()
    return matcher


_QUOTE = ord('"')
_OPEN_BRACE = ord("{")


def test_missing_root_type_permits_bare_top_level_string() -> None:
    """Untyped root schema lets the model open an unbounded top-level string.

    This is the runaway trigger: the START state allows a bare ``"`` (opening
    a top-level string), and once inside, EOS stays masked indefinitely while
    string content is emitted -- so a looping model never terminates.
    """
    matcher = _matcher_for({"properties": {"x": {}}})

    start_allowed = _allowed_tokens(matcher)
    # The bug fingerprint: a bare top-level string can be opened, and the
    # schema does not pin the response to an object.
    assert _QUOTE in start_allowed
    assert start_allowed != {_OPEN_BRACE}

    # Open the bare top-level string and emit a long repetition. EOS must
    # never become available -- the only escape is the (unlikely) close-quote.
    assert matcher.try_consume_tokens([_QUOTE]) == 1
    for _ in range(500):
        assert matcher.try_consume_tokens(list(b"shell ")) == 6
    inside_string = _allowed_tokens(matcher)
    assert _EOS not in inside_string, (
        "EOS must be masked inside an unbounded top-level string"
    )
    assert not matcher.is_accepting()


def test_normalized_root_type_forces_object_and_terminates() -> None:
    """After normalization the START state requires ``{`` and can reach EOS.

    With ``type: object`` injected, a bare top-level string is no longer
    permitted, and a valid object terminates -- EOS becomes available.
    """
    schema = normalize_response_format_schema({"properties": {"x": {}}})
    assert schema["type"] == "object"
    matcher = _matcher_for(schema)

    start_allowed = _allowed_tokens(matcher)
    # Object wrapper is forced: only ``{`` may start the response, and a bare
    # top-level string is rejected.
    assert start_allowed == {_OPEN_BRACE}
    assert _QUOTE not in start_allowed

    # A minimal valid object ``{"x":1}`` must terminate and unmask EOS.
    assert matcher.try_consume_tokens(list(b'{"x":1}')) == 7
    assert matcher.is_accepting()
    assert _EOS in _allowed_tokens(matcher)


def test_missing_type_bare_string_rejected_after_normalization() -> None:
    """The normalized grammar rejects a bare top-level string outright."""
    schema = normalize_response_format_schema({"properties": {"x": {}}})
    matcher = _matcher_for(schema)
    # Opening a top-level string is not a valid object start.
    assert matcher.try_consume_tokens([_QUOTE]) == 0


def test_nested_untyped_object_subschema_anchored_after_normalization() -> None:
    """Recursive normalization anchors a nested object-shaped subschema.

    Before the fix, a nested untyped-but-object-shaped value (``inner``) also
    permits a bare unbounded string. After recursive inference the inner value
    is object-anchored: at the ``inner`` value position only ``{`` is allowed,
    not a bare ``"``.
    """
    schema = normalize_response_format_schema(
        {"properties": {"inner": {"properties": {"y": {}}}}}
    )
    matcher = _matcher_for(schema)

    # Walk to the inner value position: ``{"inner":``.
    assert matcher.try_consume_tokens(list(b'{"inner":')) == 9
    inner_allowed = _allowed_tokens(matcher)
    assert _OPEN_BRACE in inner_allowed
    assert _QUOTE not in inner_allowed, (
        "nested inner value must be object-anchored, not a bare string"
    )
