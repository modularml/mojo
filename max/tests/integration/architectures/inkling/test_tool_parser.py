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

"""Tests for the Inkling tool-call parser.

Every input is in post-detokenization form: only
``<|content_invoke_tool_json|>`` survives, so a call reaches the parser as
``NAME<|content_invoke_tool_json|>{"name":...,"args":{...}}`` with nothing
separating one call from the next, or from preceding text.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import Any

import pytest
from _grammar_harness import (
    END_MESSAGE,
    MESSAGE_MODEL,
    STOP,
    TEXT,
    THINKING,
    UNIT_ENUM,
    Grammar,
    call,
    obj,
    opened,
    tool,
)
from max.pipelines.architectures.inkling.reasoning import (
    InklingReasoningParser,
)
from max.pipelines.architectures.inkling.tokenizer import TOOL_CALL_JSON_MARKER
from max.pipelines.architectures.inkling.tool_parser import InklingToolParser
from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import ParsedToolCallDelta


def _wire(name: str, args_json: str) -> str:
    """Renders one tool call exactly as the parser receives it."""
    return f'{name}{TOOL_CALL_JSON_MARKER}{{"name":{json.dumps(name)},"args":{args_json}}}'


def _schemas(*names: str) -> dict[str, dict[str, Any]]:
    """Declared tool schemas as the router supplies them; only keys matter."""
    return {n: {"type": "object", "properties": {}} for n in names}


def _assemble_streamed(
    parser: InklingToolParser, token_chunks: Sequence[str]
) -> tuple[str, list[dict[str, str]]]:
    """Reconstructs what a streaming client sees, per tool-call index."""
    content: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    for chunk in token_chunks:
        result = parser.parse_delta(chunk)
        if not result:
            continue
        for delta in result:
            if delta.content is not None:
                content.append(delta.content)
                continue
            call = calls.setdefault(
                delta.index, {"id": "", "name": "", "arguments": ""}
            )
            if delta.id is not None:
                call["id"] = delta.id
            if delta.name is not None:
                call["name"] = delta.name
            if delta.arguments is not None:
                call["arguments"] += delta.arguments
    return "".join(content), [calls[i] for i in sorted(calls)]


def _assert_streaming_matches_complete(
    response: str,
    declared: Sequence[str],
    chunks: Sequence[str],
    expected_content: str | None,
) -> None:
    """Asserts streaming ``chunks`` reproduce ``parse_complete(response)``.

    Arguments compare byte for byte: both paths forward the model's own ``args``
    text, so agreeing only after a JSON round-trip would not be enough.
    """
    expected = InklingToolParser().parse_complete(response)
    assert expected.content == expected_content

    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas(*declared))
    content, streamed = _assemble_streamed(parser, chunks)

    assert content == (expected.content or "")
    assert [c["name"] for c in streamed] == [
        tc.name for tc in expected.tool_calls
    ]
    assert [c["arguments"] for c in streamed] == [
        tc.arguments for tc in expected.tool_calls
    ]


_SINGLE_CALL = _wire("get_weather", '{"city":"SF"}')
_TWO_CALLS = _SINGLE_CALL + _wire("add", '{"a":2,"b":3}')
_TEXT_THEN_CALL = "Let me check both." + _SINGLE_CALL


# ---------------------------------------------------------------------------
# parse_complete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args_json", "expected"),
    [
        pytest.param(
            '{"city":"SF","days":3,"metric":true}',
            {"city": "SF", "days": 3, "metric": True},
            id="typed-args",
        ),
        pytest.param("{}", {}, id="empty-args"),
    ],
)
def test_parse_complete_single_call(
    args_json: str, expected: dict[str, Any]
) -> None:
    result = InklingToolParser().parse_complete(_wire("get_weather", args_json))

    assert result.content is None
    assert len(result.tool_calls) == 1

    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert call.id.startswith("call_")
    # OpenAI's contract: ``arguments`` is a JSON *string*, not an object.
    assert isinstance(call.arguments, str)
    assert json.loads(call.arguments) == expected


def test_parse_complete_unicode_args_round_trip() -> None:
    """Arguments are serialized with ``ensure_ascii=False``: no ``\\uXXXX``."""
    result = InklingToolParser().parse_complete(
        _wire("say", '{"text":"café ü 你好"}')
    )

    arguments = result.tool_calls[0].arguments
    assert "café ü 你好" in arguments
    assert "\\u" not in arguments


@pytest.mark.parametrize(
    ("payload_args", "expected"),
    [
        pytest.param('{"city":"SF"}', '{"city":"SF"}', id="compact"),
        pytest.param('{"city": "SF"}', '{"city": "SF"}', id="spaced"),
        pytest.param("5", "5", id="scalar-root"),
    ],
)
def test_parse_complete_arguments_are_the_models_own_bytes(
    payload_args: str, expected: str
) -> None:
    """``arguments`` is forwarded verbatim, not re-serialized.

    The grammar caps interior whitespace rather than forbidding it, and a tool
    without ``parameters`` maps to the unconstrained schema, so both a spaced
    object and a scalar root are legal generations. Re-serializing would
    rewrite them into something streaming never sends.
    """
    result = InklingToolParser().parse_complete(_wire("f", payload_args))

    assert result.tool_calls[0].arguments == expected


@pytest.mark.parametrize(
    ("payload", "expected_args"),
    [
        pytest.param(
            '{"name":"f", "args": {"a":1}}',
            '{"a":1}',
            id="whitespace-around-the-args-key",
        ),
        pytest.param(
            '{"name":"f","args":{"a":1},"extra":2}',
            '{"a":1}',
            id="key-after-args",
        ),
        pytest.param(
            '{"name":"f","args":{"a":1,"name":"not-the-tool"}}',
            '{"a":1,"name":"not-the-tool"}',
            id="name-property-nested-in-args",
        ),
    ],
)
def test_off_canonical_payloads_parse_the_same_either_way(
    payload: str, expected_args: str
) -> None:
    """Deviations from the canonical frame must not split the two paths.

    The grammar emits the frame as a const string, so none of these arise while
    enforcement holds. They are reachable with constrained decode disabled, or
    after a rejected token fails enforcement open mid-request -- and each used
    to yield a tool call non-streaming while streaming silently produced no
    call at all, or arguments that were not valid JSON.
    """
    response = f"f{TOOL_CALL_JSON_MARKER}{payload}"

    _assert_streaming_matches_complete(response, ("f",), list(response), None)

    result = InklingToolParser().parse_complete(response)
    assert result.tool_calls[0].arguments == expected_args


def test_parse_complete_plain_text_passes_through() -> None:
    response = "The forecast looks fine, no tools needed."
    result = InklingToolParser().parse_complete(response)

    assert result.content == response
    assert result.tool_calls == []


@pytest.mark.parametrize(
    ("args_json", "expected"),
    [
        pytest.param(
            r'{"q":"he said \"}\" and left","p":"back\\slash"}',
            {"q": 'he said "}" and left', "p": "back\\slash"},
            id="escaped-quote-and-backslash",
        ),
        pytest.param(
            r'{"q":"trailing backslash \\"}',
            {"q": "trailing backslash \\"},
            id="string-ending-in-escaped-backslash",
        ),
    ],
)
def test_parse_complete_brace_balancer_ignores_string_contents(
    args_json: str, expected: dict[str, Any]
) -> None:
    """A string ending in an escaped backslash must not swallow its quote."""
    result = InklingToolParser().parse_complete(_wire("search", args_json))

    assert json.loads(result.tool_calls[0].arguments) == expected


def test_parse_complete_brace_balancer_stops_at_the_next_call() -> None:
    """A braced string in call *n* must not run on into call *n+1*."""
    response = _wire("search", '{"q":"{{{"}') + _wire("add", '{"a":1}')
    result = InklingToolParser().parse_complete(response)

    assert [tc.name for tc in result.tool_calls] == ["search", "add"]
    assert json.loads(result.tool_calls[0].arguments) == {"q": "{{{"}
    assert json.loads(result.tool_calls[1].arguments) == {"a": 1}
    assert result.tool_calls[0].id != result.tool_calls[1].id


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param('{"name":"f","args":{,,}}', id="malformed-json"),
        pytest.param('{"args":{"a":1}}', id="no-name-key"),
        pytest.param("[1,2]", id="non-object-payload"),
        pytest.param('{"name":"f","args":{"a":1}', id="truncated-json"),
    ],
)
def test_parse_complete_raises_when_the_only_call_is_unusable(
    payload: str,
) -> None:
    response = f"f{TOOL_CALL_JSON_MARKER}{payload}"

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        InklingToolParser().parse_complete(response)


def test_parse_complete_skips_malformed_call_beside_a_valid_one() -> None:
    response = f'f{TOOL_CALL_JSON_MARKER}{{"name":"f","args":{{,,}}}}' + _wire(
        "add", '{"a":1}'
    )
    result = InklingToolParser().parse_complete(response)

    assert [tc.name for tc in result.tool_calls] == ["add"]
    assert json.loads(result.tool_calls[0].arguments) == {"a": 1}
    # Known wart: only the first surviving call's name is trimmed, so the
    # skipped call's bare name is left behind as content.
    assert result.content == "f"


# ---------------------------------------------------------------------------
# parse_delta (streaming)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "declared", "expected_content"),
    [
        pytest.param(_SINGLE_CALL, ("get_weather",), None, id="single-call"),
        pytest.param(_TWO_CALLS, ("get_weather", "add"), None, id="two-calls"),
        pytest.param(
            _TEXT_THEN_CALL,
            ("get_weather",),
            "Let me check both.",
            id="text-then-call",
        ),
        pytest.param(_wire("ping", "{}"), ("ping",), None, id="empty-args"),
    ],
)
def test_streaming_char_by_char_matches_parse_complete(
    response: str, declared: tuple[str, ...], expected_content: str | None
) -> None:
    """The general property, at the strictest possible split."""
    _assert_streaming_matches_complete(
        response, declared, list(response), expected_content
    )


def test_streaming_argument_deltas_concatenate_to_the_args_object() -> None:
    """OpenAI streaming contract: deltas carry only newly-arrived bytes and
    concatenate to exactly the payload's ``args`` object."""
    args_json = '{"city":"Chicago","unit":"fahrenheit"}'
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("get_weather"))

    deltas: list[ParsedToolCallDelta] = []
    for chunk in [
        "get_weather",
        TOOL_CALL_JSON_MARKER,
        '{"name":"get_',
        'weather","args":{"city":',
        '"Chicago","unit":',
        '"fahrenheit"}}',
    ]:
        deltas.extend(parser.parse_delta(chunk) or [])
    argument_deltas = [d.arguments for d in deltas if d.arguments is not None]

    assert all(argument_deltas)
    assert "".join(argument_deltas) == args_json
    assert json.loads("".join(argument_deltas)) == {
        "city": "Chicago",
        "unit": "fahrenheit",
    }


def test_streaming_undeclared_tool_name_surfaces_once_as_content() -> None:
    """Documented fallback: the router omits tools without ``parameters``, so
    such a name gets no holdback and reaches the client as content."""
    content, streamed = _assemble_streamed(
        InklingToolParser(), list(_SINGLE_CALL)
    )

    assert content == "get_weather"
    assert [c["name"] for c in streamed] == ["get_weather"]


@pytest.mark.parametrize(
    ("stream", "expected_content", "expected_names"),
    [
        pytest.param(
            "Let me check" + _SINGLE_CALL,
            "Let me check",
            ["get_weather"],
            id="exact-name-trimmed-out-of-a-fused-run",
        ),
        pytest.param(
            "please get more info",
            "please get more info",
            [],
            id="prefix-released-once-it-breaks",
        ),
        pytest.param(
            "please get", "please ", [], id="prefix-dropped-at-end-of-stream"
        ),
    ],
)
def test_streaming_name_holdback(
    stream: str, expected_content: str, expected_names: list[str]
) -> None:
    """Nothing separates content from the bare name, so a trailing run that
    still prefixes a declared tool is held back: released once a character
    breaks the prefix, dropped if the stream ends first, and trimmed at the
    exact declared name when content runs straight into it.
    """
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("get_weather"))

    content, streamed = _assemble_streamed(parser, list(stream))

    assert content == expected_content
    assert [c["name"] for c in streamed] == expected_names


def test_streaming_back_to_back_calls() -> None:
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("get_weather", "add"))

    content, streamed = _assemble_streamed(parser, list(_TWO_CALLS))

    # The second call's bare name sits between the two payloads.
    assert content == ""
    assert [c["name"] for c in streamed] == ["get_weather", "add"]
    assert streamed[0]["id"] != streamed[1]["id"]
    assert all(call["id"].startswith("call_") for call in streamed)


def test_streaming_incomplete_call_without_arguments_is_not_surfaced() -> None:
    """Generation cut off before ``,"args":`` must not leak a dangling call."""
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("emit"))

    chunks = list(_wire("emit", '{"v":1}')) + list(
        f'emit{TOOL_CALL_JSON_MARKER}{{"name":"emit",'
    )
    _, streamed = _assemble_streamed(parser, chunks)

    assert len(streamed) == 1
    assert json.loads(streamed[0]["arguments"]) == {"v": 1}


def test_reset_clears_declared_tool_names() -> None:
    """A second request that declares no tools must not inherit the first's."""
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("get_weather"))
    parser.parse_delta("get_weather")

    parser.reset()

    content, streamed = _assemble_streamed(parser, list(_SINGLE_CALL))
    assert content == "get_weather"
    assert [c["name"] for c in streamed] == ["get_weather"]


# ---------------------------------------------------------------------------
# multi-turn streaming
# ---------------------------------------------------------------------------

# Published Inkling marker ids.
_MARKER_IDS = {
    MESSAGE_MODEL: 200001,
    THINKING: 200008,
    END_MESSAGE: 200010,
    TOOL_CALL_JSON_MARKER: 200049,
}


def _encode(text: str) -> list[int]:
    """Token ids for ``text``, markers as single ids and the rest per byte."""
    ids: list[int] = []
    rest = text
    while rest:
        for marker, token_id in _MARKER_IDS.items():
            if rest.startswith(marker):
                ids.append(token_id)
                rest = rest[len(marker) :]
                break
        else:
            ids.append(ord(rest[0]))
            rest = rest[1:]
    return ids


def _detokenize(token_ids: Sequence[int]) -> str:
    """Applies the tokenizer's rule: every special id is dropped but the marker."""
    marker_id = _MARKER_IDS[TOOL_CALL_JSON_MARKER]
    pieces = []
    for token_id in token_ids:
        if token_id == marker_id:
            pieces.append(TOOL_CALL_JSON_MARKER)
        elif token_id not in _MARKER_IDS.values():
            pieces.append(chr(token_id))
    return "".join(pieces)


def _chunkings(text: str, seed: int, count: int = 40) -> list[list[str]]:
    """Char-by-char plus ``count`` seeded random splits of ``text``."""
    rng = random.Random(seed)
    splits = [list(text)]
    for _ in range(count):
        chunks: list[str] = []
        pos = 0
        while pos < len(text):
            step = rng.randint(1, 7)
            chunks.append(text[pos : pos + step])
            pos += step
        splits.append(chunks)
    return splits


def _token_chunkings(
    token_ids: Sequence[int], seed: int, count: int = 40
) -> list[list[list[int]]]:
    """One-token-at-a-time plus ``count`` seeded random splits of ``token_ids``."""
    rng = random.Random(seed)
    splits = [[[t] for t in token_ids]]
    for _ in range(count):
        chunks: list[list[int]] = []
        pos = 0
        while pos < len(token_ids):
            step = rng.randint(1, 7)
            chunks.append(list(token_ids[pos : pos + step]))
            pos += step
        splits.append(chunks)
    return splits


def _stream_turn(
    parser: InklingToolParser, chunks: Sequence[str]
) -> tuple[str, list[dict[str, str]]]:
    """Streams one turn, asserting no marker text reaches ``delta.content``.

    The check is per delta rather than on the assembled content, since a leaked
    marker can be followed by a delta that parses correctly.
    """
    content: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        for delta in parser.parse_delta(chunk) or []:
            if delta.content is not None:
                assert "<|" not in delta.content, (
                    f"marker text leaked as content: {delta.content!r}"
                )
                content.append(delta.content)
                continue
            entry = calls.setdefault(
                delta.index, {"id": "", "name": "", "arguments": ""}
            )
            if delta.id is not None:
                entry["id"] = delta.id
            if delta.name is not None:
                entry["name"] = delta.name
            if delta.arguments is not None:
                entry["arguments"] += delta.arguments
    return "".join(content), [calls[i] for i in sorted(calls)]


def _assert_turn(
    parser: InklingToolParser, response: str, chunks: Sequence[str]
) -> None:
    """Asserts one streamed turn reproduces ``parse_complete(response)``."""
    expected = InklingToolParser().parse_complete(response)
    content, streamed = _stream_turn(parser, chunks)

    assert content == (expected.content or "")
    assert [c["name"] for c in streamed] == [
        tc.name for tc in expected.tool_calls
    ]
    assert [json.loads(c["arguments"]) for c in streamed] == [
        json.loads(tc.arguments) for tc in expected.tool_calls
    ]


def test_streaming_turn_initial_call_across_two_turns() -> None:
    """One parser, two turns, each opening straight into a tool call.

    A turn-initial call is the shape a prompt ending at a bare
    ``<|message_model|>`` produces, and reusing the instance across turns is
    what would expose a stale buffer, as leaked marker text.
    """
    turns = [
        (_wire("get_weather", '{"city":"SF"}'), ("get_weather",)),
        (_wire("add", '{"a":2,"b":3}'), ("add",)),
    ]

    parser = InklingToolParser()
    for index, (response, declared) in enumerate(turns):
        for chunks in _chunkings(response, seed=index):
            parser.reset()
            parser.set_streaming_tool_schemas(_schemas(*declared))
            _assert_turn(parser, response, chunks)


def test_streaming_marker_at_buffer_position_zero() -> None:
    """A turn whose first byte is the marker: the name comes from the payload.

    Thinking that runs straight into a tool call swallows the bare name.
    """
    response = f'{TOOL_CALL_JSON_MARKER}{{"name":"get_weather","args":{{"city":"SF"}}}}'

    for chunks in _chunkings(response, seed=7):
        parser = InklingToolParser()
        parser.set_streaming_tool_schemas(_schemas("get_weather"))
        content, streamed = _stream_turn(parser, chunks)

        assert content == ""
        assert [c["name"] for c in streamed] == ["get_weather"]
        assert json.loads(streamed[0]["arguments"]) == {"city": "SF"}


def test_streaming_thinking_then_turn_initial_call() -> None:
    """The joint path: reasoning parser, tokenizer stripping, then parse_delta.

    A thinking message split mid-token still routes to reasoning, not content.
    """
    turn = (
        f"{THINKING}weighing the options{END_MESSAGE}"
        f"{MESSAGE_MODEL}get_weather{TOOL_CALL_JSON_MARKER}"
        '{"name":"get_weather","args":{"city":"SF"}}'
    )
    token_ids = _encode(turn)

    for token_chunks in _token_chunkings(token_ids, seed=11):
        reasoning = InklingReasoningParser(
            thinking_start_token_id=_MARKER_IDS[THINKING],
            end_message_token_id=_MARKER_IDS[END_MESSAGE],
            tool_call_start_token_id=_MARKER_IDS[TOOL_CALL_JSON_MARKER],
        )
        parser = InklingToolParser()
        parser.set_streaming_tool_schemas(_schemas("get_weather"))

        is_reasoning = False
        text_chunks: list[str] = []
        for token_chunk in token_chunks:
            parsed = reasoning.stream(
                token_chunk, is_currently_reasoning=is_reasoning
            )
            is_reasoning = parsed.is_still_reasoning
            text_chunks.append(
                _detokenize(parsed.span.extract_content(token_chunk))
            )

        content, streamed = _stream_turn(parser, text_chunks)
        assert content == ""
        assert [c["name"] for c in streamed] == ["get_weather"]
        assert json.loads(streamed[0]["arguments"]) == {"city": "SF"}


# ---------------------------------------------------------------------------
# reasoning handoff
# ---------------------------------------------------------------------------


def test_reasoning_parser_hands_over_only_the_tool_call() -> None:
    """A thinking block before a call must not reach this parser.

    Runs the real reasoning parser over the token sequence, then applies the
    tokenizer's rule that every special id is dropped except the tool marker,
    which is the exact text the serving path feeds to this parser. Ids are the
    published Inkling values.
    """
    thinking, end_message, message_model, tool_json = (
        200008,
        200010,
        200001,
        200049,
    )
    body = '{"name":"get_weather","args":{"city":"SF"}}'
    text = {
        tool_json: TOOL_CALL_JSON_MARKER,
        1: "weighing the options",
        2: "get_weather",
        3: body,
    }
    tokens = [thinking, 1, end_message, message_model, 2, tool_json, 3]

    span = (
        InklingReasoningParser(
            thinking_start_token_id=thinking,
            end_message_token_id=end_message,
            tool_call_start_token_id=tool_json,
        )
        .stream(tokens, is_currently_reasoning=False)
        .span
    )

    handed_over = "".join(
        text[token] for token in span.extract_content(tokens) if token in text
    )
    assert handed_over == _wire("get_weather", '{"city":"SF"}')

    parsed = InklingToolParser().parse_complete(handed_over)
    assert parsed.content is None
    assert [(c.name, json.loads(c.arguments)) for c in parsed.tool_calls] == [
        ("get_weather", {"city": "SF"})
    ]


# ---------------------------------------------------------------------------
# constrained decoding
# ---------------------------------------------------------------------------


def test_generate_tool_call_grammar_requires_xgrammar_backend() -> None:
    with pytest.raises(InputError, match="xgrammar"):
        InklingToolParser.generate_tool_call_grammar(
            tools=[tool("get_weather", UNIT_ENUM)], backend="llguidance"
        )


def test_auto_grammar_is_loose_until_the_marker_then_tight() -> None:
    """Enforcement arms on the marker: everything ahead of it is free."""
    grammar = Grammar([tool("get_weather", UNIT_ENUM)])
    matcher = grammar.matcher()
    assert len(grammar.allowed(matcher)) > 200

    assert matcher.accept_token(grammar.ids[TOOL_CALL_JSON_MARKER])
    assert grammar.allowed(matcher) == {ord("{")}


def test_auto_grammar_rejects_an_out_of_enum_argument() -> None:
    grammar = Grammar([tool("get_weather", UNIT_ENUM)])
    matcher = grammar.matcher()
    opened = (
        f"get_weather{TOOL_CALL_JSON_MARKER}"
        '{"name":"get_weather","args":{"unit":"'
    )
    assert grammar.drive(matcher, opened)
    assert grammar.allowed(matcher) == {ord("c"), ord("f")}
    assert not matcher.accept_token(ord("s"))


def test_auto_grammar_frees_the_region_after_a_call() -> None:
    """Enforcement never disarms, so the free region must pass what follows.

    Rejecting the separator or the terminator would drop enforcement for the
    rest of the request.
    """
    grammar = Grammar([tool("get_weather", UNIT_ENUM)])
    matcher = grammar.matcher()
    assert grammar.drive(matcher, call("get_weather", '{"unit":"c"}'))

    allowed = grammar.allowed(matcher)
    assert grammar.ids[MESSAGE_MODEL] in allowed
    assert grammar.ids[STOP] in allowed
    assert grammar.drive(matcher, MESSAGE_MODEL)
    assert grammar.drive(matcher, call("get_weather", '{"unit":"f"}'))


def _preamble(*messages: str) -> str:
    """The turn's leading messages, each closed and separated as Inkling does.

    The first ``<|message_model|>`` is already in the prompt, so the first
    message opens with its content marker alone.
    """
    return "".join(
        f"{marker}thinking out loud{END_MESSAGE}{MESSAGE_MODEL}"
        for marker in messages
    )


def test_required_grammar_admits_a_bounded_canonical_preamble() -> None:
    """Inkling's canonical turn is thinking, then text, then the call.

    Enforcement runs from token 0 under ``required`` with no way to suspend it,
    so a grammar admitting no preamble would forbid thinking outright.
    """
    grammar = Grammar([tool("get_weather", UNIT_ENUM)], "required")

    allowed = grammar.allowed(grammar.matcher())
    assert grammar.ids[THINKING] in allowed
    assert grammar.ids[TEXT] in allowed

    for prefix in [
        "",
        _preamble(THINKING),
        _preamble(TEXT),
        _preamble(THINKING, TEXT),
    ]:
        matcher = grammar.matcher()
        assert grammar.drive(
            matcher, prefix + call("get_weather", '{"unit":"c"}')
        ), f"rejected preamble {prefix!r}"


def test_required_grammar_bounds_the_preamble_at_two_messages() -> None:
    """The cap is what keeps a preamble from burning the whole token budget."""
    grammar = Grammar([tool("get_weather", UNIT_ENUM)], "required")
    matcher = grammar.matcher()
    assert grammar.drive(matcher, _preamble(THINKING, TEXT))

    allowed = grammar.allowed(matcher)
    assert grammar.ids[THINKING] not in allowed
    assert grammar.ids[TEXT] not in allowed
    assert grammar.ids[STOP] not in allowed


def test_required_grammar_preamble_body_cannot_reach_the_call() -> None:
    """A message body is byte-level, so no marker is emittable inside it."""
    grammar = Grammar([tool("get_weather", UNIT_ENUM)], "required")
    matcher = grammar.matcher()
    assert grammar.drive(matcher, THINKING + "weighing the options")

    assert grammar.allowed(matcher) & set(grammar.ids.values()) == {
        grammar.ids[END_MESSAGE]
    }


def test_required_grammar_forbids_ending_the_turn_after_a_preamble() -> None:
    """Talking is admitted; talking *instead* of calling is not."""
    grammar = Grammar([tool("get_weather", UNIT_ENUM)], "required")
    matcher = grammar.matcher()
    assert grammar.drive(matcher, _preamble(THINKING))

    assert grammar.ids[STOP] not in grammar.allowed(matcher)


def test_args_accept_the_sorted_key_order_the_template_emits() -> None:
    """The template renders args with ``sort_keys=true``, so keys arrive sorted.

    Sorting the schema must not degrade into ``any_order``, which only counts
    keys: a required property still cannot be dropped.
    """
    schema = obj(
        {"location": {"type": "string"}, "days": {"type": "integer"}},
        ["location", "days"],
    )
    grammar = Grammar([tool("forecast", schema)])
    assert grammar.drive(
        grammar.matcher(), call("forecast", '{"days":1,"location":"P"}')
    )

    truncated = grammar.matcher()
    assert grammar.drive(truncated, opened("forecast") + '{"days":1')
    assert not truncated.accept_token(ord("}"))


def test_nested_object_properties_are_sorted_too() -> None:
    schema = obj(
        {
            "where": obj(
                {"zip": {"type": "string"}, "city": {"type": "string"}},
                ["zip", "city"],
            )
        },
        ["where"],
    )
    grammar = Grammar([tool("forecast", schema)])
    matcher = grammar.matcher()
    assert grammar.drive(
        matcher, call("forecast", '{"where":{"city":"P","zip":"1"}}')
    )


def test_required_grammar_forbids_ending_the_turn_without_a_call() -> None:
    grammar = Grammar([tool("get_weather", UNIT_ENUM)], "required")
    assert grammar.ids[STOP] not in grammar.allowed(grammar.matcher())


# --- argument schema enforcement ---


@pytest.mark.parametrize(
    ("schema", "conforming", "violating"),
    [
        pytest.param(
            obj({"n": {"type": "integer"}}, ["n"]),
            '{"n":3}',
            '{"n":3.',
            id="integer-rejects-fractional",
        ),
        pytest.param(
            obj({"s": {"type": "string"}}, ["s"]),
            '{"s":"x"}',
            '{"s":4',
            id="string-rejects-number",
        ),
        pytest.param(
            obj({"a": {"type": "string"}, "b": {"type": "string"}}, ["a", "b"]),
            '{"a":"x","b":"y"}',
            '{"a":"x"}',
            id="required-property-cannot-be-omitted",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {"u": {"$ref": "#/$defs/U"}},
                "required": ["u"],
                "$defs": {"U": {"type": "string", "enum": ["c", "f"]}},
            },
            '{"u":"c"}',
            '{"u":"s',
            id="ref-defs-enum",
        ),
        pytest.param(
            obj(
                {"v": {"anyOf": [{"type": "integer"}, {"enum": ["auto"]}]}},
                ["v"],
            ),
            '{"v":"auto"}',
            '{"v":"other',
            id="anyof",
        ),
        pytest.param(
            obj(
                {"xs": {"type": "array", "items": {"type": "integer"}}}, ["xs"]
            ),
            '{"xs":[1,2]}',
            '{"xs":["',
            id="array-items",
        ),
        pytest.param(
            obj({"s": {"type": "string", "maxLength": 2}}, ["s"]),
            '{"s":"ab"}',
            '{"s":"abc',
            id="maxlength",
        ),
        pytest.param(
            obj({"s": {"type": "string", "pattern": "^a+$"}}, ["s"]),
            '{"s":"aa"}',
            '{"s":"b',
            id="pattern",
        ),
    ],
)
def test_args_are_enforced_against_the_declared_schema(
    schema: dict[str, Any], conforming: str, violating: str
) -> None:
    """Conforming args are accepted; violating args are rejected mid-stream."""
    grammar = Grammar([tool("f", schema)])

    accepting = grammar.matcher()
    assert grammar.drive(accepting, call("f", conforming))

    rejecting = grammar.matcher()
    assert not grammar.drive(rejecting, opened("f") + violating)


def test_unsupported_schema_keyword_stays_unenforced() -> None:
    """A keyword the converter cannot express must not fail the request.

    Rejecting it would lose the whole call rather than that one constraint.
    """
    schema = obj({"n": {"type": "integer", "multipleOf": 2}}, ["n"])
    grammar = Grammar([tool("f", schema)])
    matcher = grammar.matcher()
    # multipleOf goes unchecked, but the declared type still holds.
    assert grammar.drive(matcher, call("f", '{"n":3}'))


def test_schema_without_additional_properties_admits_extra_keys() -> None:
    """``additionalProperties`` defaults to true, so extras must be allowed."""
    grammar = Grammar([tool("f", obj({"a": {"type": "string"}}, ["a"]))])
    matcher = grammar.matcher()
    assert grammar.drive(matcher, call("f", '{"a":"x","extra":1}'))


def test_freeform_object_admits_more_than_one_key() -> None:
    """An object schema with no declared properties takes any number of pairs.

    Allowing only one would fold the rest into the first value rather than
    reject, handing the caller plausible-looking wrong arguments.
    """
    schema = obj({"headers": {"type": "object"}}, ["headers"])
    grammar = Grammar([tool("f", schema)])
    matcher = grammar.matcher()
    assert grammar.drive(
        matcher, call("f", '{"headers":{"Accept":"x","Content-Type":"y"}}')
    )


# --- envelope enforcement ---


def test_payload_name_must_be_a_declared_tool() -> None:
    grammar = Grammar([tool("get_weather", UNIT_ENUM)])
    matcher = grammar.matcher()
    assert grammar.drive(
        matcher, f'get_weather{TOOL_CALL_JSON_MARKER}{{"name":"'
    )
    assert not grammar.drive(matcher, "wrong")


def test_payload_name_selects_the_matching_schema() -> None:
    """Two tools share the marker, so the name is what picks the arg schema."""
    tools = [
        tool("a", obj({"x": {"type": "integer"}}, ["x"])),
        tool("b", obj({"y": {"type": "string"}}, ["y"])),
    ]
    grammar = Grammar(tools)

    matcher = grammar.matcher()
    assert grammar.drive(matcher, call("a", '{"x":1}'))

    crossed = grammar.matcher()
    assert not grammar.drive(crossed, opened("a") + '{"y"')


def test_payload_tolerates_but_bounds_whitespace() -> None:
    """Whitespace is permitted and capped rather than forbidden."""
    grammar = Grammar([tool("get_weather", UNIT_ENUM)])
    matcher = grammar.matcher()
    assert grammar.drive(matcher, opened("get_weather") + '{"unit":')
    assert matcher.accept_token(ord(" "))
    # Bounded: a second consecutive space is not.
    assert not matcher.accept_token(ord(" "))


def test_parameterless_tool_accepts_an_empty_object() -> None:
    """A tool without ``parameters`` maps to the unconstrained bool schema.

    Its arguments only have to be well-formed JSON, so a scalar root is not
    blocked either.
    """
    grammar = Grammar([{"type": "function", "function": {"name": "ping"}}])
    matcher = grammar.matcher()
    assert grammar.drive(matcher, call("ping", "{}"))


def test_open_string_value_cannot_swallow_a_marker() -> None:
    """Markers stay masked out of string content, so a call cannot run away."""
    grammar = Grammar([tool("note", obj({"s": {"type": "string"}}, ["s"]))])
    matcher = grammar.matcher()
    assert grammar.drive(matcher, opened("note") + '{"s":"')

    allowed = grammar.allowed(matcher)
    assert grammar.ids[END_MESSAGE] not in allowed
    assert grammar.ids[TOOL_CALL_JSON_MARKER] not in allowed


def test_closed_payload_admits_only_the_message_terminator() -> None:
    grammar = Grammar([tool("get_weather", UNIT_ENUM)])
    matcher = grammar.matcher()
    assert grammar.drive(matcher, opened("get_weather") + '{"unit":"c"}}')
    assert grammar.allowed(matcher) == {grammar.ids[END_MESSAGE]}


# --- required / forced composition ---


def test_required_grammar_pins_the_bare_name_to_a_declared_tool() -> None:
    """Under ``required`` the bare name ahead of the marker is constrained."""
    grammar = Grammar([tool("get_weather", UNIT_ENUM)], "required")
    matcher = grammar.matcher()
    assert grammar.drive(matcher, "get_weath")
    assert not matcher.accept_token(ord("z"))


def test_required_grammar_needs_the_separator_between_calls() -> None:
    grammar = Grammar([tool("get_weather", UNIT_ENUM)], "required")
    matcher = grammar.matcher()
    assert grammar.drive(matcher, call("get_weather", '{"unit":"c"}'))

    allowed = grammar.allowed(matcher)
    assert grammar.ids[MESSAGE_MODEL] in allowed
    assert grammar.ids[TOOL_CALL_JSON_MARKER] not in allowed
    assert grammar.drive(
        matcher, MESSAGE_MODEL + call("get_weather", '{"unit":"f"}')
    )


def test_forced_grammar_admits_exactly_one_call_to_the_named_tool() -> None:
    tools = [
        tool("a", obj({"x": {"type": "integer"}}, ["x"])),
        tool("b", obj({"y": {"type": "string"}}, ["y"])),
    ]
    choice = {"type": "function", "function": {"name": "a"}}
    grammar = Grammar(tools, choice)

    matcher = grammar.matcher()
    assert grammar.drive(matcher, call("a", '{"x":1}'))
    assert grammar.ids[MESSAGE_MODEL] not in grammar.allowed(matcher)

    other = grammar.matcher()
    assert not grammar.drive(other, "b")


def test_forced_grammar_admits_the_preamble_before_its_one_call() -> None:
    tools = [
        tool("a", obj({"x": {"type": "integer"}}, ["x"])),
        tool("b", obj({"y": {"type": "string"}}, ["y"])),
    ]
    grammar = Grammar(tools, {"type": "function", "function": {"name": "a"}})

    matcher = grammar.matcher()
    assert grammar.drive(
        matcher, _preamble(THINKING, TEXT) + call("a", '{"x":1}')
    )
    # Still exactly one call: no separator is offered afterwards.
    assert grammar.ids[MESSAGE_MODEL] not in grammar.allowed(matcher)

    other = grammar.matcher()
    assert grammar.drive(other, _preamble(THINKING))
    assert not grammar.drive(other, "b")


# --- response_format alternation ---

_ANSWER_SCHEMA = obj({"answer": {"type": "string"}}, ["answer"])
_ANSWER_JSON = '{"answer":"42"}'


def _json_grammar() -> Grammar:
    """Tools plus a ``response_format`` schema, combined into one grammar."""
    return Grammar(
        [tool("get_weather", UNIT_ENUM)],
        response_format_schema=_ANSWER_SCHEMA,
    )


def test_response_format_json_must_be_framed_as_a_text_message() -> None:
    """Bare JSON is not a turn Inkling's parsers can read, so it is not offered."""
    grammar = _json_grammar()
    assert ord("{") not in grammar.allowed(grammar.matcher())

    matcher = grammar.matcher()
    assert grammar.drive(matcher, TEXT + _ANSWER_JSON + END_MESSAGE)
    assert grammar.ids[STOP] in grammar.allowed(matcher)


def test_response_format_json_may_follow_a_thinking_message() -> None:
    grammar = _json_grammar()
    matcher = grammar.matcher()
    assert grammar.drive(
        matcher, _preamble(THINKING) + TEXT + _ANSWER_JSON + END_MESSAGE
    )
    assert grammar.ids[STOP] in grammar.allowed(matcher)


def test_response_format_json_is_enforced_against_the_schema() -> None:
    """A text message only ends the turn when its body conforms.

    Any byte run is a legal free-text preamble, so the schema decides whether
    the turn may terminate there rather than what the body may contain.
    """
    grammar = _json_grammar()

    conforming = grammar.matcher()
    assert grammar.drive(conforming, TEXT + _ANSWER_JSON + END_MESSAGE)
    assert grammar.ids[STOP] in grammar.allowed(conforming)

    # Right shape, wrong type for ``answer``.
    violating = grammar.matcher()
    assert grammar.drive(violating, TEXT + '{"answer":4}' + END_MESSAGE)
    assert grammar.ids[STOP] not in grammar.allowed(violating)


def test_response_format_leaves_the_tool_branch_alive_after_free_text() -> None:
    """A text message that is not the JSON answer is still a legal preamble.

    Only the tool-call continuation survives it, so the turn cannot end there.
    """
    grammar = _json_grammar()
    matcher = grammar.matcher()
    assert grammar.drive(matcher, TEXT + "just talking" + END_MESSAGE)

    allowed = grammar.allowed(matcher)
    assert grammar.ids[MESSAGE_MODEL] in allowed
    assert grammar.ids[STOP] not in allowed
    assert grammar.drive(
        matcher, MESSAGE_MODEL + call("get_weather", '{"unit":"c"}')
    )
