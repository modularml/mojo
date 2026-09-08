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
"""Tests for the structured-output grammar backends."""

import json
import logging
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines.lib.pipeline_variants import structured_output_backend
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN,
    _compiled_shape,
    _log_if_slow,
)
from max.pipelines.lib.pipeline_variants.utils import StructuredOutputHelper
from max.pipelines.modeling.types import PipelineTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

_N_VOCAB = 256

# An ordinary printable byte ("~") the runtime additionally stops on. The
# declared EOS (byte 0) is deliberately distinct: the runtime EOS set layers
# extra terminators (chat turn-end tokens and the like) on top of the
# declared EOS, which is exactly the split the grammar backend's stop set
# has to respect. Unless the backend registers the extra terminator as a
# stop token, the grammar admits it as ordinary string content.
_EXTRA_TERMINATOR_ID = 126


class _FakeTikTokenTokenizer:
    """Prod-shaped TikToken delegate over a byte-level vocab."""

    eos_token_id: int = 0
    bos_token_id: int | None = None
    all_special_ids: list[int] = []

    def __init__(self) -> None:
        self.byte_decoder = {chr(b): b for b in range(256)}

    def __len__(self) -> int:
        return _N_VOCAB

    def get_vocab(self) -> dict[str, int]:
        return {chr(i): i for i in range(256)}

    def convert_ids_to_tokens(self, idx: int) -> str:
        return chr(idx)

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        return [ord(c) for c in text]


def _fast_tokenizer_delegate() -> PreTrainedTokenizerFast:
    """``PreTrainedTokenizerFast`` delegate over the same byte-level vocab."""
    vocab = {chr(i): i for i in range(256)}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token=chr(1))),
        eos_token=chr(0),
        unk_token=chr(1),
    )


def _allowed_tokens(backend: Any, matcher: Any) -> np.ndarray:
    """Bool ``[vocab]`` mask of the tokens ``matcher`` currently allows."""
    packed = backend.allocate_token_bitmask(1, _N_VOCAB)
    backend.fill_next_token_bitmask(matcher, packed, 0)
    masks = np.int32(1) << np.arange(32, dtype=np.int32)
    bits = (packed[..., np.newaxis] & masks) != 0
    return bits.reshape(*packed.shape[:-1], -1)[0, :_N_VOCAB]


@pytest.mark.parametrize(
    "delegate_factory",
    [_FakeTikTokenTokenizer, _fast_tokenizer_delegate],
    ids=["tiktoken", "hf_fast"],
)
def test_xgrammar_stop_tokens_cover_runtime_eos_set(
    delegate_factory: Callable[[], Any],
) -> None:
    """The grammar's stop set must match the runtime EOS set, not just EOS."""
    delegate = delegate_factory()
    runtime_eos = {delegate.eos_token_id, _EXTRA_TERMINATOR_ID}
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = runtime_eos

    helper = StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name="xgrammar",
    )
    assert helper.backend is not None

    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(json.dumps(schema))
    )

    # Stop inside the string value: ordinary content bytes are allowed, but
    # no terminator may be sampled, or the runtime would end the request
    # mid-structure (silent truncation).
    for char in '{"a":"x':
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert not matcher.is_accepting()
    mid = _allowed_tokens(helper.backend, matcher)
    assert mid[ord("y")]
    assert not mid[sorted(runtime_eos)].any(), (
        "a terminator is sampleable mid-structure — the runtime would stop "
        "generation inside the constrained response and truncate it"
    )

    for char in '"}':
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert matcher.is_accepting()

    done = set(np.flatnonzero(_allowed_tokens(helper.backend, matcher)))
    assert done == runtime_eos, (
        "a completed grammar must permit exactly the runtime EOS set: a "
        "missing terminator forces an unnatural declared-EOS ending, an "
        "extra token would leak unconstrained output"
    )


def test_fill_slot_after_stop_token_accepted_forces_eos_without_crashing() -> (
    None
):
    """Once the matcher actually consumes its stop token (not merely reaching
    the "ready to stop" accepting state exercised above), xgrammar's native
    fill raises a fatal C++ check rather than returning a mask. This drives a
    real matcher through that exact transition and confirms
    ``fill_next_token_bitmask`` still forces the same stop-token-only mask
    instead of crashing -- built by hand from ``stop_token_ids`` rather than
    delegated to xgrammar's own (crashing) fill.
    """
    delegate = _FakeTikTokenTokenizer()
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = {delegate.eos_token_id}

    helper = StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name="xgrammar",
    )
    assert helper.backend is not None

    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(json.dumps(schema))
    )
    for char in "{}":
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert matcher.is_accepting()
    assert not matcher.is_stopped()

    assert matcher.try_consume_tokens([delegate.eos_token_id]) == 1
    assert matcher.is_stopped()
    assert matcher.is_accepting()

    allowed = set(np.flatnonzero(_allowed_tokens(helper.backend, matcher)))
    assert allowed == {delegate.eos_token_id}, (
        "a stopped-and-accepted xgrammar matcher must force exactly its "
        f"stop token set, not {allowed} -- xgrammar's own fill crashes here, "
        "so the mask is built by hand from stop_token_ids"
    )


def test_llguidance_completed_matcher_still_gets_eos_mask() -> None:
    """llguidance's ``is_stopped()`` becomes true the moment the grammar is
    satisfied -- before the stop token is even consumed -- so a completed
    grammar is llguidance's normal, frequent end-of-request state, not a
    rare corner case. It computes a real EOS-only mask here without
    crashing, matching xgrammar's hand-built one above.
    """
    delegate = _FakeTikTokenTokenizer()
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = {delegate.eos_token_id}

    helper = StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name="llguidance",
    )
    assert helper.backend is not None

    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(json.dumps(schema))
    )
    for char in "{}":
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert matcher.is_accepting()
    assert matcher.is_stopped()

    allowed = set(np.flatnonzero(_allowed_tokens(helper.backend, matcher)))
    assert allowed == {delegate.eos_token_id}, (
        f"a completed llguidance matcher must force exactly its stop token "
        f"set, not {allowed}"
    )


# Minimal schema for the whitespace-mode tests: one required string property.
_WS_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
)


def _make_helper(backend_name: str, **kwargs: Any) -> StructuredOutputHelper:
    # The TikToken-shaped fake exercises both backends: llguidance cannot
    # infer a decoder from the WordLevel HF fake, but both backends accept
    # the byte-level adapter path.
    delegate = _FakeTikTokenTokenizer()
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = {delegate.eos_token_id}
    return StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name=backend_name,
        **kwargs,
    )


@pytest.mark.parametrize("backend_name", ["xgrammar", "llguidance"])
def test_any_whitespace_grammar_admits_whitespace(backend_name: str) -> None:
    """``any_whitespace=True`` compiles a grammar that accepts whitespaceful JSON."""
    helper = _make_helper(backend_name, any_whitespace=True)
    assert helper.backend is not None
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(_WS_SCHEMA)
    )
    payload = '{ "a": "x" }'
    tokens = [ord(c) for c in payload]
    assert matcher.try_consume_tokens(tokens) == len(tokens), (
        f"[{backend_name}] whitespace-tolerant grammar rejected {payload!r}"
    )
    assert matcher.is_accepting()


def test_any_whitespace_grammar_bounds_whitespace_runs() -> None:
    """The whitespace-tolerant xgrammar grammar caps each whitespace run.

    Guards the runaway-generation vector that motivated the compact default:
    a model looping on whitespace must be forced to converge instead of
    emitting whitespace forever (GLM 5.2 produced exactly that runaway
    inside tool calls). xgrammar-only: llguidance has no whitespace-run cap.
    """
    backend_name = "xgrammar"
    helper = _make_helper(backend_name, any_whitespace=True)
    assert helper.backend is not None
    grammar = helper.backend.compile_json_schema(_WS_SCHEMA)

    max_run = " " * STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN
    within = helper.backend.create_matcher(grammar)
    payload = "{" + max_run + '"a":"x"}'
    tokens = [ord(c) for c in payload]
    assert within.try_consume_tokens(tokens) == len(tokens), (
        f"[{backend_name}] bounded grammar rejected a "
        f"{STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN}-char whitespace run"
    )
    assert within.is_accepting()

    beyond = helper.backend.create_matcher(grammar)
    payload = "{" + max_run + ' "a":"x"}'
    consumed = beyond.try_consume_tokens([ord(c) for c in payload])
    assert consumed == 1 + STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN, (
        f"[{backend_name}] grammar consumed {consumed} tokens of "
        f"{payload!r}; expected rejection at whitespace char "
        f"{STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN + 1}"
    )


@pytest.mark.parametrize("backend_name", ["xgrammar", "llguidance"])
def test_default_grammar_stays_compact(backend_name: str) -> None:
    """The default (``any_whitespace`` unset) keeps the compact-JSON grammar.

    Guards the Gemma-4 runaway mitigation (0c57a6bd331): flipping the global
    default is a product decision, so an unset knob must reproduce today's
    whitespace-free grammar exactly.
    """
    helper = _make_helper(backend_name)
    assert helper.backend is not None
    grammar = helper.backend.compile_json_schema(_WS_SCHEMA)

    compact = helper.backend.create_matcher(grammar)
    tokens = [ord(c) for c in '{"a":"x"}']
    assert compact.try_consume_tokens(tokens) == len(tokens)
    assert compact.is_accepting()

    spaced = helper.backend.create_matcher(grammar)
    payload = '{"a": "x"}'
    consumed = spaced.try_consume_tokens([ord(c) for c in payload])
    assert consumed == payload.index(" "), (
        f"[{backend_name}] compact grammar consumed {consumed} tokens of "
        f"{payload!r}; expected rejection at the whitespace"
    )


class _SlowBackend:
    """Minimal stand-in carrying the ``name`` the decorator reads."""

    name = "fake"

    @_log_if_slow
    def compile_json_schema(self, schema: str) -> str:
        return schema


def test_slow_grammar_compile_log_carries_structured_fields(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The duration must be a queryable attribute, not just message text."""
    monkeypatch.setattr(
        structured_output_backend, "_GRAMMAR_COMPILE_LOG_MS", -1.0
    )
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _SlowBackend().compile_json_schema("{}")

    record = next(
        r for r in caplog.records if r.msg.startswith("grammar %s took")
    )
    fields = vars(record)
    assert fields["event"] == "grammar_compile_slow"
    assert fields["grammar_compile_method"] == "compile_json_schema"
    assert fields["grammar_backend"] == "fake"
    assert fields["grammar_compile_time_ms"] > 0.0
    # ``name`` stays the logger name; the backend goes in its own key.
    assert record.name == "max.pipelines"


def test_fast_grammar_compile_logs_nothing(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        structured_output_backend, "_GRAMMAR_COMPILE_LOG_MS", 1e9
    )
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _SlowBackend().compile_json_schema("{}")
    assert not [r for r in caplog.records if r.msg.startswith("grammar %s")]


class _ShapeBackend:
    """Stand-in whose compile entry point takes any of the wire forms."""

    name = "fake"

    @_log_if_slow
    def create_matcher(self, grammar: Any) -> Any:
        return grammar


def _slow_log(
    arg: Any, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Force the slow path for ``arg`` and return the emitted record's fields."""
    monkeypatch.setattr(
        structured_output_backend, "_GRAMMAR_COMPILE_LOG_MS", -1.0
    )
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _ShapeBackend().create_matcher(arg)
    return vars(
        next(r for r in caplog.records if r.msg.startswith("grammar %s took"))
    )


def test_slow_compile_log_carries_schema_shape(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth and node count are attached to the slow-compile record."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "object", "properties": {"b": {"type": "string"}}}
        },
    }
    fields = _slow_log(json.dumps(schema), caplog, monkeypatch)
    # root -> a -> b
    assert fields["grammar_schema_depth"] == 3
    assert fields["grammar_schema_nodes"] == 3


def test_slow_compile_log_omits_shape_for_non_schema_arguments(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiled handle has no shape; the duration is still logged."""
    fields = _slow_log(object(), caplog, monkeypatch)
    assert fields["grammar_compile_time_ms"] > 0.0
    assert "grammar_schema_depth" not in fields
    assert "grammar_schema_nodes" not in fields


def _tool(name: str) -> dict[str, Any]:
    """An OpenAI tool whose arguments schema is 5 subschemas, 4 deep."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
                "required": ["query"],
            },
        },
    }


def test_compiled_shape_aggregates_schemas_inside_a_tool_grammar() -> None:
    """A tool grammar's shape is its embedded schemas, not its envelope.

    Depth is the deepest of them; node count is their sum.
    """
    one_tool = _compiled_shape(
        structured_output_backend.build_xgrammar_tool_grammar(
            "kimi", [_tool("a")], "auto"
        )
    )
    assert one_tool is not None
    depth, nodes = one_tool
    assert depth == 4, "root -> filters -> tags -> items"
    assert nodes == 5

    five_tools = _compiled_shape(
        structured_output_backend.build_xgrammar_tool_grammar(
            "kimi", [_tool(f"t{i}") for i in range(5)], "auto"
        )
    )
    assert five_tools is not None
    assert five_tools == (depth, nodes * 5), (
        "depth is the deepest tool's, node count the sum across tools"
    )


def test_slow_compile_log_carries_tool_grammar_shape(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The logged fields carry a tool grammar's aggregated shape."""
    grammar = structured_output_backend.build_xgrammar_tool_grammar(
        "kimi", [_tool(f"t{i}") for i in range(5)], "auto"
    )
    fields = _slow_log(grammar, caplog, monkeypatch)
    assert fields["grammar_schema_depth"] == 4
    assert fields["grammar_schema_nodes"] == 25, (
        "expected 5 tool schemas of 5 subschemas each"
    )


def test_slow_compile_log_carries_shape_for_a_keyword_call(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The payload is found whether it arrives positionally or by keyword."""
    monkeypatch.setattr(
        structured_output_backend, "_GRAMMAR_COMPILE_LOG_MS", -1.0
    )
    schema = _tool("a")["function"]["parameters"]
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _ShapeBackend().create_matcher(grammar=json.dumps(schema))
    fields = vars(
        next(r for r in caplog.records if r.msg.startswith("grammar %s took"))
    )
    assert fields["grammar_schema_depth"] == 4
    assert fields["grammar_schema_nodes"] == 5


def test_compiled_shape_reads_a_bare_schema_directly() -> None:
    """The response_format path hands over a schema, not a tag."""
    schema = _tool("a")["function"]["parameters"]
    assert _compiled_shape(json.dumps(schema)) == (4, 5)


def test_compiled_shape_ignores_a_lark_grammar() -> None:
    """A Lark grammar is not JSON, so no shape is reported."""
    assert _compiled_shape("start: /[a-z]+/\n") is None


def test_slow_compile_log_survives_malformed_json(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unparseable input omits the shape fields rather than raising."""
    fields = _slow_log("{not json", caplog, monkeypatch)
    assert fields["grammar_compile_time_ms"] > 0.0
    assert "grammar_schema_depth" not in fields
