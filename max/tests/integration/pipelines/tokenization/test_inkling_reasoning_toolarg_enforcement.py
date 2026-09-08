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
"""Offline CPU test that Inkling tool-argument enforcement survives reasoning.

Inkling never arms a thinking region -- the generation prompt ends at a bare
``<|message_model|>``, so the model rather than the template picks the first
content marker -- and the usual "suspend the bitmask while thinking" path never
engages. A thinking-then-tool-call turn is safe only because enforcement stays
off until the tool marker under ``auto``, and the grammar itself admits the
canonical preamble under ``required``.

Both matter because the runtime fails open: a rejected committed token disables
enforcement for the rest of the request, silently. These tests drive the real
compiled grammar and enforcement state machine over a synthetic byte + marker
RAW vocab, with no server or GPU.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import transformers
from _grammar_harness import (
    END_MESSAGE,
    MARKERS,
    MESSAGE_MODEL,
    STOP,
    TEXT,
    THINKING,
    TOOL_CALL_JSON_MARKER,
    UNIT_ENUM,
    Grammar,
    tool,
)
from max.pipelines.architectures.inkling.reasoning import (
    InklingReasoningParser,
)
from max.pipelines.architectures.inkling.tool_parser import InklingToolParser
from max.pipelines.context import (
    GrammarEnforcementState,
    StructuredOutputRegionDelimiters,
)
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    XgrammarBackend,
    special_token_ids_for,
)
from max.pipelines.lib.pipeline_variants.utils import StructuredOutputHelper

_TOOLS = [tool("get_weather", UNIT_ENUM)]


def _auto_state(harness: Grammar) -> GrammarEnforcementState:
    """The state the runtime builds for ``tool_choice=auto`` on a flat parser.

    The region is derived rather than hard-coded, so these tests follow the
    decode path.
    """
    start, end = StructuredOutputHelper._get_tool_region_tags("inkling")
    assert start is not None
    return GrammarEnforcementState(
        grammar_enforced=False,
        tools_forced=False,
        tool_region=StructuredOutputRegionDelimiters(
            start_token_ids=[harness.ids[start]],
            end_token_ids=[harness.ids[end]] if end else None,
        ),
    )


def test_inkling_derives_a_start_tag_and_no_end_tag() -> None:
    """The tool marker arms enforcement and nothing disarms it.

    An end tag would drop enforcement between consecutive calls, letting the
    model emit the next one unconstrained.
    """
    assert StructuredOutputHelper._get_tool_region_tags("inkling") == (
        TOOL_CALL_JSON_MARKER,
        None,
    )


def test_inkling_never_arms_a_thinking_region() -> None:
    """The premise both grammar shapes rest on.

    Were this True the bitmask would instead be suspended during reasoning.
    """
    parser = InklingReasoningParser(
        thinking_start_token_id=200008,
        end_message_token_id=200010,
        tool_call_start_token_id=200049,
    )
    assert parser.will_reason_after_prompt([200001]) is False


def test_auto_enforcement_starts_at_the_marker_and_never_disarms() -> None:
    """Reasoning cannot reach the matcher, so it cannot trip the fail-open."""
    harness = Grammar(_TOOLS, "auto")
    state = _auto_state(harness)
    marker = harness.ids[TOOL_CALL_JSON_MARKER]

    prelude = harness.encode(
        f"{THINKING}weighing the options{END_MESSAGE}{MESSAGE_MODEL}get_weather"
    )
    assert not any(state.update_enforcement_state(t) for t in prelude)
    assert state.grammar_enforced is False

    assert state.update_enforcement_state(marker) is True
    assert state.grammar_enforced is True

    payload = harness.encode('{"name":"get_weather","args":{"unit":"c"}}')
    assert all(state.update_enforcement_state(t) for t in payload)

    # The tool region has no end tag, so enforcement stays armed afterwards.
    assert state.update_enforcement_state(harness.ids[END_MESSAGE]) is True
    assert state.grammar_enforced is True


def test_auto_matcher_accepts_every_token_it_is_fed() -> None:
    """The other half of the fail-open proof: what reaches the matcher is legal.

    Only tokens ``update_enforcement_state`` approves are consumed, so a green
    run means enforcement survives the whole turn.
    """
    harness = Grammar(_TOOLS, "auto")
    state = _auto_state(harness)
    matcher = harness.matcher()

    turn = harness.encode(
        f"{THINKING}weighing the options{END_MESSAGE}"
        f"{MESSAGE_MODEL}get_weather{TOOL_CALL_JSON_MARKER}"
        '{"name":"get_weather","args":{"unit":"c"}}'
        f"{END_MESSAGE}"
    )
    consumed = 0
    for token in turn:
        if state.update_enforcement_state(token):
            assert matcher.accept_token(token), f"rejected token {token}"
            consumed += 1
    assert consumed > 0


def test_required_admits_a_full_thinking_turn_before_the_call() -> None:
    """The direct proof that a thinking model does not trip the fail-open.

    Every token of the canonical turn has to be accepted, and the arguments
    still have to conform.
    """
    harness = Grammar(_TOOLS, "required")
    matcher = harness.matcher()

    assert harness.ids[THINKING] in harness.allowed(matcher)

    turn = harness.encode(
        f"{THINKING}weighing the options{END_MESSAGE}{MESSAGE_MODEL}"
        f"{TEXT}checking the forecast{END_MESSAGE}{MESSAGE_MODEL}"
        f'get_weather{TOOL_CALL_JSON_MARKER}{{"name":"get_weather","args":{{"unit":"'
    )
    assert all(matcher.accept_token(t) for t in turn)

    assert harness.allowed(matcher) == {ord("c"), ord("f")}
    assert not matcher.accept_token(ord("s"))


def test_required_still_forbids_talking_instead_of_calling() -> None:
    """The preamble is admitted but capped, so the call stays mandatory."""
    harness = Grammar(_TOOLS, "required")
    matcher = harness.matcher()

    preamble = harness.encode(
        f"{THINKING}weighing the options{END_MESSAGE}{MESSAGE_MODEL}"
        f"{TEXT}checking the forecast{END_MESSAGE}{MESSAGE_MODEL}"
    )
    assert all(matcher.accept_token(t) for t in preamble)

    allowed = harness.allowed(matcher)
    assert harness.ids[STOP] not in allowed
    assert harness.ids[THINKING] not in allowed
    assert harness.ids[TEXT] not in allowed


@pytest.fixture(scope="module")
def inkling_delegate() -> Any:
    """The real Inkling tokenizer, from ``INKLING_TOKENIZER_DIR``.

    A local snapshot rather than the HF hub keeps the test off the network; it
    skips when the variable is unset.
    """
    snapshot = os.environ.get("INKLING_TOKENIZER_DIR")
    if not snapshot:
        pytest.skip("set INKLING_TOKENIZER_DIR to an Inkling snapshot to run")
    return transformers.AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True
    )


def test_real_tokenizer_masks_every_structural_marker(
    inkling_delegate: Any,
) -> None:
    """Every marker is masked without a per-model marker table entry."""
    masked = special_token_ids_for("inkling", inkling_delegate)
    for marker in MARKERS:
        token_id = inkling_delegate.convert_tokens_to_ids(marker)
        assert token_id in masked, f"{marker} is not masked"


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
    ids=["auto", "required", "forced"],
)
def test_real_tokenizer_compiles_the_grammar(
    inkling_delegate: Any, tool_choice: Any
) -> None:
    """The grammar names markers by token id, resolved against the real vocab.

    A marker that does not resolve is a hard error at compile time, so it would
    fail every tool request.
    """
    backend = XgrammarBackend.from_tokenizer_delegate(
        inkling_delegate,
        len(inkling_delegate),
        stop_token_ids=[inkling_delegate.convert_tokens_to_ids(STOP)],
        special_token_ids=special_token_ids_for("inkling", inkling_delegate),
    )
    grammar = InklingToolParser.generate_tool_call_grammar(
        tools=_TOOLS, tool_choice=tool_choice
    )
    backend.create_matcher(grammar)


def test_forked_matcher_leaves_the_argument_mask_tight() -> None:
    """Spec-decode forks the matcher; discarding a fork must not loosen it."""
    harness = Grammar(_TOOLS, "auto")
    matcher = harness.matcher()

    opened = [
        harness.ids[TOOL_CALL_JSON_MARKER],
        *harness.encode('{"name":"get_weather","args":{"unit":"'),
    ]
    for token in opened:
        fork = matcher.fork()
        fork.accept_token(ord("s"))
        del fork
        assert matcher.accept_token(token)

    assert harness.allowed(matcher) == {ord("c"), ord("f")}
