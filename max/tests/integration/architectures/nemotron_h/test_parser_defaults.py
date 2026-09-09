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
"""Regression tests for Nemotron-H reasoning/tool parser defaults.

Nemotron-3-Nano emits reasoning with an *implicit* open (the chat template
pre-fills ``<think>\\n`` in the generation prompt) and an *explicit*
``</think>`` close. Served without a reasoning parser, the chain-of-thought
and the raw ``</think>`` delimiter leak verbatim into OpenAI
``message.content`` and ``reasoning_content`` stays null. The architecture
must therefore default a ``<think>``-style reasoning parser (and the
matching ``<tool_call>`` tool parser) so any serve of this architecture
splits reasoning without a CLI flag.

The reasoning-split test mirrors the pinned fuzz repro
(``case_think_artifact_leak.sh``): a temperature-0 completion shaped
``<CoT tokens> </think> <content tokens>`` must route the CoT to the
reasoning span and keep ``</think>`` out of the content span.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import Mock

import numpy as np
from max.pipelines.architectures.nemotron_h.arch import nemotron_h_arch
from max.pipelines.lib import reasoning, tool_parsing

# Special-token ids from the Nemotron-3-Nano tokenizer.
_THINK_START = 12
_THINK_END = 13
_TOOL_CALL_START = 14

_SPECIAL_TOKENS = {
    "<think>": _THINK_START,
    "</think>": _THINK_END,
    "<tool_call>": _TOOL_CALL_START,
}


def _mock_nemotron_tokenizer() -> Mock:
    """Minimal tokenizer surface for ``ReasoningParser.from_tokenizer``.

    ``convert_token_to_id`` encodes the marker string and requires a
    single-token result, so map the three Nemotron special tokens to their
    real vocab ids and everything else to a multi-token sequence.
    """

    async def mock_encode(
        token: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        token_id = _SPECIAL_TOKENS.get(token)
        if token_id is None:
            return np.array([0, 0])
        return np.array([token_id])

    mock = Mock()
    mock.encode = mock_encode
    return mock


def _build_reasoning_parser() -> Any:
    assert nemotron_h_arch.reasoning_parser is not None, (
        "nemotron_h must default a reasoning parser: without one, CoT and a "
        "raw </think> leak into OpenAI message.content"
    )
    return asyncio.run(
        reasoning.create(
            nemotron_h_arch.reasoning_parser, _mock_nemotron_tokenizer()
        )
    )


def test_arch_defaults_registered_reasoning_parser() -> None:
    """The default reasoning parser name must resolve in the registry."""
    name = nemotron_h_arch.reasoning_parser
    assert name is not None
    assert reasoning.get_parser_cls(name) is not None, (
        f"reasoning parser {name!r} is not registered; importing the "
        "nemotron_h architecture must register it"
    )


def test_reasoning_split_mirrors_think_artifact_repro() -> None:
    """Implicit-open CoT + explicit ``</think>`` splits away from content."""
    parser = _build_reasoning_parser()
    parser.reset()

    cot = [100, 101, 102]
    content = [200, 201]
    delta = parser.stream([*cot, _THINK_END, *content])

    tokens = [*cot, _THINK_END, *content]
    assert delta.span.extract_reasoning(tokens) == cot
    # The content region must not contain the </think> delimiter.
    assert delta.span.extract_content(tokens) == content
    assert not delta.is_still_reasoning


def test_will_reason_after_multi_turn_backfilled_prompt() -> None:
    """Prior-turn ``<think></think>`` backfill must not disable reasoning.

    The Nemotron chat template inserts ``<think></think>`` into earlier
    assistant turns and ends the generation prompt with a fresh
    ``<think>``, so only the *last* delimiter reflects the current state.
    """
    parser = _build_reasoning_parser()
    prompt = [5, _THINK_START, _THINK_END, 6, 7, _THINK_START]
    assert parser.will_reason_after_prompt(prompt)


def test_arch_defaults_registered_tool_parser() -> None:
    """The default tool parser name must resolve in the registry."""
    name = nemotron_h_arch.tool_parser
    assert isinstance(name, str), (
        "nemotron_h must default a tool parser: without one, "
        "tool_choice='required' emits zero tool_calls"
    )
    assert tool_parsing.get_parser_cls(name) is not None, (
        f"tool parser {name!r} is not registered; importing the "
        "nemotron_h architecture must register it"
    )


def test_tool_parser_handles_nemotron_template_format() -> None:
    """The default tool parser parses the chat template's rendered format."""
    assert isinstance(nemotron_h_arch.tool_parser, str)
    parser = tool_parsing.create(nemotron_h_arch.tool_parser)
    # Exactly what the Nemotron-3-Nano chat template renders for an
    # assistant tool call.
    rendered = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\n"
        "Paris\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )
    parsed = parser.parse_complete(rendered)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {"city": "Paris"}
