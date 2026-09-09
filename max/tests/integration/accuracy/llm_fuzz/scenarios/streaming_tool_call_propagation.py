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
"""
Scenario: streaming — tool-call propagation regressions

Regression coverage for the streaming chat completion path on
reasoning-capable architectures (Gemma 4, Kimi K2.5, etc.).  Each
sub-test asserts a property that previously regressed in production:

  1. ``stream_does_not_500_on_active_no_text``
     Reasoning parsers consume every token in a chunk as a structural
     delimiter, leaving the chunk ACTIVE with no decoded content.  The
     old chat-stream code forced ``get_finish_reason_from_status(...,
     allow_none=False)`` on every empty chunk and raised on ACTIVE.

  2. ``stream_emits_tool_call_deltas``
     The streaming detokenizer used to call
     ``DecodeStream(skip_special_tokens=True)``, stripping wrappers
     like ``<|tool_call>``/``<tool_call|>`` before the parser saw the
     chunk.  Verify ``tool_calls`` deltas are emitted (not raw wrapper
     text leaking via ``content``).

  3. ``stream_no_raw_structural_token_leak``
     If wrappers were stripped *and* the parser couldn't match, the
     literal arg body (``call:NAME{...}``) leaked into ``content``.  No
     content delta should contain raw structural markers.

  4. ``stream_reasoning_not_in_content``
     Models can emit ``<|channel>thought\\n...<channel|>`` mid-stream
     even when not pre-seeded into reasoning.  With the always-run-
     parser fix, those tokens route to ``reasoning_content``; content
     deltas must not contain the stray ``thought\\n`` prefix.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name",
                },
            },
            "required": ["city"],
        },
    },
}

# Structural marker fragments that should never appear verbatim in a
# content delta after the streaming-tool-call fixes. Drawn from the
# Gemma 4, Kimi K2.5 and Inkling vocabularies.
_STRUCTURAL_LEAKS: tuple[str, ...] = (
    "<|content_invoke_tool_json|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|channel>",
    "<channel|>",
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    '<|"|>',
)


def _tool_streaming_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Paris? Use the tool.",
            },
        ],
        "tools": [_TOOL],
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def _iter_deltas(
    chunks: list[str],
) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Yield decoded ``delta`` dicts from an SSE chunk list."""
    for raw in chunks:
        if not raw or raw == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        yield choices[0].get("delta") or {}, choices[0].get("finish_reason")


@register_scenario
class StreamingToolCallPropagation(BaseScenario):
    name = "streaming_tool_call_propagation"
    description = (
        "Streaming tool-call regressions: ACTIVE-no-text 500s, missing "
        "tool_calls deltas, raw structural-token leaks, and mid-stream "
        "reasoning sections leaking as content."
    )
    tags = ["streaming", "tools", "reasoning", "regression"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        payload = _tool_streaming_payload(model)
        resp = await client.post_streaming(payload, read_timeout=120.0)

        # ----- 1. No 500 on ACTIVE-but-no-text intermediate chunks ----------
        if resp.status >= 500:
            results.append(
                self.make_result(
                    self.name,
                    "stream_does_not_500_on_active_no_text",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=(
                        f"streaming returned HTTP {resp.status}: "
                        f"{(resp.body or '')[:300]!r}"
                    ),
                )
            )
            # Without a successful stream, downstream checks cannot run.
            return results

        results.append(
            self.make_result(
                self.name,
                "stream_does_not_500_on_active_no_text",
                Verdict.PASS,
                status_code=resp.status,
                detail="stream completed without 5xx",
            )
        )

        chunks = resp.chunks or []
        deltas = list(_iter_deltas(chunks))

        # ----- 2. tool_calls deltas were emitted at least once --------------
        has_tool_call_delta = any(
            delta.get("tool_calls") for delta, _ in deltas
        )
        results.append(
            self.make_result(
                self.name,
                "stream_emits_tool_call_deltas",
                Verdict.PASS if has_tool_call_delta else Verdict.FAIL,
                detail=(
                    "tool_calls delta present in stream"
                    if has_tool_call_delta
                    else "no tool_calls delta emitted — wrappers likely "
                    "stripped before parser"
                ),
            )
        )

        # ----- 3. Content deltas don't carry raw structural markers ---------
        leaked_marker: str | None = None
        leaked_snippet: str = ""
        for delta, _ in deltas:
            content = delta.get("content")
            if not isinstance(content, str) or not content:
                continue
            for marker in _STRUCTURAL_LEAKS:
                if marker in content:
                    leaked_marker = marker
                    idx = content.find(marker)
                    leaked_snippet = content[
                        max(0, idx - 20) : idx + len(marker) + 40
                    ]
                    break
            if leaked_marker is not None:
                break

        results.append(
            self.make_result(
                self.name,
                "stream_no_raw_structural_token_leak",
                Verdict.PASS if leaked_marker is None else Verdict.FAIL,
                detail=(
                    "no structural-token leak in content"
                    if leaked_marker is None
                    else (
                        f"content leaked {leaked_marker!r} "
                        f"(snippet={leaked_snippet!r})"
                    )
                ),
            )
        )

        # ----- 4. Stray "thought\n" prefix not in content -------------------
        # Models can emit <|channel>thought\n...<channel|> mid-stream;
        # the always-run-parser fix routes those tokens to reasoning. If
        # any content delta begins with "thought\n", reasoning bled into
        # content.
        thought_leak: str = ""
        for delta, _ in deltas:
            content = delta.get("content")
            if not isinstance(content, str):
                continue
            stripped = content.lstrip()
            if stripped.startswith(("thought\n", "thought ")):
                thought_leak = content[:120]
                break

        results.append(
            self.make_result(
                self.name,
                "stream_reasoning_not_in_content",
                Verdict.PASS if not thought_leak else Verdict.FAIL,
                detail=(
                    "no 'thought\\n' prefix leaked into content"
                    if not thought_leak
                    else (
                        "content delta starts with reasoning prefix "
                        f"(snippet={thought_leak!r})"
                    )
                ),
            )
        )

        # ----- 5. Server still healthy after the run ------------------------
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_stream_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail=(
                    "server still healthy"
                    if health.status == 200
                    else (health.error or f"HTTP {health.status}")
                ),
            )
        )

        return results
