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
"""Scenario: agentic correctness for any tool-calling model.

Regression tests motivated by bugs found across multiple models
(Kimi K2.5, etc.) when used in IDE-style agent loops
(OpenCode). Applies to any model with tool-calling support:

- Think-token / tool-marker leaks in streamed content (reasoning and
  tool-parser protocol tokens must not reach the client verbatim).
- Tool call arguments must be valid JSON even when the model emits
  its own intermediate format (per-arg, XML, etc.) — the parser must
  normalize before streaming.
- Tool names with different case (e.g. model emits "Glob", schema
  has "glob") must be normalized to the canonical name, not dropped
  as hallucinations.
- Inter-token spacing must be preserved when tools are present
  (detokenizer / strip_tool_call_markup must not eat leading
  whitespace from streaming deltas).
- End-to-end multi-turn agentic flow: model calls tool, result is
  returned, model produces a coherent followup without leaks.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from helpers import check_no_forbidden_tokens, make_tool, validate_json_args

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig
    from validator_client import ValidatorClient

_SEARCH_TOOL = make_tool(
    "WebSearch",
    {
        "type": "object",
        "properties": {
            "search_term": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["search_term"],
        "additionalProperties": False,
    },
    description="Search the web for a term",
)

_BASH = make_tool(
    "bash",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    description="Run a bash command",
)

_READ = make_tool(
    "read",
    {
        "type": "object",
        "properties": {"filePath": {"type": "string"}},
        "required": ["filePath"],
        "additionalProperties": False,
    },
    description="Read a file",
)

_WRITE = make_tool(
    "write",
    {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["filePath", "content"],
        "additionalProperties": False,
    },
    description="Write a file",
)

_GLOB = make_tool(
    "glob",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    description="Find files matching a glob pattern",
)

_GREP = make_tool(
    "grep",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    description="Search files for a pattern",
)

_CODER_TOOLS = [_BASH, _READ, _WRITE, _GLOB, _GREP]
_CODER_TOOL_NAMES = {t["function"]["name"] for t in _CODER_TOOLS}

# Tool-parser protocol tokens that must never leak into streamed content,
# across the common tool formats (Kimi/DeepSeek/GLM). The fullwidth-bar
# codepoints (U+FF5C) in the DeepSeek-V3 markers are the actual on-the-
# wire bytes the model emits — replacing them with ASCII `|` would turn
# the leak detector into "no `|` anywhere", false-positiving on regular
# text. The RUF001 warnings on those lines are suppressed.
_LEAK_MARKERS = [
    "<think>",
    "</think>",
    # Kimi K2 style (ASCII)
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
    # DeepSeek-V3 style (fullwidth — intentional, see comment above)
    "<｜tool▁calls▁begin｜>",  # noqa: RUF001
    "<｜tool▁calls▁end｜>",  # noqa: RUF001
    "<｜tool▁call▁begin｜>",  # noqa: RUF001
    "<｜tool▁call▁end｜>",  # noqa: RUF001
    "<｜tool▁sep｜>",  # noqa: RUF001
    # GLM style
    "<tool_call>",
    "</tool_call>",
    # Inkling style
    "<|content_invoke_tool_json|>",
]

# Pre-compiled to avoid re-compiling in the hot path of
# _has_garbled_spacing. Ignore code blocks/URLs where no-space
# punctuation is legitimate.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_URL_RE = re.compile(r"https?://\S+")
_GARBLED_SPACING_RE = re.compile(r"[a-z][,.][A-Z][a-z]")


def _has_garbled_spacing(text: str) -> bool:
    if not text:
        return False
    stripped = _CODE_FENCE_RE.sub("", text)
    stripped = _INLINE_CODE_RE.sub("", stripped)
    stripped = _URL_RE.sub("", stripped)
    return bool(_GARBLED_SPACING_RE.search(stripped))


@register_scenario
class AgenticCorrectness(BaseScenario):
    """End-to-end agentic correctness for tool-calling models."""

    name = "agentic_correctness"
    description = (
        "Agentic flow correctness: marker leaks, tool-arg JSON validity, "
        "case-insensitive tool names, spacing preservation, multi-turn loop"
    )
    tags = ["validation", "tools", "agentic", "streaming"]
    requires_validator = True
    scenario_type = "validation"

    def _verdict(
        self,
        test_name: str,
        ok: bool,
        detail_fail: str,
        *,
        body: str = "",
    ) -> list[ScenarioResult]:
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS if ok else Verdict.FAIL,
                detail="ok" if ok else detail_fail,
                response_body=body[:300] if body else "",
            )
        ]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        v = config.validator
        if not v:
            results.append(
                self.make_result(
                    self.name,
                    "setup",
                    Verdict.ERROR,
                    detail="No validator client available",
                )
            )
            return results

        test_fns = [
            self._test_tool_args_valid_json_streaming,
            self._test_simple_json_args_streaming,
            self._test_no_marker_leak_streaming_no_tools,
            self._test_no_marker_leak_streaming_with_tools,
            self._test_case_insensitive_tool_name,
            self._test_spacing_preserved_tool_choice_none,
            self._test_spacing_preserved_tool_choice_auto,
            self._test_multi_turn_agentic_loop,
        ]
        loop = asyncio.get_running_loop()
        outcomes = await asyncio.gather(
            *(loop.run_in_executor(None, fn, v) for fn in test_fns),
            return_exceptions=True,
        )
        for fn, outcome in zip(test_fns, outcomes, strict=False):
            if isinstance(outcome, BaseException):
                results.append(
                    self.make_result(
                        self.name,
                        fn.__name__.removeprefix("_test_"),
                        Verdict.ERROR,
                        error=str(outcome),
                    )
                )
            else:
                results.extend(outcome)
        return results

    def _test_tool_args_valid_json_streaming(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Streaming tool call arguments must be parseable JSON.  Regression
        test for per-arg / XML / non-JSON intermediate formats that some
        models emit natively."""
        test_name = "tool_args_valid_json_streaming"
        agg = v.tc_chat_stream(
            messages=[
                {
                    "role": "user",
                    "content": "Search the web for python asyncio tutorial",
                }
            ],
            tools=[_SEARCH_TOOL],
            tool_choice="required",
            temperature=0,
            max_tokens=200,
        )
        if not agg["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="model did not emit a required tool call",
                )
            ]
        _, errs = validate_json_args(agg["tool_calls"])
        raw = agg["tool_calls"][0]["arguments"][:120]
        return self._verdict(
            test_name,
            not errs,
            f"invalid JSON: {errs}; raw={raw!r}",
        )

    def _test_simple_json_args_streaming(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        test_name = "simple_json_args_streaming"
        agg = v.tc_chat_stream(
            messages=[
                {"role": "user", "content": "what is the weather in tokyo"}
            ],
            tools=[
                make_tool(
                    "get_weather",
                    {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                )
            ],
            tool_choice="required",
            temperature=0,
            max_tokens=100,
        )
        if not agg["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="model did not emit a required tool call",
                )
            ]
        _, errs = validate_json_args(agg["tool_calls"])
        return self._verdict(test_name, not errs, f"invalid JSON: {errs}")

    def _test_no_marker_leak_streaming_no_tools(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        agg = v.chat_stream(
            messages=[{"role": "user", "content": "hi how are you?"}],
            temperature=0,
            max_tokens=100,
        )
        leaked = check_no_forbidden_tokens(agg["content"], _LEAK_MARKERS)
        return self._verdict(
            "no_marker_leak_streaming_no_tools",
            not leaked,
            f"leaked {leaked}",
            body=agg["content"],
        )

    def _test_no_marker_leak_streaming_with_tools(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Tools provided + tool_choice=none: the model must answer
        directly, and streamed content must be clean of protocol markers."""
        agg = v.tc_chat_stream(
            messages=[{"role": "user", "content": "hi how are you?"}],
            tools=_CODER_TOOLS,
            tool_choice="none",
            temperature=0,
            max_tokens=120,
        )
        leaked = check_no_forbidden_tokens(agg["content"], _LEAK_MARKERS)
        return self._verdict(
            "no_marker_leak_streaming_with_tools",
            not leaked,
            f"leaked {leaked}",
            body=agg["content"],
        )

    def _test_case_insensitive_tool_name(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Tool names differing only in case (Glob vs glob) must be
        normalized to the schema's canonical name, not dropped."""
        test_name = "case_insensitive_tool_name"
        agg = v.tc_chat_stream(
            messages=[
                {
                    "role": "user",
                    "content": "Call the Glob tool to find all Python files in /tmp. "
                    "Use the tool rather than answering directly.",
                }
            ],
            tools=_CODER_TOOLS,
            tool_choice="required",
            temperature=0,
            max_tokens=200,
        )
        if not agg["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="model did not emit a required tool call",
                )
            ]
        bad = [
            tc["name"]
            for tc in agg["tool_calls"]
            if tc["name"] not in _CODER_TOOL_NAMES
        ]
        if bad:
            return self._verdict(
                test_name,
                False,
                f"non-canonical tool name(s): {bad} "
                f"(valid: {sorted(_CODER_TOOL_NAMES)})",
            )
        saw_glob = any(tc["name"] == "glob" for tc in agg["tool_calls"])
        return self._verdict(
            test_name,
            saw_glob,
            "expected a canonical 'glob' tool call, got "
            f"{[tc['name'] for tc in agg['tool_calls']]}",
        )

    def _test_spacing_preserved_tool_choice_none(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """tool_choice=none + tools must not drop inter-token spaces
        through strip_tool_call_markup."""
        agg = v.tc_chat_stream(
            messages=[
                {
                    "role": "user",
                    "content": "Explain recursion in one sentence.",
                }
            ],
            tools=_CODER_TOOLS,
            tool_choice="none",
            temperature=0,
            max_tokens=150,
        )
        garbled = _has_garbled_spacing(agg["content"])
        return self._verdict(
            "spacing_preserved_tool_choice_none",
            not garbled,
            f"garbled spacing in: {agg['content'][:200]!r}",
            body=agg["content"],
        )

    def _test_spacing_preserved_tool_choice_auto(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        agg = v.tc_chat_stream(
            messages=[
                {
                    "role": "user",
                    "content": "Explain what a hash map is in 2-3 sentences.",
                }
            ],
            tools=_CODER_TOOLS,
            temperature=0,
            max_tokens=200,
        )
        garbled = _has_garbled_spacing(agg["content"])
        return self._verdict(
            "spacing_preserved_tool_choice_auto",
            not garbled,
            f"garbled spacing in: {agg['content'][:200]!r}",
            body=agg["content"],
        )

    def _test_multi_turn_agentic_loop(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """End-to-end mini agent: model calls tool, we return result,
        model produces a coherent followup with no leaks."""
        test_name = "multi_turn_agentic_loop"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a coding assistant. Use tools when the user asks.",
            },
            {"role": "user", "content": "Use bash to run: echo hello"},
        ]
        agg1 = v.tc_chat_stream(
            messages=messages,
            tools=_CODER_TOOLS,
            tool_choice="required",
            temperature=0,
            max_tokens=300,
        )
        if not agg1["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="model did not emit a required tool call on turn 1",
                )
            ]
        _, errs1 = validate_json_args(agg1["tool_calls"])
        if errs1:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"turn 1 invalid JSON args: {errs1}; "
                    f"raw={agg1['tool_calls'][0]['arguments'][:120]!r}",
                )
            ]

        tc = agg1["tool_calls"][0]
        tc_id = tc["id"]
        if not tc_id:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="streaming tool_call missing id — breaks real "
                    "agent loops that must reference tool_call_id",
                )
            ]
        messages.append(
            {
                "role": "assistant",
                "content": agg1["content"] or "",
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": "hello",
                "tool_call_id": tc_id,
            }
        )

        agg2 = v.tc_chat_stream(
            messages=messages,
            tools=_CODER_TOOLS,
            temperature=0,
            max_tokens=200,
        )
        leaked = check_no_forbidden_tokens(agg2["content"], _LEAK_MARKERS)
        if leaked:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"turn 2 leaked markers: {leaked}",
                    response_body=agg2["content"][:300],
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"2-turn loop ok; final content len={len(agg2['content'])}",
            )
        ]
