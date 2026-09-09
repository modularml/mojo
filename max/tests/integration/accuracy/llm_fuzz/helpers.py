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
"""Shared test utilities for llm-fuzz scenarios."""

from __future__ import annotations

import json
from typing import Any


def parse_json(body: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a JSON response body. Returns (data, None) on success or (None, error_str) on failure."""
    try:
        return json.loads(body), None
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return None, str(e)


def collect_stream(stream: Any) -> dict[str, Any]:
    """Aggregate an OpenAI SDK streaming response into a single result dict.

    Works with the openai Python SDK's stream objects. Tracks content, reasoning,
    tool calls (with correct index handling), finish_reason, usage, and protocol
    compliance metadata.
    """
    content = ""
    reasoning = ""
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    chunk_count = 0
    first_tc_chunks: dict[int, object] = {}
    all_chunks: list[Any] = []
    usage = None

    for chunk in stream:
        chunk_count += 1
        all_chunks.append(chunk)
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        if delta.content:
            content += delta.content
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning += delta.reasoning_content
        elif hasattr(delta, "reasoning") and delta.reasoning:
            reasoning += delta.reasoning
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.id or "",
                        "name": "",
                        "arguments": "",
                    }
                    first_tc_chunks[idx] = tc
                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        if hasattr(choice, "usage") and choice.usage:
            usage = choice.usage

    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": list(tool_calls.values()),
        "finish_reason": finish_reason,
        "chunk_count": chunk_count,
        "first_tc_chunks": first_tc_chunks,
        "all_chunks": all_chunks,
        "usage": usage,
    }


def make_tool(
    name: str,
    params_schema: dict[str, Any] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Create an OpenAI-format tool definition."""
    if params_schema is None:
        params_schema = {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "additionalProperties": False,
        }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Tool: {name}",
            "parameters": params_schema,
        },
    }


def budget_exhausted(resp: Any) -> bool:
    """Check if a non-streaming response hit the max_tokens limit."""
    return resp.choices[0].finish_reason == "length"


def stream_budget_exhausted(result: dict[str, Any]) -> bool:
    """Check if a collected stream result hit the max_tokens limit."""
    return result.get("finish_reason") == "length"


def validate_json_args(
    tool_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any] | None], list[str]]:
    """Parse tool call arguments as JSON, returning (parsed_list, errors).

    Each entry in parsed_list is the parsed dict (or None on failure).
    errors is a list of human-readable error strings for any parse failures.
    """
    parsed = []
    errors = []
    for i, tc in enumerate(tool_calls):
        args_str = tc.get("arguments", "")
        try:
            parsed.append(json.loads(args_str))
        except (json.JSONDecodeError, TypeError) as e:
            parsed.append(None)
            errors.append(f"tool_call[{i}] ({tc.get('name', '?')}): {e}")
    return parsed, errors


def check_streaming_protocol(first_tc_chunks: dict[int, Any]) -> list[str]:
    """Check first streaming tool call chunk for protocol compliance.

    Returns a list of violation descriptions (empty = all good).
    Checks: id present, type="function", name present.
    """
    violations = []
    for idx, tc in first_tc_chunks.items():
        if not tc.id:
            violations.append(f"tool_call[{idx}]: missing id in first chunk")
        if hasattr(tc, "type") and tc.type != "function":
            violations.append(
                f"tool_call[{idx}]: type={tc.type!r}, expected 'function'"
            )
        if tc.function and not tc.function.name:
            violations.append(f"tool_call[{idx}]: missing name in first chunk")
    return violations


def check_no_forbidden_tokens(text: str, tokens: list[str]) -> list[str]:
    """Check text for forbidden token substrings. Returns list of found tokens."""
    return [t for t in tokens if t in text]


# Structural / reasoning control tokens across tool-calling models. The serving
# layer decodes with ``skip_special_tokens=True``, so none of these should ever
# appear as literal text in ``message.content`` / ``message.reasoning``; one that
# does signals a parser/matcher desync or a dropped reasoning span (a
# regression). When adding a tool-calling architecture, add its markers
# here (sourced from its ``tool_parser.py`` / ``reasoning.py``).
STRUCTURAL_LEAK_MARKERS: tuple[str, ...] = (
    # Kimi K2.5 — architectures/kimik2_5/{tool_parser,reasoning}.py
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
    "<|im_end|>",
    # Gemma 4 — architectures/gemma4/{tokenizer,reasoning}.py (SpecialToken)
    "<|tool_call>",
    "<tool_call|>",
    "<|tool>",
    "<tool|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    "<turn|>",
    "<|channel>",
    "<channel|>",
    # MiniMax M2 — architectures/minimax_m2/tool_parser.py
    "<minimax:tool_call>",
    "</minimax:tool_call>",
    # GLM-4.5+ (GLM-5.1 / GLM-5.2) — architectures/glm5_1/tool_parser.py
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    # Shared reasoning delimiters (Kimi K2.5, MiniMax M2).
    "<think>",
    "</think>",
)
