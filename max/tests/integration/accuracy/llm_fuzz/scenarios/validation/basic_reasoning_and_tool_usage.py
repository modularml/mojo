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
"""Validation: minimal viability of tool calling x reasoning x content.

Smoke-level checks that a model + tool-parser + reasoning-parser
configuration is minimally viable over the streaming chat-completion API.
Eight base scenarios, each crossed with thinking on/off (16 variants):

- tool-call: forced tool call (tool_choice=required), one tool
- tool-call-auto: auto tool choice; the model may answer or call
- response-format: response_format=json_schema constrained decoding
- tool-call-and-response-format: both tools and a JSON schema
- open-ended: plain chat completion
- content-parts: user content as an array of text parts
- tool-call-calculate-required: forced tool call on a greeting prompt
- special-token-content: control tokens embedded in the prompt as text

Each streamed response is checked against a structured ``Expectation``:
well-formed tool-call arguments that conform to the declared parameter
schema, response_format content that conforms to its JSON schema,
reasoning presence matching the requested thinking mode (with a non-zero
reasoning-token assertion where the prompt reliably thinks), allowed
finish reasons, and the absence of parser sentinel tokens in
user-visible text. Any failed check FAILs the variant.

Thinking is toggled with both ``chat_template_kwargs.thinking`` and
``chat_template_kwargs.enable_thinking`` (models disagree on the kwarg
name; the standard llm-fuzz idiom is to send both).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from helpers import STRUCTURAL_LEAK_MARKERS
from jsonschema import Draft7Validator

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RawResponse, RunConfig


WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
                },
            },
            "required": ["location"],
        },
    },
}


STOCK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


STOCK_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "symbol": {"type": "string"},
        "price": {"type": "number"},
        "analysis": {"type": "string"},
    },
    "required": ["reasoning", "symbol", "price", "analysis"],
    "additionalProperties": False,
}


STOCK_SUMMARY_SCHEMA_NO_REASONING: dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "price": {"type": "number"},
        "analysis": {"type": "string"},
    },
    "required": ["symbol", "price", "analysis"],
    "additionalProperties": False,
}


CALCULATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate, e.g. 2 + 2"
                    ),
                }
            },
            "additionalProperties": False,
            "required": ["expression"],
        },
    },
}


# A final user turn that embeds special / control tokens inline as plain text
# (two repetitions of lorem + a special-token block). Exercises the
# special-token-in-content path: the tokenizer must treat these as ordinary
# text, and the parsers must not echo them back into user-visible content.
_LOREM = (
    "lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim "
    "ad minim veniam, quis nostrud"
)
_SPECIAL_TOKENS = (
    "<|endofprompt|>\n<|endoftext|>\n<|fim_middle|>\n<|fim_prefix|>\n"
    "<|fim_suffix|>\n<|im_end|>\n<|im_sep|>\n<|im_start|>"
)
_SPECIAL_TOKEN_BLOB = (
    f"{_LOREM}\n{_SPECIAL_TOKENS}\n{_LOREM}\n{_SPECIAL_TOKENS}\n"
)


# Sampling params reused by scenarios that stream usage. They request
# streaming usage (so the reasoning-token check below has data) and pin
# sampling so the run is reproducible.
_USAGE_STREAM_OPTIONS: dict[str, Any] = {"include_usage": True}
_EXPLICIT_SAMPLING: dict[str, Any] = {
    "temperature": 1,
    "presence_penalty": 0,
    "repetition_penalty": 1,
    "frequency_penalty": 0,
    "top_p": 1,
}


# Base tests whose thinking-on variant must report non-zero reasoning tokens in
# ``usage.completion_tokens_details.reasoning_tokens``. Scoped to free-form
# prompts that reliably trigger a thinking block (not e.g. forced tool calls,
# which may short-circuit reasoning).
REQUIRES_REASONING_TOKENS: frozenset[str] = frozenset({"special-token-content"})


@dataclass(frozen=True)
class Expectation:
    """What the checks should require of a streamed response."""

    must_have_tool_calls: bool | None
    """``True`` requires at least one tool call. ``False`` requires zero.
    ``None`` (auto) means the model may choose; in that case the
    auto-choice check only requires the turn to be non-empty (a
    natural-language preamble alongside a tool call is correct).
    """

    tool_arg_schema: dict[str, Any] | None
    """JSON schema each tool call's ``arguments`` must conform to.
    ``None`` skips per-tool schema validation (but the JSON-parse check
    still runs).
    """

    must_have_content: bool | None
    """``True`` requires non-empty assistant content. ``False`` requires
    empty. ``None`` defers to the auto-choice check.
    """

    content_response_format_schema: dict[str, Any] | None
    """When the request set ``response_format=json_schema`` and content
    is non-empty, ``content`` must JSON-parse and conform to this schema.
    """

    must_have_reasoning: bool | None
    """Requirement on the presence of streamed reasoning content.

    - ``False`` → reasoning MUST be empty. Set for every thinking-off
      variant; a violation is a "stray thinking token" regression (the
      reasoning parser surfaced output it shouldn't have).
    - ``True`` → reasoning MUST be non-empty.
    - ``None`` → reasoning is optional. Used for thinking-on variants
      where whether the model emits a thinking block is prompt-dependent
      (e.g. forced tool calls on a trivial prompt), so requiring it would
      be flaky. Such variants instead rely on
      ``require_reasoning_tokens`` where a strong assertion is reliable.
    """

    allowed_finish_reasons: frozenset[str]

    require_reasoning_tokens: bool = False
    """When ``True``, ``usage.completion_tokens_details.reasoning_tokens``
    must be present and non-zero. A stronger, token-counted assertion than
    ``must_have_reasoning`` — set only for thinking-on variants of tests in
    ``REQUIRES_REASONING_TOKENS``. Requires the request to send
    ``stream_options.include_usage``.
    """

    check_sentinel_leak: bool = True
    """Whether to scan the response for forbidden sentinel tokens. Disabled
    for variants (e.g. ``special-token-content``) that deliberately feed
    control tokens to the model as plain-text input — there the model may
    legitimately quote them back, which is not a parser leak.
    """


@dataclass(frozen=True)
class TestCase:
    name: str
    """Variant name (e.g. ``tool-call-thinking-off``)."""

    payload: dict[str, Any]
    """Chat-completion request body (sans ``model``, which ``run()``
    injects from the run config).
    """

    expectation: Expectation


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _system(content: str) -> dict[str, str]:
    return {"role": "system", "content": content}


# Stable user prompts so server-side prefix caching can amortize across runs.
_WEATHER_USER = _user("What is the weather like in Coquitlam today?")
_STOCK_USER = _user("Find the stock price for AAPL and give me a summary.")
_MEANING_USER = _user("What is the meaning of life?")


def _base_tests() -> list[tuple[str, dict[str, Any], Expectation]]:
    """The base scenarios of the viability matrix.

    Each entry is ``(name, payload-without-model, expectation)``. The
    ``must_have_reasoning`` field here is a thinking-agnostic placeholder;
    the cross-product below stamps the real per-thinking-mode value.
    """
    weather_args_schema = WEATHER_TOOL["function"]["parameters"]
    stock_args_schema = STOCK_TOOL["function"]["parameters"]
    return [
        (
            "tool-call",
            {
                "messages": [_WEATHER_USER],
                "tools": [WEATHER_TOOL],
                "tool_choice": "required",
            },
            Expectation(
                must_have_tool_calls=True,
                tool_arg_schema=weather_args_schema,
                must_have_content=False,
                content_response_format_schema=None,
                must_have_reasoning=False,
                allowed_finish_reasons=frozenset({"tool_calls"}),
            ),
        ),
        (
            "tool-call-auto",
            {
                "messages": [_WEATHER_USER],
                "tools": [WEATHER_TOOL],
                "tool_choice": "auto",
            },
            Expectation(
                must_have_tool_calls=None,
                tool_arg_schema=weather_args_schema,
                must_have_content=None,
                content_response_format_schema=None,
                must_have_reasoning=False,
                allowed_finish_reasons=frozenset({"tool_calls", "stop"}),
            ),
        ),
        (
            "response-format",
            {
                "messages": [
                    _system(
                        "You are a research assistant. Fill in this"
                        " information to the best of your knowledge."
                    ),
                    _STOCK_USER,
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "stock_summary_format",
                        "strict": True,
                        "schema": STOCK_SUMMARY_SCHEMA,
                    },
                },
            },
            Expectation(
                must_have_tool_calls=False,
                tool_arg_schema=None,
                must_have_content=True,
                content_response_format_schema=STOCK_SUMMARY_SCHEMA,
                must_have_reasoning=False,
                allowed_finish_reasons=frozenset({"stop", "length"}),
            ),
        ),
        (
            "tool-call-and-response-format",
            {
                "messages": [
                    _system(
                        "You are a research assistant. Use the tool to"
                        " find data, then summarize it."
                    ),
                    _STOCK_USER,
                ],
                "tools": [STOCK_TOOL],
                "tool_choice": "auto",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "stock_summary_format",
                        "strict": True,
                        "schema": STOCK_SUMMARY_SCHEMA_NO_REASONING,
                    },
                },
            },
            Expectation(
                must_have_tool_calls=None,
                tool_arg_schema=stock_args_schema,
                must_have_content=None,
                content_response_format_schema=STOCK_SUMMARY_SCHEMA_NO_REASONING,
                must_have_reasoning=False,
                allowed_finish_reasons=frozenset({"tool_calls", "stop"}),
            ),
        ),
        (
            "open-ended",
            {
                "messages": [_MEANING_USER],
            },
            Expectation(
                must_have_tool_calls=False,
                tool_arg_schema=None,
                must_have_content=True,
                content_response_format_schema=None,
                must_have_reasoning=False,
                allowed_finish_reasons=frozenset({"stop", "length"}),
            ),
        ),
        # Content as an array of text parts — exercises the content-array
        # parse path. Streams usage + pins sampling.
        (
            "content-parts",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What number comes after"},
                            {"type": "text", "text": " 5?"},
                        ],
                    }
                ],
                **_EXPLICIT_SAMPLING,
            },
            Expectation(
                must_have_tool_calls=False,
                tool_arg_schema=None,
                must_have_content=True,
                content_response_format_schema=None,
                must_have_reasoning=False,
                allowed_finish_reasons=frozenset({"stop", "length"}),
            ),
        ),
        # Forced tool call (tool_choice=required) with the calculate tool on a
        # plain-greeting prompt: the model must emit a conforming tool call
        # even though the prompt doesn't ask for one.
        (
            "tool-call-calculate-required",
            {
                "messages": [_user("Hi, how are you?")],
                "tools": [CALCULATE_TOOL],
                "tool_choice": "required",
                **_EXPLICIT_SAMPLING,
            },
            Expectation(
                must_have_tool_calls=True,
                tool_arg_schema=CALCULATE_TOOL["function"]["parameters"],
                must_have_content=False,
                content_response_format_schema=None,
                must_have_reasoning=False,
                allowed_finish_reasons=frozenset({"tool_calls"}),
            ),
        ),
        # Multi-turn chat whose final user turn embeds special / control tokens
        # inline as plain text. With thinking on it must spend reasoning tokens
        # (see REQUIRES_REASONING_TOKENS); the sentinel-leak check also
        # ensures the model doesn't echo any control token into its content.
        (
            "special-token-content",
            {
                "messages": [
                    _user("I need help identifying where this came from."),
                    {
                        "role": "assistant",
                        "content": "Sure, please send me the full text.",
                    },
                    _user(_SPECIAL_TOKEN_BLOB),
                ],
                **_EXPLICIT_SAMPLING,
            },
            Expectation(
                must_have_tool_calls=False,
                tool_arg_schema=None,
                must_have_content=True,
                content_response_format_schema=None,
                must_have_reasoning=False,
                allowed_finish_reasons=frozenset({"stop", "length"}),
                # The prompt embeds control tokens as plain text; the model
                # may quote them back, so a sentinel match here is not a
                # parser leak.
                check_sentinel_leak=False,
            ),
        ),
    ]


def _build_tests() -> dict[str, TestCase]:
    out: dict[str, TestCase] = {}
    for base_name, base_payload, base_exp in _base_tests():
        for thinking in (False, True):
            suffix = "thinking-on" if thinking else "thinking-off"
            name = f"{base_name}-{suffix}"
            payload = {
                # Always request streaming usage so the reasoning-token
                # check has data.
                "stream_options": _USAGE_STREAM_OPTIONS,
                **base_payload,
                "stream": True,
                # Models disagree on the kwarg that toggles thinking
                # (e.g. Kimi uses ``thinking``, Gemma4 uses
                # ``enable_thinking``); send both, per llm-fuzz idiom.
                "chat_template_kwargs": {
                    "thinking": thinking,
                    "enable_thinking": thinking,
                },
                "max_tokens": 10000,
            }
            requires_tokens = (
                thinking and base_name in REQUIRES_REASONING_TOKENS
            )
            if not thinking:
                # Thinking off: reasoning MUST be empty (stray-token guard).
                must_have_reasoning: bool | None = False
            elif requires_tokens:
                # Thinking on + reliable prompt: require reasoning content
                # and a non-zero reasoning-token count.
                must_have_reasoning = True
            else:
                # Thinking on, prompt-dependent: reasoning is optional.
                must_have_reasoning = None
            expectation = Expectation(
                must_have_tool_calls=base_exp.must_have_tool_calls,
                tool_arg_schema=base_exp.tool_arg_schema,
                must_have_content=base_exp.must_have_content,
                content_response_format_schema=(
                    base_exp.content_response_format_schema
                ),
                must_have_reasoning=must_have_reasoning,
                allowed_finish_reasons=base_exp.allowed_finish_reasons,
                require_reasoning_tokens=requires_tokens,
                check_sentinel_leak=base_exp.check_sentinel_leak,
            )
            out[name] = TestCase(
                name=name, payload=payload, expectation=expectation
            )
    return out


TESTS: dict[str, TestCase] = _build_tests()


@dataclass
class _StreamedResponse:
    """The streamed chat-completion response, reassembled from SSE chunks."""

    content: str = ""
    reasoning: str = ""
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    @property
    def reasoning_tokens(self) -> int | None:
        details = (self.usage or {}).get("completion_tokens_details") or {}
        value = details.get("reasoning_tokens")
        return value if isinstance(value, int) else None


def _accumulate_stream(chunks: Iterable[str]) -> _StreamedResponse:
    """Reassemble raw SSE ``data:`` payload strings into one response."""
    response = _StreamedResponse()
    for raw in chunks:
        if raw == "[DONE]":
            break
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue
        # The final ``include_usage`` chunk carries usage with empty
        # choices, so read usage before bailing on the choices list.
        if isinstance(chunk.get("usage"), dict):
            response.usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str):
            response.reasoning += reasoning
        content = delta.get("content")
        if isinstance(content, str):
            response.content += content

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = response.tool_calls.setdefault(
                idx, {"id": "", "name": "", "arguments": ""}
            )
            if isinstance(tc.get("id"), str):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if isinstance(fn.get("name"), str) and fn["name"]:
                slot["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["arguments"] += fn["arguments"]

        if choice.get("finish_reason"):
            response.finish_reason = choice["finish_reason"]
    return response


def _check_sentinel_leak(
    response: _StreamedResponse, expected: Expectation
) -> list[str]:
    """Fail if a structural parser marker leaked into user-visible text.

    Skipped when the variant deliberately feeds control tokens as
    plain-text input (``expected.check_sentinel_leak is False``).
    """
    if not expected.check_sentinel_leak:
        return []
    failures: list[str] = []
    scopes: list[tuple[str, str]] = [
        ("content", response.content),
        ("reasoning", response.reasoning),
    ]
    for idx, slot in sorted(response.tool_calls.items()):
        scopes.append((f"tool_calls[{idx}].arguments", slot["arguments"]))
        scopes.append((f"tool_calls[{idx}].name", slot["name"]))
    for scope_name, text in scopes:
        for sentinel in STRUCTURAL_LEAK_MARKERS:
            if sentinel in text:
                failures.append(
                    f"forbidden sentinel {sentinel!r} leaked into"
                    f" {scope_name} — tool/reasoning parser failed to"
                    " strip it"
                )
                break
    return failures


def _check_finish_reason(
    response: _StreamedResponse, expected: Expectation
) -> list[str]:
    if response.finish_reason in expected.allowed_finish_reasons:
        return []
    return [
        f"finish_reason={response.finish_reason!r} not in allowed set"
        f" {sorted(expected.allowed_finish_reasons)}"
    ]


def _check_thinking_parity(
    response: _StreamedResponse, expected: Expectation
) -> list[str]:
    """Reasoning content must match the thinking mode.

    - ``must_have_reasoning is False`` (thinking off): any reasoning
      content is a "stray thinking token" — the reasoning parser saw
      output it shouldn't have.
    - ``must_have_reasoning is True``: missing reasoning is a regression
      in the thinking pathway.
    - ``must_have_reasoning is None``: no check (prompt-dependent).
    """
    if expected.must_have_reasoning is None:
        return []
    have = len(response.reasoning) > 0
    if expected.must_have_reasoning and not have:
        return [
            "expected non-empty reasoning (thinking=True) but none was streamed"
        ]
    if (not expected.must_have_reasoning) and have:
        return [
            "stray reasoning content present despite thinking=False"
            f" ({len(response.reasoning)} chars)"
        ]
    return []


def _check_reasoning_tokens(
    response: _StreamedResponse, expected: Expectation
) -> list[str]:
    """When required, usage must report a non-zero reasoning-token count.

    A stronger assertion than the thinking-parity check for prompts that
    reliably think: the model must have actually spent reasoning tokens,
    surfaced via ``usage.completion_tokens_details.reasoning_tokens``.
    """
    if not expected.require_reasoning_tokens:
        return []
    tokens = response.reasoning_tokens
    if tokens and tokens > 0:
        return []
    return [
        "expected non-zero usage.completion_tokens_details.reasoning_tokens"
        f" but got {tokens!r}"
    ]


def _declared_tool_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for tool in payload.get("tools") or []:
        fn = tool.get("function") or {}
        if isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def _check_tool_calls(response: _StreamedResponse, case: TestCase) -> list[str]:
    """Check tool-call presence + JSON parse + per-tool schema."""
    failures: list[str] = []
    expected = case.expectation
    declared = _declared_tool_names(case.payload)
    have_calls = len(response.tool_calls) > 0

    if expected.must_have_tool_calls is True and not have_calls:
        failures.append("expected at least one tool call; got zero")
    if expected.must_have_tool_calls is False and have_calls:
        failures.append(
            f"expected zero tool calls; got {len(response.tool_calls)}"
        )

    for idx, slot in sorted(response.tool_calls.items()):
        name = slot["name"]
        args = slot["arguments"]

        if declared and name not in declared:
            failures.append(
                f"tool_call[{idx}].name={name!r} is not in the request's"
                f" declared tools {sorted(declared)}"
            )

        try:
            parsed = json.loads(args) if args else None
        except json.JSONDecodeError as exc:
            failures.append(
                f"tool_call[{idx}].arguments did not parse as JSON: {exc}"
                f" (prefix: {args[:200]!r})"
            )
            continue

        if parsed is None:
            failures.append(
                f"tool_call[{idx}].arguments is empty; expected a JSON object"
            )
            continue

        schema = expected.tool_arg_schema
        if schema is not None:
            for err in Draft7Validator(schema).iter_errors(parsed):
                failures.append(
                    f"tool_call[{idx}].arguments violates schema: {err.message}"
                )

    return failures


def _check_response_format(
    response: _StreamedResponse, case: TestCase
) -> list[str]:
    """When the request set response_format=json_schema, content must
    conform. Failure here means constrained decoding broke.
    """
    schema = case.expectation.content_response_format_schema
    if schema is None:
        return []
    if not response.content:
        return []
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError as exc:
        # A length-truncated response is legitimately incomplete JSON —
        # that's a max_tokens artifact, not a constrained-decoding bug.
        if response.finish_reason == "length":
            return []
        return [
            "response_format=json_schema was set but content is not valid"
            f" JSON: {exc} (prefix: {response.content[:200]!r})"
        ]
    return [
        f"content violates response_format schema: {err.message}"
        for err in Draft7Validator(schema).iter_errors(parsed)
    ]


def _check_content_presence(
    response: _StreamedResponse, expected: Expectation
) -> list[str]:
    have = len(response.content) > 0
    if expected.must_have_content is True and not have:
        return ["expected non-empty assistant content; got empty"]
    if expected.must_have_content is False and have:
        return [
            "expected empty assistant content; got"
            f" {len(response.content)} chars"
        ]
    return []


def _check_auto_nonempty(
    response: _StreamedResponse, expected: Expectation
) -> list[str]:
    """For tool_choice=auto variants, the model must produce *something*.

    The model decides whether to call a tool, answer in prose, or both
    (a natural-language preamble alongside a tool call is correct). The
    only regression here is an empty turn: neither a tool call nor
    content.
    """
    if expected.must_have_tool_calls is not None:
        return []
    if expected.must_have_content is not None:
        return []
    if not response.tool_calls and not response.content:
        return [
            "tool_choice=auto: response is empty — neither a tool call nor"
            " assistant content was produced"
        ]
    return []


def _run_all_checks(response: _StreamedResponse, case: TestCase) -> list[str]:
    """Run every check in deterministic order; collect all failures."""
    failures: list[str] = []
    failures.extend(_check_sentinel_leak(response, case.expectation))
    failures.extend(_check_finish_reason(response, case.expectation))
    failures.extend(_check_thinking_parity(response, case.expectation))
    failures.extend(_check_reasoning_tokens(response, case.expectation))
    failures.extend(_check_content_presence(response, case.expectation))
    failures.extend(_check_auto_nonempty(response, case.expectation))
    failures.extend(_check_tool_calls(response, case))
    failures.extend(_check_response_format(response, case))
    return failures


@register_scenario
class BasicReasoningAndToolUsage(BaseScenario):
    name = "basic_reasoning_and_tool_usage"
    description = (
        "Minimal-viability matrix over the streaming API: tool calling,"
        " structured output, and content, each crossed with thinking on/off"
    )
    tags = ["validation", "tool_calling", "reasoning", "structured_output"]
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        for case in TESTS.values():
            payload = {"model": config.model, **case.payload}
            try:
                resp: RawResponse = await client.post_streaming(
                    payload, read_timeout=config.timeout * 4
                )
            except Exception as e:
                results.append(
                    self.make_result(
                        self.name, case.name, Verdict.ERROR, error=str(e)
                    )
                )
                continue

            if resp.error:
                results.append(
                    self.make_result(
                        self.name,
                        case.name,
                        Verdict.ERROR,
                        status_code=resp.status,
                        error=resp.error,
                    )
                )
                continue
            if resp.status != 200:
                results.append(
                    self.make_result(
                        self.name,
                        case.name,
                        Verdict.FAIL,
                        status_code=resp.status,
                        detail=(
                            f"valid request rejected with HTTP {resp.status}:"
                            f" {resp.body[:300]}"
                        ),
                    )
                )
                continue

            response = _accumulate_stream(resp.chunks or [])
            failures = _run_all_checks(response, case)
            if failures:
                results.append(
                    self.make_result(
                        self.name,
                        case.name,
                        Verdict.FAIL,
                        status_code=resp.status,
                        detail="; ".join(failures),
                    )
                )
            else:
                results.append(
                    self.make_result(
                        self.name,
                        case.name,
                        Verdict.PASS,
                        status_code=resp.status,
                        detail=(
                            f"finish={response.finish_reason}"
                            f" tool_calls={len(response.tool_calls)}"
                            f" content={len(response.content)}ch"
                            f" reasoning={len(response.reasoning)}ch"
                        ),
                    )
                )
        return results
