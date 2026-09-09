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
Scenario: Provider endpoint baseline
Target: Validate functional correctness of standard OpenAI API features.

Covers all 49 text-only tests from OpenRouter's 59-template provider
endpoint validation suite, plus 3 function-name-accuracy tests (weather tool
variants for auto tool_choice). 10 multimodal templates are skipped
(text-only LLM endpoint). Plus bonus streaming and misc tests.

Validation logic, prompts, and request shapes are aligned with OR's actual
test runner (extracted from dashboard Validation Result fields). See the
companion spec document for the full per-template mapping.

OR-equivalent test groups (49 tests, 1:1 with OR dashboard slugs):
    A. Basic Chat (4)         — yes-no, multi-turn, multipart-content, max-tokens
    B. System Prompt (2)      — multi-system-prompt, system-prompt-only
    C. Logprobs (1)           — top-logprobs
    D. Tools (7)              — tool-call-step-1/5, tool-choice-{auto,auto-weather,none,required,function}
    E. Structured Output (2)  — structured-output, response-format-json-object
    F. Reasoning (4)          — reasoning, reasoning-usage, reasoning-disabled, reasoning-max-tokens
    G. Reasoning Effort (6)   — reasoning-effort-{none,minimal,low,medium,high,xhigh}
    H. Reasoning+Tools (14)   — 7 tool variants x {reasoning-enabled, reasoning-disabled}
    I. Reasoning+JSON (4)     — {structured-output, response-format-json-object} x {reasoning-enabled, reasoning-disabled}
    J. Verbosity (4)          — verbosity-{low,medium,high,max}
    K. Misc (4)               — large-prompt, developer-role, assistant-prefill, fast-apply

Bonus tests (beyond OR's catalog):
    L. Streaming Variants (10) — SSE validation, tool streaming (with arg validation),
                                 JSON streaming (with schema validation), function streaming,
                                 reasoning+tool streaming, reasoning+required streaming,
                                 usage, finish_reason
    M. Extra Misc (2)          — seed_determinism, n_multiple_choices
    + health_check

Two-tier verdict system:
    core_verdict  — Feature MUST work.     400 → FAIL
    probe_verdict — Feature MAY be absent. 400 → INTERESTING
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from helpers import parse_json

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RawResponse, RunConfig


def _get_content(data: dict[str, Any]) -> str:
    return (
        data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    )


def _get_reasoning(data: dict[str, Any]) -> str:
    msg = data.get("choices", [{}])[0].get("message", {})
    # vLLM now uses `reasoning`; older responses used `reasoning_content`.
    return msg.get("reasoning", "") or msg.get("reasoning_content", "") or ""


def _get_finish_reason(data: dict[str, Any]) -> str | None:
    return data.get("choices", [{}])[0].get("finish_reason")


def _get_tool_calls(data: dict[str, Any]) -> list[Any] | None:
    return data.get("choices", [{}])[0].get("message", {}).get("tool_calls")


def _content_contains(data: dict[str, Any], needle: str) -> str | None:
    """Return error string if content doesn't contain needle (case-insensitive)."""
    content = _get_content(data)
    if not content:
        return "Empty content"
    if needle.lower() not in content.lower():
        return f"Content does not contain '{needle}': {content[:200]}"
    return None


def _args_not_json(args: str) -> bool:
    """Return True if a tool call's arguments string is not valid JSON."""
    try:
        json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return True
    return False


def _check_completion_tokens(data: dict[str, Any]) -> str | None:
    """Return error if completion_tokens is 0 or missing (OR: completion_tokens_positive)."""
    usage = data.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    if ct <= 0:
        return "No completion tokens"
    return None


# Regex for detecting actual leaked tool markup in content.
# Keep this narrow so harmless XML/HTML-like text does not trigger false failures.
TOOL_MARKUP_RE = re.compile(
    r"(<\|tool_[^>]+\|>|</?tool_calls?>)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# OpenRouter reasoning → upstream request translation
# ---------------------------------------------------------------------------
# OpenRouter translates its standard `reasoning` object into provider-specific
# params before hitting our endpoint. We replicate that translation here so
# our tests match what OpenRouter actually sends us.
#
# Key: OR sends reasoning_effort as a TOP-LEVEL param alongside chat_template_kwargs.


def _reasoning_kwargs(
    *,
    enabled: bool | None = None,
    effort: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Translate OpenRouter reasoning params to upstream request params."""
    result: dict[str, Any] = {}
    if effort == "none" or enabled is False:
        result["chat_template_kwargs"] = {
            "thinking": False,
            "enable_thinking": False,
        }
        if effort:
            result["reasoning_effort"] = effort
        return result
    result["chat_template_kwargs"] = {"thinking": True, "enable_thinking": True}
    if effort:
        result["reasoning_effort"] = effort
    if max_tokens is not None:
        result["chat_template_kwargs"]["thinking_budget"] = max_tokens
    return result


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _get_delta(cd: dict[str, Any]) -> dict[str, Any]:
    """Safely extract the delta from a parsed SSE chunk (handles empty choices)."""
    choices = cd.get("choices", [])
    if not choices:
        return {}
    return choices[0].get("delta", {})


def _stream_has_tool_delta(chunks: list[str]) -> bool:
    """Return True if any SSE chunk contains tool_calls in its delta."""
    for raw in chunks:
        if raw == "[DONE]":
            continue
        cd, _ = parse_json(raw)
        if cd and "tool_calls" in _get_delta(cd):
            return True
    return False


def _assemble_stream_content(chunks: list[str]) -> str:
    """Reassemble content text from SSE delta chunks."""
    assembled = ""
    for raw in chunks:
        if raw == "[DONE]":
            continue
        cd, _ = parse_json(raw)
        if cd:
            assembled += _get_delta(cd).get("content", "") or ""
    return assembled


def _assemble_stream_tool_args(chunks: list[str]) -> dict[int, str]:
    """Reassemble tool call arguments from streaming deltas, keyed by tool index.

    Every tool index seen in any delta is tracked — even if no argument
    fragments were emitted — so callers can detect tool calls with empty args.
    """
    args_by_index: dict[int, str] = {}
    for raw in chunks:
        if raw == "[DONE]":
            continue
        cd, _ = parse_json(raw)
        if not cd:
            continue
        delta = _get_delta(cd)
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            fn_args = (tc.get("function") or {}).get("arguments", "")
            args_by_index.setdefault(idx, "")
            if fn_args:
                args_by_index[idx] += fn_args
    return args_by_index


def _get_stream_tool_names(chunks: list[str]) -> list[str]:
    """Extract function names from streaming tool call deltas."""
    names: list[str] = []
    for raw in chunks:
        if raw == "[DONE]":
            continue
        cd, _ = parse_json(raw)
        if not cd:
            continue
        delta = _get_delta(cd)
        for tc in delta.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if name:
                names.append(name)
    return names


def _get_stream_finish_reason(chunks: list[str]) -> str | None:
    """Get the finish_reason from the last non-DONE chunk."""
    for raw in reversed(chunks):
        if raw == "[DONE]":
            continue
        cd, _ = parse_json(raw)
        if cd:
            choices = cd.get("choices", [])
            if choices and choices[0].get("finish_reason"):
                return choices[0]["finish_reason"]
    return None


# Kimi-specific markers that should never appear in user-visible content or arguments.
_LEAKED_MARKERS = [
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_section_begin|>",
    "<|tool_call_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
]


def _check_no_markers(text: str) -> str | None:
    """Return error if text contains leaked model-specific markers."""
    for marker in _LEAKED_MARKERS:
        if marker in text:
            return f"Leaked marker {marker!r} in output: ...{text[max(0, text.index(marker) - 20) : text.index(marker) + 40]}..."
    return None


def _validate_tool_args_json(tool_calls: list[Any]) -> str | None:
    """Validate that every tool call has parseable JSON arguments."""
    for i, tc in enumerate(tool_calls):
        args_str = (tc.get("function") or {}).get("arguments", "")
        if not args_str:
            return f"tool_calls[{i}]: empty arguments"
        try:
            json.loads(args_str)
        except (json.JSONDecodeError, TypeError) as e:
            return f"tool_calls[{i}]: arguments not valid JSON: {e} (got {args_str[:100]!r})"
    return None


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# OR uses get_current_weather for tool-choice-none/function tests where the
# prompt naturally triggers tool use, exposing tool_choice enforcement bugs.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

WEATHER_PROMPT = "What is the weather like in Boston, MA in fahrenheit?"

# OR uses a calculate tool for tool-call-step and tool-choice-required tests.
CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '2 + 2'",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# Forced-tool runaway thresholds for the tool-choice-required validators.
# Above this many tool calls the repetition attractor is certain, even if the
# model escaped the grammar before the token cap.
_RUNAWAY_CALL_FAIL_THRESHOLD = 8
# Completion-token ceiling for a clean forced-tool answer to the scenario's
# trivial prompts: reasoning plus a handful of calls fits in a few hundred
# tokens, so a response burning more than this is a runaway even when it
# stops before the request's max_tokens (which stays at OR's 65536 shape).
_RUNAWAY_COMPLETION_BUDGET = 4096

WEATHER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "weather",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "temperature", "unit"],
            "additionalProperties": False,
        },
    },
}

# OR's multi-step tool conversation fixture
TOOL_RESULT_MESSAGES = [
    {"role": "user", "content": "What is 2 + 2?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "2 + 2"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_001", "content": '{"result": 4}'},
]

# OR's reasoning test prompt (multi-turn apple riddle)
REASONING_MESSAGES = [
    {
        "role": "user",
        "content": "I have 4 apples. I give 2 to my friend. How many apples do we have now?",
    },
    {"role": "assistant", "content": "What do you want me to do?"},
    {"role": "user", "content": "Please think and answer my original question"},
]


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@register_scenario
class ProviderBaseline(BaseScenario):
    name = "openrouter_tests"
    description = (
        "OpenRouter provider endpoint baseline: validates functional "
        "correctness of standard OpenAI API features"
    )
    tags = ["validation", "baseline", "compliance", "functional"]
    scenario_type = "validation"

    # -- request builders --------------------------------------------------

    def _req(
        self,
        model: str,
        content: str | None,
        *,
        messages: list[Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        # OR sends max_tokens=65536 on almost every test. For reasoning models,
        # low values cause empty content (all tokens burned on reasoning).
        # Respect the model's actual context limit when running locally.
        max_tokens = min(65536, self._fuzz_config.model_config.max_num_tokens)
        p: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
        if messages is not None:
            p["messages"] = messages
        else:
            p["messages"] = [{"role": "user", "content": content}]
        p.update(extra)
        # OR always injects thinking kwargs into upstream requests. Replicate
        # that here so tests that don't explicitly set reasoning params still
        # get thinking enabled (matching OR's actual behavior).
        if "chat_template_kwargs" not in p:
            p["chat_template_kwargs"] = {
                "thinking": True,
                "enable_thinking": True,
            }
        return p

    def _with_tools(
        self,
        model: str,
        content: str = "Hi, how are you?",
        **extra: Any,
    ) -> dict[str, Any]:
        p = self._req(model, content, tools=[CALCULATE_TOOL])
        p.update(extra)
        return p

    # -- verdict helpers ---------------------------------------------------

    @staticmethod
    def _core_verdict(
        status: int,
        data: dict[str, Any] | None = None,
        *,
        check: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> tuple[Verdict, str]:
        """Verdict for core features: 200+valid=PASS, 400=FAIL, 5xx=FAIL."""
        if status >= 500:
            return Verdict.FAIL, f"Server error {status}"
        if status == 400:
            return Verdict.FAIL, "Core feature rejected (400)"
        if status != 200:
            return Verdict.INTERESTING, f"Status {status}"
        if data is None:
            return Verdict.FAIL, "Invalid JSON response"
        if check:
            err = check(data)
            if err:
                return Verdict.FAIL, err
        return Verdict.PASS, "OK"

    @staticmethod
    def _probe_verdict(
        status: int,
        data: dict[str, Any] | None = None,
        *,
        check: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> tuple[Verdict, str]:
        """Verdict for optional/probe features: 400=INTERESTING, 5xx=FAIL."""
        if status >= 500:
            return Verdict.FAIL, f"Server error {status}"
        if status == 400:
            return Verdict.INTERESTING, "Feature not supported (400)"
        if status != 200:
            return Verdict.INTERESTING, f"Status {status}"
        if data is None:
            return Verdict.FAIL, "Invalid JSON response"
        if check:
            err = check(data)
            if err:
                return Verdict.FAIL, err
        return Verdict.PASS, "OK"

    @staticmethod
    def _validate_weather_schema(parsed: object) -> str | None:
        """Validate the strict weather schema used by this scenario."""
        if not isinstance(parsed, dict):
            return f"Expected JSON object, got {type(parsed).__name__}"

        expected_keys = {"location", "temperature", "unit"}
        actual_keys = set(parsed.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            parts = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if extra:
                parts.append(f"extra keys: {extra}")
            return "; ".join(parts)

        if not isinstance(parsed["location"], str):
            return "location must be a string"
        if not isinstance(parsed["temperature"], (int, float)) or isinstance(
            parsed["temperature"], bool
        ):
            return "temperature must be a number"
        if parsed["unit"] not in {"celsius", "fahrenheit"}:
            return "unit must be one of ['celsius', 'fahrenheit']"
        return None

    @classmethod
    def _validate_json_content(
        cls,
        resp: RawResponse,
        *,
        label: str = "",
        validator: Callable[[object], str | None] | None = None,
    ) -> tuple[Verdict, str]:
        """Validate that a 200 response contains parseable JSON in content."""
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                content = _get_content(data)
                if content:
                    parsed, json_err = parse_json(content)
                    if not json_err:
                        if validator:
                            schema_err = validator(parsed)
                            if schema_err:
                                return (
                                    Verdict.FAIL,
                                    f"JSON schema mismatch: {schema_err}",
                                )
                        return (
                            Verdict.PASS,
                            f"Valid JSON content{' (' + label + ')' if label else ''}",
                        )
                    return Verdict.FAIL, f"Content not valid JSON: {json_err}"
                return Verdict.FAIL, "Empty content"
            return Verdict.FAIL, "Invalid JSON response"
        if resp.status == 400:
            return (
                Verdict.INTERESTING,
                f"Server rejects {label or 'request'} (400)",
            )
        if resp.status >= 500:
            return Verdict.FAIL, f"Server error {resp.status}"
        return Verdict.INTERESTING, f"Status {resp.status}"

    @staticmethod
    def _check_tool_choice_none(data: dict[str, Any]) -> str | None:
        """OR validation for tool_choice=none: no tool_calls AND no tool markup in content."""
        tc = _get_tool_calls(data)
        if tc:
            return "tool_calls present despite tool_choice=none"
        content = _get_content(data)
        if content and TOOL_MARKUP_RE.search(content):
            return f"Tool markup in content despite tool_choice=none: {content[:100]}"
        return None

    @staticmethod
    def _check_tool_choice_required(data: dict[str, Any]) -> str | None:
        """OR validation for tool_choice=required.

        Beyond OR's own finish_reason=tool_calls check, this classifies
        forced-tool runaway: a ``required`` grammar masks EOS until the
        tool-call envelope closes, so a model that reasoned its way to "no
        tool needed" can fall into a repetition attractor — it keeps opening
        new calls instead of closing the envelope, running until the token
        cap (observed on Kimi-K2.6 via OpenRouter: ~1900 incrementing
        calculate calls, with the truncated trailing call's non-JSON
        arguments leaked on the length stop).

        Runaway signatures, most severe first: finish_reason="length", an
        excessive call count, an excessive completion-token count, or a call
        with non-JSON arguments.
        """
        fr = _get_finish_reason(data)
        tool_calls = _get_tool_calls(data) or []
        n_calls = len(tool_calls)
        completion_tokens = (data.get("usage") or {}).get("completion_tokens")
        stats = (
            f"calls={n_calls}, finish_reason={fr}, "
            f"completion_tokens={completion_tokens}"
        )

        if fr == "length":
            return f"Forced-tool runaway: hit token cap ({stats})"
        if n_calls > _RUNAWAY_CALL_FAIL_THRESHOLD:
            return f"Forced-tool runaway: excessive call count ({stats})"
        if (
            isinstance(completion_tokens, int)
            and completion_tokens > _RUNAWAY_COMPLETION_BUDGET
        ):
            return f"Forced-tool runaway: excessive completion tokens ({stats})"

        bad_args = [
            i
            for i, tc in enumerate(tool_calls)
            if _args_not_json((tc.get("function") or {}).get("arguments", ""))
        ]
        if bad_args:
            return f"Non-JSON arguments in tool call(s) {bad_args} ({stats})"
        if not tool_calls:
            return f"No tool_calls ({stats})"
        if fr != "tool_calls":
            return f"tool_calls present but finish_reason={fr} (expected tool_calls)"
        return None

    @staticmethod
    def _tool_choice_function_verdict(
        data: dict[str, Any], func_name: str
    ) -> tuple[Verdict, str]:
        """Validation for tool_choice=function with compatibility fallback.

        Some OpenAI-compatible providers/adapters return finish_reason="stop"
        for forced/named tool choice while still emitting valid tool_calls.
        Treat that as INTERESTING rather than FAIL, but still verify the
        selected function name.
        """
        fr = _get_finish_reason(data)
        tc = _get_tool_calls(data)
        if not tc:
            return Verdict.FAIL, f"No tool_calls, finish_reason={fr}"
        actual_name = tc[0].get("function", {}).get("name")
        if actual_name != func_name:
            return (
                Verdict.FAIL,
                f"Wrong function: {actual_name} (expected {func_name})",
            )
        if fr == "tool_calls":
            return (
                Verdict.PASS,
                "tool_calls present with finish_reason=tool_calls",
            )
        if fr == "stop":
            return (
                Verdict.INTERESTING,
                "tool_calls present with finish_reason=stop for named tool_choice; "
                "some OpenAI-compatible APIs historically use stop here",
            )
        return (
            Verdict.FAIL,
            f"tool_calls present but finish_reason={fr} (expected tool_calls or stop)",
        )

    @classmethod
    def _check_tool_choice_function(
        cls, data: dict[str, Any], func_name: str
    ) -> str | None:
        """Compatibility wrapper for existing core/probe helper call sites."""
        verdict, detail = cls._tool_choice_function_verdict(data, func_name)
        return None if verdict == Verdict.PASS else detail

    def _exchange_verbose(
        self,
        payload: dict[str, Any] | str | None,
        resp: RawResponse | None,
        *,
        alt_resp: RawResponse | None = None,
    ) -> dict[str, str]:
        """When RunConfig.verbose is set, attach request/response/error for JSON export."""
        cfg = getattr(self, "_fuzz_config", None)
        if cfg is None or not getattr(cfg, "verbose", False):
            return {}
        out: dict[str, str] = {
            "request_body": "",
            "response_body": "",
            "error": "",
        }
        if payload is not None:
            if isinstance(payload, dict):
                try:
                    out["request_body"] = json.dumps(
                        payload, ensure_ascii=False
                    )
                except (TypeError, ValueError):
                    out["request_body"] = str(payload)
            else:
                out["request_body"] = str(payload)
        if resp is not None:
            body = getattr(resp, "body", None) or ""
            chunks = getattr(resp, "chunks", None)
            if not body and chunks:
                try:
                    body = json.dumps(chunks, ensure_ascii=False)
                except (TypeError, ValueError):
                    body = "\n".join(str(c) for c in chunks)
            if alt_resp is not None:
                b2 = getattr(alt_resp, "body", None) or ""
                try:
                    body = json.dumps(
                        {"first": body, "second": b2}, ensure_ascii=False
                    )
                except (TypeError, ValueError):
                    body = f"first={body!r}\nsecond={b2!r}"
            out["response_body"] = body
            errs: list[str] = []
            e1 = getattr(resp, "error", None)
            if e1:
                errs.append(str(e1))
            if alt_resp is not None:
                e2 = getattr(alt_resp, "error", None)
                if e2:
                    errs.append(f"second: {e2}")
            out["error"] = "; ".join(errs) if errs else ""
        return out

    # -- main entry --------------------------------------------------------

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        self._fuzz_config = config
        results: list[ScenarioResult] = []
        model = config.model

        # --- 49 tests matching OR's text-only templates 1:1 ---
        results.extend(await self._basic_chat(client, model))  # 4
        results.extend(await self._system_prompt(client, model))  # 2
        results.extend(await self._logprobs(client, model))  # 1
        results.extend(await self._tools(client, model))  # 6
        results.extend(await self._structured_output(client, model))  # 2
        results.extend(await self._reasoning(client, model))  # 4
        results.extend(await self._reasoning_effort(client, model))  # 6
        results.extend(await self._reasoning_tools(client, model))  # 12
        results.extend(await self._reasoning_json(client, model))  # 4
        results.extend(await self._verbosity(client, model))  # 4
        results.extend(await self._misc(client, config))  # 4
        # Total: 49

        # --- Bonus tests (beyond OR's 59-template catalog) ---
        results.extend(await self._streaming_variants(client, model))
        results.extend(await self._extra_misc(client, config))

        # Health check (payload mirrors FuzzClient.health_check)
        resp = await client.health_check()
        hc_payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        results.append(
            self.make_result(
                self.name,
                "health_check",
                Verdict.PASS if resp.status == 200 else Verdict.FAIL,
                status_code=resp.status,
                detail="Post-scenario health check",
                **self._exchange_verbose(hc_payload, resp),
            )
        )

        return results

    # =====================================================================
    # A. Basic Chat (4 tests — OR: YesNo, MultiTurn, MultipartContent, MaxTokens)
    # =====================================================================

    async def _basic_chat(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # OR: YesNo — deliberately ambiguous; do not require a literal "yes".
        # This test sends thinking kwargs (via _req default), so reasoning
        # should be present. If reasoning fields exist they must be non-empty.
        pl_yesno = self._req(model, "Yes or no?")
        resp = await client.post_json(
            pl_yesno, timeout=self._fuzz_config.timeout * 2
        )
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        if resp.status == 200 and data:
            content = _get_content(data)
            reasoning = _get_reasoning(data)
            msg = data.get("choices", [{}])[0].get("message", {})
            reasoning_fields = "reasoning" in msg or "reasoning_content" in msg
            if not content:
                v, d = Verdict.FAIL, "Empty content"
            elif not reasoning:
                if reasoning_fields:
                    v, d = Verdict.FAIL, "Reasoning field present but empty"
                else:
                    v, d = (
                        Verdict.FAIL,
                        "No reasoning in response (thinking kwargs sent)",
                    )
            else:
                v, d = (
                    Verdict.PASS,
                    (
                        f"Answer present ({len(content)} chars), "
                        f"reasoning present ({len(reasoning)} chars)"
                    ),
                )
        else:
            v, d = self._core_verdict(resp.status, data)
        result(
            "yes-no",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_yesno, resp),
        )

        # OR: MultiTurn — content_contains("42")
        pl_multi = self._req(
            model,
            None,
            messages=[
                {"role": "user", "content": "Remember this number: 42"},
                {
                    "role": "assistant",
                    "content": "I'll remember the number 42.",
                },
                {
                    "role": "user",
                    "content": "What number did I ask you to remember? Reply with just the number.",
                },
            ],
        )
        resp = await client.post_json(pl_multi)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status, data, check=lambda d: _content_contains(d, "42")
        )
        result(
            "multi-turn",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_multi, resp),
        )

        # OR: MultipartContent — content_contains("4")
        pl_multi_part = self._req(
            model,
            None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is 2+2?"},
                        {"type": "text", "text": "Reply with just the number."},
                    ],
                }
            ],
        )
        resp = await client.post_json(pl_multi_part)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status, data, check=lambda d: _content_contains(d, "4")
        )
        result(
            "multipart-content",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_multi_part, resp),
        )

        # OR: MaxTokens — completion_tokens_positive (OR uses max_tokens=500)
        pl_max_tok = self._req(
            model,
            "Tell me a 1000 word bedtime story",
            max_tokens=500,
        )
        resp = await client.post_json(
            pl_max_tok, timeout=self._fuzz_config.timeout * 2
        )
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status, data, check=_check_completion_tokens
        )
        result(
            "max-tokens",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_max_tok, resp),
        )

        return results

    # =====================================================================
    # B. System Prompt (2 tests — OR: MultiSystemPrompt, SystemPromptOnly)
    # =====================================================================

    async def _system_prompt(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # OR: MultiSystemPrompt — content_contains("Paris")
        pl_ms = self._req(
            model,
            None,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "system",
                    "content": "Always respond in exactly 3 words.",
                },
                {"role": "user", "content": "What is the capital of France?"},
            ],
        )
        resp = await client.post_json(
            pl_ms, timeout=self._fuzz_config.timeout * 2
        )
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status, data, check=lambda d: _content_contains(d, "Paris")
        )
        result(
            "multi-system-prompt",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_ms, resp),
        )

        # OR: SystemPromptOnly — status_ok + content_non_empty
        pl_spo = self._req(
            model,
            None,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Start by greeting the user and introducing yourself.",
                },
            ],
        )
        resp = await client.post_json(pl_spo)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status,
            data,
            check=lambda d: None if _get_content(d) else "Empty response",
        )
        result(
            "system-prompt-only",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_spo, resp),
        )

        return results

    # =====================================================================
    # C. Logprobs (1 test — OR: TopLogprobs)
    # =====================================================================

    async def _logprobs(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        pl_lp = self._req(model, "Hello", logprobs=True, top_logprobs=5)
        resp = await client.post_json(pl_lp)
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                choice = data.get("choices", [{}])[0]
                lp = choice.get("logprobs")
                if lp is None or not isinstance(lp, dict):
                    v, d = Verdict.FAIL, "logprobs missing or not an object"
                else:
                    content_lps = lp.get("content")
                    if isinstance(content_lps, list) and len(content_lps) > 0:
                        tl = content_lps[0].get("top_logprobs")
                        if isinstance(tl, list) and len(tl) > 0:
                            v, d = (
                                Verdict.PASS,
                                f"logprobs valid, {len(content_lps)} tokens, top_logprobs present",
                            )
                        else:
                            v, d = Verdict.FAIL, "top_logprobs missing or empty"
                    else:
                        v, d = (
                            Verdict.FAIL,
                            "logprobs.content empty or wrong type",
                        )
            else:
                v, d = Verdict.FAIL, "Invalid JSON"
        elif resp.status == 400:
            # MAX's overlap pipeline does not currently support logprobs, so a
            # clean 400 rejection is the expected, correct behavior here rather
            # than a divergence to investigate.
            v, d = (
                Verdict.PASS,
                "Server correctly rejects unsupported logprobs (400)",
            )
        else:
            v = Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            d = f"Status {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "top-logprobs",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_lp, resp),
            )
        )

        return results

    # =====================================================================
    # D. Tools (6 tests — OR: ToolCallStep1/5, ToolChoice{Auto,None,Required,Function})
    # =====================================================================

    async def _tools(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # OR: ToolCallStep1 — tool_calls_present + function name matches + valid args
        pl_t1 = self._with_tools(model, "What is 2 + 2?")
        resp = await client.post_json(pl_t1)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        if resp.status == 200 and data:
            tc = _get_tool_calls(data)
            if not tc:
                v, d = Verdict.FAIL, "tool_calls absent"
            else:
                actual_name = tc[0].get("function", {}).get("name")
                args_err = _validate_tool_args_json(tc)
                marker_err = _check_no_markers(str(tc))
                if actual_name != "calculate":
                    v, d = (
                        Verdict.FAIL,
                        f"Wrong function name: {actual_name!r} (expected 'calculate')",
                    )
                elif args_err:
                    v, d = Verdict.FAIL, args_err
                elif marker_err:
                    v, d = Verdict.FAIL, marker_err
                else:
                    v, d = (
                        Verdict.PASS,
                        "tool_calls present, correct name, valid JSON args",
                    )
        else:
            v, d = self._core_verdict(resp.status, data)
        result(
            "tool-call-step-1",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_t1, resp),
        )

        # OR: ToolCallStep5 — content_non_empty (text response after tool result)
        pl_t5 = self._req(
            model,
            None,
            messages=TOOL_RESULT_MESSAGES,
            tools=[CALCULATE_TOOL],
        )
        resp = await client.post_json(pl_t5)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status,
            data,
            check=lambda d: None if _get_content(d) else "Empty response",
        )
        result(
            "tool-call-step-5",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_t5, resp),
        )

        # OR: ToolChoiceAuto — status_ok + function name validation
        pl_tca = self._with_tools(model, tool_choice="auto")
        resp = await client.post_json(pl_tca)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        if resp.status == 200 and data:
            tc = _get_tool_calls(data)
            if tc:
                actual_name = tc[0].get("function", {}).get("name")
                if actual_name != "calculate":
                    v, d = (
                        Verdict.FAIL,
                        f"Wrong function name: {actual_name!r} (expected 'calculate')",
                    )
                else:
                    v, d = (
                        Verdict.PASS,
                        "tool_calls present with correct function name",
                    )
            else:
                v, d = self._core_verdict(resp.status, data)
        else:
            v, d = self._core_verdict(resp.status, data)
        result(
            "tool-choice-auto",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_tca, resp),
        )

        # OR: ToolChoiceAuto with weather tool — function name accuracy
        # Matches OR's weather-tool variant that validates returned function name.
        pl_tca_w = self._req(
            model,
            WEATHER_PROMPT,
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        )
        resp = await client.post_json(pl_tca_w)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        if resp.status == 200 and data:
            tc = _get_tool_calls(data)
            if not tc:
                v, d = Verdict.FAIL, "tool_calls absent for weather prompt"
            else:
                actual_name = tc[0].get("function", {}).get("name")
                if actual_name != "get_current_weather":
                    v, d = (
                        Verdict.FAIL,
                        f"Wrong function name: {actual_name!r} (expected 'get_current_weather')",
                    )
                else:
                    v, d = (
                        Verdict.PASS,
                        "tool_calls present with correct function name",
                    )
        else:
            v, d = self._core_verdict(resp.status, data)
        result(
            "tool-choice-auto-weather",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_tca_w, resp),
        )

        # OR: ToolChoiceNone — content_not_matches_regex("<[^>]+>")
        # OR uses weather prompt + weather tool to stress-test tool_choice enforcement.
        pl_tcn = self._req(
            model,
            WEATHER_PROMPT,
            tools=[WEATHER_TOOL],
            tool_choice="none",
        )
        resp = await client.post_json(pl_tcn)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status, data, check=self._check_tool_choice_none
        )
        result(
            "tool-choice-none",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_tcn, resp),
        )

        # OR: ToolChoiceRequired — finish_reason_equals("tool_calls")
        pl_tcr = self._with_tools(model, tool_choice="required")
        resp = await client.post_json(pl_tcr)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status, data, check=self._check_tool_choice_required
        )
        result(
            "tool-choice-required",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_tcr, resp),
        )

        # OR: ToolChoiceFunction — finish_reason_equals("tool_calls") AND name matches
        # OR uses weather prompt + both tools, forcing the calculate function.
        pl_tcf = self._req(
            model,
            WEATHER_PROMPT,
            tools=[WEATHER_TOOL, CALCULATE_TOOL],
            tool_choice={"type": "function", "function": {"name": "calculate"}},
        )
        resp = await client.post_json(pl_tcf)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        if resp.status == 200 and data:
            v, d = self._tool_choice_function_verdict(data, "calculate")
        else:
            v, d = self._core_verdict(resp.status, data)
        result(
            "tool-choice-function",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_tcf, resp),
        )

        return results

    # =====================================================================
    # E. Structured Output (2 tests — OR: StructuredOutput, ResponseFormatJSONObject)
    # =====================================================================

    async def _structured_output(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # OR: StructuredOutput — valid_json_content AND matches json_schema
        pl_so = self._req(
            model,
            "Give me the weather for San Francisco in fahrenheit.",
            response_format=WEATHER_SCHEMA,
        )
        resp = await client.post_json(pl_so)
        v, d = self._validate_json_content(
            resp,
            label="json_schema",
            validator=self._validate_weather_schema,
        )
        result(
            "structured-output",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_so, resp),
        )

        # OR: ResponseFormatJSONObject — valid_json_content
        pl_jo = self._req(
            model,
            "Return a JSON object with a greeting field.",
            response_format={"type": "json_object"},
        )
        resp = await client.post_json(pl_jo)
        v, d = self._validate_json_content(resp, label="json_object")
        result(
            "response-format-json-object",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_jo, resp),
        )

        return results

    # =====================================================================
    # F. Reasoning (4 probe tests)
    #    Uses OR's apple riddle prompt and chat_template_kwargs translation.
    # =====================================================================

    async def _reasoning(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # OR: Reasoning — content non-empty AND reasoning_content non-empty
        pl_r = self._req(
            model,
            None,
            messages=REASONING_MESSAGES,
            **_reasoning_kwargs(enabled=True),
        )
        resp = await client.post_json(
            pl_r, timeout=self._fuzz_config.timeout * 2
        )
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        if resp.status == 200 and data:
            content = _get_content(data)
            reasoning = _get_reasoning(data)
            if not content:
                v, d = Verdict.FAIL, "Empty content"
            elif not reasoning:
                v, d = Verdict.FAIL, "No reasoning_content in response"
            else:
                v, d = (
                    Verdict.PASS,
                    f"Content present ({len(content)} chars), reasoning present ({len(reasoning)} chars)",
                )
        else:
            v, d = self._probe_verdict(resp.status, data)
        result(
            "reasoning",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_r, resp),
        )

        # OR: ReasoningUsage — completion_tokens_details.reasoning_tokens > 0
        pl_ru = self._req(
            model,
            None,
            messages=REASONING_MESSAGES,
            **_reasoning_kwargs(enabled=True),
        )
        resp = await client.post_json(
            pl_ru, timeout=self._fuzz_config.timeout * 2
        )
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                usage = data.get("usage", {})
                ctd = usage.get("completion_tokens_details", {})
                if isinstance(ctd, dict) and ctd.get("reasoning_tokens", 0) > 0:
                    v, d = (
                        Verdict.PASS,
                        f"reasoning_tokens={ctd['reasoning_tokens']}",
                    )
                elif isinstance(ctd, dict) and "reasoning_tokens" in ctd:
                    v, d = (
                        Verdict.FAIL,
                        f"reasoning_tokens={ctd['reasoning_tokens']} (expected >0)",
                    )
                else:
                    v, d = Verdict.INTERESTING, "No reasoning_tokens in usage"
            else:
                v, d = Verdict.FAIL, "Invalid JSON"
        else:
            v, d = self._probe_verdict(resp.status)
        result(
            "reasoning-usage",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_ru, resp),
        )

        # OR: ReasoningDisabled — content non-empty AND reasoning_content empty
        pl_rd = self._req(
            model,
            None,
            messages=REASONING_MESSAGES,
            **_reasoning_kwargs(enabled=False),
        )
        resp = await client.post_json(pl_rd)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        if resp.status == 200 and data:
            content = _get_content(data)
            reasoning = _get_reasoning(data)
            if not content:
                v, d = Verdict.FAIL, "Empty content"
            elif reasoning:
                v, d = (
                    Verdict.FAIL,
                    f"reasoning_content present despite disabled ({len(reasoning)} chars)",
                )
            else:
                v, d = Verdict.PASS, "Content present, no reasoning"
        else:
            v, d = self._probe_verdict(resp.status, data)
        result(
            "reasoning-disabled",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_rd, resp),
        )

        # OR: ReasoningMaxTokens — completion_tokens_positive
        pl_rmt = self._req(
            model,
            None,
            messages=REASONING_MESSAGES,
            **_reasoning_kwargs(max_tokens=128),
        )
        resp = await client.post_json(
            pl_rmt, timeout=self._fuzz_config.timeout * 2
        )
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._probe_verdict(
            resp.status, data, check=_check_completion_tokens
        )
        result(
            "reasoning-max-tokens",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_rmt, resp),
        )

        return results

    # =====================================================================
    # G. Reasoning Effort (6 probe tests — OR: ReasoningEffort{None..XHigh})
    #    OR validation: completion_tokens_positive
    # =====================================================================

    async def _reasoning_effort(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        for level in ("none", "minimal", "low", "medium", "high", "xhigh"):
            pl_eff = self._req(
                model,
                None,
                messages=REASONING_MESSAGES,
                **_reasoning_kwargs(effort=level),
            )
            resp = await client.post_json(
                pl_eff, timeout=self._fuzz_config.timeout * 2
            )
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            v, d = self._probe_verdict(
                resp.status, data, check=_check_completion_tokens
            )
            result(
                f"reasoning-effort-{level}",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_eff, resp),
            )

        return results

    # =====================================================================
    # H. Reasoning + Tools (12 probe tests)
    #    OR validation: status_ok for step1/step5/auto,
    #                   content_not_matches_regex for none,
    #                   finish_reason=tool_calls for required/function
    # =====================================================================

    async def _reasoning_tools(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        for reasoning_on in (True, False):
            prefix = (
                "reasoning-enabled" if reasoning_on else "reasoning-disabled"
            )
            rkw = _reasoning_kwargs(enabled=reasoning_on)

            # Step-1: single tool call — tool_calls_present + function name matches
            pl_rt1 = self._with_tools(model, "What is 2 + 2?", **rkw)
            resp = await client.post_json(
                pl_rt1, timeout=self._fuzz_config.timeout * 2
            )
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            if resp.status == 200 and data:
                tc = _get_tool_calls(data)
                if not tc:
                    v, d = Verdict.FAIL, "tool_calls absent"
                else:
                    actual_name = tc[0].get("function", {}).get("name")
                    if actual_name != "calculate":
                        v, d = (
                            Verdict.FAIL,
                            f"Wrong function name: {actual_name!r} (expected 'calculate')",
                        )
                    else:
                        v, d = (
                            Verdict.PASS,
                            "tool_calls present with correct function name",
                        )
            else:
                v, d = self._probe_verdict(resp.status, data)
            result(
                f"{prefix}-tool-call-step-1",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_rt1, resp),
            )

            # Step-5: multi-step tool conversation — status_ok
            pl_rt5 = self._req(
                model,
                None,
                messages=TOOL_RESULT_MESSAGES,
                tools=[CALCULATE_TOOL],
                **rkw,
            )
            resp = await client.post_json(
                pl_rt5, timeout=self._fuzz_config.timeout * 2
            )
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            v, d = self._probe_verdict(resp.status, data)
            result(
                f"{prefix}-tool-call-step-5",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_rt5, resp),
            )

            # ToolChoiceAuto — status_ok + function name validation
            pl_rta = self._with_tools(model, tool_choice="auto", **rkw)
            resp = await client.post_json(
                pl_rta, timeout=self._fuzz_config.timeout * 2
            )
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            if resp.status == 200 and data:
                tc = _get_tool_calls(data)
                if tc:
                    actual_name = tc[0].get("function", {}).get("name")
                    if actual_name != "calculate":
                        v, d = (
                            Verdict.FAIL,
                            f"Wrong function name: {actual_name!r} (expected 'calculate')",
                        )
                    else:
                        v, d = (
                            Verdict.PASS,
                            "tool_calls present with correct function name",
                        )
                else:
                    v, d = self._probe_verdict(resp.status, data)
            else:
                v, d = self._probe_verdict(resp.status, data)
            result(
                f"{prefix}-tool-choice-auto",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_rta, resp),
            )

            # ToolChoiceAuto with weather tool — function name accuracy
            # Matches OR's weather-tool variant (reasoning + auto + weather).
            pl_rta_w = self._req(
                model,
                WEATHER_PROMPT,
                tools=[WEATHER_TOOL],
                tool_choice="auto",
                **rkw,
            )
            resp = await client.post_json(
                pl_rta_w, timeout=self._fuzz_config.timeout * 2
            )
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            if resp.status == 200 and data:
                tc = _get_tool_calls(data)
                if not tc:
                    v, d = Verdict.FAIL, "tool_calls absent for weather prompt"
                else:
                    actual_name = tc[0].get("function", {}).get("name")
                    if actual_name != "get_current_weather":
                        v, d = (
                            Verdict.FAIL,
                            f"Wrong function name: {actual_name!r} (expected 'get_current_weather')",
                        )
                    else:
                        v, d = (
                            Verdict.PASS,
                            "tool_calls present with correct function name",
                        )
            else:
                v, d = self._probe_verdict(resp.status, data)
            result(
                f"{prefix}-tool-choice-auto-weather",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_rta_w, resp),
            )

            # ToolChoiceNone — content_not_matches_regex("<[^>]+>")
            # OR uses weather prompt + weather tool to stress-test enforcement.
            pl_rtn = self._req(
                model,
                WEATHER_PROMPT,
                tools=[WEATHER_TOOL],
                tool_choice="none",
                **rkw,
            )
            resp = await client.post_json(
                pl_rtn, timeout=self._fuzz_config.timeout * 2
            )
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            v, d = self._probe_verdict(
                resp.status, data, check=self._check_tool_choice_none
            )
            result(
                f"{prefix}-tool-choice-none",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_rtn, resp),
            )

            # ToolChoiceRequired — finish_reason_equals("tool_calls")
            pl_rtr = self._with_tools(model, tool_choice="required", **rkw)
            resp = await client.post_json(
                pl_rtr, timeout=self._fuzz_config.timeout * 2
            )
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            v, d = self._probe_verdict(
                resp.status, data, check=self._check_tool_choice_required
            )
            result(
                f"{prefix}-tool-choice-required",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_rtr, resp),
            )

            # ToolChoiceFunction — finish_reason=tool_calls AND name matches
            # OR uses weather prompt + both tools, forcing the calculate function.
            pl_rtf = self._req(
                model,
                WEATHER_PROMPT,
                tools=[WEATHER_TOOL, CALCULATE_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "calculate"},
                },
                **rkw,
            )
            resp = await client.post_json(
                pl_rtf, timeout=self._fuzz_config.timeout * 2
            )
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            if resp.status == 200 and data:
                v, d = self._tool_choice_function_verdict(data, "calculate")
            else:
                v, d = self._probe_verdict(resp.status, data)
            result(
                f"{prefix}-tool-choice-function",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_rtf, resp),
            )

        return results

    # =====================================================================
    # I. Reasoning + JSON (4 probe tests)
    #    OR validation: valid_json_content
    # =====================================================================

    async def _reasoning_json(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        format_variants = [
            (
                "structured-output",
                WEATHER_SCHEMA,
                "Give me the weather for San Francisco in fahrenheit.",
            ),
            (
                "response-format-json-object",
                {"type": "json_object"},
                "Return a JSON object with a greeting field.",
            ),
        ]

        for reasoning_on in (True, False):
            prefix = (
                "reasoning-enabled" if reasoning_on else "reasoning-disabled"
            )
            rkw = _reasoning_kwargs(enabled=reasoning_on)
            for fmt_label, fmt_value, prompt in format_variants:
                pl_rj = self._req(
                    model,
                    prompt,
                    response_format=fmt_value,
                    **rkw,
                )
                resp = await client.post_json(
                    pl_rj, timeout=self._fuzz_config.timeout * 2
                )
                validator = (
                    self._validate_weather_schema
                    if fmt_label == "structured-output"
                    else None
                )
                v, d = self._validate_json_content(
                    resp,
                    label=f"{prefix}+{fmt_label}",
                    validator=validator,
                )
                result(
                    f"{prefix}-{fmt_label}",
                    v,
                    status_code=resp.status,
                    detail=d,
                    **self._exchange_verbose(pl_rj, resp),
                )

        return results

    # =====================================================================
    # J. Verbosity (4 probe tests — OR: Verbosity{Low,Medium,High,Max})
    #    OR sends verbosity as top-level param. Matches OpenAI: low/medium/high
    #    are valid (accepted, 200; do not affect output for now), "max" is not
    #    an OpenAI verbosity level so the invalid enum must be rejected with 400.
    # =====================================================================

    async def _verbosity(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # OpenAI defines verbosity as Literal["low","medium","high"]. These are
        # accepted (200, positive completion tokens) but need not affect output.
        for level in ("low", "medium", "high"):
            # OR sends verbosity as top-level param (not chat_template_kwargs)
            pl_v = self._req(
                model,
                "Explain what a CPU is.",
                verbosity=level,
            )
            resp = await client.post_json(pl_v)
            data, _ = (
                parse_json(resp.body) if resp.status == 200 else (None, None)
            )
            v, d = self._probe_verdict(
                resp.status, data, check=_check_completion_tokens
            )
            result(
                f"verbosity-{level}",
                v,
                status_code=resp.status,
                detail=d,
                **self._exchange_verbose(pl_v, resp),
            )

        # "max" is not a valid OpenAI verbosity value; the invalid enum must be
        # rejected with 400 (a 200 would mean the bad value was silently accepted).
        pl_max = self._req(model, "Explain what a CPU is.", verbosity="max")
        resp = await client.post_json(pl_max)
        if resp.status == 400:
            v, d = Verdict.PASS, "Invalid verbosity rejected (400)"
        elif resp.status >= 500:
            v, d = Verdict.FAIL, f"Server error {resp.status}"
        else:
            v, d = (
                Verdict.FAIL,
                f"Invalid verbosity not rejected (got {resp.status})",
            )
        result(
            "verbosity-max",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_max, resp),
        )

        return results

    # =====================================================================
    # K. Misc (4 tests — OR: LargePrompt, DeveloperRole, AssistantPrefill, FastApply)
    # =====================================================================

    async def _misc(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # OR: LargePrompt — status_ok (200 or 400 accepted)
        ctx_size = config.model_config.large_input_tokens
        lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * (
            ctx_size // 8
        )
        pl_large = self._req(
            model,
            f"Summarize the following text in one sentence: {lorem}",
        )
        resp = await client.post_json(
            pl_large, timeout=self._fuzz_config.timeout * 2
        )
        if resp.status in (200, 400):
            v, d = (
                Verdict.PASS,
                f"Status {resp.status} for ~{ctx_size} token prompt",
            )
        elif resp.status >= 500:
            v, d = Verdict.FAIL, f"Server error {resp.status} on large prompt"
        else:
            v, d = Verdict.INTERESTING, f"Status {resp.status}"
        result(
            "large-prompt",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_large, resp),
        )

        # OR: DeveloperRole — must be accepted (normalized to system at the
        # OpenAI-compat route layer per OpenAI model spec compatibility).
        pl_dev = self._req(
            model,
            None,
            messages=[
                {
                    "role": "developer",
                    "content": "You are a helpful coding assistant. Always respond with code.",
                },
                {"role": "user", "content": "Write a hello world in Python."},
            ],
        )
        resp = await client.post_json(pl_dev)
        if resp.status == 200:
            v, d = (
                Verdict.PASS,
                "Status 200 (developer role normalized to system)",
            )
        elif resp.status >= 500:
            v, d = Verdict.FAIL, f"Server error {resp.status}"
        else:
            v, d = (
                Verdict.FAIL,
                f"Status {resp.status} (developer role should be accepted)",
            )
        result(
            "developer-role",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_dev, resp),
        )

        # OR: AssistantPrefill — content_non_empty
        pl_ap = self._req(
            model,
            None,
            messages=[
                {"role": "user", "content": "What is the meaning of life?"},
                {
                    "role": "assistant",
                    "content": "I'm not sure, but my best guess is",
                },
            ],
        )
        resp = await client.post_json(pl_ap)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._core_verdict(
            resp.status,
            data,
            check=lambda d: None if _get_content(d) else "Empty response",
        )
        result(
            "assistant-prefill",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_ap, resp),
        )

        # OR: FastApply — completion_tokens_positive
        pl_fa = self._req(
            model,
            "Apply this diff to the code: change 'hello' to 'world'.",
            chat_template_kwargs={"fast_apply": True},
        )
        resp = await client.post_json(pl_fa)
        data, _ = parse_json(resp.body) if resp.status == 200 else (None, None)
        v, d = self._probe_verdict(
            resp.status, data, check=_check_completion_tokens
        )
        result(
            "fast-apply",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_fa, resp),
        )

        return results

    # =====================================================================
    # L. Streaming Variants (7 bonus tests)
    # =====================================================================

    async def _streaming_variants(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # SSE chunk structure validation
        pl_bs = self._req(model, "Say hello.")
        resp = await client.post_streaming(pl_bs)
        if resp.status == 200 and resp.chunks:
            errors = []
            chunk_id = None
            first_has_role = False
            has_done = resp.chunks[-1] == "[DONE]" if resp.chunks else False

            for i, raw in enumerate(resp.chunks):
                if raw == "[DONE]":
                    continue
                cd, _ = parse_json(raw)
                if not cd:
                    errors.append(f"chunk[{i}] not valid JSON")
                    continue
                cid = cd.get("id")
                if chunk_id is None:
                    chunk_id = cid
                elif cid != chunk_id:
                    errors.append(
                        f"Inconsistent chunk IDs: {chunk_id} vs {cid}"
                    )
                    break
                if i == 0:
                    delta = _get_delta(cd)
                    if "role" in delta:
                        first_has_role = True

            if not first_has_role:
                errors.append("First chunk missing role in delta")
            if not has_done:
                errors.append("Stream missing [DONE] terminator")

            if errors:
                result(
                    "basic_streaming",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail="; ".join(errors[:3]),
                    **self._exchange_verbose(pl_bs, resp),
                )
            else:
                result(
                    "basic_streaming",
                    Verdict.PASS,
                    status_code=resp.status,
                    detail=f"Valid SSE: {len(resp.chunks)} chunks, consistent IDs, [DONE] present",
                    **self._exchange_verbose(pl_bs, resp),
                )
        else:
            result(
                "basic_streaming",
                Verdict.FAIL,
                status_code=resp.status,
                detail=resp.error or f"Status {resp.status}",
                **self._exchange_verbose(pl_bs, resp),
            )

        # Streaming tool calls — validate tool deltas, assembled args are JSON, no markers
        pl_tcs = self._with_tools(model, "What is 2 + 2?")
        resp = await client.post_streaming(pl_tcs)
        if resp.status == 200 and resp.chunks:
            has_tool = _stream_has_tool_delta(resp.chunks)
            if not has_tool:
                v, d = Verdict.INTERESTING, "No tool_calls in stream"
            else:
                errors = []
                # Check function name
                names = _get_stream_tool_names(resp.chunks)
                if not names:
                    errors.append(
                        "No function name seen in any tool_calls delta"
                    )
                elif names[0] != "calculate":
                    errors.append(f"Wrong function name: {names[0]!r}")
                # Check assembled arguments are valid JSON
                args_map = _assemble_stream_tool_args(resp.chunks)
                for idx, args_str in args_map.items():
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError) as e:
                        errors.append(
                            f"tool[{idx}] args not JSON: {e} ({args_str[:80]!r})"
                        )
                    marker_err = _check_no_markers(args_str)
                    if marker_err:
                        errors.append(marker_err)
                # Check finish_reason
                fr = _get_stream_finish_reason(resp.chunks)
                if fr != "tool_calls":
                    errors.append(
                        f"finish_reason={fr!r}, expected 'tool_calls'"
                    )
                if errors:
                    v, d = Verdict.FAIL, "; ".join(errors[:3])
                else:
                    v, d = (
                        Verdict.PASS,
                        "tool_calls streamed, valid JSON args, correct name",
                    )
        elif resp.status == 200:
            v, d = Verdict.INTERESTING, "Empty stream"
        else:
            v, d = self._core_verdict(resp.status)
        result(
            "tool_call_streaming",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_tcs, resp),
        )

        # Streaming multi-step tool conversation — assembled content must be non-empty
        pl_t5s = self._req(
            model,
            None,
            messages=TOOL_RESULT_MESSAGES,
            tools=[CALCULATE_TOOL],
        )
        resp = await client.post_streaming(pl_t5s)
        if resp.status == 200 and resp.chunks:
            assembled = _assemble_stream_content(resp.chunks)
            marker_err = _check_no_markers(assembled) if assembled else None
            if not assembled.strip():
                v, d = Verdict.FAIL, "Assembled stream content is empty"
            elif marker_err:
                v, d = Verdict.FAIL, marker_err
            else:
                v, d = (
                    Verdict.PASS,
                    f"Content streamed OK ({len(assembled)} chars)",
                )
        else:
            v, d = self._core_verdict(resp.status)
        result(
            "tool_call_step5_streaming",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_t5s, resp),
        )

        # Streaming JSON schema — validate JSON + schema match + no markers
        pl_sjs = self._req(
            model,
            "Give me the weather for San Francisco in fahrenheit.",
            response_format=WEATHER_SCHEMA,
        )
        resp = await client.post_streaming(pl_sjs)
        if resp.status == 200 and resp.chunks:
            assembled = _assemble_stream_content(resp.chunks)
            if not assembled:
                v, d = Verdict.FAIL, "No content assembled from stream"
            else:
                parsed, json_err = parse_json(assembled)
                marker_err = _check_no_markers(assembled)
                if json_err:
                    v, d = (
                        Verdict.FAIL,
                        f"Assembled stream not JSON: {json_err}",
                    )
                elif marker_err:
                    v, d = Verdict.FAIL, marker_err
                elif parsed is not None:
                    schema_err = self._validate_weather_schema(parsed)
                    if schema_err:
                        v, d = (
                            Verdict.FAIL,
                            f"JSON valid but schema mismatch: {schema_err}",
                        )
                    else:
                        v, d = (
                            Verdict.PASS,
                            "Assembled stream is valid JSON matching schema",
                        )
                else:
                    v, d = Verdict.FAIL, "JSON parsed to None unexpectedly"
        elif resp.status == 400:
            v, d = (
                Verdict.INTERESTING,
                "Server rejects streaming json_schema (400)",
            )
        else:
            v = Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            d = resp.error or f"Status {resp.status}"
        result(
            "structured_json_streaming",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_sjs, resp),
        )

        # Streaming tool_choice=required — validate args JSON + finish_reason
        pl_str = self._with_tools(model, tool_choice="required")
        resp = await client.post_streaming(pl_str)
        if resp.status == 200 and resp.chunks:
            has_tool = _stream_has_tool_delta(resp.chunks)
            if not has_tool:
                v, d = Verdict.FAIL, "No tool_calls in stream despite required"
            else:
                errors = []
                args_map = _assemble_stream_tool_args(resp.chunks)
                if len(args_map) > _RUNAWAY_CALL_FAIL_THRESHOLD:
                    errors.append(
                        f"Forced-tool runaway: {len(args_map)} streamed calls"
                    )
                for idx, args_str in args_map.items():
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError) as e:
                        errors.append(f"tool[{idx}] args not JSON: {e}")
                    marker_err = _check_no_markers(args_str)
                    if marker_err:
                        errors.append(marker_err)
                fr = _get_stream_finish_reason(resp.chunks)
                if fr != "tool_calls":
                    errors.append(
                        f"finish_reason={fr!r}, expected 'tool_calls'"
                    )
                if errors:
                    v, d = Verdict.FAIL, "; ".join(errors[:3])
                else:
                    v, d = (
                        Verdict.PASS,
                        "required tool_calls streamed, valid JSON args",
                    )
        else:
            v, d = self._core_verdict(resp.status)
        result(
            "streaming_tool_choice_required",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_str, resp),
        )

        # Streaming tool_choice=function — validate named function, args, finish_reason
        pl_stf = self._req(
            model,
            WEATHER_PROMPT,
            tools=[WEATHER_TOOL, CALCULATE_TOOL],
            tool_choice={"type": "function", "function": {"name": "calculate"}},
        )
        resp = await client.post_streaming(pl_stf)
        if resp.status == 200 and resp.chunks:
            has_tool = _stream_has_tool_delta(resp.chunks)
            if not has_tool:
                v, d = (
                    Verdict.FAIL,
                    "No tool_calls in stream for function choice",
                )
            else:
                errors = []
                names = _get_stream_tool_names(resp.chunks)
                if not names:
                    errors.append(
                        "No function name seen in any tool_calls delta"
                    )
                elif names[0] != "calculate":
                    errors.append(
                        f"Wrong function: {names[0]!r} (expected 'calculate')"
                    )
                args_map = _assemble_stream_tool_args(resp.chunks)
                for idx, args_str in args_map.items():
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError) as e:
                        errors.append(f"tool[{idx}] args not JSON: {e}")
                    marker_err = _check_no_markers(args_str)
                    if marker_err:
                        errors.append(marker_err)
                fr = _get_stream_finish_reason(resp.chunks)
                if fr not in ("tool_calls", "stop"):
                    errors.append(
                        f"finish_reason={fr!r}, expected 'tool_calls' or 'stop'"
                    )
                if errors:
                    v, d = Verdict.FAIL, "; ".join(errors[:3])
                else:
                    v, d = (
                        Verdict.PASS,
                        "function tool_calls streamed, valid JSON args",
                    )
        else:
            v, d = self._core_verdict(resp.status)
        result(
            "streaming_tool_choice_function",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_stf, resp),
        )

        # Streaming reasoning + tool call — reasoning enabled, auto tool choice
        pl_srt = self._with_tools(model, "What is 2 + 2?")
        pl_srt.update(_reasoning_kwargs(enabled=True))
        resp = await client.post_streaming(pl_srt)
        if resp.status == 200 and resp.chunks:
            has_tool = _stream_has_tool_delta(resp.chunks)
            if not has_tool:
                v, d = (
                    Verdict.INTERESTING,
                    "No tool_calls in reasoning+streaming",
                )
            else:
                errors = []
                args_map = _assemble_stream_tool_args(resp.chunks)
                for idx, args_str in args_map.items():
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError) as e:
                        errors.append(f"tool[{idx}] args not JSON: {e}")
                    marker_err = _check_no_markers(args_str)
                    if marker_err:
                        errors.append(marker_err)
                if errors:
                    v, d = Verdict.FAIL, "; ".join(errors[:3])
                else:
                    v, d = (
                        Verdict.PASS,
                        "reasoning+streaming tool_calls, valid JSON args",
                    )
        else:
            v, d = self._core_verdict(resp.status)
        result(
            "streaming_reasoning_tool_call",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_srt, resp),
        )

        # Streaming reasoning + tool_choice=required
        pl_srr = self._with_tools(model, tool_choice="required")
        pl_srr.update(_reasoning_kwargs(enabled=True))
        resp = await client.post_streaming(pl_srr)
        if resp.status == 200 and resp.chunks:
            has_tool = _stream_has_tool_delta(resp.chunks)
            if not has_tool:
                v, d = (
                    Verdict.FAIL,
                    "No tool_calls in reasoning+required stream",
                )
            else:
                errors = []
                args_map = _assemble_stream_tool_args(resp.chunks)
                for idx, args_str in args_map.items():
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError) as e:
                        errors.append(f"tool[{idx}] args not JSON: {e}")
                fr = _get_stream_finish_reason(resp.chunks)
                if fr != "tool_calls":
                    errors.append(
                        f"finish_reason={fr!r}, expected 'tool_calls'"
                    )
                if errors:
                    v, d = Verdict.FAIL, "; ".join(errors[:3])
                else:
                    v, d = (
                        Verdict.PASS,
                        "reasoning+required streamed, valid JSON args",
                    )
        else:
            v, d = self._core_verdict(resp.status)
        result(
            "streaming_reasoning_tool_choice_required",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_srr, resp),
        )

        # Streaming include_usage
        pl_siu = self._req(model, "Say hello.")
        pl_siu["stream_options"] = {"include_usage": True}
        resp = await client.post_streaming(pl_siu)
        if resp.status == 200 and resp.chunks:
            has_usage = False
            for raw in resp.chunks:
                if raw == "[DONE]":
                    continue
                cd, _ = parse_json(raw)
                if cd and cd.get("usage"):
                    has_usage = True
                    break
            v = Verdict.PASS if has_usage else Verdict.INTERESTING
            d = (
                "Usage in final chunk"
                if has_usage
                else "No usage in stream chunks"
            )
        else:
            v, d = self._core_verdict(resp.status)
        result(
            "streaming_include_usage",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_siu, resp),
        )

        # Streaming finish_reason
        pl_sfr = self._req(model, "Say hello.")
        resp = await client.post_streaming(pl_sfr)
        if resp.status == 200 and resp.chunks:
            last_finish = None
            for raw in reversed(resp.chunks):
                if raw == "[DONE]":
                    continue
                cd, _ = parse_json(raw)
                if cd:
                    choices = cd.get("choices", [])
                    if choices:
                        fr = choices[0].get("finish_reason")
                        if fr is not None:
                            last_finish = fr
                            break
            v = Verdict.PASS if last_finish else Verdict.FAIL
            d = (
                f"finish_reason={last_finish}"
                if last_finish
                else "No finish_reason in stream"
            )
        else:
            v, d = self._core_verdict(resp.status)
        result(
            "streaming_finish_reason",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_sfr, resp),
        )

        return results

    # =====================================================================
    # M. Extra Misc (2 bonus tests)
    # =====================================================================

    async def _extra_misc(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # Seed determinism
        seed_payload = self._req(model, "What is 2+2?", seed=42, temperature=0)
        resp_a = await client.post_json(seed_payload)
        resp_b = await client.post_json(seed_payload)
        if resp_a.status == 200 and resp_b.status == 200:
            da, _ = parse_json(resp_a.body)
            db, _ = parse_json(resp_b.body)
            if da and db:
                ca = _get_content(da)
                cb = _get_content(db)
                if ca == cb:
                    v, d = Verdict.PASS, "Deterministic output with same seed"
                else:
                    v, d = (
                        Verdict.INTERESTING,
                        "Different output with same seed (spec says best-effort)",
                    )
            else:
                v, d = Verdict.FAIL, "Invalid JSON"
        else:
            v, d = (
                Verdict.INTERESTING,
                f"Status a={resp_a.status}, b={resp_b.status}",
            )
        result(
            "seed_determinism",
            v,
            status_code=resp_a.status,
            detail=d,
            **self._exchange_verbose(seed_payload, resp_a, alt_resp=resp_b),
        )

        # n=2 multiple choices
        pl_n = self._req(model, "Say hello.", n=2)
        resp = await client.post_json(pl_n)
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                choices = data.get("choices", [])
                if len(choices) == 2:
                    v, d = Verdict.PASS, "2 choices returned"
                elif len(choices) == 1:
                    v, d = (
                        Verdict.INTERESTING,
                        "Only 1 choice returned (n=2 may not be supported)",
                    )
                else:
                    v, d = (
                        Verdict.INTERESTING,
                        f"{len(choices)} choices returned",
                    )
            else:
                v, d = Verdict.FAIL, "Invalid JSON"
        elif resp.status == 400:
            v, d = Verdict.INTERESTING, "Server rejects n=2 (400)"
        else:
            v = Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            d = f"Status {resp.status}"
        result(
            "n_multiple_choices",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(pl_n, resp),
        )

        return results
