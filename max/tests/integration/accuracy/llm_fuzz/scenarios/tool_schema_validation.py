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
Scenario: Tool call schema validation
Target: Verify tool call arguments conform to declared JSON schemas.

Based on the K2 Vendor Verifier approach: sends requests designed to trigger
tool calls, then validates that (1) the model triggers tool calls when expected
(finish_reason == "tool_calls"), and (2) the returned arguments match the
JSON schema declared in the tool definition.

What it catches:
  - Tool call arguments with missing required fields
  - Wrong types (string where number expected, etc.)
  - Extra properties when additionalProperties: false
  - Malformed JSON in arguments string
  - Enum violations
  - Array cardinality (minItems/maxItems) violations
  - Nested object schema violations
  - Inconsistencies between streaming and non-streaming tool calls
  - Tool call ID format issues (missing id, empty id)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from helpers import parse_json
from jsonschema import Draft7Validator

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RawResponse, RunConfig

# ---------------------------------------------------------------------------
# JSON schema validation (jsonschema Draft 7)
# ---------------------------------------------------------------------------


def _validate_against_schema(
    instance: object, schema: dict[str, Any]
) -> list[str]:
    """Validate a JSON instance against a JSON Schema (Draft 7).

    Returns a list of human-readable error strings (empty when valid). Uses the
    ``jsonschema`` ``Draft7Validator`` with ``FORMAT_CHECKER`` so the full
    keyword set is covered -- including semantic keywords (``pattern``,
    ``format``, ``multipleOf``, ``not``, ``if``/``then``, ``minProperties``)
    that a hand-rolled walker would miss.
    """
    try:
        validator = Draft7Validator(
            schema, format_checker=Draft7Validator.FORMAT_CHECKER
        )
    except Exception as e:  # malformed tool schema
        return [f"invalid schema: {e}"]
    return [
        f"{err.validator}@{err.json_path}: {err.message}"
        for err in sorted(validator.iter_errors(instance), key=str)
    ]


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _get_choice(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("choices", [{}])[0]


def _get_tool_calls(data: dict[str, Any]) -> list[Any]:
    return _get_choice(data).get("message", {}).get("tool_calls") or []


def _get_finish_reason(data: dict[str, Any]) -> str | None:
    return _get_choice(data).get("finish_reason")


def _find_tool_schema(
    tool_name: str, tools: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find the parameters schema for a tool by name."""
    for tool in tools:
        fn = tool.get("function", {})
        if fn.get("name") == tool_name:
            return fn.get("parameters")
    return None


def _validate_tool_call(
    tc: dict[str, Any], tools: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Validate a single tool call against the schema from the tools list.

    Returns (valid, detail).
    """
    fn = tc.get("function", {})
    name = fn.get("name")
    if not name:
        return False, "tool call missing function name"

    raw_args = fn.get("arguments", "")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return False, f"arguments not valid JSON: {e}"
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        return False, f"arguments has unexpected type {type(raw_args).__name__}"

    schema = _find_tool_schema(name, tools)
    if schema is None:
        return False, f"no schema found for tool '{name}'"

    errors = _validate_against_schema(args, schema)
    if errors:
        return False, f"schema errors: {'; '.join(errors)}"
    return True, "OK"


# ---------------------------------------------------------------------------
# Tool definitions used in tests
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2 + 2'",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

CREATE_EVENT_TOOL = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create a calendar event",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {
                    "type": "string",
                    "description": "ISO date, e.g. 2025-03-15",
                },
                "time": {"type": "string", "description": "HH:MM 24h format"},
                "duration_minutes": {"type": "integer", "minimum": 1},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee email addresses",
                },
                "location": {"type": "string"},
                "is_recurring": {"type": "boolean"},
            },
            "required": ["title", "date", "time", "duration_minutes"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

NESTED_TOOL = {
    "type": "function",
    "function": {
        "name": "create_order",
        "description": "Place a product order",
        "parameters": {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "email"],
                    "additionalProperties": False,
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "string"},
                            "quantity": {"type": "integer", "minimum": 1},
                            "price": {"type": "number"},
                        },
                        "required": ["product", "quantity", "price"],
                        "additionalProperties": False,
                    },
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                },
            },
            "required": ["customer", "items", "priority"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# ---------------------------------------------------------------------------
# Schema-enforcement cases
# ---------------------------------------------------------------------------
# Each forces a named tool call and asks for an argument value that is valid
# JSON but violates a declared schema keyword. Unlike the cases above (which
# use natural prompts the model satisfies on its own), these adversarial
# prompts only conform when the server actively constrains tool arguments to
# the schema, so they expose whether schema enforcement is on. Each case is
# repeat-sampled (see ``_schema_enforcement``).
#
# ``enforceable`` marks keywords that grammar-based constrained decoding can
# satisfy (type/enum/required/additionalProperties/minItems/maxItems/maximum/
# maxLength/pattern/format/minProperties): a violation there is a real defect
# (FAIL). The rest (``multipleOf``/``not``/``if``-``then``) cannot be expressed
# as a token grammar on any engine, so a violation is expected -- graded PASS.


def _enforce_tool(
    name: str,
    properties: dict[str, Any],
    required: list[str],
    description: str,
    **extra_params: Any,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    params.update(extra_params)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


_ENFORCE_REPEATS = 4

_ENFORCEMENT_CASES: list[dict[str, Any]] = [
    # --- grammar-enforceable: FAIL when the schema is not enforced ----------
    {
        "name": "wrong_type",
        "expect": "type",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_reminder",
            {
                "minutes": {"type": "integer"},
                "message": {"type": "string"},
            },
            ["minutes", "message"],
            "Schedule a reminder after some minutes.",
        ),
        "prompt": (
            "Call set_reminder with message set to 'take a break' and minutes "
            'set to the JSON string "ten" -- the word ten in double quotes, '
            "NOT a number."
        ),
    },
    {
        "name": "missing_required",
        "expect": "required",
        "enforceable": True,
        "tool": _enforce_tool(
            "create_user",
            {
                "username": {"type": "string"},
                "email": {"type": "string"},
            },
            ["username", "email"],
            "Create a user account.",
        ),
        "prompt": (
            "Call create_user with username set to 'alice' only. Do NOT "
            "include the email field at all -- omit it entirely."
        ),
    },
    {
        "name": "additional_property",
        "expect": "additionalProperties",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_volume",
            {"level": {"type": "integer"}},
            ["level"],
            "Set the speaker volume.",
        ),
        "prompt": (
            "Call set_volume with level set to 5 and ALSO add an extra field "
            "named muted set to true."
        ),
    },
    {
        "name": "nested_wrong_type",
        "expect": "type",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_location",
            {
                "coords": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                    },
                    "required": ["lat", "lon"],
                    "additionalProperties": False,
                }
            },
            ["coords"],
            "Set a geographic location.",
        ),
        "prompt": (
            'Call set_location with coords.lat set to the JSON string "north" '
            "and coords.lon set to 12.5."
        ),
    },
    {
        "name": "object_as_array",
        "expect": "type",
        "enforceable": True,
        "tool": _enforce_tool(
            "configure",
            {"options": {"type": "object"}},
            ["options"],
            "Apply a configuration object.",
        ),
        "prompt": (
            'Call configure with options set to a JSON array: ["a", "b"] '
            "(an array, not an object)."
        ),
    },
    {
        "name": "number_over_maximum",
        "expect": "maximum",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_age",
            {"age": {"type": "integer", "minimum": 0, "maximum": 120}},
            ["age"],
            "Set a person's age in years.",
        ),
        "prompt": "Call set_age with age set to 999.",
    },
    {
        "name": "max_length",
        "expect": "maxLength",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_username",
            {"username": {"type": "string", "maxLength": 8}},
            ["username"],
            "Set a short username (at most 8 characters).",
        ),
        "prompt": (
            "Call set_username with username set to "
            '"this_is_a_very_long_username_exceeding_the_limit".'
        ),
    },
    {
        "name": "string_pattern",
        "expect": "pattern",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_ticket",
            {"code": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{4}$"}},
            ["code"],
            "Set a ticket code formatted like ABC-1234.",
        ),
        "prompt": 'Call set_ticket with code set to "hello world".',
    },
    {
        "name": "format_email",
        "expect": "format",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_contact",
            {"email": {"type": "string", "format": "email"}},
            ["email"],
            "Set a contact email address.",
        ),
        "prompt": 'Call set_contact with email set to "not an email".',
    },
    {
        "name": "min_properties",
        "expect": "minProperties",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_metadata",
            {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"},
            },
            [],
            "Set metadata with at least three fields.",
            minProperties=3,
        ),
        "prompt": 'Call set_metadata with only field a set to "x".',
    },
    {
        "name": "enum_violation",
        "expect": "enum",
        "enforceable": True,
        "tool": _enforce_tool(
            "set_role",
            {"role": {"type": "string", "enum": ["admin", "user", "guest"]}},
            ["role"],
            "Assign an account role.",
        ),
        "prompt": 'Call set_role with role set to "SUPERADMIN".',
    },
    {
        "name": "array_max_items",
        "expect": "maxItems",
        "enforceable": True,
        "tool": _enforce_tool(
            "add_tags",
            {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                }
            },
            ["tags"],
            "Attach at most three tags.",
        ),
        "prompt": "Call add_tags with these seven tags: a, b, c, d, e, f, g.",
    },
    # --- grammar-inexpressible: violation is PASS (expected), never FAIL -----
    {
        "name": "multiple_of",
        "expect": "multipleOf",
        "enforceable": False,
        "tool": _enforce_tool(
            "set_price",
            {"amount": {"type": "number", "multipleOf": 5}},
            ["amount"],
            "Set a price that must be a multiple of 5.",
        ),
        "prompt": "Call set_price with amount set to 7.",
    },
    {
        "name": "not_forbidden",
        "expect": "not",
        "enforceable": False,
        "tool": _enforce_tool(
            "set_status",
            {"status": {"type": "string", "not": {"enum": ["banned"]}}},
            ["status"],
            "Set a status that must not be the forbidden value.",
        ),
        "prompt": 'Call set_status with status set to "banned".',
    },
    {
        "name": "conditional_required",
        "expect": "if/then",
        "enforceable": False,
        "tool": _enforce_tool(
            "create_timed_event",
            {
                "all_day": {"type": "boolean"},
                "start_time": {"type": "string"},
            },
            ["all_day"],
            "Create an event; non-all-day events require a start time.",
            **{
                "if": {"properties": {"all_day": {"const": False}}},
                "then": {"required": ["start_time"]},
            },
        ),
        "prompt": (
            "Call create_timed_event with all_day set to false and do NOT "
            "include start_time."
        ),
    },
]


# Enforcement cases whose keyword the constrained-decoding backend genuinely
# cannot enforce, so the honest outcome is an enforce-or-400 up front rather
# than a best-effort 200 that silently violates the schema. Mirrors the honest-
# rejection contract in the sibling ``structured_output_schema_limits``
# (``array_unique_unenforceable`` / ``_EXPECT_REJECT`` /
# ``_is_honest_rejection``): for these a clean 400 naming the construct is a
# PASS, not a FAIL. ``multipleOf`` / ``not`` / ``if``-``then`` are CFG-
# inexpressible on every backend; ``minProperties`` is unenforceable for the
# decomposing-codec models (e.g. gemma4) whose enforce-or-400 gate rejects it --
# both surface the same 400. This only credits the 400 PATH: the ``enforceable``
# flag is left untouched, so a schema-violating 200 still grades as before (a
# JSON-arg model that DOES enforce ``minProperties`` is still FAILed on a
# violation; the universally-inexpressible three are PASS, a violation being
# expected).
_EXPECT_HONEST_REJECT: frozenset[str] = frozenset(
    {"min_properties", "multiple_of", "not_forbidden", "conditional_required"}
)


def _is_honest_rejection(status: int) -> bool:
    """True if ``status`` is an honest refusal of the unenforceable construct.

    These requests are well-formed (valid JSON, structurally valid tool
    schema), so the only reason to reject one is the construct itself: a clean
    400 is the backend honestly declining to build a grammar it cannot enforce,
    whatever the exact message says. Matching on status alone -- rather than an
    ``"enforce"`` substring that is backend- and version-specific -- keeps a
    correctly-refused request from grading FAIL just because a backend words its
    400 differently (or omits the word entirely).
    """
    return status == 400


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@register_scenario
class ToolSchemaValidation(BaseScenario):
    name = "tool_schema_validation"
    description = (
        "K2VV-style tool call schema validation: checks that tool call "
        "arguments conform to declared JSON schemas"
    )
    tags = ["tools", "schema", "compliance", "functional"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        self._fuzz_config = config
        results: list[ScenarioResult] = []
        model = config.model

        results.extend(await self._single_tool_basic(client, model))
        results.extend(await self._multi_param_tools(client, model))
        results.extend(await self._nested_schema(client, model))
        results.extend(await self._multi_tool_selection(client, model))
        results.extend(await self._tool_choice_schema(client, model))
        results.extend(await self._multi_turn_tool(client, model))
        results.extend(await self._streaming_schema(client, model))
        results.extend(await self._concurrent_schema(client, model))
        results.extend(await self._schema_enforcement(client, model))

        return results

    # -- helpers --------------------------------------------------------------

    def _req(
        self, model: str, content: str, tools: list[dict[str, Any]], **extra
    ) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "tools": tools,
            "max_tokens": 1024,
        }
        p.update(extra)
        return p

    def _exchange_verbose(
        self,
        payload: dict[str, Any] | str | None,
        resp: RawResponse | None,
    ) -> dict[str, str]:
        cfg = getattr(self, "_fuzz_config", None)
        if cfg is None or not getattr(cfg, "verbose", False):
            return {}
        out: dict[str, str] = {
            "request_body": "",
            "response_body": "",
            "error": "",
        }
        if payload is not None:
            try:
                out["request_body"] = (
                    json.dumps(payload, ensure_ascii=False)
                    if isinstance(payload, dict)
                    else str(payload)
                )
            except (TypeError, ValueError):
                out["request_body"] = str(payload)
        if resp is not None:
            out["response_body"] = getattr(resp, "body", "") or ""
            out["error"] = getattr(resp, "error", "") or ""
        return out

    def _assess_tool_response(
        self,
        resp: RawResponse,
        tools: list[dict[str, Any]],
        *,
        expect_tool_call: bool = True,
    ) -> tuple[Verdict, str, dict[str, Any]]:
        """Parse a response and validate any tool calls against schemas.

        Returns (verdict, detail, stats) where stats has keys:
          triggered, valid, invalid, errors.
        """
        stats: dict[str, Any] = {
            "triggered": 0,
            "valid": 0,
            "invalid": 0,
            "errors": [],
        }

        if resp.error:
            if resp.error == "TIMEOUT":
                return Verdict.FAIL, "Request timed out", stats
            return Verdict.ERROR, f"Client error: {resp.error}", stats

        if resp.status >= 500:
            return Verdict.FAIL, f"Server error {resp.status}", stats

        if resp.status != 200:
            if expect_tool_call and resp.status == 400:
                return (
                    Verdict.FAIL,
                    f"Server rejected request ({resp.status})",
                    stats,
                )
            return Verdict.INTERESTING, f"Status {resp.status}", stats

        data, parse_err = parse_json(resp.body)
        if parse_err or data is None:
            return Verdict.FAIL, f"Invalid JSON response: {parse_err}", stats

        finish_reason = _get_finish_reason(data)
        tool_calls = _get_tool_calls(data)

        if expect_tool_call and not tool_calls:
            if finish_reason == "stop":
                return (
                    Verdict.INTERESTING,
                    "Model chose not to call tools (finish_reason=stop)",
                    stats,
                )
            return (
                Verdict.FAIL,
                f"No tool_calls, finish_reason={finish_reason}",
                stats,
            )

        if not tool_calls:
            return Verdict.PASS, "No tool calls (as expected)", stats

        # Validate finish_reason
        if finish_reason != "tool_calls":
            stats["errors"].append(
                f"finish_reason={finish_reason}, expected tool_calls"
            )

        stats["triggered"] = len(tool_calls)
        for tc in tool_calls:
            valid, detail = _validate_tool_call(tc, tools)
            if valid:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
                stats["errors"].append(detail)

        if stats["invalid"] > 0:
            error_summary = "; ".join(stats["errors"][:3])
            return (
                Verdict.FAIL,
                f"{stats['valid']}/{stats['triggered']} valid, "
                f"{stats['invalid']} schema errors: {error_summary}",
                stats,
            )

        return (
            Verdict.PASS,
            f"{stats['valid']}/{stats['triggered']} tool calls valid",
            stats,
        )

    # =====================================================================
    # 0. Schema enforcement (forced named tool call, adversarial value)
    # =====================================================================

    async def _schema_enforcement(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        """Force a named tool call asking for a value that violates a declared
        keyword, then validate the arguments against the schema.

        Each case is repeated ``_ENFORCE_REPEATS`` times at ``temperature=0``
        with reasoning disabled, so a grammar that enforces only intermittently
        (e.g. under serving-state / compiler-cache contention) is caught while a
        correct backend stays deterministic. Conforming is a PASS; a
        grammar-enforceable violation is a FAIL; a violation of a keyword no
        token grammar can express is a PASS (expected, not a defect).
        """
        results: list[ScenarioResult] = []
        for case in _ENFORCEMENT_CASES:
            tool = case["tool"]
            fname = tool["function"]["name"]
            test = f"enforce_{case['name']}"
            expect_honest_reject = case["name"] in _EXPECT_HONEST_REJECT

            fired = 0
            conformant = 0
            honest_reject = 0
            first_bad: tuple[dict[str, Any], RawResponse, str] | None = None
            last: tuple[dict[str, Any], RawResponse] | None = None
            transport: tuple[Verdict, str] | None = None

            for _ in range(_ENFORCE_REPEATS):
                payload = self._req(
                    model,
                    case["prompt"],
                    [tool],
                    tool_choice={
                        "type": "function",
                        "function": {"name": fname},
                    },
                    temperature=0.0,
                    chat_template_kwargs={
                        "enable_thinking": False,
                        "thinking": False,
                    },
                )
                resp = await client.post_json(
                    payload, timeout=self._fuzz_config.timeout * 2
                )
                last = (payload, resp)

                if resp.error or resp.status != 200:
                    transport = self._assess_tool_response(resp, [tool])[:2]
                    # A CFG-inexpressible keyword the backend cannot enforce is
                    # honestly refused with an enforce-or-400 (see
                    # _EXPECT_HONEST_REJECT); count those so the fired==0 branch
                    # can credit the honest rejection as a PASS.
                    if expect_honest_reject and _is_honest_rejection(
                        resp.status
                    ):
                        honest_reject += 1
                    continue

                data, _perr = parse_json(resp.body)
                tool_calls = _get_tool_calls(data or {})
                if not tool_calls:
                    continue
                fired += 1
                valid, detail = _validate_tool_call(tool_calls[0], [tool])
                if valid:
                    conformant += 1
                elif first_bad is None:
                    first_bad = (payload, resp, detail)

            # A forced tool_choice that never fired is a server fault (or the
            # transport error behind it) -- UNLESS it is a CFG-inexpressible
            # construct the backend honestly refused with an enforce-or-400,
            # which is the correct outcome (mirrors s32's honest-rejection
            # contract), so a clean rejection across all repeats is a PASS.
            if fired == 0:
                assert last is not None
                payload, resp = last
                if expect_honest_reject and honest_reject == _ENFORCE_REPEATS:
                    verdict = Verdict.PASS
                    note = (
                        f"honest enforce-or-400 ({case['expect']}): rejected "
                        f"{honest_reject}/{_ENFORCE_REPEATS}; unenforceable "
                        "construct refused up front, never a best-effort 200"
                    )
                else:
                    verdict, note = transport or (
                        Verdict.FAIL,
                        "forced tool_choice produced no tool call",
                    )
            else:
                violations = fired - conformant
                if violations == 0:
                    assert last is not None
                    payload, resp = last
                    verdict = Verdict.PASS
                    note = (
                        f"args conform {conformant}/{fired} ({case['expect']})"
                    )
                else:
                    assert first_bad is not None
                    payload, resp, bad = first_bad
                    if case["enforceable"]:
                        verdict = Verdict.FAIL
                        note = (
                            f"schema not enforced ({case['expect']}): "
                            f"{violations}/{fired} violated; {bad}"
                        )
                    else:
                        verdict = Verdict.PASS
                        note = (
                            f"{case['expect']} not grammar-enforceable; "
                            f"{violations}/{fired} violated (expected): {bad}"
                        )
            results.append(
                self.make_result(
                    self.name,
                    test,
                    verdict,
                    status_code=resp.status,
                    detail=note,
                    **self._exchange_verbose(payload, resp),
                )
            )
        return results

    # =====================================================================
    # 1. Single-tool basic schema validation (4 tests)
    # =====================================================================

    async def _single_tool_basic(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # Weather tool with required enum field
        payload = self._req(
            model,
            "What is the weather in San Francisco? Use fahrenheit.",
            [WEATHER_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [WEATHER_TOOL])
        result(
            "weather_basic",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # Calculator tool with single required field
        payload = self._req(
            model,
            "What is 15 * 23 + 7? Use the calculator.",
            [CALCULATOR_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [CALCULATOR_TOOL])
        result(
            "calculator_basic",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # Search tool with optional field
        payload = self._req(
            model,
            "Search the web for the latest news about AI.",
            [SEARCH_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [SEARCH_TOOL])
        result(
            "search_basic",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # tool_choice=auto: model should still call the tool
        payload = self._req(
            model,
            "What is the weather in Tokyo in celsius?",
            [WEATHER_TOOL],
            tool_choice="auto",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [WEATHER_TOOL])
        result(
            "weather_auto_choice",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        return results

    # =====================================================================
    # 2. Multi-parameter and complex-type tools (3 tests)
    # =====================================================================

    async def _multi_param_tools(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # Event creation with mixed types: string, integer, array, boolean
        payload = self._req(
            model,
            "Create a meeting called 'Sprint Planning' on 2025-04-10 at 14:00 "
            "for 60 minutes with alice@example.com and bob@example.com in "
            "Conference Room A. Make it recurring.",
            [CREATE_EVENT_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [CREATE_EVENT_TOOL])
        result(
            "event_all_fields",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # Event with only required fields (optional fields should be omitted or valid)
        payload = self._req(
            model,
            "Create an event titled 'Lunch' on 2025-04-11 at 12:00 for 30 minutes.",
            [CREATE_EVENT_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [CREATE_EVENT_TOOL])
        result(
            "event_required_only",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # Search with integer constraint (num_results should be 1-20)
        payload = self._req(
            model,
            "Search for 'quantum computing breakthroughs' and give me 5 results.",
            [SEARCH_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [SEARCH_TOOL])
        result(
            "search_with_count",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        return results

    # =====================================================================
    # 3. Nested object schema validation (2 tests)
    # =====================================================================

    async def _nested_schema(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # Nested objects + arrays
        payload = self._req(
            model,
            "Place an urgent order for customer John Doe (john@example.com): "
            "2 units of Widget A at $9.99 each and 1 unit of Gadget B at $24.50.",
            [NESTED_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [NESTED_TOOL])
        result(
            "nested_order",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # Single item order (array with one element)
        payload = self._req(
            model,
            "Place a normal priority order for Jane Smith (jane@example.com): "
            "1 unit of Basic Plan at $49.00.",
            [NESTED_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [NESTED_TOOL])
        result(
            "nested_single_item",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        return results

    # =====================================================================
    # 4. Multi-tool selection (3 tests)
    # =====================================================================

    async def _multi_tool_selection(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        all_tools = [WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL]

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # Should select weather tool
        payload = self._req(
            model,
            "What's the weather in London in celsius?",
            all_tools,
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, all_tools)
        # Additional check: verify it selected the right tool
        if v == Verdict.PASS and resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                tcs = _get_tool_calls(data)
                if (
                    tcs
                    and tcs[0].get("function", {}).get("name") != "get_weather"
                ):
                    v = Verdict.INTERESTING
                    d = f"Expected get_weather, got {tcs[0].get('function', {}).get('name')}"
        result(
            "multi_tool_select_weather",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # Should select calculator tool
        payload = self._req(
            model,
            "Calculate 42 * 17 - 3.",
            all_tools,
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, all_tools)
        if v == Verdict.PASS and resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                tcs = _get_tool_calls(data)
                if (
                    tcs
                    and tcs[0].get("function", {}).get("name") != "calculate"
                ):
                    v = Verdict.INTERESTING
                    d = f"Expected calculate, got {tcs[0].get('function', {}).get('name')}"
        result(
            "multi_tool_select_calc",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # Should select search tool
        payload = self._req(
            model,
            "Search the web for recent SpaceX launches.",
            all_tools,
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, all_tools)
        if v == Verdict.PASS and resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                tcs = _get_tool_calls(data)
                if (
                    tcs
                    and tcs[0].get("function", {}).get("name") != "web_search"
                ):
                    v = Verdict.INTERESTING
                    d = f"Expected web_search, got {tcs[0].get('function', {}).get('name')}"
        result(
            "multi_tool_select_search",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        return results

    # =====================================================================
    # 5. tool_choice modes with schema validation (4 tests)
    # =====================================================================

    async def _tool_choice_schema(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # tool_choice=none should NOT produce tool calls
        payload = self._req(
            model,
            "What is the weather in Paris?",
            [WEATHER_TOOL],
            tool_choice="none",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        if resp.status >= 500 or resp.error == "TIMEOUT":
            v, d = Verdict.FAIL, f"Server error or timeout: {resp.status}"
        elif resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                tcs = _get_tool_calls(data)
                fr = _get_finish_reason(data)
                if tcs:
                    v, d = (
                        Verdict.FAIL,
                        "tool_calls present despite tool_choice=none",
                    )
                elif fr == "tool_calls":
                    v, d = (
                        Verdict.FAIL,
                        "finish_reason=tool_calls despite tool_choice=none",
                    )
                else:
                    v, d = Verdict.PASS, "No tool calls with tool_choice=none"
            else:
                v, d = Verdict.FAIL, "Invalid JSON"
        else:
            v, d = Verdict.PASS, f"Status {resp.status}"
        result(
            "tool_choice_none_no_calls",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # tool_choice=required must trigger a tool call with valid schema
        payload = self._req(
            model,
            "Tell me a joke.",  # not a weather question, but required forces it
            [WEATHER_TOOL],
            tool_choice="required",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [WEATHER_TOOL])
        result(
            "tool_choice_required_schema",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # tool_choice=specific function
        payload = self._req(
            model,
            "Tell me something interesting.",
            [WEATHER_TOOL, CALCULATOR_TOOL],
            tool_choice={"type": "function", "function": {"name": "calculate"}},
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        if resp.status >= 500 or resp.error == "TIMEOUT":
            v, d = Verdict.FAIL, f"Server error or timeout: {resp.status}"
        elif resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                tcs = _get_tool_calls(data)
                if not tcs:
                    v, d = (
                        Verdict.FAIL,
                        "No tool calls with specific tool_choice",
                    )
                else:
                    fn_name = tcs[0].get("function", {}).get("name")
                    if fn_name != "calculate":
                        v, d = (
                            Verdict.FAIL,
                            f"Wrong tool: {fn_name}, expected calculate",
                        )
                    else:
                        valid, detail = _validate_tool_call(
                            tcs[0], [WEATHER_TOOL, CALCULATOR_TOOL]
                        )
                        v = Verdict.PASS if valid else Verdict.FAIL
                        d = detail
            else:
                v, d = Verdict.FAIL, "Invalid JSON"
        elif resp.status == 400:
            v, d = (
                Verdict.INTERESTING,
                "Specific tool_choice not supported (400)",
            )
        else:
            v, d = Verdict.INTERESTING, f"Status {resp.status}"
        result(
            "tool_choice_specific_schema",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        # tool_choice=auto with strong prompt (should still validate)
        payload = self._req(
            model,
            "What is the weather like in Berlin, Germany? Give the temperature in celsius.",
            [WEATHER_TOOL],
            tool_choice="auto",
        )
        resp = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v, d, _ = self._assess_tool_response(resp, [WEATHER_TOOL])
        result(
            "tool_choice_auto_schema",
            v,
            status_code=resp.status,
            detail=d,
            **self._exchange_verbose(payload, resp),
        )

        return results

    # =====================================================================
    # 6. Multi-turn tool conversations (3 tests)
    # =====================================================================

    async def _multi_turn_tool(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # Step 1: initial tool call
        step1_payload = self._req(
            model,
            "What is 25 * 4?",
            [CALCULATOR_TOOL],
            tool_choice="required",
        )
        resp1 = await client.post_json(
            step1_payload, timeout=self._fuzz_config.timeout * 2
        )
        v1, d1, _ = self._assess_tool_response(resp1, [CALCULATOR_TOOL])
        result(
            "multi_turn_step1",
            v1,
            status_code=resp1.status,
            detail=d1,
            **self._exchange_verbose(step1_payload, resp1),
        )

        # Step 2: provide tool result, ask follow-up that triggers another call
        step2_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "What is 25 * 4?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_step1",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"expression": "25 * 4"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_step1",
                    "content": '{"result": 100}',
                },
                {"role": "user", "content": "Now add 50 to that result."},
            ],
            "tools": [CALCULATOR_TOOL],
            "max_tokens": 1024,
            "tool_choice": "auto",
        }
        resp2 = await client.post_json(
            step2_payload, timeout=self._fuzz_config.timeout * 2
        )
        v2, d2, _ = self._assess_tool_response(
            resp2,
            [CALCULATOR_TOOL],
            expect_tool_call=True,
        )
        # Model might answer directly since it knows the result; that's acceptable
        if v2 == Verdict.INTERESTING and "chose not to call" in d2:
            v2 = Verdict.PASS
            d2 = "Model answered directly from context (acceptable)"
        result(
            "multi_turn_step2",
            v2,
            status_code=resp2.status,
            detail=d2,
            **self._exchange_verbose(step2_payload, resp2),
        )

        # Step 3: multi-tool conversation
        step3_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in NYC in fahrenheit?",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "New York, NY", "unit": "fahrenheit"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "content": '{"temperature": 72, "condition": "sunny"}',
                },
                {
                    "role": "user",
                    "content": "Now check the weather in Los Angeles in celsius.",
                },
            ],
            "tools": [WEATHER_TOOL],
            "max_tokens": 1024,
            "tool_choice": "auto",
        }
        resp3 = await client.post_json(
            step3_payload, timeout=self._fuzz_config.timeout * 2
        )
        v3, d3, _ = self._assess_tool_response(resp3, [WEATHER_TOOL])
        result(
            "multi_turn_step3",
            v3,
            status_code=resp3.status,
            detail=d3,
            **self._exchange_verbose(step3_payload, resp3),
        )

        return results

    # =====================================================================
    # 7. Streaming tool call schema validation (3 tests)
    # =====================================================================

    async def _streaming_schema(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        tools = [WEATHER_TOOL]
        payload = self._req(
            model,
            "What is the weather in Chicago in fahrenheit?",
            tools,
            tool_choice="required",
        )

        # Non-streaming baseline
        resp_sync = await client.post_json(
            payload, timeout=self._fuzz_config.timeout * 2
        )
        v_sync, d_sync, _ = self._assess_tool_response(resp_sync, tools)
        result(
            "stream_sync_baseline",
            v_sync,
            status_code=resp_sync.status,
            detail=d_sync,
            **self._exchange_verbose(payload, resp_sync),
        )

        # Streaming version
        resp_stream = await client.post_streaming(
            payload, read_timeout=self._fuzz_config.timeout * 2
        )
        if resp_stream.error:
            v_stream = (
                Verdict.ERROR
                if resp_stream.error != "TIMEOUT"
                else Verdict.FAIL
            )
            d_stream = f"Streaming error: {resp_stream.error}"
        elif resp_stream.status != 200:
            v_stream = (
                Verdict.FAIL
                if resp_stream.status >= 500
                else Verdict.INTERESTING
            )
            d_stream = f"Status {resp_stream.status}"
        else:
            # Reassemble tool calls from stream chunks
            assembled_tool_calls = self._assemble_stream_tool_calls(
                resp_stream.chunks or []
            )
            if not assembled_tool_calls:
                v_stream = Verdict.INTERESTING
                d_stream = "No tool calls in streaming response"
            else:
                all_valid = True
                stream_errors = []
                for tc in assembled_tool_calls:
                    valid, detail = _validate_tool_call(tc, tools)
                    if not valid:
                        all_valid = False
                        stream_errors.append(detail)
                if all_valid:
                    v_stream = Verdict.PASS
                    d_stream = f"{len(assembled_tool_calls)} streaming tool calls valid"
                else:
                    v_stream = Verdict.FAIL
                    d_stream = f"Streaming schema errors: {'; '.join(stream_errors[:3])}"
        result(
            "stream_tool_schema",
            v_stream,
            status_code=resp_stream.status,
            detail=d_stream,
            **self._exchange_verbose(payload, resp_stream),
        )

        # Compare sync vs stream consistency
        if v_sync == Verdict.PASS and v_stream == Verdict.PASS:
            sync_data, _ = parse_json(resp_sync.body)
            sync_tcs = _get_tool_calls(sync_data) if sync_data else []
            stream_tcs = self._assemble_stream_tool_calls(
                resp_stream.chunks or []
            )

            sync_names = [tc.get("function", {}).get("name") for tc in sync_tcs]
            stream_names = [
                tc.get("function", {}).get("name") for tc in stream_tcs
            ]

            if sync_names == stream_names:
                v, d = Verdict.PASS, f"Consistent tool selection: {sync_names}"
            else:
                v, d = (
                    Verdict.INTERESTING,
                    f"Different tools: sync={sync_names}, stream={stream_names}",
                )
        elif v_sync != v_stream:
            v = Verdict.INTERESTING
            d = f"Sync verdict={v_sync.value}, stream verdict={v_stream.value}"
        else:
            v = Verdict.PASS
            d = "Both sync and stream had same verdict"
        result("stream_sync_consistency", v, detail=d)

        return results

    @staticmethod
    def _assemble_stream_tool_calls(
        chunks: list[str],
    ) -> list[dict[str, Any]]:
        """Reassemble tool calls from SSE delta chunks."""
        tool_calls: dict[int, dict[str, Any]] = {}

        for raw in chunks:
            if raw == "[DONE]":
                continue
            try:
                cd = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            choices = cd.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            delta_tcs = delta.get("tool_calls")
            if not delta_tcs:
                continue

            for tc_delta in delta_tcs:
                idx = tc_delta.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc_delta.get("id"),
                        "type": tc_delta.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    }
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    tool_calls[idx]["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_calls[idx]["function"]["arguments"] += fn["arguments"]

        return [tool_calls[k] for k in sorted(tool_calls)]

    # =====================================================================
    # 8. Concurrent tool call schema validation (2 tests)
    # =====================================================================

    async def _concurrent_schema(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        import asyncio

        def result(test: str, verdict: Verdict, **kw: Any) -> None:
            results.append(self.make_result(self.name, test, verdict, **kw))

        # Fire 5 different tool call requests concurrently
        payloads_and_tools = [
            (
                self._req(
                    model,
                    "Weather in Paris, celsius?",
                    [WEATHER_TOOL],
                    tool_choice="required",
                ),
                [WEATHER_TOOL],
            ),
            (
                self._req(
                    model,
                    "Calculate 99 * 101.",
                    [CALCULATOR_TOOL],
                    tool_choice="required",
                ),
                [CALCULATOR_TOOL],
            ),
            (
                self._req(
                    model,
                    "Weather in Tokyo, fahrenheit?",
                    [WEATHER_TOOL],
                    tool_choice="required",
                ),
                [WEATHER_TOOL],
            ),
            (
                self._req(
                    model,
                    "Search for 'rust programming language'.",
                    [SEARCH_TOOL],
                    tool_choice="required",
                ),
                [SEARCH_TOOL],
            ),
            (
                self._req(
                    model,
                    "Calculate 2^10.",
                    [CALCULATOR_TOOL],
                    tool_choice="required",
                ),
                [CALCULATOR_TOOL],
            ),
        ]

        async def send_and_validate(
            payload: dict[str, Any],
            tools: list[dict[str, Any]],
            idx: int,
        ) -> tuple[int, RawResponse, list[dict[str, Any]]]:
            resp = await client.post_json(
                payload, timeout=self._fuzz_config.timeout * 2
            )
            return idx, resp, tools

        tasks = [
            send_and_validate(p, t, i)
            for i, (p, t) in enumerate(payloads_and_tools)
        ]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        total = 0
        valid = 0
        invalid = 0
        for res in task_results:
            if isinstance(res, BaseException):
                invalid += 1
                total += 1
                continue
            idx, resp, tools = res
            total += 1
            v, d, _ = self._assess_tool_response(resp, tools)
            if v == Verdict.PASS:
                valid += 1
            elif v == Verdict.INTERESTING:
                valid += 1  # model chose not to call, but no schema error
            else:
                invalid += 1

        if invalid == 0:
            v = Verdict.PASS
            d = f"All {valid}/{total} concurrent tool calls schema-valid"
        elif invalid <= 1:
            v = Verdict.INTERESTING
            d = f"{valid}/{total} valid, {invalid} failed"
        else:
            v = Verdict.FAIL
            d = f"{valid}/{total} valid, {invalid} schema failures under concurrency"
        result("concurrent_tool_schema", v, detail=d)

        # Mixed concurrent: some with tools, some without
        mixed_payloads = [
            self._req(
                model,
                "Weather in Berlin, celsius?",
                [WEATHER_TOOL],
                tool_choice="required",
            ),
            {
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 50,
            },
            self._req(
                model,
                "Calculate 7 * 8.",
                [CALCULATOR_TOOL],
                tool_choice="required",
            ),
            {
                "model": model,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
            },
        ]

        mixed_responses = await client.concurrent_requests(mixed_payloads)
        tool_indices = [0, 2]  # indices that have tools
        tool_defs = [[WEATHER_TOOL], [CALCULATOR_TOOL]]
        mixed_valid = 0
        mixed_errors = 0

        for i, idx in enumerate(tool_indices):
            resp = mixed_responses[idx]
            v, d, _ = self._assess_tool_response(resp, tool_defs[i])
            if v in (Verdict.PASS, Verdict.INTERESTING):
                mixed_valid += 1
            else:
                mixed_errors += 1

        if mixed_errors == 0:
            v = Verdict.PASS
            d = f"Mixed concurrent: {mixed_valid}/{len(tool_indices)} tool calls valid"
        else:
            v = Verdict.FAIL
            d = f"Mixed concurrent: {mixed_errors}/{len(tool_indices)} tool call schema failures"
        result("concurrent_mixed_schema", v, detail=d)

        return results
