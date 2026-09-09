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
"""Tool calling schema enforcement: constrained decoding and JSON schema support.

Verifies that tool calls match the requested tool call schema, and that the
complete JSON schema specification is supported.

Each test sends a tool-calling request with a carefully designed schema and
validates the returned tool call against it. Every test runs under both
tool_choice="required" and tool_choice="auto"; in auto mode the model may
legitimately decline to call a tool, which counts as a pass.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from helpers import budget_exhausted, make_tool
from pydantic import BaseModel

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

# Run every test under BOTH tool_choice modes: "required" (enforced from token
# 0) and "auto" (conditional enforcement that only arms once the model opens a
# tool call).
_env_choice = os.environ.get("TC_SCHEMA_TOOL_CHOICE")
_TOOL_CHOICE_MODES: tuple[str, ...] = (
    (_env_choice,) if _env_choice else ("required", "auto")
)

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


def _validate_args(
    args: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """Validate tool call args against a JSON schema. Returns error strings."""
    errors: list[str] = []
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    for req in required:
        if req not in args:
            errors.append(f"missing required field: {req}")

    for key, val in args.items():
        if key not in props:
            continue
        prop_schema = props[key]

        if "enum" in prop_schema:
            if val not in prop_schema["enum"]:
                errors.append(
                    f"{key}: {val!r} not in enum {prop_schema['enum']}"
                )

        expected_type = prop_schema.get("type")
        if expected_type == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                errors.append(
                    f"{key}: expected integer, got {type(val).__name__}"
                )
            if isinstance(val, float) and val != int(val):
                errors.append(f"{key}: got float {val}, expected integer")
        elif expected_type == "number":
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(
                    f"{key}: expected number, got {type(val).__name__}"
                )
        elif expected_type == "string":
            if not isinstance(val, str):
                errors.append(
                    f"{key}: expected string, got {type(val).__name__}"
                )
        elif expected_type == "boolean":
            if not isinstance(val, bool):
                errors.append(
                    f"{key}: expected boolean, got {type(val).__name__}"
                )
        elif expected_type == "object" and isinstance(val, dict):
            nested_errors = _validate_args(val, prop_schema)
            errors.extend(f"{key}.{e}" for e in nested_errors)
        elif expected_type == "array" and isinstance(val, list):
            items_schema = prop_schema.get("items", {})
            for i, item in enumerate(val):
                if (
                    isinstance(item, dict)
                    and items_schema.get("type") == "object"
                ):
                    nested_errors = _validate_args(item, items_schema)
                    errors.extend(f"{key}[{i}].{e}" for e in nested_errors)

    return errors


def _extract_tc_args(resp: Any) -> tuple[dict[str, Any] | None, str]:
    """Extract parsed tool call arguments from an OpenAI SDK response.

    Returns (args_dict, error_string). On success error_string is empty.
    """
    tcs = resp.choices[0].message.tool_calls
    if not tcs:
        return None, "no tool_calls returned"
    raw = tcs[0].function.arguments
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON args: {e}"
    if not isinstance(args, dict):
        return None, f"args not dict: {type(args).__name__}"
    return args, ""


@register_scenario
class TCSchemaEnforcement(BaseScenario):
    name = "tc_schema_enforcement"
    description = (
        "Tool calling schema enforcement: required fields, enums, "
        "integer vs number, null, nested objects, fixed property order"
    )
    tags = ["validation", "tool_calling", "schema", "grammar"]
    requires_validator = True
    scenario_type = "validation"

    # Set per-mode pass in run().
    _tool_choice: str = "required"

    def _nocall_ok(self, err: str) -> bool:
        """In auto mode the model may legitimately decline to call a tool, which
        is a PASS not an enforcement failure. Any other extraction error
        (malformed JSON, non-dict args) is still a real failure."""
        return self._tool_choice == "auto" and err == "no tool_calls returned"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        v = config.validator
        if not v:
            return [
                self.make_result(
                    self.name,
                    "setup",
                    Verdict.ERROR,
                    detail="No validator client available",
                )
            ]
        loop = asyncio.get_running_loop()
        all_results: list[ScenarioResult] = []
        # Run every test under each tool_choice mode. Tag each result so the
        # auto and required variants stay distinct in the report.
        for mode in _TOOL_CHOICE_MODES:
            self._tool_choice = mode
            mode_results = await self._run_all_tests(v, loop)
            for r in mode_results:
                r.test_name = f"{r.test_name}[tool_choice={mode}]"
            all_results.extend(mode_results)
        return all_results

    async def _run_all_tests(self, v: Any, loop: Any) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        results.extend(await self._required_fields(v, loop))
        results.extend(await self._enum_enforcement(v, loop))
        results.extend(await self._integer_vs_number(v, loop))
        results.extend(await self._nested_object_typed(v, loop))
        results.extend(await self._many_properties(v, loop))
        results.extend(await self._required_consistency(v, loop))
        results.extend(await self._scientific_notation(v, loop))
        results.extend(await self._null_value(v, loop))
        results.extend(await self._multiline_string(v, loop))
        results.extend(await self._hyphenated_keys_freeform(v, loop))
        results.extend(await self._enum_with_object_value(v, loop))
        results.extend(await self._integer_scientific_notation(v, loop))
        results.extend(await self._no_type_field(v, loop))
        results.extend(await self._type_array_number_null(v, loop))
        results.extend(await self._ref_defs_fail_open(v, loop))
        results.extend(await self._const_value_fail_open(v, loop))
        results.extend(await self._nullable_type_list(v, loop))
        results.extend(await self._null_type_standalone(v, loop))
        results.extend(await self._anyof_nullable_string(v, loop))
        results.extend(await self._anyof_multi_object(v, loop))
        results.extend(await self._oneof_string_int(v, loop))
        results.extend(await self._enum_mixed_with_null(v, loop))
        results.extend(await self._enum_bool_null(v, loop))
        results.extend(await self._additional_properties_true(v, loop))
        results.extend(await self._deep_nesting_5_levels(v, loop))
        results.extend(await self._required_optional_mix(v, loop))
        results.extend(await self._all_optional_empty_args(v, loop))
        results.extend(await self._additional_properties_false(v, loop))
        results.extend(await self._ref_defs_recursive(v, loop))
        results.extend(await self._type_list_object_with_properties(v, loop))
        results.extend(await self._type_list_array_with_items(v, loop))
        results.extend(await self._properties_without_type_object(v, loop))
        results.extend(await self._enum_dict_literal_exact(v, loop))
        results.extend(await self._additional_properties_typed_values(v, loop))
        results.extend(await self._ref_defs_array_enforcement(v, loop))
        results.extend(await self._ref_defs_enum_enforced(v, loop))
        results.extend(await self._ref_defs_chain_a_b_c(v, loop))
        results.extend(await self._ref_defs_reused_same_def(v, loop))
        results.extend(await self._ref_defs_anyof_nullable(v, loop))
        results.extend(await self._ref_defs_required_propagation(v, loop))
        results.extend(await self._ref_defs_nested_array_in_object(v, loop))
        results.extend(await self._ref_defs_additional_props_false(v, loop))
        results.extend(await self._ref_defs_cross_referencing(v, loop))
        results.extend(await self._ref_defs_adversarial_extra_fields(v, loop))
        results.extend(await self._ref_defs_array_of_enums(v, loop))
        results.extend(await self._ref_defs_integer_enforcement(v, loop))
        results.extend(await self._ref_defs_recursive_enum_leaf(v, loop))
        results.extend(await self._ref_defs_boolean_type(v, loop))
        results.extend(await self._ref_defs_type_array_nullable(v, loop))
        results.extend(await self._anyof_nullable_integer(v, loop))
        results.extend(await self._anyof_nullable_boolean(v, loop))
        results.extend(await self._anyof_nullable_number(v, loop))
        results.extend(await self._anyof_nullable_array(v, loop))
        results.extend(await self._anyof_nullable_object(v, loop))
        results.extend(await self._anyof_string_or_integer(v, loop))
        results.extend(await self._anyof_nested_in_array_items(v, loop))
        results.extend(await self._anyof_multiple_required_fields(v, loop))
        results.extend(await self._anyof_with_enum_branch(v, loop))
        results.extend(await self._anyof_with_ref_and_null(v, loop))
        results.extend(await self._anyof_deeply_nested(v, loop))
        results.extend(
            await self._ref_defs_adversarial_all_constraints(v, loop)
        )
        results.extend(await self._ref_defs_required_only_minimal(v, loop))
        results.extend(
            await self._ref_defs_nested_items_type_enforcement(v, loop)
        )

        return results

    def _tc(
        self,
        v: Any,
        loop: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        return loop.run_in_executor(
            None,
            lambda: v.tc_chat(
                messages,
                tools,
                tool_choice=self._tool_choice,
                max_tokens=1024,
                **kwargs,
            ),
        )

    async def _run_tc_test(
        self,
        v: Any,
        loop: Any,
        test_name: str,
        schema: dict[str, Any],
        tool_name: str,
        tool_desc: str,
        user_message: str,
        validate: Callable[[dict[str, Any]], tuple[Verdict, str]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[ScenarioResult]:
        """Run a single tool-call test with standard boilerplate.

        Handles the try/budget_exhausted/extract_args/except envelope.
        If ``validate`` is provided, it receives the parsed args dict and
        must return ``(verdict, detail)``.  Otherwise falls back to
        ``_validate_args`` against the schema.
        """
        if tool_choice is None:
            tool_choice = self._tool_choice
        if tools is None:
            tools = [make_tool(tool_name, schema, tool_desc)]
        chat_kwargs: dict[str, Any] = {
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
        }
        # Enforcement is a per-step mask property, so greedy (temperature=0)
        # decoding is the deterministic probe; only set it when a case opts in.
        # A case asserting the model VOLUNTEERS something must pin this, or it
        # measures the server's default rather than the schema. Better still,
        # put the requirement in the schema, as additional_properties_true does.
        if temperature is not None:
            chat_kwargs["temperature"] = temperature
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [{"role": "user", "content": user_message}],
                    tools,
                    **chat_kwargs,
                ),
            )
            # Judge the payload before the budget: a backend may report
            # finish_reason="length" beside a call it did finish, and only
            # non-MAX-Serve backends hit that -- the wrong way round for a
            # suite that compares them.
            args, err = _extract_tc_args(resp)
            if err:
                # No call to judge. If the response was truncated, that is the
                # explanation, and it is not a verdict on the grammar.
                if budget_exhausted(resp):
                    return [
                        self.make_result(
                            self.name,
                            test_name,
                            Verdict.INTERESTING,
                            detail="Budget exhausted",
                        )
                    ]
                if self._nocall_ok(err):
                    return [
                        self.make_result(
                            self.name,
                            test_name,
                            Verdict.PASS,
                            detail="auto: model declined to call a tool",
                        )
                    ]
                return [
                    self.make_result(
                        self.name, test_name, Verdict.FAIL, detail=err
                    )
                ]
            assert args is not None
            if validate is not None:
                verdict, detail = validate(args)
            else:
                errors = _validate_args(args, schema)
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                detail = "; ".join(errors) or f"OK: {json.dumps(args)}"
            # A truncated payload is trusted only when it passes: a cut-off
            # object looks exactly like a missing required key.
            if verdict is Verdict.FAIL and budget_exhausted(resp):
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.INTERESTING,
                        detail=f"Budget exhausted, payload incomplete: {detail}",
                    )
                ]
            return [
                self.make_result(self.name, test_name, verdict, detail=detail)
            ]
        except Exception as e:
            return [
                self.make_result(
                    self.name, test_name, Verdict.ERROR, error=str(e)
                )
            ]

    # ------------------------------------------------------------------
    # 1. Required fields always present
    # ------------------------------------------------------------------

    async def _required_fields(self, v: Any, loop: Any) -> list[ScenarioResult]:
        return await self._run_tc_test(
            v,
            loop,
            test_name="required_fields_present",
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "country", "unit"],
                "additionalProperties": False,
            },
            tool_name="get_weather",
            tool_desc="Get weather for a city",
            user_message="Weather in Paris, France in celsius?",
        )

    # ------------------------------------------------------------------
    # 2. Enum enforcement
    # ------------------------------------------------------------------

    async def _enum_enforcement(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        }
        tool = make_tool("get_temperature", schema, "Get temperature")

        # Run 5 times to increase chance of catching non-deterministic violations
        enum_pass = 0
        enum_fail_details: list[str] = []
        for i in range(5):
            try:
                resp = await self._tc(
                    v,
                    loop,
                    [
                        {
                            "role": "user",
                            "content": f"Temperature in city #{i + 1}: Tokyo?",
                        }
                    ],
                    [tool],
                )
                args, err = _extract_tc_args(resp)
                if err:
                    # Truncation explains a missing call, so skip the sample
                    # rather than charge it to the grammar.
                    if not budget_exhausted(resp) and not self._nocall_ok(err):
                        enum_fail_details.append(f"run {i}: {err}")
                    continue
                assert args is not None
                unit = args.get("unit")
                if unit in ("celsius", "fahrenheit"):
                    enum_pass += 1
                elif not budget_exhausted(resp):
                    enum_fail_details.append(f"run {i}: unit={unit!r}")
            except Exception as e:
                enum_fail_details.append(f"run {i}: {e}")

        if enum_fail_details:
            results.append(
                self.make_result(
                    self.name,
                    "enum_string_enforcement",
                    Verdict.FAIL,
                    detail=f"{enum_pass}/5 valid; {'; '.join(enum_fail_details[:3])}",
                )
            )
        elif enum_pass == 0:
            # Every run was unjudgeable (budget exhausted or no call), so
            # there is nothing to charge to the grammar either way.
            results.append(
                self.make_result(
                    self.name,
                    "enum_string_enforcement",
                    Verdict.INTERESTING,
                    detail="All 5 runs unjudgeable (budget exhausted or no call)",
                )
            )
        else:
            results.append(
                self.make_result(
                    self.name,
                    "enum_string_enforcement",
                    Verdict.PASS,
                    detail=f"{enum_pass}/5 all valid enum values",
                )
            )

        # Integer enum
        int_schema = {
            "type": "object",
            "properties": {
                "status_code": {
                    "type": "integer",
                    "enum": [200, 404, 500],
                },
            },
            "required": ["status_code"],
            "additionalProperties": False,
        }
        int_tool = make_tool(
            "get_status",
            int_schema,
            "Return an HTTP status code for a URL check",
        )

        try:
            resp = await self._tc(
                v,
                loop,
                [
                    {
                        "role": "user",
                        "content": "Check if example.com is reachable and return the status code.",
                    }
                ],
                [int_tool],
            )
            args, err = _extract_tc_args(resp)
            if err and budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "enum_integer_enforcement",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                if err and self._nocall_ok(err):
                    verdict = Verdict.PASS
                    detail = "auto: model declined to call a tool"
                elif err:
                    verdict = Verdict.FAIL
                    detail = err
                else:
                    assert args is not None
                    sc = args.get("status_code")
                    if sc in (200, 404, 500):
                        verdict = Verdict.PASS
                        detail = f"status_code={sc}"
                    elif budget_exhausted(resp):
                        verdict = Verdict.INTERESTING
                        detail = (
                            "Budget exhausted, payload incomplete: "
                            f"status_code={sc!r} not in enum [200, 404, 500]"
                        )
                    else:
                        verdict = Verdict.FAIL
                        detail = (
                            f"status_code={sc!r} not in enum [200, 404, 500]"
                        )
                results.append(
                    self.make_result(
                        self.name,
                        "enum_integer_enforcement",
                        verdict,
                        detail=detail,
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "enum_integer_enforcement",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        return results

    # ------------------------------------------------------------------
    # 3. Integer vs number typing
    # ------------------------------------------------------------------

    async def _integer_vs_number(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            dur = args.get("duration_minutes")
            if isinstance(dur, float):
                return Verdict.FAIL, (
                    f"duration_minutes is float ({dur}), expected int"
                )
            if isinstance(dur, int) and not isinstance(dur, bool):
                return Verdict.PASS, (
                    f"duration_minutes={dur} (int), rating={args.get('rating')}"
                )
            return Verdict.FAIL, (
                f"duration_minutes has type {type(dur).__name__}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="integer_no_decimal",
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                    "rating": {"type": "number"},
                },
                "required": ["title", "duration_minutes", "rating"],
                "additionalProperties": False,
            },
            tool_name="create_event",
            tool_desc="Create an event with a title, duration in minutes, and rating",
            user_message=(
                "Create an event titled 'Team Standup' that lasts "
                "30 minutes with a rating of 4.5."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 4. Nested object inner types enforced
    # ------------------------------------------------------------------

    async def _nested_object_typed(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        schema = {
            "type": "object",
            "properties": {
                "server": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "tls": {"type": "boolean"},
                    },
                    "required": ["host", "port", "tls"],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["server", "tags"],
            "additionalProperties": False,
        }

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors = _validate_args(args, schema)
            port = args.get("server", {}).get("port")
            if isinstance(port, str):
                errors.append(
                    f"server.port is string '{port}', expected integer"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, "; ".join(errors) or f"OK: {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="nested_object_types",
            schema=schema,
            tool_name="configure_server",
            tool_desc="Configure a server connection with host, port, TLS, and tags",
            user_message=(
                "Configure server at api.example.com on port 443 "
                "with TLS enabled, tags: production, primary."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 5. Many properties (20 required)
    # ------------------------------------------------------------------

    async def _many_properties(self, v: Any, loop: Any) -> list[ScenarioResult]:
        field_names = [
            "name",
            "email",
            "phone",
            "address",
            "city",
            "state",
            "zip_code",
            "country",
            "company",
            "title",
            "department",
            "team",
            "manager",
            "start_date",
            "office",
            "floor",
            "desk",
            "badge_id",
            "role",
            "level",
        ]
        schema = {
            "type": "object",
            "properties": {f: {"type": "string"} for f in field_names},
            "required": field_names,
            "additionalProperties": False,
        }

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors = _validate_args(args, schema)
            missing = [f for f in field_names if f not in args]
            if missing:
                errors.append(f"missing {len(missing)} fields: {missing[:5]}")
            if len(args) < len(field_names):
                errors.append(
                    f"only {len(args)} keys, expected {len(field_names)}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, "; ".join(
                errors
            ) or f"OK: {len(args)} fields present"

        return await self._run_tc_test(
            v,
            loop,
            test_name="many_properties_20",
            schema=schema,
            tool_name="register_employee",
            tool_desc="Register a new employee with all fields",
            user_message=(
                "Register employee: Jane Smith, jane@acme.com, "
                "555-1234, 100 Main St, Springfield, IL, 62701, "
                "US, Acme Corp, Senior Engineer, Engineering, "
                "Platform, Bob Jones, 2025-01-15, HQ, 3, D42, "
                "B-9876, engineer, senior."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 6. Required fields consistency (N runs)
    # ------------------------------------------------------------------

    async def _required_consistency(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
            },
            "required": ["a", "b", "c"],
            "additionalProperties": False,
        }
        tool = make_tool(
            "submit_form",
            schema,
            "Submit a form with fields a (string), b (integer), c (boolean)",
        )

        n_runs = 5
        passes = 0
        fail_details: list[str] = []
        for i in range(n_runs):
            try:
                resp = await self._tc(
                    v,
                    loop,
                    [
                        {
                            "role": "user",
                            "content": f"Submit form: a='hello{i}', b={i + 1}, c=true.",
                        }
                    ],
                    [tool],
                )
                args, err = _extract_tc_args(resp)
                if err:
                    if not budget_exhausted(resp) and not self._nocall_ok(err):
                        fail_details.append(f"run {i}: {err}")
                    continue
                assert args is not None
                errors = _validate_args(args, schema)
                if errors:
                    if not budget_exhausted(resp):
                        fail_details.append(f"run {i}: {'; '.join(errors)}")
                else:
                    passes += 1
            except Exception as e:
                fail_details.append(f"run {i}: {e}")

        if fail_details:
            results.append(
                self.make_result(
                    self.name,
                    "required_consistency_5_runs",
                    Verdict.FAIL,
                    detail=f"{passes}/{n_runs} valid; {'; '.join(fail_details[:3])}",
                )
            )
        elif passes == 0:
            # Every run was unjudgeable (budget exhausted or no call), so
            # there is nothing to charge to the grammar either way.
            results.append(
                self.make_result(
                    self.name,
                    "required_consistency_5_runs",
                    Verdict.INTERESTING,
                    detail=f"All {n_runs} runs unjudgeable (budget exhausted or no call)",
                )
            )
        else:
            results.append(
                self.make_result(
                    self.name,
                    "required_consistency_5_runs",
                    Verdict.PASS,
                    detail=f"{passes}/{n_runs} all valid across runs",
                )
            )

        return results

    # ------------------------------------------------------------------
    # 7. Scientific notation accepted in number fields
    # ------------------------------------------------------------------

    async def _scientific_notation(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that number fields accept scientific notation (e.g. 1.5e10).

        We can't force the model to emit sci notation, so we validate that
        the returned number field parses as a valid Python float/int -- the
        grammar allows sci notation and the parser handles it.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            dist = args.get("distance_meters")
            pop = args.get("population")
            errors: list[str] = []
            if not isinstance(dist, (int, float)) or isinstance(dist, bool):
                errors.append(
                    f"distance_meters: expected number, got {type(dist).__name__}"
                )
            if not isinstance(pop, int) or isinstance(pop, bool):
                errors.append(
                    f"population: expected integer, got {type(pop).__name__}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: distance={dist}, population={pop}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="number_field_valid",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "distance_meters": {"type": "number"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "distance_meters", "population"],
                "additionalProperties": False,
            },
            tool_name="record_celestial_body",
            tool_desc="Record a celestial body with name, distance in meters, and population",
            user_message=(
                "Record celestial body: Earth, distance from Sun "
                "is 149597870700 meters, population 8000000000."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 8. Null value accepted for optional fields
    # ------------------------------------------------------------------

    async def _null_value(self, v: Any, loop: Any) -> list[ScenarioResult]:
        """Verify that the grammar does not crash when a field could be null.

        The model may or may not emit null -- we just verify the request
        succeeds and produces valid JSON args with correct required fields.
        The grammar's null_val rule ensures null is parseable if emitted.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "name" not in args:
                errors.append("missing required field: name")
            if "age" not in args:
                errors.append("missing required field: age")
            nickname = args.get("nickname")
            if nickname is not None and not isinstance(nickname, str):
                errors.append(
                    f"nickname: expected string or null, got {type(nickname).__name__}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, "; ".join(errors) or f"OK: nickname={nickname!r}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="optional_field_nullable",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "nickname": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
            tool_name="register_person",
            tool_desc="Register a person. nickname is optional and may be omitted.",
            user_message=(
                "Register person: name is John Doe, age 30. No nickname."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 9. Multiline strings accepted in string fields
    # ------------------------------------------------------------------

    async def _multiline_string(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify the grammar accepts newlines inside string values.

        STRING_CONTENT must match newline characters so the model can
        produce multi-line strings (e.g. poems, addresses, code).
        """
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["title", "content"],
            "additionalProperties": False,
        }

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors = _validate_args(args, schema)
            content = args.get("content", "")
            has_newline = "\n" in content if isinstance(content, str) else False
            if errors:
                return Verdict.FAIL, "; ".join(errors)
            if has_newline:
                return (
                    Verdict.PASS,
                    f"OK: content has newline, len={len(content)}",
                )
            return Verdict.FAIL, (
                f"grammar likely blocked newlines: "
                f"content={content!r} (no newline)"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="multiline_string_value",
            schema=schema,
            tool_name="save_note",
            tool_desc="Save a note with a title and multi-line content",
            user_message=(
                "Save a note titled 'Grocery List'. The content "
                "field MUST contain actual newline characters "
                "between items. Format it exactly like:\n"
                "- milk\n- eggs\n- bread\n"
                "Include the newlines in the content string."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 10. Hyphenated keys in free-form object properties
    # ------------------------------------------------------------------

    async def _hyphenated_keys_freeform(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify free-form objects accept property names with hyphens/dots.

        The KEY terminal must allow characters beyond [a-zA-Z0-9_] so
        that keys like 'Content-Type' or 'x.custom' work in untyped
        object properties.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            headers = args.get("headers")
            if not isinstance(headers, dict):
                return Verdict.FAIL, (
                    f"headers not a dict: {type(headers).__name__}"
                )
            if any("-" in k for k in headers):
                return Verdict.PASS, f"keys={list(headers.keys())}"
            return Verdict.FAIL, (
                f"grammar likely blocked hyphens in keys: "
                f"keys={list(headers.keys())}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="freeform_object_hyphen_keys",
            schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                },
                "required": ["url", "headers"],
                "additionalProperties": False,
            },
            tool_name="send_request",
            tool_desc="Send an HTTP request with a URL and custom headers object",
            user_message=(
                "Send a request to https://api.example.com/data "
                "with headers Content-Type=application/json and "
                "Accept=text/html."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 11. Enum with object/array values accepted
    # ------------------------------------------------------------------

    async def _enum_with_object_value(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify enum alternatives that are objects or arrays are not dropped.

        _enum_value_rule must handle dict/list enum values instead of
        silently discarding them, which would make a valid alternative
        impossible to produce.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            config = args.get("config")
            if isinstance(config, dict):
                return Verdict.PASS, f"config={config!r} (object selected)"
            return Verdict.FAIL, (
                f"grammar likely dropped object enum alternative: "
                f"config={config!r} (type={type(config).__name__})"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="enum_with_object_value",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "config": {
                        "enum": [
                            "default",
                            {"mode": "advanced", "level": 3},
                        ],
                    },
                },
                "required": ["name", "config"],
                "additionalProperties": False,
            },
            tool_name="apply_config",
            tool_desc=(
                "Apply a configuration. config is either 'default' or "
                "an object {mode: 'advanced', level: 3}."
            ),
            user_message=(
                "Apply the advanced configuration (mode=advanced, "
                "level=3) for component 'router'."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 12. Integer field accepts scientific notation
    # ------------------------------------------------------------------

    async def _integer_scientific_notation(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify integer fields accept scientific notation (e.g. 1e50).

        Uses astronomically large numbers that are impractical to write
        in digit form, forcing the model to use scientific notation.
        The INTEGER terminal must allow an exponent suffix for this to work.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            qty = args.get("quantity")
            if isinstance(qty, float) and qty >= 1e20:
                return Verdict.PASS, (
                    f"quantity={qty} (sci notation parsed as float)"
                )
            if isinstance(qty, int) and qty >= 10**20:
                return Verdict.PASS, f"quantity={qty} (large integer)"
            return Verdict.FAIL, (
                f"quantity={qty!r} (type={type(qty).__name__}) -- "
                f"expected a very large number via sci notation"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="integer_scientific_notation",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["name", "quantity"],
                "additionalProperties": False,
            },
            tool_name="record_quantity",
            tool_desc=(
                "Record a named quantity. The quantity field is an integer. "
                "You MUST use scientific notation (e.g. 1e50) for large values."
            ),
            user_message=(
                "Record quantity: name='atoms_in_universe', "
                "quantity=1e80. Write the quantity using "
                "scientific notation exactly as 1e80."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 13. Nullable type list (e.g. ["string", "null"])
    # ------------------------------------------------------------------

    async def _nullable_type_list(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify schemas with union types like ["string", "null"] don't crash.

        JSON Schema allows type to be a list. When the grammar generator
        receives a list for type, dict.get() raises TypeError (unhashable
        type: 'list') unless handled explicitly.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "name" not in args:
                errors.append("missing required field: name")
            if "nickname" not in args:
                errors.append("missing required field: nickname")
            else:
                nn = args["nickname"]
                if nn is not None and not isinstance(nn, str):
                    errors.append(
                        f"nickname: expected string or null, got {type(nn).__name__}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: name={args.get('name')!r}, nickname={args.get('nickname')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="nullable_type_list",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "nickname": {"type": ["string", "null"]},
                },
                "required": ["name", "nickname"],
                "additionalProperties": False,
            },
            tool_name="greet_user",
            tool_desc="Greet a user. nickname can be a string or null.",
            user_message=(
                "Greet user: name is 'Alice', nickname is null "
                "(she has no nickname). Set nickname to null."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 14. Standalone null type property
    # ------------------------------------------------------------------

    async def _null_type_standalone(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify a property with type "null" produces null and uses null_val rule.

        _JSON_TYPE_TO_GRAMMAR_RULE must have an entry for "null" mapping
        to "null_val", otherwise the property falls to generic "value"
        (fail-open).
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "action" not in args:
                errors.append("missing required field: action")
            if "reason" not in args:
                errors.append("missing required field: reason")
            elif args["reason"] is not None:
                errors.append(
                    f"reason: expected null, got "
                    f"{type(args['reason']).__name__}={args['reason']!r}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: action={args.get('action')!r}, reason=null"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="null_type_standalone",
            schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "reason": {"type": "null"},
                },
                "required": ["action", "reason"],
                "additionalProperties": False,
            },
            tool_name="cancel_request",
            tool_desc="Cancel a request. reason must always be null.",
            user_message=(
                "Cancel the pending request. The action is 'cancel' "
                "and reason is null (must be null, not a string)."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 15. anyOf with nullable string: anyOf: [{type:string},{type:null}]
    # ------------------------------------------------------------------

    async def _anyof_nullable_string(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "name" not in args:
                errors.append("missing required field: name")
            if "middle_name" not in args:
                errors.append("missing required field: middle_name")
            else:
                mn = args["middle_name"]
                if mn is not None and not isinstance(mn, str):
                    errors.append(
                        f"middle_name: expected string or null, "
                        f"got {type(mn).__name__}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: middle_name={args.get('middle_name')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_nullable_string",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "middle_name": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                },
                "required": ["name", "middle_name"],
                "additionalProperties": False,
            },
            tool_name="register_name",
            tool_desc="Register a name. middle_name can be a string or null.",
            user_message=(
                "Register: name='Alice', middle_name is null "
                "(she has no middle name). Set middle_name to null."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 16. anyOf with multiple object branches
    # ------------------------------------------------------------------

    async def _anyof_multi_object(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            payment = args.get("payment")
            if not isinstance(payment, dict):
                return Verdict.FAIL, (
                    f"payment not dict: {type(payment).__name__}"
                )
            has_method = isinstance(payment.get("method"), str)
            has_card = isinstance(payment.get("card_last4"), str)
            has_bank = isinstance(payment.get("bank_name"), str)
            if has_method and (has_card or has_bank):
                return Verdict.PASS, f"OK: payment={payment!r}"
            return Verdict.FAIL, (
                f"payment missing expected fields: {payment!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_multi_object",
            schema={
                "type": "object",
                "properties": {
                    "payment": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "method": {"type": "string"},
                                    "card_last4": {"type": "string"},
                                },
                                "required": ["method", "card_last4"],
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "method": {"type": "string"},
                                    "bank_name": {"type": "string"},
                                },
                                "required": ["method", "bank_name"],
                            },
                        ],
                    },
                },
                "required": ["payment"],
                "additionalProperties": False,
            },
            tool_name="process_payment",
            tool_desc=(
                "Process a payment. Payment is either a card (with card_last4) "
                "or bank transfer (with bank_name)."
            ),
            user_message=(
                "Process a credit card payment, card last 4 digits are 1234."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 17. oneOf with string or integer
    # ------------------------------------------------------------------

    async def _oneof_string_int(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            ident = args.get("identifier")
            if isinstance(ident, (str, int)) and not isinstance(ident, bool):
                return Verdict.PASS, (
                    f"OK: identifier={ident!r} ({type(ident).__name__})"
                )
            return Verdict.FAIL, (
                f"identifier={ident!r} ({type(ident).__name__}) "
                f"-- expected string or integer"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="oneof_string_int",
            schema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "oneOf": [{"type": "string"}, {"type": "integer"}],
                    },
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
            tool_name="lookup_entity",
            tool_desc="Look up an entity by identifier (string name or integer ID).",
            user_message="Look up entity with ID 42.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 18. Enum with null among string values
    # ------------------------------------------------------------------

    async def _enum_mixed_with_null(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            status = args.get("status")
            if status in ("active", "inactive", None):
                return Verdict.PASS, f"OK: status={status!r}"
            return Verdict.FAIL, (
                f"status={status!r} not in enum ['active','inactive',null]"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="enum_mixed_with_null",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {"enum": ["active", "inactive", None]},
                },
                "required": ["name", "status"],
                "additionalProperties": False,
            },
            tool_name="set_status",
            tool_desc="Set user status. Status is 'active', 'inactive', or null.",
            user_message=(
                "Set status for user 'Bob' to null (no status assigned)."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 19. Enum with boolean and null values
    # ------------------------------------------------------------------

    async def _enum_bool_null(self, v: Any, loop: Any) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            enabled = args.get("enabled")
            if enabled in (True, False, None):
                return Verdict.PASS, f"OK: enabled={enabled!r}"
            return Verdict.FAIL, (
                f"enabled={enabled!r} ({type(enabled).__name__}) "
                f"not in [true, false, null]"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="enum_bool_null",
            schema={
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "enabled": {"enum": [True, False, None]},
                },
                "required": ["feature", "enabled"],
                "additionalProperties": False,
            },
            tool_name="toggle_feature",
            tool_desc="Toggle a feature. enabled is true, false, or null (unknown).",
            user_message="Toggle feature 'dark_mode' to enabled=true.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 20. $ref/$defs
    # ------------------------------------------------------------------

    async def _ref_defs_fail_open(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that $ref/$defs are resolved and enforced by the grammar."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            item = args.get("item")
            if not isinstance(item, dict):
                return Verdict.FAIL, (
                    f"item={item!r} is not a dictionary with name and price"
                )
            has_name = isinstance(item.get("name"), str)
            has_price = isinstance(item.get("price"), (int, float))
            if has_name and has_price:
                return Verdict.PASS, f"OK: item={item!r}"
            return Verdict.FAIL, (
                f"item={item!r} is not a dictionary with name and price"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_fail_open",
            schema={
                "type": "object",
                "properties": {
                    "item": {"$ref": "#/$defs/Item"},
                },
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                        },
                        "required": ["name", "price"],
                    },
                },
                "required": ["item"],
                "additionalProperties": False,
            },
            tool_name="add_item",
            tool_desc="Add an item with name and price.",
            user_message="Add item: name='Widget', price=9.99.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 21. const keyword
    # ------------------------------------------------------------------

    async def _const_value_fail_open(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that the const keyword is enforced."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            version = args.get("version")
            if version == "2.0":
                return Verdict.PASS, (
                    f"OK: version={version!r} (model produced correct const)"
                )
            return Verdict.FAIL, (
                f"version={version!r} is not '2.0' "
                f"(model did not produce correct const)"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="const_value_fail_open",
            schema={
                "type": "object",
                "properties": {
                    "version": {"const": "2.0"},
                    "name": {"type": "string"},
                },
                "required": ["version", "name"],
                "additionalProperties": False,
            },
            tool_name="create_spec",
            tool_desc="Create a spec. version must be exactly '2.0'.",
            user_message="Create a spec named 'my-api' with version '2.0'.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 22. additionalProperties: true (default) — extra props allowed
    # ------------------------------------------------------------------

    async def _additional_properties_true(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """When additionalProperties is not set (defaults to true in JSON
        Schema), the grammar must not reject undeclared properties.

        ``minProperties`` is what makes this an assertion about the grammar
        rather than about the server's default temperature: with three required
        and one declared, a grammar that admits undeclared keys must produce
        them, and one that forbids them cannot compile at all.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            if "name" not in args:
                return Verdict.FAIL, "missing required field: name"
            extra_keys = set(args.keys()) - {"name"}
            if len(extra_keys) >= 2:
                return Verdict.PASS, (
                    f"OK: undeclared properties accepted: {sorted(extra_keys)}"
                )
            return Verdict.FAIL, (
                f"minProperties=3 requires two undeclared properties beside "
                f"'name', got {sorted(extra_keys)} -- the grammar admitted "
                f"fewer keys than the schema demands"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="additional_properties_true",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
                "minProperties": 3,
            },
            tool_name="flexible_input",
            tool_desc=(
                "Accept flexible input. name is required, but you may include "
                "any additional fields like age, email, etc."
            ),
            user_message=(
                "Create input: name='Alice', age=30, "
                "email='alice@example.com'. Include all three fields."
            ),
            validate=validate,
            # The schema forces the outcome, so nothing is left to sampling.
            temperature=0,
        )

    # ------------------------------------------------------------------
    # 23. Deep nesting (5 levels of nested objects)
    # ------------------------------------------------------------------

    async def _deep_nesting_5_levels(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            try:
                l5_val = args["l1"]["l2"]["l3"]["l4"]["l5"]
                if not isinstance(l5_val, str):
                    errors.append(
                        f"l5: expected string, got {type(l5_val).__name__}"
                    )
            except (KeyError, TypeError) as e:
                errors.append(f"nesting traversal failed: {e}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: l1.l2.l3.l4.l5={args['l1']['l2']['l3']['l4']['l5']!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="deep_nesting_5_levels",
            schema={
                "type": "object",
                "properties": {
                    "l1": {
                        "type": "object",
                        "properties": {
                            "l2": {
                                "type": "object",
                                "properties": {
                                    "l3": {
                                        "type": "object",
                                        "properties": {
                                            "l4": {
                                                "type": "object",
                                                "properties": {
                                                    "l5": {"type": "string"},
                                                },
                                                "required": ["l5"],
                                            },
                                        },
                                        "required": ["l4"],
                                    },
                                },
                                "required": ["l3"],
                            },
                        },
                        "required": ["l2"],
                    },
                },
                "required": ["l1"],
                "additionalProperties": False,
            },
            tool_name="deep_store",
            tool_desc="Store a deeply nested value at l1.l2.l3.l4.l5.",
            user_message=(
                "Store the value 'deep_value' at the path l1.l2.l3.l4.l5."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 24. Mix of required and optional properties
    # ------------------------------------------------------------------

    async def _required_optional_mix(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            for req in ("first_name", "last_name", "email"):
                if req not in args:
                    errors.append(f"missing required field: {req}")
                elif not isinstance(args[req], str):
                    errors.append(
                        f"{req}: expected string, got {type(args[req]).__name__}"
                    )
            for opt in ("phone", "bio", "website"):
                if opt in args and not isinstance(args[opt], str):
                    errors.append(
                        f"{opt}: expected string, got {type(args[opt]).__name__}"
                    )
            allowed = {
                "first_name",
                "last_name",
                "email",
                "phone",
                "bio",
                "website",
            }
            extra = set(args.keys()) - allowed
            if extra:
                errors.append(f"undeclared properties: {sorted(extra)}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: keys={sorted(args.keys())}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="required_optional_mix",
            schema={
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "bio": {"type": "string"},
                    "website": {"type": "string"},
                },
                "required": ["first_name", "last_name", "email"],
                "additionalProperties": False,
            },
            tool_name="create_profile",
            tool_desc=(
                "Create a user profile. first_name, last_name, email are "
                "required. phone, bio, website are optional."
            ),
            user_message=(
                "Create profile: first_name='Jane', "
                "last_name='Doe', email='jane@example.com'. "
                "Skip the optional fields."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 25. All optional — model may produce empty args
    # ------------------------------------------------------------------

    async def _all_optional_empty_args(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "note" in args and not isinstance(args["note"], str):
                errors.append(
                    f"note: expected string, got {type(args['note']).__name__}"
                )
            if "priority" in args:
                p = args["priority"]
                if not isinstance(p, int) or isinstance(p, bool):
                    errors.append(
                        f"priority: expected integer, got {type(p).__name__}"
                    )
            extra = set(args.keys()) - {"note", "priority"}
            if extra:
                errors.append(f"undeclared properties: {sorted(extra)}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: args={args!r} (empty or optional fields)"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="all_optional_empty_args",
            schema={
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "priority": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            tool_name="ping",
            tool_desc="Send a ping. All parameters are optional.",
            user_message="Send a simple ping with no parameters.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 26. Property with no type field
    # ------------------------------------------------------------------

    async def _no_type_field(self, v: Any, loop: Any) -> list[ScenarioResult]:
        """A property with no 'type' key should fall back to generic 'value'
        and accept any JSON. Must not crash.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "label" not in args:
                errors.append("missing required field: label")
            if "data" not in args:
                errors.append("missing required field: data")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: data={args.get('data')!r} ({type(args.get('data')).__name__})"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="no_type_field",
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "data": {
                        "description": "Arbitrary data, no type specified."
                    },
                },
                "required": ["label", "data"],
                "additionalProperties": False,
            },
            tool_name="store_data",
            tool_desc="Store data. The data field has no type -- it can be anything.",
            user_message="Store data: label='test', data=42.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 27. type: ["number", "null"] — another type-as-array variant
    # ------------------------------------------------------------------

    async def _type_array_number_null(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Like nullable_type_list but with number instead of string.
        Exercises the same dict.get() crash path with a different type.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "label" not in args:
                errors.append("missing required field: label")
            if "score" not in args:
                errors.append("missing required field: score")
            else:
                sc = args["score"]
                if sc is not None and (
                    not isinstance(sc, (int, float)) or isinstance(sc, bool)
                ):
                    errors.append(
                        f"score: expected number or null, "
                        f"got {type(sc).__name__}={sc!r}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: score={args.get('score')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="type_array_number_null",
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "score": {"type": ["number", "null"]},
                },
                "required": ["label", "score"],
                "additionalProperties": False,
            },
            tool_name="record_score",
            tool_desc="Record a score. score is a number or null.",
            user_message="Record: label='test_run', score=95.5.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 28. additionalProperties: false — grammar rejects undeclared keys
    # ------------------------------------------------------------------

    async def _additional_properties_false(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify additionalProperties: false prevents undeclared properties.

        The prompt deliberately asks the model to include extra fields that
        are NOT in the schema. The grammar should prevent them from appearing.
        """
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city", "country"],
            "additionalProperties": False,
        }

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "city" not in args:
                errors.append("missing required field: city")
            if "country" not in args:
                errors.append("missing required field: country")
            extra = set(args.keys()) - {"city", "country"}
            if extra:
                errors.append(
                    f"additionalProperties violated: "
                    f"undeclared keys {sorted(extra)}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: keys={sorted(args.keys())} (no extras)"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="additional_properties_false",
            schema=schema,
            tool_name="get_location",
            tool_desc=(
                "Get a location. Only city and country are accepted. "
                "No other fields."
            ),
            user_message=(
                "Get the location for Tokyo, Japan. Also include the "
                "population (1400000), the continent (Asia), and "
                "the timezone (JST). Include all five fields."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 29. $ref/$defs recursive schema (linked list)
    # ------------------------------------------------------------------

    async def _ref_defs_recursive(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify recursive $ref/$defs schemas work (linked list pattern).

        This is the key use case from the OpenAI structured outputs docs:
        a node type that references itself via $ref inside anyOf.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            node = args.get("list")
            if not isinstance(node, dict):
                return Verdict.FAIL, (
                    f"list: expected object, got {type(node).__name__}"
                )
            depth = 0
            current = node
            while isinstance(current, dict):
                depth += 1
                if not isinstance(current.get("value"), str):
                    return Verdict.FAIL, (
                        f"node at depth {depth}: value is "
                        f"{type(current.get('value')).__name__}, "
                        f"expected string"
                    )
                nxt = current.get("next")
                if nxt is None:
                    break
                if not isinstance(nxt, dict):
                    return Verdict.FAIL, (
                        f"node at depth {depth}: next is "
                        f"{type(nxt).__name__}, expected object or null"
                    )
                current = nxt
            if depth < 2:
                return Verdict.FAIL, (
                    f"only {depth} node(s), expected at least 2"
                )
            return Verdict.PASS, f"OK: linked list with {depth} nodes"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_recursive",
            schema={
                "type": "object",
                "properties": {
                    "list": {"$ref": "#/$defs/node"},
                },
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "next": {
                                "anyOf": [
                                    {"$ref": "#/$defs/node"},
                                    {"type": "null"},
                                ],
                            },
                        },
                        "required": ["value", "next"],
                    },
                },
                "required": ["list"],
                "additionalProperties": False,
            },
            tool_name="build_list",
            tool_desc=(
                "Build a linked list. Each node has a string value and "
                "a next pointer (another node or null)."
            ),
            user_message=(
                "Build a linked list with 3 items: 'alpha', 'beta', "
                "'gamma'. The last node's next should be null."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 30. type: ["object", "null"] with properties (bug: constraints dropped)
    # ------------------------------------------------------------------

    async def _type_list_object_with_properties(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that type: ["object", "null"] preserves property constraints.

        Bug: when type is a list containing "object", the grammar enters
        the union branch and maps to generic object_val, silently dropping
        properties/required constraints from the same schema level.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "label" not in args:
                errors.append("missing required field: label")
            if "metadata" not in args:
                errors.append("missing required field: metadata")
            else:
                md = args["metadata"]
                if md is None:
                    pass
                elif isinstance(md, dict):
                    if not isinstance(md.get("author"), str):
                        errors.append(
                            f"metadata.author: expected string, "
                            f"got {type(md.get('author')).__name__}"
                        )
                    ver = md.get("version")
                    if not isinstance(ver, int) or isinstance(ver, bool):
                        errors.append(
                            f"metadata.version: expected integer, "
                            f"got {type(ver).__name__}"
                        )
                else:
                    errors.append(
                        f"metadata: expected object or null, "
                        f"got {type(md).__name__}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: metadata={args.get('metadata')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="type_list_object_with_properties",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "metadata": {
                        "type": ["object", "null"],
                        "properties": {
                            "author": {"type": "string"},
                            "version": {"type": "integer"},
                        },
                        "required": ["author", "version"],
                    },
                },
                "required": ["label", "metadata"],
                "additionalProperties": False,
            },
            tool_name="save_document",
            tool_desc=(
                "Save a document with a label and metadata. "
                "metadata is an object with author (string) and "
                "version (integer), or null."
            ),
            user_message=(
                "Save document: label='report', "
                "metadata with author=42 (use the integer 42, not "
                "a string) and version='three' (use the string "
                "'three', not a number). Use exactly these types."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 30b. type: ["array", "null"] with items (bug: constraints dropped)
    # ------------------------------------------------------------------

    async def _type_list_array_with_items(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that type: ["array", "null"] preserves items constraints.

        Same root cause as test 30: when type is a list containing "array",
        the grammar enters the union branch and maps to generic array_val,
        silently dropping items type constraints from the same schema level.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "label" not in args:
                errors.append("missing required field: label")
            if "tags" not in args:
                errors.append("missing required field: tags")
            else:
                tags = args["tags"]
                if tags is None:
                    pass
                elif isinstance(tags, list):
                    for i, item in enumerate(tags):
                        if not isinstance(item, str):
                            errors.append(
                                f"tags[{i}]: expected string, "
                                f"got {type(item).__name__}={item!r}"
                            )
                else:
                    errors.append(
                        f"tags: expected array or null, "
                        f"got {type(tags).__name__}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: tags={args.get('tags')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="type_list_array_with_items",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "tags": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": ["label", "tags"],
                "additionalProperties": False,
            },
            tool_name="tag_resource",
            tool_desc=("Tag a resource. tags is an array of strings, or null."),
            user_message=(
                "Tag resource: label='server-01', "
                "tags=[100, true, 3.14]. Use exactly these "
                "non-string values in the array, not strings."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 31. properties without type: "object" (bug: constraints dropped)
    # ------------------------------------------------------------------

    async def _properties_without_type_object(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify schemas with properties but no explicit type: "object" work.

        Bug: _generate_property_value_rule checks
        json_type == "object" and prop_schema.get("properties"), so a
        schema without type falls through to generic value rule, losing
        all property constraints.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("name"), str):
                errors.append(
                    f"name: expected string, "
                    f"got {type(args.get('name')).__name__}"
                )
            age = args.get("age")
            if not isinstance(age, int) or isinstance(age, bool):
                errors.append(
                    f"age: expected integer, got {type(age).__name__}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: name={args.get('name')!r}, age={age}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="properties_without_type_object",
            schema={
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
            tool_name="register_user",
            tool_desc="Register a user with name and age.",
            user_message="Register user: name='Bob', age=25.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 32. Enum with dict literals — exact match enforcement
    # ------------------------------------------------------------------

    async def _enum_dict_literal_exact(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify enum with dict values constrains to the exact literal.

        Bug: _enum_value_rule maps dict enum values to the generic
        object_val rule, which accepts ANY object instead of only the
        specific literals declared in the enum.
        """
        valid_presets = [
            {"mode": "fast", "threads": 4},
            {"mode": "safe", "threads": 1},
        ]

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            preset = args.get("preset")
            if not isinstance(preset, dict):
                return Verdict.FAIL, (
                    f"preset: expected dict, got {type(preset).__name__}"
                )
            if preset in valid_presets:
                return Verdict.PASS, f"OK: preset={preset!r} (exact match)"
            return Verdict.FAIL, (
                f"preset={preset!r} does not exactly match any "
                f"enum literal: {valid_presets}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="enum_dict_literal_exact",
            schema={
                "type": "object",
                "properties": {
                    "preset": {
                        "enum": valid_presets,
                    },
                },
                "required": ["preset"],
                "additionalProperties": False,
            },
            tool_name="apply_preset",
            tool_desc=(
                "Apply a preset. preset must be exactly one of: "
                "{mode:'fast', threads:4} or {mode:'safe', threads:1}."
            ),
            user_message=("Apply the fast preset (mode='fast', threads=4)."),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 33. additionalProperties with typed values
    #     Tests that constrained decoding enforces value types when
    #     objects use additionalProperties: {type: ...} instead of
    #     explicit properties. Uses forced tool_choice + decoy tool.
    # ------------------------------------------------------------------

    async def _additional_properties_typed_values(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        decoy_tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": "__decoy_noop",
                "description": "No-op decoy tool",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        }

        def _validate_typed_dict(
            obj: object,
            label: str,
            check_value: Callable[[object], bool],
            expected_type_name: str,
        ) -> list[str]:
            errors: list[str] = []
            if not isinstance(obj, dict):
                return [f"{label}: expected dict, got {type(obj).__name__}"]
            for k, val in obj.items():
                if not check_value(val):
                    errors.append(
                        f"{label}[{k!r}]: expected {expected_type_name}, "
                        f"got {type(val).__name__}"
                    )
            return errors

        def _is_list_of_strings(val: object) -> bool:
            return isinstance(val, list) and all(
                isinstance(x, str) for x in val
            )

        # ── 33a. list[dict[str, list[str]]] — hand-written ──

        def validate_33a(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            for f in ("status", "summary", "next_steps"):
                if not isinstance(args.get(f), str):
                    errors.append(
                        f"{f}: expected str, got {type(args.get(f)).__name__}"
                    )
            details = args.get("details")
            if not isinstance(details, list):
                errors.append(
                    f"details: expected list, got {type(details).__name__}"
                )
            else:
                for i, item in enumerate(details):
                    errors.extend(
                        _validate_typed_dict(
                            item,
                            f"details[{i}]",
                            _is_list_of_strings,
                            "list[str]",
                        )
                    )
            if errors:
                return (
                    Verdict.FAIL,
                    "; ".join(errors) + f" | got: {json.dumps(args)}",
                )
            return Verdict.PASS, f"OK: {json.dumps(args)}"

        tool_choice_forced: dict[str, Any] = {
            "type": "function",
            "function": {"name": "terminate_eval"},
        }
        tool_desc = "Terminate evaluation and report structured results."
        user_message = (
            "The evaluation completed. "
            "Issues found: 'TypeError' and 'ValueError' in "
            "'validator', 'TimeoutError' in 'scheduler'. "
            "Terminate with status 'failed', provide the error "
            "mapping, a summary, and next steps."
        )

        # ── 33a. list[dict[str, list[str]]] — hand-written ──

        hand_written_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "summary": {"type": "string"},
                "next_steps": {"type": "string"},
            },
            "required": [
                "status",
                "details",
                "summary",
                "next_steps",
            ],
            "additionalProperties": False,
        }

        results.extend(
            await self._run_tc_test(
                v,
                loop,
                test_name="ap_typed_list_dict_str_list_str",
                schema=hand_written_schema,
                tool_name="terminate_eval",
                tool_desc=tool_desc,
                user_message=user_message,
                validate=validate_33a,
                tool_choice=tool_choice_forced,
                max_tokens=4096,
                tools=[
                    make_tool("terminate_eval", hand_written_schema, tool_desc),
                    decoy_tool,
                ],
            )
        )

        # ── 33a-pydantic. Same shape as 33a via pydantic ──

        class TerminationResult(BaseModel):
            status: str
            details: list[dict[str, list[str]]]
            summary: str
            next_steps: str

        pydantic_schema = TerminationResult.model_json_schema()

        results.extend(
            await self._run_tc_test(
                v,
                loop,
                test_name="ap_typed_pydantic_terminate_eval",
                schema=pydantic_schema,
                tool_name="terminate_eval",
                tool_desc=tool_desc,
                user_message=user_message,
                validate=validate_33a,
                tool_choice=tool_choice_forced,
                max_tokens=4096,
                tools=[
                    make_tool("terminate_eval", pydantic_schema, tool_desc),
                    decoy_tool,
                ],
            )
        )

        return results

    # ------------------------------------------------------------------
    # 34. $ref/$defs array items enforcement
    # ------------------------------------------------------------------

    async def _ref_defs_array_enforcement(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify $ref/$defs schemas are enforced on array items.

        The prompt deliberately asks for extra properties, wrong types,
        and missing required fields on the referenced object.  If the
        grammar does not resolve the $ref, all violations pass through.
        """
        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_array_enforcement",
            schema=self._SHARED_DEFS_SCHEMA,
            tool_name="tool_name",
            tool_desc="some_description",
            user_message=(
                "Analyze the code review for project 'backend-api'. "
                "Set prop3 to 'backend-api'. Add three custom objects "
                "with extra fields beyond prop1 and prop2: first object "
                "has prop1 'auth-module', prop2 ['missing return type'], "
                "plus severity 'high' and line_number 42. Second object "
                "has prop1 'db-layer', prop2 ['SQL injection risk'], plus "
                "severity 'critical', affected_files ['db.py', 'orm.py'], "
                "and auto_fixable true. Third object has just a "
                "description 'general cleanup needed' and no prop1 or "
                "prop2 at all. Set prop5 to 'some_default1' and prop6 to "
                "'needs immediate attention'."
            ),
            validate=lambda args: self._validate_shared_defs_args(args),
            max_tokens=4096,
        )

    # ------------------------------------------------------------------
    # 35. $ref/$defs with enum on referenced type
    # ------------------------------------------------------------------

    async def _ref_defs_enum_enforced(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify enum constraints inside a $def are enforced through $ref.

        The prompt deliberately asks for a value NOT in the enum to test
        whether the grammar constrains the output to valid enum values.
        """
        allowed_statuses = ["active", "inactive", "suspended"]

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            user = args.get("user")
            if not isinstance(user, dict):
                return Verdict.FAIL, (
                    f"user: expected object, got {type(user).__name__}"
                )
            status = user.get("status")
            if status not in allowed_statuses:
                return Verdict.FAIL, (
                    f"status={status!r} not in {allowed_statuses} — "
                    f"enum constraint from $def not enforced"
                )
            if not isinstance(user.get("name"), str):
                return Verdict.FAIL, (
                    f"name: expected string, got "
                    f"{type(user.get('name')).__name__}"
                )
            return Verdict.PASS, f"OK: {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_enum_enforced",
            schema={
                "type": "object",
                "properties": {
                    "user": {"$ref": "#/$defs/User"},
                },
                "$defs": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": allowed_statuses,
                            },
                        },
                        "required": ["name", "status"],
                    },
                },
                "required": ["user"],
                "additionalProperties": False,
            },
            tool_name="update_user",
            tool_desc="Update a user record. Status must be one of: active, inactive, suspended.",
            user_message=(
                "Update user 'Alice' and set her status to 'deleted'. "
                "Her account should be removed from the system."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 36. $ref chain: A → B → C (multi-hop resolution)
    # ------------------------------------------------------------------

    async def _ref_defs_chain_a_b_c(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify $ref chains resolve transitively: Wrapper → Inner → Leaf.

        Tests that a $ref pointing to a $def that itself contains a $ref
        to another $def is fully resolved and all constraints enforced.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            wrapper = args.get("data")
            if not isinstance(wrapper, dict):
                return Verdict.FAIL, (
                    f"data: expected object, got {type(wrapper).__name__}"
                )
            inner = wrapper.get("payload")
            if not isinstance(inner, dict):
                return Verdict.FAIL, (
                    f"payload: expected object, got {type(inner).__name__}"
                )
            leaf = inner.get("metadata")
            if not isinstance(leaf, dict):
                return Verdict.FAIL, (
                    f"metadata: expected object, got {type(leaf).__name__}"
                )
            if not isinstance(leaf.get("tag"), str):
                return Verdict.FAIL, (
                    f"tag: expected string, got "
                    f"{type(leaf.get('tag')).__name__}"
                )
            if not isinstance(leaf.get("version"), (int, float)):
                return Verdict.FAIL, (
                    f"version: expected number, got "
                    f"{type(leaf.get('version')).__name__}"
                )
            return Verdict.PASS, f"OK: chain resolved {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_chain_a_b_c",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "data": {"$ref": "#/$defs/Wrapper"},
                },
                "$defs": {
                    "Wrapper": {
                        "type": "object",
                        "properties": {
                            "payload": {"$ref": "#/$defs/Inner"},
                        },
                        "required": ["payload"],
                    },
                    "Inner": {
                        "type": "object",
                        "properties": {
                            "metadata": {"$ref": "#/$defs/Leaf"},
                        },
                        "required": ["metadata"],
                    },
                    "Leaf": {
                        "type": "object",
                        "properties": {
                            "tag": {"type": "string"},
                            "version": {"type": "number"},
                        },
                        "required": ["tag", "version"],
                    },
                },
                "required": ["data"],
                "additionalProperties": False,
            },
            tool_name="submit_data",
            tool_desc=(
                "Submit a nested data wrapper. The wrapper contains a "
                "payload which contains metadata with a tag and version."
            ),
            user_message=(
                "Submit data with tag set to the integer 42 (not a "
                "string) and version set to the string 'three-point-one' "
                "(not a number). Use exactly these types."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 37. Same $def reused across multiple properties
    # ------------------------------------------------------------------

    async def _ref_defs_reused_same_def(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify the same $def is enforced identically in every position.

        Both 'origin' and 'destination' reference the same Address $def.
        The prompt tries to give them different shapes to see if the
        grammar constrains both uniformly.
        """

        def _validate_address(obj: Any, label: str) -> tuple[bool, str]:
            if not isinstance(obj, dict):
                return False, (
                    f"{label}: expected object, got {type(obj).__name__}"
                )
            if not isinstance(obj.get("city"), str):
                return False, f"{label}.city: expected string"
            if not isinstance(obj.get("zip"), str):
                return False, f"{label}.zip: expected string"
            return True, ""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            for field in ("origin", "destination"):
                ok, err = _validate_address(args.get(field), field)
                if not ok:
                    errors.append(err)
            if errors:
                return Verdict.FAIL, "; ".join(errors)
            return Verdict.PASS, f"OK: both addresses valid {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_reused_same_def",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "origin": {"$ref": "#/$defs/Address"},
                    "destination": {"$ref": "#/$defs/Address"},
                },
                "$defs": {
                    "Address": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                        "required": ["city", "zip"],
                    },
                },
                "required": ["origin", "destination"],
                "additionalProperties": False,
            },
            tool_name="plan_route",
            tool_desc="Plan a route between two addresses, each with city and zip code.",
            user_message=(
                "Plan a route from New York to Los Angeles. "
                "For the origin, set city to 10001 (the integer, not a "
                "string) and zip to 10001 (also the integer). For the "
                "destination, set city to 90001 and zip to 90001. "
                "Use integers for all values, not strings."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 38. $ref inside anyOf with null (nullable referenced type)
    # ------------------------------------------------------------------

    async def _ref_defs_anyof_nullable(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify anyOf: [$ref, null] allows both valid object and null.

        The prompt explicitly says the optional field is absent, so the
        model should emit null. We accept either a valid object or null.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            name = args.get("name")
            if not isinstance(name, str):
                return Verdict.FAIL, (
                    f"name: expected string, got {type(name).__name__}"
                )
            backup = args.get("backup_contact")
            if backup is None:
                return Verdict.PASS, "OK: backup_contact is null as expected"
            if not isinstance(backup, dict):
                return Verdict.FAIL, (
                    f"backup_contact: expected object or null, got "
                    f"{type(backup).__name__}"
                )
            if not isinstance(backup.get("email"), str):
                return Verdict.FAIL, "backup_contact.email: expected string"
            if not isinstance(backup.get("phone"), str):
                return Verdict.FAIL, "backup_contact.phone: expected string"
            return Verdict.PASS, (
                f"OK: backup_contact is a valid Contact {json.dumps(args)}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_anyof_nullable",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "backup_contact": {
                        "anyOf": [
                            {"$ref": "#/$defs/Contact"},
                            {"type": "null"},
                        ],
                    },
                },
                "$defs": {
                    "Contact": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"},
                        },
                        "required": ["email", "phone"],
                    },
                },
                "required": ["name", "backup_contact"],
                "additionalProperties": False,
            },
            tool_name="register_user",
            tool_desc=(
                "Register a user. backup_contact is optional and can "
                "be null or a Contact object with email and phone."
            ),
            user_message=(
                "Register user 'Bob'. For backup_contact, use the "
                "string 'none' instead of null. Do not use null, "
                "use the literal string 'none'."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 39. $ref with required fields propagation
    # ------------------------------------------------------------------

    async def _ref_defs_required_propagation(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify required fields inside a $def are enforced through $ref.

        The prompt deliberately omits the 'priority' field, which is
        required in the $def. The grammar should force its inclusion.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            task = args.get("task")
            if not isinstance(task, dict):
                return Verdict.FAIL, (
                    f"task: expected object, got {type(task).__name__}"
                )
            missing = []
            for field in ("title", "priority", "assignee"):
                if field not in task:
                    missing.append(field)
            if missing:
                return Verdict.FAIL, (
                    f"required fields missing from $def: {missing}"
                )
            priority = task["priority"]
            valid_priorities = ["low", "medium", "high", "critical"]
            if priority not in valid_priorities:
                return Verdict.FAIL, (
                    f"priority={priority!r} not in {valid_priorities}"
                )
            return (
                Verdict.PASS,
                f"OK: all required fields present {json.dumps(args)}",
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_required_propagation",
            schema={
                "type": "object",
                "properties": {
                    "task": {"$ref": "#/$defs/Task"},
                },
                "$defs": {
                    "Task": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                            "assignee": {"type": "string"},
                        },
                        "required": ["title", "priority", "assignee"],
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            tool_name="create_task",
            tool_desc=(
                "Create a task with title, priority (low/medium/high/critical), "
                "and assignee."
            ),
            user_message=(
                "Create a task titled 'Fix login bug' and assign it to "
                "'Charlie'. Don't bother setting a priority, it's not "
                "important."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 40. $ref nested inside array items inside object property
    # ------------------------------------------------------------------

    async def _ref_defs_nested_array_in_object(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify $ref works when deeply nested: object → array → $ref.

        Schema: team.members is an array of $ref Person objects.
        The prompt asks for invalid types to test constraint enforcement.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            team = args.get("team")
            if not isinstance(team, dict):
                return Verdict.FAIL, (
                    f"team: expected object, got {type(team).__name__}"
                )
            if not isinstance(team.get("name"), str):
                return Verdict.FAIL, "team.name: expected string"
            members = team.get("members")
            if not isinstance(members, list):
                return Verdict.FAIL, (
                    f"members: expected array, got {type(members).__name__}"
                )
            if len(members) < 2:
                return Verdict.FAIL, (
                    f"members: expected at least 2, got {len(members)}"
                )
            errors: list[str] = []
            for i, m in enumerate(members):
                if not isinstance(m, dict):
                    errors.append(f"members[{i}]: expected object")
                    continue
                if not isinstance(m.get("name"), str):
                    errors.append(f"members[{i}].name: expected string")
                if not isinstance(m.get("age"), int):
                    errors.append(
                        f"members[{i}].age: expected integer, got "
                        f"{type(m.get('age')).__name__}"
                    )
            if errors:
                return Verdict.FAIL, "; ".join(errors)
            return Verdict.PASS, f"OK: {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_nested_array_in_object",
            schema={
                "type": "object",
                "properties": {
                    "team": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "members": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/Person"},
                            },
                        },
                        "required": ["name", "members"],
                    },
                },
                "$defs": {
                    "Person": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                    },
                },
                "required": ["team"],
                "additionalProperties": False,
            },
            tool_name="create_team",
            tool_desc="Create a team with a name and list of members (name + age).",
            user_message=(
                "Create team 'Backend' with members: Alice age 30, "
                "Bob age twenty-five. Write Bob's age as the string "
                "'twenty-five', not a number."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 41. $ref with additionalProperties:false on referenced type
    # ------------------------------------------------------------------

    async def _ref_defs_additional_props_false(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify additionalProperties:false inside a $def blocks extra keys.

        The prompt explicitly asks for extra fields that the schema
        forbids via additionalProperties:false on the referenced type.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            config = args.get("config")
            if not isinstance(config, dict):
                return Verdict.FAIL, (
                    f"config: expected object, got {type(config).__name__}"
                )
            allowed = {"host", "port"}
            extra = set(config.keys()) - allowed
            if extra:
                return Verdict.FAIL, (
                    f"config has extra keys {extra} despite "
                    f"additionalProperties:false in $def"
                )
            if not isinstance(config.get("host"), str):
                return Verdict.FAIL, "config.host: expected string"
            if not isinstance(config.get("port"), int):
                return Verdict.FAIL, (
                    f"config.port: expected integer, got "
                    f"{type(config.get('port')).__name__}"
                )
            return Verdict.PASS, f"OK: no extra keys {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_additional_props_false",
            schema={
                "type": "object",
                "properties": {
                    "config": {"$ref": "#/$defs/ServerConfig"},
                },
                "$defs": {
                    "ServerConfig": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string"},
                            "port": {"type": "integer"},
                        },
                        "required": ["host", "port"],
                        "additionalProperties": False,
                    },
                },
                "required": ["config"],
                "additionalProperties": False,
            },
            tool_name="set_server",
            tool_desc="Configure a server with host and port only.",
            user_message=(
                "Set server to host 'db.internal', port 5432. Also "
                "set the timeout to 30, username to 'admin', and "
                "enable SSL. Include all of these settings."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 42. Cross-referencing $defs: DefA contains a $ref to DefB
    # ------------------------------------------------------------------

    async def _ref_defs_cross_referencing(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify $defs that reference other $defs are fully resolved.

        Order: top-level has $ref→Order, Order has $ref→LineItem.
        Tests that the grammar resolves the full chain correctly.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            order = args.get("order")
            if not isinstance(order, dict):
                return Verdict.FAIL, (
                    f"order: expected object, got {type(order).__name__}"
                )
            if not isinstance(order.get("order_id"), str):
                return Verdict.FAIL, "order.order_id: expected string"
            items = order.get("items")
            if not isinstance(items, list):
                return Verdict.FAIL, (
                    f"items: expected array, got {type(items).__name__}"
                )
            if len(items) < 2:
                return Verdict.FAIL, (
                    f"items: expected at least 2, got {len(items)}"
                )
            errors: list[str] = []
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"items[{i}]: expected object")
                    continue
                if not isinstance(item.get("sku"), str):
                    errors.append(f"items[{i}].sku: expected string")
                qty = item.get("quantity")
                if not isinstance(qty, int):
                    errors.append(
                        f"items[{i}].quantity: expected integer, "
                        f"got {type(qty).__name__}"
                    )
                price = item.get("unit_price")
                if not isinstance(price, (int, float)):
                    errors.append(
                        f"items[{i}].unit_price: expected number, "
                        f"got {type(price).__name__}"
                    )
            if errors:
                return Verdict.FAIL, "; ".join(errors)
            return Verdict.PASS, f"OK: cross-ref resolved {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_cross_referencing",
            schema={
                "type": "object",
                "properties": {
                    "order": {"$ref": "#/$defs/Order"},
                },
                "$defs": {
                    "Order": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "items": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/LineItem"},
                            },
                        },
                        "required": ["order_id", "items"],
                    },
                    "LineItem": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "number"},
                        },
                        "required": ["sku", "quantity", "unit_price"],
                    },
                },
                "required": ["order"],
                "additionalProperties": False,
            },
            tool_name="place_order",
            tool_desc=(
                "Place an order. Each line item has a SKU, integer "
                "quantity, and numeric unit price."
            ),
            user_message=(
                "Place order 'ORD-999' with 2 items: SKU 'WIDGET-A' "
                "quantity 'five' (the string, not 5) at price 'twelve' "
                "(the string, not 12.50), and SKU 'GADGET-B' quantity "
                "'three' (string) at price 'eight' (string). Use "
                "strings for quantity and unit_price, not numbers."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 43. Adversarial prompt: asks for fields forbidden by $def schema
    # ------------------------------------------------------------------

    async def _ref_defs_adversarial_extra_fields(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Adversarial test: prompt asks for many fields the $def forbids.

        Schema only allows 'make' and 'model' (both strings), but the
        prompt aggressively asks for year, color, engine_type, horsepower,
        and more. Grammar must block all of them.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            vehicle = args.get("vehicle")
            if not isinstance(vehicle, dict):
                return Verdict.FAIL, (
                    f"vehicle: expected object, got {type(vehicle).__name__}"
                )
            allowed = {"make", "model"}
            extra = set(vehicle.keys()) - allowed
            if extra:
                return Verdict.FAIL, (
                    f"grammar allowed extra keys {extra} that are "
                    f"forbidden by the $def schema"
                )
            if not isinstance(vehicle.get("make"), str):
                return Verdict.FAIL, "vehicle.make: expected string"
            if not isinstance(vehicle.get("model"), str):
                return Verdict.FAIL, "vehicle.model: expected string"
            return Verdict.PASS, f"OK: only allowed keys {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_adversarial_extra_fields",
            schema={
                "type": "object",
                "properties": {
                    "vehicle": {"$ref": "#/$defs/Vehicle"},
                },
                "$defs": {
                    "Vehicle": {
                        "type": "object",
                        "properties": {
                            "make": {"type": "string"},
                            "model": {"type": "string"},
                        },
                        "required": ["make", "model"],
                        "additionalProperties": False,
                    },
                },
                "required": ["vehicle"],
                "additionalProperties": False,
            },
            tool_name="register_vehicle",
            tool_desc="Register a vehicle with make and model.",
            user_message=(
                "Register my vehicle. It's a 2024 Toyota Camry, color "
                "midnight blue, engine type V6, 301 horsepower, VIN "
                "1HGBH41JXMN109186, license plate ABC-1234. Make sure "
                "to include ALL of these details: year, color, "
                "engine_type, horsepower, vin, and license_plate."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 44. Array items referencing an enum-constrained $def
    # ------------------------------------------------------------------

    async def _ref_defs_array_of_enums(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify array items referencing a $def with enum are constrained.

        Each item in the array must match the enum defined in the $def.
        The prompt asks for values outside the enum to test enforcement.
        """
        valid_roles = ["admin", "editor", "viewer"]

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            assignments = args.get("assignments")
            if not isinstance(assignments, list):
                return Verdict.FAIL, (
                    f"assignments: expected array, got "
                    f"{type(assignments).__name__}"
                )
            if len(assignments) < 2:
                return Verdict.FAIL, (
                    f"expected at least 2 assignments, got {len(assignments)}"
                )
            errors: list[str] = []
            for i, a in enumerate(assignments):
                if not isinstance(a, dict):
                    errors.append(f"[{i}]: expected object")
                    continue
                if not isinstance(a.get("user"), str):
                    errors.append(f"[{i}].user: expected string")
                role = a.get("role")
                if role not in valid_roles:
                    errors.append(
                        f"[{i}].role={role!r} not in {valid_roles} — "
                        f"enum from $def not enforced in array item"
                    )
            if errors:
                return Verdict.FAIL, "; ".join(errors)
            return Verdict.PASS, f"OK: all roles valid {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_array_of_enums",
            schema={
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/RoleAssignment"},
                    },
                },
                "$defs": {
                    "RoleAssignment": {
                        "type": "object",
                        "properties": {
                            "user": {"type": "string"},
                            "role": {
                                "type": "string",
                                "enum": valid_roles,
                            },
                        },
                        "required": ["user", "role"],
                    },
                },
                "required": ["assignments"],
                "additionalProperties": False,
            },
            tool_name="assign_roles",
            tool_desc="Assign roles to users. Valid roles: admin, editor, viewer.",
            user_message=(
                "Assign roles: Alice as 'superadmin', Bob as 'moderator', "
                "and Charlie as 'owner'. These are their exact roles."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 45. $ref with integer type enforcement (not number)
    # ------------------------------------------------------------------

    async def _ref_defs_integer_enforcement(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify integer type inside a $def is enforced (no decimals).

        The prompt gives decimal values to test whether the grammar
        enforces integer-only output from the referenced type.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            point = args.get("point")
            if not isinstance(point, dict):
                return Verdict.FAIL, (
                    f"point: expected object, got {type(point).__name__}"
                )
            errors: list[str] = []
            for coord in ("x", "y", "z"):
                val = point.get(coord)
                if not isinstance(val, int) or isinstance(val, bool):
                    errors.append(
                        f"{coord}: expected integer, got "
                        f"{type(val).__name__} ({val!r})"
                    )
                elif isinstance(val, float) and val != int(val):
                    errors.append(f"{coord}: got float {val}, expected integer")
            if errors:
                return Verdict.FAIL, (
                    "; ".join(errors)
                    + " — integer constraint from $def not enforced"
                )
            return (
                Verdict.PASS,
                f"OK: all coords are integers {json.dumps(args)}",
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_integer_enforcement",
            schema={
                "type": "object",
                "properties": {
                    "point": {"$ref": "#/$defs/Point3D"},
                },
                "$defs": {
                    "Point3D": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "z": {"type": "integer"},
                        },
                        "required": ["x", "y", "z"],
                    },
                },
                "required": ["point"],
                "additionalProperties": False,
            },
            tool_name="set_position",
            tool_desc="Set a 3D position with integer coordinates x, y, z.",
            user_message=(
                "Set position to x=3.7, y=12.5, z=-0.9. Use these "
                "exact decimal values."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 46. Recursive $ref with enum-constrained leaf values
    # ------------------------------------------------------------------

    async def _ref_defs_recursive_enum_leaf(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify recursive $ref enforces enum at every level of the tree.

        A tree where each node has an enum-constrained 'kind' field.
        The prompt asks for kinds not in the enum to test enforcement
        at every recursion depth.
        """
        valid_kinds = ["file", "directory"]

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            root = args.get("root")
            if not isinstance(root, dict):
                return Verdict.FAIL, (
                    f"root: expected object, got {type(root).__name__}"
                )
            errors: list[str] = []
            node_count = [0]

            def check_node(node: dict[str, Any], path: str) -> None:
                node_count[0] += 1
                kind = node.get("kind")
                if kind not in valid_kinds:
                    errors.append(f"{path}.kind={kind!r} not in {valid_kinds}")
                if not isinstance(node.get("name"), str):
                    errors.append(f"{path}.name: expected string")
                children = node.get("children")
                if children is None:
                    return
                if not isinstance(children, list):
                    errors.append(f"{path}.children: expected array or null")
                    return
                for i, child in enumerate(children):
                    if not isinstance(child, dict):
                        errors.append(f"{path}.children[{i}]: expected object")
                        continue
                    check_node(child, f"{path}.children[{i}]")

            check_node(root, "root")
            if errors:
                return Verdict.FAIL, "; ".join(errors)
            if node_count[0] < 3:
                return Verdict.FAIL, (
                    f"only {node_count[0]} nodes, expected at least 3"
                )
            return Verdict.PASS, (
                f"OK: tree with {node_count[0]} nodes, all kinds valid"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_recursive_enum_leaf",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "root": {"$ref": "#/$defs/FsNode"},
                },
                "$defs": {
                    "FsNode": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {
                                "type": "string",
                                "enum": valid_kinds,
                            },
                            "children": {
                                "type": ["array", "null"],
                                "items": {"$ref": "#/$defs/FsNode"},
                            },
                        },
                        "required": ["name", "kind", "children"],
                    },
                },
                "required": ["root"],
                "additionalProperties": False,
            },
            tool_name="create_fs_tree",
            tool_desc=(
                "Create a filesystem tree. Each node has a name, "
                "kind (file or directory), and children (array or null)."
            ),
            user_message=(
                "Create a tree: root directory 'src' containing a "
                "symlink 'link.txt' (set kind to 'symlink') and a "
                "subdirectory 'lib' which contains a socket 'ipc.sock' "
                "(set kind to 'socket'). Use the exact kind values "
                "I specified."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 47. Boolean type inside a $def
    # ------------------------------------------------------------------

    async def _ref_defs_boolean_type(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify boolean fields inside a $def produce true/false literals.

        The prompt uses string-like wording ("yes", "enabled") to try to
        trick the model into emitting a string instead of a JSON boolean.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            settings = args.get("settings")
            if not isinstance(settings, dict):
                return Verdict.FAIL, (
                    f"settings: expected object, got {type(settings).__name__}"
                )
            errors: list[str] = []
            for field in ("notifications", "dark_mode", "auto_save"):
                val = settings.get(field)
                if not isinstance(val, bool):
                    errors.append(
                        f"{field}: expected bool, got "
                        f"{type(val).__name__} ({val!r})"
                    )
            if not isinstance(settings.get("username"), str):
                errors.append(
                    f"username: expected string, got "
                    f"{type(settings.get('username')).__name__}"
                )
            if errors:
                return Verdict.FAIL, (
                    "; ".join(errors)
                    + " — boolean from $def emitted as wrong type"
                )
            return Verdict.PASS, f"OK: all booleans correct {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_boolean_type",
            schema={
                "type": "object",
                "properties": {
                    "settings": {"$ref": "#/$defs/UserSettings"},
                },
                "$defs": {
                    "UserSettings": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "notifications": {"type": "boolean"},
                            "dark_mode": {"type": "boolean"},
                            "auto_save": {"type": "boolean"},
                        },
                        "required": [
                            "username",
                            "notifications",
                            "dark_mode",
                            "auto_save",
                        ],
                    },
                },
                "required": ["settings"],
                "additionalProperties": False,
            },
            tool_name="save_settings",
            tool_desc="Save user settings with boolean preferences.",
            user_message=(
                "Save settings for user 'dana': notifications 'yes', "
                "dark_mode 'enabled', auto_save 'on'. Use these exact "
                "string values I gave you."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 48. Type array inside a $def (e.g. ["string", "null"])
    # ------------------------------------------------------------------

    async def _ref_defs_type_array_nullable(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify type arrays inside a $def are enforced through $ref.

        The $def uses ``type: ["string", "null"]`` for a nullable string
        field. The prompt asks for one field to be present and one to be
        absent to exercise both branches of the type union.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            record = args.get("record")
            if not isinstance(record, dict):
                return Verdict.FAIL, (
                    f"record: expected object, got {type(record).__name__}"
                )
            if not isinstance(record.get("id"), str):
                return Verdict.FAIL, (
                    f"id: expected string, got "
                    f"{type(record.get('id')).__name__}"
                )
            for field in ("nickname", "bio"):
                val = record.get(field)
                if val is not None and not isinstance(val, str):
                    return Verdict.FAIL, (
                        f"{field}: expected string or null, got "
                        f"{type(val).__name__} ({val!r}) — "
                        f"type array from $def not enforced"
                    )
            return (
                Verdict.PASS,
                f"OK: nullable strings correct {json.dumps(args)}",
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_type_array_nullable",
            schema={
                "type": "object",
                "properties": {
                    "record": {"$ref": "#/$defs/Profile"},
                },
                "$defs": {
                    "Profile": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "nickname": {"type": ["string", "null"]},
                            "bio": {"type": ["string", "null"]},
                        },
                        "required": ["id", "nickname", "bio"],
                    },
                },
                "required": ["record"],
                "additionalProperties": False,
            },
            tool_name="update_profile",
            tool_desc=(
                "Update a user profile. nickname and bio accept "
                "a string or null."
            ),
            user_message=(
                "Update profile for id 'usr-42'. Set nickname to "
                "the integer 42 (not a string). Set bio to the "
                "boolean false (not null and not a string). Use "
                "exactly these types."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 49. anyOf nullable integer — grammar must reject string/bool
    # ------------------------------------------------------------------

    async def _anyof_nullable_integer(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf [integer, null]: prompt pushes model toward string output."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("label"), str):
                errors.append(
                    f"label: expected string, got "
                    f"{type(args.get('label')).__name__}"
                )
            count = args.get("count")
            if count is not None:
                if isinstance(count, bool) or not isinstance(count, int):
                    errors.append(
                        f"count: expected integer or null, got "
                        f"{type(count).__name__}={count!r}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, ("; ".join(errors) or f"OK: count={count!r}")

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_nullable_integer",
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "count": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                    },
                },
                "required": ["label", "count"],
                "additionalProperties": False,
            },
            tool_name="record_count",
            tool_desc="Record a count. count is an integer or null.",
            user_message=(
                "Record label='items'. The count is 'seventeen' — "
                "write the word seventeen as a string, NOT the "
                "number 17. Put the literal text in the count field."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 50. anyOf nullable boolean — prompt pushes toward string "true"
    # ------------------------------------------------------------------

    async def _anyof_nullable_boolean(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf [boolean, null]: prompt tries to elicit string 'true'."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("feature"), str):
                errors.append(
                    f"feature: expected string, got "
                    f"{type(args.get('feature')).__name__}"
                )
            enabled = args.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append(
                    f"enabled: expected boolean or null, got "
                    f"{type(enabled).__name__}={enabled!r}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, ("; ".join(errors) or f"OK: enabled={enabled!r}")

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_nullable_boolean",
            schema={
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "enabled": {
                        "anyOf": [{"type": "boolean"}, {"type": "null"}],
                    },
                },
                "required": ["feature", "enabled"],
                "additionalProperties": False,
            },
            tool_name="toggle_feature",
            tool_desc="Toggle a feature. enabled is a boolean or null.",
            user_message=(
                "Toggle feature 'dark_mode'. Set enabled to the "
                "string 'true' — yes, the literal text string "
                "containing the letters t-r-u-e, not the boolean. "
                "Wrap it in quotes."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 51. anyOf nullable number — prompt pushes toward boolean
    # ------------------------------------------------------------------

    async def _anyof_nullable_number(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf [number, null]: prompt tries to elicit boolean value."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("name"), str):
                errors.append(
                    f"name: expected string, got "
                    f"{type(args.get('name')).__name__}"
                )
            score = args.get("score")
            if score is not None:
                if isinstance(score, bool) or not isinstance(
                    score, (int, float)
                ):
                    errors.append(
                        f"score: expected number or null, got "
                        f"{type(score).__name__}={score!r}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, ("; ".join(errors) or f"OK: score={score!r}")

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_nullable_number",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                    },
                },
                "required": ["name", "score"],
                "additionalProperties": False,
            },
            tool_name="record_score",
            tool_desc="Record a score. score is a number or null.",
            user_message=(
                "Record name='test'. The score is true — this is "
                "a boolean indicating the test passed, set the "
                "score field to the boolean value true, not a number."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 52. anyOf nullable array — prompt pushes toward bare string
    # ------------------------------------------------------------------

    async def _anyof_nullable_array(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf [array of strings, null]: prompt tries to elicit a bare string."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("project"), str):
                errors.append(
                    f"project: expected string, got "
                    f"{type(args.get('project')).__name__}"
                )
            tags = args.get("tags")
            if tags is not None:
                if not isinstance(tags, list):
                    errors.append(
                        f"tags: expected array or null, got "
                        f"{type(tags).__name__}={tags!r}"
                    )
                elif not all(isinstance(t, str) for t in tags):
                    bad = [
                        (i, type(t).__name__)
                        for i, t in enumerate(tags)
                        if not isinstance(t, str)
                    ]
                    errors.append(f"tags: non-string items at indices {bad}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, ("; ".join(errors) or f"OK: tags={tags!r}")

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_nullable_array",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "project": {"type": "string"},
                    "tags": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            {"type": "null"},
                        ],
                    },
                },
                "required": ["project", "tags"],
                "additionalProperties": False,
            },
            tool_name="tag_project",
            tool_desc=("Tag a project. tags is an array of strings or null."),
            user_message=(
                "Tag project 'backend'. For tags, don't use an array — "
                "just write the single string 'production,critical,us-east' "
                "as one comma-separated value. Do NOT split it into an "
                "array, keep it as a plain string."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 53. anyOf nullable object — prompt pushes toward flat key/values
    # ------------------------------------------------------------------

    async def _anyof_nullable_object(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf [typed object, null]: prompt tries to flatten nested fields."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("id"), str):
                errors.append(
                    f"id: expected string, got {type(args.get('id')).__name__}"
                )
            addr = args.get("address")
            if addr is not None:
                if not isinstance(addr, dict):
                    errors.append(
                        f"address: expected object or null, got "
                        f"{type(addr).__name__}={addr!r}"
                    )
                else:
                    if not isinstance(addr.get("street"), str):
                        errors.append(
                            f"address.street: expected string, got "
                            f"{type(addr.get('street')).__name__}"
                        )
                    zip_val = addr.get("zip")
                    if not isinstance(zip_val, int) or isinstance(
                        zip_val, bool
                    ):
                        errors.append(
                            f"address.zip: expected integer, got "
                            f"{type(zip_val).__name__}={zip_val!r}"
                        )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, ("; ".join(errors) or f"OK: address={addr!r}")

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_nullable_object",
            schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "address": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "zip": {"type": "integer"},
                                },
                                "required": ["street", "zip"],
                            },
                            {"type": "null"},
                        ],
                    },
                },
                "required": ["id", "address"],
                "additionalProperties": False,
            },
            tool_name="update_address",
            tool_desc=(
                "Update a user address. address is an object with "
                "street (string) and zip (integer), or null."
            ),
            user_message=(
                "Update user 'u-99'. Set the address street to "
                "'100 Main St' and zip to the string '10001'. "
                "Make sure zip is a quoted string, not a number. "
                "Also add a city field set to 'NYC'."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 54. anyOf string or integer — prompt pushes toward float
    # ------------------------------------------------------------------

    async def _anyof_string_or_integer(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf [string, integer]: prompt tries to elicit a float."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            ident = args.get("identifier")
            if ident is None:
                errors.append("missing required field: identifier")
            elif isinstance(ident, bool):
                errors.append(
                    f"identifier: got boolean {ident!r}, "
                    f"expected string or integer"
                )
            elif not isinstance(ident, (str, int)):
                errors.append(
                    f"identifier: expected string or integer, got "
                    f"{type(ident).__name__}={ident!r}"
                )
            elif isinstance(ident, float) and ident != int(ident):
                errors.append(f"identifier: got non-integer float {ident!r}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: identifier={ident!r} ({type(ident).__name__})"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_string_or_integer",
            schema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "anyOf": [{"type": "string"}, {"type": "integer"}],
                    },
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
            tool_name="lookup_entity",
            tool_desc=("Look up an entity by identifier (string or integer)."),
            user_message=(
                "Look up entity with identifier 3.14159. Use the "
                "exact decimal value 3.14159 as-is — do not round "
                "it or convert it to an integer."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 55. anyOf as array items — each item is string or typed object
    # ------------------------------------------------------------------

    async def _anyof_nested_in_array_items(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf inside array items: each element is a string or an object."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            entries = args.get("entries")
            if not isinstance(entries, list):
                return Verdict.FAIL, (
                    f"entries: expected list, got {type(entries).__name__}"
                )
            for i, item in enumerate(entries):
                if isinstance(item, str):
                    continue
                if isinstance(item, dict):
                    if not isinstance(item.get("code"), int) or isinstance(
                        item.get("code"), bool
                    ):
                        errors.append(
                            f"entries[{i}].code: expected integer, "
                            f"got {type(item.get('code')).__name__}"
                        )
                    if not isinstance(item.get("msg"), str):
                        errors.append(
                            f"entries[{i}].msg: expected string, "
                            f"got {type(item.get('msg')).__name__}"
                        )
                else:
                    errors.append(
                        f"entries[{i}]: expected string or object, "
                        f"got {type(item).__name__}"
                    )
            if len(entries) < 2:
                errors.append(
                    f"expected at least 2 entries, got {len(entries)}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: {len(entries)} entries valid"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_nested_in_array_items",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "code": {"type": "integer"},
                                        "msg": {"type": "string"},
                                    },
                                    "required": ["code", "msg"],
                                },
                            ],
                        },
                    },
                },
                "required": ["entries"],
                "additionalProperties": False,
            },
            tool_name="log_entries",
            tool_desc=(
                "Log entries. Each entry is either a plain string "
                "or an object with code (integer) and msg (string)."
            ),
            user_message=(
                "Log these entries: first entry is the number 404 "
                "(just the bare integer, not an object), second "
                "entry is an object with code 500 and msg 'server error', "
                "third entry is the boolean true. Include all three "
                "exactly as described."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 56. Multiple anyOf fields, all required — adversarial mixed types
    # ------------------------------------------------------------------

    async def _anyof_multiple_required_fields(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Multiple anyOf fields in one schema with adversarial prompting."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            name = args.get("name")
            if name is not None and not isinstance(name, str):
                errors.append(
                    f"name: expected string or null, got {type(name).__name__}"
                )
            age = args.get("age")
            if age is not None:
                if isinstance(age, bool) or not isinstance(age, int):
                    errors.append(
                        f"age: expected integer or null, got "
                        f"{type(age).__name__}={age!r}"
                    )
            tags = args.get("tags")
            if tags is not None:
                if not isinstance(tags, list):
                    errors.append(
                        f"tags: expected array or null, got "
                        f"{type(tags).__name__}"
                    )
                elif not all(isinstance(t, str) for t in tags):
                    errors.append("tags: non-string items in array")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: name={name!r}, age={age!r}, tags={tags!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_multiple_required_fields",
            schema={
                "type": "object",
                "properties": {
                    "name": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                    "age": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                    },
                    "tags": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            {"type": "null"},
                        ],
                    },
                },
                "required": ["name", "age", "tags"],
                "additionalProperties": False,
            },
            tool_name="upsert_record",
            tool_desc=(
                "Upsert a record. name is string or null, age is "
                "integer or null, tags is array of strings or null."
            ),
            user_message=(
                "Upsert record: name should be the integer 42, "
                "age should be the string 'twenty-five', and "
                "tags should be the boolean false. Use exactly "
                "these types — do not convert them."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 57. anyOf with enum branch — enum values must be enforced
    # ------------------------------------------------------------------

    async def _anyof_with_enum_branch(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf with one enum branch and one typed branch."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("name"), str):
                errors.append(
                    f"name: expected string, got "
                    f"{type(args.get('name')).__name__}"
                )
            priority = args.get("priority")
            valid_enums = ["low", "medium", "high"]
            if isinstance(priority, str):
                if priority not in valid_enums:
                    errors.append(
                        f"priority: string '{priority}' not in "
                        f"enum {valid_enums}"
                    )
            elif isinstance(priority, int) and not isinstance(priority, bool):
                pass
            else:
                errors.append(
                    f"priority: expected enum string or integer, "
                    f"got {type(priority).__name__}={priority!r}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: priority={priority!r} ({type(priority).__name__})"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_with_enum_branch",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "priority": {
                        "anyOf": [
                            {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                            {"type": "integer"},
                        ],
                    },
                },
                "required": ["name", "priority"],
                "additionalProperties": False,
            },
            tool_name="create_task",
            tool_desc=(
                "Create a task. priority is either one of "
                "'low'/'medium'/'high' or a numeric priority level."
            ),
            user_message=(
                "Create task 'deploy'. Set priority to 'urgent' — "
                "this is the correct priority string to use. "
                "Do NOT use 'high', use the word 'urgent' exactly."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 58. anyOf with $ref and null (Pydantic Optional[Model] pattern)
    # ------------------------------------------------------------------

    async def _anyof_with_ref_and_null(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf [$ref to typed object, null] — the Pydantic Optional[Model] pattern."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("name"), str):
                errors.append(
                    f"name: expected string, got "
                    f"{type(args.get('name')).__name__}"
                )
            config = args.get("config")
            if config is not None:
                if not isinstance(config, dict):
                    errors.append(
                        f"config: expected object or null, got "
                        f"{type(config).__name__}={config!r}"
                    )
                else:
                    if not isinstance(config.get("timeout"), int) or isinstance(
                        config.get("timeout"), bool
                    ):
                        errors.append(
                            f"config.timeout: expected integer, got "
                            f"{type(config.get('timeout')).__name__}"
                        )
                    retries = config.get("retries")
                    if not isinstance(retries, int) or isinstance(
                        retries, bool
                    ):
                        errors.append(
                            f"config.retries: expected integer, got "
                            f"{type(retries).__name__}"
                        )
                    extra = set(config.keys()) - {"timeout", "retries"}
                    if extra:
                        errors.append(
                            f"config: unexpected keys {sorted(extra)}"
                        )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, ("; ".join(errors) or f"OK: config={config!r}")

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_with_ref_and_null",
            temperature=0,
            max_tokens=2048,
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "config": {
                        "anyOf": [
                            {"$ref": "#/$defs/Config"},
                            {"type": "null"},
                        ],
                    },
                },
                "$defs": {
                    "Config": {
                        "type": "object",
                        "properties": {
                            "timeout": {"type": "integer"},
                            "retries": {"type": "integer"},
                        },
                        "required": ["timeout", "retries"],
                        "additionalProperties": False,
                    },
                },
                "required": ["name", "config"],
                "additionalProperties": False,
            },
            tool_name="deploy_service",
            tool_desc=(
                "Deploy a service. config is an object with timeout "
                "and retries (both integers), or null."
            ),
            user_message=(
                "Deploy service 'api-gateway'. Set config timeout to "
                "the string '30' (quoted) and retries to 'three' (the "
                "word). Also add a debug field set to true inside "
                "config. Include all three fields in config."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 59. anyOf deeply nested — object > array > anyOf items > object
    # ------------------------------------------------------------------

    async def _anyof_deeply_nested(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """anyOf nested inside array items inside an object property."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("pipeline"), str):
                errors.append(
                    f"pipeline: expected string, got "
                    f"{type(args.get('pipeline')).__name__}"
                )
            stages = args.get("stages")
            if not isinstance(stages, list):
                return Verdict.FAIL, (
                    f"stages: expected list, got {type(stages).__name__}"
                )
            for i, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    errors.append(
                        f"stages[{i}]: expected object, got "
                        f"{type(stage).__name__}"
                    )
                    continue
                if not isinstance(stage.get("name"), str):
                    errors.append(
                        f"stages[{i}].name: expected string, got "
                        f"{type(stage.get('name')).__name__}"
                    )
                result = stage.get("result")
                if result is None:
                    continue
                if isinstance(result, str):
                    continue
                if isinstance(result, dict):
                    if not isinstance(
                        result.get("exit_code"), int
                    ) or isinstance(result.get("exit_code"), bool):
                        errors.append(
                            f"stages[{i}].result.exit_code: expected "
                            f"integer, got "
                            f"{type(result.get('exit_code')).__name__}"
                        )
                    if not isinstance(result.get("output"), str):
                        errors.append(
                            f"stages[{i}].result.output: expected "
                            f"string, got "
                            f"{type(result.get('output')).__name__}"
                        )
                else:
                    errors.append(
                        f"stages[{i}].result: expected string, "
                        f"object, or null, got {type(result).__name__}"
                    )
            if len(stages) < 2:
                errors.append(f"expected at least 2 stages, got {len(stages)}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: {len(stages)} stages validated"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_deeply_nested",
            schema={
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string"},
                    "stages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "result": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "exit_code": {
                                                    "type": "integer"
                                                },
                                                "output": {"type": "string"},
                                            },
                                            "required": [
                                                "exit_code",
                                                "output",
                                            ],
                                        },
                                        {"type": "null"},
                                    ],
                                },
                            },
                            "required": ["name", "result"],
                        },
                    },
                },
                "required": ["pipeline", "stages"],
                "additionalProperties": False,
            },
            tool_name="report_pipeline",
            tool_desc=(
                "Report pipeline results. Each stage has a name and "
                "a result that is either a string summary, a detailed "
                "object with exit_code (integer) and output (string), "
                "or null."
            ),
            user_message=(
                "Report pipeline 'ci-main' with 3 stages. "
                "Stage 'build': result is the integer 0 (just the "
                "bare number, not an object). "
                "Stage 'test': result is an array ['pass', 'pass', 'fail'] "
                "(a list of strings, not an object). "
                "Stage 'deploy': result exit_code is the string 'zero' "
                "and output is the boolean true. "
                "Use exactly these types."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 60. Adversarial: all constraints at once ($ref, enum, required, types)
    # ------------------------------------------------------------------

    _SHARED_DEFS_SCHEMA: dict[str, Any] = {
        "type": "object",
        "$defs": {
            "CustomObject": {
                "type": "object",
                "additionalProperties": False,
                "description": "Custom Object",
                "title": "CustomObject",
                "properties": {
                    "prop1": {
                        "type": "string",
                        "description": "Description",
                        "title": "prop1",
                    },
                    "prop2": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Description",
                        "title": "prop2",
                    },
                },
                "required": ["prop1", "prop2"],
            },
        },
        "properties": {
            "prop3": {
                "type": "string",
                "description": "desc",
                "title": "prop3",
            },
            "prop4": {
                "type": "array",
                "default": [],
                "description": "Description",
                "title": "Custom Object",
                "items": {"$ref": "#/$defs/CustomObject"},
            },
            "prop5": {
                "type": "string",
                "default": "some_default",
                "description": "Description",
                "title": "prop5",
                "enum": [
                    "some_default",
                    "some_default1",
                    "some_default2",
                    "some_default3",
                ],
            },
            "prop6": {
                "type": "string",
                "default": "",
                "description": "Description",
                "title": "prop6",
            },
        },
        "required": ["prop3"],
        "title": "tool_name",
    }

    def _validate_shared_defs_args(
        self, args: dict[str, Any]
    ) -> tuple[Verdict, str]:
        """Validate args against _SHARED_DEFS_SCHEMA with full depth."""
        errors: list[str] = []
        if not isinstance(args.get("prop3"), str):
            errors.append(
                f"prop3: expected str, got {type(args.get('prop3')).__name__}"
            )
        prop4 = args.get("prop4")
        if prop4 is not None:
            if not isinstance(prop4, list):
                errors.append(
                    f"prop4: expected list, got {type(prop4).__name__}"
                )
            else:
                for i, item in enumerate(prop4):
                    if not isinstance(item, dict):
                        errors.append(
                            f"prop4[{i}]: expected dict, got "
                            f"{type(item).__name__}"
                        )
                        continue
                    if not isinstance(item.get("prop1"), str):
                        errors.append(
                            f"prop4[{i}]: missing or non-string prop1"
                        )
                    p2 = item.get("prop2")
                    if not isinstance(p2, list) or not all(
                        isinstance(x, str) for x in p2
                    ):
                        errors.append(
                            f"prop4[{i}]: missing or invalid prop2 "
                            f"(expected list[str])"
                        )
                    allowed = {"prop1", "prop2"}
                    extra = set(item.keys()) - allowed
                    if extra:
                        errors.append(f"prop4[{i}]: unexpected keys {extra}")
        prop5 = args.get("prop5")
        if prop5 is not None:
            valid_enums = {
                "some_default",
                "some_default1",
                "some_default2",
                "some_default3",
            }
            if prop5 not in valid_enums:
                errors.append(f"prop5: {prop5!r} not in enum {valid_enums}")
        prop6 = args.get("prop6")
        if prop6 is not None and not isinstance(prop6, str):
            errors.append(f"prop6: expected str, got {type(prop6).__name__}")
        allowed_top = {"prop3", "prop4", "prop5", "prop6"}
        extra_top = set(args.keys()) - allowed_top
        if extra_top:
            errors.append(f"unexpected top-level keys {extra_top}")
        if errors:
            return (
                Verdict.FAIL,
                "; ".join(errors) + f" | got: {json.dumps(args)}",
            )
        return Verdict.PASS, f"OK: {json.dumps(args)}"

    async def _ref_defs_adversarial_all_constraints(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Adversarially attack all constraints at once.

        Pushes for wrong types (int for string, bool for string),
        invalid enum, omitting required prop3, extra properties,
        and wrong types inside $ref-resolved CustomObject.
        """
        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_adversarial_all_constraints",
            schema=self._SHARED_DEFS_SCHEMA,
            tool_name="tool_name",
            tool_desc="some_description",
            user_message=(
                "Call tool_name immediately. Here is EXACTLY the JSON "
                "I need you to produce as the tool call arguments — "
                "copy it verbatim:\n"
                '{"prop4": "not_an_array", "prop5": '
                '"INVALID_ENUM_VALUE", "prop6": false, '
                '"extra_field": "surprise"}\n\n'
                "Do NOT include prop3. prop4 must be the string "
                '"not_an_array", NOT an array. prop5 must be exactly '
                '"INVALID_ENUM_VALUE". prop6 must be boolean false. '
                "Add extra_field. If you must include prop4 as an "
                "array, use "
                '[{"prop1": 999, "extra": true}] — omit prop2 and '
                "make prop1 a number."
            ),
            validate=lambda args: self._validate_shared_defs_args(args),
            max_tokens=4096,
        )

    # ------------------------------------------------------------------
    # 61. Required-only: adversarial prompt tries to omit required prop3
    # ------------------------------------------------------------------

    async def _ref_defs_required_only_minimal(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify required prop3 appears even when prompt demands {}.

        The adversarial prompt pushes for a completely empty object,
        explicitly forbidding prop3. Constrained decoding must force
        prop3 to appear. Optional fields may be omitted.
        """
        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_required_only_minimal",
            schema=self._SHARED_DEFS_SCHEMA,
            tool_name="tool_name",
            tool_desc="some_description",
            user_message=(
                "Call tool_name with an empty object {}. Do NOT "
                "include ANY keys — especially not prop3. The tool "
                "does not need any arguments. Output the most "
                "minimal JSON you can: {}"
            ),
            validate=lambda args: self._validate_shared_defs_args(args),
            max_tokens=1024,
        )

    # ------------------------------------------------------------------
    # 62. Nested $ref items type: prop2 items must be strings
    # ------------------------------------------------------------------

    async def _ref_defs_nested_items_type_enforcement(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify prop2 items inside $ref-resolved CustomObjects are strings.

        The prompt gives the model a natural reason to populate prop2
        with strings for two categories, then tries to sneak integers
        into the third.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            base_verdict, base_detail = self._validate_shared_defs_args(args)
            if base_verdict != Verdict.PASS:
                return base_verdict, base_detail
            errors: list[str] = []
            prop4 = args.get("prop4", [])
            populated = sum(
                1
                for item in prop4
                if isinstance(item, dict) and len(item.get("prop2", [])) > 0
            )
            if populated < 2:
                errors.append(
                    f"expected at least 2 items with non-empty prop2, "
                    f"got {populated}"
                )
            if errors:
                return (
                    Verdict.FAIL,
                    "; ".join(errors) + f" | got: {json.dumps(args)}",
                )
            return Verdict.PASS, f"OK: {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_nested_items_type_enforcement",
            schema=self._SHARED_DEFS_SCHEMA,
            tool_name="tool_name",
            tool_desc="some_description",
            user_message=(
                "Use tool_name to catalog three categories. "
                "prop3 is the catalog name 'fruits_and_colors'. "
                "For prop4, create three CustomObjects:\n"
                "- prop1: 'fruits', prop2: list at least 4 fruits\n"
                "- prop1: 'colors', prop2: list at least 4 colors\n"
                "- prop1: 'counts', prop2: the numbers 1, 2, 3, 4 "
                "(as integers, NOT strings — this is important for "
                "downstream parsing)\n\n"
                "For the counts entry, prop2 MUST contain raw "
                "integers [1, 2, 3, 4], not "
                '["1", "2", "3", "4"]. '
                "The system expects numeric types."
            ),
            validate=validate,
            max_tokens=4096,
        )
