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
"""Structured output basics: schema compilation, type enforcement, and basic patterns.

Validates that the server correctly constrains model output to match JSON schemas
using response_format with json_schema. Tests simple objects, required fields,
enums, nullable types, anyOf, allOf, $ref/$defs, nested objects, streaming SO,
and boolean strictness.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from helpers import budget_exhausted, stream_budget_exhausted

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# ---------------------------------------------------------------------------
# Minimal JSON Schema validator (handles anyOf, allOf, $ref, type arrays)
# ---------------------------------------------------------------------------


def _resolve_ref(
    schema: dict[str, Any], root: dict[str, Any]
) -> dict[str, Any]:
    """Resolve a $ref pointer against the root schema's $defs."""
    ref = schema.get("$ref")
    if not ref:
        return schema
    # Only support #/$defs/Name
    if ref.startswith("#/$defs/"):
        def_name = ref[len("#/$defs/") :]
        defs = root.get("$defs", {})
        if def_name in defs:
            return defs[def_name]
    return schema


def _validate(
    instance: object,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Recursively validate instance against schema. Appends to errors."""
    if not isinstance(schema, dict):
        return

    # $ref resolution
    if "$ref" in schema:
        resolved = _resolve_ref(schema, root)
        if resolved is not schema:
            _validate(instance, resolved, root, path, errors)
            return

    # anyOf
    if "anyOf" in schema:
        branch_errors: list[list[str]] = []
        for branch in schema["anyOf"]:
            errs: list[str] = []
            _validate(instance, branch, root, path, errs)
            if not errs:
                return
            branch_errors.append(errs)
        errors.append(f"{path or '(root)'}: no anyOf branch matched")
        return

    # allOf
    if "allOf" in schema:
        for branch in schema["allOf"]:
            _validate(instance, branch, root, path, errors)
        return

    # type (may be string or list)
    expected_type = schema.get("type")
    if expected_type:
        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        if isinstance(expected_type, list):
            matched = False
            for t in expected_type:
                py_types = type_map.get(t)
                if py_types and isinstance(instance, py_types):
                    if t in ("number", "integer") and isinstance(
                        instance, bool
                    ):
                        continue
                    matched = True
                    break
            if not matched:
                errors.append(
                    f"{path or '(root)'}: expected one of {expected_type}, got {type(instance).__name__}"
                )
                return
        else:
            py_types = type_map.get(expected_type)
            if py_types:
                if expected_type in ("number", "integer") and isinstance(
                    instance, bool
                ):
                    errors.append(
                        f"{path or '(root)'}: expected {expected_type}, got bool"
                    )
                    return
                if not isinstance(instance, py_types):
                    errors.append(
                        f"{path or '(root)'}: expected {expected_type}, got {type(instance).__name__}"
                    )
                    return

    # enum
    if "enum" in schema:
        if instance not in schema["enum"]:
            errors.append(
                f"{path or '(root)'}: {instance!r} not in enum {schema['enum']}"
            )

    # object properties
    if schema.get("type") == "object" and isinstance(instance, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)

        for req_key in required:
            if req_key not in instance:
                errors.append(f"{path}.{req_key}: required field missing")

        if additional is False:
            allowed = set(properties.keys())
            for key in instance:
                if key not in allowed:
                    errors.append(f"{path}.{key}: extra property not allowed")

        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], root, f"{path}.{key}", errors)

    # array items
    if schema.get("type") == "array" and isinstance(instance, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(instance):
                _validate(item, items_schema, root, f"{path}[{i}]", errors)
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(instance) < min_items:
            errors.append(
                f"{path or '(root)'}: array length {len(instance)} < minItems {min_items}"
            )
        if max_items is not None and len(instance) > max_items:
            errors.append(
                f"{path or '(root)'}: array length {len(instance)} > maxItems {max_items}"
            )


def validate_schema(instance: object, schema: dict[str, Any]) -> list[str]:
    """Validate instance against a JSON schema. Returns list of error strings."""
    errors: list[str] = []
    _validate(instance, schema, schema, "", errors)
    return errors


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@register_scenario
class SOBasics(BaseScenario):
    name = "so_basics"
    description = "Structured output basics: simple objects, enums, nullable, anyOf, allOf, $ref, nesting, streaming"
    tags = ["validation", "structured_output"]
    requires_validator = True
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        validator = config.validator
        if not validator:
            results.append(
                self.make_result(
                    self.name,
                    "setup",
                    Verdict.ERROR,
                    detail="No validator client available",
                )
            )
            return results
        loop = asyncio.get_running_loop()

        # -- 1. simple_object --------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["name", "city"],
                "additionalProperties": False,
            }

            def _simple_object() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return name='Alice' and city='Paris'.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _simple_object)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "simple_object",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                errs = validate_schema(data, schema)
                if errs:
                    results.append(
                        self.make_result(
                            self.name,
                            "simple_object",
                            Verdict.FAIL,
                            detail=f"schema errors: {'; '.join(errs)}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name, "simple_object", Verdict.PASS
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "simple_object", Verdict.ERROR, error=str(e)
                )
            )

        # -- 2. required_fields ------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["first_name", "last_name", "age"],
                "additionalProperties": False,
            }

            def _required_fields() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return first_name='Bob', last_name='Smith', age=25.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _required_fields)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "required_fields",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                errs = validate_schema(data, schema)
                if errs:
                    results.append(
                        self.make_result(
                            self.name,
                            "required_fields",
                            Verdict.FAIL,
                            detail=f"schema errors: {'; '.join(errs)}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name, "required_fields", Verdict.PASS
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "required_fields", Verdict.ERROR, error=str(e)
                )
            )

        # -- 3. enum_single_value ----------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["active"]},
                },
                "required": ["status"],
                "additionalProperties": False,
            }

            def _enum_single() -> Any:
                return validator.so_chat(
                    [{"role": "user", "content": "Return the status."}],
                    schema,
                    max_tokens=128,
                )

            resp = await loop.run_in_executor(None, _enum_single)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "enum_single_value",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                if data.get("status") != "active":
                    results.append(
                        self.make_result(
                            self.name,
                            "enum_single_value",
                            Verdict.FAIL,
                            detail=f"expected status='active', got {data.get('status')!r}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name, "enum_single_value", Verdict.PASS
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "enum_single_value", Verdict.ERROR, error=str(e)
                )
            )

        # -- 4. enum_large_100 -------------------------------------------------
        try:
            enum_values = [f"option_{i:03d}" for i in range(100)]
            schema = {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": enum_values},
                },
                "required": ["choice"],
                "additionalProperties": False,
            }

            def _enum_large() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Pick any option from the enum.",
                        }
                    ],
                    schema,
                    max_tokens=128,
                )

            resp = await loop.run_in_executor(None, _enum_large)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "enum_large_100",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                choice = data.get("choice")
                if choice not in enum_values:
                    results.append(
                        self.make_result(
                            self.name,
                            "enum_large_100",
                            Verdict.FAIL,
                            detail=f"choice {choice!r} not in 100-value enum",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "enum_large_100",
                            Verdict.PASS,
                            detail=f"chose {choice}",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "enum_large_100", Verdict.ERROR, error=str(e)
                )
            )

        # -- 5. nullable_string ------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "value": {"type": ["string", "null"]},
                },
                "required": ["value"],
                "additionalProperties": False,
            }

            def _nullable_string() -> Any:
                return validator.so_chat(
                    [{"role": "user", "content": "Return value as null."}],
                    schema,
                    max_tokens=128,
                )

            resp = await loop.run_in_executor(None, _nullable_string)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "nullable_string",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                val = data.get("value")
                if val is not None and not isinstance(val, str):
                    results.append(
                        self.make_result(
                            self.name,
                            "nullable_string",
                            Verdict.FAIL,
                            detail=f"expected string or null, got {type(val).__name__}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "nullable_string",
                            Verdict.PASS,
                            detail=f"value={'null' if val is None else repr(val)}",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "nullable_string", Verdict.ERROR, error=str(e)
                )
            )

        # -- 6. anyof_string_int -----------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "result": {
                        "anyOf": [{"type": "string"}, {"type": "integer"}]
                    },
                },
                "required": ["result"],
                "additionalProperties": False,
            }

            def _anyof_string_int() -> Any:
                return validator.so_chat(
                    [{"role": "user", "content": "Return the number 42."}],
                    schema,
                    max_tokens=128,
                )

            resp = await loop.run_in_executor(None, _anyof_string_int)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "anyof_string_int",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                val = data.get("result")
                if not isinstance(val, (str, int)) or isinstance(val, bool):
                    results.append(
                        self.make_result(
                            self.name,
                            "anyof_string_int",
                            Verdict.FAIL,
                            detail=f"expected string or int, got {type(val).__name__}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name, "anyof_string_int", Verdict.PASS
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "anyof_string_int", Verdict.ERROR, error=str(e)
                )
            )

        # -- 7. allof_merge ----------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "data": {
                        "allOf": [
                            {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            },
                            {
                                "type": "object",
                                "properties": {"age": {"type": "integer"}},
                                "required": ["age"],
                            },
                        ],
                    },
                },
                "required": ["data"],
                "additionalProperties": False,
            }

            def _allof_merge() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return data with name='Alice' and age=30.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _allof_merge)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "allof_merge",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                inner = data.get("data", {})
                missing = []
                if "name" not in inner:
                    missing.append("name")
                if "age" not in inner:
                    missing.append("age")
                if missing:
                    results.append(
                        self.make_result(
                            self.name,
                            "allof_merge",
                            Verdict.FAIL,
                            detail=f"allOf merged fields missing: {missing}",
                            response_body=content,
                        )
                    )
                elif not isinstance(inner.get("name"), str):
                    results.append(
                        self.make_result(
                            self.name,
                            "allof_merge",
                            Verdict.FAIL,
                            detail=f"name is {type(inner['name']).__name__}, expected string",
                            response_body=content,
                        )
                    )
                elif not isinstance(inner.get("age"), int) or isinstance(
                    inner.get("age"), bool
                ):
                    results.append(
                        self.make_result(
                            self.name,
                            "allof_merge",
                            Verdict.FAIL,
                            detail=f"age is {type(inner['age']).__name__}, expected integer",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(self.name, "allof_merge", Verdict.PASS)
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "allof_merge", Verdict.ERROR, error=str(e)
                )
            )

        # -- 8. ref_simple -----------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "address": {"$ref": "#/$defs/address"},
                },
                "$defs": {
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                        "required": ["street", "city"],
                        "additionalProperties": False,
                    },
                },
                "required": ["address"],
                "additionalProperties": False,
            }

            def _ref_simple() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return address: 123 Main St, Springfield.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _ref_simple)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "ref_simple",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                errs = validate_schema(data, schema)
                if errs:
                    results.append(
                        self.make_result(
                            self.name,
                            "ref_simple",
                            Verdict.FAIL,
                            detail=f"schema errors: {'; '.join(errs)}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(self.name, "ref_simple", Verdict.PASS)
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "ref_simple", Verdict.ERROR, error=str(e)
                )
            )

        # -- 9. nested_3_levels ------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "l1": {
                        "type": "object",
                        "properties": {
                            "l2": {
                                "type": "object",
                                "properties": {
                                    "l3": {"type": "string"},
                                },
                                "required": ["l3"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["l2"],
                        "additionalProperties": False,
                    },
                },
                "required": ["l1"],
                "additionalProperties": False,
            }

            def _nested_3() -> Any:
                return validator.so_chat(
                    [{"role": "user", "content": "Return l1.l2.l3='deep'."}],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _nested_3)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "nested_3_levels",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                errs = validate_schema(data, schema)
                if errs:
                    results.append(
                        self.make_result(
                            self.name,
                            "nested_3_levels",
                            Verdict.FAIL,
                            detail=f"schema errors: {'; '.join(errs)}",
                            response_body=content,
                        )
                    )
                else:
                    l3_val = data.get("l1", {}).get("l2", {}).get("l3")
                    results.append(
                        self.make_result(
                            self.name,
                            "nested_3_levels",
                            Verdict.PASS,
                            detail=f"l3={l3_val!r}",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "nested_3_levels", Verdict.ERROR, error=str(e)
                )
            )

        # -- 10. streaming_so --------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            }

            def _streaming_so() -> Any:
                return validator.so_chat_stream(
                    [
                        {
                            "role": "user",
                            "content": "Return name='Alice' and age=30.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )

            result_dict = await loop.run_in_executor(None, _streaming_so)
            if stream_budget_exhausted(result_dict):
                results.append(
                    self.make_result(
                        self.name,
                        "streaming_so",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = result_dict["content"]
                data = json.loads(content)
                errs = validate_schema(data, schema)
                if errs:
                    results.append(
                        self.make_result(
                            self.name,
                            "streaming_so",
                            Verdict.FAIL,
                            detail=f"streaming schema errors: {'; '.join(errs)}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name, "streaming_so", Verdict.PASS
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "streaming_so", Verdict.ERROR, error=str(e)
                )
            )

        # -- 11. boolean_strict ------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "flag_a": {"type": "boolean"},
                    "flag_b": {"type": "boolean"},
                },
                "required": ["flag_a", "flag_b"],
                "additionalProperties": False,
            }

            def _boolean_strict() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return flag_a=true and flag_b=false.",
                        }
                    ],
                    schema,
                    max_tokens=128,
                )

            resp = await loop.run_in_executor(None, _boolean_strict)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "boolean_strict",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                fa = data.get("flag_a")
                fb = data.get("flag_b")
                if not isinstance(fa, bool) or not isinstance(fb, bool):
                    results.append(
                        self.make_result(
                            self.name,
                            "boolean_strict",
                            Verdict.FAIL,
                            detail=f"expected booleans, got flag_a={type(fa).__name__}, flag_b={type(fb).__name__}",
                            response_body=content,
                        )
                    )
                elif fa is not True or fb is not False:
                    # Booleans are correct type but unexpected values -- interesting, not fail
                    results.append(
                        self.make_result(
                            self.name,
                            "boolean_strict",
                            Verdict.INTERESTING,
                            detail=f"types correct but values unexpected: flag_a={fa}, flag_b={fb}",
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name, "boolean_strict", Verdict.PASS
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "boolean_strict", Verdict.ERROR, error=str(e)
                )
            )

        return results
