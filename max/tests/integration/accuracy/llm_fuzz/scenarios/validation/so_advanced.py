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
"""Advanced structured output: complex unions, circular refs, concurrency, large schemas.

Pushes the grammar compilation and guided decoding further with discriminated unions,
chained references, special-character enums, numeric enums, array constraints,
real-world schemas, concurrent isolation, error handling, and large property counts.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from helpers import budget_exhausted

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# ---------------------------------------------------------------------------
# Minimal JSON Schema validator (same core as v04, extended)
# ---------------------------------------------------------------------------


def _resolve_ref(
    schema: dict[str, Any], root: dict[str, Any]
) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
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
    depth: int = 0,
) -> None:
    if not isinstance(schema, dict) or depth > 50:
        return

    if "$ref" in schema:
        resolved = _resolve_ref(schema, root)
        if resolved is not schema:
            _validate(instance, resolved, root, path, errors, depth + 1)
            return

    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            errs: list[str] = []
            _validate(instance, branch, root, path, errs, depth + 1)
            if not errs:
                return
        errors.append(f"{path or '(root)'}: no anyOf branch matched")
        return

    if "allOf" in schema:
        for branch in schema["allOf"]:
            _validate(instance, branch, root, path, errors, depth + 1)
        return

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

    if "enum" in schema:
        if instance not in schema["enum"]:
            errors.append(f"{path or '(root)'}: {instance!r} not in enum")

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
                _validate(
                    value,
                    properties[key],
                    root,
                    f"{path}.{key}",
                    errors,
                    depth + 1,
                )

    if schema.get("type") == "array" and isinstance(instance, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(instance):
                _validate(
                    item, items_schema, root, f"{path}[{i}]", errors, depth + 1
                )
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
    errors: list[str] = []
    _validate(instance, schema, schema, "", errors)
    return errors


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@register_scenario
class SOAdvanced(BaseScenario):
    name = "so_advanced"
    description = "Advanced structured output: discriminated unions, circular refs, concurrency, large schemas"
    tags = ["validation", "structured_output", "advanced"]
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

        # -- 1. anyof_object_variants ------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "result": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "kind": {
                                        "type": "string",
                                        "enum": ["text"],
                                    },
                                    "text": {"type": "string"},
                                },
                                "required": ["kind", "text"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "kind": {
                                        "type": "string",
                                        "enum": ["error"],
                                    },
                                    "error_code": {"type": "integer"},
                                },
                                "required": ["kind", "error_code"],
                                "additionalProperties": False,
                            },
                        ],
                    },
                },
                "required": ["result"],
                "additionalProperties": False,
            }

            def _anyof_objects() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return a text result with text='hello'.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _anyof_objects)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "anyof_object_variants",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                r = data.get("result", {})
                kind = r.get("kind")
                if kind == "text" and "text" in r:
                    results.append(
                        self.make_result(
                            self.name, "anyof_object_variants", Verdict.PASS
                        )
                    )
                elif kind == "error" and "error_code" in r:
                    results.append(
                        self.make_result(
                            self.name,
                            "anyof_object_variants",
                            Verdict.INTERESTING,
                            detail="model chose error branch instead of text",
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "anyof_object_variants",
                            Verdict.FAIL,
                            detail=f"result does not match either anyOf branch: {r}",
                            response_body=content,
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "anyof_object_variants",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # -- 2. allof_three_schemas --------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "point": {
                        "allOf": [
                            {
                                "type": "object",
                                "properties": {"x": {"type": "integer"}},
                                "required": ["x"],
                            },
                            {
                                "type": "object",
                                "properties": {"y": {"type": "integer"}},
                                "required": ["y"],
                            },
                            {
                                "type": "object",
                                "properties": {"z": {"type": "integer"}},
                                "required": ["z"],
                            },
                        ],
                    },
                },
                "required": ["point"],
                "additionalProperties": False,
            }

            def _allof_three() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return point with x=1, y=2, z=3.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _allof_three)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "allof_three_schemas",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                point = data.get("point", {})
                missing = [k for k in ("x", "y", "z") if k not in point]
                if missing:
                    results.append(
                        self.make_result(
                            self.name,
                            "allof_three_schemas",
                            Verdict.FAIL,
                            detail=f"allOf merged fields missing: {missing}",
                            response_body=content,
                        )
                    )
                elif not all(
                    isinstance(point[k], int) and not isinstance(point[k], bool)
                    for k in ("x", "y", "z")
                ):
                    results.append(
                        self.make_result(
                            self.name,
                            "allof_three_schemas",
                            Verdict.FAIL,
                            detail=f"not all values are integers: {point}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name, "allof_three_schemas", Verdict.PASS
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "allof_three_schemas",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # -- 3. ref_circular ---------------------------------------------------
        # Tree structure with recursive $ref. Grammar backends (xgrammar,
        # llguidance) support recursive schemas with depth limits, so the
        # server should compile and produce valid output.
        try:
            schema = {
                "type": "object",
                "properties": {
                    "tree": {"$ref": "#/$defs/node"},
                },
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "children": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/node"},
                            },
                        },
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
                "required": ["tree"],
                "additionalProperties": False,
            }

            def _ref_circular() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return a tree with root='A' and one child 'B'.",
                        }
                    ],
                    schema,
                    max_tokens=512,
                )

            try:
                resp = await loop.run_in_executor(None, _ref_circular)
                if budget_exhausted(resp):
                    results.append(
                        self.make_result(
                            self.name,
                            "ref_circular",
                            Verdict.PASS,
                            detail="budget exhausted",
                        )
                    )
                else:
                    content = resp.choices[0].message.content
                    data = json.loads(content)
                    # If it compiles and produces valid output, that's a pass
                    tree = data.get("tree", {})
                    if "value" in tree:
                        results.append(
                            self.make_result(
                                self.name,
                                "ref_circular",
                                Verdict.PASS,
                                detail="circular $ref compiled and produced valid output",
                            )
                        )
                    else:
                        results.append(
                            self.make_result(
                                self.name,
                                "ref_circular",
                                Verdict.FAIL,
                                detail=f"output missing 'value' field: {tree}",
                                response_body=content,
                            )
                        )
            except Exception as inner_e:
                err_str = str(inner_e).lower()
                if "500" in err_str or "internal" in err_str:
                    results.append(
                        self.make_result(
                            self.name,
                            "ref_circular",
                            Verdict.FAIL,
                            detail=f"server crashed on recursive schema: {inner_e}",
                        )
                    )
                elif (
                    "400" in err_str
                    or "recursive" in err_str
                    or "unsupported" in err_str
                ):
                    results.append(
                        self.make_result(
                            self.name,
                            "ref_circular",
                            Verdict.INTERESTING,
                            detail=f"server rejected recursive schema (should be supported): {inner_e}",
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "ref_circular",
                            Verdict.INTERESTING,
                            detail=f"unexpected error on recursive schema: {inner_e}",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "ref_circular", Verdict.ERROR, error=str(e)
                )
            )

        # -- 4. ref_chained ----------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "result": {"$ref": "#/$defs/TypeA"},
                },
                "$defs": {
                    "TypeA": {
                        "type": "object",
                        "properties": {
                            "a_value": {"type": "string"},
                            "nested": {"$ref": "#/$defs/TypeB"},
                        },
                        "required": ["a_value", "nested"],
                        "additionalProperties": False,
                    },
                    "TypeB": {
                        "type": "object",
                        "properties": {
                            "b_value": {"type": "integer"},
                            "leaf": {"$ref": "#/$defs/TypeC"},
                        },
                        "required": ["b_value", "leaf"],
                        "additionalProperties": False,
                    },
                    "TypeC": {
                        "type": "object",
                        "properties": {
                            "c_value": {"type": "boolean"},
                        },
                        "required": ["c_value"],
                        "additionalProperties": False,
                    },
                },
                "required": ["result"],
                "additionalProperties": False,
            }

            def _ref_chained() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return result with a_value='hello', nested.b_value=42, nested.leaf.c_value=true.",
                        }
                    ],
                    schema,
                    max_tokens=512,
                )

            resp = await loop.run_in_executor(None, _ref_chained)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "ref_chained",
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
                            "ref_chained",
                            Verdict.FAIL,
                            detail=f"chained $ref errors: {'; '.join(errs)}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(self.name, "ref_chained", Verdict.PASS)
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "ref_chained", Verdict.ERROR, error=str(e)
                )
            )

        # -- 5. enum_special_chars ---------------------------------------------
        try:
            special_values = [
                "hello world",
                "it's a test",
                'value with "quotes"',
                "line\nbreak",
                "tab\there",
                "emoji: \U0001f600",
                "path/to/file",
                "a&b<c>d",
            ]
            schema = {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": special_values},
                },
                "required": ["choice"],
                "additionalProperties": False,
            }

            def _enum_special() -> Any:
                return validator.so_chat(
                    [{"role": "user", "content": "Pick any enum value."}],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _enum_special)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "enum_special_chars",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                choice = data.get("choice")
                if choice in special_values:
                    results.append(
                        self.make_result(
                            self.name,
                            "enum_special_chars",
                            Verdict.PASS,
                            detail=f"chose {choice!r}",
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "enum_special_chars",
                            Verdict.FAIL,
                            detail=f"choice {choice!r} not in special-char enum",
                            response_body=content,
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "enum_special_chars", Verdict.ERROR, error=str(e)
                )
            )

        # -- 6. enum_numeric ---------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "integer",
                        "enum": [200, 201, 400, 404, 500],
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            }

            def _enum_numeric() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return HTTP status code 404.",
                        }
                    ],
                    schema,
                    max_tokens=128,
                )

            resp = await loop.run_in_executor(None, _enum_numeric)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "enum_numeric",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                code = data.get("code")
                if code not in [200, 201, 400, 404, 500]:
                    results.append(
                        self.make_result(
                            self.name,
                            "enum_numeric",
                            Verdict.FAIL,
                            detail=f"code={code!r} not in integer enum",
                            response_body=content,
                        )
                    )
                elif not isinstance(code, int) or isinstance(code, bool):
                    results.append(
                        self.make_result(
                            self.name,
                            "enum_numeric",
                            Verdict.FAIL,
                            detail=f"code is {type(code).__name__}, expected integer",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "enum_numeric",
                            Verdict.PASS,
                            detail=f"code={code}",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "enum_numeric", Verdict.ERROR, error=str(e)
                )
            )

        # -- 7. array_min_max_items --------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 5,
                    },
                },
                "required": ["tags"],
                "additionalProperties": False,
            }

            def _array_min_max() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return exactly 3 tags: 'python', 'rust', 'go'.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )

            resp = await loop.run_in_executor(None, _array_min_max)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "array_min_max_items",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                tags = data.get("tags", [])
                if not isinstance(tags, list):
                    results.append(
                        self.make_result(
                            self.name,
                            "array_min_max_items",
                            Verdict.FAIL,
                            detail=f"tags is {type(tags).__name__}, expected array",
                            response_body=content,
                        )
                    )
                elif len(tags) < 2:
                    results.append(
                        self.make_result(
                            self.name,
                            "array_min_max_items",
                            Verdict.FAIL,
                            detail=f"array length {len(tags)} < minItems 2",
                            response_body=content,
                        )
                    )
                elif len(tags) > 5:
                    results.append(
                        self.make_result(
                            self.name,
                            "array_min_max_items",
                            Verdict.FAIL,
                            detail=f"array length {len(tags)} > maxItems 5",
                            response_body=content,
                        )
                    )
                elif not all(isinstance(t, str) for t in tags):
                    results.append(
                        self.make_result(
                            self.name,
                            "array_min_max_items",
                            Verdict.FAIL,
                            detail="not all array items are strings",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "array_min_max_items",
                            Verdict.PASS,
                            detail=f"array length={len(tags)}",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "array_min_max_items",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # -- 8. real_world_schema ----------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "date": {"type": "string"},
                    "total": {"type": "number"},
                    "currency": {
                        "type": "string",
                        "enum": ["USD", "EUR", "GBP", "JPY"],
                    },
                    "paid": {"type": "boolean"},
                    "customer": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "address": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "city": {"type": "string"},
                                    "country": {"type": "string"},
                                },
                                "required": ["street", "city", "country"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["name", "email", "address"],
                        "additionalProperties": False,
                    },
                    "line_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "integer"},
                                "unit_price": {"type": "number"},
                            },
                            "required": [
                                "description",
                                "quantity",
                                "unit_price",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "notes": {"type": ["string", "null"]},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "status": {
                        "type": "string",
                        "enum": ["draft", "sent", "paid", "overdue"],
                    },
                    "tax_rate": {"type": "number"},
                },
                "required": [
                    "invoice_number",
                    "date",
                    "total",
                    "currency",
                    "paid",
                    "customer",
                    "line_items",
                    "notes",
                    "tags",
                    "status",
                    "tax_rate",
                ],
                "additionalProperties": False,
            }

            def _real_world() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Generate an invoice: INV-001, 2025-01-15, $250.00 USD, paid, "
                                "customer John Doe (john@example.com, 123 Oak St, Portland, US), "
                                "2 line items (Widget x2 at $50, Service x1 at $150), "
                                "no notes, tags=['consulting'], status=paid, tax_rate=0.08."
                            ),
                        }
                    ],
                    schema,
                    max_tokens=1024,
                )

            resp = await loop.run_in_executor(None, _real_world)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "real_world_schema",
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
                            "real_world_schema",
                            Verdict.FAIL,
                            detail=f"real-world schema errors ({len(errs)}): {'; '.join(errs[:5])}",
                            response_body=content,
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "real_world_schema",
                            Verdict.PASS,
                            detail=f"{len(data.get('line_items', []))} line items, "
                            f"currency={data.get('currency')}, status={data.get('status')}",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "real_world_schema", Verdict.ERROR, error=str(e)
                )
            )

        # -- 9. concurrent_so_mixed --------------------------------------------
        try:
            schemas = [
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"flag": {"type": "boolean"}},
                    "required": ["flag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["items"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["ok", "error"]},
                    },
                    "required": ["status"],
                    "additionalProperties": False,
                },
            ]

            def _run_one_so(
                idx: int,
            ) -> tuple[int, Any | None, str | None]:
                s = schemas[idx]
                resp = validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": f"Generate output for schema {idx}.",
                        }
                    ],
                    s,
                    max_tokens=256,
                )
                if budget_exhausted(resp):
                    return idx, None, "budget exhausted"
                content = resp.choices[0].message.content
                data = json.loads(content)
                errs = validate_schema(data, s)
                if errs:
                    return idx, content, f"schema errors: {'; '.join(errs)}"
                return idx, content, None

            concurrent_results = await loop.run_in_executor(
                None,
                lambda: validator.concurrent_run(
                    _run_one_so,
                    [(i,) for i in range(5)],
                    max_workers=5,
                ),
            )

            failures = []
            budget_hits = 0
            for idx, result_val, item_err in concurrent_results:
                if item_err:
                    if isinstance(result_val, tuple) and len(result_val) == 3:
                        _, _, detail = result_val
                        if detail == "budget exhausted":
                            budget_hits += 1
                            continue
                    failures.append(f"request {idx}: {item_err}")
                elif result_val is not None:
                    if isinstance(result_val, tuple) and len(result_val) == 3:
                        _, _, detail = result_val
                        if detail == "budget exhausted":
                            budget_hits += 1
                        elif detail:
                            failures.append(f"request {idx}: {detail}")

            if failures:
                results.append(
                    self.make_result(
                        self.name,
                        "concurrent_so_mixed",
                        Verdict.FAIL,
                        detail=f"{len(failures)} failures: {'; '.join(failures[:3])}",
                    )
                )
            else:
                results.append(
                    self.make_result(
                        self.name,
                        "concurrent_so_mixed",
                        Verdict.PASS,
                        detail=f"5 concurrent SO requests OK (budget_exhausted={budget_hits})",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "concurrent_so_mixed",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # -- 10. error_invalid_schema ------------------------------------------
        try:
            # Schema where top-level type is not "object" -- servers typically
            # require the root schema to be an object type for json_schema mode
            invalid_schema = {
                "type": "string",
            }

            def _invalid_schema() -> Any:
                return validator.so_chat(
                    [{"role": "user", "content": "Return anything."}],
                    invalid_schema,
                    max_tokens=128,
                )

            try:
                resp = await loop.run_in_executor(None, _invalid_schema)
                # If it succeeds, that's interesting but not necessarily wrong
                results.append(
                    self.make_result(
                        self.name,
                        "error_invalid_schema",
                        Verdict.INTERESTING,
                        detail=f"server accepted non-object root schema (finish_reason={resp.choices[0].finish_reason})",
                    )
                )
            except Exception as inner_e:
                err_str = str(inner_e).lower()
                if (
                    "400" in err_str
                    or "invalid" in err_str
                    or "must be" in err_str
                ):
                    results.append(
                        self.make_result(
                            self.name,
                            "error_invalid_schema",
                            Verdict.PASS,
                            detail=f"server correctly rejected invalid schema: {inner_e}",
                        )
                    )
                elif "500" in err_str or "internal" in err_str:
                    results.append(
                        self.make_result(
                            self.name,
                            "error_invalid_schema",
                            Verdict.FAIL,
                            detail=f"server 500 on invalid schema (should be 4xx): {inner_e}",
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "error_invalid_schema",
                            Verdict.INTERESTING,
                            detail=f"unexpected error on invalid schema: {inner_e}",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "error_invalid_schema",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # -- 11. 50_property_object --------------------------------------------
        try:
            properties = {}
            required = []
            for i in range(50):
                prop_name = f"field_{i:02d}"
                properties[prop_name] = {"type": "string"}
                required.append(prop_name)

            schema = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

            def _fifty_props() -> Any:
                return validator.so_chat(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Fill all 50 fields. For each field_XX, return the string 'val_XX'. "
                                "For example field_00='val_00', field_01='val_01', etc."
                            ),
                        }
                    ],
                    schema,
                    max_tokens=4096,
                )

            resp = await loop.run_in_executor(None, _fifty_props)
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "50_property_object",
                        Verdict.PASS,
                        detail="budget exhausted",
                    )
                )
            else:
                content = resp.choices[0].message.content
                data = json.loads(content)
                errs = validate_schema(data, schema)
                present_count = sum(1 for k in required if k in data)
                if errs:
                    results.append(
                        self.make_result(
                            self.name,
                            "50_property_object",
                            Verdict.FAIL,
                            detail=f"{present_count}/50 fields present, errors: {'; '.join(errs[:5])}",
                            response_body=content[:500],
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "50_property_object",
                            Verdict.PASS,
                            detail=f"all {present_count}/50 fields present and valid",
                        )
                    )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "50_property_object", Verdict.ERROR, error=str(e)
                )
            )

        return results
