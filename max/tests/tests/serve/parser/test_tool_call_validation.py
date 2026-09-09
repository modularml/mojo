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
"""Tests for observability-only tool-call schema-conformance checking."""

from __future__ import annotations

from typing import Any

from max.serve.parser.tool_call_validation import (
    _VALIDATOR_CACHE_SIZE,
    _build_validator,
    check_response_format_conformance,
    check_tool_call_conformance,
)

_WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "count": {"type": "integer"},
        "unit": {"type": "string", "enum": ["C", "F"]},
    },
    "required": ["location"],
    "additionalProperties": False,
}
_SCHEMAS = {"get_weather": _WEATHER_SCHEMA, "no_args": {"type": "object"}}


def _only(calls: list[tuple[str, object]]) -> Any:
    [result] = check_tool_call_conformance(calls, _SCHEMAS)
    return result


def test_valid_args_are_conforming() -> None:
    r = _only([("get_weather", '{"location": "NYC", "unit": "F"}')])
    assert r.outcome == "valid"
    assert r.errors == []


def test_empty_args_string_is_treated_as_empty_object() -> None:
    # A no-arg tool whose schema requires nothing: empty string == {}.
    assert _only([("no_args", "")]).outcome == "valid"
    assert _only([("no_args", "   ")]).outcome == "valid"


def test_already_decoded_mapping_is_accepted() -> None:
    assert _only([("get_weather", {"location": "NYC"})]).outcome == "valid"


def test_invalid_json_is_flagged() -> None:
    r = _only([("get_weather", '{"location": "NYC"')])  # unterminated
    assert r.outcome == "invalid_json"
    assert r.errors == []


def test_unknown_tool_has_no_schema() -> None:
    assert _only([("nonexistent", "{}")]).outcome == "unknown_tool"


def test_missing_required_is_schema_mismatch() -> None:
    r = _only([("get_weather", '{"unit": "F"}')])
    assert r.outcome == "schema_mismatch"
    assert any(e.startswith("required@") for e in r.errors)


def test_wrong_type_reports_keyword_and_path() -> None:
    r = _only([("get_weather", '{"location": "NYC", "count": "five"}')])
    assert r.outcome == "schema_mismatch"
    assert "type@$.count" in r.errors


def test_enum_violation_reports_keyword() -> None:
    r = _only([("get_weather", '{"location": "NYC", "unit": "K"}')])
    assert r.outcome == "schema_mismatch"
    assert any(e.startswith("enum@") for e in r.errors)


def test_additional_properties_violation() -> None:
    r = _only([("get_weather", '{"location": "NYC", "x": 1}')])
    assert r.outcome == "schema_mismatch"
    assert any(e.startswith("additionalProperties@") for e in r.errors)


def test_errors_never_contain_argument_values() -> None:
    # PII guarantee: only schema-defined names (keyword@json_path) are recorded.
    secret = "user-secret-12345"
    r = _only([("get_weather", f'{{"location": "NYC", "count": "{secret}"}}')])
    assert r.outcome == "schema_mismatch"
    assert all(secret not in e for e in r.errors)


def test_multiple_calls_independently_classified() -> None:
    results = check_tool_call_conformance(
        [
            ("get_weather", '{"location": "NYC"}'),
            ("get_weather", "{}"),
            ("nonexistent", "{}"),
        ],
        _SCHEMAS,
    )
    assert [r.outcome for r in results] == [
        "valid",
        "schema_mismatch",
        "unknown_tool",
    ]


def test_draft7_tuple_items_is_enforced() -> None:
    """Confirms validation runs under Draft 7: array-form ``items`` is validated
    as a tuple (item i against schema i), a Draft 7 feature. Here the second
    tuple element must be an integer, so a string there is a mismatch."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pair": {
                "type": "array",
                "items": [{"type": "string"}, {"type": "integer"}],
            },
        },
        "required": ["pair"],
    }
    [r] = check_tool_call_conformance(
        [("f", '{"pair": ["a", "b"]}')], {"f": schema}
    )
    assert r.outcome == "schema_mismatch"
    assert "type@$.pair[1]" in r.errors


def test_oneof_still_flagged_under_draft7() -> None:
    """Draft 7 enforces ``oneOf`` as exactly-one, matching the evaluator, so an
    argument matching more than one branch is still a mismatch. We intentionally
    keep counting it (consistency with the evaluator)."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {"oneOf": [{"type": "number"}, {"type": "integer"}]},
        },
        "required": ["x"],
    }
    # ``5`` matches both the number and integer branches -> oneOf violated.
    [r] = check_tool_call_conformance([("f", '{"x": 5}')], {"f": schema})
    assert r.outcome == "schema_mismatch"
    assert any(e.startswith("oneOf@") for e in r.errors)


def test_unresolvable_ref_does_not_raise() -> None:
    """A schema that ``check_schema`` accepts but whose ``$ref`` cannot be
    resolved makes jsonschema's ``iter_errors`` raise. The check must swallow
    that and report ``valid`` rather than propagate into the request path."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {"$ref": "https://example.com/does-not-exist.json"}
        },
    }
    [r] = check_tool_call_conformance([("f", '{"x": 1}')], {"f": schema})
    assert r.outcome == "valid"
    assert r.errors == []


def test_declared_schemaless_tool_is_valid_not_unknown() -> None:
    """A tool declared without a ``parameters`` schema has no entry in
    ``schemas_by_name`` but is still a known tool: calling it is ``valid``,
    whereas a name that was never declared is ``unknown_tool``."""
    results = check_tool_call_conformance(
        [("no_params", "{}"), ("hallucinated", "{}")],
        {"get_weather": _WEATHER_SCHEMA},
        known_tools={"get_weather", "no_params"},
    )
    assert [r.outcome for r in results] == ["valid", "unknown_tool"]


def test_known_tools_defaults_to_schema_keys() -> None:
    """When ``known_tools`` is omitted the declared set is the schema keys, so a
    name absent from the schemas is still ``unknown_tool`` (prior behavior)."""
    [r] = check_tool_call_conformance([("no_params", "{}")], {})
    assert r.outcome == "unknown_tool"


def test_response_format_conforming_content_is_valid() -> None:
    r = check_response_format_conformance(
        '{"location": "NYC", "count": 3}', _WEATHER_SCHEMA
    )
    assert r.outcome == "valid"
    assert r.errors == []


def test_response_format_empty_content_is_invalid_json() -> None:
    """Unlike a no-arg tool call, empty content is not a valid no-op: the
    client asked for a JSON document and got none."""
    assert check_response_format_conformance("", _WEATHER_SCHEMA).outcome == (
        "invalid_json"
    )
    assert check_response_format_conformance(
        "   ", _WEATHER_SCHEMA
    ).outcome == ("invalid_json")


def test_response_format_truncated_json_is_invalid_json() -> None:
    r = check_response_format_conformance('{"location": "NYC"', _WEATHER_SCHEMA)
    assert r.outcome == "invalid_json"
    assert r.errors == []


def test_response_format_mismatch_reports_keyword_and_path() -> None:
    r = check_response_format_conformance(
        '{"location": "NYC", "count": "five"}', _WEATHER_SCHEMA
    )
    assert r.outcome == "schema_mismatch"
    assert "type@$.count" in r.errors


def test_response_format_errors_never_contain_content_values() -> None:
    secret = "hunter2-super-secret"
    r = check_response_format_conformance(
        f'{{"location": "{secret}", "count": "{secret}"}}', _WEATHER_SCHEMA
    )
    assert r.outcome == "schema_mismatch"
    assert all(secret not in e for e in r.errors)


def test_response_format_json_object_mode_schema() -> None:
    """json_object mode normalizes to ``{"type": "object"}``: any JSON object
    conforms, a bare scalar or array does not."""
    schema: dict[str, Any] = {"type": "object"}
    assert (
        check_response_format_conformance('{"a": 1}', schema).outcome == "valid"
    )
    assert (
        check_response_format_conformance("[1, 2]", schema).outcome
        == "schema_mismatch"
    )


def test_response_format_uncompilable_schema_is_valid() -> None:
    """A schema the validator cannot compile yields ``valid`` -- the check
    never invents a failure it cannot substantiate."""
    r = check_response_format_conformance(
        '{"a": 1}', {"type": "not-a-real-type"}
    )
    assert r.outcome == "valid"


def test_validator_cache_is_bounded() -> None:
    """Schemas are client-supplied, so the validator cache must stay bounded
    rather than grow without limit (a memory leak). Feeding more distinct
    schemas than the cap must not push the live cache past ``maxsize``."""
    assert _build_validator.cache_info().maxsize == _VALIDATOR_CACHE_SIZE
    for i in range(_VALIDATOR_CACHE_SIZE + 200):
        check_tool_call_conformance(
            [("f", "{}")],
            {"f": {"type": "object", "properties": {f"p{i}": {}}}},
        )
    assert _build_validator.cache_info().currsize <= _VALIDATOR_CACHE_SIZE
