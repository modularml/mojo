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
"""Observability-only schema-conformance checks for generated output.

Runs JSON Schema validation on parsed tool-call arguments and on
``response_format`` (json_schema / json_object) final content purely to emit
structured, PII-free signals to the serve log. It never mutates the response,
never repairs arguments, and never raises into the request path. The goal is to
turn the coarse production "Schema Mismatch" rate into a per-keyword/per-path
distribution so the failure modes can be sized.

PII: only schema-defined names reach the log -- the function name, the failing
validator keyword, and the JSON path (object/property names, which come from
the developer-supplied schema). Argument *values* and validator error messages
(which embed the offending value) are never logged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from jsonschema import Draft7Validator
from jsonschema.protocols import Validator

logger = logging.getLogger("max.serve")

# Cap errors recorded per call so a deeply-wrong object can't bloat a log line.
_MAX_ERRORS_PER_CALL = 5

# Bound on the number of compiled validators kept in memory. Schemas are
# client-supplied per request, so an unbounded cache would grow without limit
# under high schema cardinality (a memory leak). Building a validator is the
# only non-trivial cost (validation itself is microseconds); an LRU keeps that
# off the hot path while capping retained memory. lru_cache is internally
# locked, so concurrent builds are safe.
_VALIDATOR_CACHE_SIZE = 1024

ToolCallOutcome = Literal[
    "valid", "invalid_json", "unknown_tool", "schema_mismatch"
]

ResponseFormatOutcome = Literal["valid", "invalid_json", "schema_mismatch"]


@dataclass
class ToolCallConformance:
    """Result of validating one tool call's arguments against its schema."""

    function: str
    outcome: ToolCallOutcome
    # "keyword@json_path" pairs; schema-defined names only, no instance values.
    errors: list[str] = field(default_factory=list)
    # Count of errors beyond _MAX_ERRORS_PER_CALL that were not recorded.
    additional_error_count: int = 0


@dataclass
class ResponseFormatConformance:
    """Result of validating final content against a response_format schema."""

    outcome: ResponseFormatOutcome
    # "keyword@json_path" pairs; schema-defined names only, no instance values.
    errors: list[str] = field(default_factory=list)
    # Count of errors beyond _MAX_ERRORS_PER_CALL that were not recorded.
    additional_error_count: int = 0


@lru_cache(maxsize=_VALIDATOR_CACHE_SIZE)
def _build_validator(schema_key: str) -> Validator | None:
    """Returns an LRU-cached Draft 7 validator for a schema, or ``None``.

    *schema_key* is the canonical ``json.dumps`` of the schema (sorted keys),
    used both as the cache key and as the source to rebuild from, so it is
    always valid JSON. Returns ``None`` when the schema cannot be compiled.

    Validates under JSON Schema Draft 7 to match the OpenRouter evaluator that
    scores our tool-call error rate under that dialect. Draft 7 is pinned
    unconditionally -- any ``$schema`` the caller declares is ignored.
    """
    schema = json.loads(schema_key)
    # check_schema rejects an invalid tool-definition schema (SchemaError); a
    # bad tool definition is not a model failure, so skip rather than blame it.
    # Broad except keeps this observability path from ever raising.
    try:
        Draft7Validator.check_schema(schema)
        return Draft7Validator(schema)
    except Exception:
        return None


def check_tool_call_conformance(
    calls: list[tuple[str, object]],
    schemas_by_name: Mapping[str, Mapping[str, Any]],
    known_tools: Collection[str] | None = None,
) -> list[ToolCallConformance]:
    """Validates each ``(name, arguments)`` call against its declared schema.

    Pure and side-effect free; never raises. ``arguments`` is the raw JSON
    string emitted by the model (an already-decoded mapping is also accepted).
    An empty/whitespace argument string is treated as ``{}`` (a no-arg call).
    A schema that cannot be compiled yields ``valid`` -- this check never
    invents a failure it cannot substantiate.

    ``known_tools`` is the full set of declared tool names. A call to a tool in
    ``known_tools`` that has no entry in ``schemas_by_name`` (a legitimately
    parameter-less tool) yields ``valid``; only a call to a name that was never
    declared yields ``unknown_tool``. When ``known_tools`` is ``None`` the
    declared set is taken to be ``schemas_by_name`` (the prior behavior).
    """
    declared = (
        known_tools if known_tools is not None else schemas_by_name.keys()
    )
    results: list[ToolCallConformance] = []
    for name, raw_args in calls:
        schema = schemas_by_name.get(name)
        if schema is None:
            # A declared-but-schemaless tool has nothing to validate against;
            # only a name that was never declared is a genuine unknown tool.
            outcome: ToolCallOutcome = (
                "valid" if name in declared else "unknown_tool"
            )
            results.append(ToolCallConformance(name, outcome))
            continue

        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                results.append(ToolCallConformance(name, "invalid_json"))
                continue
        else:
            parsed = raw_args

        # Canonical JSON doubles as the validator's LRU cache key. An
        # unserializable schema (or one the backend cannot compile) yields no
        # validator, so we don't invent a failure we cannot substantiate.
        try:
            schema_key = json.dumps(
                schema, sort_keys=True, separators=(",", ":")
            )
            validator = _build_validator(schema_key)
        except (TypeError, ValueError):
            validator = None
        if validator is None:
            results.append(ToolCallConformance(name, "valid"))
            continue

        errors: list[str] = []
        additional_error_count = 0
        # iter_errors can raise on a schema that check_schema accepts but cannot
        # execute (e.g. an unresolvable ``$ref``). This path is observability
        # only and must never raise into the request, so treat any such failure
        # as ``valid`` rather than inventing a mismatch it cannot substantiate.
        try:
            for err in validator.iter_errors(parsed):
                if len(errors) < _MAX_ERRORS_PER_CALL:
                    errors.append(f"{err.validator}@{err.json_path}")
                else:
                    additional_error_count += 1
        except Exception:
            results.append(ToolCallConformance(name, "valid"))
            continue
        results.append(
            ToolCallConformance(
                name,
                "schema_mismatch" if errors else "valid",
                errors,
                additional_error_count,
            )
        )
    return results


def check_response_format_conformance(
    content: str,
    schema: Mapping[str, Any],
) -> ResponseFormatConformance:
    """Validates final message content against a ``response_format`` schema.

    Pure and side-effect free; never raises. *content* is the assembled
    assistant message content of a request that asked for
    ``response_format`` json_schema/json_object. Unlike tool-call arguments,
    empty content is not a valid no-op: the client asked for a JSON document
    and got none, so it classifies as ``invalid_json``. A schema that cannot
    be compiled yields ``valid`` -- this check never invents a failure it
    cannot substantiate.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return ResponseFormatConformance("invalid_json")

    try:
        schema_key = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        validator = _build_validator(schema_key)
    except (TypeError, ValueError):
        validator = None
    if validator is None:
        return ResponseFormatConformance("valid")

    errors: list[str] = []
    additional_error_count = 0
    # Mirrors check_tool_call_conformance: iter_errors can raise on a schema
    # check_schema accepts but cannot execute; observability must not raise.
    try:
        for err in validator.iter_errors(parsed):
            if len(errors) < _MAX_ERRORS_PER_CALL:
                errors.append(f"{err.validator}@{err.json_path}")
            else:
                additional_error_count += 1
    except Exception:
        return ResponseFormatConformance("valid")
    return ResponseFormatConformance(
        "schema_mismatch" if errors else "valid", errors, additional_error_count
    )
