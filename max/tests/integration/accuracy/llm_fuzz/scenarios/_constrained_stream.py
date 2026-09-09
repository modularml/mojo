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
"""Reassembles and validates constrained streaming responses."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from jsonschema import Draft7Validator


@dataclass
class ToolCall:
    name: str = ""
    arguments: str = ""


@dataclass
class StreamedTurn:
    reasoning: str = ""
    content: str = ""
    tool_calls: dict[int, ToolCall] = field(default_factory=dict)
    finish_reason: str | None = None
    protocol_errors: list[str] = field(default_factory=list)


def _as_object(
    value: object, label: str, errors: list[str]
) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    errors.append(f"{label} is not an object")
    return None


def _append_str(value: object, label: str, errors: list[str]) -> str:
    if isinstance(value, str):
        return value
    if value is not None:
        errors.append(f"{label} is not a string")
    return ""


def accumulate_stream(chunks: Iterable[str]) -> StreamedTurn:
    """Reassembles one chat-completion response from SSE data payloads."""
    response = StreamedTurn()
    errors = response.protocol_errors
    for raw in chunks:
        if raw == "[DONE]":
            break
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(f"stream chunk is not JSON: {raw[:60]!r}")
            continue
        parsed = _as_object(chunk, "stream chunk", errors)
        if parsed is None:
            continue
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = _as_object(choices[0], "stream choice", errors)
        if choice is None:
            continue
        delta_value = choice.get("delta")
        delta = _as_object(
            {} if delta_value is None else delta_value, "stream delta", errors
        )
        if delta is None:
            continue

        response.content += _append_str(
            delta.get("content"), "content fragment", errors
        )
        response.reasoning += _append_str(
            delta.get("reasoning_content") or delta.get("reasoning"),
            "reasoning fragment",
            errors,
        )

        calls = delta.get("tool_calls")
        if calls is None:
            calls = []
        elif not isinstance(calls, list):
            errors.append("tool_calls fragment is not a list")
            calls = []
        for call in calls:
            parsed_call = _as_object(call, "tool call fragment", errors)
            if parsed_call is None:
                continue
            index = parsed_call.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                errors.append("tool call index is not an integer")
                continue
            tool_call = response.tool_calls.setdefault(index, ToolCall())
            function_value = parsed_call.get("function")
            function = _as_object(
                {} if function_value is None else function_value,
                "tool call function",
                errors,
            )
            if function is None:
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                tool_call.name = name
            elif name is not None:
                errors.append("tool call name is not a string")
            tool_call.arguments += _append_str(
                function.get("arguments"), "tool call arguments", errors
            )

        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            response.finish_reason = finish_reason
    if response.finish_reason is None:
        errors.append("stream finished without a finish_reason")
    return response


def parse_json_document(
    document: object, *, label: str
) -> tuple[object, str | None]:
    """Parses a JSON document or returns a bounded diagnostic."""
    if not isinstance(document, (str, bytes, bytearray)):
        return None, f"{label} is not JSON: {repr(document)[:60]}"
    try:
        return json.loads(document), None
    except json.JSONDecodeError:
        return None, f"{label} is not JSON: {repr(document)[:60]}"


def classify_served_json(
    body: str, schema: Mapping[str, object]
) -> tuple[str, str | None]:
    """Grades a non-streaming chat-completion body against ``schema``.

    Returns ``truncated``, ``conformant``, or ``invalid``. Length
    truncation is not conformance.
    """
    try:
        choice = json.loads(body)["choices"][0]
        content = choice["message"]["content"]
        finish = choice.get("finish_reason")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "invalid", "served but body was not schema JSON"
    if finish == "length":
        return "truncated", None
    parsed, parse_error = parse_json_document(content, label="served content")
    if parse_error:
        return "invalid", parse_error
    try:
        errors = sorted(Draft7Validator(schema).iter_errors(parsed), key=str)
    except Exception as error:
        return "invalid", f"schema not judgeable: {error}"
    if not errors:
        return "conformant", None
    shown = [f"{item.validator}@{item.json_path}" for item in errors[:2]]
    return "invalid", f"{repr(content)[:70]} -> {shown}"


def grade_admission(
    expect: str,
    attempts: int,
    refused: int,
    served: int,
    conformant: int,
    truncated: int,
) -> str:
    """Grades one admission case from its attempt counters.

    A serve case is ``pass`` only when every attempt was a 200 and every
    non-truncated body conformed. ``interesting`` is reserved for an
    all-200 all-truncated run. A refuse case is ``pass`` only when every
    attempt was a 400.
    """
    if expect == "refuse":
        return "pass" if refused == attempts else "fail"
    if served != attempts:
        return "fail"
    if truncated == attempts:
        return "interesting"
    judged = served - truncated
    if judged > 0 and conformant == judged:
        return "pass"
    return "fail"


def constrained_document_errors(
    response: StreamedTurn,
    schema: Mapping[str, object],
    *,
    content_is_constrained: bool,
    required_tool_name: str | None,
) -> list[str]:
    """Returns protocol and schema failures for every emitted document."""
    errors = list(response.protocol_errors)
    documents: list[tuple[str, str]] = []
    if content_is_constrained and response.content:
        documents.append(("content", response.content))

    for index, tool_call in sorted(response.tool_calls.items()):
        label = f"tool_calls[{index}]"
        documents.append((f"{label}.arguments", tool_call.arguments))
        if required_tool_name and tool_call.name != required_tool_name:
            errors.append(
                f"{label}.name is {tool_call.name!r}, expected {required_tool_name!r}"
            )

    if required_tool_name and not response.tool_calls:
        errors.append(f"required tool {required_tool_name!r} was not called")
    if not documents and not errors:
        errors.append("response emitted no constrained document")

    validator = Draft7Validator(schema)
    for label, document in documents:
        instance, parse_error = parse_json_document(document, label=label)
        if parse_error:
            errors.append(parse_error)
            continue
        errors.extend(
            f"{label}: {error.validator}@{error.json_path}"
            for error in sorted(validator.iter_errors(instance), key=str)
        )
    return errors
