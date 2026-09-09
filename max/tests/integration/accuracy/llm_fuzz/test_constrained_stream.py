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

import json

from scenarios._constrained_stream import (
    StreamedTurn,
    ToolCall,
    accumulate_stream,
    classify_served_json,
    constrained_document_errors,
    grade_admission,
    parse_json_document,
)

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "unit": {"type": "string", "enum": ["seconds", "minutes"]},
    },
    "required": ["answer", "unit"],
    "additionalProperties": False,
}

_VALID_DOCUMENT = '{"answer": 3, "unit": "seconds"}'


def _chunk(delta: object, finish_reason: str | None = None) -> str:
    return json.dumps(
        {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    )


def test_accumulate_stream_keeps_content_and_indexed_tool_calls() -> None:
    response = accumulate_stream(
        [
            _chunk({"reasoning_content": "work "}),
            _chunk(
                {
                    "content": '{"answer": 3, ',
                    "tool_calls": [
                        {
                            "index": 1,
                            "function": {
                                "name": "record_duration",
                                "arguments": '{"answer": 4, ',
                            },
                        },
                        {
                            "index": 0,
                            "function": {
                                "name": "record_duration",
                                "arguments": '{"answer": 3, ',
                            },
                        },
                    ],
                }
            ),
            _chunk(
                {
                    "content": '"unit": "seconds"}',
                    "reasoning": "done",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '"unit": "seconds"}'},
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '"unit": "minutes"}'},
                        },
                    ],
                },
                "stop",
            ),
            "[DONE]",
        ]
    )

    assert response.reasoning == "work done"
    assert response.content == _VALID_DOCUMENT
    assert response.finish_reason == "stop"
    assert set(response.tool_calls) == {0, 1}
    assert response.tool_calls[0] == ToolCall(
        name="record_duration", arguments=_VALID_DOCUMENT
    )
    assert response.tool_calls[1] == ToolCall(
        name="record_duration",
        arguments='{"answer": 4, "unit": "minutes"}',
    )


def test_validation_checks_content_and_every_tool_call() -> None:
    response = StreamedTurn(
        content='{"answer": "three", "unit": "seconds"}',
        tool_calls={
            0: ToolCall("record_duration", _VALID_DOCUMENT),
            1: ToolCall("record_duration", '{"answer": 4, "unit": "hours"}'),
        },
    )

    errors = constrained_document_errors(
        response,
        _SCHEMA,
        content_is_constrained=True,
        required_tool_name=None,
    )

    assert any(error.startswith("content:") for error in errors)
    assert any(error.startswith("tool_calls[1].arguments:") for error in errors)
    assert not any(
        error.startswith("tool_calls[0].arguments:") for error in errors
    )


def test_forced_tool_requires_the_named_call() -> None:
    missing_errors = constrained_document_errors(
        StreamedTurn(content=_VALID_DOCUMENT),
        _SCHEMA,
        content_is_constrained=False,
        required_tool_name="record_duration",
    )
    wrong_name_errors = constrained_document_errors(
        StreamedTurn(tool_calls={0: ToolCall("other", _VALID_DOCUMENT)}),
        _SCHEMA,
        content_is_constrained=False,
        required_tool_name="record_duration",
    )
    valid_errors = constrained_document_errors(
        StreamedTurn(
            tool_calls={0: ToolCall("record_duration", _VALID_DOCUMENT)}
        ),
        _SCHEMA,
        content_is_constrained=False,
        required_tool_name="record_duration",
    )

    assert missing_errors == ["required tool 'record_duration' was not called"]
    assert any(
        "expected 'record_duration'" in error for error in wrong_name_errors
    )
    assert valid_errors == []


def test_combined_constraint_allows_a_tool_only_response() -> None:
    errors = constrained_document_errors(
        StreamedTurn(
            tool_calls={0: ToolCall("record_duration", _VALID_DOCUMENT)}
        ),
        _SCHEMA,
        content_is_constrained=True,
        required_tool_name=None,
    )

    assert errors == []


def test_null_content_with_a_tool_call_is_allowed() -> None:
    response = accumulate_stream(
        [
            _chunk(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "name": "record_duration",
                                "arguments": _VALID_DOCUMENT,
                            },
                        }
                    ],
                },
                "stop",
            )
        ]
    )

    assert (
        constrained_document_errors(
            response,
            _SCHEMA,
            content_is_constrained=True,
            required_tool_name=None,
        )
        == []
    )


def test_null_document_returns_a_bounded_error() -> None:
    document, error = parse_json_document(None, label="served content")

    assert document is None
    assert error == "served content is not JSON: None"


def test_non_json_stream_chunk_fails_an_otherwise_valid_response() -> None:
    response = accumulate_stream(
        [
            "not-json",
            _chunk({"content": _VALID_DOCUMENT}, "stop"),
        ]
    )

    assert constrained_document_errors(
        response,
        _SCHEMA,
        content_is_constrained=True,
        required_tool_name=None,
    ) == ["stream chunk is not JSON: 'not-json'"]


def test_missing_tool_call_index_is_a_protocol_error() -> None:
    response = accumulate_stream(
        [
            _chunk(
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "record_duration",
                                "arguments": _VALID_DOCUMENT,
                            }
                        }
                    ]
                },
                "stop",
            )
        ]
    )

    assert response.tool_calls == {}
    assert constrained_document_errors(
        response,
        _SCHEMA,
        content_is_constrained=False,
        required_tool_name="record_duration",
    ) == [
        "tool call index is not an integer",
        "required tool 'record_duration' was not called",
    ]


def test_length_truncation_is_not_conformance() -> None:
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": "{"},
                    "finish_reason": "length",
                }
            ]
        }
    )

    assert classify_served_json(body, _SCHEMA) == ("truncated", None)


def test_complete_invalid_json_is_not_conformance() -> None:
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": "{"},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    grade, detail = classify_served_json(body, _SCHEMA)
    assert grade == "invalid"
    assert detail is not None


def test_complete_valid_json_is_conformant() -> None:
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": _VALID_DOCUMENT},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert classify_served_json(body, _SCHEMA) == ("conformant", None)


def test_falsey_malformed_delta_is_a_protocol_error() -> None:
    response = accumulate_stream(
        [
            _chunk([]),
            _chunk({"content": _VALID_DOCUMENT}, "stop"),
        ]
    )

    assert "stream delta is not an object" in response.protocol_errors
    assert constrained_document_errors(
        response,
        _SCHEMA,
        content_is_constrained=True,
        required_tool_name=None,
    ) == ["stream delta is not an object"]


def test_falsey_malformed_tool_calls_is_a_protocol_error() -> None:
    response = accumulate_stream(
        [_chunk({"content": _VALID_DOCUMENT, "tool_calls": {}}, "stop")]
    )

    assert response.protocol_errors == ["tool_calls fragment is not a list"]


def test_boolean_tool_call_index_is_a_protocol_error() -> None:
    response = accumulate_stream(
        [
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": True,
                            "function": {
                                "name": "record_duration",
                                "arguments": _VALID_DOCUMENT,
                            },
                        }
                    ]
                },
                "stop",
            )
        ]
    )

    assert response.tool_calls == {}
    assert response.protocol_errors == ["tool call index is not an integer"]


def test_missing_finish_reason_is_a_protocol_error() -> None:
    response = accumulate_stream(
        [_chunk({"content": _VALID_DOCUMENT}), "[DONE]"]
    )

    assert response.finish_reason is None
    assert response.protocol_errors == [
        "stream finished without a finish_reason"
    ]
    assert constrained_document_errors(
        response,
        _SCHEMA,
        content_is_constrained=True,
        required_tool_name=None,
    ) == ["stream finished without a finish_reason"]


def test_falsey_malformed_function_is_a_protocol_error() -> None:
    response = accumulate_stream(
        [
            _chunk(
                {
                    "tool_calls": [
                        {"index": 0, "function": []},
                    ]
                },
                "stop",
            )
        ]
    )

    assert response.protocol_errors == ["tool call function is not an object"]


def test_serve_case_passes_when_non_truncated_bodies_conform() -> None:
    assert (
        grade_admission(
            "serve", attempts=3, refused=0, served=3, conformant=1, truncated=2
        )
        == "pass"
    )


def test_serve_case_fails_when_attempts_are_refused() -> None:
    assert (
        grade_admission(
            "serve", attempts=3, refused=3, served=0, conformant=0, truncated=0
        )
        == "fail"
    )


def test_serve_case_fails_on_partial_transport_loss() -> None:
    assert (
        grade_admission(
            "serve", attempts=3, refused=0, served=2, conformant=2, truncated=0
        )
        == "fail"
    )


def test_serve_case_all_truncated_is_interesting() -> None:
    assert (
        grade_admission(
            "serve", attempts=3, refused=0, served=3, conformant=0, truncated=3
        )
        == "interesting"
    )


def test_serve_case_requires_every_attempt_served() -> None:
    assert (
        grade_admission(
            "serve", attempts=3, refused=0, served=3, conformant=3, truncated=0
        )
        == "pass"
    )


def test_refuse_case_requires_every_attempt_refused() -> None:
    assert (
        grade_admission(
            "refuse", attempts=3, refused=3, served=0, conformant=0, truncated=0
        )
        == "pass"
    )
    assert (
        grade_admission(
            "refuse", attempts=3, refused=2, served=1, conformant=1, truncated=0
        )
        == "fail"
    )
