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
Scenario: OpenAI API specification compliance
Target: Validate that responses conform to the OpenAI Chat Completions API spec.

Unlike other scenarios that try to break things, this scenario sends valid
requests and validates that the response structure, types, and values match
the OpenAI specification exactly. Failures here indicate spec divergence,
not crashes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from helpers import parse_json

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

TypeSpec = type | tuple[type, ...]


def _type_name(expected_type: TypeSpec) -> str:
    if isinstance(expected_type, tuple):
        return " | ".join(t.__name__ for t in expected_type)
    return expected_type.__name__


def _unexpected_status_verdict(status: int) -> Verdict:
    return Verdict.FAIL if status >= 500 else Verdict.INTERESTING


def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
    """Safely get the first choice from a response/chunk, handling empty choices arrays."""
    choices = data.get("choices", [])
    return choices[0] if choices else {}


def _check_fields(
    data: dict[str, Any],
    required: dict[str, TypeSpec],
    optional: dict[str, TypeSpec] | None = None,
) -> list[str]:
    """Validate required and optional fields exist and have correct types.

    Args:
        data: dict to validate
        required: mapping of field name -> expected type (use object for any)
        optional: mapping of field name -> expected type for optional fields

    Returns:
        list of error strings (empty = valid)
    """
    errors = []
    for field_name, expected_type in required.items():
        if field_name not in data:
            errors.append(f"missing required field '{field_name}'")
        elif expected_type is not object and not isinstance(
            data[field_name], expected_type
        ):
            actual = type(data[field_name]).__name__
            errors.append(
                f"field '{field_name}' expected {_type_name(expected_type)}, got {actual}"
            )
    if optional:
        for field_name, expected_type in optional.items():
            if field_name in data and data[field_name] is not None:
                if expected_type is not object and not isinstance(
                    data[field_name], expected_type
                ):
                    actual = type(data[field_name]).__name__
                    errors.append(
                        f"optional field '{field_name}' expected {_type_name(expected_type)}, got {actual}"
                    )
    return errors


def _validate_chat_completion(data: dict[str, Any]) -> list[str]:
    """Validate a ChatCompletion response object against the OpenAI spec."""
    errors = _check_fields(
        data,
        {
            "id": str,
            "object": str,
            "created": int,
            "model": str,
            "choices": list,
        },
        optional={
            "usage": dict,
            "system_fingerprint": str,
            "service_tier": str,
        },
    )

    if data.get("object") != "chat.completion":
        errors.append(
            f"'object' must be 'chat.completion', got '{data.get('object')}'"
        )

    if not data.get("id", "").strip():
        errors.append("'id' must be a non-empty string")

    usage = data.get("usage")
    if usage is not None:
        errors.extend(_validate_usage(usage))

    choices = data.get("choices", [])
    if not isinstance(choices, list):
        return errors

    for i, choice in enumerate(choices):
        if not isinstance(choice, dict):
            errors.append(
                f"choice[{i}] must be an object, got {type(choice).__name__}"
            )
            continue
        errors.extend(_validate_choice(choice, i))

    return errors


def _validate_usage(usage: dict[str, Any]) -> list[str]:
    errors = _check_fields(
        usage,
        {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
        },
        optional={
            "prompt_tokens_details": dict,
            "completion_tokens_details": dict,
        },
    )
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    tt = usage.get("total_tokens", 0)
    if isinstance(pt, int) and isinstance(ct, int) and isinstance(tt, int):
        if tt != pt + ct:
            errors.append(
                f"total_tokens ({tt}) != prompt_tokens ({pt}) + completion_tokens ({ct})"
            )
        if pt < 0:
            errors.append(f"prompt_tokens is negative: {pt}")
        if ct < 0:
            errors.append(f"completion_tokens is negative: {ct}")
    return errors


VALID_FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
}


def _validate_choice(choice: dict[str, Any], index: int) -> list[str]:
    prefix = f"choices[{index}]"
    errors = _check_fields(
        choice,
        {
            "index": int,
            "message": dict,
            "finish_reason": str,
        },
        optional={
            "logprobs": (dict, type(None)),
        },
    )

    fr = choice.get("finish_reason")
    if isinstance(fr, str) and fr not in VALID_FINISH_REASONS:
        errors.append(
            f"{prefix}.finish_reason '{fr}' not in {VALID_FINISH_REASONS}"
        )

    if choice.get("index") != index:
        errors.append(
            f"{prefix}.index is {choice.get('index')}, expected {index}"
        )

    msg = choice.get("message")
    if isinstance(msg, dict):
        errors.extend(_validate_message(msg, prefix))
    return errors


def _validate_message(msg: dict[str, Any], prefix: str) -> list[str]:
    errors = []
    if "role" not in msg:
        errors.append(f"{prefix}.message missing 'role'")
    elif msg["role"] != "assistant":
        errors.append(
            f"{prefix}.message.role must be 'assistant', got '{msg['role']}'"
        )

    has_content = "content" in msg and msg["content"] is not None
    has_tool_calls = "tool_calls" in msg and msg["tool_calls"] is not None
    has_function_call = (
        "function_call" in msg and msg["function_call"] is not None
    )

    if not has_content and not has_tool_calls and not has_function_call:
        errors.append(
            f"{prefix}.message has no content, tool_calls, or function_call"
        )

    if has_content and not isinstance(msg["content"], str):
        errors.append(
            f"{prefix}.message.content must be string, got {type(msg['content']).__name__}"
        )

    if has_tool_calls:
        errors.extend(_validate_tool_calls(msg["tool_calls"], prefix))

    return errors


def _validate_tool_calls(tool_calls: object, prefix: str) -> list[str]:
    errors = []
    if not isinstance(tool_calls, list):
        return [f"{prefix}.message.tool_calls must be array"]
    for i, tc in enumerate(tool_calls):
        tc_prefix = f"{prefix}.message.tool_calls[{i}]"
        if not isinstance(tc, dict):
            errors.append(f"{tc_prefix} must be object")
            continue
        tc_errors = _check_fields(tc, {"id": str, "type": str})
        errors.extend(f"{tc_prefix} {err}" for err in tc_errors)

        tc_type = tc.get("type")
        if not isinstance(tc_type, str):
            continue
        if tc_type != "function":
            errors.append(
                f"{tc_prefix}.type must be 'function', got '{tc_type}'"
            )
            continue

        fn = tc.get("function")
        if not isinstance(fn, dict):
            errors.append(f"{tc_prefix}.function must be object")
            continue

        fn_errors = _check_fields(fn, {"name": str, "arguments": str})
        errors.extend(f"{tc_prefix}.function {err}" for err in fn_errors)
    return errors


def _validate_chunk(
    chunk: dict[str, Any], chunk_index: int, is_first: bool
) -> list[str]:
    """Validate a single SSE chunk against the ChatCompletionChunk spec."""
    errors = _check_fields(
        chunk,
        {
            "id": str,
            "object": str,
            "created": int,
            "model": str,
            "choices": list,
        },
        optional={
            "system_fingerprint": str,
            "service_tier": str,
            "usage": (dict, type(None)),
        },
    )

    if chunk.get("object") != "chat.completion.chunk":
        errors.append(
            f"chunk[{chunk_index}].object must be 'chat.completion.chunk', got '{chunk.get('object')}'"
        )

    choices = chunk.get("choices", [])
    if not isinstance(choices, list):
        return errors

    for ci, choice in enumerate(choices):
        cp = f"chunk[{chunk_index}].choices[{ci}]"
        if not isinstance(choice, dict):
            errors.append(f"{cp} must be object, got {type(choice).__name__}")
            continue
        if "delta" not in choice:
            errors.append(f"{cp} missing 'delta'")
        if "index" not in choice:
            errors.append(f"{cp} missing 'index'")

        delta = choice.get("delta", {})
        if isinstance(delta, dict):
            if is_first and chunk_index == 0:
                if (
                    delta.get("role") is not None
                    and delta["role"] != "assistant"
                ):
                    errors.append(
                        f"{cp}.delta.role should be 'assistant' in first chunk, got '{delta.get('role')}'"
                    )

        fr = choice.get("finish_reason")
        if fr is not None and fr not in VALID_FINISH_REASONS:
            errors.append(
                f"{cp}.finish_reason '{fr}' not in {VALID_FINISH_REASONS}"
            )

    return errors


@register_scenario
class OpenAISpecCompliance(BaseScenario):
    name = "openai_spec_compliance"
    description = "Validate endpoint responses conform to the OpenAI Chat Completions API specification"
    tags = ["spec", "compliance", "validation", "openai"]
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        def req(content: str = "Say hello", **extra: Any) -> dict[str, Any]:
            # 200 lets reasoning models finish their <think> block and still
            # emit a visible answer (json_object reasoning alone needs ~140
            # tokens). Well-behaved completions stop well under the cap; the
            # truncation test overrides max_tokens explicitly.
            p = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 200,
            }
            p.update(extra)
            return p

        # ----- 1. Basic response structure -----
        resp = await client.post_json(req())
        if resp.error:
            results.append(
                self.make_result(
                    self.name,
                    "basic_response_structure",
                    Verdict.ERROR,
                    detail=f"Request failed: {resp.error}",
                )
            )
        elif resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "basic_response_structure",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"Expected 200, got {resp.status}",
                )
            )
        else:
            data, err = parse_json(resp.body)
            if err or data is None:
                results.append(
                    self.make_result(
                        self.name,
                        "basic_response_structure",
                        Verdict.FAIL,
                        status_code=200,
                        detail=f"Response is not valid JSON: {err}",
                    )
                )
            else:
                errors = _validate_chat_completion(data)
                if errors:
                    results.append(
                        self.make_result(
                            self.name,
                            "basic_response_structure",
                            Verdict.FAIL,
                            status_code=200,
                            detail="; ".join(errors),
                            response_body=resp.body[:2000],
                        )
                    )
                else:
                    results.append(
                        self.make_result(
                            self.name,
                            "basic_response_structure",
                            Verdict.PASS,
                            status_code=200,
                            detail="Response conforms to ChatCompletion spec",
                        )
                    )

        # ----- 2. finish_reason=stop on normal completion -----
        resp = await client.post_json(req("Say the word yes"))
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                fr = _first_choice(data).get("finish_reason")
                if fr == "stop":
                    verdict, detail = Verdict.PASS, "finish_reason is 'stop'"
                elif fr in VALID_FINISH_REASONS:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        f"finish_reason is '{fr}', expected 'stop'",
                    )
                else:
                    verdict, detail = (
                        Verdict.FAIL,
                        f"Invalid finish_reason: '{fr}'",
                    )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        else:
            verdict = _unexpected_status_verdict(resp.status)
            detail = f"Status {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "finish_reason_stop",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 3. finish_reason=length when truncated -----
        resp = await client.post_json(
            req(
                "Write a very long essay about the history of computing",
                max_tokens=5,
            )
        )
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                fr = _first_choice(data).get("finish_reason")
                if fr == "length":
                    verdict, detail = (
                        Verdict.PASS,
                        "finish_reason is 'length' when max_tokens hit",
                    )
                elif fr == "stop":
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "finish_reason is 'stop' despite very low max_tokens",
                    )
                else:
                    verdict, detail = (
                        Verdict.FAIL,
                        f"Expected 'length', got '{fr}'",
                    )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        else:
            verdict = _unexpected_status_verdict(resp.status)
            detail = f"Status {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "finish_reason_length",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 4. Usage token counts are consistent -----
        resp = await client.post_json(req("Hello"))
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                usage = data.get("usage")
                if usage is None:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "No usage field in response",
                    )
                else:
                    errs = _validate_usage(usage)
                    if errs:
                        verdict, detail = Verdict.FAIL, "; ".join(errs)
                    else:
                        verdict, detail = (
                            Verdict.PASS,
                            f"Usage valid: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}, total={usage.get('total_tokens')}",
                        )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        else:
            verdict = _unexpected_status_verdict(resp.status)
            detail = f"Status {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "usage_token_consistency",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 5. Multiple choices with n > 1 -----
        resp = await client.post_json(req("Say a number", n=3))
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                choices = data.get("choices", [])
                if len(choices) == 3:
                    indices = [c.get("index") for c in choices]
                    if indices == [0, 1, 2]:
                        verdict, detail = (
                            Verdict.PASS,
                            "3 choices with correct indices",
                        )
                    else:
                        verdict, detail = (
                            Verdict.FAIL,
                            f"Choice indices are {indices}, expected [0,1,2]",
                        )
                elif len(choices) == 1:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "Server returned 1 choice despite n=3 (may not support n>1)",
                    )
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        f"Expected 3 choices, got {len(choices)}",
                    )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        elif resp.status == 400:
            verdict, detail = Verdict.INTERESTING, "Server rejects n>1 (400)"
        else:
            verdict = _unexpected_status_verdict(resp.status)
            detail = f"Status {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "n_multiple_choices",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 6. System message accepted -----
        resp = await client.post_json(
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": "Say hello"},
                ],
                "max_tokens": 30,
            }
        )
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                errs = _validate_chat_completion(data)
                verdict = Verdict.FAIL if errs else Verdict.PASS
                detail = (
                    "; ".join(errs)
                    if errs
                    else "System message accepted, valid response"
                )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        else:
            verdict = (
                Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            )
            detail = f"Status {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "system_message",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 7. Multi-turn conversation -----
        resp = await client.post_json(
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "max_tokens": 30,
            }
        )
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                errs = _validate_chat_completion(data)
                verdict = Verdict.FAIL if errs else Verdict.PASS
                detail = (
                    "; ".join(errs)
                    if errs
                    else "Multi-turn conversation accepted"
                )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        else:
            verdict = (
                Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            )
            detail = f"Status {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "multi_turn_conversation",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 8. Content as array of parts -----
        resp = await client.post_json(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": " world"},
                        ],
                    }
                ],
                "max_tokens": 30,
            }
        )
        if resp.status == 200:
            data, _ = parse_json(resp.body)
            if data:
                errs = _validate_chat_completion(data)
                verdict = Verdict.FAIL if errs else Verdict.PASS
                detail = (
                    "; ".join(errs) if errs else "Array content parts accepted"
                )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        elif resp.status == 400:
            verdict, detail = (
                Verdict.INTERESTING,
                "Server rejects array content parts (400)",
            )
        else:
            verdict = (
                Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            )
            detail = f"Status {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "content_array_parts",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 9. Streaming response spec compliance -----
        resp = await client.post_streaming(req("Count from 1 to 5"))
        if resp.error:
            results.append(
                self.make_result(
                    self.name,
                    "streaming_chunk_format",
                    Verdict.ERROR,
                    detail=f"Stream error: {resp.error}",
                )
            )
        elif resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "streaming_chunk_format",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"Expected 200, got {resp.status}",
                )
            )
        else:
            chunks = resp.chunks or []
            all_errors = []
            has_done = False
            data_chunks = []
            for raw in chunks:
                if raw == "[DONE]":
                    has_done = True
                    continue
                chunk_data, err = parse_json(raw)
                if err or chunk_data is None:
                    all_errors.append(f"Unparsable chunk: {raw[:100]}")
                    continue
                data_chunks.append(chunk_data)

            for ci, cd in enumerate(data_chunks):
                all_errors.extend(_validate_chunk(cd, ci, ci == 0))

            # Verify consistent id across chunks
            if len(data_chunks) >= 2:
                ids = {cd.get("id") for cd in data_chunks}
                if len(ids) > 1:
                    all_errors.append(f"Inconsistent chunk IDs: {ids}")

            if not has_done:
                all_errors.append("Stream did not end with [DONE]")

            if not data_chunks:
                all_errors.append("No data chunks received")

            if all_errors:
                results.append(
                    self.make_result(
                        self.name,
                        "streaming_chunk_format",
                        Verdict.FAIL,
                        status_code=200,
                        detail="; ".join(all_errors[:10]),
                    )
                )
            else:
                results.append(
                    self.make_result(
                        self.name,
                        "streaming_chunk_format",
                        Verdict.PASS,
                        status_code=200,
                        detail=f"All {len(data_chunks)} chunks conform to ChatCompletionChunk spec",
                    )
                )

        # ----- 10. Streaming: first chunk has role=assistant -----
        if resp.chunks and resp.status == 200:
            first_data = None
            for raw in resp.chunks:
                if raw != "[DONE]":
                    first_data, _ = parse_json(raw)
                    if first_data:
                        break
            if first_data:
                delta = _first_choice(first_data).get("delta", {})
                if delta.get("role") == "assistant":
                    verdict, detail = (
                        Verdict.PASS,
                        "First chunk delta has role=assistant",
                    )
                elif "role" not in delta:
                    verdict, detail = (
                        Verdict.FAIL,
                        "First chunk delta missing role",
                    )
                else:
                    verdict, detail = (
                        Verdict.FAIL,
                        f"First chunk delta role is '{delta.get('role')}', expected 'assistant'",
                    )
            else:
                verdict, detail = Verdict.FAIL, "Could not parse first chunk"
            results.append(
                self.make_result(
                    self.name,
                    "streaming_first_chunk_role",
                    verdict,
                    status_code=200,
                    detail=detail,
                )
            )

        # ----- 11. Streaming: last content chunk has finish_reason -----
        if resp.chunks and resp.status == 200:
            last_data = None
            for raw in reversed(resp.chunks):
                if raw == "[DONE]":
                    continue
                last_data, _ = parse_json(raw)
                if last_data and last_data.get("choices"):
                    break
            if last_data:
                fr = _first_choice(last_data).get("finish_reason")
                if fr is not None and fr in VALID_FINISH_REASONS:
                    verdict, detail = (
                        Verdict.PASS,
                        f"Last chunk has finish_reason='{fr}'",
                    )
                elif fr is None:
                    verdict, detail = (
                        Verdict.FAIL,
                        "Last chunk has finish_reason=null",
                    )
                else:
                    verdict, detail = (
                        Verdict.FAIL,
                        f"Last chunk has invalid finish_reason='{fr}'",
                    )
            else:
                verdict, detail = Verdict.FAIL, "Could not parse last chunk"
            results.append(
                self.make_result(
                    self.name,
                    "streaming_last_chunk_finish_reason",
                    verdict,
                    status_code=200,
                    detail=detail,
                )
            )

        # ----- 12. Streaming with include_usage -----
        resp_usage = await client.post_streaming(
            {**req("Say hi"), "stream_options": {"include_usage": True}}
        )
        if resp_usage.status == 200 and resp_usage.chunks:
            usage_chunk = None
            for raw in reversed(resp_usage.chunks):
                if raw == "[DONE]":
                    continue
                usage_cd, _ = parse_json(raw)
                if usage_cd and usage_cd.get("usage") is not None:
                    usage_chunk = usage_cd
                    break
            if usage_chunk:
                errs = _validate_usage(usage_chunk["usage"])
                if errs:
                    verdict, detail = (
                        Verdict.FAIL,
                        f"Stream usage chunk invalid: {'; '.join(errs)}",
                    )
                else:
                    verdict, detail = (
                        Verdict.PASS,
                        "Stream usage chunk has valid token counts",
                    )
            else:
                verdict, detail = (
                    Verdict.FAIL,
                    "include_usage=true but no usage chunk received",
                )
        elif resp_usage.status == 200:
            verdict, detail = Verdict.FAIL, "No chunks received"
        elif resp_usage.status == 400:
            verdict, detail = (
                Verdict.INTERESTING,
                "Server rejects stream_options (400)",
            )
        else:
            verdict = Verdict.ERROR if resp_usage.error else Verdict.FAIL
            detail = resp_usage.error or f"Status {resp_usage.status}"
        results.append(
            self.make_result(
                self.name,
                "streaming_include_usage",
                verdict,
                status_code=resp_usage.status,
                detail=detail,
            )
        )

        # ----- 13. Error response format (4xx) -----
        bad_resp = await client.post_json({"model": model})
        if bad_resp.status == 400 or bad_resp.status == 422:
            data, err = parse_json(bad_resp.body)
            if err:
                verdict, detail = (
                    Verdict.FAIL,
                    f"Error response is not JSON: {err}",
                )
            elif data:
                error_obj = data.get("error")
                if not isinstance(error_obj, dict):
                    verdict, detail = (
                        Verdict.FAIL,
                        f"Error response missing 'error' object, got keys: {list(data.keys())}",
                    )
                else:
                    errs = []
                    if "message" not in error_obj:
                        errs.append("error.message missing")
                    if "type" not in error_obj:
                        errs.append("error.type missing")
                    if errs:
                        verdict, detail = Verdict.FAIL, "; ".join(errs)
                    else:
                        verdict, detail = (
                            Verdict.PASS,
                            f"Error response follows spec: type={error_obj['type']}",
                        )
            else:
                verdict, detail = Verdict.FAIL, "Empty error response body"
        elif bad_resp.status == 200:
            verdict, detail = (
                Verdict.FAIL,
                "Server accepted request without 'messages' field",
            )
        else:
            verdict, detail = (
                Verdict.INTERESTING,
                f"Status {bad_resp.status} for missing messages",
            )
        results.append(
            self.make_result(
                self.name,
                "error_response_format",
                verdict,
                status_code=bad_resp.status,
                detail=detail,
            )
        )

        # ----- 14. response_format json_object -----
        resp_json = await client.post_json(
            req(
                "Return JSON with key 'greeting' and value 'hello'",
                response_format={"type": "json_object"},
            )
        )
        if resp_json.status == 200:
            data, _ = parse_json(resp_json.body)
            if data:
                errs = _validate_chat_completion(data)
                content = (
                    _first_choice(data).get("message", {}).get("content", "")
                )
                if errs:
                    verdict, detail = Verdict.FAIL, "; ".join(errs)
                elif content:
                    _, json_err = parse_json(content)
                    if json_err:
                        verdict, detail = (
                            Verdict.FAIL,
                            f"json_object mode returned non-JSON content: {json_err}",
                        )
                    else:
                        verdict, detail = (
                            Verdict.PASS,
                            "json_object mode returned valid JSON content",
                        )
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "Empty content with json_object mode",
                    )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        elif resp_json.status == 400:
            verdict, detail = (
                Verdict.INTERESTING,
                "Server rejects json_object response_format (400)",
            )
        else:
            verdict = (
                Verdict.FAIL if resp_json.status >= 500 else Verdict.INTERESTING
            )
            detail = f"Status {resp_json.status}"
        results.append(
            self.make_result(
                self.name,
                "response_format_json_object",
                verdict,
                status_code=resp_json.status,
                detail=detail,
            )
        )

        # ----- 15. response_format json_schema -----
        resp_schema = await client.post_json(
            req(
                "Return a greeting",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "greeting",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "greeting": {"type": "string"},
                            },
                            "required": ["greeting"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
        )
        if resp_schema.status == 200:
            data, _ = parse_json(resp_schema.body)
            if data:
                errs = _validate_chat_completion(data)
                content = (
                    _first_choice(data).get("message", {}).get("content", "")
                )
                if errs:
                    verdict, detail = Verdict.FAIL, "; ".join(errs)
                elif content:
                    parsed, json_err = parse_json(content)
                    if json_err or parsed is None:
                        verdict, detail = (
                            Verdict.FAIL,
                            f"json_schema mode returned non-JSON: {json_err}",
                        )
                    elif "greeting" not in parsed:
                        verdict, detail = (
                            Verdict.FAIL,
                            f"json_schema response missing 'greeting' key, got: {list(parsed.keys())}",
                        )
                    else:
                        verdict, detail = (
                            Verdict.PASS,
                            "json_schema response matches schema",
                        )
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "Empty content with json_schema mode",
                    )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        elif resp_schema.status == 400:
            verdict, detail = (
                Verdict.INTERESTING,
                "Server rejects json_schema response_format (400)",
            )
        else:
            verdict = (
                Verdict.FAIL
                if resp_schema.status >= 500
                else Verdict.INTERESTING
            )
            detail = f"Status {resp_schema.status}"
        results.append(
            self.make_result(
                self.name,
                "response_format_json_schema",
                verdict,
                status_code=resp_schema.status,
                detail=detail,
            )
        )

        # ----- 16. Tool calling response structure -----
        tool_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The city name",
                                },
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
            "max_tokens": 100,
        }
        resp_tool = await client.post_json(tool_payload)
        if resp_tool.status == 200:
            data, _ = parse_json(resp_tool.body)
            if data:
                errs = _validate_chat_completion(data)
                choice = _first_choice(data)
                fr = choice.get("finish_reason")
                tc = choice.get("message", {}).get("tool_calls")
                if errs:
                    verdict, detail = Verdict.FAIL, "; ".join(errs)
                elif fr == "tool_calls" and tc:
                    tc_errs = _validate_tool_calls(tc, "choices[0]")
                    if tc_errs:
                        verdict, detail = Verdict.FAIL, "; ".join(tc_errs)
                    else:
                        verdict, detail = (
                            Verdict.PASS,
                            f"Tool call response valid: {len(tc)} tool call(s)",
                        )
                elif fr == "stop" and not tc:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "Model chose not to call tool despite tool_choice forcing",
                    )
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        f"finish_reason={fr}, tool_calls={'present' if tc else 'absent'}",
                    )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        elif resp_tool.status == 400:
            verdict, detail = (
                Verdict.INTERESTING,
                "Server rejects tool calling (400)",
            )
        else:
            verdict = (
                Verdict.FAIL if resp_tool.status >= 500 else Verdict.INTERESTING
            )
            detail = f"Status {resp_tool.status}"
        results.append(
            self.make_result(
                self.name,
                "tool_call_response_structure",
                verdict,
                status_code=resp_tool.status,
                detail=detail,
            )
        )

        # ----- 17. Tool message in conversation -----
        tool_conv = {
            "model": model,
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"SF"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "72F and sunny",
                },
            ],
            "max_tokens": 50,
        }
        resp_toolmsg = await client.post_json(tool_conv)
        if resp_toolmsg.status == 200:
            data, _ = parse_json(resp_toolmsg.body)
            if data:
                errs = _validate_chat_completion(data)
                verdict = Verdict.FAIL if errs else Verdict.PASS
                detail = (
                    "; ".join(errs)
                    if errs
                    else "Tool message conversation accepted"
                )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON response"
        elif resp_toolmsg.status == 400:
            verdict, detail = (
                Verdict.INTERESTING,
                "Server rejects tool messages (400)",
            )
        else:
            verdict = (
                Verdict.FAIL
                if resp_toolmsg.status >= 500
                else Verdict.INTERESTING
            )
            detail = f"Status {resp_toolmsg.status}"
        results.append(
            self.make_result(
                self.name,
                "tool_message_conversation",
                verdict,
                status_code=resp_toolmsg.status,
                detail=detail,
            )
        )

        # ----- 18. temperature and top_p parameters -----
        resp_temp = await client.post_json(
            req("Say hello", temperature=0.0, top_p=1.0)
        )
        if resp_temp.status == 200:
            data, _ = parse_json(resp_temp.body)
            if data:
                errs = _validate_chat_completion(data)
                verdict = Verdict.FAIL if errs else Verdict.PASS
                detail = (
                    "; ".join(errs)
                    if errs
                    else "temperature=0, top_p=1 accepted"
                )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON"
        else:
            verdict = _unexpected_status_verdict(resp_temp.status)
            detail = f"Status {resp_temp.status}"
        results.append(
            self.make_result(
                self.name,
                "temperature_top_p",
                verdict,
                status_code=resp_temp.status,
                detail=detail,
            )
        )

        # ----- 19. stop sequences -----
        resp_stop = await client.post_json(
            req("Count: 1, 2, 3, 4, 5", stop=[","])
        )
        if resp_stop.status == 200:
            data, _ = parse_json(resp_stop.body)
            if data:
                errs = _validate_chat_completion(data)
                verdict = Verdict.FAIL if errs else Verdict.PASS
                detail = (
                    "; ".join(errs)
                    if errs
                    else "stop sequence parameter accepted"
                )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON"
        else:
            verdict = _unexpected_status_verdict(resp_stop.status)
            detail = f"Status {resp_stop.status}"
        results.append(
            self.make_result(
                self.name,
                "stop_sequences",
                verdict,
                status_code=resp_stop.status,
                detail=detail,
            )
        )

        # ----- 20. logprobs response structure -----
        resp_lp = await client.post_json(
            req("Hello", logprobs=True, top_logprobs=3)
        )
        if resp_lp.status == 200:
            data, _ = parse_json(resp_lp.body)
            if data:
                errs = _validate_chat_completion(data)
                choice = _first_choice(data)
                lp = choice.get("logprobs")
                if errs:
                    verdict, detail = Verdict.FAIL, "; ".join(errs)
                elif lp is None:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "logprobs=true but no logprobs in response",
                    )
                elif not isinstance(lp, dict):
                    verdict, detail = (
                        Verdict.FAIL,
                        f"logprobs should be object, got {type(lp).__name__}",
                    )
                else:
                    content_lps = lp.get("content")
                    if content_lps is None:
                        verdict, detail = (
                            Verdict.INTERESTING,
                            "logprobs.content is null",
                        )
                    elif not isinstance(content_lps, list):
                        verdict, detail = (
                            Verdict.FAIL,
                            f"logprobs.content should be array, got {type(content_lps).__name__}",
                        )
                    elif len(content_lps) > 0:
                        tok = content_lps[0]
                        tok_errs = []
                        if "token" not in tok:
                            tok_errs.append("missing 'token'")
                        if "logprob" not in tok:
                            tok_errs.append("missing 'logprob'")
                        elif not isinstance(tok["logprob"], (int, float)):
                            tok_errs.append(
                                f"logprob not numeric: {type(tok['logprob']).__name__}"
                            )
                        tl = tok.get("top_logprobs")
                        if tl is not None and not isinstance(tl, list):
                            tok_errs.append(
                                f"top_logprobs not array: {type(tl).__name__}"
                            )
                        elif isinstance(tl, list) and len(tl) > 3:
                            tok_errs.append(
                                f"top_logprobs has {len(tl)} entries, requested 3"
                            )
                        if tok_errs:
                            verdict, detail = Verdict.FAIL, "; ".join(tok_errs)
                        else:
                            verdict, detail = (
                                Verdict.PASS,
                                f"logprobs structure valid, {len(content_lps)} tokens",
                            )
                    else:
                        verdict, detail = (
                            Verdict.PASS,
                            "logprobs.content is empty array",
                        )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON"
        elif resp_lp.status == 400:
            # MAX's overlap pipeline does not currently support logprobs, so a
            # clean 400 rejection is the expected, correct behavior here rather
            # than a divergence to investigate.
            verdict, detail = (
                Verdict.PASS,
                "Server correctly rejects unsupported logprobs (400)",
            )
        else:
            verdict = (
                Verdict.FAIL if resp_lp.status >= 500 else Verdict.INTERESTING
            )
            detail = f"Status {resp_lp.status}"
        results.append(
            self.make_result(
                self.name,
                "logprobs_structure",
                verdict,
                status_code=resp_lp.status,
                detail=detail,
            )
        )

        # ----- 21. seed parameter for determinism -----
        seed_payload = req("What is 2+2?", seed=42, temperature=0)
        resp_a = await client.post_json(seed_payload)
        resp_b = await client.post_json(seed_payload)
        if resp_a.status == 200 and resp_b.status == 200:
            da, _ = parse_json(resp_a.body)
            db, _ = parse_json(resp_b.body)
            if da and db:
                errs_a = _validate_chat_completion(da)
                errs_b = _validate_chat_completion(db)
                if errs_a or errs_b:
                    verdict, detail = (
                        Verdict.FAIL,
                        f"Spec errors: {'; '.join(errs_a + errs_b)}",
                    )
                else:
                    ca = _first_choice(da).get("message", {}).get("content", "")
                    cb = _first_choice(db).get("message", {}).get("content", "")
                    if ca == cb:
                        verdict, detail = (
                            Verdict.PASS,
                            "Deterministic output with same seed",
                        )
                    else:
                        verdict, detail = (
                            Verdict.INTERESTING,
                            "Different output with same seed (spec says best-effort)",
                        )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON"
        else:
            verdict = Verdict.INTERESTING
            detail = f"Status a={resp_a.status}, b={resp_b.status}"
        results.append(
            self.make_result(
                self.name,
                "seed_determinism",
                verdict,
                status_code=resp_a.status,
                detail=detail,
            )
        )

        # ----- 22. Content-Type header is application/json -----
        resp_ct = await client.post_json(req("Hello"))
        if resp_ct.status == 200:
            ct = resp_ct.headers.get(
                "Content-Type", resp_ct.headers.get("content-type", "")
            )
            if "application/json" in ct:
                verdict, detail = Verdict.PASS, f"Content-Type: {ct}"
            else:
                verdict, detail = (
                    Verdict.FAIL,
                    f"Expected application/json, got Content-Type: {ct}",
                )
        else:
            verdict = _unexpected_status_verdict(resp_ct.status)
            detail = f"Status {resp_ct.status}"
        results.append(
            self.make_result(
                self.name,
                "content_type_header",
                verdict,
                status_code=resp_ct.status,
                detail=detail,
            )
        )

        # ----- 23. Streaming Content-Type is text/event-stream -----
        resp_sct = await client.post_streaming(req("Hello"))
        if resp_sct.status == 200:
            ct = resp_sct.headers.get(
                "Content-Type", resp_sct.headers.get("content-type", "")
            )
            if "text/event-stream" in ct:
                verdict, detail = Verdict.PASS, f"Content-Type: {ct}"
            else:
                verdict, detail = (
                    Verdict.FAIL,
                    f"Expected text/event-stream for streaming, got Content-Type: {ct}",
                )
        else:
            verdict = _unexpected_status_verdict(resp_sct.status)
            detail = f"Status {resp_sct.status}"
        results.append(
            self.make_result(
                self.name,
                "streaming_content_type",
                verdict,
                status_code=resp_sct.status,
                detail=detail,
            )
        )

        # ----- 24. frequency_penalty and presence_penalty -----
        resp_pen = await client.post_json(
            req("Hello", frequency_penalty=0.5, presence_penalty=0.5)
        )
        if resp_pen.status == 200:
            data, _ = parse_json(resp_pen.body)
            if data:
                errs = _validate_chat_completion(data)
                verdict = Verdict.FAIL if errs else Verdict.PASS
                detail = (
                    "; ".join(errs) if errs else "penalty parameters accepted"
                )
            else:
                verdict, detail = Verdict.FAIL, "Invalid JSON"
        else:
            verdict = _unexpected_status_verdict(resp_pen.status)
            detail = f"Status {resp_pen.status}"
        results.append(
            self.make_result(
                self.name,
                "penalty_parameters",
                verdict,
                status_code=resp_pen.status,
                detail=detail,
            )
        )

        # ----- 25. max_tokens vs max_completion_tokens -----
        max_token_limit = 10
        for param_name, param in [
            ("max_tokens", {"max_tokens": max_token_limit}),
            (
                "max_completion_tokens",
                {"max_completion_tokens": max_token_limit},
            ),
        ]:
            resp_mt = await client.post_json(
                {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "Write a long story"}
                    ],
                    **param,
                }
            )
            if resp_mt.status == 200:
                data, _ = parse_json(resp_mt.body)
                if data:
                    errs = _validate_chat_completion(data)
                    usage = data.get("usage", {})
                    ct = usage.get("completion_tokens", 0)
                    if ct > max_token_limit:
                        errs.append(
                            f"completion_tokens ({ct}) exceeds"
                            f" {param_name}={max_token_limit}"
                        )
                    verdict = Verdict.FAIL if errs else Verdict.PASS
                    detail = (
                        "; ".join(errs)
                        if errs
                        else f"{param_name} honored ({ct} tokens)"
                    )
                else:
                    verdict, detail = Verdict.FAIL, "Invalid JSON"
            elif resp_mt.status == 400:
                verdict, detail = (
                    Verdict.INTERESTING,
                    f"Server rejects {param_name} (400)",
                )
            else:
                verdict = (
                    Verdict.FAIL
                    if resp_mt.status >= 500
                    else Verdict.INTERESTING
                )
                detail = f"Status {resp_mt.status}"
            results.append(
                self.make_result(
                    self.name,
                    f"param_{param_name}",
                    verdict,
                    status_code=resp_mt.status,
                    detail=detail,
                )
            )

        # ----- 26. Streaming tool call chunks have type field -----
        tool_stream_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Get the weather in Paris"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
            "max_tokens": 100,
        }
        resp_ts = await client.post_streaming(tool_stream_payload)
        if resp_ts.status == 200 and resp_ts.chunks:
            tool_call_chunks = []
            for raw in resp_ts.chunks:
                if raw == "[DONE]":
                    continue
                ts_cd, _ = parse_json(raw)
                if ts_cd:
                    delta = _first_choice(ts_cd).get("delta", {})
                    if "tool_calls" in delta:
                        tool_call_chunks.append(delta["tool_calls"])

            if tool_call_chunks:
                first_tc = tool_call_chunks[0]
                if isinstance(first_tc, list) and len(first_tc) > 0:
                    tc = first_tc[0]
                    missing = []
                    if "index" not in tc:
                        missing.append("index")
                    if "type" not in tc and "id" not in tc:
                        missing.append(
                            "type or id (expected in first tool call chunk)"
                        )
                    if missing:
                        verdict, detail = (
                            Verdict.FAIL,
                            f"First streaming tool_call chunk missing: {', '.join(missing)}",
                        )
                    else:
                        verdict, detail = (
                            Verdict.PASS,
                            "Streaming tool call chunks have required fields",
                        )
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "tool_calls in delta is not array or empty",
                    )
            else:
                verdict, detail = (
                    Verdict.INTERESTING,
                    "No tool_calls in stream delta chunks",
                )
        elif resp_ts.status == 400:
            verdict, detail = (
                Verdict.INTERESTING,
                "Server rejects streaming tool calls (400)",
            )
        else:
            verdict = Verdict.ERROR if resp_ts.error else Verdict.INTERESTING
            detail = resp_ts.error or f"Status {resp_ts.status}"
        results.append(
            self.make_result(
                self.name,
                "streaming_tool_call_format",
                verdict,
                status_code=resp_ts.status,
                detail=detail,
            )
        )

        return results
