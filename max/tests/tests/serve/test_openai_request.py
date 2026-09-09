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

from max.pipelines.context.exceptions import InputError
from max.serve.config import Settings
from max.serve.router.openai_routes import (
    _create_response_format,
    openai_parse_chat_completion_request,
)

"""
It is unclear why the type ignore for CreateChatCompletionRequest is necessary.
bazel+mypy complain about this import not being available even though it is part of the serving package.
Explicitly importing //max/python/max/serve/schemas in the test's BUILD file hasn't worked either.
"""

from typing import Any, cast

import pytest
from max.serve.schemas.openai import CreateChatCompletionRequest
from pydantic import AnyUrl, ValidationError


@pytest.mark.skip
async def test_openai_extract_image_from_requests() -> None:
    request_images = {
        "smily_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII=",
        "boardwark_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        "mountain_url": "https://picsum.photos/seed/picsum/200/300",
    }

    system_message = {
        "role": "system",
        "content": "You are an opinionated chat-bot.",
    }
    user_message_no_images = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What'''s in this image?"},
        ],
    }
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [system_message, user_message_no_images]}
    )

    settings = Settings()
    (
        messages,
        images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(request, False, settings)
    assert len(messages) == 2
    assert len(images) == 0
    assert isinstance(messages[0].content, str)
    assert isinstance(messages[1].content, list)
    assert hasattr(messages[1].content[0], "text")

    (
        messages,
        images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(request, True, settings)
    assert len(messages) == 2
    assert len(images) == 0
    assert isinstance(messages[0].content, list)
    assert isinstance(messages[1].content, list)
    assert hasattr(messages[1].content[0], "text")

    user_message_image_with_url = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What'''s in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": request_images["boardwark_url"]},
            },
        ],
    }
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [user_message_image_with_url]}
    )
    (
        messages,
        images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request,
        False,
        settings,
    )
    assert len(messages) == 1
    assert len(images) == 1
    assert isinstance(messages[0].content, list)
    # When wrap_content=False, content items are dicts
    assert isinstance(messages[0].content[1], dict)
    assert "image_url" in messages[0].content[1]
    assert images[0] == AnyUrl(request_images["boardwark_url"])

    (
        messages,
        images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request,
        True,
        settings,
    )
    assert len(messages) == 1
    assert len(images) == 1
    assert isinstance(messages[0].content, list)
    assert messages[0].content[1].type == "image"
    assert images[0] == AnyUrl(request_images["boardwark_url"])

    user_message_image_two_urls = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What'''s in these images?"},
            {
                "type": "image_url",
                "image_url": {"url": request_images["boardwark_url"]},
            },
            {
                "type": "image_url",
                "image_url": {"url": request_images["mountain_url"]},
            },
        ],
    }
    (
        messages,
        images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        CreateChatCompletionRequest(
            model="test", messages=[system_message, user_message_image_two_urls]
        ),
        False,
        settings,
    )
    assert len(messages) == 2
    assert len(images) == 2
    assert images[0] == AnyUrl(request_images["boardwark_url"])
    assert images[1] == AnyUrl(request_images["mountain_url"])

    user_message_mixed_url_b64 = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What'''s in these images?"},
            {
                "type": "image_url",
                "image_url": {"url": request_images["smily_b64"]},
            },
            {
                "type": "image_url",
                "image_url": {"url": request_images["mountain_url"]},
            },
        ],
    }
    request = CreateChatCompletionRequest(
        model="test", messages=[user_message_mixed_url_b64]
    )
    (
        messages,
        images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(request, False, settings)
    assert len(messages) == 1
    assert len(images) == 2
    assert images[0] == AnyUrl(request_images["smily_b64"])
    assert images[1] == AnyUrl(request_images["mountain_url"])

    (
        messages,
        images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(request, True, settings)
    assert len(messages) == 1
    assert len(images) == 2
    assert isinstance(messages[0].content, list)
    assert messages[0].content[1].type == "image"
    assert messages[0].content[2].type == "image"
    assert images[0] == AnyUrl(request_images["smily_b64"])
    assert images[1] == AnyUrl(request_images["mountain_url"])


async def test_openai_user_message_with_null_content() -> None:
    """Test that user messages with null content are accepted and handled."""
    # Test with explicit null content
    user_message_null_content = {
        "role": "user",
        "content": None,
    }
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [user_message_null_content]}
    )
    settings = Settings()
    (
        messages,
        _images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(request, False, settings)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == ""

    # Test with missing content field (should default to None)
    user_message_no_content = {
        "role": "user",
    }
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [user_message_no_content]}
    )
    (
        messages,
        _images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(request, False, settings)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == ""


async def test_openai_parse_normalizes_developer_role_to_system() -> None:
    """``role: "developer"`` must be accepted and normalized to ``"system"``.

    OpenAI's model chat-completion spec uses ``developer`` as the
    system-equivalent role. The internal ``TextGenerationRequestMessage``
    ``_MessageRole`` literal only enumerates the five spec-supported roles
    (``system``, ``user``, ``assistant``, ``tool``, ``function``), so the
    request was previously rejected with a 422. Normalize at the
    OpenAI-compat seam so requests from OpenAI model spec compliant clients are accepted.
    """
    request_data = {
        "model": "test",
        "messages": [
            {"role": "developer", "content": "You are a coding assistant."},
            {"role": "user", "content": "hi"},
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    settings = Settings()

    (
        messages,
        _images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == "You are a coding assistant."
    assert messages[1].role == "user"


def test_openai_parse_rejects_unknown_role() -> None:
    """Roles outside the spec-supported set still surface as a validation error.

    The ``developer`` normalization must not become a permissive sink:
    arbitrary role strings remain rejected by the schema so genuine malformed
    requests still produce a 4xx. The role union is now validated by pydantic
    at ``model_validate`` time (SERVSYS-1257), so the rejection happens here
    rather than later inside ``openai_parse_chat_completion_request``.
    """
    request_data = {
        "model": "test",
        "messages": [{"role": "wizard", "content": "abracadabra"}],
    }
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(request_data)


def test_openai_chat_completion_accepts_prompt_tokens() -> None:
    """Schema must accept ``prompt_tokens`` (orchestrator pre-tokenized input).

    The Mammoth orchestrator tokenizes incoming requests once and forwards
    the integer token IDs to MAX Serve under the ``prompt_tokens`` field,
    bypassing re-tokenization. Regression coverage for SERVSYS-1239: the
    schema rewrite in #84789 dropped this MAX-only field, causing
    ``CreateChatCompletionRequest.model_validate_json`` to fail with
    ``ValidationError: prompt_tokens - Extra inputs are not permitted``.
    """
    body = (
        '{"model":"test","ignore_eos":true,"max_tokens":4,'
        '"messages":[{"role":"user","content":"hi"}],'
        '"prompt_tokens":[101,202,303]}'
    )
    request = CreateChatCompletionRequest.model_validate_json(body)
    assert request.prompt_tokens == [101, 202, 303]


async def test_openai_parse_forwards_tool_call_metadata() -> None:
    """Multi-turn tool-use messages must keep ``tool_calls`` and ``tool_call_id``.

    The router previously dropped these fields when building the internal
    ``TextGenerationRequestMessage`` list, so the chat-templated prompt
    rendered with an empty ``<think>`` block and a bare ``## Return of``
    header instead of the originating function name (for example Kimi-K2).
    """
    request_data = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I'll call the search tool.",
                "tool_calls": [
                    {
                        "id": "call_9e53d2d2_0",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q":"cats"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_9e53d2d2_0",
                "content": "1. fluffy cat\n2. orange cat",
            },
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    settings = Settings()

    (
        messages,
        _images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].tool_calls is None
    assert messages[0].tool_call_id is None

    assert messages[1].role == "assistant"
    assert messages[1].tool_calls is not None
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0]["id"] == "call_9e53d2d2_0"
    assert messages[1].tool_calls[0]["function"]["name"] == "search"
    # ``function.arguments`` is decoded from the OpenAI JSON-string wire
    # format into a mapping so tool-use chat templates can iterate it.
    assert messages[1].tool_calls[0]["function"]["arguments"] == {"q": "cats"}
    assert messages[1].reasoning_content == "I'll call the search tool."

    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "call_9e53d2d2_0"
    assert messages[2].content == "1. fluffy cat\n2. orange cat"


async def test_openai_parse_drops_empty_tool_calls() -> None:
    """Empty assistant ``tool_calls`` lists must be dropped (vLLM parity).

    Some clients echo back ``"tool_calls": []`` on assistant turns even
    when the assistant did not call any tools. Letting an empty list reach
    the chat template causes tool-use branches to fire with no entries,
    which renders broken multi-turn prompts.
    """
    request_data = {
        "model": "test",
        "messages": [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [],
            },
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    settings = Settings()

    (
        messages,
        _images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].tool_calls is None
    assert messages[0].content == "ok"


def test_openai_request_rejects_tool_call_missing_name() -> None:
    """Assistant ``tool_calls`` missing ``function.name`` is rejected.

    Regression for the llm-fuzz ``tool_calling/bad_tool_call_missing_name``
    case. With the strongly-typed ``messages: list[ChatCompletionMessageParam]``
    field, pydantic validates ``function.name`` (declared ``Required[str]`` on
    the OpenAI SDK ``TypedDict``) at ``model_validate`` time and surfaces a
    422 - no hand-coded check needed in the route.
    """
    request_data = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {"arguments": "{}"},
                    }
                ],
            },
        ],
    }
    with pytest.raises(ValidationError) as excinfo:
        CreateChatCompletionRequest.model_validate(request_data)
    assert "function.name" in str(excinfo.value)


def test_openai_request_rejects_tool_call_empty_function() -> None:
    """Assistant ``tool_calls`` with an empty ``function`` object is rejected.

    Regression for the llm-fuzz ``tool_calling/bad_tool_call_empty_function``
    case. Same mechanism as ``test_openai_request_rejects_tool_call_missing_name``:
    ``function.name`` is ``Required[str]`` on the OpenAI SDK ``TypedDict`` so
    pydantic refuses ``function = {}`` at request validation.
    """
    request_data = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_test", "type": "function", "function": {}},
                ],
            },
        ],
    }
    with pytest.raises(ValidationError) as excinfo:
        CreateChatCompletionRequest.model_validate(request_data)
    assert "function.name" in str(excinfo.value)


async def test_openai_parse_coerces_empty_tool_call_arguments() -> None:
    """Empty ``function.arguments`` are coerced to ``{}``.

    Mirrors vLLM's ``_postprocess_messages``: clients that send no-arg
    tool calls (for example a ``get_time()`` invocation) emit
    ``"arguments": ""`` over the wire. Chat templates that iterate the
    mapping must see an empty dict, not the empty string.
    """
    request_data = {
        "model": "test",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": ""},
                    },
                ],
            },
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    settings = Settings()

    (
        messages,
        _images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assert len(messages) == 1
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) == 1
    assert messages[0].tool_calls[0]["function"]["arguments"] == {}


def test_openai_request_rejects_tool_call_missing_arguments() -> None:
    """Assistant ``tool_calls`` missing ``function.arguments`` is rejected.

    OpenAI's spec marks ``Function.arguments`` as ``Required[str]``.
    Requests that omit it entirely must fail schema validation rather
    than being silently coerced.
    """
    request_data = {
        "model": "test",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_date"},
                    },
                ],
            },
        ],
    }
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(request_data)


def test_openai_chat_completion_accepts_explicit_null_tool_choice() -> None:
    """Schema must accept ``"tool_choice": null`` from OpenAI-compatible clients.

    OpenAI's ``ChatCompletionToolChoiceOptionParam`` is a non-Optional union
    of ``Literal["none","auto","required"]`` and tool-choice objects;
    omission is expressed via ``NotRequired`` on the TypedDict. Some clients
    (LangChain, certain JS SDKs, anything that serializes a dataclass with
    a ``None`` field) explicitly emit ``"tool_choice": null`` instead of
    omitting the key, which must be treated as equivalent to omission.
    """
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": None,
    }
    request = CreateChatCompletionRequest.model_validate(body)
    assert request.tool_choice is None


def test_openai_response_format_schema_advertises_boolean() -> None:
    """The generated JSON/OpenAPI schema must advertise the boolean ``schema``
    so the published docs match what the endpoint actually accepts."""
    schema = CreateChatCompletionRequest.model_json_schema()
    schema_field = schema["$defs"]["JSONSchema"]["properties"]["schema"]
    types = {
        opt.get("type") for opt in schema_field.get("anyOf", [schema_field])
    }
    assert "boolean" in types


@pytest.mark.parametrize("schema", [True, False])
def test_openai_response_format_accepts_boolean_schema(schema: bool) -> None:
    """A boolean is a valid JSON Schema; the request must accept it verbatim
    (de-sugaring to a dict happens later, in ``_create_response_format``)."""
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "target", "schema": schema},
        },
    }
    request = CreateChatCompletionRequest.model_validate(body)
    response_format = cast(dict[str, Any], request.response_format)
    assert response_format["json_schema"]["schema"] is schema


def test_create_response_format_normalizes_schema() -> None:
    """The worker compiles what this returns, so it must carry the
    *normalized* schema: an untyped root defaults to "object", which is what
    keeps a looping model from running to max_length."""
    response_format = cast(
        Any,
        {
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": {"properties": {}}},
        },
    )

    result = _create_response_format(
        response_format, enable_response_format_schema=True
    )

    assert result is not None
    assert result.json_schema is not None
    assert result.json_schema["type"] == "object"


def test_create_response_format_rejects_schema_without_the_flag() -> None:
    """The flag check stays at the route: the worker's grammar gate skips
    requests ``update_context`` would reject, so one reaching the worker
    would raise from inside ``execute()``."""
    response_format = cast(
        Any,
        {
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": {"type": "object"}},
        },
    )

    with pytest.raises(InputError):
        _create_response_format(
            response_format, enable_response_format_schema=False
        )


def test_openai_chat_message_validates_structure() -> None:
    """Regression for SERVSYS-1257: ``messages`` should not be typed as
    ``list[dict[str, Any]]`` (the ``Any`` clobbered OpenAI SDK validation).

    Confirms that ``CreateChatCompletionRequest`` now validates each
    message against the OpenAI ``ChatCompletionMessageParam`` union -
    invalid roles, missing required fields, and malformed content parts
    surface as a 422 instead of being silently accepted.
    """
    base = {"model": "test"}

    # Invalid role is rejected (was accepted previously).
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {**base, "messages": [{"role": "wizard", "content": "magic"}]}
        )

    # Missing required ``role`` is rejected.
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {**base, "messages": [{"content": "no role"}]}
        )


def test_openai_chat_message_multipart_content_preserves_list() -> None:
    """Regression for SERVSYS-1257: validated multi-part ``content`` is a
    concrete ``list`` of plain dicts that can be re-iterated.

    Before this fix, ``messages`` was typed as ``list[dict[str, Any]]`` to
    sidestep pydantic stashing the inner ``Iterable[ContentPart]`` as a
    one-shot ``ValidatorIterator``. With the recursive ``Iterable -> list``
    normalization in place, the content array survives validation as a
    real list and the route can index/iterate it freely.
    """
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/foo.png"},
                        },
                    ],
                }
            ],
        }
    )
    content = request.messages[0].get("content")
    assert isinstance(content, list)
    assert len(content) == 2
    # Re-iteration must yield the same dicts (used to break on a
    # ``ValidatorIterator`` that consumed itself after the first pass).
    first_pass = [c.get("type") for c in content]
    second_pass = [c.get("type") for c in content]
    assert first_pass == second_pass == ["text", "image_url"]
    # Items are plain dicts (TypedDict -> dict at the JSON layer).
    assert content[0]["text"] == "what's in this image?"
    assert content[1]["image_url"]["url"] == "https://example.com/foo.png"


def test_openai_chat_message_preserves_vendor_extensions() -> None:
    """``reasoning_content`` (vLLM-style extension on assistant turns) and
    other vendor-specific message fields must survive validation.

    The normalized chat-message ``TypedDict`` mirrors are configured with
    ``extra='allow'`` so passthrough keys aren't silently dropped.
    """
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "preserved",
                }
            ],
        }
    )
    assert request.messages[0].get("reasoning_content") == "preserved"


def test_openai_chat_message_validates_vendor_extension_type() -> None:
    """``reasoning_content`` is a first-class field; pydantic enforces its type.

    Previously ``reasoning_content`` rode through ``extra='allow'`` on
    the message TypedDict, so a non-string value passed validation and
    later tripped a route-level ``assert`` (-> unhandled ``AssertionError``
    -> 500). Declaring it on :class:`ChatCompletionMessageParam` makes
    pydantic reject the bad type at ``model_validate`` time so the
    request surfaces as a clean 4xx via the chat route's existing
    ``ValidationError`` handler.
    """
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "ok",
                        "reasoning_content": 123,
                    }
                ],
            }
        )


def test_openai_chat_message_rejects_non_string_tool_call_id() -> None:
    """``tool_call_id`` must be a string; non-string values are rejected."""
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [
                    {
                        "role": "tool",
                        "content": "result",
                        "tool_call_id": 123,
                    }
                ],
            }
        )


def test_openai_chat_message_rejects_invalid_content_type() -> None:
    """``content`` must be a string, list, or null; integers are rejected."""
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [
                    {
                        "role": "user",
                        "content": 42,
                    }
                ],
            }
        )


def test_openai_user_message_content_nullable_schema() -> None:
    """Test that the CreateChatCompletionRequest schema accepts null user content."""
    # Test with explicit null content in user message
    request_data = {
        "model": "test",
        "messages": [{"role": "user", "content": None}],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    assert request.messages[0].get("content") is None

    # Test with omitted content field
    request_data_no_content = {
        "model": "test",
        "messages": [{"role": "user"}],
    }
    request = CreateChatCompletionRequest.model_validate(
        request_data_no_content
    )
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    assert request.messages[0].get("content") is None

    # Test mixed messages with null user content
    request_data_mixed = {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": None},
            {"role": "assistant", "content": "How can I help you?"},
            {"role": "user", "content": "Hello!"},
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data_mixed)
    assert len(request.messages) == 4
    assert request.messages[0]["content"] == "You are a helpful assistant."
    assert request.messages[1].get("content") is None
    assert request.messages[2]["content"] == "How can I help you?"
    assert request.messages[3]["content"] == "Hello!"


def test_openai_image_url_accepts_non_string_sizing_hints() -> None:
    """image_url/video_url objects accept and preserve non-string vendor hints."""
    request_data = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "auto",
                            "max_long_side_pixel": 504,
                        },
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "data:video/mp4;base64,AAAA",
                            "fps": 2.0,
                            "max_long_side_pixel": 1008,
                        },
                    },
                ],
            }
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    content = request.messages[0]["content"]
    assert isinstance(content, list)
    image_url = content[1]["image_url"]
    # The int hint round-trips as an int and all keys are preserved.
    assert image_url["max_long_side_pixel"] == 504
    assert isinstance(image_url["max_long_side_pixel"], int)
    assert image_url["url"].startswith("data:image/png")
    assert image_url["detail"] == "auto"
    video_url = content[2]["video_url"]
    assert video_url["fps"] == 2.0
    assert video_url["max_long_side_pixel"] == 1008


async def test_openai_wrap_content_carries_detail_hint() -> None:
    """``detail`` is carried onto wrapped image/video content parts."""
    smily_b64 = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
        "AAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": smily_b64, "detail": "high"},
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": "data:video/mp4;base64,AAAA",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        }
    )
    settings = Settings()
    parsed = await openai_parse_chat_completion_request(request, True, settings)
    content = parsed.messages[0].content
    assert isinstance(content, list)
    assert content[1].type == "image"
    assert content[1].detail == "high"
    assert content[2].type == "video"
    assert content[2].detail == "low"


async def test_openai_root_role_accepted_and_passed_through() -> None:
    """The MiniMax ``root`` role validates and passes through unchanged as the first message."""
    request_data = {
        "model": "test",
        "messages": [
            {"role": "root", "content": "You are MiniMax."},
            {"role": "system", "content": "You are a generic assistant."},
            {"role": "user", "content": "Who are you?"},
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    assert request.messages[0]["role"] == "root"

    settings = Settings()
    (
        messages,
        _images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(request, False, settings)
    # ``root`` survives normalization unchanged and stays first.
    assert messages[0].role == "root"
    assert messages[0].content == "You are MiniMax."
    assert messages[1].role == "system"


def test_thinking_translates_to_standard_reasoning_flags() -> None:
    """The vendor ``thinking`` control is translated to standard reasoning flags.

    ``enabled``/``disabled`` set ``enable_thinking``/``thinking`` on
    ``chat_template_kwargs``; ``adaptive`` leaves them unset (templates default
    to adaptive when no flag is given). The vendor field is not retained.
    """

    def _kwargs(mode: str) -> dict[str, Any] | None:
        request = CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": mode},
            }
        )
        # The vendor field is consumed, not stored.
        assert not hasattr(request, "thinking")
        return request.chat_template_kwargs

    assert _kwargs("enabled") == {"enable_thinking": True, "thinking": True}
    assert _kwargs("disabled") == {"enable_thinking": False, "thinking": False}
    # adaptive == unset: no reasoning flags injected.
    assert _kwargs("adaptive") is None

    # Omitted -> no flags.
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert request.chat_template_kwargs is None

    # Unknown thinking type is rejected.
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "sometimes"},
            }
        )

    # A malformed thinking object (extra keys) is rejected.
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "enabled", "budget": 100},
            }
        )


def test_thinking_does_not_override_explicit_chat_template_kwargs() -> None:
    """Client-set ``chat_template_kwargs`` win over the ``thinking`` translation."""
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    # Explicit kwarg is preserved; ``thinking`` only fills what is unset.
    assert request.chat_template_kwargs == {
        "enable_thinking": False,
        "thinking": True,
    }


def _chat_template_kwargs_for(payload: dict[str, Any]) -> dict[str, Any] | None:
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            **payload,
        }
    )
    return request.resolved_chat_template_kwargs


@pytest.mark.parametrize(
    "payload,expected",
    [
        pytest.param({}, None, id="no_reasoning_controls"),
        pytest.param(
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"enable_thinking": False},
            id="passthrough_without_reasoning_controls",
        ),
        pytest.param(
            {"reasoning_effort": "high"},
            {
                "enable_thinking": True,
                "thinking": True,
                "reasoning_effort": "high",
            },
            id="top_level_effort",
        ),
        pytest.param(
            {"reasoning_effort": "none"},
            {
                "enable_thinking": False,
                "thinking": False,
                "reasoning_effort": "none",
            },
            id="effort_none_disables_thinking",
        ),
        pytest.param(
            {"chat_template_kwargs": {"reasoning_effort": "none"}},
            {
                "enable_thinking": False,
                "thinking": False,
                "reasoning_effort": "none",
            },
            id="client_effort_none_disables_thinking",
        ),
        pytest.param(
            {"chat_template_kwargs": {"reasoning_effort": "low"}},
            {
                "enable_thinking": True,
                "thinking": True,
                "reasoning_effort": "low",
            },
            id="client_effort_enables_thinking",
        ),
        pytest.param(
            {
                "chat_template_kwargs": {
                    "reasoning_effort": "none",
                    "enable_thinking": True,
                }
            },
            {
                "enable_thinking": True,
                "thinking": True,
                "reasoning_effort": "none",
            },
            id="client_toggle_wins_over_client_effort",
        ),
        pytest.param(
            {"reasoning": {"effort": "low"}},
            {
                "enable_thinking": True,
                "thinking": True,
                "reasoning_effort": "low",
            },
            id="openrouter_effort",
        ),
        pytest.param(
            {"reasoning": {"enabled": False, "effort": "high"}},
            {
                "enable_thinking": False,
                "thinking": False,
                "reasoning_effort": "high",
            },
            id="openrouter_enabled_wins_over_effort",
        ),
        pytest.param(
            {"reasoning_effort": "high", "reasoning": {"effort": "low"}},
            {
                "enable_thinking": True,
                "thinking": True,
                "reasoning_effort": "high",
            },
            id="top_level_effort_wins",
        ),
        pytest.param(
            {
                "reasoning_effort": "high",
                "chat_template_kwargs": {
                    "reasoning_effort": "low",
                    "enable_thinking": False,
                },
            },
            {
                "reasoning_effort": "low",
                "enable_thinking": False,
                "thinking": False,
            },
            id="client_kwargs_win",
        ),
    ],
)
def test_resolved_chat_template_kwargs(
    payload: dict[str, Any], expected: dict[str, Any] | None
) -> None:
    """Reasoning controls are folded into the kwargs the chat template sees."""
    assert _chat_template_kwargs_for(payload) == expected


# ---------------------------------------------------------------------------
# MiniMax v1/chat/completions format-correctness conformance.
# ---------------------------------------------------------------------------


def test_openai_accepts_both_max_tokens_and_max_completion_tokens() -> None:
    """Both token-limit fields are accepted; max_completion_tokens wins (verifier 06_03/06_04)."""
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "Write a paragraph"}],
            "max_tokens": 50,
            "max_completion_tokens": 100,
        }
    )
    # max_completion_tokens wins; max_tokens is reconciled to match it.
    assert request.max_completion_tokens == 100
    assert request.max_tokens == 100

    # Equal values are left untouched.
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "max_completion_tokens": 64,
        }
    )
    assert request.max_tokens == 64
    assert request.max_completion_tokens == 64


def test_openai_tool_function_name_charset_is_per_model() -> None:
    """Tool-name charset defaults to OpenAI's set and widens per model (verifier 16_11)."""
    import re

    from max.pipelines.context.exceptions import InputError
    from max.serve.router.openai_routes import (
        _convert_chat_completion_tools_to_token_generator_tools,
        _validate_tool_function_name,
    )

    minimax_re = re.compile(r"^[a-zA-Z0-9_.-]+$")

    # Undotted names pass under the default (OpenAI) charset.
    for name in ("plain_name", "my-tool"):
        _validate_tool_function_name(name)

    # The default charset rejects the dot (model-specific relaxation only).
    for dotted in ("my-tool.v2", "weather.get_current"):
        with pytest.raises(InputError):
            _validate_tool_function_name(dotted)
        # ...but a model that opts in (e.g. MiniMax M3) accepts it.
        _validate_tool_function_name(dotted, minimax_re)

    tools = _convert_chat_completion_tools_to_token_generator_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "my-tool.v2",
                    "description": "Tool with special chars in name",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
        ],
        minimax_re,
    )
    assert tools is not None
    assert tools[0]["function"]["name"] == "my-tool.v2"

    # Names with disallowed characters are still rejected under any charset.
    for bad in ("has space", "comma,name", ""):
        with pytest.raises(InputError):
            _validate_tool_function_name(bad, minimax_re)


def test_openai_tool_function_name_length_limit() -> None:
    """Tool-name length cap is 1024 chars; empty and invalid charsets rejected."""
    from max.pipelines.context.exceptions import InputError
    from max.serve.router.openai_routes import (
        _MAX_TOOL_NAME_LEN,
        _validate_tool_function_name,
    )

    assert _MAX_TOOL_NAME_LEN == 1024

    # A name exactly at the cap is accepted.
    _validate_tool_function_name("a" * _MAX_TOOL_NAME_LEN)

    # One character past the cap is rejected.
    with pytest.raises(InputError):
        _validate_tool_function_name("a" * (_MAX_TOOL_NAME_LEN + 1))

    # Empty and invalid-character names are still rejected regardless of length.
    with pytest.raises(InputError):
        _validate_tool_function_name("")
    with pytest.raises(InputError):
        _validate_tool_function_name("has space")


async def test_openai_rejects_invalid_json_tool_call_arguments() -> None:
    """Assistant ``tool_calls.arguments`` that isn't valid JSON -> 400 (verifier 16_12)."""
    from max.pipelines.context.exceptions import InputError

    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{invalid json}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            ],
        }
    )
    with pytest.raises(InputError):
        await openai_parse_chat_completion_request(
            request, wrap_content=True, settings=Settings()
        )


async def test_openai_rejects_text_content_part_without_text_field() -> None:
    """``{'type': 'text'}`` parts must include ``text`` and error cleanly."""
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": [{"type": "text"}]}],
        }
    )

    with pytest.raises(
        InputError,
        match=r"Content part of type 'text' must include a 'text' field\.",
    ):
        await openai_parse_chat_completion_request(
            request, wrap_content=True, settings=Settings()
        )


async def test_openai_rejects_tool_call_id_mismatch() -> None:
    """A ``tool`` reply whose ``tool_call_id`` matches no tool_call -> 400 (verifier 16_08)."""
    from max.pipelines.context.exceptions import InputError

    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Beijing"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_999_wrong",
                    "content": "sunny",
                },
            ],
        }
    )
    with pytest.raises(InputError):
        await openai_parse_chat_completion_request(
            request, wrap_content=False, settings=Settings()
        )


async def test_openai_rejects_partial_tool_call_reply() -> None:
    """Answering only some of an assistant's tool_calls -> 400 (verifier 16_09)."""
    from max.pipelines.context.exceptions import InputError

    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Weather in Beijing and Shanghai?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Beijing"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Shanghai"}',
                            },
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
        }
    )
    with pytest.raises(InputError):
        await openai_parse_chat_completion_request(
            request, wrap_content=False, settings=Settings()
        )


async def test_openai_accepts_complete_tool_call_replies() -> None:
    """A fully-answered multi-call tool exchange is accepted (guards 16_08/16_09 over-rejection)."""
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Weather in Beijing and Shanghai?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Beijing"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Shanghai"}',
                            },
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_2", "content": "rainy"},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                {"role": "user", "content": "thanks"},
            ],
        }
    )
    (
        messages,
        _images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=Settings()
    )
    assert len(messages) == 5


async def test_openai_rejects_oversized_image() -> None:
    """An image whose resolved bytes exceed 10MB -> 400 (verifier 11_04)."""
    import base64 as _base64

    from max.pipelines.context.exceptions import InputError

    oversized = _base64.b64encode(b"\x00" * (10 * 1024 * 1024 + 1)).decode()
    data_url = f"data:image/png;base64,{oversized}"
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "What?"},
                    ],
                }
            ],
        }
    )
    with pytest.raises(InputError):
        await openai_parse_chat_completion_request(
            request,
            wrap_content=True,
            settings=Settings(),
            max_image_bytes=10 * 1024 * 1024,
        )


# A 1x1 PNG: tiny and decodable, so a batch passes the per-image decode check.
_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _request_with_n_images(n: int) -> CreateChatCompletionRequest:
    content = [
        {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}}
        for _ in range(n)
    ]
    return CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [{"role": "user", "content": content}]}
    )


async def test_openai_rejects_too_many_images() -> None:
    """More than 200 images on a single request -> 400 (spec 3b.i)."""
    from max.pipelines.context.exceptions import InputError

    request = _request_with_n_images(201)
    with pytest.raises(InputError):
        await openai_parse_chat_completion_request(
            request,
            wrap_content=True,
            settings=Settings(),
            max_images_per_request=200,
        )


async def test_openai_accepts_image_count_at_limit() -> None:
    """Exactly 200 images is accepted (the cap is exclusive, no off-by-one)."""
    request = _request_with_n_images(200)
    (
        _messages,
        images,
        _videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request,
        wrap_content=True,
        settings=Settings(),
        max_images_per_request=200,
    )
    assert len(images) == 200


async def test_openai_accepts_64mb_request_body() -> None:
    """A request body of at least 64MB parses without error (spec 3d)."""
    # ~64MiB of text content, well past the 64M floor once JSON-encoded.
    big_text = "a" * (64 * 1024 * 1024)
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": big_text}],
        }
    )
    (
        messages,
        images,
        videos,
        _decoded,
    ) = await openai_parse_chat_completion_request(
        request, wrap_content=True, settings=Settings()
    )
    assert len(messages) == 1
    assert not images
    assert not videos


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, True),  # defaults to True
        ({"reasoning_split": True}, True),
        ({"reasoning_split": False}, False),  # False is accepted and preserved
    ],
)
def test_reasoning_split(payload: dict[str, Any], expected: bool) -> None:
    """``reasoning_split`` defaults to True; both True and False are accepted."""
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        **payload,
    }
    request = CreateChatCompletionRequest.model_validate(body)
    assert request.reasoning_split is expected


def test_merge_tool_call_deltas_coalesces_same_index_in_one_chunk() -> None:
    """A name/id delta and its first args delta in one chunk merge to one entry.

    Regression for CENG-768: with ``STREAM_MIN_CHUNK_TOKENS`` batching a single
    ``parse_delta`` return carries both the call-start and the opening args for
    the same call. Emitting them as two ``tool_calls`` entries that share an
    index breaks strict OpenAI clients; the router must coalesce them into one
    first-frame entry ``{index, id, type, function: {name, arguments}}``.
    """
    from max.pipelines.modeling.types import ParsedToolCallDelta
    from max.serve.router.openai_routes import _merge_tool_call_deltas

    merged = _merge_tool_call_deltas(
        [
            ParsedToolCallDelta(index=0, id="call_abc", name="save_config"),
            ParsedToolCallDelta(index=0, arguments="{"),
        ]
    )
    assert len(merged) == 1
    entry = merged[0]
    assert entry.index == 0
    assert entry.id == "call_abc"
    assert entry.type == "function"
    assert entry.function is not None
    assert entry.function.name == "save_config"
    assert entry.function.arguments == "{"


def test_merge_tool_call_deltas_two_calls_one_chunk_one_entry_each() -> None:
    """Two calls firing in one chunk yield one entry each, ordered by index."""
    from max.pipelines.modeling.types import ParsedToolCallDelta
    from max.serve.router.openai_routes import _merge_tool_call_deltas

    merged = _merge_tool_call_deltas(
        [
            ParsedToolCallDelta(index=0, arguments=', "days": 2}'),
            ParsedToolCallDelta(index=1, id="call_xyz", name="get_time"),
            ParsedToolCallDelta(index=1, arguments='{"tz": "UTC"}'),
        ]
    )
    assert [e.index for e in merged] == [0, 1]
    # index 0 is a continuation: args only, no id/name/type.
    assert merged[0].id is None
    assert merged[0].type is None
    assert merged[0].function is not None
    assert merged[0].function.name is None
    assert merged[0].function.arguments == ', "days": 2}'
    # index 1 is a first frame: id + type + name + concatenated args.
    assert merged[1].id == "call_xyz"
    assert merged[1].type == "function"
    assert merged[1].function is not None
    assert merged[1].function.name == "get_time"
    assert merged[1].function.arguments == '{"tz": "UTC"}'


def test_merge_tool_call_deltas_first_frame_has_empty_args_string() -> None:
    """A first frame with no args in its chunk still carries ``arguments == ""``.

    Matches OpenAI's first-frame shape so a client that reads
    ``function.arguments`` unconditionally does not see ``null``.
    """
    from max.pipelines.modeling.types import ParsedToolCallDelta
    from max.serve.router.openai_routes import _merge_tool_call_deltas

    merged = _merge_tool_call_deltas(
        [ParsedToolCallDelta(index=0, id="call_1", name="ping")]
    )
    assert len(merged) == 1
    assert merged[0].function is not None
    assert merged[0].function.arguments == ""


def test_merge_tool_call_deltas_ignores_content_only_deltas() -> None:
    """Content-only deltas carry no tool call and produce no entry."""
    from max.pipelines.modeling.types import ParsedToolCallDelta
    from max.serve.router.openai_routes import _merge_tool_call_deltas

    assert (
        _merge_tool_call_deltas(
            [ParsedToolCallDelta(index=0, content="hello ")]
        )
        == []
    )


def test_merge_tool_call_deltas_empty_string_arg_is_present_not_absent() -> (
    None
):
    """An empty-string ``arguments`` is present-but-empty, not absent.

    Pins the ``is not None`` presence semantics: a lone ``arguments=""`` delta
    still yields one continuation entry (args-only, no id/type/name) rather
    than being dropped or promoted to a first frame. ``parse_delta`` never
    emits empty-string fields today, so this only fixes the edge's intent; it
    must stay behavior-identical on real input.
    """
    from max.pipelines.modeling.types import ParsedToolCallDelta
    from max.serve.router.openai_routes import _merge_tool_call_deltas

    merged = _merge_tool_call_deltas(
        [ParsedToolCallDelta(index=0, arguments="")]
    )
    assert len(merged) == 1
    assert merged[0].index == 0
    assert merged[0].id is None
    assert merged[0].type is None
    assert merged[0].function is not None
    assert merged[0].function.name is None
    assert merged[0].function.arguments == ""
