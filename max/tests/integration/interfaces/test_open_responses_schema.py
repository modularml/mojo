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
"""Tests for OpenResponses Pydantic schema."""

import json

import numpy as np
import pytest
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.context.status import GenerationStatus
from max.pipelines.modeling.types import RequestID
from max.pipelines.request.open_responses import (
    AssistantMessage,
    FunctionCall,
    FunctionToolParam,
    ImageGenerationDetails,
    InputTextContent,
    OpenResponsesRequestBody,
    OutputTextContent,
    ResponseResource,
    SystemMessage,
    ToolChoiceValueEnum,
    Usage,
    UserMessage,
)
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    MaxProviderOptions,
    ProviderOptions,
)
from pydantic import ValidationError


def test_import_all_types() -> None:
    """Test that all OpenResponses types can be imported."""

    # If we get here, all imports succeeded
    assert True


def test_create_response_body_minimal() -> None:
    """Test creating a minimal OpenResponsesRequestBody with required fields only."""
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="Hello, world!",
    )

    assert request.model == "gpt-4"
    assert request.input == "Hello, world!"
    assert request.temperature is None
    assert request.max_output_tokens is None


def test_create_response_body_with_messages() -> None:
    """Test OpenResponsesRequestBody with structured message input."""
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input=[
            UserMessage(
                role="user",
                content="What's the weather in San Francisco?",
            )
        ],
        temperature=0.7,
        max_output_tokens=1000,
    )

    assert request.model == "gpt-4"
    assert len(request.input) == 1
    assert isinstance(request.input[0], UserMessage)
    assert request.temperature == 0.7
    assert request.max_output_tokens == 1000


def test_create_response_body_with_tools() -> None:
    """Test OpenResponsesRequestBody with tool definitions."""
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="What's the weather?",
        tools=[
            FunctionToolParam(
                type="function",
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            )
        ],
        tool_choice=ToolChoiceValueEnum.auto,
    )

    assert request.tools is not None
    assert len(request.tools) == 1
    assert request.tools[0].name == "get_weather"
    assert request.tool_choice == ToolChoiceValueEnum.auto


def test_user_message_with_string_content() -> None:
    """Test UserMessage with simple string content."""
    msg = UserMessage(
        role="user",
        content="Hello!",
    )

    assert msg.role == "user"
    assert msg.content == "Hello!"
    assert msg.name is None


def test_user_message_with_structured_content() -> None:
    """Test UserMessage with structured content."""
    msg = UserMessage(
        role="user",
        content=[
            InputTextContent(type="input_text", text="Look at this image:"),
        ],
    )

    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], InputTextContent)
    assert msg.content[0].text == "Look at this image:"


def test_assistant_message_with_tool_calls() -> None:
    """Test AssistantMessage with tool calls."""
    msg = AssistantMessage(
        role="assistant",
        content="I'll check the weather for you.",
        tool_calls=[
            FunctionCall(
                id="call_123",
                type="function",
                name="get_weather",
                arguments='{"location": "San Francisco"}',
            )
        ],
    )

    assert msg.role == "assistant"
    assert msg.content == "I'll check the weather for you."
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "get_weather"


def test_system_message() -> None:
    """Test SystemMessage creation."""
    msg = SystemMessage(
        role="system",
        content="You are a helpful assistant.",
    )

    assert msg.role == "system"
    assert msg.content == "You are a helpful assistant."


def test_response_resource_minimal() -> None:
    """Test creating a minimal ResponseResource."""
    response = ResponseResource(
        id="resp_123",
        object="response",
        created_at=1234567890,
        status="completed",
        model="gpt-4",
    )

    assert response.id == "resp_123"
    assert response.object == "response"
    assert response.status == "completed"
    assert response.model == "gpt-4"
    assert response.output is None


def test_models_are_frozen() -> None:
    """Test that all models are frozen (immutable)."""
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="Hello!",
    )

    # Attempting to modify should raise an error
    # Pydantic frozen models raise ValidationError on attribute assignment
    with pytest.raises(ValidationError):
        request.model = "gpt-3.5"  # type: ignore[misc]


def test_temperature_validation() -> None:
    """Test temperature field validation constraints."""
    # Valid temperature
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="Hello!",
        temperature=0.7,
    )
    assert request.temperature == 0.7

    # Temperature too high should fail
    with pytest.raises(ValidationError):
        OpenResponsesRequestBody(
            model="gpt-4",
            input="Hello!",
            temperature=2.5,
        )

    # Temperature too low should fail
    with pytest.raises(ValidationError):
        OpenResponsesRequestBody(
            model="gpt-4",
            input="Hello!",
            temperature=-0.5,
        )


def test_top_p_validation() -> None:
    """Test top_p field validation constraints."""
    # Valid top_p
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="Hello!",
        top_p=0.9,
    )
    assert request.top_p == 0.9

    # top_p too high should fail
    with pytest.raises(ValidationError):
        OpenResponsesRequestBody(
            model="gpt-4",
            input="Hello!",
            top_p=1.5,
        )


def test_json_serialization() -> None:
    """Test that models can be serialized to JSON."""
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input=[
            UserMessage(
                role="user",
                content="Hello!",
            )
        ],
        temperature=0.7,
    )

    # Serialize to JSON
    json_str = request.model_dump_json()
    json_data = json.loads(json_str)

    assert json_data["model"] == "gpt-4"
    assert json_data["temperature"] == 0.7
    assert len(json_data["input"]) == 1
    assert json_data["input"][0]["role"] == "user"


def test_json_deserialization() -> None:
    """Test that models can be deserialized from JSON."""
    json_data = {
        "model": "gpt-4",
        "input": "Hello!",
        "temperature": 0.8,
        "max_output_tokens": 500,
    }

    request = OpenResponsesRequestBody.model_validate(json_data)

    assert request.model == "gpt-4"
    assert request.input == "Hello!"
    assert request.temperature == 0.8
    assert request.max_output_tokens == 500


def test_output_text_content_with_annotations() -> None:
    """Test OutputTextContent with annotations."""
    content = OutputTextContent(
        type="output_text",
        text="The weather in San Francisco is 65°F.",
        annotations=[0, 1],
    )

    assert content.type == "output_text"
    assert content.text == "The weather in San Francisco is 65°F."
    assert content.annotations == [0, 1]


def test_create_response_body_with_provider_options() -> None:
    """Test OpenResponsesRequestBody with provider options."""
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="Hello, world!",
        provider_options=ProviderOptions(
            max=MaxProviderOptions(target_endpoint="instance-123")
        ),
    )

    assert request.model == "gpt-4"
    assert request.provider_options is not None
    assert request.provider_options.max is not None
    assert request.provider_options.max.target_endpoint == "instance-123"
    assert request.provider_options.image is None
    assert request.provider_options.video is None


def test_create_response_body_with_max_and_image_options() -> None:
    """Test OpenResponsesRequestBody with both MAX and image modality options."""
    request = OpenResponsesRequestBody(
        model="vision-model",
        input="Describe this image",
        provider_options=ProviderOptions(
            max=MaxProviderOptions(target_endpoint="instance-456"),
            image=ImageProviderOptions(width=512, height=512),
        ),
    )

    assert request.provider_options is not None
    assert request.provider_options.max is not None
    assert request.provider_options.max.target_endpoint == "instance-456"
    assert request.provider_options.image is not None
    assert request.provider_options.image.width == 512
    assert request.provider_options.image.height == 512


def test_create_response_body_provider_options_json_serialization() -> None:
    """Test that OpenResponsesRequestBody with provider options serializes correctly."""
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="Hello!",
        provider_options=ProviderOptions(
            max=MaxProviderOptions(target_endpoint="instance-123"),
            image=ImageProviderOptions(width=1024, height=768),
        ),
    )

    json_str = request.model_dump_json()
    json_data = json.loads(json_str)

    assert json_data["model"] == "gpt-4"
    assert json_data["input"] == "Hello!"
    assert (
        json_data["provider_options"]["max"]["target_endpoint"]
        == "instance-123"
    )
    assert json_data["provider_options"]["image"]["width"] == 1024
    assert json_data["provider_options"]["image"]["height"] == 768


def test_create_response_body_provider_options_json_deserialization() -> None:
    """Test that OpenResponsesRequestBody with provider options deserializes correctly."""
    json_data = {
        "model": "gpt-4",
        "input": "Hello!",
        "provider_options": {
            "max": {"target_endpoint": "instance-123"},
            "image": {"width": 512, "height": 512},
        },
    }

    request = OpenResponsesRequestBody.model_validate(json_data)

    assert request.model == "gpt-4"
    assert request.input == "Hello!"
    assert request.provider_options is not None
    assert request.provider_options.max is not None
    assert request.provider_options.max.target_endpoint == "instance-123"
    assert request.provider_options.image is not None
    assert request.provider_options.image.width == 512
    assert request.provider_options.image.height == 512


def test_create_response_body_without_provider_options() -> None:
    """Test that OpenResponsesRequestBody works without provider options."""
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="Hello!",
    )

    assert request.model == "gpt-4"
    # provider_options defaults to ProviderOptions with default ImageProviderOptions
    assert request.provider_options is not None
    assert request.provider_options.max is None
    assert request.provider_options.image is not None
    assert request.provider_options.image.guidance_scale == 3.5
    assert request.provider_options.image.true_cfg_scale == 1.0
    assert request.provider_options.image.steps is None
    assert request.provider_options.image.num_images == 1
    assert request.provider_options.video is None


def test_create_response_body_with_partial_provider_options() -> None:
    """Test OpenResponsesRequestBody with only some provider options fields."""
    # Only MAX options
    request = OpenResponsesRequestBody(
        model="gpt-4",
        input="Hello!",
        provider_options=ProviderOptions(
            max=MaxProviderOptions(target_endpoint="instance-123")
        ),
    )
    assert request.provider_options is not None
    assert request.provider_options.max is not None
    assert request.provider_options.image is None
    assert request.provider_options.video is None

    # Only image options
    request = OpenResponsesRequestBody(
        model="vision-model",
        input="Describe this",
        provider_options=ProviderOptions(
            image=ImageProviderOptions(width=512, height=512)
        ),
    )
    assert request.provider_options is not None
    assert request.provider_options.max is None
    assert request.provider_options.image is not None
    assert request.provider_options.image.width == 512
    assert request.provider_options.image.height == 512


def test_image_generation_details_from_images() -> None:
    """Details are measured from the actual output arrays."""
    pixel_data = np.zeros((2, 512, 768, 3), dtype=np.uint8)

    details = ImageGenerationDetails.from_images(pixel_data, steps=28)

    assert details.width == 768
    assert details.height == 512
    assert details.megapixels == pytest.approx(0.393216)
    assert details.steps == 28
    assert details.image_count == 2


def test_image_generation_details_from_images_empty_raises() -> None:
    """Empty pixel data cannot produce image generation details."""
    with pytest.raises(ValueError, match="empty pixel data"):
        ImageGenerationDetails.from_images(
            np.zeros((0, 512, 512, 3), dtype=np.uint8), steps=28
        )


def test_from_generation_output_populates_usage() -> None:
    """Pipeline-reported usage flows through to the response."""
    usage = Usage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        image_generation_details=ImageGenerationDetails(
            width=1024,
            height=1024,
            megapixels=1.048576,
            steps=28,
            image_count=1,
        ),
    )
    generation_output = GenerationOutput(
        request_id=RequestID(value="req-1"),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[OutputTextContent(type="output_text", text="ok")],
        usage=usage,
    )

    response = ResponseResource.from_generation_output(
        generation_output, model="flux.2-klein-9b"
    )

    assert response.usage == usage
    assert response.usage is not None
    assert response.usage.image_generation_details is not None
    assert response.usage.image_generation_details.megapixels == pytest.approx(
        1.048576
    )


def test_from_generation_output_without_usage() -> None:
    """Outputs that carry no usage keep the response usage unset."""
    generation_output = GenerationOutput(
        request_id=RequestID(value="req-1"),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[OutputTextContent(type="output_text", text="ok")],
    )

    response = ResponseResource.from_generation_output(
        generation_output, model="flux.2-klein-9b"
    )

    assert response.usage is None
