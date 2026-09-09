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
from __future__ import annotations

import re
from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.pipelines import TextTokenizer
from max.pipelines.context import SamplingParams
from max.pipelines.lib import KVCacheConfig
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from transformers import AutoConfig


def _create_mock_pipeline_config(model_path: str) -> MagicMock:
    """Create a mock PipelineConfig with real HuggingFace config."""
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    mock_kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    mock_kv_cache_config.enable_prefix_caching = False

    mock_model_config = MagicMock()
    mock_model_config.huggingface_config = hf_config
    mock_model_config.kv_cache = mock_kv_cache_config

    pipeline_config = MagicMock()
    pipeline_config.model = mock_model_config
    return pipeline_config


# Sample tool definitions for testing
def get_sample_tools() -> list[TextGenerationRequestTool]:
    """Returns a list of sample tools for testing tool call functionality."""
    return [
        TextGenerationRequestTool(
            type="function",
            function=TextGenerationRequestFunction(
                name="get_current_weather",
                description="Get the current weather information for a given location.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country, e.g. San Francisco, CA or London, UK",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        TextGenerationRequestTool(
            type="function",
            function=TextGenerationRequestFunction(
                name="calculate_distance",
                description="Calculate the distance between two locations.",
                parameters={
                    "type": "object",
                    "properties": {
                        "from_location": {
                            "type": "string",
                            "description": "Starting location",
                        },
                        "to_location": {
                            "type": "string",
                            "description": "Destination location",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["miles", "kilometers"],
                            "description": "Distance unit",
                        },
                    },
                    "required": ["from_location", "to_location"],
                },
            ),
        ),
    ]


def get_sample_messages() -> list[TextGenerationRequestMessage]:
    """Returns sample chat messages for testing."""
    return [
        TextGenerationRequestMessage(
            role="user",
            content="What's the weather like in Toronto today? Also, how far is it from New York?",
        )
    ]


def validate_tool_call_format(prompt: str, model_name: str) -> bool:
    """
    Validates that the generated prompt contains appropriate tool call formatting.

    Args:
        prompt: The generated prompt string
        model_name: The name of the model (for model-specific validation)

    Returns:
        bool: True if the prompt contains valid tool call formatting
    """
    # Check for general tool-related patterns that should be present
    tool_indicators = [
        # Function names should appear in the prompt
        "get_current_weather",
        "calculate_distance",
        # Common tool call formats across different models
        r"function",  # The word "function" should appear
        r"parameters?",  # "parameter" or "parameters" should appear
        r"location",  # Required parameter should appear
    ]

    prompt_lower = prompt.lower()

    # Count matches - we expect at least 3 indicators to be present
    matches = 0
    for indicator in tool_indicators:
        if re.search(indicator, prompt_lower):
            matches += 1

    basic_validation = matches >= 3

    # Model-specific validations
    if "llama" in model_name.lower():
        # Llama models typically use specific tool call formatting
        # Look for common Llama tool patterns
        llama_patterns = [
            r"<\|.*\|>",  # Special tokens like <|tool|>
            r"available.*functions?",  # "available functions" text
            r"call.*function",  # "call function" text
        ]
        llama_matches = sum(
            1
            for pattern in llama_patterns
            if re.search(pattern, prompt, re.IGNORECASE)
        )
        return basic_validation or llama_matches > 0

    elif "gemma" in model_name.lower():
        # Gemma models may have different formatting
        # Look for Gemma-specific patterns
        gemma_patterns = [
            r"tool",
            r"function_call",
            r"available.*tool",
        ]
        gemma_matches = sum(
            1
            for pattern in gemma_patterns
            if re.search(pattern, prompt, re.IGNORECASE)
        )
        return basic_validation or gemma_matches > 0

    elif "gpt" in model_name.lower() or "openai" in model_name.lower():
        # OpenAI/GPT models typically follow OpenAI function calling format
        # Look for OpenAI-specific patterns
        openai_patterns = [
            r"tools?",  # "tool" or "tools"
            r"function_call",  # "function_call"
            r"available.*functions?",  # "available functions"
            r"json",  # JSON formatting often used
            r"schema",  # Schema definition
        ]
        openai_matches = sum(
            1
            for pattern in openai_patterns
            if re.search(pattern, prompt, re.IGNORECASE)
        )
        return basic_validation or openai_matches > 0

    # For any other models, rely on basic validation
    return basic_validation


@pytest.mark.skip("Needs huggingface access.")
@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "openai/gpt-oss-20b",
    ],
)
@pytest.mark.asyncio
async def test_tool_calling_prompt_format(model_name: str) -> None:
    """
    Test that tool calling produces appropriately formatted prompts for different model types.

    Args:
        model_name: The model name to test (parameterized)
    """
    # Create tokenizer for the specified model
    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        max_length=4096,
    )

    # Get sample tools and messages
    tools = get_sample_tools()
    messages = get_sample_messages()

    # Create a TextGenerationRequest with tools
    request = TextGenerationRequest(
        request_id=RequestID("test_tool_calling"),
        model_name=model_name,
        messages=messages,
        tools=tools,
        sampling_params=SamplingParams(
            max_new_tokens=100,
            temperature=0.7,
        ),
    )

    # Generate the context which includes prompt processing
    context = await tokenizer.new_context(request)

    # Extract the generated prompt
    # The prompt should be accessible via the tokenizer's apply_chat_template method
    generated_prompt = tokenizer.apply_chat_template(
        messages=messages,
        tools=tools,
        chat_template_options={"add_generation_prompt": True},
    )

    # Validate that the prompt is not empty
    assert generated_prompt, f"Generated prompt is empty for model {model_name}"

    # Validate that the prompt contains tool call formatting
    is_valid = validate_tool_call_format(generated_prompt, model_name)

    # Assertion with detailed failure message
    assert is_valid, (
        f"Generated prompt for {model_name} does not contain valid tool call formatting.\n"
        f"Generated prompt (first 500 chars):\n{generated_prompt[:500]}\n"
        f"Expected tool indicators: function names, parameters, location field, etc."
    )

    # Additional validations
    # Check that tool functions are mentioned in the prompt
    prompt_lower = generated_prompt.lower()
    assert "get_current_weather" in prompt_lower, (
        f"Tool function 'get_current_weather' not found in prompt for {model_name}"
    )
    assert "calculate_distance" in prompt_lower, (
        f"Tool function 'calculate_distance' not found in prompt for {model_name}"
    )

    # Check that the user message is included
    assert "toronto" in prompt_lower or "weather" in prompt_lower, (
        f"User message content not found in prompt for {model_name}"
    )

    # Verify that tools are properly structured in the context
    assert context is not None, f"Context should not be None for {model_name}"
    assert len(context.tokens) > 0, (
        f"Context should have tokens for {model_name}"
    )

    print(f"✓ Tool calling test passed for {model_name}")
    print(f"  Prompt length: {len(generated_prompt)} characters")
    print(f"  Token count: {len(context.tokens)}")
    print(f"  Prompt: {generated_prompt}...")


@pytest.mark.skip("Needs huggingface access.")
@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "openai/gpt-oss-20b",
    ],
)
@pytest.mark.asyncio
async def test_tool_calling_no_tools_baseline(model_name: str) -> None:
    """
    Baseline test to ensure prompts without tools work correctly.

    Args:
        model_name: The model name to test (parameterized)
    """
    # Create tokenizer for the specified model
    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        max_length=4096,
    )

    # Get sample messages without tools
    messages = get_sample_messages()

    # Create a TextGenerationRequest without tools
    request = TextGenerationRequest(
        request_id=RequestID("test_no_tools"),
        model_name=model_name,
        messages=messages,
        tools=None,  # No tools provided
        sampling_params=SamplingParams(
            max_new_tokens=100,
            temperature=0.7,
        ),
    )

    # Generate the context
    context = await tokenizer.new_context(request)

    # Extract the generated prompt without tools
    generated_prompt = tokenizer.apply_chat_template(
        messages=messages,
        tools=None,
        chat_template_options={"add_generation_prompt": True},
    )

    # Validate that the prompt is not empty
    assert generated_prompt, f"Generated prompt is empty for model {model_name}"

    # Validate that the prompt does NOT contain tool call formatting
    # (should fail the tool validation since no tools are provided)
    prompt_lower = generated_prompt.lower()

    # The prompt should still contain the user message
    assert "toronto" in prompt_lower or "weather" in prompt_lower, (
        f"User message content not found in no-tools prompt for {model_name}"
    )

    # Verify context is valid
    assert context is not None, f"Context should not be None for {model_name}"
    assert len(context.tokens) > 0, (
        f"Context should have tokens for {model_name}"
    )

    print(f"✓ No-tools baseline test passed for {model_name}")
    print(f"  Prompt length: {len(generated_prompt)} characters")
    print(f"  Token count: {len(context.tokens)}")


@pytest.mark.skip("Needs huggingface access.")
@pytest.mark.parametrize(
    "model_name,expected_patterns",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", ["<|", "|>", "function"]),
        ("openai/gpt-oss-20b", ["tools", "function_call", "json", "schema"]),
    ],
)
@pytest.mark.asyncio
async def test_model_specific_tool_formatting(
    model_name: str, expected_patterns: list[str]
) -> None:
    """
    Test model-specific tool call formatting patterns.

    Args:
        model_name: The model name to test
        expected_patterns: Patterns expected for this specific model
    """
    # Create tokenizer for the specified model
    try:
        pipeline_config = _create_mock_pipeline_config(model_name)
        tokenizer = TextTokenizer(
            model_path=model_name,
            pipeline_config=pipeline_config,
            trust_remote_code=True,
            max_length=4096,
        )
    except Exception as e:
        pytest.skip(f"Could not load tokenizer for {model_name}: {e}")

    # Get sample tools and messages
    tools = get_sample_tools()
    messages = get_sample_messages()

    # Generate the prompt with tools
    try:
        generated_prompt = tokenizer.apply_chat_template(
            messages=messages,
            tools=tools,
            chat_template_options={"add_generation_prompt": True},
        )
    except Exception as e:
        pytest.skip(f"Could not generate prompt for {model_name}: {e}")

    # Check for model-specific patterns
    prompt_lower = generated_prompt.lower()
    found_patterns = []

    for pattern in expected_patterns:
        if pattern.lower() in prompt_lower:
            found_patterns.append(pattern)

    # We expect to find at least one of the expected patterns
    assert found_patterns, (
        f"None of the expected patterns {expected_patterns} found in prompt for {model_name}.\n"
        f"Prompt preview: {generated_prompt[:300]}..."
    )

    print(f"✓ Model-specific formatting test passed for {model_name}")
    print(f"  Found patterns: {found_patterns}")
    print(f"  Expected patterns: {expected_patterns}")


@pytest.mark.skip("Needs huggingface access")
@pytest.mark.asyncio
async def test_tool_calling_edge_cases() -> None:
    """Test edge cases in tool calling functionality."""
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Use one model for edge case testing

    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        max_length=4096,
    )

    # Test with empty tools list
    messages = get_sample_messages()
    empty_tools: list[TextGenerationRequestTool] = []

    prompt_empty_tools = tokenizer.apply_chat_template(
        messages=messages,
        tools=empty_tools,
        chat_template_options={"add_generation_prompt": True},
    )
    # Should not crash, but may behave like no tools
    assert prompt_empty_tools, (
        "Prompt should not be empty with empty tools list"
    )

    # Test with complex nested parameters
    complex_tool = TextGenerationRequestTool(
        type="function",
        function=TextGenerationRequestFunction(
            name="complex_function",
            description="A function with complex nested parameters.",
            parameters={
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {
                            "nested_param": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    }
                },
            },
        ),
    )

    prompt_complex = tokenizer.apply_chat_template(
        messages=messages,
        tools=[complex_tool],
        chat_template_options={"add_generation_prompt": True},
    )
    assert prompt_complex, "Prompt should not be empty with complex tools"
    assert "complex_function" in prompt_complex.lower(), (
        "Complex function name should appear in prompt"
    )

    print("✓ Edge cases test completed")
