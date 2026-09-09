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
"""Test Suite for Unit Testing the TextGenerationPipeline with structured output."""

import asyncio
import json
from typing import cast

import numpy as np
from max.driver import DeviceSpec
from max.pipelines import (
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
    SamplingConfig,
    TextGenerationPipeline,
)
from max.pipelines.context import (
    SamplingParams,
    TextContext,
    TextGenerationResponseFormat,
)
from max.pipelines.lib import (
    OverlapTextGenerationPipeline,
    TextTokenizer,
)
from max.pipelines.lib.registry import PipelineRegistry
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)

pytest_plugins = "test_common.registry"


def test_smollm_with_structured_output_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    pipeline_config = PipelineArgs(
        model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=8192,
        sampling=SamplingConfig(enable_structured_output=True),
        runtime=PipelineRuntimeConfig(max_batch_size=1),
    )

    retrieved = pipeline_registry.retrieve_factory(
        PipelineConfig.from_args(pipeline_config)
    )
    tokenizer = retrieved.tokenizer
    pipeline_factory = retrieved.factory
    assert isinstance(tokenizer, TextTokenizer)

    prompt = """
    Please provide a json response, with the person's name and age extracted from the excerpt.
    For example, provided an excerpt 'Bob Dylan is 83 years old.' return with {"badnamey": "Bob Dylan", "badagey": 83}.
    Please extract the person's name and age from the following excerpt:
    'John Mayer is 47 years old.'
    """

    request_id = RequestID("request_0")
    sampling_params = SamplingParams(max_new_tokens=50, top_k=1)
    request = TextGenerationRequest(
        model_name=pipeline_config.model_path,
        request_id=request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        sampling_params=sampling_params,
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "age": {
                        "type": "integer",
                    },
                },
                "required": ["name", "age"],
            },
            grammar_enforced=True,
            tools_forced=False,
        ),
    )

    # Get Context
    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    # Pipeline may be TextGenerationPipeline or OverlapTextGenerationPipeline
    # depending on auto-enable settings. Both support structured output.
    assert isinstance(
        pipeline, (TextGenerationPipeline, OverlapTextGenerationPipeline)
    )
    # Cast for type checker - both have the same interface for what we need.
    pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context)

    tokens: list[int] = []
    max_iterations = 60
    for _ in range(max_iterations):
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]]
        )
        kv_manager.alloc(context)
        response = pipeline.execute(inputs)

        # Handle overlap pipeline which may return empty response on first iteration
        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)

            if response[request_id].is_done:
                break

    # Flush remaining outputs with empty batch (for overlap pipeline)
    if isinstance(pipeline, OverlapTextGenerationPipeline):
        empty_inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[]]
        )
        response = pipeline.execute(empty_inputs)
        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)

    response_content = asyncio.run(
        tokenizer.decode(np.array(tokens), skip_special_tokens=True)
    )

    result = json.loads(response_content)

    assert result["name"] == "John Mayer"
    assert result["age"] == 47


def test_multistep_structured_output_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test structured output over multiple single-step pipeline invocations."""
    pipeline_config = PipelineArgs(
        model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=8192,
        sampling=SamplingConfig(enable_structured_output=True),
        # Disable overlap scheduler: multi-step execution (num_steps > 1) is
        # only supported in TextGenerationPipeline, not OverlapTextGenerationPipeline.
        # Use force=True to prevent auto-enable from overriding our setting.
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            enable_overlap_scheduler=False,
            force=True,
        ),
    )

    retrieved = pipeline_registry.retrieve_factory(
        PipelineConfig.from_args(pipeline_config)
    )
    tokenizer = retrieved.tokenizer
    pipeline_factory = retrieved.factory
    assert isinstance(tokenizer, TextTokenizer)

    prompt = """Extract the person's name and age from: 'Alice Smith is 30 years old.'"""

    request_id = RequestID("multistep_request")
    sampling_params = SamplingParams(max_new_tokens=50, top_k=1)
    request = TextGenerationRequest(
        model_name=pipeline_config.model_path,
        request_id=request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        sampling_params=sampling_params,
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            grammar_enforced=True,
            tools_forced=False,
        ),
    )

    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    assert isinstance(pipeline, TextGenerationPipeline)
    pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context)

    tokens = []
    while True:
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]]
        )
        kv_manager.alloc(context)
        response = pipeline.execute(inputs)

        for token in response[request_id].tokens:
            tokens.append(token)

        if response[request_id].is_done:
            break

    response_content = asyncio.run(
        tokenizer.decode(np.array(tokens), skip_special_tokens=True)
    )

    # Verify we got valid JSON matching the schema
    result = json.loads(response_content)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)


def test_multi_step_guided_decoding_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test that guided decoding works over repeated single-step invocations."""
    pipeline_config = PipelineArgs(
        model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=8192,
        sampling=SamplingConfig(enable_structured_output=True),
        # Disable overlap scheduler: multi-step execution (num_steps > 1) is
        # only supported in TextGenerationPipeline, not OverlapTextGenerationPipeline.
        # Use force=True to prevent auto-enable from overriding our setting.
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            enable_overlap_scheduler=False,
            force=True,
        ),
    )

    retrieved = pipeline_registry.retrieve_factory(
        PipelineConfig.from_args(pipeline_config)
    )
    tokenizer = retrieved.tokenizer
    pipeline_factory = retrieved.factory
    assert isinstance(tokenizer, TextTokenizer)

    prompt = """Return JSON: 'Bob Jones is 25.'"""

    request_id = RequestID("multi_step_guided_test")
    request = TextGenerationRequest(
        model_name=pipeline_config.model_path,
        request_id=request_id,
        messages=[TextGenerationRequestMessage(role="user", content=prompt)],
        sampling_params=SamplingParams(max_new_tokens=30, top_k=1),
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            grammar_enforced=True,
            tools_forced=False,
        ),
    )

    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    assert isinstance(pipeline, TextGenerationPipeline)
    pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context)

    # Single-step execution with guided decoding per scheduler iteration.
    max_iterations = 20

    for _ in range(max_iterations):
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]]
        )
        kv_manager.alloc(context)
        response = pipeline.execute(inputs)

        if response[request_id].is_done:
            break


def test_overlap_pipeline_structured_output_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test overlap pipeline with structured output.

    This verifies that the OverlapTextGenerationPipeline correctly handles
    structured output (guided decoding), producing
    valid JSON output conforming to the schema.
    """

    pipeline_config = PipelineArgs(
        model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=8192,
        sampling=SamplingConfig(enable_structured_output=True),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            enable_overlap_scheduler=True,
        ),
    )

    retrieved = pipeline_registry.retrieve_factory(
        PipelineConfig.from_args(pipeline_config)
    )
    tokenizer = retrieved.tokenizer
    pipeline_factory = retrieved.factory
    assert isinstance(tokenizer, TextTokenizer)

    prompt = """Extract name and age from: 'Charlie Brown is 8 years old.'"""

    request_id = RequestID("overlap_structured")
    request = TextGenerationRequest(
        model_name=pipeline_config.model_path,
        request_id=request_id,
        messages=[TextGenerationRequestMessage(role="user", content=prompt)],
        sampling_params=SamplingParams(max_new_tokens=50, top_k=1),
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            grammar_enforced=True,
            tools_forced=False,
        ),
    )

    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    # Verify we got the overlap pipeline
    assert isinstance(pipeline, OverlapTextGenerationPipeline)
    pipeline = cast(OverlapTextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context)

    tokens: list[int] = []
    max_iterations = 60  # More iterations needed with single-step execution

    for _ in range(max_iterations):
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]]
        )
        kv_manager.alloc(context)

        # For structured output, overlap pipeline syncs immediately (no overlap)
        # so results are returned in the same call, not delayed.
        response = pipeline.execute(inputs)

        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)
            if response[request_id].is_done:
                break

    # For normal overlap (non-structured-output), this would flush pending outputs.
    # For structured output with immediate sync, there are no pending outputs.
    empty_inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[]]
    )
    response = pipeline.execute(empty_inputs)
    if request_id in response:
        for token in response[request_id].tokens:
            tokens.append(token)

    response_content = asyncio.run(
        tokenizer.decode(np.array(tokens), skip_special_tokens=True)
    )

    # Verify valid JSON matching schema
    result = json.loads(response_content)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)


def test_heterogeneous_batch_structured_output_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test mixed batch with both structured and non-structured output requests.

    Verifies that when a batch contains both requests with json_schema (structured
    output) and requests without json_schema (free-form), each request is handled
    correctly:
    - Structured output requests produce valid JSON conforming to their schema
    - Non-structured requests generate unconstrained output (not blocked by bitmask)
    """
    pipeline_config = PipelineArgs(
        model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=8192,
        sampling=SamplingConfig(enable_structured_output=True),
        # Use batch_size=2 for heterogeneous batch testing.
        # Disable overlap scheduler for simpler output handling.
        runtime=PipelineRuntimeConfig(
            max_batch_size=2,
            enable_overlap_scheduler=False,
            force=True,
        ),
    )

    retrieved = pipeline_registry.retrieve_factory(
        PipelineConfig.from_args(pipeline_config)
    )
    tokenizer = retrieved.tokenizer
    pipeline_factory = retrieved.factory
    assert isinstance(tokenizer, TextTokenizer)

    # Request 1: Structured output with JSON schema
    structured_request_id = RequestID("structured_request")
    structured_request = TextGenerationRequest(
        model_name=pipeline_config.model_path,
        request_id=structured_request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="Extract: 'Alice is 30 years old.'",
            )
        ],
        sampling_params=SamplingParams(max_new_tokens=50, top_k=1),
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
                "strict": True,
            },
            grammar_enforced=True,
            tools_forced=False,
        ),
    )

    # Request 2: Non-structured output (no json_schema)
    freeform_request_id = RequestID("freeform_request")
    freeform_request = TextGenerationRequest(
        model_name=pipeline_config.model_path,
        request_id=freeform_request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="Say hello in one sentence.",
            )
        ],
        sampling_params=SamplingParams(max_new_tokens=20, top_k=1),
        # No response_format - this is free-form generation
    )

    # Create contexts
    structured_ctx: TextContext = asyncio.run(
        tokenizer.new_context(structured_request)
    )
    freeform_ctx: TextContext = asyncio.run(
        tokenizer.new_context(freeform_request)
    )

    # Verify one has json_schema and one doesn't
    assert structured_ctx.json_schema is not None
    assert freeform_ctx.json_schema is None

    pipeline = pipeline_factory()
    assert isinstance(pipeline, TextGenerationPipeline)
    pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager

    # Claim KV cache for both requests
    kv_manager.claim(structured_ctx)
    kv_manager.claim(freeform_ctx)

    structured_tokens: list[int] = []
    freeform_tokens: list[int] = []
    max_iterations = 40

    # Run both contexts in the same batch
    active_contexts = [structured_ctx, freeform_ctx]

    for _ in range(max_iterations):
        if not active_contexts:
            break

        # Allocate KV cache for active contexts
        for ctx in active_contexts:
            kv_manager.alloc(ctx)

        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[active_contexts]
        )
        response = pipeline.execute(inputs)

        # Collect tokens and check completion
        contexts_to_remove = []
        for ctx in active_contexts:
            if ctx.request_id in response:
                resp = response[ctx.request_id]
                if ctx.request_id == structured_request_id:
                    structured_tokens.extend(resp.tokens)
                else:
                    freeform_tokens.extend(resp.tokens)

                if resp.is_done:
                    contexts_to_remove.append(ctx)

        # Remove completed contexts from active batch
        for ctx in contexts_to_remove:
            active_contexts.remove(ctx)

    # Verify structured output produced valid JSON
    structured_response = asyncio.run(
        tokenizer.decode(np.array(structured_tokens), skip_special_tokens=True)
    )
    result = json.loads(structured_response)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)

    # Verify free-form output was generated (not blocked)
    assert len(freeform_tokens) > 0, "Free-form request should generate tokens"
    freeform_response = asyncio.run(
        tokenizer.decode(np.array(freeform_tokens), skip_special_tokens=True)
    )
    # Free-form response should contain some text (not empty or just whitespace)
    assert len(freeform_response.strip()) > 0, (
        "Free-form request should produce non-empty output"
    )
