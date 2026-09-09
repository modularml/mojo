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
"""Integration tests for structured output with speculative decoding (EAGLE).

These tests verify that grammar constraints (JSON schema) are correctly applied
during speculative decoding target verification, producing valid JSON output.

NOTE: These tests require downloading EAGLE models (~7GB for 3B model pair) and
take significant time to compile. They are marked for the HF workflow.
"""

import asyncio
import json
import time

import numpy as np
import pytest
from max.driver import DeviceSpec
from max.pipelines import (
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
    SamplingConfig,
)
from max.pipelines.context import (
    SamplingParams,
    TextContext,
    TextGenerationResponseFormat,
)
from max.pipelines.lib import MAXModelConfig, TextTokenizer
from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
)
from max.pipelines.lib.registry import PipelineRegistry
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)

pytest_plugins = "test_common.registry"


@pytest.mark.timeout(600)  # 10 minutes for model download + compile
def test_eagle_structured_output_json_schema_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test that EAGLE speculative decoding with structured output produces valid JSON.

    This test verifies the end-to-end integration of:
    1. EAGLE speculative decoding (draft + target verification)
    2. Structured output via JSON schema grammar constraints
    3. Bitmask-constrained acceptance sampling during target verification

    The grammar constraints are applied only during target model verification,
    not during draft token generation (following vLLM's approach).
    """
    # Use Llama-3.2-3B-Instruct as target with EAGLE-Llama-3.2-3B-Instruct-bf16
    # as the draft model. This is the smallest EAGLE model pair available.
    pipeline_config = PipelineArgs(
        model_path="meta-llama/Llama-3.2-3B-Instruct",
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=2048,
        draft_model=MAXModelConfig(
            model_path="atomicapple0/EAGLE-Llama-3.2-3B-Instruct-bf16",
            quantization_encoding="bfloat16",
            device_specs=[DeviceSpec.accelerator()],
        ),
        speculative=SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=2,
        ),
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

    prompt = """Extract the person's name and age from: 'David Smith is 35 years old.'"""

    request_id = RequestID("eagle_structured")
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

    # Verify context has json_schema set (required for structured output)
    assert context.json_schema is not None

    pipeline = pipeline_factory()
    # The pipeline should support spec decode + structured output
    assert isinstance(pipeline, OverlapTextGenerationPipeline)

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

        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)
            if response[request_id].is_done:
                break

    # Flush any remaining outputs
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
    assert "name" in result, f"Missing 'name' in response: {response_content}"
    assert "age" in result, f"Missing 'age' in response: {response_content}"
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)


@pytest.mark.timeout(600)  # 10 minutes for model download + compile
def test_eagle_structured_output_heterogeneous_batch_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test mixed batch with structured and non-structured requests under EAGLE.

    Verifies that when a batch contains both:
    - Requests with json_schema (structured output)
    - Requests without json_schema (free-form)

    Each request is handled correctly during speculative decoding:
    - Structured requests use grammar-constrained bitmasks during verification
    - Free-form requests use unconstrained (all-True) bitmasks
    """
    pipeline_config = PipelineArgs(
        model_path="meta-llama/Llama-3.2-3B-Instruct",
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=2048,
        draft_model=MAXModelConfig(
            model_path="atomicapple0/EAGLE-Llama-3.2-3B-Instruct-bf16",
            quantization_encoding="bfloat16",
            device_specs=[DeviceSpec.accelerator()],
        ),
        speculative=SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=2,
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        runtime=PipelineRuntimeConfig(
            max_batch_size=2,
            enable_overlap_scheduler=True,
        ),
    )

    retrieved = pipeline_registry.retrieve_factory(
        PipelineConfig.from_args(pipeline_config)
    )
    tokenizer = retrieved.tokenizer
    pipeline_factory = retrieved.factory
    assert isinstance(tokenizer, TextTokenizer)

    # Request 1: Structured output with JSON schema
    structured_request_id = RequestID("eagle_structured_batch")
    structured_request = TextGenerationRequest(
        model_name=pipeline_config.model_path,
        request_id=structured_request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="Extract: 'Emma Wilson is 28 years old.'",
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
            },
            grammar_enforced=True,
            tools_forced=False,
        ),
    )

    # Request 2: Non-structured output (no json_schema)
    freeform_request_id = RequestID("eagle_freeform_batch")
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
    )

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
    assert isinstance(pipeline, OverlapTextGenerationPipeline)

    kv_manager = pipeline.kv_manager
    kv_manager.claim(structured_ctx)
    kv_manager.claim(freeform_ctx)

    structured_tokens: list[int] = []
    freeform_tokens: list[int] = []
    max_iterations = 60

    # Keep both contexts in batch throughout execution.
    # Speculative decoding maintains state that expects consistent batch sizes.
    contexts: list[TextContext] = [structured_ctx, freeform_ctx]
    structured_done = False
    freeform_done = False

    for _ in range(max_iterations):
        if structured_done and freeform_done:
            break

        # Allocate KV cache for all contexts in the batch.
        # Even done contexts need consistent allocation for spec decode.
        for ctx in contexts:
            kv_manager.alloc(ctx)

        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[contexts]
        )
        response = pipeline.execute(inputs)

        for ctx in contexts:
            if ctx.request_id in response:
                resp = response[ctx.request_id]
                if ctx.request_id == structured_request_id:
                    structured_tokens.extend(resp.tokens)
                    if resp.is_done:
                        structured_done = True
                else:
                    freeform_tokens.extend(resp.tokens)
                    if resp.is_done:
                        freeform_done = True

    # Flush remaining outputs
    empty_inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[]]
    )
    response = pipeline.execute(empty_inputs)
    if structured_request_id in response:
        structured_tokens.extend(response[structured_request_id].tokens)
    if freeform_request_id in response:
        freeform_tokens.extend(response[freeform_request_id].tokens)

    # Verify structured output produced valid JSON
    structured_response = asyncio.run(
        tokenizer.decode(np.array(structured_tokens), skip_special_tokens=True)
    )
    result = json.loads(structured_response)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)

    # Verify free-form output was generated (not blocked by bitmask)
    assert len(freeform_tokens) > 0, "Free-form request should generate tokens"
    freeform_response = asyncio.run(
        tokenizer.decode(np.array(freeform_tokens), skip_special_tokens=True)
    )
    assert len(freeform_response.strip()) > 0, (
        "Free-form request should produce non-empty output"
    )


@pytest.mark.timeout(600)  # 10 minutes for model download + compile
def test_eagle_structured_output_no_first_decode_stall_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Regression test for MXSERV-189: first constrained decode must not stall.

    Without the warmup fix, the first async bitmask kickoff callback is
    stream-ordered behind the captured graph's same-stream wait_host_value at
    the prefill->decode boundary, producing a self-deadlock that idles the GPU
    ~7-9 seconds per server lifetime. This test detects that stall by measuring
    the wall time of each execute() call and asserting that no single call
    exceeds a conservative multiple (30x) of the median steady-state decode
    time across iterations 3..N.

    The multiplier is generous: the observed stall is ~7-9s vs ~0.06-0.18s
    per iteration (50-150x), while legitimate variance from graph-capture
    jitter or KV eviction is at most 2-3x. 30x reliably separates the two.
    """
    pipeline_config = PipelineArgs(
        model_path="meta-llama/Llama-3.2-3B-Instruct",
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=2048,
        draft_model=MAXModelConfig(
            model_path="atomicapple0/EAGLE-Llama-3.2-3B-Instruct-bf16",
            quantization_encoding="bfloat16",
            device_specs=[DeviceSpec.accelerator()],
        ),
        speculative=SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=2,
        ),
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

    prompt = """Extract the person's name and age from: 'Maria Garcia is 42 years old.'"""

    request_id = RequestID("eagle_stall_regression")
    request = TextGenerationRequest(
        model_name=pipeline_config.model_path,
        request_id=request_id,
        messages=[TextGenerationRequestMessage(role="user", content=prompt)],
        sampling_params=SamplingParams(max_new_tokens=60, top_k=1),
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
    assert context.json_schema is not None

    pipeline = pipeline_factory()
    assert isinstance(pipeline, OverlapTextGenerationPipeline)

    kv_manager = pipeline.kv_manager
    kv_manager.claim(context)

    # Collect per-call wall times. The overlap pipeline is pipelined:
    #   call 0: prefill (drives model forward, no output yet)
    #   call 1: first decode step (returns prefill output, drives 1st decode)
    #   call 2+: steady-state decode
    # The MXSERV-189 stall hits at call 1 (first decode): the bitmask kickoff
    # callback enqueued by call 1 is stream-ordered behind the same-stream
    # wait_host_value in the captured decode graph, deadlocking until an
    # external poke frees it (~7-9s).
    call_times: list[float] = []
    tokens: list[int] = []
    max_iterations = 60

    for _ in range(max_iterations):
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]]
        )
        kv_manager.alloc(context)

        t0 = time.monotonic()
        response = pipeline.execute(inputs)
        call_times.append(time.monotonic() - t0)

        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)
            if response[request_id].is_done:
                break

    # Flush remaining outputs
    empty_inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[]]
    )
    response = pipeline.execute(empty_inputs)
    if request_id in response:
        for token in response[request_id].tokens:
            tokens.append(token)

    # Need at least 4 timed calls: prefill (0), first decode (1), and two
    # steady-state iterations (2, 3) to compute a meaningful median.
    assert len(call_times) >= 4, (
        f"Expected at least 4 execute() calls, got {len(call_times)}"
    )

    # Median of calls 3..N as the steady-state reference (skip prefill and the
    # first two decode iterations to let the pipeline reach steady throughput).
    steady_times = call_times[3:]
    median_steady = float(np.median(steady_times))

    # The first decode call (index 1) is where the MXSERV-189 stall manifests.
    # With the fix it should be within 30x of steady-state; without the fix it
    # is 50-150x slower.
    first_decode_time = call_times[1]
    threshold = 30.0 * median_steady

    assert first_decode_time < threshold, (
        f"First decode execute() took {first_decode_time:.3f}s, "
        f"which is {first_decode_time / median_steady:.1f}x the median "
        f"steady-state time ({median_steady:.4f}s). "
        f"Expected < 30x ({threshold:.3f}s). "
        "This suggests the MXSERV-189 bitmask kickoff stall has regressed."
    )

    # Verify the output is still valid JSON (functional correctness).
    response_content = asyncio.run(
        tokenizer.decode(np.array(tokens), skip_special_tokens=True)
    )
    # raw_decode extracts the first JSON object and ignores trailing text;
    # the Llama 3 chat template can append a non-special "assistant" role token
    # after the closing brace that would cause json.loads to raise.
    result, _ = json.JSONDecoder().raw_decode(response_content)
    assert "name" in result, f"Missing 'name' in response: {response_content}"
    assert "age" in result, f"Missing 'age' in response: {response_content}"
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)
