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
"""Serve smoke test for SmolLM2 (Llama3) LoRA on the ModuleV3 pipeline path."""

import pytest
from max.pipelines import TextGenerationPipeline
from max.pipelines.context import SamplingParams
from max.pipelines.context.context import TextContext
from max.pipelines.lora import LoRAManagerV3
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
)
from test_common.lora_utils import (
    REPO_ID,
    create_pipeline_with_lora,
    create_test_lora_adapter,
    create_tokenizer,
)


def generate_tokens_from_contexts(
    pipeline: TextGenerationPipeline,  # type: ignore[type-arg]
    contexts: dict[RequestID, TextContext],
) -> dict[RequestID, list[int]]:
    """Generate tokens from multiple contexts using the same pipeline.

    Args:
        pipeline: The text generation pipeline to use
        contexts: Dictionary mapping request_id to TextContext

    Returns:
        Dictionary mapping request_id to list of generated tokens
    """
    all_tokens: dict[RequestID, list[int]] = {req_id: [] for req_id in contexts}
    active_contexts = contexts.copy()
    kv_manager = pipeline.kv_manager
    for context in active_contexts.values():
        kv_manager.claim(context)

    while active_contexts:
        for context in active_contexts.values():
            kv_manager.alloc(context)
        response = pipeline.execute(
            TextGenerationInputs(batches=[list(active_contexts.values())])
        )
        for req_id, resp in response.items():
            all_tokens[req_id].extend(resp.tokens)
            if resp.is_done:
                del active_contexts[req_id]

    return all_tokens


@pytest.mark.asyncio
async def test_smollm2_lora_modulev3_serves() -> None:
    """Serve SmolLM2 with a LoRA adapter on the ModuleV3 path (base vs adapter).

    Exercises the ModuleV3 adapters-as-inputs serving path end to end: the
    ``LoRAManagerV3`` loads a synthetic adapter, the graph compiles with the
    SGMV routing inputs, and both a base and an adapter request generate tokens.
    The synthetic adapter perturbs every attention projection, so its greedy
    output must diverge from the base's.
    """
    lora_path = create_test_lora_adapter(seed=0)
    pipeline = create_pipeline_with_lora([lora_path], prefer_module_v3=True)
    tokenizer = create_tokenizer()

    # Confirm the request actually routed through ModuleV3, not a V2 fallback.
    manager = pipeline._pipeline_model._lora_manager
    assert isinstance(manager, LoRAManagerV3)
    assert manager.is_lora(lora_path)

    # pipeline.execute() bypasses the serve scheduler, which normally activates a
    # request's adapter; activate here or the routing selects no adapter and the
    # LoRA delta lands on zero rows (a silent no-op).
    manager.activate_adapter(lora_path)

    prompt = "The future of AI is"
    sampling_params = SamplingParams(
        max_new_tokens=20,
        temperature=0.0,
        top_k=1,
    )

    base_context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id=RequestID("base"),
            prompt=prompt,
            model_name=REPO_ID,
            sampling_params=sampling_params,
        )
    )
    lora_context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id=RequestID("lora"),
            prompt=prompt,
            model_name=lora_path,
            sampling_params=sampling_params,
        )
    )

    base_tokens = generate_tokens_from_contexts(
        pipeline, {base_context.request_id: base_context}
    )[base_context.request_id]
    lora_tokens = generate_tokens_from_contexts(
        pipeline, {lora_context.request_id: lora_context}
    )[lora_context.request_id]

    assert len(base_tokens) > 0
    assert len(lora_tokens) > 0
    assert base_tokens != lora_tokens

    pipeline.release(base_context.request_id)
    pipeline.release(lora_context.request_id)
