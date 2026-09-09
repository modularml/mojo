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
"""End-to-end test for ``MAXModelWorker`` against a real small model.

Deploys the worker on the local runtime, which spawns a ``max.serve``
model-worker subprocess for ``modularai/SmolLM-135M-Instruct-FP32`` on CPU, and
streams generated token ids through the ``decode`` worker method. This
exercises the full path: ``PipelineConfig`` build, ``retrieve_factory``
resolution, subprocess spawn, ZMQ proxy, and per-request streaming.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import DeviceSpec
from max.experimental.cascade import (
    ChatMessage,
    GenAIRequest,
    GenAITextChunk,
    LocalRuntime,
    TextGenOptions,
)
from max.experimental.cascade.core.pipeline_method import _pipeline_method_scope
from max.experimental.cascade.pipelines.all_pipelines import build_pipeline
from max.experimental.cascade.pipelines.common_textgen import (
    CommonTextGenPipeline,
)
from max.pipelines.architectures import register_all_models
from max.pipelines.lib import PipelineArgs, generate_local_model_path
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig

REPO_ID = "modularai/SmolLM-135M-Instruct-FP32"


def _model_path() -> str:
    """Resolve a cached local path for the test model, else the repo id."""
    try:
        return generate_local_model_path(REPO_ID)
    except FileNotFoundError:
        return REPO_ID


async def _text_pipeline(model_path: str) -> CommonTextGenPipeline:
    """Build the cascade text pipeline for the test model (CPU, float32).

    Goes through ``build_pipeline``, which constructs the config from the raw
    args and builds/binds the model factory exactly as the serve entrypoint
    does. Tests reuse this single path -- including the ones that exercise
    ``MAXModelWorker.decode`` directly, via the pipeline's already-bound
    ``model`` worker -- rather than re-implementing construction/binding. A
    fresh config per call gives each test its own instance.
    """
    # `build_pipeline` resolves the arch by name, so the registry has to be
    # populated first. Importing `max.pipelines.lib` alone does not do it.
    register_all_models()
    args = PipelineArgs(
        model_path=model_path,
        device_specs=[DeviceSpec.cpu()],
        max_length=512,
        quantization_encoding="float32",
        runtime=PipelineRuntimeConfig(
            max_batch_size=8,
            enable_chunked_prefill=True,
            enable_in_flight_batching=False,
        ),
    )
    pipeline = await build_pipeline(args)
    assert isinstance(pipeline, CommonTextGenPipeline)
    return pipeline


@pytest.fixture(scope="module")
def model_path() -> str:
    return _model_path()


@pytest.mark.asyncio
async def test_decode_streams_requested_token_count(model_path: str) -> None:
    # A short prompt of valid, low token ids; content is irrelevant since
    # ``ignore_eos`` forces exactly ``num_tokens`` to be generated.
    prompt = np.array([1, 2, 3, 4], dtype=np.int32)
    pipeline = await _text_pipeline(model_path)

    async with LocalRuntime() as rt, _pipeline_method_scope():
        proxy = await rt.deploy(pipeline.model)

        # One deploy (one model-worker subprocess), several decode requests.
        for num_tokens in (4, 8):
            req = TextGenOptions(num_tokens=num_tokens, ignore_eos=True)
            chunks = [chunk async for chunk in await proxy.decode(req, prompt)]
            generated = (
                np.concatenate(chunks)
                if chunks
                else np.array([], dtype=np.int32)
            )
            assert generated.dtype == np.int32
            assert len(generated) == num_tokens


@pytest.mark.asyncio
async def test_decode_stops_on_eos(model_path: str) -> None:
    prompt = np.array([1, 2, 3, 4], dtype=np.int32)
    pipeline = await _text_pipeline(model_path)

    async with LocalRuntime() as rt, _pipeline_method_scope():
        proxy = await rt.deploy(pipeline.model)
        req = TextGenOptions(num_tokens=16, ignore_eos=False)
        chunks = [chunk async for chunk in await proxy.decode(req, prompt)]
        generated = (
            np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)
        )
    # With EOS honored, generation may stop early but never exceeds the cap.
    assert generated.dtype == np.int32
    assert 0 <= len(generated) <= 16


@pytest.mark.asyncio
async def test_greedy_decode_answers_capital_of_france(model_path: str) -> None:
    # A true end-to-end LLM check that exercises the full cascade pipeline: a
    # ``MAXTokenizer`` worker applies the chat template, the ``MAXModelWorker``
    # greedily decodes (temperature 0), and the tokenizer worker decodes the
    # generated ids back to text. Confirm the model answers "Paris".
    messages = [ChatMessage.text("user", "What is the capital of France?")]
    pipeline = await _text_pipeline(model_path)

    async with LocalRuntime() as rt, _pipeline_method_scope():
        tokenizer = await rt.deploy(pipeline.tokenizer)
        model = await rt.deploy(pipeline.model)

        prompt = await (await tokenizer.encode(messages))
        req = TextGenOptions(num_tokens=20, temperature=0.0)
        chunks = [chunk async for chunk in await model.decode(req, prompt)]
        generated = (
            np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)
        )
        answer = await (await tokenizer.decode(generated, True))

    assert "paris" in answer.lower(), f"unexpected answer: {answer!r}"


@pytest.mark.asyncio
async def test_pipeline_answers_capital_of_france(model_path: str) -> None:
    # Same factual check, but routed through the cascade
    # ``CommonTextGenPipeline`` that ``build_pipeline`` selects from the model
    # path, so tokenization and detokenization run as pipeline workers rather
    # than being driven by the test.
    pipeline = await _text_pipeline(model_path)
    messages = [ChatMessage.text("user", "What is the capital of France?")]

    async with LocalRuntime() as rt:
        await pipeline.deploy(rt)

        req = GenAIRequest(
            messages=messages,
            text=TextGenOptions(num_tokens=20, temperature=0.0),
        )
        answer = "".join(
            [
                chunk.text
                async for chunk in pipeline.generate(req)
                if isinstance(chunk, GenAITextChunk)
            ]
        )

    assert "paris" in answer.lower(), f"unexpected answer: {answer!r}"
