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
"""Test serving a Llama 3 model on the GPU."""

import asyncio
import json
from typing import Any

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineArgs
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    CreateCompletionResponse,
)
from test_common.test_data import DEFAULT_PROMPTS

MAX_READ_SIZE = 10 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineArgs(
            model_path="HuggingFaceTB/SmolLM2-135M",
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="bfloat16",
            kv_cache=KVCacheConfig(),
            max_length=512,
            runtime=PipelineRuntimeConfig(max_batch_size=16),
        )
    ],
    indirect=True,
)
async def test_smollm_serve_gpu(app: FastAPI) -> None:
    # Arbitrary - just demonstrate we can submit multiple async
    # requests and collect the results later
    N_REQUESTS = 3

    async with TestClient(app, timeout=90.0) as client:
        tasks = [
            client.post(
                "/v1/chat/completions",
                json=simple_openai_request(
                    model_name="HuggingFaceTB/SmolLM2-135M"
                ),
            )
            for _ in range(N_REQUESTS)
        ]

        responses = await asyncio.gather(*tasks)

        for raw_response in responses:
            print(raw_response)

            # This is not a streamed completion - There is no [DONE] at the end.
            response = CreateChatCompletionResponse.model_validate_json(
                raw_response.text
            )
            print(response)

            assert len(response.choices) == 1
            assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineArgs(
            model_path="HuggingFaceTB/SmolLM2-135M",
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="bfloat16",
            kv_cache=KVCacheConfig(),
            max_length=512,
            runtime=PipelineRuntimeConfig(max_batch_size=16),
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompt,expected_choices",
    [
        ("Hello world", 1),
        (["Hello world"], 1),
        (["Hello world", "Why hello"], 2),
        ([1, 2, 3], 1),
        ([[1, 2, 3]], 1),
    ],
)
async def test_smollm_serve_gpu_nonchat_completions(
    app: FastAPI,
    prompt: str | list[str] | list[int] | list[list[int]],
    expected_choices: int,
) -> None:
    async with TestClient(app, timeout=90.0) as client:
        # Completions endpoint instead of chat completions
        raw_response = await client.post(
            "/v1/completions",
            json={"model": "HuggingFaceTB/SmolLM2-135M", "prompt": prompt},
        )
        response = CreateCompletionResponse.model_validate(raw_response.json())

        assert len(response.choices) == expected_choices
        assert response.choices[0].finish_reason == "stop"


@pytest.mark.skip("TODO(ylou): Fix!!")
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineArgs(
            model_path="HuggingFaceTB/SmolLM2-135M",
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="bfloat16",
            kv_cache=KVCacheConfig(),
            max_length=512,
            runtime=PipelineRuntimeConfig(max_batch_size=16),
        )
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_tinyllama_serve_gpu_stream(app: FastAPI) -> None:
    NUM_TASKS = 16

    def openai_completion_request(content: str) -> dict[str, Any]:
        """Create the json request for /v1/completion (not chat)."""
        return {
            "model": "test/tinyllama",
            "prompt": content,
            "temperature": 0.7,
        }

    async def main_stream(client: TestClient, msg: str) -> str:
        r = await client.post(
            "/v1/completions",
            json=openai_completion_request(msg)
            | {
                "stream": True,
            },
            stream=True,
        )
        response_text = ""
        async for response in r.iter_content(MAX_READ_SIZE):
            response = response.decode("utf-8").strip()
            if response.startswith("data: [DONE]"):
                break
            try:
                data = json.loads(response[len("data: ") :])
                content = data["choices"][0]["text"]
                response_text += content
            except Exception as e:
                # Just suppress the exception as it might be a ping message.
                print(f"Exception {e} at '{response}'")
        return response_text

    tasks = []
    resp = []
    async with TestClient(app, timeout=5.0) as client:
        j = 1
        for i in range(NUM_TASKS):
            # we skip the first prompt as it is longer than 512
            if i >= len(DEFAULT_PROMPTS):
                j = 1
            else:
                j += 1

            msg = DEFAULT_PROMPTS[j]
            tasks.append(asyncio.create_task(main_stream(client, msg)))
        for task in tasks:
            resp.append(await task)
