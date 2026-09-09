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
"""Test serving a Llama 3 model on the CPU."""

import asyncio
import json
from typing import Any

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineArgs
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    CreateCompletionResponse,
)
from test_common.test_data import DEFAULT_PROMPTS

MAX_READ_SIZE = 10 * 1024

MODEL_NAME = "modularai/SmolLM-135M-Instruct-FP32"

pipeline_config = PipelineArgs(
    model_path=MODEL_NAME,
    device_specs=[DeviceSpec.cpu()],
    quantization_encoding="float32",
    kv_cache=KVCacheConfig(),
    max_length=512,
    runtime=PipelineRuntimeConfig(max_batch_size=16),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [pipeline_config],
    indirect=True,
)
async def test_tinyllama_serving_cpu(app: FastAPI) -> None:
    """Test chat completions, completions, and streaming completions."""

    async with TestClient(app, timeout=720.0) as client:
        # ---- Non-streaming chat completion ----
        raw_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "tell me a joke"}],
                "stream": False,
                "max_tokens": 3,
            },
        )
        response = CreateChatCompletionResponse.model_validate(
            raw_response.json()
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "length"

        # ---- Non-streaming completions with different prompt formats ----
        prompt_num = [
            ("Hello world", 1),
            (["Hello world", "hello there"], 2),
            ([1, 2, 3], 1),
            ([[1, 2, 3], [4, 5, 6]], 2),
        ]
        for prompt, n_prompts in prompt_num:
            raw_response = await client.post(
                "/v1/completions",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": 3,
                },
            )
            response2 = CreateCompletionResponse.model_validate(
                raw_response.json()
            )
            assert len(response2.choices) == n_prompts
            assert response2.choices[0].finish_reason in ["length", "stop"]

        # ---- Streaming completions ----
        def openai_completion_request(content: str) -> dict[str, Any]:
            return {
                "model": MODEL_NAME,
                "prompt": content,
                "temperature": 0.7,
                "max_tokens": 3,
            }

        async def stream_completion(client: TestClient, msg: str) -> str:
            r = await client.post(
                "/v1/completions",
                json=openai_completion_request(msg) | {"stream": True},
                stream=True,
            )
            response_text = ""
            async for chunk in r.iter_content(MAX_READ_SIZE):
                chunk_str = chunk.decode("utf-8").strip()
                if chunk_str.startswith("data: [DONE]"):
                    break
                try:
                    data = json.loads(chunk_str[len("data: ") :])
                    content = data["choices"][0]["text"]
                    response_text += content
                except Exception:
                    pass
            return response_text

        tasks = []
        for prompt in DEFAULT_PROMPTS[1:]:
            tasks.append(asyncio.create_task(stream_completion(client, prompt)))
        for task in tasks:
            await task
