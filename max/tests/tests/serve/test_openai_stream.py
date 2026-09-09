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

import asyncio
import json
import sys

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.pipelines.context import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.modeling.types import PipelineTask
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)

# FIXME: SERVSYS-1275 — serve integration tests run for many minutes on
# contended macOS CI workers, pushing the macOS job past its 2h cap. Skip on
# macOS until the serve-test slowness is addressed.
pytestmark = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="SERVSYS-1275: too slow on macOS CI; exceeds the 2h job timeout",
)

MAX_CHUNK_TO_READ_BYTES: int = 1024 * 10


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to always return TextContext."""

    def _mock_retrieve_context_type(
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> type[TextContext]:
        return TextContext

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        _mock_retrieve_context_type,
    )


@pytest.fixture(autouse=True)
def reset_sse_appstatus() -> None:
    """
    Fixture that resets the appstatus event in the sse_starlette app.

    Should be used on any test that uses sse_starlette to stream events.
    """
    # See https://github.com/sysid/sse-starlette/issues/59
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def decode_and_strip(text: bytes, prefix: str | None):  # noqa: ANN201
    decoded = text.decode("utf-8").strip()
    if prefix:
        return remove_prefix(decoded, prefix)
    return decoded


@pytest.fixture(scope="function")
def stream_app(mock_pipeline_config: PipelineConfig) -> FastAPI:
    settings = Settings(api_types=[APIType.OPENAI], use_heartbeat=False)
    serving_settings = ServingTokenGeneratorSettings(
        model_factory=EchoTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=EchoPipelineTokenizer(),
    )
    fast_app = fastapi_app(settings, serving_settings)
    return fast_app


@pytest.mark.asyncio
@pytest.mark.parametrize("num_tasks", [16])
async def test_openai_chat_completion_streamed(
    stream_app: FastAPI,
    num_tasks: int,
) -> None:
    async def stream_request(client: TestClient, idx: int) -> None:
        request_content = f"Who was the {idx} president?"
        response_text = ""
        response = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name="echo", content=request_content, stream=True
            ),
            stream=True,
        )
        async for decoded_response in response.iter_content(
            MAX_CHUNK_TO_READ_BYTES
        ):
            decoded_response = decode_and_strip(decoded_response, "data: ")
            if decoded_response.startswith("[DONE]"):
                break
            if decoded_response.startswith("ping -"):
                continue

            json_response = json.loads(decoded_response)
            response_content = json_response["choices"][0]["delta"]["content"]
            response_text += response_content
        assert response_text == (request_content)

    async with TestClient(stream_app, timeout=20.0) as client:
        tasks = []
        for i in range(num_tasks):
            tasks.append(asyncio.create_task(stream_request(client, i)))
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_openai_chat_completion_stream_with_usage(
    stream_app: FastAPI,
) -> None:
    """Test that stream_options.include_usage returns a final chunk with usage stats."""
    async with TestClient(stream_app, timeout=20.0) as client:
        request_content = "Hello, world!"
        response_text = ""
        has_usage_chunk = False
        usage_chunk_data = None
        regular_chunks_have_null_usage = True

        request_json = simple_openai_request(
            model_name="echo", content=request_content, stream=True
        )
        request_json["stream_options"] = {"include_usage": True}

        response = await client.post(
            "/v1/chat/completions",
            json=request_json,
            stream=True,
        )

        chunks = []
        async for decoded_response in response.iter_content(
            MAX_CHUNK_TO_READ_BYTES
        ):
            decoded_response = decode_and_strip(decoded_response, "data: ")
            if decoded_response.startswith("[DONE]"):
                break
            if decoded_response.startswith("ping -"):
                continue

            json_response = json.loads(decoded_response)
            chunks.append(json_response)

            if (
                "choices" in json_response
                and len(json_response["choices"]) == 0
            ):
                # Final usage chunk
                if json_response.get("usage"):
                    has_usage_chunk = True
                    usage_chunk_data = json_response["usage"]
            else:
                # Regular content chunk
                if json_response.get("choices"):
                    response_content = json_response["choices"][0]["delta"][
                        "content"
                    ]
                    response_text += response_content

                    if (
                        "usage" in json_response
                        and json_response["usage"] is not None
                    ):
                        regular_chunks_have_null_usage = False

        assert response_text == request_content
        assert has_usage_chunk, "Expected a final chunk with usage statistics"
        assert usage_chunk_data is not None
        assert "prompt_tokens" in usage_chunk_data
        assert "completion_tokens" in usage_chunk_data
        assert "total_tokens" in usage_chunk_data
        assert regular_chunks_have_null_usage, (
            "Regular chunks should have null usage when include_usage is True"
        )
