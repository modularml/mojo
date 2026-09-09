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

import json
from collections.abc import Callable

import httpx
import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.pipelines.context import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.modeling.types import PipelineTask

request = {
    "model": "echo",
    "messages": [
        {
            "role": "user",
            "content": "I like to ride my bicycle, I like to ride my bike" * 10,
        }
    ],
    "stream": False,
    "stop": ["doesn't show up", "bicycle, I like", "bicycle"],
}


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


@pytest.mark.asyncio
async def test_stop_sequence(echo_app: FastAPI) -> None:
    async with TestClient(echo_app, timeout=720.0) as client:
        # Test with streaming set to False
        raw_response = await client.post("/v1/chat/completions", json=request)

        result = raw_response.json()

        # Expected continuation stops at the first match ("ekil I ,elcycib")
        # and does not include the stop sequence
        expected = "I like to ride my "

        assert expected == result["choices"][0]["message"]["content"]


@pytest.mark.skip(reason="Flaky test -- MAXSERV-947")
@pytest.mark.asyncio
async def test_stop_sequence_streaming(
    echo_app: FastAPI,
    reset_sse_starlette_appstatus_event: Callable[[], None],
) -> None:
    async with TestClient(echo_app, timeout=720.0) as client:
        # Test with streaming set to False
        raw_response = await client.post(
            "/v1/chat/completions", json=request | {"stream": True}, stream=True
        )

        response_text = await _stream_response(raw_response)

        # In the streaming case, guarantees are softer. We do best effort
        # to end early but it won't be deterministic and we can't
        # clean up the response in the same way we can in the sync case.
        # The full requested tokens was 512, so 128 is a meaningful cutoff
        # (also keeping in mind additional padding for noisy neighbors in CI)
        expected = "to ride my bike"
        assert len(response_text) - len(expected) < 128, (
            "Got too many extra characters after stop sequence"
        )


async def _stream_response(raw_response: httpx.Response) -> str:
    response_text = ""
    async for response_bytes in raw_response.aiter_bytes(1024 * 20):
        response = response_bytes.decode("utf-8").strip()
        if response.startswith("data: [DONE]"):
            break
        try:
            data = json.loads(response[len("data: ") :])
            content = data["choices"][0]["delta"]["content"]
            response_text += content
        except Exception as e:
            # Just suppress the exception as it might be a ping message.
            print(f"Exception {e} at '{response}'")

    return response_text
