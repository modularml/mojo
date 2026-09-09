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

import logging

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.pipelines.context import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.modeling.types import PipelineTask

logging.basicConfig(
    level=logging.DEBUG,
)

request = {
    "model": "echo",
    "messages": [
        {
            "role": "user",
            "content": "I like to ride my bicycle, I like to ride my bike" * 10,
        },
    ],
    "stream": False,
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
async def test_invocations(echo_app: FastAPI) -> None:
    async with TestClient(echo_app, timeout=720.0) as client:
        raw_response = await client.post(
            "/invocations",
            json=request,
        )

        assert raw_response.status_code == 200
