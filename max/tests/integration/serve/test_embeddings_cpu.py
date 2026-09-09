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

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineArgs
from max.serve.schemas.openai import CreateEmbeddingResponse

MPNET_REPO_ID = "sentence-transformers/all-mpnet-base-v2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineArgs(
            model_path=MPNET_REPO_ID,
            device_specs=[DeviceSpec.cpu()],
            max_length=256,
        )
    ],
    indirect=True,
)
async def test_serve_embeddings(app: FastAPI) -> None:
    async with TestClient(app, timeout=720.0) as client:
        raw_response = await client.post(
            "/v1/embeddings",
            json={
                "input": "Turn this sentence into embeddings",
                "model": MPNET_REPO_ID,
            },
        )

        response = CreateEmbeddingResponse.model_validate(raw_response.json())

        assert response.data[0].embedding is not None
