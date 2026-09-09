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
from fastapi.testclient import TestClient
from max.pipelines.context import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.modeling.types import PipelineTask
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.router import openai_routes


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


@pytest.fixture
def app(mock_pipeline_config: PipelineConfig):  # noqa: ANN201
    settings = Settings(api_types=[APIType.KSERVE], use_heartbeat=False)

    pipeline_settings = ServingTokenGeneratorSettings(
        model_factory=EchoTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=EchoPipelineTokenizer(),
    )
    return fastapi_app(settings, pipeline_settings)


# In Bazel, we throw PackageNotFoundError since we're not running as a proper package.
def test_version_endpoint_exists(app) -> None:  # noqa: ANN001
    with TestClient(app) as client:
        response = client.get("/version")
        assert response.status_code == 200
        assert response.json()
        assert response.json()["version"]


def test_health_endpoint_exists(app) -> None:  # noqa: ANN001
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_prompts() -> None:
    prompts = openai_routes.get_prompts_from_openai_request(
        "Why is the sky blue?"
    )
    assert len(prompts) == 1

    prompts = openai_routes.get_prompts_from_openai_request(
        ["Why is the sky blue?", "what time is it?"]
    )
    assert len(prompts) == 2

    prompts = openai_routes.get_prompts_from_openai_request([[1, 2, 3]])
    assert len(prompts) == 1

    prompts = openai_routes.get_prompts_from_openai_request([1, 2, 3])
    assert len(prompts) == 1

    # prompt item
    prompts = openai_routes.get_prompts_from_openai_request([[1, 2, 3]])
    assert len(prompts) == 1

    prompts = openai_routes.get_prompts_from_openai_request(
        [[1, 2, 3], [4, 5, 6]]
    )
    assert len(prompts) == 2
