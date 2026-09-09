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
"""Shared fixtures for the serve-entrypoint tests (model worker + graceful
shutdown)."""

from __future__ import annotations

import pytest
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    MAXModelConfig,
    ModelManifest,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.modeling.types import PipelineTask


@pytest.fixture
def mock_pipeline_config() -> PipelineConfig:
    runtime = PipelineRuntimeConfig.model_construct(
        max_batch_size=1,
    )

    model_config = MAXModelConfig.model_construct(served_model_name="echo")
    return PipelineConfig.model_construct(
        runtime=runtime,
        models=ModelManifest({"main": model_config}),
    )


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to always return TextContext.

    The tests in this package use simple mock pipeline configs that do not
    correspond to a registered architecture. The default implementation of
    `retrieve_context_type` would raise in this case, but for these tests we
    only care that a valid context type is provided, not which one.
    """

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
