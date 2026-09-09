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

import asyncio
import functools
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import pytest
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    ModelManifest,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


@pytest.fixture(scope="session")
def fixture_testdatadirectory() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("MAX_SERVE_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture(scope="session")
def fixture_tokenizer(
    fixture_testdatadirectory: Path,
) -> PreTrainedTokenizerFast | PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(fixture_testdatadirectory)
    return tokenizer


@pytest.fixture
def enable_prefix_caching(request: pytest.FixtureRequest) -> bool:
    """Fixture for a whether prefix caching is enabled
    This is bound indirectly - hence the request.param pattern.
    See https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    # defaults to False if not specified
    return request.param if hasattr(request, "param") else False


@pytest.fixture
def runtime_overrides(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Runtime config fields for ``mock_pipeline_config``'s construction.
    This is bound indirectly - hence the request.param pattern.
    See https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    return request.param if hasattr(request, "param") else {}


@pytest.fixture
def mock_pipeline_config(
    enable_prefix_caching: bool, runtime_overrides: dict[str, Any]
) -> PipelineConfig:
    runtime = PipelineRuntimeConfig.model_construct(
        max_batch_size=1,
        **runtime_overrides,
    )

    kv_cache_config = KVCacheConfig.model_construct(
        enable_prefix_caching=enable_prefix_caching,
    )

    model_config = MAXModelConfig.model_construct(
        served_model_name="echo",
        kv_cache=kv_cache_config,
    )
    model_config._huggingface_config = PretrainedConfig()

    return PipelineConfig.model_construct(
        runtime=runtime,
        models=ModelManifest({"main": model_config}),
    )


# simple decorator to make hung test cases fail faster than the bazel 300s timeout
def async_timeout(
    timeout: float,
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, _R]:
        @pytest.mark.asyncio
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return await asyncio.wait_for(func(*args, **kwargs), timeout)

        return wrapper

    return decorator
