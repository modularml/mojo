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

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar
from unittest.mock import patch

from max.pipelines.lib import MemoryEstimator
from max.pipelines.lib.memory_estimation import MemoryPlan
from typing_extensions import ParamSpec

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _fake_plan_from_sizes(
    pipeline_config: Any,
    model_config: Any,
    *args: Any,
    **kwargs: Any,
) -> MemoryPlan:
    """Stands in for ``plan_from_sizes`` without touching devices.

    Mirrors the config's own values so readers that consume the plan see
    the construction-resolved configuration.
    """
    return MemoryPlan(
        planned_max_batch_size=1,
        footprint=0,
        planned_max_length=model_config.max_length,
        device_specs=tuple(model_config.device_specs),
        planned_max_batch_total_tokens=pipeline_config.runtime.max_batch_total_tokens,
    )


def mock_plan_from_sizes(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Mock the MemoryEstimator.plan_from_sizes method.

    The KV-capacity clamp reads real device stats, but it runs inside
    ``plan_from_sizes``, so stubbing that covers it too. This
    decorator works with both sync and async functions.
    """
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with patch.object(
                MemoryEstimator,
                "plan_from_sizes",
                side_effect=_fake_plan_from_sizes,
            ):
                return await func(*args, **kwargs)

        return async_wrapper  # type: ignore
    else:

        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with patch.object(
                MemoryEstimator,
                "plan_from_sizes",
                side_effect=_fake_plan_from_sizes,
            ):
                return func(*args, **kwargs)

        return wrapper
