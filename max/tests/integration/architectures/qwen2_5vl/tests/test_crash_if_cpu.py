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
"""Checks that Qwen2.5VL correctly crashes if the first device is CPU."""

from typing import Any
from unittest.mock import Mock

import pytest
from max.driver import CPU
from max.pipelines.architectures.qwen2_5vl.model import Qwen2_5VLModel
from max.pipelines.lib import PipelineModel


def noop_init(_self: Any, *_args: Any, **_kwargs: Any) -> None:
    pass


def test_crash_if_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch PipelineModel.__init__ to be a no-op
    monkeypatch.setattr(PipelineModel, "__init__", noop_init)
    with pytest.raises(
        ValueError, match=r"Qwen2\.5VL currently only supports GPU devices"
    ):
        Qwen2_5VLModel(
            pipeline_config=Mock(),
            session=Mock(),
            devices=[CPU()],
            kv_cache_config=Mock(),
            weights=Mock(),
            memory_plan=Mock(),
        )
