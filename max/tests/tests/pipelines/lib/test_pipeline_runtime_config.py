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
"""Tests for PipelineRuntimeConfig."""

import pytest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.config_enums import PipelineRole
from pydantic import ValidationError


def test_emit_reasoning_content_defaults_false() -> None:
    assert PipelineRuntimeConfig().emit_reasoning_content is False


def test_emit_reasoning_content_can_be_enabled() -> None:
    assert (
        PipelineRuntimeConfig(
            emit_reasoning_content=True
        ).emit_reasoning_content
        is True
    )


@pytest.mark.parametrize("utilization", [0, 0.05, 1])
def test_vision_cache_utilization_accepts_in_range_values(
    utilization: float,
) -> None:
    assert (
        PipelineRuntimeConfig(
            vision_cache_utilization=utilization
        ).vision_cache_utilization
        == utilization
    )


@pytest.mark.parametrize(
    ("pipeline_role", "expected"),
    [
        ("prefill_and_decode", False),
        ("prefill_only", True),
        ("decode_only", True),
    ],
)
def test_is_disaggregated(pipeline_role: PipelineRole, expected: bool) -> None:
    runtime = PipelineRuntimeConfig(pipeline_role=pipeline_role)
    assert runtime.is_disaggregated is expected


@pytest.mark.parametrize("utilization", [-0.01, -1, 1.01, 2, 100])
def test_vision_cache_utilization_rejects_out_of_range_values(
    utilization: float,
) -> None:
    """Out-of-range values must fail at config validation with a message
    naming the field, not surface later as a confusing allocation failure
    (a negative or over-100% byte budget) deep in memory estimation."""
    with pytest.raises(ValidationError, match="vision_cache_utilization"):
        PipelineRuntimeConfig(vision_cache_utilization=utilization)
