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
"""Tests for the worker-side helper that retrieves the EPLB accumulator."""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest
from max.pipelines.lib.eplb_stats import (
    EplbStatsAccumulator,
    EplbStatsMetadata,
)
from max.pipelines.modeling.types.pipeline import Pipeline
from max.serve.pipelines.model_worker import _get_eplb_stats_accumulator


def _md() -> EplbStatsMetadata:
    return EplbStatsMetadata(
        num_layers=2,
        num_moe_layers=2,
        moe_layer_indices=(0, 1),
        num_logical_experts=4,
        num_experts_per_token=2,
    )


class _MockPipelineModelWithAccumulator:
    def __init__(self, accumulator: EplbStatsAccumulator) -> None:
        self._eplb_stats_accumulator = accumulator


class _MockPipelineModelWithoutAccumulator:
    pass


class _MockPipelineWrapping:
    def __init__(self, pipeline_model: object) -> None:
        self._pipeline_model = pipeline_model


def test_disabled_short_circuits() -> None:
    pipeline = cast(Pipeline[Any, Any], _MockPipelineWrapping(object()))
    assert _get_eplb_stats_accumulator(pipeline, enabled=False) is None


def test_enabled_with_accumulator_returns_it() -> None:
    accumulator = EplbStatsAccumulator(_md(), devices=[])
    pipeline_model = _MockPipelineModelWithAccumulator(accumulator)
    pipeline = cast(Pipeline[Any, Any], _MockPipelineWrapping(pipeline_model))
    assert _get_eplb_stats_accumulator(pipeline, enabled=True) is accumulator


def test_enabled_without_accumulator_warns_and_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = cast(
        Pipeline[Any, Any],
        _MockPipelineWrapping(_MockPipelineModelWithoutAccumulator()),
    )
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        result = _get_eplb_stats_accumulator(pipeline, enabled=True)
    assert result is None
    assert "does not expose an EplbStatsAccumulator" in caplog.text
