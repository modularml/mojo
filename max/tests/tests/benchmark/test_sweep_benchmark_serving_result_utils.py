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
"""Tests for the serving sweep uploader protocol + percentile validation."""

from __future__ import annotations

import pytest
from max.benchmark.sweep_benchmark_serving_result_utils import (
    SUPPORTED_SWEEP_SERVING_PERCENTILES,
    validate_sweep_serving_percentiles,
)


def test_supported_percentiles_frozen_set() -> None:
    assert SUPPORTED_SWEEP_SERVING_PERCENTILES == frozenset((50, 90, 95, 99))


def test_validate_sweep_serving_percentiles_accepts_supported() -> None:
    validate_sweep_serving_percentiles([50, 90, 95, 99])
    validate_sweep_serving_percentiles([50])


def test_validate_sweep_serving_percentiles_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported percentiles: \\[75\\]"):
        validate_sweep_serving_percentiles([50, 75])
