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
"""Tests for the pipeline-side weight-loading env helpers."""

from __future__ import annotations

import pytest
from max.pipelines.weights.weight_loading import (
    AUTO_CAST_ENV_VAR,
    auto_cast_weights_from_env,
)


def test_auto_cast_weights_from_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AUTO_CAST_ENV_VAR, raising=False)
    assert auto_cast_weights_from_env() is True


@pytest.mark.parametrize(
    "value", ["true", "TRUE", "1", "yes", "on", "  True  "]
)
def test_auto_cast_weights_from_env_truthy(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(AUTO_CAST_ENV_VAR, value)
    assert auto_cast_weights_from_env() is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
def test_auto_cast_weights_from_env_falsy(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(AUTO_CAST_ENV_VAR, value)
    assert auto_cast_weights_from_env() is False


def test_auto_cast_weights_from_env_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUTO_CAST_ENV_VAR, "maybe")
    with pytest.raises(ValueError, match=AUTO_CAST_ENV_VAR):
        auto_cast_weights_from_env()
