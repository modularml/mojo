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
"""Tests for the construction-time vision cache budget resolution."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from max.pipelines.lib import PipelineRuntimeConfig
from max.pipelines.lib.config.config import (
    _resolve_preprocess_cache_budgets,
    _resolve_vision_cache_utilization,
)

GIB = 1024**3
_CONFIG = "max.pipelines.lib.config.config"


def _stub_arch(
    entry_bytes: int = 1,
    row_spec: tuple[int, Any] | None = (1024, "bfloat16"),
) -> SimpleNamespace:
    """An architecture whose config reports the given vision-cache facts."""

    class _VisionArchConfig:
        @classmethod
        def estimate_vision_cache_entry_bytes(cls, hf: Any) -> int:
            return entry_bytes

        @classmethod
        def get_vision_cache_row_spec(cls, hf: Any) -> tuple[int, Any] | None:
            return row_spec

    return SimpleNamespace(name="stub-arch", config=_VisionArchConfig)


def _model_stub() -> MagicMock:
    """A model config stand-in: the resolvers read only its HF config."""
    model = MagicMock()
    model.huggingface_config = MagicMock()
    return model


def _clamp_budgets(
    image_bytes: int,
    video_bytes: int,
    host_bytes: int | None,
    has_vision_tower: bool = True,
) -> tuple[int, int]:
    """Run the host-memory clamp and return the resulting budgets."""
    runtime = PipelineRuntimeConfig(
        max_vision_preprocess_cache_bytes=image_bytes,
        max_video_preprocess_cache_bytes=video_bytes,
    )
    model = _model_stub()
    arch = _stub_arch(entry_bytes=1 if has_vision_tower else 0)

    with patch(f"{_CONFIG}._host_memory_limit", return_value=host_bytes):
        return _resolve_preprocess_cache_budgets(runtime, model, arch)


def test_preprocess_cache_budgets__left_alone_when_they_fit() -> None:
    """A ceiling within the host fraction is not a ceiling worth lowering."""
    assert _clamp_budgets(2 * GIB, 1 * GIB, host_bytes=64 * GIB) == (
        2 * GIB,
        1 * GIB,
    )


def test_preprocess_cache_budgets__reduced_proportionally_when_too_large() -> (
    None
):
    """An oversized pair shrinks to the cap, keeping the image:video ratio.

    Clamping each budget independently would let one starve the other; scaling
    both preserves whatever split the operator asked for.
    """
    image, video = _clamp_budgets(8 * GIB, 2 * GIB, host_bytes=16 * GIB)

    cap = int(16 * GIB * 0.25)
    assert image + video <= cap
    assert image == 4 * video  # the configured 8:2 split, preserved


def test_preprocess_cache_budgets__untouched_without_a_vision_tower() -> None:
    """A text-only model never builds the caches, so nothing to bound."""
    assert _clamp_budgets(
        8 * GIB, 8 * GIB, host_bytes=1 * GIB, has_vision_tower=False
    ) == (8 * GIB, 8 * GIB)


def test_preprocess_cache_budgets__untouched_when_host_memory_unknown() -> None:
    """An unbounded guess would be worse than the configured ceiling."""
    assert _clamp_budgets(8 * GIB, 8 * GIB, host_bytes=None) == (
        8 * GIB,
        8 * GIB,
    )


def test_preprocess_cache_budgets__disabled_caches_stay_disabled() -> None:
    """Zero means off, and a clamp must never turn a cache back on."""
    assert _clamp_budgets(0, 0, host_bytes=1 * GIB) == (0, 0)


def test_preprocess_cache_budgets__clamp_is_idempotent() -> None:
    """Clamping twice must not shrink the budgets twice.

    Construction runs the clamp once, but a config can be re-constructed from
    already-clamped values (a worker rebuilding from serialized args), so the
    proportional reduction must be a fixed point -- afterwards the sum equals
    the cap, so a second pass returns early. This pins that, because geometric
    shrinking would be silent.
    """
    once = _clamp_budgets(8 * GIB, 2 * GIB, host_bytes=16 * GIB)

    assert _clamp_budgets(once[0], once[1], host_bytes=16 * GIB) == once


def test_vision_cache_utilization__disabled_when_no_row_spec(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An arch config with a per-entry size but no row spec cannot back the cache."""
    runtime = PipelineRuntimeConfig(vision_cache_utilization=0.05)

    with caplog.at_level("WARNING", logger="max.pipelines"):
        resolved = _resolve_vision_cache_utilization(
            runtime, _model_stub(), _stub_arch(entry_bytes=64, row_spec=None)
        )

    assert resolved == 0.0
    assert any(
        "Disabling vision encoder cache" in record.message
        for record in caplog.records
    )


def test_vision_cache_utilization__kept_with_a_row_spec() -> None:
    runtime = PipelineRuntimeConfig(vision_cache_utilization=0.05)

    resolved = _resolve_vision_cache_utilization(
        runtime,
        _model_stub(),
        _stub_arch(entry_bytes=64, row_spec=(1024, "bfloat16")),
    )

    assert resolved == 0.05


def test_vision_cache_utilization__kept_without_a_vision_tower() -> None:
    """Text-only architectures never consult the row spec."""
    runtime = PipelineRuntimeConfig(vision_cache_utilization=0.05)

    resolved = _resolve_vision_cache_utilization(
        runtime, _model_stub(), _stub_arch(entry_bytes=0, row_spec=None)
    )

    assert resolved == 0.05


def test_vision_cache_utilization__zero_skips_the_arch_config() -> None:
    """An explicit 0 is final; the arch config is never consulted."""

    class _BoomArchConfig:
        @classmethod
        def estimate_vision_cache_entry_bytes(cls, hf: Any) -> int:
            raise AssertionError(
                "arch config consulted despite utilization == 0"
            )

        @classmethod
        def get_vision_cache_row_spec(cls, hf: Any) -> tuple[int, Any] | None:
            raise AssertionError(
                "arch config consulted despite utilization == 0"
            )

    runtime = PipelineRuntimeConfig(vision_cache_utilization=0.0)
    arch = SimpleNamespace(name="stub-arch", config=_BoomArchConfig)

    assert (
        _resolve_vision_cache_utilization(runtime, _model_stub(), arch) == 0.0
    )
