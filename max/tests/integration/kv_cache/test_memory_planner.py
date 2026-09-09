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

"""Unit tests for MemoryPlanner and PagedMemoryPlanner."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
)
from max.pipelines.kv_cache import (
    ModelConfig,
    ModelConfigWithKVCache,
    PagedMemoryPlanner,
)
from max.pipelines.lib.memory_estimation import MemoryEstimator
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.vision_encoder_cache import VisionCachePlan

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig
    from max.pipelines.lib.config.model_config import MAXModelConfig
    from max.pipelines.lib.registry import SupportedArchitecture

# ---------------------------------------------------------------------------
# Minimal protocol conformers
# ---------------------------------------------------------------------------


class _MinimalConfig:
    """Satisfies ModelConfig (has ``devices``)."""

    @property
    def devices(self) -> list[Device]:
        return []


class _KVConfig(_MinimalConfig):
    """Satisfies ModelConfigWithKVCache (adds ``get_kv_params``)."""

    def get_kv_params(self) -> KVCacheParams:
        return MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=8,
            head_dim=128,
            num_layers=1,
            page_size=128,
            data_parallel_degree=1,
            devices=[DeviceRef.CPU()],
            kvcache_quant_config=KVCacheQuantizationConfig(),
        )


class _BadConfig:
    """Does NOT satisfy ModelConfigWithKVCache (no ``get_kv_params``)."""

    @property
    def devices(self) -> list[Device]:
        return []


# ---------------------------------------------------------------------------
# Protocol isinstance checks
# ---------------------------------------------------------------------------


def test_minimal_config_satisfies_model_config() -> None:
    assert isinstance(_MinimalConfig(), ModelConfig)


def test_kv_config_satisfies_model_config_with_kv_cache() -> None:
    assert isinstance(_KVConfig(), ModelConfigWithKVCache)


def test_bad_config_does_not_satisfy_model_config_with_kv_cache() -> None:
    assert not isinstance(_BadConfig(), ModelConfigWithKVCache)


# ---------------------------------------------------------------------------
# PagedMemoryPlanner
# ---------------------------------------------------------------------------


def test_paged_planner_rejects_non_kv_config() -> None:
    with pytest.raises(TypeError, match="ModelConfigWithKVCache"):
        PagedMemoryPlanner(_BadConfig())


def test_paged_planner_accepts_kv_config() -> None:
    planner = PagedMemoryPlanner(_KVConfig())
    assert planner is not None


def test_paged_planner_estimate_activation_memory_zero_by_default() -> None:
    """Default estimate_activation_memory should return 0."""
    planner = PagedMemoryPlanner(_KVConfig())
    assert planner.estimate_activation_memory(MagicMock(), MagicMock()) == 0


def test_paged_planner_infer_max_batch_size_none_by_default() -> None:
    """Default infer_max_batch_size defers to the framework inference."""
    planner = PagedMemoryPlanner(_KVConfig())
    assert planner.infer_max_batch_size(MagicMock(), [], 0) is None


def test_with_activation_reservation_returns_correct_bytes() -> None:
    """with_activation_reservation should return the configured value."""
    reservation = 15 * 1024**3
    planner_cls = PagedMemoryPlanner.with_activation_reservation(reservation)
    planner = planner_cls(_KVConfig())
    assert (
        planner.estimate_activation_memory(MagicMock(), MagicMock())
        == reservation
    )


def _block_reserve(
    utilization: float,
    row_bytes: int,
    available_memory: int,
    n_devices: int = 1,
) -> tuple[int, VisionCachePlan, "PipelineRuntimeConfig"]:
    """Run _reserve_vision_cache_blocks against a real runtime config."""
    runtime = PipelineRuntimeConfig(vision_cache_utilization=utilization)
    pipeline_config = SimpleNamespace(runtime=runtime)
    total, plan = MemoryEstimator._reserve_vision_cache_blocks(
        cast("PipelineConfig", pipeline_config),
        (row_bytes, DType.uint8),
        available_memory,
        n_devices,
    )
    return total, plan, runtime


def test_reserve_vision_cache_blocks_rounds_to_whole_blocks() -> None:
    available = 1024**3
    total, plan, runtime = _block_reserve(
        utilization=0.001,
        row_bytes=10,
        available_memory=available,
    )
    block_bytes = 128 * 10
    assert total == (int(available * 0.001) // block_bytes) * block_bytes
    assert 0 < total <= available * 0.001
    assert plan.bytes_per_device == total
    assert (plan.hidden_size, plan.dtype) == (10, DType.uint8)
    assert runtime.vision_cache_utilization == 0.001


def test_reserve_vision_cache_blocks_default_auto_sizes() -> None:
    available = 1024**3
    total, plan, _ = _block_reserve(
        utilization=PipelineRuntimeConfig().vision_cache_utilization,
        row_bytes=10,
        available_memory=available,
    )
    block_bytes = 128 * 10
    assert total == (int(available * 0.05) // block_bytes) * block_bytes
    assert plan is not None


def test_reserve_vision_cache_blocks_scales_with_pool() -> None:
    small, _, _ = _block_reserve(
        utilization=0.5,
        row_bytes=8,
        available_memory=10 * 1024**2,
    )
    large, _, _ = _block_reserve(
        utilization=0.5,
        row_bytes=8,
        available_memory=20 * 1024**2,
    )
    assert 0 < small <= 5 * 1024**2
    assert small < large <= 10 * 1024**2


def test_reserve_vision_cache_blocks_raises_when_explicitly_set() -> None:
    with pytest.raises(ValueError, match="too small to fit one"):
        _block_reserve(
            utilization=0.001,
            row_bytes=1024**2,
            available_memory=10 * 1024**2,
        )


def test_reserve_vision_cache_blocks_shards_budget_across_devices() -> None:
    available = 1024**3
    total, plan, _ = _block_reserve(
        utilization=0.001,
        row_bytes=10,
        available_memory=available,
        n_devices=2,
    )
    block_bytes = 128 * 10
    requested = int(available * 0.001) // 2
    num_blocks = requested // block_bytes // 2 * 2
    assert total == num_blocks * block_bytes
    assert plan.bytes_per_device == total // 2


def _memory_reserve(
    utilization: float,
    per_entry_bytes: int,
    available_memory: int,
    n_devices: int = 1,
    row_spec: tuple[int, DType] | None = None,
) -> tuple[int, VisionCachePlan | None, "PipelineRuntimeConfig"]:
    """Run _reserve_vision_cache_memory against a real runtime config."""
    runtime = PipelineRuntimeConfig(vision_cache_utilization=utilization)
    pipeline_config = SimpleNamespace(runtime=runtime)

    class _VisionArchConfig:
        @classmethod
        def estimate_vision_cache_entry_bytes(
            cls, huggingface_config: object
        ) -> int:
            return per_entry_bytes

        @classmethod
        def get_vision_cache_row_spec(
            cls, huggingface_config: object
        ) -> tuple[int, DType] | None:
            return row_spec

    arch = SimpleNamespace(name="test-arch", config=_VisionArchConfig)
    total, plan = MemoryEstimator._reserve_vision_cache_memory(
        cast("PipelineConfig", pipeline_config),
        cast("MAXModelConfig", SimpleNamespace(huggingface_config=None)),
        available_memory,
        [cast(Device, MagicMock())] * n_devices,
        arch=cast("SupportedArchitecture", arch),
    )
    return total, plan, runtime


def test_reserve_vision_cache_memory_disabled_when_utilization_zero() -> None:
    total, plan, _ = _memory_reserve(
        utilization=0.0,
        per_entry_bytes=1024,
        available_memory=1024**3,
        row_spec=(10, DType.uint8),
    )
    assert total == 0
    assert plan is None


def test_reserve_vision_cache_memory_disabled_when_no_row_spec() -> None:
    default_utilization = PipelineRuntimeConfig().vision_cache_utilization
    total, plan, runtime = _memory_reserve(
        utilization=default_utilization,
        per_entry_bytes=1024,
        available_memory=1024**3,
        row_spec=None,
    )
    assert total == 0
    assert plan is None
    # Planning no longer writes the config; construction owns the disable.
    assert runtime.vision_cache_utilization == default_utilization


def test_reserve_vision_cache_memory_disabled_when_no_vision_tower() -> None:
    total, plan, _ = _memory_reserve(
        utilization=PipelineRuntimeConfig().vision_cache_utilization,
        per_entry_bytes=0,
        available_memory=1024**3,
    )
    assert total == 0
    assert plan is None


def test_reserve_vision_cache_memory_block_mode_returns_plan() -> None:
    total, plan, _ = _memory_reserve(
        utilization=0.001,
        per_entry_bytes=1024,
        available_memory=1024**3,
        row_spec=(10, DType.uint8),
    )
    assert plan is not None
    assert plan.bytes_per_device == total
    assert (plan.hidden_size, plan.dtype) == (10, DType.uint8)
