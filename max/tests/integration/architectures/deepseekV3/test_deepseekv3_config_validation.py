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
"""Tests for DeepseekV3 configuration validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import NonCallableMock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV3.deepseekV3 import (
    _validate_parallelism_config,
)
from max.pipelines.architectures.deepseekV3.model_config import DeepseekV3Config
from max.pipelines.architectures.deepseekV3_nextn.model_config import (
    DeepseekV3NextNConfig,
)


def make_mock_config(
    *,
    ep_config: NonCallableMock | None = None,
    data_parallel_degree: int = 1,
    num_devices: int = 8,
) -> NonCallableMock:
    """Create a mock DeepseekV3Config for testing."""
    config = NonCallableMock(spec=DeepseekV3Config)
    config.ep_config = ep_config
    config.data_parallel_degree = data_parallel_degree
    config.devices = [
        NonCallableMock(spec=DeviceRef) for _ in range(num_devices)
    ]
    return config


def test_multi_gpu_requires_ep() -> None:
    """Test that multi-GPU requires EP config."""
    # num_devices=8 with matching data_parallel_degree but no EP config
    config = make_mock_config(
        ep_config=None, data_parallel_degree=8, num_devices=8
    )

    with pytest.raises(ValueError, match=r"Expert-parallel.*must be enabled"):
        _validate_parallelism_config(config)


def test_valid_single_gpu_config() -> None:
    """Test that single GPU config is valid without EP."""
    config = make_mock_config(
        ep_config=None, data_parallel_degree=1, num_devices=1
    )
    # Should not raise
    _validate_parallelism_config(config)


def test_valid_multi_gpu_with_ep() -> None:
    """Test that multi-GPU config with EP is valid."""
    ep_config = NonCallableMock()
    config = make_mock_config(
        ep_config=ep_config, data_parallel_degree=8, num_devices=8
    )
    # Should not raise
    _validate_parallelism_config(config)


# ── DeepseekV3NextNConfig validation tests ──


def _make_nextn_config_kwargs(
    *, data_parallel_degree: int, num_devices: int
) -> dict[str, Any]:
    """Return minimal kwargs for constructing a DeepseekV3NextNConfig."""
    from max.nn.kv_cache import MLAKVCacheParams

    devices = [DeviceRef("gpu", i) for i in range(num_devices)]
    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        devices=devices,
        data_parallel_degree=data_parallel_degree,
        num_q_heads=128,
    )
    return dict(
        dtype=DType.bfloat16,
        kv_params=kv_params,
        devices=devices,
        data_parallel_degree=data_parallel_degree,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_key_value_heads=128,
        n_shared_experts=1,
        n_routed_experts=256,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        first_k_dense_replace=3,
        vocab_size=129280,
        max_position_embeddings=4096,
        max_seq_len=163840,
        rope_scaling={
            "type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )


def test_nextn_config_tp_mode_valid() -> None:
    """Test that NextN config accepts TP mode (dp=1, multi-GPU)."""
    kwargs = _make_nextn_config_kwargs(data_parallel_degree=1, num_devices=8)
    # Should not raise
    DeepseekV3NextNConfig(**kwargs)


def test_nextn_config_dp_mode_valid() -> None:
    """Test that NextN config accepts DP mode (dp=num_devices)."""
    kwargs = _make_nextn_config_kwargs(data_parallel_degree=8, num_devices=8)
    # Should not raise
    DeepseekV3NextNConfig(**kwargs)
