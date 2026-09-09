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
"""Tests for LoRAModel class functionality."""

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.pipelines.lora.lora import LoRAModel
from safetensors.numpy import save_file

LoRAWeights = dict[str, npt.NDArray[Any]]

TEST_RANK = 8
TEST_MAX_RANK = 16
TEST_IN_FEATURES = 64
TEST_N_HEADS = 4
TEST_N_KV_HEADS = 2
TEST_HEAD_DIM = 16
TEST_Q_OUT_FEATURES = TEST_N_HEADS * TEST_HEAD_DIM
TEST_KV_OUT_FEATURES = TEST_N_KV_HEADS * TEST_HEAD_DIM


def create_adapter_config(
    rank: int = TEST_RANK,
    lora_alpha: int = 16,
    target_modules: list[str] | None = None,
    bias: str = "none",
) -> dict[str, object]:
    """Create a minimal adapter config dict."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj"]
    return {
        "r": rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "bias": bias,
    }


def create_lora_weights(
    layer_idx: int = 0,
    rank: int = TEST_RANK,
    include_q: bool = True,
    include_k: bool = True,
    include_v: bool = True,
    include_o: bool = False,
    in_features: int = TEST_IN_FEATURES,
    q_out_features: int = TEST_Q_OUT_FEATURES,
    kv_out_features: int = TEST_KV_OUT_FEATURES,
    dtype: npt.DTypeLike = np.float32,
) -> LoRAWeights:
    """Create dummy LoRA weights for testing.

    LoRA A shape: [rank, in_features]
    LoRA B shape: [out_features, rank]
    """
    weights = {}
    prefix = f"base_model.model.layers.{layer_idx}.self_attn"

    if include_q:
        weights[f"{prefix}.q_proj.lora_A.weight"] = np.random.randn(
            rank, in_features
        ).astype(dtype)
        weights[f"{prefix}.q_proj.lora_B.weight"] = np.random.randn(
            q_out_features, rank
        ).astype(dtype)

    if include_k:
        weights[f"{prefix}.k_proj.lora_A.weight"] = np.random.randn(
            rank, in_features
        ).astype(dtype)
        weights[f"{prefix}.k_proj.lora_B.weight"] = np.random.randn(
            kv_out_features, rank
        ).astype(dtype)

    if include_v:
        weights[f"{prefix}.v_proj.lora_A.weight"] = np.random.randn(
            rank, in_features
        ).astype(dtype)
        weights[f"{prefix}.v_proj.lora_B.weight"] = np.random.randn(
            kv_out_features, rank
        ).astype(dtype)

    if include_o:
        weights[f"{prefix}.o_proj.lora_A.weight"] = np.random.randn(
            rank, q_out_features
        ).astype(dtype)
        weights[f"{prefix}.o_proj.lora_B.weight"] = np.random.randn(
            q_out_features, rank
        ).astype(dtype)

    return weights


@pytest.fixture
def temp_lora_adapter() -> Callable[..., str]:
    """Create a temporary LoRA adapter directory with config and weights."""

    def _create_adapter(
        config: dict[str, object] | None = None,
        weights: LoRAWeights | None = None,
    ) -> str:
        tmpdir = tempfile.mkdtemp()

        if config is None:
            config = create_adapter_config()
        config_path = Path(tmpdir) / "adapter_config.json"
        config_path.write_text(json.dumps(config))

        if weights is None:
            weights = create_lora_weights()
        weights_path = Path(tmpdir) / "adapter_model.safetensors"
        save_file(weights, str(weights_path))

        return tmpdir

    return _create_adapter


# =============================================================================
# Basic Loading Tests
# =============================================================================


def test_load_basic_qkv_adapter(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test loading a basic adapter with Q, K, V weights."""
    adapter_path = temp_lora_adapter()

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    assert model.name == "test_lora"
    assert model.rank == TEST_RANK
    assert model.target_modules == ["q_proj", "k_proj", "v_proj"]


def test_load_adapter_with_o_proj(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test loading adapter with Q, K, V, and O projection weights."""
    config = create_adapter_config(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    weights = create_lora_weights(include_o=True)
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    assert "o_proj" in model.target_modules
    assert "layers.0.self_attn.o_proj.lora_A.weight" in model.lora_A
    assert "layers.0.self_attn.o_proj.lora_B.weight" in model.lora_B


def test_adapter_config_properties(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that adapter config is properly loaded."""
    config = create_adapter_config(rank=4, lora_alpha=32)
    weights = create_lora_weights(rank=4)
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    assert model.rank == 4
    assert model.adapter_config["lora_alpha"] == 32


# =============================================================================
# QKV Combining Tests
# =============================================================================


def test_qkv_weights_are_combined(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that Q, K, V weights are combined into qkv_lora keys."""
    adapter_path = temp_lora_adapter()

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    assert "layers.0.self_attn.q_proj.lora_A.weight" not in model.lora_A
    assert "layers.0.self_attn.k_proj.lora_A.weight" not in model.lora_A
    assert "layers.0.self_attn.v_proj.lora_A.weight" not in model.lora_A

    assert "layers.0.self_attn.qkv_lora.lora_A.weight" in model.lora_A
    assert "layers.0.self_attn.qkv_lora.lora_B.weight" in model.lora_B


def test_combined_lora_a_shape(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that combined LoRA A has correct shape [3*max_rank, in_features]."""
    adapter_path = temp_lora_adapter()

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    combined_a = model.lora_A["layers.0.self_attn.qkv_lora.lora_A.weight"]
    weight = Buffer.from_dlpack(combined_a.data)

    expected_shape = (3 * TEST_MAX_RANK, TEST_IN_FEATURES)
    assert tuple(weight.shape) == expected_shape


def test_combined_lora_b_shape(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test the fused LoRA B shape [q_out + 2*kv_out, max_rank] (Q | K | V)."""
    adapter_path = temp_lora_adapter()

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    combined_b = model.lora_B["layers.0.self_attn.qkv_lora.lora_B.weight"]
    weight = Buffer.from_dlpack(combined_b.data)

    expected_shape = (
        TEST_Q_OUT_FEATURES + 2 * TEST_KV_OUT_FEATURES,
        TEST_MAX_RANK,
    )
    assert tuple(weight.shape) == expected_shape


# =============================================================================
# Partial QKV Tests (missing some projections)
# =============================================================================


def test_missing_k_projection(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test handling when K projection is missing - should create zeros."""
    config = create_adapter_config(target_modules=["q_proj", "v_proj"])
    weights = create_lora_weights(include_k=False)
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    assert "layers.0.self_attn.qkv_lora.lora_A.weight" in model.lora_A
    assert "layers.0.self_attn.qkv_lora.lora_B.weight" in model.lora_B

    combined_a = model.lora_A["layers.0.self_attn.qkv_lora.lora_A.weight"]
    weight_a = Buffer.from_dlpack(combined_a.data).to_numpy()

    k_portion = weight_a[TEST_MAX_RANK : 2 * TEST_MAX_RANK, :]
    assert np.allclose(k_portion, 0.0)

    q_portion = weight_a[:TEST_MAX_RANK, :]
    v_portion = weight_a[2 * TEST_MAX_RANK :, :]
    assert not np.allclose(q_portion, 0.0)
    assert not np.allclose(v_portion, 0.0)

    # Fused B is [q_out | k_out | v_out] along axis 0; the K region is zeroed.
    combined_b = model.lora_B["layers.0.self_attn.qkv_lora.lora_B.weight"]
    weight_b = Buffer.from_dlpack(combined_b.data).to_numpy()
    k_region_b = weight_b[
        TEST_Q_OUT_FEATURES : TEST_Q_OUT_FEATURES + TEST_KV_OUT_FEATURES, :
    ]
    assert np.allclose(k_region_b, 0.0)


def test_missing_v_projection(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test handling when V projection is missing - should create zeros."""
    config = create_adapter_config(target_modules=["q_proj", "k_proj"])
    weights = create_lora_weights(include_v=False)
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    combined_a = model.lora_A["layers.0.self_attn.qkv_lora.lora_A.weight"]
    weight_a = Buffer.from_dlpack(combined_a.data).to_numpy()

    v_portion = weight_a[2 * TEST_MAX_RANK :, :]
    assert np.allclose(v_portion, 0.0)

    combined_b = model.lora_B["layers.0.self_attn.qkv_lora.lora_B.weight"]
    weight_b = Buffer.from_dlpack(combined_b.data).to_numpy()

    # Fused B is [q_out | k_out | v_out] along axis 0.
    v_portion_b = weight_b[TEST_Q_OUT_FEATURES + TEST_KV_OUT_FEATURES :, :]
    assert np.allclose(v_portion_b, 0.0)

    k_portion_b = weight_b[
        TEST_Q_OUT_FEATURES : TEST_Q_OUT_FEATURES + TEST_KV_OUT_FEATURES, :
    ]
    assert not np.allclose(k_portion_b, 0.0)


def test_only_q_projection(temp_lora_adapter: Callable[..., str]) -> None:
    """Test handling when only Q projection exists."""
    config = create_adapter_config(target_modules=["q_proj"])
    weights = create_lora_weights(include_k=False, include_v=False)
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    combined_a = model.lora_A["layers.0.self_attn.qkv_lora.lora_A.weight"]
    weight_a = Buffer.from_dlpack(combined_a.data).to_numpy()

    q_portion = weight_a[:TEST_MAX_RANK, :]
    k_portion = weight_a[TEST_MAX_RANK : 2 * TEST_MAX_RANK, :]
    v_portion = weight_a[2 * TEST_MAX_RANK :, :]

    assert not np.allclose(q_portion, 0.0)
    assert np.allclose(k_portion, 0.0)
    assert np.allclose(v_portion, 0.0)


# =============================================================================
# Padding Tests
# =============================================================================


def test_padding_when_rank_less_than_max(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that weights are padded when rank < max_lora_rank."""
    small_rank = 4
    config = create_adapter_config(rank=small_rank)
    weights = create_lora_weights(rank=small_rank)
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    combined_a = model.lora_A["layers.0.self_attn.qkv_lora.lora_A.weight"]
    weight_a = Buffer.from_dlpack(combined_a.data).to_numpy()

    assert weight_a.shape == (3 * TEST_MAX_RANK, TEST_IN_FEATURES)

    q_padded = weight_a[small_rank:TEST_MAX_RANK, :]
    assert np.allclose(q_padded, 0.0)

    k_padded = weight_a[TEST_MAX_RANK + small_rank : 2 * TEST_MAX_RANK, :]
    assert np.allclose(k_padded, 0.0)

    v_padded = weight_a[2 * TEST_MAX_RANK + small_rank :, :]
    assert np.allclose(v_padded, 0.0)


def test_no_padding_when_rank_equals_max(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that no extra padding when rank == max_lora_rank."""
    config = create_adapter_config(rank=TEST_MAX_RANK)
    weights = create_lora_weights(rank=TEST_MAX_RANK)
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    combined_a = model.lora_A["layers.0.self_attn.qkv_lora.lora_A.weight"]
    weight_a = Buffer.from_dlpack(combined_a.data).to_numpy()

    assert weight_a.shape == (3 * TEST_MAX_RANK, TEST_IN_FEATURES)


# =============================================================================
# Scaling Tests
# =============================================================================


def test_lora_b_is_scaled(temp_lora_adapter: Callable[..., str]) -> None:
    """Test that LoRA B weights are pre-multiplied by scale = alpha/rank."""
    rank = 8
    alpha = 16
    scale = alpha / rank

    config = create_adapter_config(rank=rank, lora_alpha=alpha)

    np.random.seed(42)
    original_b_weight = np.random.randn(TEST_Q_OUT_FEATURES, rank).astype(
        np.float32
    )

    weights = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(
            rank, TEST_IN_FEATURES
        ).astype(np.float32),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": original_b_weight.copy(),
        "base_model.model.layers.0.self_attn.k_proj.lora_A.weight": np.random.randn(
            rank, TEST_IN_FEATURES
        ).astype(np.float32),
        "base_model.model.layers.0.self_attn.k_proj.lora_B.weight": np.random.randn(
            TEST_KV_OUT_FEATURES, rank
        ).astype(np.float32),
        "base_model.model.layers.0.self_attn.v_proj.lora_A.weight": np.random.randn(
            rank, TEST_IN_FEATURES
        ).astype(np.float32),
        "base_model.model.layers.0.self_attn.v_proj.lora_B.weight": np.random.randn(
            TEST_KV_OUT_FEATURES, rank
        ).astype(np.float32),
    }

    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    combined_b = model.lora_B["layers.0.self_attn.qkv_lora.lora_B.weight"]
    actual_b = Buffer.from_dlpack(combined_b.data).to_numpy()

    expected_scaled = original_b_weight * scale
    # Q region is the first q_out_features rows of the fused B.
    actual_non_padded = actual_b[:TEST_Q_OUT_FEATURES, :rank]

    assert np.allclose(actual_non_padded, expected_scaled, rtol=1e-5)


# =============================================================================
# Dtype Casting Tests
# =============================================================================


def test_weights_cast_to_base_dtype(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that weights are cast to base_dtype."""
    weights = create_lora_weights(dtype=np.float32)
    adapter_path = temp_lora_adapter(weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.bfloat16,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    combined_a = model.lora_A["layers.0.self_attn.qkv_lora.lora_A.weight"]
    weight = Buffer.from_dlpack(combined_a.data)
    assert weight.dtype == DType.bfloat16


# =============================================================================
# Multiple Layers Tests
# =============================================================================


def test_multiple_layers(temp_lora_adapter: Callable[..., str]) -> None:
    """Test loading adapter with weights for multiple layers."""
    config = create_adapter_config()

    weights = {}
    for layer_idx in range(3):
        layer_weights = create_lora_weights(layer_idx=layer_idx)
        weights.update(layer_weights)

    adapter_path = temp_lora_adapter(config=config, weights=weights)

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    for layer_idx in range(3):
        assert (
            f"layers.{layer_idx}.self_attn.qkv_lora.lora_A.weight"
            in model.lora_A
        )
        assert (
            f"layers.{layer_idx}.self_attn.qkv_lora.lora_B.weight"
            in model.lora_B
        )


# =============================================================================
# Validation Tests
# =============================================================================


def test_unsupported_target_module_raises(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that unsupported target modules raise ValueError."""
    config = create_adapter_config(
        target_modules=["q_proj", "unsupported_module"]
    )
    weights = create_lora_weights()
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    with pytest.raises(ValueError, match="unsupported target modules"):
        LoRAModel(
            name="test_lora",
            path=adapter_path,
            base_dtype=DType.float32,
            max_lora_rank=TEST_MAX_RANK,
            n_heads=TEST_N_HEADS,
            n_kv_heads=TEST_N_KV_HEADS,
            head_dim=TEST_HEAD_DIM,
        )


def test_bias_not_none_raises(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that bias != 'none' raises ValueError."""
    config = create_adapter_config(bias="all")
    weights = create_lora_weights()
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    with pytest.raises(
        ValueError, match="bias training is not currently supported"
    ):
        LoRAModel(
            name="test_lora",
            path=adapter_path,
            base_dtype=DType.float32,
            max_lora_rank=TEST_MAX_RANK,
            n_heads=TEST_N_HEADS,
            n_kv_heads=TEST_N_KV_HEADS,
            head_dim=TEST_HEAD_DIM,
        )


def test_missing_config_raises() -> None:
    """Test that missing adapter_config.json raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        weights = create_lora_weights()
        weights_path = Path(tmpdir) / "adapter_model.safetensors"
        save_file(weights, str(weights_path))

        with pytest.raises(ValueError, match="Adapter config file not found"):
            LoRAModel(
                name="test_lora",
                path=tmpdir,
                base_dtype=DType.float32,
                max_lora_rank=TEST_MAX_RANK,
                n_heads=TEST_N_HEADS,
                n_kv_heads=TEST_N_KV_HEADS,
                head_dim=TEST_HEAD_DIM,
            )


def test_greater_than_max_rank_raises(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that missing adapter_config.json raises ValueError."""
    weights = create_lora_weights(rank=17)
    config = create_adapter_config(rank=17)
    adapter_path = temp_lora_adapter(config=config, weights=weights)

    with pytest.raises(
        ValueError, match="LoRA of rank 17 exceeds maximum rank"
    ):
        LoRAModel(
            name="test_lora",
            path=adapter_path,
            base_dtype=DType.float32,
            max_lora_rank=TEST_MAX_RANK,
            n_heads=TEST_N_HEADS,
            n_kv_heads=TEST_N_KV_HEADS,
            head_dim=TEST_HEAD_DIM,
        )


# =============================================================================
# Get Method Tests
# =============================================================================


def test_get_existing_weight(temp_lora_adapter: Callable[..., str]) -> None:
    """Test getting an existing weight."""
    adapter_path = temp_lora_adapter()

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    weight = model.get("layers.0.self_attn.qkv_lora.lora_A.weight")
    assert weight is not None


def test_get_nonexistent_weight_returns_none(
    temp_lora_adapter: Callable[..., str],
) -> None:
    """Test that getting a nonexistent weight returns None."""
    adapter_path = temp_lora_adapter()

    model = LoRAModel(
        name="test_lora",
        path=adapter_path,
        base_dtype=DType.float32,
        max_lora_rank=TEST_MAX_RANK,
        n_heads=TEST_N_HEADS,
        n_kv_heads=TEST_N_KV_HEADS,
        head_dim=TEST_HEAD_DIM,
    )

    weight = model.get("nonexistent.weight.key")
    assert weight is None
