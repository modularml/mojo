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

"""Quantization-metadata tests for Qwen3.5.

``RadixArk/Qwen3.8-27B-NVFP4`` used to load without raising and emit garbage:
its ``quant_algo`` is ``MIXED_PRECISION``, which the generic modelopt parser
read as uniform NVFP4, and ``load_state_dict(strict=False)`` then dropped every
scale tensor. These tests pin the two gates that make that loud -- the scheme
parser refusing anything it cannot represent, and the weight check refusing a
load that drops a scale.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.quant_config import QuantFormat, ScaleGranularity, ScaleOrigin
from max.pipelines.architectures.qwen3_5.model import _check_weights_match
from max.pipelines.architectures.qwen3_5.quantization import parse_quant_scheme

NUM_LAYERS = 4
# Layer 3 is full attention; 0-2 are linear attention (Gated DeltaNet).
_FULL_ATTN_LAYERS = (3,)


def _quantized_layers() -> dict[str, dict[str, object]]:
    """The `quantized_layers` map modelopt writes, at four layers."""
    entries: dict[str, dict[str, object]] = {
        "lm_head": {"quant_algo": "NVFP4", "group_size": 16}
    }
    for i in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{i}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            entries[f"{prefix}.mlp.{proj}"] = {
                "quant_algo": "NVFP4",
                "group_size": 16,
            }
        if i in _FULL_ATTN_LAYERS:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                entries[f"{prefix}.self_attn.{proj}"] = {"quant_algo": "FP8"}
        else:
            for proj in ("in_proj_qkv", "in_proj_z", "out_proj"):
                entries[f"{prefix}.linear_attn.{proj}"] = {"quant_algo": "FP8"}
    return entries


def _hf_quant_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": _quantized_layers(),
    }
    config.update(overrides)
    return config


def _state_dict() -> dict[str, WeightData]:
    """Only the tensors the scheme parser reads: the embedding and the scales."""
    weights = {
        "embed_tokens.weight": WeightData.from_numpy(
            np.zeros((8, 4), dtype=np.float32), "embed_tokens.weight"
        )
    }
    for i in range(NUM_LAYERS):
        for proj in ("gate_proj", "up_proj"):
            base = f"layers.{i}.mlp.{proj}"
            for suffix, value in (
                ("weight_scale_2", 1.5e-4),
                ("input_scale", 1.4e-3),
            ):
                weights[f"{base}.{suffix}"] = WeightData.from_numpy(
                    np.array(value, dtype=np.float32), f"{base}.{suffix}"
                )
    return weights


def test_mixed_precision_map_splits_into_two_configs() -> None:
    scheme = parse_quant_scheme(_hf_quant_config(), _state_dict(), NUM_LAYERS)
    assert scheme is not None

    assert scheme.mlp is not None
    assert scheme.mlp.format == QuantFormat.NVFP4
    assert scheme.mlp.weight_scale.block_size == (1, 16)
    assert scheme.mlp.input_scale.origin == ScaleOrigin.STATIC
    assert scheme.mlp_layers == frozenset(range(NUM_LAYERS))
    assert scheme.quantize_lm_head

    assert scheme.attn is not None
    assert scheme.attn.weight_scale.granularity == ScaleGranularity.TENSOR
    assert scheme.attn.input_scale.granularity == ScaleGranularity.TENSOR
    assert scheme.attn.input_scale.origin == ScaleOrigin.STATIC
    assert scheme.attn.attn_quantized_layers == set(range(NUM_LAYERS))

    # Both families of layer are covered, and each gets only its own format.
    for layer in range(NUM_LAYERS):
        assert scheme.mlp_config(layer) is scheme.mlp
        assert scheme.attn_config(layer) is scheme.attn

    # The unquantized weights stay at the embedding's dtype, not the packed
    # `uint8` the encoding implies.
    assert scheme.compute_dtype == DType.float32


def test_no_quantization_config_is_not_quantized() -> None:
    assert parse_quant_scheme(None, _state_dict(), NUM_LAYERS) is None


def test_mixed_precision_without_the_map_raises() -> None:
    config = _hf_quant_config()
    del config["quantized_layers"]
    with pytest.raises(ValueError, match="no 'quantized_layers' map"):
        parse_quant_scheme(config, _state_dict(), NUM_LAYERS)


def test_unknown_quant_algo_raises() -> None:
    with pytest.raises(ValueError, match="cannot read quant_algo"):
        parse_quant_scheme(
            _hf_quant_config(quant_algo="W4A8_AWQ"), _state_dict(), NUM_LAYERS
        )


def test_partially_quantized_mlp_raises() -> None:
    entries = _quantized_layers()
    del entries["model.language_model.layers.0.mlp.up_proj"]
    with pytest.raises(ValueError, match="partially quantized"):
        parse_quant_scheme(
            _hf_quant_config(quantized_layers=entries),
            _state_dict(),
            NUM_LAYERS,
        )


def test_unrecognized_module_raises() -> None:
    entries = _quantized_layers()
    entries["model.language_model.layers.0.mlp.experts.0.w1"] = {
        "quant_algo": "NVFP4"
    }
    with pytest.raises(ValueError, match="Refusing to guess"):
        parse_quant_scheme(
            _hf_quant_config(quantized_layers=entries),
            _state_dict(),
            NUM_LAYERS,
        )


def test_wrong_nvfp4_group_size_raises() -> None:
    entries = _quantized_layers()
    entries["model.language_model.layers.0.mlp.gate_proj"] = {
        "quant_algo": "NVFP4",
        "group_size": 32,
    }
    with pytest.raises(ValueError, match="group_size 32"):
        parse_quant_scheme(
            _hf_quant_config(quantized_layers=entries),
            _state_dict(),
            NUM_LAYERS,
        )


def test_attention_quantized_as_nvfp4_raises() -> None:
    entries = _quantized_layers()
    entries["model.language_model.layers.3.self_attn.q_proj"] = {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }
    with pytest.raises(ValueError, match="per-tensor FP8 only"):
        parse_quant_scheme(
            _hf_quant_config(quantized_layers=entries),
            _state_dict(),
            NUM_LAYERS,
        )


def test_unconsumed_scale_tensor_fails_the_load() -> None:
    expected = {"layers.0.mlp.gate_proj.weight"}
    provided = expected | {"layers.0.mlp.gate_proj.weight_scale_2"}
    with pytest.raises(ValueError, match="no layer consumes"):
        _check_weights_match(expected=expected, provided=provided)


def test_missing_weight_fails_the_load() -> None:
    with pytest.raises(ValueError, match="missing 1 weight"):
        _check_weights_match(
            expected={"layers.0.mlp.gate_proj.weight_scale"},
            provided=set(),
        )


def test_skipped_mtp_scales_do_not_fail_the_load() -> None:
    # The weight adapter drops `mtp.*`, so its scales are unused by design.
    _check_weights_match(
        expected={"layers.0.mlp.gate_proj.weight"},
        provided={
            "layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.gate_proj.input_scale",
        },
    )
