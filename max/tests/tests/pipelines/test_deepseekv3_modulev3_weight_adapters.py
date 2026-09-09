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

"""Name-remapping checks for the DeepseekV3 ModuleV3 weight adapters.

The adapter picks a naming scheme from the checkpoint's own scale entries, then
has to land every leaf on the matching module parameter -- so the assertions
here are the parameter names the quantized layers expose (see
``max/tests/tests/nn/module_v3/quant_layers``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from max.dtype import DType
from max.graph.weights import Weights
from max.pipelines.architectures.deepseekV3_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)

_NUM_LAYERS = 1
# Layer index _NUM_LAYERS is the MTP layer, which the adapter drops.
_MTP_LAYER = _NUM_LAYERS


def _weight(dtype: DType) -> Any:
    """A stand-in ``Weights`` whose ``data()`` reports only its dtype."""
    weights = MagicMock(spec=Weights)
    weights.data.return_value = SimpleNamespace(dtype=dtype)
    return weights


def _convert(state_dict: dict[str, Any]) -> dict[str, Any]:
    return convert_safetensor_state_dict(
        state_dict, DeepseekV3Config(num_hidden_layers=_NUM_LAYERS)
    )


def test_nvfp4_projection_leaves_land_on_the_quantized_weight() -> None:
    """Each modelopt scale entry maps onto its ``NVFP4Tensor`` leaf."""
    converted = _convert(
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": _weight(
                DType.uint8
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale": _weight(
                DType.float8_e4m3fn
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2": _weight(
                DType.float32
            ),
            "model.layers.0.mlp.experts.0.gate_proj.input_scale": _weight(
                DType.float32
            ),
        }
    )

    prefix = "language_model.layers.0.mlp.experts.0.gate_proj.weight"
    assert set(converted) == {
        f"{prefix}.data",
        f"{prefix}.weight_scale",
        f"{prefix}.weight_scale_2",
        f"{prefix}.input_scale",
    }


def test_nvfp4_quantizes_o_proj_but_not_the_latent_projections() -> None:
    """In an NVFP4 checkpoint the MLA q/kv projections stay bf16 raw tensors."""
    converted = _convert(
        {
            "model.layers.0.self_attn.o_proj.weight": _weight(DType.uint8),
            "model.layers.0.self_attn.o_proj.weight_scale": _weight(
                DType.float8_e4m3fn
            ),
            "model.layers.0.self_attn.o_proj.weight_scale_2": _weight(
                DType.float32
            ),
            "model.layers.0.self_attn.q_a_proj.weight": _weight(DType.bfloat16),
            "model.layers.0.self_attn.kv_b_proj.weight": _weight(
                DType.bfloat16
            ),
        }
    )

    assert set(converted) == {
        "language_model.layers.0.self_attn.o_proj.weight.data",
        "language_model.layers.0.self_attn.o_proj.weight.weight_scale",
        "language_model.layers.0.self_attn.o_proj.weight.weight_scale_2",
        # Raw-tensor MLA projections drop the ``.weight`` suffix entirely.
        "language_model.layers.0.self_attn.q_a_proj",
        "language_model.layers.0.self_attn.kv_b_proj",
    }


def test_nvfp4_applies_the_shared_bf16_renames() -> None:
    """The cross-cutting V3 renames still run on an NVFP4 checkpoint."""
    converted = _convert(
        {
            "model.layers.0.mlp.gate.weight_scale_2": _weight(DType.float32),
            "model.layers.0.mlp.gate.weight": _weight(DType.bfloat16),
            "model.layers.0.self_attn.kv_a_layernorm.weight": _weight(
                DType.bfloat16
            ),
            "model.layers.0.input_layernorm.weight": _weight(DType.bfloat16),
            "model.embed_tokens.weight": _weight(DType.bfloat16),
            "lm_head.weight": _weight(DType.bfloat16),
        }
    )

    assert "language_model.layers.0.mlp.gate.gate_score.weight" in converted
    assert "language_model.layers.0.self_attn.kv_a_proj_layernorm" in converted
    assert "language_model.layers.0.input_layernorm.weight" in converted
    assert "language_model.embed_tokens.weight" in converted
    assert "language_model.lm_head.weight" in converted


def test_nvfp4_drops_kv_cache_scales_and_mtp_layer() -> None:
    """modelopt emits KV-cache scales and an MTP layer that MAX doesn't load."""
    converted = _convert(
        {
            "model.layers.0.self_attn.o_proj.weight_scale_2": _weight(
                DType.float32
            ),
            "model.layers.0.self_attn.k_proj.k_scale": _weight(DType.float32),
            "model.layers.0.self_attn.v_proj.v_scale": _weight(DType.float32),
            f"model.layers.{_MTP_LAYER}.mlp.experts.0.gate_proj.weight": _weight(
                DType.uint8
            ),
            f"model.layers.{_MTP_LAYER}.input_layernorm.weight": _weight(
                DType.bfloat16
            ),
        }
    )

    assert set(converted) == {
        "language_model.layers.0.self_attn.o_proj.weight.weight_scale_2"
    }


def test_scale_entries_pick_the_naming_scheme() -> None:
    """``weight_scale_2`` selects NVFP4; ``weight_scale_inv`` selects FP8."""
    fp8 = _convert(
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": _weight(
                DType.float8_e4m3fn
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": _weight(
                DType.float32
            ),
        }
    )
    prefix = "language_model.layers.0.mlp.experts.0.gate_proj.weight"
    assert set(fp8) == {f"{prefix}.data", f"{prefix}.weight_scale_inv"}

    bf16 = _convert(
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": _weight(
                DType.bfloat16
            ),
        }
    )
    assert set(bf16) == {prefix}
