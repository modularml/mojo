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

from __future__ import annotations

import dataclasses

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights

from .model_config import Gemma4ForConditionalGenerationConfig

GEMMA4_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "model.language_model.": "",
    "language_model.model.": "",
    "router.proj.weight": "moe_block.gate.gate_score.weight",
    "router.scale": "moe_block.gate.scale",
    "router.per_expert_scale": "moe_block.gate.per_expert_scale",
    "pre_feedforward_layernorm_2.weight": "moe_block.pre_expert_norm.weight",
    "experts.": "moe_block.experts.",
}

GEMMA4_VISION_SAFETENSOR_MAP: dict[str, str] = {
    "model.vision_tower.": "",
    "model.embed_vision": "embed_vision",
    ".linear.": ".",
}


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the language model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        # modelopt checkpoints may carry FP8 KV-cache scales (k_scale /
        # v_scale) that MAX's BF16 KV cache does not consume; drop them.
        if weight_name.endswith((".k_scale", ".v_scale")):
            continue
        if not (
            weight_name.startswith(("language_model.", "model.language_model."))
        ):
            continue

        max_name = weight_name
        for before, after in GEMMA4_LANGUAGE_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        data = value.data()

        if max_name.endswith(".weight_scale") and data.dtype == DType.uint8:
            data = dataclasses.replace(data, dtype=DType.float8_e8m0fnu)

        # Stacked MoE expert weights: split into individual per-expert weights.
        # HF stores gate_up_proj [num_experts, 2*moe_dim, hidden_dim]
        # and down_proj [num_experts, hidden_dim, moe_dim] as single tensors.
        if "moe_block.experts.gate_up_proj" in max_name:
            prefix = max_name.split("moe_block.experts.")[0]
            buf = Buffer.from_dlpack(data.data)
            num_experts = buf.shape[0]
            half = buf.shape[1] // 2
            expert_shape = [half, buf.shape[2]]
            for j in range(num_experts):
                for proj, s in [
                    ("gate_proj", slice(None, half)),
                    ("up_proj", slice(half, None)),
                ]:
                    name = f"{prefix}moe_block.experts.{j}.{proj}.weight"
                    proj_buf = buf[j : j + 1, s, :].view(
                        data.dtype, expert_shape
                    )
                    new_state_dict[name] = WeightData(
                        proj_buf, name, data.dtype, Shape(expert_shape)
                    )
            continue

        if "moe_block.experts.down_proj" in max_name:
            prefix = max_name.split("moe_block.experts.")[0]
            buf = Buffer.from_dlpack(data.data)
            num_experts = buf.shape[0]
            expert_shape = list(buf.shape[1:])
            for j in range(num_experts):
                name = f"{prefix}moe_block.experts.{j}.down_proj.weight"
                expert_buf = buf[j : j + 1, :, :].view(data.dtype, expert_shape)
                new_state_dict[name] = WeightData(
                    expert_buf, name, data.dtype, Shape(expert_shape)
                )
            continue

        new_state_dict[max_name] = data

    return new_state_dict


def _row_concat(parts: list[WeightData], name: str) -> WeightData:
    """Concatenates weight tensors along axis 0 (row-major, dtype-agnostic).

    Views each part as raw bytes so the concat works for any dtype (bf16 has
    no native numpy type); the result keeps the parts' dtype and gets the
    summed row count.
    """
    dtype = parts[0].dtype
    itemsize = dtype.size_in_bytes
    byte_rows = []
    for part in parts:
        part_shape = [int(d) for d in part.shape]
        buf = Buffer.from_dlpack(part.data)
        flat = np.from_dlpack(
            buf.view(DType.uint8, [int(np.prod(part_shape)) * itemsize])
        )
        byte_rows.append(flat.reshape(part_shape[0], -1))
    concatenated = np.ascontiguousarray(np.concatenate(byte_rows, axis=0))
    out_shape = [sum(int(p.shape[0]) for p in parts)] + [
        int(d) for d in parts[0].shape[1:]
    ]
    fused = Buffer.from_dlpack(concatenated).view(dtype, out_shape)
    return WeightData(fused, name, dtype, Shape(out_shape))


def gemma4_uses_fused_projections(
    config: Gemma4ForConditionalGenerationConfig,
) -> bool:
    """Returns whether gemma4 builds fused projection layers (DISTINF-194).

    Single source of truth shared by the graph side (which selects
    :class:`~max.nn.FusedMLP` / stacked qkv layers) and the weight-adapter
    side (:func:`fuse_gemma4_projection_weights`): checkpoint keys must be
    fused if and only if the layers expect fused weights, otherwise strict
    loading fails or fused keys target an unfused graph. Fused projections
    are single-device, unquantized only.
    """
    return (
        config.text_config.fused_projection_weights
        and len(config.devices) == 1
        and config.text_config.quant_config is None
    )


def fuse_gemma4_projection_weights(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Pre-fuse MLP gate/up and attention qkv/qk projections (DISTINF-194).

    Each decoder layer's ``gate_proj``/``up_proj`` are concatenated into a single
    ``mlp.gate_up_proj_fused`` and its ``q_proj``/``k_proj``(/``v_proj``) into a
    single ``self_attn.qkv_proj.weight`` (or ``self_attn.qk_proj.weight`` when
    the checkpoint has no ``v_proj``), matching the :class:`~max.nn.FusedMLP` /
    ``StackedLinear(stacked=True)`` layers those layers build. The source
    per-projection keys are dropped. Call this AFTER MAX-name conversion and
    BEFORE any ``target.``/``draft.`` prefixing.
    """
    fused = dict(state_dict)

    mlp_prefixes = [
        key.removesuffix(".gate_proj.weight")
        for key in state_dict
        if key.endswith(".mlp.gate_proj.weight")
    ]
    for prefix in mlp_prefixes:
        gate = fused.pop(f"{prefix}.gate_proj.weight")
        up = fused.pop(f"{prefix}.up_proj.weight")
        name = f"{prefix}.gate_up_proj_fused"
        fused[name] = _row_concat([gate, up], name)

    attn_prefixes = [
        key.removesuffix(".q_proj.weight")
        for key in state_dict
        if key.endswith(".self_attn.q_proj.weight")
    ]
    for prefix in attn_prefixes:
        parts = [
            fused.pop(f"{prefix}.q_proj.weight"),
            fused.pop(f"{prefix}.k_proj.weight"),
        ]
        v_key = f"{prefix}.v_proj.weight"
        if v_key in fused:
            parts.append(fused.pop(v_key))
            name = f"{prefix}.qkv_proj.weight"
        else:
            name = f"{prefix}.qk_proj.weight"
        fused[name] = _row_concat(parts, name)

    return fused


def convert_safetensor_vision_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the vision model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if not (
            weight_name.startswith(
                ("model.vision_tower.", "model.embed_vision.")
            )
        ):
            continue

        max_name = weight_name
        for before, after in GEMMA4_VISION_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        new_state_dict[max_name] = value.data()

    return new_state_dict
