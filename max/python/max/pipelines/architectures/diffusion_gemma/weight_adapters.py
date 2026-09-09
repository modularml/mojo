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
"""Weight adapters for the DiffusionGemma NVFP4 checkpoint.

The checkpoint stores the shared transformer weights once, under
``model.decoder.*`` (HF ties every encoder layer weight to its decoder twin,
so only the decoder side plus the encoder-only ``layer_scalar`` buffers,
vision tower, and ``embed_vision`` are serialized). ``lm_head`` is absent —
tied to ``model.decoder.embed_tokens``.

MoE experts are stored per expert (``experts.J.{gate,up,down}_proj``), each as
an NVFP4 quadruplet: packed ``weight`` (u8), ``weight_scale`` (fp8-e4m3 group
scales), ``weight_scale_2`` and ``input_scale`` (fp32) — exactly the per-expert
fields ``max.nn.moe.MoEQuantized`` reads, so unlike the gemma4 donor adapter no
stacked-tensor splitting is needed.

Both the encoder and decoder MAX graphs load from the same converted dict;
shared FQNs in the combined weights registry give weight sharing for free.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

logger = logging.getLogger(__name__)

# Ordered prefix/infix rewrites: checkpoint name -> MAX graph FQN.
# The decoder prefix is stripped first so the remaining rules see bare
# ``layers.N...`` names. MoE renames mirror the gemma4 donor adapter.
_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "model.decoder.": "",
    "router.proj.weight": "moe_block.gate.gate_score.weight",
    "router.scale": "moe_block.gate.scale",
    "router.per_expert_scale": "moe_block.gate.per_expert_scale",
    "pre_feedforward_layernorm_2.weight": "moe_block.pre_expert_norm.weight",
    "experts.": "moe_block.experts.",
}

_VISION_SAFETENSOR_MAP: dict[str, str] = {
    "model.encoder.vision_tower.": "",
    "model.encoder.embed_vision": "embed_vision",
    ".linear.": ".",
}

_ENCODER_LAYER_SCALAR_PREFIX = "model.encoder.language_model.layers."
_DECODER_PREFIX = "model.decoder."


def _weight_bytes(data: WeightData) -> np.ndarray:
    """Returns the raw bytes of a weight for dtype-agnostic comparison."""
    nbytes = math.prod(int(d) for d in data.shape) * data.dtype.size_in_bytes
    return Buffer.from_dlpack(data.data).view(DType.uint8, [nbytes]).to_numpy()


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Converts checkpoint keys to the FQNs the text graphs expect.

    Keeps one copy of each tied weight (the ``model.decoder.*`` canonical
    one). Encoder ``layer_scalar`` buffers are the only untied per-layer
    tensors in the checkpoint; this adapter asserts they are byte-identical
    to the decoder's and drops them, because the MAX port shares a single
    weight set between the encoder and decoder graphs. If a future checkpoint
    trains them apart, the graphs need per-graph scalar names instead.
    """
    new_state_dict: dict[str, WeightData] = {}
    encoder_layer_scalars: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if weight_name.startswith(_ENCODER_LAYER_SCALAR_PREFIX):
            if weight_name.endswith(".layer_scalar"):
                encoder_layer_scalars[
                    weight_name.removeprefix("model.encoder.language_model.")
                ] = value.data()
            continue
        if not weight_name.startswith(_DECODER_PREFIX):
            continue

        max_name = weight_name
        for before, after in _LANGUAGE_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        data = value.data()

        # The BF16 base checkpoint (google/diffusiongemma-26B-A4B-it) stores
        # experts stacked as [num_experts, 2*moe_dim, hidden] /
        # [num_experts, hidden, moe_dim] parameters; split per expert like
        # the gemma4 donor adapter. The NVFP4 checkpoint is already
        # per-expert and never hits these branches.
        if max_name.endswith("moe_block.experts.gate_up_proj"):
            prefix = max_name.split("moe_block.experts.")[0]
            buf = Buffer.from_dlpack(data.data)
            half = buf.shape[1] // 2
            expert_shape = [half, buf.shape[2]]
            for j in range(buf.shape[0]):
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
        if max_name.endswith("moe_block.experts.down_proj"):
            prefix = max_name.split("moe_block.experts.")[0]
            buf = Buffer.from_dlpack(data.data)
            expert_shape = list(buf.shape[1:])
            for j in range(buf.shape[0]):
                name = f"{prefix}moe_block.experts.{j}.down_proj.weight"
                expert_buf = buf[j : j + 1, :, :].view(data.dtype, expert_shape)
                new_state_dict[name] = WeightData(
                    expert_buf, name, data.dtype, Shape(expert_shape)
                )
            continue

        new_state_dict[max_name] = data

    for name, enc_data in encoder_layer_scalars.items():
        dec_data = new_state_dict.get(name)
        if dec_data is None:
            raise ValueError(
                f"Encoder buffer {name!r} has no decoder twin in checkpoint."
            )
        if not np.array_equal(_weight_bytes(enc_data), _weight_bytes(dec_data)):
            raise NotImplementedError(
                f"Encoder and decoder {name!r} differ; the single-weight-set"
                " port assumes they are equal. Give the encoder graph its own"
                " scalar FQNs before serving this checkpoint."
            )

    if encoder_layer_scalars:
        logger.info(
            "Verified %d encoder layer_scalar buffers equal decoder twins.",
            len(encoder_layer_scalars),
        )
    return new_state_dict


def convert_safetensor_vision_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Converts vision tower + multimodal embedder keys to donor FQNs.

    Same targets as the gemma4 donor vision adapter; only the source prefixes
    differ (DiffusionGemma nests the tower under ``model.encoder.``). The
    ``.linear.`` strip flattens HF's ClippableLinear wrapper.
    """
    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        if not (
            weight_name.startswith(
                ("model.encoder.vision_tower.", "model.encoder.embed_vision.")
            )
        ):
            continue
        max_name = weight_name
        for before, after in _VISION_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    return new_state_dict


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Registered adapter: merged language + vision conversion.

    The pipeline model's ``load_model`` calls the two specific converters
    directly (mirroring the gemma4 donor); this merged form exists for
    registry tooling and offline audits.
    """
    return {
        **convert_safetensor_language_state_dict(state_dict),
        **convert_safetensor_vision_state_dict(state_dict),
    }
