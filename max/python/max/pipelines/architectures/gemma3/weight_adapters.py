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

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights


def _to_bfloat16(
    weight_data: WeightData, *, subtract_one: bool = False
) -> WeightData:
    """Converts float16/float32 weight data to bfloat16.

    Goes through raw bytes + a `Buffer` rather than `WeightData.astype` because
    NumPy/DLPack cannot represent bfloat16. bfloat16 is the high 16 bits of a
    float32, so we truncate after widening to float32.

    When `subtract_one` is set, 1.0 is subtracted from the values first. This is
    used for Gemma's RMSNorm weights: llama.cpp bakes the `+1` of Gemma's
    `(1 + weight)` normalization into the stored GGUF weight, whereas MAX (like
    HuggingFace) stores `weight` and applies `(1 + weight)` at runtime.
    """
    f32 = np.ascontiguousarray(np.from_dlpack(weight_data.data)).astype(
        np.float32
    )
    if subtract_one:
        f32 = f32 - np.float32(1.0)
    bf16_bits = (f32.view(np.uint32) >> 16).astype(np.uint16)
    buffer = Buffer.from_dlpack(bf16_bits.view(np.uint8)).view(
        DType.bfloat16, [int(d) for d in weight_data.shape]
    )
    return WeightData(
        buffer, weight_data.name, DType.bfloat16, weight_data.shape
    )


# Maps from Safetensor to MAX weight names.
GEMMA3_SAFETENSOR_MAP: dict[str, str] = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "lm_head.": "language_model.lm_head.",
    "model.layers.": "language_model.layers.",
}

# Maps from GGUF (llama.cpp) to MAX weight names. The replacements are applied
# in order as substring substitutions, so more specific keys (e.g. the `*_norm`
# tensors) must come before the shorter keys they contain (e.g. `attn_q`).
GEMMA3_GGUF_MAP: dict[str, str] = {
    "token_embd": "language_model.embed_tokens",
    "output_norm": "language_model.norm",
    "post_attention_norm": "post_attention_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "attn_output": "self_attn.o_proj",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_norm": "input_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_norm": "pre_feedforward_layernorm",
    "blk": "language_model.layers",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    # Remap HuggingFace -> MAX-style names
    for weight_name, value in state_dict.items():
        max_name = weight_name
        for before, after in GEMMA3_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    return new_state_dict


def convert_gguf_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    # Remap GGUF (llama.cpp) -> MAX-style names.
    for weight_name, value in state_dict.items():
        max_name = weight_name
        for before, after in GEMMA3_GGUF_MAP.items():
            max_name = max_name.replace(before, after)
        weight_data = value.data()
        # MAX runs Gemma 3 in bfloat16 on GPU. GGUF files mix dtypes — the bulk
        # of the weights are float16/bfloat16 while the (layer)norm weights are
        # kept in float32 — so convert the non-bfloat16 weights to bfloat16 to
        # match the graph encoding. RMSNorm weights additionally need the `+1`
        # that llama.cpp baked into them removed (see `_to_bfloat16`).
        is_norm = max_name.endswith("norm.weight")
        if is_norm:
            weight_data = _to_bfloat16(weight_data, subtract_one=True)
        elif weight_data.dtype in (DType.float16, DType.float32):
            weight_data = _to_bfloat16(weight_data)
        new_state_dict[max_name] = weight_data

    # GGUF bakes the rotary embedding frequencies into the weights file, while
    # MAX computes them at runtime from the config. Drop it if present.
    new_state_dict.pop("rope_freqs.weight", None)

    return new_state_dict
