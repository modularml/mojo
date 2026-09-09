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

from max.graph.weights import WeightData, Weights

# Maps HuggingFace safetensor weight names to V3 module-attribute paths from the
# DeepseekV2 root module. The root module stores the language model under
# ``self.language_model``, so all inner attributes are prefixed accordingly.
# Multi-Latent Attention stores several projections as raw `Tensor` parameters
# (e.g. ``q_a_proj``) instead of ``Linear``, so the trailing ``.weight`` suffix
# from HF must be stripped for those names.
DEEPSEEK_SAFETENSOR_MAP = {
    # Strip ``.weight`` for raw-Tensor MLA projections.
    "self_attn.q_a_proj.weight": "self_attn.q_a_proj",
    "self_attn.q_b_proj.weight": "self_attn.q_b_proj",
    "self_attn.q_proj.weight": "self_attn.q_proj",
    "self_attn.kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_a_layernorm.weight": "self_attn.kv_a_proj_layernorm",
    "self_attn.kv_b_proj.weight": "self_attn.kv_b_proj",
    # MoE gate uses an internal ``gate_score`` Linear.
    "mlp.gate.weight": "mlp.gate.gate_score.weight",
    # Place under the ``language_model`` submodule of the root module.
    "model.": "language_model.",
    # ``lm_head`` is a top-level HF weight; move it under ``language_model``.
    "lm_head.": "language_model.lm_head.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    for name, value in state_dict.items():
        max_name = name
        for before, after in DEEPSEEK_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    return new_state_dict
