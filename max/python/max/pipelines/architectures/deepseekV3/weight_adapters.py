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
from transformers.configuration_utils import PretrainedConfig

# Maps from Safetensor to MAX weight names.
DEEPSEEK_SAFETENSOR_MAP = {
    "model.": "",  # Removes the "model" prefix.
    "gate.weight": "gate.gate_score.weight",
    "weight_scale_inv": "weight_scale",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    # Map the weight names.
    for name, value in state_dict.items():
        max_name = name
        for before, after in DEEPSEEK_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        # Drop FP8 KV-cache static scales emitted by modelopt NVFP4
        # checkpoints (e.g. `k_proj.k_scale`, `v_proj.v_scale`). MAX reads
        # KV cache scales from a separate configuration path, so these keys
        # would otherwise trigger a strict load_state_dict failure.
        if max_name.endswith((".k_scale", ".v_scale")):
            continue
        new_state_dict[max_name] = value.data()

    # TODO(E2EOPT-673): Support MTP. We currently delete the MTP weights
    # This is also done in the official DeepSeek HF checkpoint converter:
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/4592be48c07f036b32ef971474068aebc489e3e7/inference/convert.py#L53-L54
    mtp_layer_idx = huggingface_config.num_hidden_layers
    for key in list(new_state_dict.keys()):
        if key.startswith(f"layers.{mtp_layer_idx}."):
            del new_state_dict[key]

    return new_state_dict
