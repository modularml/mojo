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

from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

from ..deepseekV3.weight_adapters import DEEPSEEK_SAFETENSOR_MAP

# MTP layer key mappings: from checkpoint MTP-layer suffix to draft module path.
# The MTP layer in the GLM-5.2 checkpoint is at index ``num_hidden_layers``.
# ``embed_tokens`` and ``shared_head.head`` (lm_head) are shared with the target
# and are not emitted under the MTP layer in this checkpoint, so they need no
# mapping here.
_MTP_LAYER_MAP = {
    "shared_head.norm.": "draft.shared_head_norm.",
    "enorm.": "draft.enorm.",
    "hnorm.": "draft.hnorm.",
    "eh_proj.": "draft.eh_proj.",
}


def convert_with_mtp_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor weights for the unified GLM-5.2 MTP model.

    Produces keys prefixed with ``target.*`` and ``draft.*`` to match the
    composed :class:`UnifiedMTPGlm5_2` module hierarchy. Shared weights
    (``embed_tokens``, ``lm_head``) appear only under ``target.*`` — weight
    sharing via module aliasing causes ``state_dict()`` to deduplicate them.
    """
    new_state_dict: dict[str, WeightData] = {}
    mtp_layer_idx = huggingface_config.num_hidden_layers

    for name, value in state_dict.items():
        max_name = name
        for before, after in DEEPSEEK_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        # Drop FP8 KV-cache static scales emitted by some checkpoints; MAX
        # reads KV cache scales from a separate configuration path.
        if max_name.endswith((".k_scale", ".v_scale")):
            continue

        mtp_prefix = f"layers.{mtp_layer_idx}."
        if max_name.startswith(mtp_prefix):
            suffix = max_name[len(mtp_prefix) :]

            mapped = False
            for before, after in _MTP_LAYER_MAP.items():
                if suffix.startswith(before):
                    new_state_dict[after + suffix[len(before) :]] = value.data()
                    mapped = True
                    break

            if not mapped:
                new_state_dict[f"draft.decoder_layer.{suffix}"] = value.data()
        else:
            new_state_dict[f"target.{max_name}"] = value.data()

    return new_state_dict
