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
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from ..gemma4.weight_adapters import convert_safetensor_language_state_dict


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Convert safetensor state dict for target Gemma4 language weights only.

    Draft weights are loaded separately in model.py from the draft_model
    checkpoint.
    """
    return convert_safetensor_language_state_dict(state_dict)


def convert_unified_safetensor_state_dict(
    target_state_dict: dict[str, WeightData],
    draft_state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Merge target + draft weights with target.*/draft.* prefixes.

    The draft's ``embed_tokens`` is renamed to ``draft_embed_tokens``
    (the assistant's own 1024-dim embedding used for the tied lm_head).
    The target's ``embed_tokens`` is shared via module aliasing for the
    concat(embed, hidden_states) input step.
    """
    unified: dict[str, WeightData] = {}

    for name, value in target_state_dict.items():
        unified[f"target.{name}"] = value

    for name, value in draft_state_dict.items():
        # Strip the HF `model.` prefix (e.g. `model.layers.0...` → `layers.0...`).
        name = name.removeprefix("model.")
        # Rename embed_tokens → draft_embed_tokens (the assistant's own
        # 1024-dim embedding, NOT shared from target).
        if name.startswith("embed_tokens."):
            name = "draft_" + name
        unified[f"draft.{name}"] = value

    return unified
