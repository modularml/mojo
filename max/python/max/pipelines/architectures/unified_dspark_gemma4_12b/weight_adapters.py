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
"""Weight adapters for the unified DSpark Gemma4 pipeline."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

from ..gemma4.weight_adapters import convert_safetensor_language_state_dict

# Draft checkpoint weights aliased to the target's modules instead of being
# loaded from the DSpark checkpoint's frozen copies.
SHARED_DRAFT_PREFIXES = ("embed_tokens.", "lm_head.")
# Draft checkpoint weights with no M1 consumer (the confidence head gates
# dynamic proposal lengths, which MAX does not use).
SKIPPED_DRAFT_PREFIXES = ("confidence_head.",)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Converts the target Gemma4 checkpoint's language weights.

    Draft weights are loaded separately in ``model.py`` from the
    ``draft_model`` checkpoint (its keys already match the draft module).
    """
    del huggingface_config
    return convert_safetensor_language_state_dict(state_dict)


def convert_unified_safetensor_state_dict(
    target_state_dict: dict[str, WeightData],
    draft_state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Merges target + draft weights for the unified model.

    Prefixes target weights with ``target.*`` and draft weights with
    ``draft.*`` verbatim (the DSpark checkpoint carries no ``model.``
    prefix). The draft's frozen ``embed_tokens`` / ``lm_head`` copies are
    dropped in favor of module aliases to the target's — after asserting
    their shapes match — and the unused ``confidence_head`` is skipped.
    """
    unified: dict[str, WeightData] = {}

    for name, value in target_state_dict.items():
        unified[f"target.{name}"] = value

    target_embed = target_state_dict.get("embed_tokens.weight")
    for name, value in draft_state_dict.items():
        if name.startswith(SHARED_DRAFT_PREFIXES):
            if target_embed is not None and tuple(value.shape) != tuple(
                target_embed.shape
            ):
                raise ValueError(
                    f"DSpark draft weight '{name}' has shape"
                    f" {tuple(value.shape)}, which does not match the"
                    f" target's embed_tokens {tuple(target_embed.shape)};"
                    " it cannot be aliased to the target's modules."
                )
            continue
        if name.startswith(SKIPPED_DRAFT_PREFIXES):
            continue
        unified[f"draft.{name}"] = value

    return unified
