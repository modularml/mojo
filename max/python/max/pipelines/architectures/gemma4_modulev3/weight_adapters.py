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
"""Weight adapters for Gemma4 ModuleV3: reuse the graph arch's converters."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.gemma4.weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Full-checkpoint adapter used by the registry: language + vision keys,
    each re-prefixed to match the eager module trees.

    The two slices never collide: the language converter only accepts
    ``(model.)language_model.*`` keys and the vision converter only accepts
    ``model.vision_tower.*`` / ``model.embed_vision.*``.
    """
    converted = convert_language_state_dict_for_module(state_dict)
    converted.update(convert_vision_state_dict_for_module(state_dict))
    return converted


def convert_language_state_dict_for_module(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Language slice, re-prefixed under ``language_model.``.

    The graph arch's converter filters checkpoint keys down to
    `model.language_model.*` / `language_model.*` and strips those prefixes
    (the graph arch's module tree consumes bare, unprefixed parameter paths).
    Our ModuleV3 module tree nests everything under `language_model`, so
    re-add the prefix here to match the module's parameter paths.
    """
    language = convert_safetensor_language_state_dict(state_dict)
    return {f"language_model.{name}": data for name, data in language.items()}


def convert_vision_state_dict_for_module(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Vision slice, unprefixed.

    The eager vision tower mirrors the graph tower's attribute names, so the
    graph converter's output keys (``patch_embedder.*``, ``encoder.layers.*``,
    ``embed_vision.*``, ``std_bias``/``std_scale``) already match
    ``Gemma4VisionModel``'s parameter paths. Text-only checkpoints simply
    match no key, yielding an empty dict.
    """
    return dict(convert_safetensor_vision_state_dict(state_dict))
