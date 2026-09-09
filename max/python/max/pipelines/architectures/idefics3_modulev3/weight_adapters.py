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
"""Weight adapters for Idefics3 ModuleV3.

V3 weight names must match the module attribute hierarchy from the root
compiled module. For the language model, the root wrapper has
``self.language_model = Idefics3TextModel(...)`` so all inner weights need
the ``language_model.`` prefix.

For the vision model, it is compiled as the root module directly, so no
extra prefix is needed (same as V2).
"""

from __future__ import annotations

from max.graph.weights import WeightData, Weights

# V3 language model weight mapping.
# HuggingFace checkpoint -> V3 module hierarchy.
#
# The compiled root module stores ``self.language_model = Idefics3TextModel(...)``
# so HuggingFace weights like ``model.text_model.layers.0.self_attn.q_proj.weight``
# become ``language_model.layers.0.self_attn.q_proj.weight``.
IDEFICS3_LANGUAGE_MODEL_MAPPING: dict[str, str] = {
    "model.text_model.": "language_model.",
    # Remap unfused Q/K/V projections into StackedLinear namespace.
}

# V3 vision model weight mapping.
# The vision model is compiled as root, so prefixes are stripped.
IDEFICS3_VISION_MODEL_MAPPING: dict[str, str] = {
    "model.vision_model.": "",
    "model.connector.": "connector.",
}


def convert_idefics3_language_model_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert Idefics3 language model weights for V3 module hierarchy.

    Maps HuggingFace checkpoint names to the V3 module attribute path.
    The ``language_model.`` prefix is required because the compiled root
    stores the text model as ``self.language_model``.

    Args:
        state_dict: The raw Idefics3 checkpoint weights.

    Returns:
        Mapped weights for the V3 language model.
    """
    llm_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        if checkpoint_name.startswith("lm_head."):
            # Map lm_head to language_model.lm_head
            llm_state_dict[f"language_model.{checkpoint_name}"] = weight.data()

        elif checkpoint_name.startswith("model.text_model."):
            llm_name = checkpoint_name
            for before, after in IDEFICS3_LANGUAGE_MODEL_MAPPING.items():
                llm_name = llm_name.replace(before, after)
            llm_state_dict[llm_name] = weight.data()

    return llm_state_dict


def convert_idefics3_vision_model_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert Idefics3 vision model weights for V3 module hierarchy.

    The vision model is compiled as the root module, so weight names map
    directly to module attributes (same structure as V2).

    Note: Unlike V2, V3's Conv2d with ``permute=True`` expects weights in
    PyTorch FCRS format [out_channels, in_channels, height, width], so the
    patch embedding weight transpose is NOT needed.

    Args:
        state_dict: The raw Idefics3 checkpoint weights.

    Returns:
        Mapped weights for the V3 vision model.
    """
    vision_model_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        if checkpoint_name.startswith(
            ("model.connector.", "model.vision_model.")
        ):
            vision_model_name = checkpoint_name
            for before, after in IDEFICS3_VISION_MODEL_MAPPING.items():
                vision_model_name = vision_model_name.replace(before, after)

            # No weight transpose needed for V3 Conv2d with permute=True.
            vision_model_state_dict[vision_model_name] = weight.data()

    return vision_model_state_dict
