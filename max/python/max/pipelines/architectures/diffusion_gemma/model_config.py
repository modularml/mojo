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
"""Config for the DiffusionGemmaForBlockDiffusion port.

DiffusionGemma's text backbone is configured exactly like Gemma4's, but its
HF config class *removes* several Gemma4 fields (``attention_k_eq_v``,
``enable_moe_block``, PLE and shared-KV fields) that
``Gemma4TextConfig.initialize_from_config`` reads unconditionally. This module
overlays those fields with their DiffusionGemma-correct constants through a
read-only config view, and adds the block-diffusion fields (``canvas_length``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from max.graph.weights import WeightData
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self


class _ConfigView:
    """Read-only attribute view over an HF config with explicit overrides.

    Attribute lookups consult ``overrides`` first and fall through to the
    wrapped config. This avoids mutating transformers' strict config objects,
    which reject both unknown attributes and assignment.
    """

    def __init__(self, base: Any, **overrides: Any) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_base"), name)


def _diffusion_hf_config_view(huggingface_config: Any) -> _ConfigView:
    """Wraps the DiffusionGemma HF config so Gemma4 config code can read it.

    The overlaid values are constants of the DiffusionGemma architecture, per
    ``transformers/models/diffusion_gemma/modular_diffusion_gemma.py``:

    - ``attention_k_eq_v=True``: full-attention layers have no ``v_proj``;
      V is computed from the ``k_proj`` output (modular line ~228).
    - ``enable_moe_block=True``: every layer runs the parallel dense-MLP +
      MoE branch; there is no dense-only variant.
    - PLE (``vocab_size_per_layer_input``/``hidden_size_per_layer_input``)
      and shared-KV (``num_kv_shared_layers``) are removed from
      DiffusionGemma; the values below make the donor treat them as absent.
    """
    text = huggingface_config.text_config
    text_view = _ConfigView(
        text,
        attention_k_eq_v=True,
        enable_moe_block=True,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        vocab_size_per_layer_input=text.vocab_size,
        hidden_size_per_layer_input=0,
    )
    return _ConfigView(huggingface_config, text_config=text_view)


@dataclass(kw_only=True)
class DiffusionGemmaForBlockDiffusionConfig(
    Gemma4ForConditionalGenerationConfig
):
    """Top-level MAX config for DiffusionGemma block diffusion.

    Extends the Gemma4 multimodal config with the block-diffusion canvas
    geometry. KV cache parameters, per-layer-type attention geometry, MoE
    sizing, and the vision tower config are all inherited from the donor,
    which reads them through the view installed by ``initialize_from_config``.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float4_e2m1fnx2"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float4_e2m1fnx2",
        "bfloat16",
    }

    canvas_length: int = 256
    """Number of tokens denoised per block-diffusion canvas."""

    boi_token_id: int = 255_999
    """Begin-of-image token id wrapping image prompts."""

    eoi_token_id: int = 258_882
    """End-of-image token id wrapping image prompts."""

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        *,
        max_seq_len: int,
    ) -> Self:
        view = _diffusion_hf_config_view(huggingface_config)
        cfg = super().initialize_from_config(
            pipeline_config, view, max_seq_len=max_seq_len
        )
        cfg.canvas_length = huggingface_config.canvas_length
        cfg.boi_token_id = huggingface_config.boi_token_id
        cfg.eoi_token_id = huggingface_config.eoi_token_id
        return cfg

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        super().finalize(
            _diffusion_hf_config_view(huggingface_config),
            state_dict,
            return_logits,
        )
