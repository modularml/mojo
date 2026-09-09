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
"""Config for LFM2 models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from max.graph.weights import WeightData
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig

from ..llama3.model_config import Llama3Config

DEFAULT_ROPE_THETA = 10000.0


@dataclass(kw_only=True)
class LFM2Config(Llama3Config):
    """Model configuration for LFM2 graph construction/execution."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float32"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    layer_types: list[str] = field(default_factory=list)
    conv_L_cache: int = 3
    conv_bias: bool = False
    norm_eps: float = 1e-5

    @staticmethod
    def _get_rope_theta(huggingface_config: AutoConfig) -> float:
        rope_params = getattr(huggingface_config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            return float(rope_params.get("rope_theta", DEFAULT_ROPE_THETA))
        return DEFAULT_ROPE_THETA

    @classmethod
    def _ensure_optional_rope_fields(
        cls, huggingface_config: AutoConfig
    ) -> None:
        # Some test runtimes use a slimmer transformers config object that
        # omits optional rope fields accessed by Llama3Config initialization.
        if not hasattr(huggingface_config, "rope_scaling"):
            huggingface_config.rope_scaling = None
        if not hasattr(huggingface_config, "rope_theta"):
            huggingface_config.rope_theta = cls._get_rope_theta(
                huggingface_config
            )

    @staticmethod
    def _resolve_intermediate_size(
        cfg: LFM2Config, huggingface_config: AutoConfig
    ) -> int:
        intermediate = cfg.intermediate_size
        if not bool(
            getattr(huggingface_config, "block_auto_adjust_ff_dim", False)
        ):
            return intermediate

        # SwiGLU FFN uses two gated projections, so the effective width is
        # multiplied by 2/3 to keep the total parameter count comparable to a
        # standard 4x FFN (same convention as LLaMA / Meta Llama models).
        intermediate = int(2 * intermediate / 3)
        multiplier = getattr(
            huggingface_config, "block_ffn_dim_multiplier", None
        )
        if multiplier is not None:
            intermediate = int(float(multiplier) * intermediate)
        multiple_of = int(getattr(huggingface_config, "block_multiple_of", 1))
        if multiple_of > 1:
            intermediate = multiple_of * (
                (intermediate + multiple_of - 1) // multiple_of
            )
        return intermediate

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: Any,
        huggingface_config: AutoConfig,
        model_config: Any = None,
        *,
        max_seq_len: int,
    ) -> LFM2Config:
        cls._ensure_optional_rope_fields(huggingface_config)
        cfg = super().initialize_from_config(
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )
        cfg.layer_types = list(getattr(huggingface_config, "layer_types", []))
        cfg.conv_L_cache = int(getattr(huggingface_config, "conv_L_cache", 3))
        cfg.conv_bias = bool(getattr(huggingface_config, "conv_bias", False))
        cfg.norm_eps = float(getattr(huggingface_config, "norm_eps", 1e-5))
        cfg.intermediate_size = cls._resolve_intermediate_size(
            cfg, huggingface_config
        )
        rope_params = getattr(huggingface_config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            cfg.rope_theta = float(
                rope_params.get("rope_theta", cfg.rope_theta)
            )
        return cfg

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        # LFM2 stores its norm epsilon under ``norm_eps``, not ``rms_norm_eps``.
        # Llama3Config.finalize reads ``huggingface_config.rms_norm_eps`` when
        # norm_method="rms_norm", which would raise AttributeError on LFM2
        # configs.  Passing "layer_norm" skips that attribute lookup entirely;
        # we then override norm_method back to "rms_norm" and set rms_norm_eps
        # ourselves from the correct ``norm_eps`` field.
        super().finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            norm_method="layer_norm",
            attention_bias=False,
        )
        self.norm_method = "rms_norm"
        self.rms_norm_eps = float(getattr(huggingface_config, "norm_eps", 1e-5))
        # transformers >= 5 folds LiquidAI's custom `tie_embedding` config key
        # into the standard `tie_word_embeddings` field (and drops the original),
        # so read that first and fall back to the legacy key for older versions.
        self.tie_word_embeddings = bool(
            getattr(
                huggingface_config,
                "tie_word_embeddings",
                getattr(huggingface_config, "tie_embedding", False),
            )
        )
