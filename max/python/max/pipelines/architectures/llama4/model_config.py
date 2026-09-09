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
"""Config for Llama4 (text-only) models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.rotary_embedding import Llama3RopeScalingParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
    supported_encoding_quantization,
)
from max.pipelines.weights import gptq_quant_config
from transformers import AutoConfig
from typing_extensions import Self, override


def get_text_config(huggingface_config: AutoConfig) -> AutoConfig:
    """Returns the text sub-config for Llama4.

    The full multimodal ``Llama4Config`` nests text fields under
    ``text_config``; a text-only ``Llama4TextConfig`` is already flat. This
    accessor lets the rest of this module read fields uniformly regardless of
    which shape was loaded.
    """
    return (
        getattr(huggingface_config, "text_config", None) or huggingface_config
    )


def _default_no_rope_layers(
    no_rope_layers: list[int] | None,
    no_rope_layer_interval: int,
    num_hidden_layers: int,
) -> list[int]:
    """Mirrors ``Llama4TextConfig.__post_init__`` for ``no_rope_layers``.

    A ``1`` marks a RoPE layer, a ``0`` marks a NoPE layer (every
    ``no_rope_layer_interval``-th layer).
    """
    if no_rope_layers:
        return list(no_rope_layers)
    return [
        int((layer_idx + 1) % no_rope_layer_interval != 0)
        for layer_idx in range(num_hidden_layers)
    ]


def _default_moe_layers(
    moe_layers: list[int] | None,
    interleave_moe_layer_step: int,
    num_hidden_layers: int,
) -> list[int]:
    """Mirrors ``Llama4TextConfig.__post_init__`` for ``moe_layers``."""
    if moe_layers is not None:
        return list(moe_layers)
    return list(
        range(
            interleave_moe_layer_step - 1,
            num_hidden_layers,
            interleave_moe_layer_step,
        )
    )


def _build_llama4_fp8_quant_config(
    huggingface_config: AutoConfig,
    dtype: DType,
) -> QuantConfig | None:
    """Builds the :class:`QuantConfig` for a compressed-tensors float-quantized
    (FP8 e4m3) Llama4 checkpoint.

    The released ``RedHatAI/Llama-4-Scout-...-FP8-dynamic`` checkpoint quantizes
    only the feed-forward linears (routed experts + shared expert) to FP8 e4m3
    with per-output-channel (rowwise) static weight scales and per-token
    (colwise) dynamic activation scales. Attention, the router gate, ``lm_head``
    and the vision tower stay bf16 (they are in the ``ignore`` list).

    The stock ``parse_quant_config`` assumes dense ``mlp.{gate,up,down}_proj``
    naming and directly indexes ``layers.0.mlp.down_proj.weight_scale``, which
    does not exist for Llama4's ``feed_forward.experts.*`` MoE layout. This
    builder reads the scheme straight from the HuggingFace ``config_groups``
    instead so it works for the split-per-expert layout.
    """
    if dtype != DType.float8_e4m3fn:
        return None
    hf_quant = getattr(
        huggingface_config, "quantization_config", None
    ) or getattr(
        get_text_config(huggingface_config), "quantization_config", None
    )
    if not hf_quant:
        return None
    # ``quantization_config`` is a plain dict in config.json; normalize objects.
    if not isinstance(hf_quant, dict):
        hf_quant = getattr(hf_quant, "to_dict", lambda: hf_quant.__dict__)()
    quant_method = hf_quant.get("quant_method")
    if quant_method and quant_method != "compressed-tensors":
        raise ValueError(
            "Llama4 FP8 only supports the compressed-tensors quant_method, got "
            f"{quant_method!r}"
        )

    group = hf_quant["config_groups"]["group_0"]
    weight_cfg = group["weights"]
    input_cfg = group["input_activations"]

    def _granularity(strategy: str) -> ScaleGranularity:
        if strategy == "channel":
            return ScaleGranularity.ROWWISE
        if strategy == "token":
            return ScaleGranularity.COLWISE
        if strategy == "tensor":
            return ScaleGranularity.TENSOR
        raise ValueError(f"unsupported FP8 scale strategy: {strategy!r}")

    if weight_cfg.get("dynamic"):
        raise ValueError(
            "compressed-tensors FP8 weights must use static scales for Llama4"
        )

    num_hidden_layers = int(
        get_text_config(huggingface_config).num_hidden_layers
    )
    # Weight/activation scales are consumed as float32 by the FP8 kernels; the
    # weight adapter casts the stored scales to float32 on load.
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=_granularity(input_cfg["strategy"]),
            origin=ScaleOrigin.DYNAMIC
            if input_cfg.get("dynamic")
            else ScaleOrigin.STATIC,
            dtype=DType.float32,
        ),
        weight_scale=WeightScaleSpec(
            granularity=_granularity(weight_cfg["strategy"]),
            dtype=DType.float32,
        ),
        # Only feed-forward linears are FP8; the model build keeps attention,
        # router, embedding and lm_head bf16 by construction, so
        # attn_quantized_layers is empty.
        mlp_quantized_layers=set(range(num_hidden_layers)),
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        format=QuantFormat.COMPRESSED_TENSORS_FP8,
    )


@dataclass(kw_only=True)
class Llama4Config(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Model configuration for Llama4 (text-only) graph construction."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float8_e4m3fn",
    }

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int
    rope_theta: float
    rope_scaling_params: Llama3RopeScalingParams | None
    max_seq_len: int
    intermediate_size: int
    intermediate_size_mlp: int
    vocab_size: int
    dtype: DType
    model_quantization_encoding: QuantizationEncoding | None
    quantization_config: QuantizationConfig | None
    kv_params: KVCacheParams
    devices: list[DeviceRef]

    # Mixture-of-experts.
    num_local_experts: int
    num_experts_per_tok: int
    moe_layers: list[int]

    # iRoPE / chunked attention.
    no_rope_layers: list[int]
    attention_chunk_size: int

    # Attention specials.
    use_qk_norm: bool
    attn_temperature_tuning: bool
    floor_scale: float
    attn_scale: float
    attention_bias: bool
    attention_multiplier: float

    rms_norm_eps: float = 1e-5
    norm_dtype: DType | None = None
    tie_word_embeddings: bool = False
    quant_config: QuantConfig | None = None
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    data_parallel_degree: int = 1
    quantization_encoding: SupportedEncoding | None = None

    @staticmethod
    @override
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        text_config = get_text_config(huggingface_config)
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is not None:
            return int(head_dim)
        return int(text_config.hidden_size // text_config.num_attention_heads)

    @staticmethod
    @override
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return int(get_text_config(huggingface_config).num_hidden_layers)

    @classmethod
    @override
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Bounds ``max_length`` by the text config's ``max_position_embeddings``.

        The multimodal ``Llama4Config`` exposes ``max_position_embeddings`` only
        under ``text_config``; the base implementation reads it off the
        top-level config, so route it through :func:`get_text_config` first.
        """
        return super().calculate_max_seq_len(
            huggingface_config=get_text_config(huggingface_config),
            model_config=model_config,
        )

    @classmethod
    @override
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParams:
        text_config = get_text_config(huggingface_config)
        return kv_cache_config.to_params(
            allow_kv_head_replication=allow_kv_head_replication,
            dtype=cache_dtype,
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            num_layers=cls.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}'"
                ", but config could not be loaded. Please ensure the model "
                "repository contains a valid config.json file."
            )
        return cls.initialize_from_config(
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        text_config = get_text_config(huggingface_config)
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )
        n_devices = len(pipeline_config.model.device_specs)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs[:n_devices]
        ]

        num_hidden_layers = int(text_config.num_hidden_layers)

        # Llama4 always uses the interleaved (complex) RoPE variant, so the
        # llama3-style frequency scaling, when present, applies on top of it.
        # Transformers v5 stores rope settings under ``rope_parameters``; v4
        # uses a flat ``rope_scaling`` dict. Support both shapes.
        rope_scaling_params: Llama3RopeScalingParams | None = None
        rope_scaling = getattr(text_config, "rope_scaling", None) or getattr(
            text_config, "rope_parameters", None
        )
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type") or rope_scaling.get(
                "type"
            )
            if rope_type == "llama3":
                rope_scaling_params = Llama3RopeScalingParams(
                    factor=rope_scaling["factor"],
                    low_freq_factor=rope_scaling["low_freq_factor"],
                    high_freq_factor=rope_scaling["high_freq_factor"],
                    orig_max_position=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )

        head_dim = cls.get_head_dim(huggingface_config)

        return cls(
            hidden_size=text_config.hidden_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            head_dim=head_dim,
            rope_theta=float(get_rope_theta(text_config)),
            rope_scaling_params=rope_scaling_params,
            intermediate_size=text_config.intermediate_size,
            intermediate_size_mlp=text_config.intermediate_size_mlp,
            vocab_size=text_config.vocab_size,
            dtype=dtype,
            model_quantization_encoding=supported_encoding_quantization(
                quantization_encoding
            ),
            quantization_config=gptq_quant_config(
                quantization_encoding,
                pipeline_config.model.huggingface_config,
            ),
            max_seq_len=max_seq_len,
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            devices=device_refs,
            num_local_experts=text_config.num_local_experts,
            num_experts_per_tok=text_config.num_experts_per_tok,
            moe_layers=_default_moe_layers(
                getattr(text_config, "moe_layers", None),
                int(getattr(text_config, "interleave_moe_layer_step", 1)),
                num_hidden_layers,
            ),
            no_rope_layers=_default_no_rope_layers(
                getattr(text_config, "no_rope_layers", None),
                int(getattr(text_config, "no_rope_layer_interval", 4)),
                num_hidden_layers,
            ),
            attention_chunk_size=int(
                getattr(text_config, "attention_chunk_size", 8192)
            ),
            use_qk_norm=bool(getattr(text_config, "use_qk_norm", True)),
            attn_temperature_tuning=bool(
                getattr(text_config, "attn_temperature_tuning", True)
            ),
            floor_scale=float(getattr(text_config, "floor_scale", 8192)),
            attn_scale=float(getattr(text_config, "attn_scale", 0.1)),
            attention_bias=bool(getattr(text_config, "attention_bias", False)),
            attention_multiplier=math.sqrt(1.0 / float(head_dim)),
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        """Sets parameters that depend on the loaded state dict."""
        text_config = get_text_config(huggingface_config)

        def _strip(name: str) -> str:
            return name.removeprefix("language_model.").removeprefix("model.")

        normalized_state_dict = {
            _strip(k): v
            for k, v in state_dict.items()
            if not (k.startswith(("vision_model.", "multi_modal_projector.")))
        }

        self.quant_config = _build_llama4_fp8_quant_config(
            huggingface_config, self.dtype
        )

        norm_dtype = None
        if "layers.0.input_layernorm.weight" in normalized_state_dict:
            norm_dtype = normalized_state_dict[
                "layers.0.input_layernorm.weight"
            ].dtype
        self.norm_dtype = norm_dtype

        if "tie_word_embeddings" in huggingface_config:
            self.tie_word_embeddings = huggingface_config.tie_word_embeddings
        else:
            self.tie_word_embeddings = (
                getattr(text_config, "tie_word_embeddings", False)
                or "lm_head.weight" not in normalized_state_dict
            )

        self.rms_norm_eps = float(getattr(text_config, "rms_norm_eps", 1e-5))
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
