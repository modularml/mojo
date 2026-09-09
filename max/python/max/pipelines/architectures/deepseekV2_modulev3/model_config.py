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
"""Config for DeepseekV2 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class DeepseekV2Config(ArchConfigWithKVCache):
    """Configuration for DeepseekV2 models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    # MAX specific fields
    dtype: DType
    kv_params: KVCacheParams
    devices: list[DeviceRef]
    quantization_encoding: SupportedEncoding | None = None

    # Default values lifted from Transformers DeepseekV2Config
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: int = 0
    n_routed_experts: int = 0
    ep_size: int = 1
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "greedy"
    n_group: int = 0
    topk_group: int = 0
    num_experts_per_tok: int = 0
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    norm_topk_prob: bool = False
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int = 100000
    eos_token_id: int = 100001
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    max_batch_context_length: int = 131072

    graph_mode: str = "auto"  # "auto" | "prefill" | "decode"

    @override
    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    @override
    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        return upper_bounded_default(
            upper_bound=huggingface_config.max_position_embeddings,
            default=model_config.max_length,
        )

    def __post_init__(self) -> None:
        if self.hidden_act != "silu":
            raise ValueError(
                "'silu' is the only hidden_act currently supported"
            )

        rope_type = self.rope_scaling and self.rope_scaling.get(
            "rope_type", self.rope_scaling.get("type")
        )
        if rope_type and rope_type != "yarn":
            raise ValueError(
                "'yarn' is the only rope_scaling type currently supported"
            )

        if self.norm_topk_prob:
            raise ValueError("norm_topk_prob is not supported yet")

        if self.pretraining_tp != 1:
            raise ValueError("Training not supported at this time")

        if self.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported yet")

        if self.pad_token_id is not None:
            raise ValueError("Padding token is not supported yet")

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            # n_kv_heads should always be 1 because we only cache a single latent vector
            # in LatentAttention
            n_kv_heads=1,
            head_dim=huggingface_config.kv_lora_rank
            + huggingface_config.qk_rope_head_dim,
            num_layers=DeepseekV2Config.get_num_layers(huggingface_config),
            devices=devices,
            is_mla=True,
            num_q_heads=huggingface_config.num_attention_heads,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

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
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )
        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        if pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        return cls(
            attention_bias=huggingface_config.attention_bias,
            attention_dropout=huggingface_config.attention_dropout,
            aux_loss_alpha=huggingface_config.aux_loss_alpha,
            bos_token_id=huggingface_config.bos_token_id,
            eos_token_id=huggingface_config.eos_token_id,
            first_k_dense_replace=huggingface_config.first_k_dense_replace,
            hidden_act=huggingface_config.hidden_act,
            hidden_size=huggingface_config.hidden_size,
            initializer_range=huggingface_config.initializer_range,
            intermediate_size=huggingface_config.intermediate_size,
            kv_lora_rank=huggingface_config.kv_lora_rank,
            max_position_embeddings=max_seq_len,
            moe_intermediate_size=huggingface_config.moe_intermediate_size,
            moe_layer_freq=huggingface_config.moe_layer_freq,
            n_group=huggingface_config.n_group,
            n_routed_experts=huggingface_config.n_routed_experts,
            n_shared_experts=huggingface_config.n_shared_experts,
            norm_topk_prob=huggingface_config.norm_topk_prob,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_experts_per_tok=huggingface_config.num_experts_per_tok,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            pretraining_tp=huggingface_config.pretraining_tp,
            q_lora_rank=huggingface_config.q_lora_rank,
            qk_nope_head_dim=huggingface_config.qk_nope_head_dim,
            qk_rope_head_dim=huggingface_config.qk_rope_head_dim,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            rope_scaling=huggingface_config.rope_scaling,
            rope_theta=get_rope_theta(huggingface_config),
            routed_scaling_factor=huggingface_config.routed_scaling_factor,
            scoring_func=huggingface_config.scoring_func,
            seq_aux=huggingface_config.seq_aux,
            tie_word_embeddings=huggingface_config.tie_word_embeddings,
            topk_group=huggingface_config.topk_group,
            topk_method=huggingface_config.topk_method,
            use_cache=huggingface_config.use_cache,
            v_head_dim=huggingface_config.v_head_dim,
            vocab_size=huggingface_config.vocab_size,
            dtype=supported_encoding_dtype(quantization_encoding),
            kv_params=kv_params,
            devices=devices,
            graph_mode=graph_mode,
            quantization_encoding=quantization_encoding,
        )
