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
"""Config for DeepseekV3 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from max.dtype import DType
from max.experimental.sharding import DeviceMesh
from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.nn.kv_cache import KVCacheParams, spec_decode_cache_slack
from max.nn.quant_config import QuantConfig
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
class DeepseekV3Config(ArchConfigWithKVCache):
    """Configuration for DeepseekV3 models (single-GPU, ModuleV3)."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    }

    # MAX specific fields
    dtype: DType
    kv_params: KVCacheParams
    devices: list[DeviceRef]
    quantization_encoding: SupportedEncoding | None = None

    mesh: DeviceMesh | None = None
    """Device mesh for sharding across multiple devices."""

    ep_config: EPConfig | None = None
    """Expert-parallel configuration, or ``None`` for replicated/TP experts.

    Set in :meth:`DeepseekV3Model.load_model` when ``--ep-size > 1``.
    """

    data_parallel_degree: int = 1
    """Number of data-parallel attention replicas. Selects the decoder layer's
    parallelism mode together with :attr:`ep_config`."""

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    norm_topk_prob: bool = True
    hidden_act: str = "silu"

    max_position_embeddings: int = 4096
    """Maximum positional embeddings as defined by the original model."""
    max_seq_len: int
    """Maximum sequence length as defined by the MAX Engine pipeline
    configuration. Required so that a subclass ``initialize`` that forgets to
    resolve it fails at construction instead of inheriting another model's
    limit."""

    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rope_interleave: bool = True
    scoring_func: str = "sigmoid"
    attention_bias: bool = False
    attention_dropout: float = 0.0

    correction_bias_dtype: DType | None = None
    max_batch_context_length: int = 131072
    graph_mode: str = "auto"  # "auto" | "prefill" | "decode"

    quant_config: QuantConfig | None = None
    """Block-scaled quantization config when the checkpoint is FP8/FP4.

    Set from the state-dict-dependent finalization step in
    :meth:`DeepseekV3Model.load_model`; ``None`` for bf16 checkpoints.
    """

    mla_o_proj_quantized: bool = True
    """Whether the MLA output projection is quantized."""

    dense_mlp_layers_without_quant: frozenset[int] = frozenset()
    """Dense prefix layers (indices ``< first_k_dense_replace``) that skip MLP quant."""

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

        if self.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported yet")

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

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
            num_layers=DeepseekV3Config.get_num_layers(huggingface_config),
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
        """Initializes a DeepseekV3Config instance from pipeline configuration."""
        model_config = model_config or pipeline_config.model
        config = model_config.huggingface_config
        if config is None:
            raise ValueError(
                "HuggingFace config is required for"
                f" '{model_config.model_path}', but config could not be loaded."
                " Please ensure the model repository contains a valid"
                " config.json file."
            )
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        kv_params = cls.construct_kv_params(
            huggingface_config=config,
            pipeline_config=pipeline_config,
            devices=device_refs,
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
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            moe_layer_freq=config.moe_layer_freq,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            routed_scaling_factor=config.routed_scaling_factor,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            num_experts_per_tok=config.num_experts_per_tok,
            first_k_dense_replace=config.first_k_dense_replace,
            norm_topk_prob=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings
            + spec_decode_cache_slack(kv_params),
            max_seq_len=max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=get_rope_theta(config),
            rope_scaling=config.rope_scaling,
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            graph_mode=graph_mode,
            data_parallel_degree=model_config.data_parallel_degree,
            quantization_encoding=quantization_encoding,
        )
