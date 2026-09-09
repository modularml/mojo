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
"""Tests for DeepseekV3.2 model instantiation."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef
from max.nn import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.deepseekV3_2.deepseekV3_2 import DeepseekV3_2
from max.pipelines.architectures.deepseekV3_2.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from transformers import DeepseekV32Config


def make_test_huggingface_config() -> DeepseekV32Config:
    """Create a minimal HuggingFace config for testing DeepSeekV3.2."""
    config_dict = {
        "architectures": ["DeepseekV32ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "ep_size": 1,
        "first_k_dense_replace": 10,
        "hidden_act": "silu",
        "hidden_size": 512,
        "index_head_dim": 128,
        "index_n_heads": 4,
        "index_topk": 64,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "kv_lora_rank": 128,
        "max_position_embeddings": 2048,
        "model_type": "deepseek_v32",
        "moe_intermediate_size": 256,
        "moe_layer_freq": 1,
        "n_group": 2,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "num_nextn_predict_layers": 1,
        "q_lora_rank": 128,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        "rope_theta": 10000,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 2,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "v_head_dim": 64,
        "vocab_size": 1024,
    }
    return DeepseekV32Config.from_dict(config_dict)


def make_test_config() -> DeepseekV3_2Config:
    """Create a minimal DeepseekV3_2Config for testing."""
    hf_config = make_test_huggingface_config()
    device = DeviceRef.CPU()

    mla_kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.v_head_dim,
        num_layers=hf_config.num_hidden_layers,
        devices=[device],
    )
    indexer_kv_params = MHAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=1,
        head_dim=hf_config.index_head_dim,
        num_layers=hf_config.num_hidden_layers,
        devices=[device],
    )
    kv_params = MultiKVCacheParams.from_params(
        {"mla": mla_kv_params, "indexer": indexer_kv_params}
    )

    return DeepseekV3_2Config(
        dtype=DType.float8_e4m3fn,
        kv_params=kv_params,
        devices=[device],
        quant_config=QuantConfig(
            weight_scale=WeightScaleSpec(
                dtype=DType.float32,
                granularity=ScaleGranularity.BLOCK,
                block_size=(128, 128),
            ),
            input_scale=InputScaleSpec(
                dtype=DType.float32,
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                block_size=(1, 128),
            ),
            mlp_quantized_layers=set(),
            attn_quantized_layers=set(),
            format=QuantFormat.BLOCKSCALED_FP8,
            embedding_output_dtype=None,
        ),
        use_subgraphs=False,
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        moe_intermediate_size=hf_config.moe_intermediate_size,
        moe_layer_freq=hf_config.moe_layer_freq,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        n_shared_experts=hf_config.n_shared_experts,
        n_routed_experts=hf_config.n_routed_experts,
        routed_scaling_factor=hf_config.routed_scaling_factor,
        kv_lora_rank=hf_config.kv_lora_rank,
        q_lora_rank=hf_config.q_lora_rank,
        qk_rope_head_dim=hf_config.qk_rope_head_dim,
        v_head_dim=hf_config.v_head_dim,
        qk_nope_head_dim=hf_config.qk_nope_head_dim,
        topk_method=hf_config.topk_method,
        n_group=hf_config.n_group,
        topk_group=hf_config.topk_group,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        first_k_dense_replace=hf_config.first_k_dense_replace,
        norm_topk_prob=hf_config.norm_topk_prob,
        hidden_act=hf_config.hidden_act,
        max_position_embeddings=hf_config.max_position_embeddings,
        max_seq_len=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        rope_theta=get_rope_theta(hf_config),
        rope_scaling=hf_config.rope_scaling,
        scoring_func=hf_config.scoring_func,
        attention_bias=hf_config.attention_bias,
        attention_dropout=hf_config.attention_dropout,
        # DeepseekV3.2 specific fields
        index_head_dim=hf_config.index_head_dim,
        index_n_heads=hf_config.index_n_heads,
        index_topk=hf_config.index_topk,
        norm_dtype=DType.bfloat16,
        correction_bias_dtype=DType.float32,
        max_batch_context_length=2048,
        graph_mode="auto",
        data_parallel_degree=1,
    )


def test_deepseekv3_2_model_instantiation() -> None:
    """Test that DeepseekV3_2 model can be instantiated with a small config."""
    config = make_test_config()
    model = DeepseekV3_2(config)

    assert model.config == config
    assert len(model.layers) == config.num_hidden_layers
    assert model.embed_tokens is not None
    assert model.norm is not None
    assert model.lm_head is not None
