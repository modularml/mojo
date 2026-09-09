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
"""Tests for Eagle3 Kimi K2.5 weight loading and namespacing."""

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV3.model_config import DeepseekV3Config
from max.pipelines.architectures.eagle3_deepseekV3.unified_eagle import (
    Eagle3DeepseekV3Unified,
)


def _kv_params(num_layers: int) -> KVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=16 + 64,  # qk_rope_head_dim + kv_lora_rank
        num_layers=num_layers,
        devices=[DeviceRef.GPU(0)],
        data_parallel_degree=1,
        page_size=1,
    )


def _target_config() -> DeepseekV3Config:
    return DeepseekV3Config(
        dtype=DType.bfloat16,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        moe_layer_freq=1,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=96,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        correction_bias_dtype=DType.bfloat16,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        hidden_act="silu",
        vocab_size=256,
        max_position_embeddings=512,
        max_seq_len=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 64.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
        },
        devices=[DeviceRef.GPU(0)],
        data_parallel_degree=1,
        kv_params=_kv_params(num_layers=2),
        return_logits=ReturnLogits.VARIABLE,
        return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
        eagle_aux_hidden_state_layer_ids=[0, 1],
    )


def _draft_config() -> DeepseekV3Config:
    return DeepseekV3Config(
        dtype=DType.bfloat16,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        moe_layer_freq=1,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=96,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        correction_bias_dtype=DType.bfloat16,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        hidden_act="silu",
        vocab_size=256,
        max_position_embeddings=512,
        max_seq_len=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 64.0,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
        },
        devices=[DeviceRef.GPU(0)],
        data_parallel_degree=1,
        kv_params=_kv_params(num_layers=1),
        return_logits=ReturnLogits.LAST_TOKEN,
        return_hidden_states=ReturnHiddenStates.LAST,
    )


def test_draft_weights_are_independent_from_target() -> None:
    """Draft norm and lm_head must be separate Weight objects from target."""
    model = Eagle3DeepseekV3Unified(_target_config(), _draft_config())

    # Share only embed_tokens (matching production code).
    assert model.draft is not None
    model.draft.embed_tokens = model.target.embed_tokens

    assert model.draft.norm is not model.target.norm
    assert model.draft.lm_head is not model.target.lm_head
    assert model.draft.embed_tokens is model.target.embed_tokens


def test_state_dict_namespacing() -> None:
    """Draft weights must be prefixable without colliding with target."""
    model = Eagle3DeepseekV3Unified(_target_config(), _draft_config())
    assert model.draft is not None
    model.draft.embed_tokens = model.target.embed_tokens

    target_keys = set(model.target.raw_state_dict().keys())
    draft_keys = set(model.draft.raw_state_dict().keys())

    # norm and lm_head exist in both — they would collide without the
    # "draft." prefix added in production code.
    colliding = target_keys & draft_keys
    shared = {k for k in colliding if k.startswith("embed_tokens.")}
    non_shared_collisions = colliding - shared
    assert non_shared_collisions, (
        "Expected colliding keys (norm, lm_head) between draft and target"
    )
    assert any("norm" in k for k in non_shared_collisions)
    assert any("lm_head" in k for k in non_shared_collisions)

    # Draft-only weights (fc, decoder_layer) must not appear in target.
    draft_only = draft_keys - target_keys
    assert any("fc." in k for k in draft_only)
    assert any("decoder_layer." in k for k in draft_only)
