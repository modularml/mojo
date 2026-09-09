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

"""CPU-only checks for Qwen3-VL MoE attention dims under tensor-parallel sharding."""

from __future__ import annotations

from unittest.mock import MagicMock

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.linear import Linear
from max.pipelines.architectures.qwen3vl_moe.nn.text_attention import (
    Qwen3VLMoEDecoderAttentionWithRope,
)


def test_attention_stores_per_module_kv_heads_distinct_from_kv_cache_global() -> (
    None
):
    """Tensor-parallel shards must use local num_key_value_heads for QKV width.

    `KVCacheParams.n_kv_heads` stays global for the cache; StackedLinear output
    widths follow the per-shard head counts passed into the module (see MODELS-1366).
    """
    head_dim = 128
    global_kv_heads = 8
    local_kv_heads = 2
    local_q_heads = 8
    hidden_size = 1024

    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=global_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        devices=(DeviceRef.CPU(),),
    )

    attn = Qwen3VLMoEDecoderAttentionWithRope(
        rope=MagicMock(),
        num_attention_heads=local_q_heads,
        num_key_value_heads=local_kv_heads,
        hidden_size=hidden_size,
        kv_params=kv_params,
        devices=(DeviceRef.CPU(),),
        dtype=DType.bfloat16,
        linear_cls=Linear,
        has_bias=False,
    )

    assert attn.num_key_value_heads == local_kv_heads
    assert isinstance(attn.kv_params, MHAKVCacheParams)
    assert attn.kv_params.n_kv_heads == global_kv_heads

    q_w, k_w, v_w = attn.qkv_proj._out_dims
    assert q_w == head_dim * local_q_heads
    assert k_w == head_dim * local_kv_heads
    assert v_w == head_dim * local_kv_heads
    assert k_w != head_dim * global_kv_heads
