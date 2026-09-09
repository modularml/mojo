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

"""Tests for AttentionWithRope in max.nn.attention."""

from __future__ import annotations

from typing import cast
from unittest import mock

import pytest
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Graph, ops
from max.nn.attention import (
    TensorParallelAttentionWithRope,
)
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from max.nn.rotary_embedding import RotaryEmbedding


def create_kv_params(n_kv_heads: int = 8) -> KVCacheParams:
    return MHAKVCacheParams(
        n_kv_heads=n_kv_heads,
        head_dim=64,
        num_layers=1,
        page_size=128,
        dtype=DType.float32,
        devices=[DeviceRef.GPU()],
    )


def test_distributed_attention_with_rope_device_validation() -> None:
    """Tests that TensorParallelAttentionWithRope raises ValueError for CPU."""
    rope = RotaryEmbedding(
        dim=64,
        n_heads=32,
        theta=10000.0,
        max_seq_len=2048,
    )

    kv_params = create_kv_params()

    # Test that CPU devices raises ValueError.
    with pytest.raises(
        ValueError, match="TensorParallelAttentionWithRope does not support CPU"
    ):
        TensorParallelAttentionWithRope(
            rope=rope,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=2048,
            kv_params=kv_params,
            devices=[DeviceRef.CPU(), DeviceRef.CPU()],
        )


@mock.patch("max.nn.attention.attention_with_rope.Allreduce")
def test_distributed_attention_with_rope_call_validation(
    allreduce_mock: mock.Mock,
) -> None:
    """Tests input validation in TensorParallelAttentionWithRope.__call__."""
    rope = RotaryEmbedding(
        dim=64,
        n_heads=32,
        theta=10000.0,
        max_seq_len=2048,
    )

    kv_params = create_kv_params()

    devices = [DeviceRef("gpu", i) for i in range(2)]
    dist_attn = TensorParallelAttentionWithRope(
        rope=rope,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2048,
        kv_params=kv_params,
        devices=devices,
    )

    # Dummy inputs for __call__
    with Graph(name="test_graph"):
        layer_idx = ops.constant(0, dtype=DType.int32, device=DeviceRef.CPU())
        dummy_tensor = ops.constant(
            0.0, dtype=DType.float32, device=DeviceRef.CPU()
        )
        x = [dummy_tensor, dummy_tensor]
        signal_buffers = [mock.Mock(spec=BufferValue) for _ in devices]
        kv_collections = [mock.Mock(spec=PagedCacheValues) for _ in devices]

        # Test wrong number of input_row_offsets
        with pytest.raises(
            ValueError, match="Expected 2 input_row_offsets, got 1"
        ):
            dist_attn(
                layer_idx,
                x,
                cast(list[BufferValue], signal_buffers),
                cast(
                    list[PagedCacheValues],
                    kv_collections,
                ),
                freqs_cis=[rope.freqs_cis, rope.freqs_cis],
                input_row_offsets=[dummy_tensor],
            )

        # Test wrong type in input_row_offsets
        with pytest.raises(
            TypeError,
            match="All elements in input_row_offsets must be TensorValue instances",
        ):
            dist_attn(
                layer_idx,
                x,
                cast(list[BufferValue], signal_buffers),
                cast(list[PagedCacheValues], kv_collections),
                freqs_cis=[rope.freqs_cis, rope.freqs_cis],
                input_row_offsets=[dummy_tensor, "not-a-tensor-value"],  # type: ignore
            )


@mock.patch("max.nn.attention.attention_with_rope.Allreduce")
def test_distributed_attention_with_rope_non_divisible_heads(
    allreduce_mock: mock.Mock,
) -> None:
    """Tests TensorParallelAttentionWithRope with non-divisible number of heads."""
    rope = RotaryEmbedding(
        dim=64,
        n_heads=30,  # Not divisible by 4
        theta=10000.0,
        max_seq_len=2048,
    )

    kv_params = create_kv_params()

    devices = [DeviceRef("gpu", i) for i in range(4)]

    # Should not raise an error anymore
    dist_attn = TensorParallelAttentionWithRope(
        rope=rope,
        num_attention_heads=30,
        num_key_value_heads=8,
        hidden_size=1920,  # 30 * 64
        kv_params=kv_params,
        devices=devices,
    )

    # Verify that heads are distributed correctly
    # Expected distribution: 8, 8, 7, 7
    assert len(dist_attn.list_of_attentions) == 4
    assert dist_attn.list_of_attentions[0].n_heads == 8
    assert dist_attn.list_of_attentions[1].n_heads == 8
    assert dist_attn.list_of_attentions[2].n_heads == 7
    assert dist_attn.list_of_attentions[3].n_heads == 7


@mock.patch("max.nn.attention.attention_with_rope.Allreduce")
def test_distributed_attention_with_rope_stacked_qkv(
    allreduce_mock: mock.Mock,
) -> None:
    """Tests TensorParallelAttentionWithRope with stacked QKV configuration."""
    rope = RotaryEmbedding(
        dim=64,
        n_heads=32,
        theta=10000.0,
        max_seq_len=2048,
    )

    kv_params = create_kv_params()

    devices = [DeviceRef("gpu", i) for i in range(4)]

    # Test with stacked QKV
    dist_attn = TensorParallelAttentionWithRope(
        rope=rope,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2048,
        kv_params=kv_params,
        devices=devices,
        stacked_qkv=True,
    )

    # Verify sharding strategies are set correctly
    assert dist_attn.qkv_proj.sharding_strategy is not None
    assert dist_attn.o_proj.sharding_strategy is not None
    assert dist_attn.qkv_proj.sharding_strategy.is_stacked_qkv
    assert dist_attn.o_proj.sharding_strategy.is_head_aware_colwise


@mock.patch("max.nn.attention.attention_with_rope.Allreduce")
def test_distributed_attention_with_rope_stacked_qkv_non_divisible(
    allreduce_mock: mock.Mock,
) -> None:
    """Tests TensorParallelAttentionWithRope with stacked QKV and non-divisible heads."""
    rope = RotaryEmbedding(
        dim=64,
        n_heads=30,  # Not divisible by 4
        theta=10000.0,
        max_seq_len=2048,
    )

    kv_params = create_kv_params(n_kv_heads=10)  # Not divisible by 4

    devices = [DeviceRef("gpu", i) for i in range(4)]

    # Should work with stacked QKV even with non-divisible heads
    dist_attn = TensorParallelAttentionWithRope(
        rope=rope,
        num_attention_heads=30,
        num_key_value_heads=10,
        hidden_size=1920,  # 30 * 64
        kv_params=kv_params,
        devices=devices,
        stacked_qkv=True,
    )

    # Verify sharding strategies
    assert dist_attn.qkv_proj.sharding_strategy is not None
    assert dist_attn.o_proj.sharding_strategy is not None
    assert dist_attn.qkv_proj.sharding_strategy.is_stacked_qkv
    assert dist_attn.o_proj.sharding_strategy.is_head_aware_colwise

    # Verify head distribution
    total_heads = sum(attn.n_heads for attn in dist_attn.list_of_attentions)
    assert total_heads == 30


@mock.patch("max.nn.attention.attention_with_rope.Allreduce")
def test_distributed_attention_with_rope_separate_projections(
    allreduce_mock: mock.Mock,
) -> None:
    """Tests TensorParallelAttentionWithRope with separate Q, K, V projections."""
    rope = RotaryEmbedding(
        dim=64,
        n_heads=30,  # Not divisible by 4
        theta=10000.0,
        max_seq_len=2048,
    )

    kv_params = create_kv_params(n_kv_heads=10)  # Not divisible by 4

    devices = [DeviceRef("gpu", i) for i in range(4)]

    # Test with separate projections (stacked_qkv=False)
    dist_attn = TensorParallelAttentionWithRope(
        rope=rope,
        num_attention_heads=30,
        num_key_value_heads=10,
        hidden_size=1920,
        kv_params=kv_params,
        devices=devices,
        stacked_qkv=False,
    )

    # Verify sharding strategies for separate projections.
    q_ss = dist_attn.qkv_proj._child("q_proj").sharding_strategy
    k_ss = dist_attn.qkv_proj._child("k_proj").sharding_strategy
    v_ss = dist_attn.qkv_proj._child("v_proj").sharding_strategy
    o_ss = dist_attn.o_proj.sharding_strategy
    assert q_ss is not None
    assert k_ss is not None
    assert v_ss is not None
    assert o_ss is not None
    assert q_ss.is_rowwise
    assert k_ss.is_rowwise
    assert v_ss.is_rowwise
    assert o_ss.is_head_aware_colwise
