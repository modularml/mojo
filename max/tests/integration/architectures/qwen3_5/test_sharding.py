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

"""Tensor-parallel sharding tests for Qwen3.5's two mixers.

Both mixers pack several logical blocks into one weight, and in both cases a
split that ignores the packing produces the right *shapes* and the wrong
numbers. Shape assertions therefore cannot gate this; the declared segments
are the thing under test.

- ``q_proj`` stores each head as one ``[query | gate]`` block of
  ``2 * head_dim`` consecutive rows. Splitting it as ``[all queries | all
  gates]`` compiles and pairs each head's attention output with a different
  head's gate.
- ``in_proj_qkv`` and the depthwise ``conv1d`` share the conv channel layout
  ``[Q | K | V]``, each block head-major, and the recurrence maps value head
  ``v`` onto key head ``v // (num_value_heads // num_key_heads)``. A flat
  split of ``conv_dim`` hands a device K channels belonging to another
  device's value heads.
"""

from __future__ import annotations

from functools import partial

import pytest
from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, Weight
from max.graph.weight import Segment
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.linear import Linear
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.stacked_linear import StackedLinear
from max.pipelines.architectures.qwen3_5.layers.attention import (
    Qwen3_5Attention,
)
from max.pipelines.architectures.qwen3_5.layers.gated_deltanet import (
    GatedDeltaNet,
)

# Qwen3.8-27B's real geometry: 24 query heads over 4 KV heads at head dim
# 256, and 48 value heads over 16 key heads at head dim 128.
NUM_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
NUM_VALUE_HEADS = 48
NUM_KEY_HEADS = 16
SSM_HEAD_DIM = 128
HIDDEN_SIZE = 5120

DEVICES = [DeviceRef.GPU(0), DeviceRef.GPU(1)]


def _child(stacked: StackedLinear, name: str) -> Linear:
    """Returns one projection of an unfused stack, which holds them as
    dynamically named attributes."""
    child = getattr(stacked, name)
    assert isinstance(child, Linear)
    return child


def _strategy(shardable: Linear | Weight) -> ShardingStrategy:
    strategy = shardable.sharding_strategy
    assert strategy is not None
    return strategy


def _segments(shardable: Linear | Weight) -> tuple[Segment, ...]:
    """Returns the segments a segmented strategy declares.

    Reaching into the bound arguments is the point: the packing a segment
    list describes is exactly what a wrong split gets wrong, and every
    observable shape survives the mistake unchanged.
    """
    strategy = _strategy(shardable)
    assert strategy.is_segmented
    assert isinstance(strategy.shard, partial)
    return tuple(strategy.shard.keywords["segments"])


def _attention() -> Qwen3_5Attention:
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=HEAD_DIM,
        devices=[DEVICES[0]],
    )
    rope = Llama3RotaryEmbedding(
        dim=HIDDEN_SIZE,
        n_heads=NUM_HEADS,
        theta=1e7,
        max_seq_len=4096,
        head_dim=int(HEAD_DIM * 0.25),
        interleaved=True,
    )
    return Qwen3_5Attention(
        rope=rope,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        hidden_size=HIDDEN_SIZE,
        head_dim=HEAD_DIM,
        kv_params=kv_params,
        layer_idx=0,
        dtype=DType.bfloat16,
        devices=[DEVICES[0]],
    )


def _gated_deltanet() -> GatedDeltaNet:
    return GatedDeltaNet(
        hidden_size=HIDDEN_SIZE,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        key_head_dim=SSM_HEAD_DIM,
        value_head_dim=SSM_HEAD_DIM,
        conv_kernel_size=4,
        dtype=DType.bfloat16,
        device=DEVICES[0],
    )


def test_attention_q_proj_splits_by_head_not_by_query_gate_halves() -> None:
    attention = _attention()
    attention.sharding_strategy = ShardingStrategy.tensor_parallel(2)

    # One head-aware segment of `2 * head_dim` slots: the query and its gate
    # travel together. `(Segment.even(6144), Segment.even(6144))` — the
    # `[all queries | all gates]` reading — would pass every shape check.
    assert _segments(_child(attention.qkv_proj, "q_proj")) == (
        Segment.head_aware(NUM_HEADS, HEAD_DIM * 2),
    )
    for name in ("k_proj", "v_proj"):
        assert _segments(_child(attention.qkv_proj, name)) == (
            Segment.head_aware(NUM_KV_HEADS, HEAD_DIM),
        )

    # Row-parallel output projection, split by query head, plus replicated
    # per-head-dim norm gammas.
    assert _strategy(attention.o_proj).is_head_aware_colwise
    assert _strategy(attention.q_norm.weight).is_replicate
    assert _strategy(attention.k_norm.weight).is_replicate


def test_attention_shards_carry_half_the_heads() -> None:
    attention = _attention()
    attention.sharding_strategy = ShardingStrategy.tensor_parallel(2)
    shards = attention.shard(DEVICES)

    assert len(shards) == 2
    for shard, device in zip(shards, DEVICES, strict=True):
        assert shard.n_heads == NUM_HEADS // 2
        assert shard.num_key_value_heads == NUM_KV_HEADS // 2
        assert shard.head_dim == HEAD_DIM
        assert shard.devices == [device]


def test_gated_deltanet_splits_conv_channels_at_head_boundaries() -> None:
    gdn = _gated_deltanet()
    gdn.sharding_strategy = ShardingStrategy.tensor_parallel(2)

    key_segment = Segment.head_aware(NUM_KEY_HEADS, SSM_HEAD_DIM)
    expected = (
        key_segment,
        key_segment,
        Segment.head_aware(NUM_VALUE_HEADS, SSM_HEAD_DIM),
    )
    # The projection and the depthwise conv index the same channels, so they
    # must be cut identically.
    assert _segments(_child(gdn.in_proj, "in_proj_qkv")) == expected
    assert _segments(gdn.conv1d) == expected

    # Everything else is indexed by value head alone.
    for weight in (gdn.dt_bias, gdn.A_log):
        assert _strategy(weight).is_rowwise
    for name in ("in_proj_z", "in_proj_b", "in_proj_a"):
        assert _strategy(_child(gdn.in_proj, name)).is_rowwise
    assert _strategy(gdn.norm.weight).is_replicate
    assert _strategy(gdn.out_proj).is_head_aware_colwise


def test_gated_deltanet_shards_preserve_the_key_to_value_head_ratio() -> None:
    gdn = _gated_deltanet()
    gdn.sharding_strategy = ShardingStrategy.tensor_parallel(2)
    shards = gdn.shard(DEVICES)

    assert len(shards) == 2
    for shard in shards:
        assert shard.num_value_heads == NUM_VALUE_HEADS // 2
        assert shard.num_key_heads == NUM_KEY_HEADS // 2
        # The recurrence kernel derives `key_head_idx = value_head_idx //
        # (nv // nk)` at runtime, so the shard is only correct on the
        # unmodified kernel while the ratio is unchanged.
        assert (
            shard.num_value_heads // shard.num_key_heads
            == NUM_VALUE_HEADS // NUM_KEY_HEADS
        )
        assert shard.conv_dim == (2 * 2048 + 6144) // 2


@pytest.mark.parametrize("num_devices", [5, 7])
def test_indivisible_head_counts_are_rejected(num_devices: int) -> None:
    """Neither mixer may silently round a head split."""
    with pytest.raises(ValueError, match="divisible"):
        _attention().sharding_strategy = ShardingStrategy.tensor_parallel(
            num_devices
        )
    with pytest.raises(ValueError, match="divisible"):
        _gated_deltanet().sharding_strategy = ShardingStrategy.tensor_parallel(
            num_devices
        )


def test_data_parallel_sharding_is_refused() -> None:
    """Both mixers hold per-device state that a replicate strategy would
    duplicate without splitting; ``construct_kv_params`` refuses DP too."""
    with pytest.raises(ValueError, match="tensor-parallel"):
        _attention().sharding_strategy = ShardingStrategy.replicate(2)
    with pytest.raises(ValueError, match="tensor-parallel"):
        _gated_deltanet().sharding_strategy = ShardingStrategy.replicate(2)
