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
"""Tests for attention head distribution and TP sharding."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy
from max.nn.attention import num_heads_for_device
from max.nn.attention.attention_with_rope import AttentionWithRope
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.rotary_embedding import RotaryEmbedding


class TestNumHeadsForDevice:
    """Tests for distributing attention heads across devices."""

    def test_even_split(self) -> None:
        """Heads divide evenly across devices."""
        assert (
            num_heads_for_device(num_heads=8, device_idx=0, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=8, device_idx=1, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=8, device_idx=2, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=8, device_idx=3, num_devices=4) == 2
        )

    def test_remainder_goes_to_earlier_devices(self) -> None:
        """When heads don't divide evenly, earlier devices get one extra."""
        # 7 heads across 4 devices: 2, 2, 2, 1
        assert (
            num_heads_for_device(num_heads=7, device_idx=0, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=7, device_idx=1, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=7, device_idx=2, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=7, device_idx=3, num_devices=4) == 1
        )

    def test_single_device(self) -> None:
        """Single device gets all heads."""
        assert (
            num_heads_for_device(num_heads=16, device_idx=0, num_devices=1)
            == 16
        )

    def test_one_head_per_device(self) -> None:
        """Each device gets exactly one head."""
        assert (
            num_heads_for_device(num_heads=4, device_idx=0, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=4, device_idx=1, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=4, device_idx=2, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=4, device_idx=3, num_devices=4) == 1
        )

    def test_more_devices_than_heads(self) -> None:
        """Excess devices get zero heads."""
        # 2 heads across 4 devices: 1, 1, 0, 0
        assert (
            num_heads_for_device(num_heads=2, device_idx=0, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=2, device_idx=1, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=2, device_idx=2, num_devices=4) == 0
        )
        assert (
            num_heads_for_device(num_heads=2, device_idx=3, num_devices=4) == 0
        )

    @pytest.mark.parametrize("num_heads", [1, 7, 16, 31, 64])
    @pytest.mark.parametrize("num_devices", [1, 2, 3, 4, 8])
    def test_total_across_all_devices_equals_num_heads(
        self, num_heads: int, num_devices: int
    ) -> None:
        """Sum of heads across all devices equals the total."""
        total = sum(
            num_heads_for_device(
                num_heads=num_heads,
                device_idx=i,
                num_devices=num_devices,
            )
            for i in range(num_devices)
        )
        assert total == num_heads


class TestAttentionWithRopeTPShard:
    """Tests that TP sharding correctly splits KV heads (SERVOPT-1164)."""

    @staticmethod
    def _make_attention(
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 4096,
        use_qk_norm: bool = False,
    ) -> AttentionWithRope:
        rope = RotaryEmbedding(
            dim=hidden_size,
            n_heads=n_heads,
            theta=10000.0,
            max_seq_len=2048,
            head_dim=head_dim,
        )
        kv_params = MHAKVCacheParams(
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=DType.bfloat16,
            num_layers=1,
            devices=[DeviceRef.GPU(0)],
        )
        return AttentionWithRope(
            rope=rope,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            dtype=DType.bfloat16,
            use_qk_norm=use_qk_norm,
        )

    @pytest.mark.parametrize("num_devices", [2, 4])
    def test_tp_shard_splits_kv_heads(self, num_devices: int) -> None:
        """Each TP shard gets num_key_value_heads // num_devices."""
        attn = self._make_attention(n_heads=32, n_kv_heads=8, use_qk_norm=True)
        devices = [DeviceRef.GPU(i) for i in range(num_devices)]
        attn.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
        shards = attn.shard(devices)

        for shard in shards:
            assert shard.num_key_value_heads == 8 // num_devices
            assert shard.n_heads == 32 // num_devices

    def test_tp_shard_kv_heads_without_qk_norm(self) -> None:
        """KV heads are sharded correctly even without qk_norm."""
        attn = self._make_attention(n_heads=32, n_kv_heads=8, use_qk_norm=False)
        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        attn.sharding_strategy = ShardingStrategy.tensor_parallel(2)
        shards = attn.shard(devices)

        for shard in shards:
            assert shard.num_key_value_heads == 4
            assert shard.n_heads == 16
