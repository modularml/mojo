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
from dataclasses import dataclass

from typing_extensions import Self


def dkv_tier_degraded(connected_clients: int, total_clients: int) -> bool:
    """Returns whether a dKV tier is present but not fully connected.

    True when at least one dKV client exists and fewer than all of them are
    connected to the external tier, which is the state an operator alerts on for
    a dead or degraded dKV deployment. Shared by KVCacheMetrics.dkv_degraded and
    the scheduler's per-batch log so the two cannot fall out of sync.
    """
    return total_clients > 0 and connected_clients < total_clients


@dataclass
class KVCacheMetrics:
    """Metrics for the KV cache.

    Tracks token usage and transfer statistics for KV cache operations.

    Device (G0) figures are in blocks, the unit the manager allocates in. The
    connector's external host and disk tiers are in bytes: those are byte
    budgets the operator sizes in bytes, their block width need not match the
    device's, and bytes rate directly against PCIe and disk bandwidth.
    """

    input_tokens: int = 0
    """Number of tokens processed as new input (cache misses)."""
    cache_tokens: int = 0
    """Number of tokens retrieved from cache (cache hits)."""
    device_blocks_served: int = 0
    """Number of cache blocks served directly from the local device prefix
    cache, with no host/disk promotion or cross-replica copy needed."""
    h2d_bytes_copied: int = 0
    """Bytes of KV copied from the connector's host tier to device."""
    d2h_bytes_copied: int = 0
    """Bytes of KV copied from device to the connector's host tier."""
    cross_replica_blocks_copied: int = 0
    """Number of cache blocks copied device-to-device across DP replicas."""
    cross_replica_bytes_copied: int = 0
    """Bytes moved by device-to-device copies across DP replicas."""
    disk_bytes_written: int = 0
    """Bytes of KV written to disk."""
    disk_bytes_read: int = 0
    """Bytes of KV read from disk."""
    inflight_disk_ops: int = 0
    """Number of in-flight disk operations."""
    nixl_read_blocks: int = 0
    """Number of cache blocks read via NIXL (dKV GET)."""
    nixl_write_blocks: int = 0
    """Number of cache blocks written via NIXL (dKV PUT)."""

    # dKV latency pairs: total_ms + count sum correctly across DP replicas.
    nixl_read_latency_total_ms: float = 0.0
    """Cumulative NIXL READ transfer latency in milliseconds."""
    nixl_read_latency_count: int = 0
    """Number of NIXL READ transfer completions."""
    nixl_write_latency_total_ms: float = 0.0
    """Cumulative NIXL WRITE transfer latency in milliseconds."""
    nixl_write_latency_count: int = 0
    """Number of NIXL WRITE transfer completions."""
    rpc_acquire_latency_total_ms: float = 0.0
    """Cumulative dKV acquire_blocks RPC latency in milliseconds."""
    rpc_acquire_latency_count: int = 0
    """Number of acquire_blocks RPC calls."""
    rpc_read_latency_total_ms: float = 0.0
    """Cumulative dKV read_blocks RPC latency in milliseconds."""
    rpc_read_latency_count: int = 0
    """Number of read_blocks RPC calls."""
    nixl_read_bytes: int = 0
    """Total bytes transferred via NIXL READ."""
    nixl_write_bytes: int = 0
    """Total bytes transferred via NIXL WRITE."""
    nixl_read_blocks_local: int = 0
    """NIXL reads from co-located (default) block store."""
    nixl_read_blocks_remote: int = 0
    """NIXL reads from non-default (remote) block stores."""

    # dKV external-tier health. These are a level and a lifetime-cumulative
    # counter read live from the connector rather than per-batch transfer
    # deltas, so they export as gauges and reset_metrics does not clear them.
    dkv_connected_clients: int = 0
    """Number of dKV connector clients currently connected to the external tier."""
    dkv_total_clients: int = 0
    """Total number of dKV connector clients, one per data-parallel replica."""
    dkv_reconnect_attempts: int = 0
    """Cumulative dKV reconnect attempts across all clients over the process lifetime."""

    @property
    def prompt_tokens(self) -> int:
        """Total number of prompt tokens (input + cached).

        Returns:
            Sum of input_tokens and cache_tokens.
        """
        return self.input_tokens + self.cache_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Proportion of prompt tokens that were retrieved from cache.

        Returns:
            Ratio of cache_tokens to total prompt_tokens, or 0.0 if no tokens
            were processed.
        """
        if self.prompt_tokens == 0:
            return 0.0
        return self.cache_tokens / self.prompt_tokens

    @property
    def nixl_read_latency_avg_ms(self) -> float:
        """Average NIXL READ transfer latency in milliseconds."""
        if self.nixl_read_latency_count == 0:
            return 0.0
        return self.nixl_read_latency_total_ms / self.nixl_read_latency_count

    @property
    def nixl_write_latency_avg_ms(self) -> float:
        """Average NIXL WRITE transfer latency in milliseconds."""
        if self.nixl_write_latency_count == 0:
            return 0.0
        return self.nixl_write_latency_total_ms / self.nixl_write_latency_count

    @property
    def rpc_acquire_latency_avg_ms(self) -> float:
        """Average dKV acquire_blocks RPC latency in milliseconds."""
        if self.rpc_acquire_latency_count == 0:
            return 0.0
        return (
            self.rpc_acquire_latency_total_ms / self.rpc_acquire_latency_count
        )

    @property
    def rpc_read_latency_avg_ms(self) -> float:
        """Average dKV read_blocks RPC latency in milliseconds."""
        if self.rpc_read_latency_count == 0:
            return 0.0
        return self.rpc_read_latency_total_ms / self.rpc_read_latency_count

    @property
    def nixl_read_gib_per_s(self) -> float:
        """NIXL READ throughput in GiB/s."""
        if self.nixl_read_latency_total_ms <= 0:
            return 0.0
        return (self.nixl_read_bytes / (1 << 30)) / (
            self.nixl_read_latency_total_ms / 1000
        )

    @property
    def nixl_write_gib_per_s(self) -> float:
        """NIXL WRITE throughput in GiB/s."""
        if self.nixl_write_latency_total_ms <= 0:
            return 0.0
        return (self.nixl_write_bytes / (1 << 30)) / (
            self.nixl_write_latency_total_ms / 1000
        )

    @property
    def remote_read_ratio(self) -> float:
        """Fraction of NIXL reads hitting non-default (remote) block stores."""
        total = self.nixl_read_blocks_local + self.nixl_read_blocks_remote
        if total == 0:
            return 0.0
        return self.nixl_read_blocks_remote / total

    @property
    def dkv_degraded(self) -> bool:
        """Whether a dKV tier is present but not every client is connected.

        Delegates to the module-level dkv_tier_degraded so this predicate has a
        single definition shared with the scheduler's per-batch log.
        """
        return dkv_tier_degraded(
            self.dkv_connected_clients, self.dkv_total_clients
        )

    def __add__(self, other: Self) -> Self:
        """Combine two KVCacheMetrics by summing their respective fields.

        Args:
            other: Another KVCacheMetrics instance to add.

        Returns:
            A new KVCacheMetrics instance with summed values.
        """
        return type(self)(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_tokens=self.cache_tokens + other.cache_tokens,
            device_blocks_served=self.device_blocks_served
            + other.device_blocks_served,
            h2d_bytes_copied=self.h2d_bytes_copied + other.h2d_bytes_copied,
            d2h_bytes_copied=self.d2h_bytes_copied + other.d2h_bytes_copied,
            cross_replica_blocks_copied=self.cross_replica_blocks_copied
            + other.cross_replica_blocks_copied,
            cross_replica_bytes_copied=self.cross_replica_bytes_copied
            + other.cross_replica_bytes_copied,
            disk_bytes_written=self.disk_bytes_written
            + other.disk_bytes_written,
            disk_bytes_read=self.disk_bytes_read + other.disk_bytes_read,
            inflight_disk_ops=self.inflight_disk_ops + other.inflight_disk_ops,
            nixl_read_blocks=self.nixl_read_blocks + other.nixl_read_blocks,
            nixl_write_blocks=self.nixl_write_blocks + other.nixl_write_blocks,
            nixl_read_latency_total_ms=self.nixl_read_latency_total_ms
            + other.nixl_read_latency_total_ms,
            nixl_read_latency_count=self.nixl_read_latency_count
            + other.nixl_read_latency_count,
            nixl_write_latency_total_ms=self.nixl_write_latency_total_ms
            + other.nixl_write_latency_total_ms,
            nixl_write_latency_count=self.nixl_write_latency_count
            + other.nixl_write_latency_count,
            rpc_acquire_latency_total_ms=self.rpc_acquire_latency_total_ms
            + other.rpc_acquire_latency_total_ms,
            rpc_acquire_latency_count=self.rpc_acquire_latency_count
            + other.rpc_acquire_latency_count,
            rpc_read_latency_total_ms=self.rpc_read_latency_total_ms
            + other.rpc_read_latency_total_ms,
            rpc_read_latency_count=self.rpc_read_latency_count
            + other.rpc_read_latency_count,
            nixl_read_bytes=self.nixl_read_bytes + other.nixl_read_bytes,
            nixl_write_bytes=self.nixl_write_bytes + other.nixl_write_bytes,
            nixl_read_blocks_local=self.nixl_read_blocks_local
            + other.nixl_read_blocks_local,
            nixl_read_blocks_remote=self.nixl_read_blocks_remote
            + other.nixl_read_blocks_remote,
            dkv_connected_clients=self.dkv_connected_clients
            + other.dkv_connected_clients,
            dkv_total_clients=self.dkv_total_clients + other.dkv_total_clients,
            dkv_reconnect_attempts=self.dkv_reconnect_attempts
            + other.dkv_reconnect_attempts,
        )
