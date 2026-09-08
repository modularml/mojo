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

"""Regression tests for per-batch KV connector metric reset (MXSERV-203)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    CompletedTransfer,
    KVConnectorTransfer,
    TransferDirection,
)
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager


class _CountingConnector:
    """Minimal connector stub with mutable transfer counters."""

    def __init__(self) -> None:
        self._h2d_bytes_copied = 0
        self._d2h_bytes_copied = 0
        self._disk_bytes_written = 0
        self._disk_bytes_read = 0
        # dKV health that mirrors the real connector, so connected and total
        # are a live level and reconnect_attempts is a lifetime counter that
        # reset_metrics deliberately does not clear.
        self._connected_clients = 0
        self._total_clients = 0
        self._reconnect_attempts = 0

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return {"full": KVCacheGroupId.full()}

    @property
    def name(self) -> str:
        return "counting"

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVConnectorTransfer:
        return CompletedTransfer.load()

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        return CompletedTransfer(TransferDirection.OFFLOAD)

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        return (0, 0)

    def touch(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> None: ...

    def wait_for_loads(self) -> None: ...
    def wait_for_offloads(self) -> None: ...
    def shutdown(self) -> None: ...
    def reset_prefix_cache(self) -> None: ...

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(free=0, total=0)

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(free=0, total=0)

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics(
            h2d_bytes_copied=self._h2d_bytes_copied,
            d2h_bytes_copied=self._d2h_bytes_copied,
            disk_bytes_written=self._disk_bytes_written,
            disk_bytes_read=self._disk_bytes_read,
            dkv_connected_clients=self._connected_clients,
            dkv_total_clients=self._total_clients,
            dkv_reconnect_attempts=self._reconnect_attempts,
        )

    def reset_metrics(self) -> None:
        # Only the per-batch transfer counters reset. The health fields are a
        # live level and a lifetime counter, so they are intentionally left
        # alone here, mirroring the real dKV connector.
        self._h2d_bytes_copied = 0
        self._d2h_bytes_copied = 0
        self._disk_bytes_written = 0
        self._disk_bytes_read = 0


def test_block_manager_reset_metrics_clears_connector_counters() -> None:
    connector = _CountingConnector()
    connector._d2h_bytes_copied = 5
    connector._h2d_bytes_copied = 2
    bm = BlockManager(
        total_num_blocks=64,
        block_size=16,
        enable_prefix_caching=True,
        connector=connector,
    )

    assert bm.metrics.d2h_bytes_copied == 5
    assert bm.metrics.h2d_bytes_copied == 2

    bm.reset_metrics()

    assert bm.metrics.d2h_bytes_copied == 0
    assert bm.metrics.h2d_bytes_copied == 0

    connector._d2h_bytes_copied = 3
    assert bm.metrics.d2h_bytes_copied == 3


def test_scheduler_sampling_cycle_reports_per_batch_deltas() -> None:
    """Models BatchMetrics.create(): sample aggregated metrics, then reset.

    Before MXSERV-203, batch 2 would report 8 (5+3 cumulative) because
    connector counters were never reset. Telemetry must emit 5 then 3.
    """
    connector = _CountingConnector()
    bm = BlockManager(
        total_num_blocks=64,
        block_size=16,
        enable_prefix_caching=True,
        connector=connector,
    )

    def sample_and_reset(d2h_delta: int, h2d_delta: int) -> KVCacheMetrics:
        connector._d2h_bytes_copied += d2h_delta
        connector._h2d_bytes_copied += h2d_delta
        sampled = bm.metrics
        bm.reset_metrics()
        return sampled

    batch_one = sample_and_reset(d2h_delta=5, h2d_delta=2)
    batch_two = sample_and_reset(d2h_delta=3, h2d_delta=0)

    assert batch_one.d2h_bytes_copied == 5
    assert batch_one.h2d_bytes_copied == 2
    assert batch_two.d2h_bytes_copied == 3
    assert batch_two.h2d_bytes_copied == 0

    # OTEL counter.add() with these per-batch samples totals 8, not 23.
    otel_counter_total = batch_one.d2h_bytes_copied + batch_two.d2h_bytes_copied
    assert otel_counter_total == 8
    assert otel_counter_total != 5 + 8  # pre-fix cumulative double-count


def test_null_connector_reset_metrics_is_noop() -> None:
    NullConnector().reset_metrics()


def test_kv_cache_metrics_add_sums_dkv_health_fields() -> None:
    """The dKV health fields sum across per-replica connector clients."""
    connected = KVCacheMetrics(
        dkv_connected_clients=1,
        dkv_total_clients=1,
        dkv_reconnect_attempts=2,
    )
    down = KVCacheMetrics(
        dkv_connected_clients=0,
        dkv_total_clients=1,
        dkv_reconnect_attempts=5,
    )

    total = connected + down

    assert total.dkv_connected_clients == 1
    assert total.dkv_total_clients == 2
    assert total.dkv_reconnect_attempts == 7
    # one of two clients down, so the aggregate is degraded
    assert total.dkv_degraded


def test_kv_cache_metrics_add_sums_tier_attribution_fields() -> None:
    """device_blocks_served and cross_replica_{blocks,bytes}_copied sum
    across replicas (CENG-845 G0/G0-DP tier-attribution counters)."""
    replica_0 = KVCacheMetrics(
        device_blocks_served=5,
        cross_replica_blocks_copied=2,
        cross_replica_bytes_copied=1024,
        h2d_bytes_copied=10,
        disk_bytes_read=3,
    )
    replica_1 = KVCacheMetrics(
        device_blocks_served=7,
        cross_replica_blocks_copied=1,
        cross_replica_bytes_copied=512,
        h2d_bytes_copied=4,
        disk_bytes_read=1,
    )

    total = replica_0 + replica_1

    assert total.device_blocks_served == 12
    assert total.cross_replica_blocks_copied == 3
    assert total.cross_replica_bytes_copied == 1536
    assert total.h2d_bytes_copied == 14
    assert total.disk_bytes_read == 4
    # defaults stay zero
    assert KVCacheMetrics().device_blocks_served == 0
    assert KVCacheMetrics().cross_replica_bytes_copied == 0


def test_kv_cache_metrics_dkv_degraded_predicate() -> None:
    """dkv_degraded is true only with a dKV tier and a client not connected."""
    # no dKV tier attached
    assert not KVCacheMetrics().dkv_degraded
    # every client connected
    assert not KVCacheMetrics(
        dkv_connected_clients=2, dkv_total_clients=2
    ).dkv_degraded
    # a client down
    assert KVCacheMetrics(
        dkv_connected_clients=1, dkv_total_clients=2
    ).dkv_degraded


def test_dkv_health_persists_across_sample_and_reset() -> None:
    """dKV health is a level and a cumulative value, not a per-batch delta.

    The connector reports connected and reconnect_attempts live and does not
    clear them on reset_metrics, so every batch reads the current level and the
    running cumulative total. This is why they export as gauges rather than
    counters, because a counter fed the cumulative value each batch would
    double-count.
    """
    connector = _CountingConnector()
    connector._connected_clients = 1
    connector._total_clients = 2
    connector._reconnect_attempts = 3
    bm = BlockManager(
        total_num_blocks=64,
        block_size=16,
        enable_prefix_caching=True,
        connector=connector,
    )

    batch_one = bm.metrics
    bm.reset_metrics()
    batch_two = bm.metrics

    # reset_metrics did not clear the health fields, so both samples agree
    for batch in (batch_one, batch_two):
        assert batch.dkv_connected_clients == 1
        assert batch.dkv_total_clients == 2
        assert batch.dkv_reconnect_attempts == 3
        assert batch.dkv_degraded

    # a later reconnect bumps the running total, and the next sample reflects
    # the new cumulative value rather than a per-batch delta of 1
    connector._reconnect_attempts = 4
    assert bm.metrics.dkv_reconnect_attempts == 4
