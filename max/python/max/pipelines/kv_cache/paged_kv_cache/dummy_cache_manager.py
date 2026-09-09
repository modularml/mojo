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

"""Provides a no-op :class:`DummyKVCache` implementation for testing when KV caching is disabled."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import (
    CompletedTransfer,
    KVConnectorTransfer,
)
from max.pipelines.modeling.types import RequestID

from .cache_manager import BlockCount, ByteCount, PagedKVCacheManager


class DummyKVCache(PagedKVCacheManager):
    """No-op KV cache implementation for testing or when cache is disabled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the dummy cache with a single replica and no host swapping."""
        self.reqs = set[RequestID]()
        self.params = MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            devices=[DeviceRef.CPU()],
        )

    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """No-op."""

    def alloc(self, *args: Any, **kwargs: Any) -> KVConnectorTransfer:
        """No-op; returns an already-complete transfer (nothing to onload)."""
        return CompletedTransfer.load()

    def step(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""

    def contains(self, ctx: TextContext) -> bool:
        """Returns True for any request."""
        return True

    def release(self, ctx: TextContext) -> None:
        """No-op."""

    def block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns a single block; this cache never allocates, so it stays free."""
        return BlockCount(free=1, total=1)

    def host_byte_count(self, replica_idx: int = 0) -> ByteCount:
        """Returns one permanently used byte."""
        return ByteCount(free=0, total=1)

    def disk_byte_count(self, replica_idx: int = 0) -> ByteCount:
        """Returns one permanently used byte."""
        return ByteCount(free=0, total=1)

    def get_metrics_aggregated(self) -> KVCacheMetrics:
        """Returns empty aggregated metrics."""
        return KVCacheMetrics()

    def reset_metrics(self) -> None:
        """No-op."""
