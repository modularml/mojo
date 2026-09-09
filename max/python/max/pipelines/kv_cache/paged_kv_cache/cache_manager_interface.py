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

"""Common interface shared by paged KV cache manager implementations.

TODO(bez): temporary scaffolding for the Jenga cutover. Once
``JengaKVCacheManager`` replaces ``PagedKVCacheManager`` outright, delete
this interface, drop the ``use_jenga_cache`` flag, and inline everything
back into a single concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from max.driver import Buffer
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheParamInterface,
)
from max.nn.kv_cache.cache_params import KVCacheBufferInterface
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    ByteCount,
    KVConnectorTransfer,
)

from .block_manager import PrefixCacheHits


class PagedKVCacheManagerInterface(ABC):
    """Interface for the KVCache manager.

    This is largely a temporarily class to help phase in the JengaKVCacheManager.
    We want to delete the legacy PagedKVCacheManager and rename JengaKVCacheManager
    to PagedKVCacheManager soon.
    """

    params: KVCacheParamInterface

    @abstractmethod
    def get_prefix_cache_hit_counts(
        self, ctx: TextContext
    ) -> list[PrefixCacheHits]:
        """Counts each replica's contiguous cached prefix for a request."""

    @abstractmethod
    def alloc(self, ctx: TextContext) -> KVConnectorTransfer:
        """Allocates blocks for a request."""

    @abstractmethod
    def runtime_inputs(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Gets the graph inputs for per-replica batches of requests."""

    @abstractmethod
    def runtime_inputs_for_leaf(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Returns :meth:`runtime_inputs` narrowed to a single leaf cache."""

    @abstractmethod
    def alloc_dummy(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Claims a dummy request and maps it to the replica's null block."""

    @abstractmethod
    def block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns the device KV cache block occupancy for the given replica."""

    @abstractmethod
    def host_byte_count(self, replica_idx: int = 0) -> ByteCount:
        """Returns the host KV tier occupancy in bytes for the given replica."""

    @abstractmethod
    def disk_byte_count(self, replica_idx: int = 0) -> ByteCount:
        """Returns the disk KV tier occupancy in bytes for the given replica."""

    @abstractmethod
    def release(self, ctx: TextContext) -> None:
        """Releases the blocks the request holds on the replica it was claimed on."""

    @abstractmethod
    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Pins a request to one replica, which owns it until it is released."""

    @abstractmethod
    def step(self, ctx: TextContext) -> None:
        """Commits the request's newly written tokens into the prefix cache."""

    @abstractmethod
    def poll_transfers(self) -> None:
        """Drains completed async KV transfers (onloads and offloads)."""

    @abstractmethod
    def pending_transfers_exist(self, replica_idx: int = 0) -> bool:
        """Returns whether any async KV transfer is in flight on the replica."""

    @abstractmethod
    def contains(self, ctx: TextContext) -> bool:
        """Returns whether the request is claimed on any replica."""

    @abstractmethod
    def reset_metrics(self) -> None:
        """Resets metrics for the block manager."""

    @abstractmethod
    def reset_prefix_cache(self) -> None:
        """Resets the device prefix caches and every connector's tiers."""

    @abstractmethod
    def shutdown(self) -> None:
        """Releases the KV connector's external resources."""

    @abstractmethod
    def get_metrics_aggregated(self) -> KVCacheMetrics:
        """Returns aggregated metrics across all replicas."""

    @abstractmethod
    def get_req_blocks(self, ctx: TextContext) -> list[int]:
        """Returns block IDs the request holds on the replica it was claimed on."""

    @abstractmethod
    def get_device_buffer(self, replica_idx: int) -> KVCacheBufferInterface:
        """Returns the replica's KV buffer (single leaf or tree)."""

    @property
    @abstractmethod
    def effective_max_seq_length(self) -> int | None:
        """Returns the effective maximum sequence length that can be served by the block manager."""
