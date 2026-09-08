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

"""Null connector implementation for KV cache.

Provides a no-op connector for use when external caching is disabled.
All operations are no-ops that return immediately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.kv_connector import (
    CompletedTransfer,
    KVConnector,
    KVConnectorTransfer,
)


class NullConnector(KVConnector):
    """No-op connector for when external caching is disabled."""

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return {"full": KVCacheGroupId.full()}

    @property
    def name(self) -> str:
        return "NullConnector"

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
        return CompletedTransfer.offload()

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()
