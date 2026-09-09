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

"""KV cache management for MAX pipelines."""

from max.nn.kv_cache import KVCacheGroupId

from .config import (
    KVCacheConfig,
    KVConnectorConfig,
    cache_dtype_for_encoding,
)
from .memory_planner import (
    MemoryPlanner,
    ModelConfig,
    ModelConfigWithKVCache,
    PagedMemoryPlanner,
)
from .paged_kv_cache import (
    BlockCount,
    ByteCount,
    DummyKVCache,
    InsufficientBlocksError,
    JengaKVCacheManager,
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
    PagedKVCacheManagerInterface,
    TransferReqData,
    available_port,
)
from .registry import load_kv_manager

__all__ = [
    "BlockCount",
    "ByteCount",
    "DummyKVCache",
    "InsufficientBlocksError",
    "JengaKVCacheManager",
    "KVCacheConfig",
    "KVConnectorConfig",
    "KVTransferEngine",
    "KVTransferEngineMetadata",
    "MemoryPlanner",
    "ModelConfig",
    "ModelConfigWithKVCache",
    "PagedKVCacheManager",
    "PagedKVCacheManagerInterface",
    "PagedMemoryPlanner",
    "TransferReqData",
    "available_port",
    "cache_dtype_for_encoding",
    "load_kv_manager",
]
