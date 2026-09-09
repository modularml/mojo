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

from __future__ import annotations

from collections.abc import Sequence

from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import KVCacheMemory


def view_2d_uint8(buf: Buffer, total_num_pages: int) -> Buffer:
    """Views a raw buffer as a 2-D uint8 ``[total_num_pages, bytes_per_page]`` array."""
    bytes_per_page = (
        buf.num_elements * buf.dtype.size_in_bytes // total_num_pages
    )
    return buf.view(DType.uint8, [total_num_pages, bytes_per_page])


def kv_memory(buf: Buffer, total_num_pages: int) -> KVCacheMemory:
    """Wraps a single raw buffer as a one-shard, non-replicated NIXL group."""
    return kv_group([buf], total_num_pages)


def kv_group(
    bufs: Sequence[Buffer],
    total_num_pages: int,
    *,
    replicated: bool = False,
) -> KVCacheMemory:
    """Wraps raw TP-shard buffers as one authored NIXL group.

    Every buffer must share a shape after the uint8 page-view; the group carries
    all shards of one logical ``(child, kind)`` tensor.
    """
    return KVCacheMemory(
        replicated=replicated,
        buffers=[view_2d_uint8(b, total_num_pages) for b in bufs],
    )
