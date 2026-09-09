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

"""Deprecated: use max.pipelines.kv_cache instead."""

import warnings

warnings.warn(
    "max.kv_cache is deprecated and will be removed in a future release."
    " Use max.pipelines.kv_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)

from max.pipelines.kv_cache import DummyKVCache as DummyKVCache
from max.pipelines.kv_cache import (
    InsufficientBlocksError as InsufficientBlocksError,
)
from max.pipelines.kv_cache import KVTransferEngine as KVTransferEngine
from max.pipelines.kv_cache import (
    KVTransferEngineMetadata as KVTransferEngineMetadata,
)
from max.pipelines.kv_cache import PagedKVCacheManager as PagedKVCacheManager
from max.pipelines.kv_cache import TransferReqData as TransferReqData
from max.pipelines.kv_cache import available_port as available_port
from max.pipelines.kv_cache import load_kv_manager as load_kv_manager
