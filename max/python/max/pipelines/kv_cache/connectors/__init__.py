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

"""KV cache connectors for external cache tiers.

- `NullConnector`: No-op connector when external caching is disabled
- `RustTierConnector`: GPU <-> CPU <-> Disk offloading, backed by the Rust
  ``kv_tier_connector`` extension. Also serves the ``tiered`` alias, whose
  Python implementation it replaced.
- `create_connector()`: Factory function
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from max.driver import Device
from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.cache_params import (
    KVCacheMemory,
    KVCacheParamInterface,
    KVConnectorType,
)
from max.pipelines.kv_cache.kv_connector import KVConnector

from .null_connector import NullConnector

logger = logging.getLogger("max.pipelines")


def create_connector(
    leaves: Mapping[str, KVCacheGroupId],
    devices: Sequence[Device],
    replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]],
    params: KVCacheParamInterface,
    device_memory_bytes: int,
) -> KVConnector:
    """Create a KV cache connector instance from ``params.kv_connector_config``.

    A single connector serves every DP replica for all connector types:
    ``replica_kv_memory`` holds each replica's device buffers, and load/offload
    select the replica via ``replica_idx`` (SERVOPT-1501). The host/disk tiers
    back this with one shared pinned host buffer / disk cache, which the tiered
    connector sizes from the config's ``host_offload_max_gb`` /
    ``disk_offload_max_gb`` (or from its own device page pool when either is
    unset); the distributed ``dkv`` connector owns one Rust client per replica
    internally.

    ``tiered`` is a backward-compatible alias for the Rust ``rust_tiered``
    connector, which replaced its deleted Python implementation, and therefore
    inherits ``rust_tiered``'s CUDA/HIP requirement.

    Args:
        devices: Devices for the KV cache tensors (all participating devices).
        replica_kv_memory: Per-replica offload-ready KV memory units, one
            mapping per DP replica, keyed by the leaf ids ``params.leaves()``
            names.
        params: KV-cache parameters. Carries the connector config (type and
            settings); the ``dkv`` connector also uses them to derive its
            multi-tenant per-GPU handshake identity.
        device_memory_bytes: The device page pool this connector sizes its
            tiers against. The caller states it rather than deriving it from
            ``params``, because the managers measure it differently: the paged
            manager counts its flat pages, Jenga its huge blocks.

    Returns:
        A connector instance implementing the KVConnector protocol.
    """
    cfg = params.kv_connector_config
    connector = cfg.type

    if connector == KVConnectorType.dkv:
        from .dkv import DKVConnector

        if not all(group_id.is_full() for group_id in leaves.values()):
            raise ValueError(
                "DKV KVConnector requires all leaves to be full attention groups. "
                f"Found: {leaves}"
            )

        if not cfg.block_store_endpoint:
            raise ValueError(
                "kv_connector_config must include 'block_store_endpoint' "
                "when its type is 'dkv'"
            )
        logger.info(
            "Creating DKVConnector: endpoint=%s",
            cfg.block_store_endpoint,
        )
        # list[dict] -> list[list]
        replica_kv_memory_list = [
            list(memory.values()) for memory in replica_kv_memory
        ]
        return DKVConnector(
            replica_kv_memory=replica_kv_memory_list,
            local_block_store_endpoint=cfg.block_store_endpoint,
            devices=devices,
            params=params,
        )

    # ``tiered`` is a backward-compatible alias for ``rust_tiered``, kept after
    # its Python implementation was deleted.
    if connector in (KVConnectorType.tiered, KVConnectorType.rust_tiered):
        from .rust_tier_connector import RustTierConnector

        return RustTierConnector.create(
            leaves=leaves,
            replica_kv_memory=replica_kv_memory,
            params=params,
            device_memory_bytes=device_memory_bytes,
        )

    logger.debug("Creating NullConnector: no KV cache connector configured")
    return NullConnector()


__all__ = [
    "KVConnector",
    "KVConnectorType",
    "NullConnector",
    "create_connector",
]
