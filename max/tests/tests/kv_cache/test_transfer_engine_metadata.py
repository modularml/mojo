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

"""CPU tests for the transfer-engine metadata struct split.

``KVTransferEngineMetadata`` now extends the transport-only
``TransferEngineMetadata``. The structural tests (field membership, subclass
relationship) run everywhere; the msgspec round-trip tests pin the wire
contract and require the ``nixl`` enums, which only exist on linux-x86_64.
"""

from __future__ import annotations

import msgspec
import pytest
from max._core import nixl
from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
    KVTransferEngineMetadata,
    TensorAgentMetadata,
    TransferEngineMetadata,
)

# ``max._core.nixl`` is an empty stub outside linux-x86_64 (the upstream NIXL
# shared library is unavailable), so ``nixl.MemoryType`` and msgspec's
# resolution of the ``memory_type`` field only work there.
_HAS_NIXL = hasattr(nixl, "MemoryType")
requires_nixl = pytest.mark.skipif(
    not _HAS_NIXL,
    reason="nixl enums unavailable on this platform (requires linux-x86_64)",
)


def _agent_meta(name: str = "a0") -> TensorAgentMetadata:
    return TensorAgentMetadata(
        agent_name=name,
        metadata=b"nixl-blob",
        base_addrs=[0x1000, 0x5000],
        device_id=0,
    )


# ---------------------------------------------------------------------------
# Structural: the split without touching nixl (runs on every platform)
# ---------------------------------------------------------------------------


def test_kv_metadata_extends_with_kv_fields() -> None:
    """KV subclass carries the base transport fields plus the KV additions.

    msgspec orders a subclass's own fields ahead of inherited ones; the structs
    encode as name-keyed maps, so order is wire-irrelevant. We assert on the
    field *set* to stay robust to that ordering.
    """
    base_fields = set(TransferEngineMetadata.__struct_fields__)
    kv_fields = set(KVTransferEngineMetadata.__struct_fields__)
    assert base_fields <= kv_fields
    assert kv_fields - base_fields == {
        "total_num_pages",
        "bytes_per_page",
        "bytes_per_group",
        "replicated_per_group",
    }


# ---------------------------------------------------------------------------
# msgspec round-trip: pins the wire contract (linux-x86_64 only)
# ---------------------------------------------------------------------------


@requires_nixl
def test_kv_metadata_roundtrip_all_fields() -> None:
    """KV subclass round-trips with every field populated."""
    md = KVTransferEngineMetadata(
        name="engine",
        memory_type=nixl.MemoryType.VRAM,
        hostname="host-b",
        agents_meta=[[_agent_meta()]],
        total_num_pages=17,
        bytes_per_page=800,
        bytes_per_group=[784, 16],
        replicated_per_group=[True, True],
    )
    dec = msgspec.json.decode(
        msgspec.json.encode(md), type=KVTransferEngineMetadata
    )
    assert dec == md
    assert dec.total_num_pages == 17
    assert dec.bytes_per_group == [784, 16]
    assert dec.replicated_per_group == [True, True]
