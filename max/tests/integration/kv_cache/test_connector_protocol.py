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

"""CPU conformance tests for the ``KVConnector`` protocol additions.

Covers the ``wait_for_loads`` / ``wait_for_offloads`` barriers and the
fire-and-forget ``touch`` method, verified against the no-op ``NullConnector``
(which needs no device) plus the host/disk connector.
"""

from __future__ import annotations

from max.pipelines.kv_cache.connectors import NullConnector
from max.pipelines.kv_cache.connectors.rust_tier_connector import (
    RustTierConnector,
)
from max.pipelines.kv_cache.kv_connector import KVConnector


def test_null_connector_satisfies_protocol() -> None:
    assert isinstance(NullConnector(), KVConnector)


def test_barrier_methods_are_callable() -> None:
    connector = NullConnector()
    # Both are no-op barriers (return ``None``); just ensure they are callable.
    connector.wait_for_loads()
    connector.wait_for_offloads()


def test_touch_returns_none_and_never_raises() -> None:
    # ``touch`` is fire-and-forget with a ``-> None`` annotation, so just call
    # it: accepts ``replica_idx``, tolerates an empty payload, never raises.
    connector = NullConnector()
    connector.touch([b"\x01" * 8])
    connector.touch([b"\x01" * 8, b"\x02" * 8], replica_idx=1)
    connector.touch([])

    # ``RustTierConnector`` needs device buffers and the Rust extension to fully
    # construct (see the GPU connector tests), but ``touch`` ignores instance
    # state, so exercise it on an uninitialized instance -- the same ``__new__``
    # pattern used in ``test_kv_connector_reset_metrics.py`` -- to prove it is a
    # no-op that never raises.
    tier_connector = RustTierConnector.__new__(RustTierConnector)
    tier_connector.touch([b"\x01" * 8])
    tier_connector.touch([b"\x01" * 8], replica_idx=2)


def test_touch_preserves_protocol_conformance() -> None:
    # Adding ``touch`` to the Protocol must not break structural conformance.
    # ``NullConnector`` is CPU-constructable, so assert the full runtime check;
    # for the host/disk connector ``@runtime_checkable`` ``isinstance`` invokes
    # its state-reading properties (GPU-bound to construct), so assert the new
    # Protocol member is present at the class level instead.
    assert isinstance(NullConnector(), KVConnector)
    assert hasattr(RustTierConnector, "touch")
