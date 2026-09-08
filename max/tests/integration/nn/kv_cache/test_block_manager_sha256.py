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

"""End-to-end tests for sha256 hashing through BlockManager.

Exercises the kv_hash_algo / kv_hash_seed / cache_salt plumbing added to
BlockManager.compute_hashes_for_request:

- sha256 produces 32-byte bytes hashes per block; sha256_64 truncates to 8.
- Identical tokens + identical seed/salt => identical hash chain (cache hit).
- Different cache_salt => different hash chain (multi-tenant isolation).
- Different kv_hash_seed => different hash chain (cluster isolation).
- kv_hash_algo="ahash64" (default) still yields int hashes (no regression).
- kv_hash_algo="ahash64" also supports cache_salt/kv_hash_seed isolation.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.kv_cache.paged_kv_cache.block_utils import KVHashAlgo
from max.pipelines.modeling.types import RequestID


def _make_ctx(
    tokens: np.ndarray,
    request_id: RequestID = RequestID("req-1"),  # noqa: B008
    *,
    cache_salt: str | None = None,
) -> TextContext:
    """Build a minimal TextGenerationContext-like stub.
    BlockManager.compute_hashes_for_request accesses ``ctx.request_id``,
    ``len(ctx.tokens)``, ``ctx.tokens[i:j]``, ``ctx.images`` (via an
    ``isinstance`` check that fails for SimpleNamespace), and
    ``ctx.cache_salt`` (direct attribute access — the real ``TextContext``
    always defines this attribute, so the stub must too, even when no
    caller-supplied salt is set).
    """
    ctx = SimpleNamespace(
        request_id=request_id,
        tokens=tokens,
        cache_salt=cache_salt,
    )
    return cast(TextContext, ctx)


def _make_block_manager(
    *,
    block_size: int = 8,
    total_blocks: int = 32,
    kv_hash_algo: KVHashAlgo = "ahash64",
    kv_hash_seed: bytes | None = None,
) -> BlockManager:
    return BlockManager(
        total_num_blocks=total_blocks,
        block_size=block_size,
        connector=cast(object, NullConnector()),  # type: ignore[arg-type]
        enable_prefix_caching=True,
        kv_hash_algo=kv_hash_algo,
        kv_hash_seed=kv_hash_seed,
    )


def test_sha256_produces_32_byte_hashes() -> None:
    bm = _make_block_manager(kv_hash_algo="sha256")
    # 33 tokens => 32 hashable (last reserved) => 4 full blocks of 8.
    tokens = np.arange(33, dtype=np.int32)
    ctx = _make_ctx(tokens)

    bm.compute_hashes_for_request(ctx)

    hashes = bm.req_to_hashes[ctx.request_id]
    assert len(hashes) == 4
    for h in hashes:
        assert isinstance(h, bytes)
        assert len(h) == 32


def test_sha256_64_produces_8_byte_hashes() -> None:
    """sha256_64 truncates each digest to the canonical 8-byte hash form."""
    bm = _make_block_manager(kv_hash_algo="sha256_64")
    tokens = np.arange(33, dtype=np.int32)
    ctx = _make_ctx(tokens)

    bm.compute_hashes_for_request(ctx)

    hashes = bm.req_to_hashes[ctx.request_id]
    assert len(hashes) == 4
    for h in hashes:
        assert isinstance(h, bytes)
        assert len(h) == 8


def test_sha256_same_tokens_same_hashes() -> None:
    """Two identical contexts produce identical hash chains (cache-hit potential)."""
    tokens = np.arange(33, dtype=np.int32)

    bm1 = _make_block_manager(kv_hash_algo="sha256")
    bm1.compute_hashes_for_request(_make_ctx(tokens, RequestID("req-A")))

    bm2 = _make_block_manager(kv_hash_algo="sha256")
    bm2.compute_hashes_for_request(_make_ctx(tokens, RequestID("req-B")))

    assert (
        bm1.req_to_hashes[RequestID("req-A")]
        == bm2.req_to_hashes[RequestID("req-B")]
    )


def test_sha256_salt_isolation() -> None:
    """Same tokens + different cache_salt => disjoint hashes (multi-tenant safety)."""
    tokens = np.arange(33, dtype=np.int32)
    bm = _make_block_manager(kv_hash_algo="sha256")

    bm.compute_hashes_for_request(
        _make_ctx(tokens, RequestID("req-tenant-A"), cache_salt="tenant-A")
    )
    bm.compute_hashes_for_request(
        _make_ctx(tokens, RequestID("req-tenant-B"), cache_salt="tenant-B")
    )

    a = bm.req_to_hashes[RequestID("req-tenant-A")]
    b = bm.req_to_hashes[RequestID("req-tenant-B")]
    assert a != b
    assert set(a).isdisjoint(set(b))


def test_sha256_seed_isolation() -> None:
    """Same tokens + different kv_hash_seed => disjoint hashes (cluster isolation)."""
    tokens = np.arange(33, dtype=np.int32)

    bm1 = _make_block_manager(kv_hash_algo="sha256", kv_hash_seed=b"\x00" * 32)
    bm1.compute_hashes_for_request(_make_ctx(tokens))

    bm2 = _make_block_manager(kv_hash_algo="sha256", kv_hash_seed=b"\x01" * 32)
    bm2.compute_hashes_for_request(_make_ctx(tokens))

    a = bm1.req_to_hashes[RequestID("req-1")]
    b = bm2.req_to_hashes[RequestID("req-1")]
    assert a != b
    assert set(a).isdisjoint(set(b))


def test_ahash64_default_unchanged() -> None:
    """Default kv_hash_algo yields canonical 8-byte hashes; legacy path unchanged."""
    bm = _make_block_manager()  # default = ahash64

    tokens = np.arange(33, dtype=np.int32)
    bm.compute_hashes_for_request(_make_ctx(tokens))

    hashes = bm.req_to_hashes[RequestID("req-1")]
    assert len(hashes) == 4
    for h in hashes:
        assert isinstance(h, bytes)
        assert len(h) == 8


def test_ahash64_with_cache_salt_isolates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Under ahash64, different request-supplied cache_salt values now
    produce disjoint hash chains for identical tokens (multi-tenant
    isolation) -- the same guarantee the sha256 path already had, with
    no warning emitted (cache_salt is no longer dropped)."""
    tokens = np.arange(33, dtype=np.int32)
    bm = _make_block_manager()  # default ahash64

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        bm.compute_hashes_for_request(
            _make_ctx(tokens, RequestID("req-A"), cache_salt="tenant-A")
        )
        bm.compute_hashes_for_request(
            _make_ctx(tokens, RequestID("req-B"), cache_salt="tenant-B")
        )

    a = bm.req_to_hashes[RequestID("req-A")]
    b = bm.req_to_hashes[RequestID("req-B")]
    assert a != b
    assert set(a).isdisjoint(set(b))
    assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_ahash64_seed_isolation() -> None:
    """Same tokens + different kv_hash_seed => disjoint hashes (cluster isolation)."""
    tokens = np.arange(33, dtype=np.int32)

    bm1 = _make_block_manager(kv_hash_seed=b"\x00" * 32)  # default ahash64
    bm1.compute_hashes_for_request(_make_ctx(tokens))

    bm2 = _make_block_manager(kv_hash_seed=b"\x01" * 32)
    bm2.compute_hashes_for_request(_make_ctx(tokens))

    a = bm1.req_to_hashes[RequestID("req-1")]
    b = bm2.req_to_hashes[RequestID("req-1")]
    assert a != b
    assert set(a).isdisjoint(set(b))
