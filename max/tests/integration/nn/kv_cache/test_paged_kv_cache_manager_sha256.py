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

"""End-to-end tests for sha256 KV-cache hashing through PagedKVCacheManager.

Walks the chain that operator-facing config feeds:

    KVCacheParams (with kv_hash_algo / kv_hash_seed)
      -> PagedKVCacheManager.__init__
        -> BlockManager.__init__
          -> compute_hashes_for_request
            -> hash_request_tokens

Constructs ``KVCacheParams`` directly to keep this target free of the
heavy ``max.pipelines.lib`` dep tree. The ``KVCacheConfig.to_params``
plumbing (Step 5d) is covered separately by a unit test for
``resolve_kv_hash_seed``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context


def _make_kv_manager(
    *,
    kv_hash_algo: Literal["ahash64", "sha256", "sha256_64"] = "ahash64",
    kv_hash_seed: bytes | None = None,
    page_size: int = 8,
    total_num_pages: int = 16,
) -> PagedKVCacheManager:
    """Build a minimal CPU PagedKVCacheManager with the given hash settings."""
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=2,
        page_size=page_size,
        devices=[DeviceRef.CPU()],
        enable_prefix_caching=True,
        kv_hash_algo=kv_hash_algo,
        kv_hash_seed=kv_hash_seed,
    )
    return PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[CPU()]),
        total_num_pages=total_num_pages,
        max_batch_size=8,
    )


def _drive_one_request(
    kv_manager: PagedKVCacheManager, prompt_len: int = 33
) -> TextContext:
    """Run a minimal claim+alloc+step cycle so req_to_hashes is populated."""
    ctx = create_text_context(np.empty(prompt_len))
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)
    kv_manager.runtime_inputs([[ctx]])
    ctx.update(42)
    kv_manager.step(ctx)
    return ctx


@pytest.mark.asyncio
async def test_sha256_produces_bytes_hashes() -> None:
    """kv_hash_algo='sha256' produces 32-byte hashes through the full chain."""
    kv_manager = _make_kv_manager(kv_hash_algo="sha256")
    ctx = _drive_one_request(kv_manager)

    block_manager = kv_manager._replica[0].block_manager
    hashes = block_manager.req_to_hashes[ctx.request_id]

    assert len(hashes) >= 1, "expected at least one full block hashed"
    for h in hashes:
        assert isinstance(h, bytes)
        assert len(h) == 32


@pytest.mark.asyncio
async def test_explicit_seed_is_deterministic() -> None:
    """Two managers with the same explicit seed produce identical chains."""
    seed = b"\xab" * 32

    bm1 = _make_kv_manager(kv_hash_algo="sha256", kv_hash_seed=seed)
    ctx1 = _drive_one_request(bm1)
    h1 = bm1._replica[0].block_manager.req_to_hashes[ctx1.request_id]

    bm2 = _make_kv_manager(kv_hash_algo="sha256", kv_hash_seed=seed)
    ctx2 = _drive_one_request(bm2)
    h2 = bm2._replica[0].block_manager.req_to_hashes[ctx2.request_id]

    assert h1 == h2


@pytest.mark.asyncio
async def test_ahash64_default_produces_int_hashes() -> None:
    """Default kv_hash_algo yields canonical 8-byte hashes."""
    kv_manager = _make_kv_manager()  # default = ahash64
    ctx = _drive_one_request(kv_manager)

    block_manager = kv_manager._replica[0].block_manager
    hashes = block_manager.req_to_hashes[ctx.request_id]

    assert len(hashes) >= 1
    for h in hashes:
        assert isinstance(h, bytes)
        assert len(h) == 8
