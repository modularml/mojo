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

"""Characterization tests pinning the canonical `bytes` hash encoding.

These assert the invariants that prove the int->bytes refactor preserves
behaviour: every block hash is the byte-image of the value the pre-refactor
code produced (a bijection), so no dict/prefix-cache semantics change.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
    _truncate_to_signed64,
    hash_request_tokens,
)


@pytest.mark.parametrize("block_size", [1, 2, 64, 128])
def test_ahash64_returns_8_byte_bytes(block_size: int) -> None:
    tokens = np.arange(block_size * 5, dtype=np.int32)
    out = hash_request_tokens(tokens, block_size)
    assert len(out) == 5
    assert all(isinstance(h, bytes) and len(h) == 8 for h in out)


@pytest.mark.parametrize("block_size", [1, 2, 64, 128])
def test_sha256_64_returns_8_byte_bytes(block_size: int) -> None:
    tokens = np.arange(block_size * 5, dtype=np.int32)
    out = hash_request_tokens(tokens, block_size, algo="sha256_64")
    assert len(out) == 5
    assert all(isinstance(h, bytes) and len(h) == 8 for h in out)


@pytest.mark.parametrize("block_size", [1, 2, 64, 128])
def test_sha256_returns_32_byte_bytes(block_size: int) -> None:
    tokens = np.arange(block_size * 5, dtype=np.int32)
    out = hash_request_tokens(tokens, block_size, algo="sha256")
    assert len(out) == 5
    assert all(isinstance(h, bytes) and len(h) == 32 for h in out)


def test_sha256_64_is_byte_image_of_truncated_full_digest() -> None:
    """sha256_64[i] == first 8 bytes of the full SHA-256 digest[i].

    This equals the byte encoding of the pre-refactor signed-64-bit int
    (``_truncate_to_signed64(d).to_bytes(8, "big", signed=True) == d[:8]``),
    proving the stored value is a bijection with the legacy int.
    """
    tokens = np.arange(128 * 6, dtype=np.int32)
    full = cast(list[bytes], hash_request_tokens(tokens, 128, algo="sha256"))
    short = cast(
        list[bytes], hash_request_tokens(tokens, 128, algo="sha256_64")
    )
    assert short == [d[:8] for d in full]
    assert short == [
        _truncate_to_signed64(d).to_bytes(8, "big", signed=True) for d in full
    ]


def test_ahash64_chaining_matches_batched() -> None:
    """Chaining ahash64 block-by-block (parent = previous 8-byte hash)
    reproduces the batched result exactly."""
    block_size = 128
    prompt = np.arange(block_size * 5, dtype=np.int32)
    batched = hash_request_tokens(prompt, block_size)

    for i in range(1, len(batched)):
        block_tokens = prompt[i * block_size : (i + 1) * block_size]
        parent = batched[i - 1]
        assert isinstance(parent, bytes) and len(parent) == 8
        chained = hash_request_tokens(block_tokens, block_size, parent)[0]
        assert chained == batched[i]
