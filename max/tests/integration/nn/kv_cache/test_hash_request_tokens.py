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
import hashlib
from typing import cast

import numpy as np
import pytest
from max.pipelines.context import TokenHashOverride
from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
    _ZERO_SEED,
    _make_root_parent_hash,
    _truncate_to_signed64,
    hash_request_tokens,
)

# --- behavior preservation -------------------------------------------------


def test_default_algo_matches_legacy() -> None:
    """Default invocation (no kwargs) must match the pre-refactor behavior."""
    tokens = np.arange(640, dtype=np.int32)
    legacy = hash_request_tokens(tokens, 128)
    explicit = hash_request_tokens(tokens, 128, algo="ahash64")
    assert legacy == explicit
    assert all(isinstance(h, bytes) and len(h) == 8 for h in legacy)


def test_ahash64_salt_isolation() -> None:
    """Two requests with same prompt but different salt must not collide."""
    tokens = np.arange(128, dtype=np.int32)
    a = hash_request_tokens(tokens, 128, algo="ahash64", salt="user-1")
    b = hash_request_tokens(tokens, 128, algo="ahash64", salt="user-2")
    assert a != b
    assert all(x != y for x, y in zip(a, b, strict=False))


def test_ahash64_seed_isolation() -> None:
    """Different seeds also produce different chains."""
    tokens = np.arange(128, dtype=np.int32)
    a = hash_request_tokens(tokens, 128, algo="ahash64", seed=b"\x00" * 32)
    b = hash_request_tokens(tokens, 128, algo="ahash64", seed=b"\xab" * 32)
    assert a != b


def test_token_hash_override_replaces_only_target_token_and_restores() -> None:
    tokens = np.arange(16, dtype=np.int64)
    original = tokens.copy()
    override = TokenHashOverride(token_idx=5, token_hash=99_001)

    got = hash_request_tokens(
        tokens, 4, prefix_length=0, token_hash_overrides=[override]
    )
    manual = original.copy()
    manual[5] = override.token_hash
    expected = hash_request_tokens(manual, 4)

    assert got == expected
    assert np.array_equal(tokens, original)


def test_token_hash_override_honors_prefix_length() -> None:
    full_tokens = np.arange(16, dtype=np.int64)
    prefix_length = 4
    token_slice = full_tokens[prefix_length:12].copy()
    original = token_slice.copy()
    override = TokenHashOverride(token_idx=6, token_hash=77_003)

    got = hash_request_tokens(
        token_slice,
        4,
        prefix_length=prefix_length,
        token_hash_overrides=[override],
    )
    manual = original.copy()
    manual[override.token_idx - prefix_length] = override.token_hash
    expected = hash_request_tokens(manual, 4)

    assert got == expected
    assert np.array_equal(token_slice, original)


def test_duplicate_token_hash_override_rejects_without_mutating() -> None:
    tokens = np.arange(16, dtype=np.int64)
    original = tokens.copy()

    with pytest.raises(ValueError, match="same token index"):
        hash_request_tokens(
            tokens,
            4,
            prefix_length=0,
            token_hash_overrides=[
                TokenHashOverride(token_idx=5, token_hash=99_001),
                TokenHashOverride(token_idx=5, token_hash=77_003),
            ],
        )

    assert np.array_equal(tokens, original)


# --- sha256 path -----------------------------------------------------------


def test_sha256_returns_bytes() -> None:
    tokens = np.arange(640, dtype=np.int32)
    out = hash_request_tokens(tokens, 128, algo="sha256")
    assert all(isinstance(h, bytes) and len(h) == 32 for h in out)


def test_sha256_salt_isolation() -> None:
    """Two requests with same prompt but different salt must not collide."""
    tokens = np.arange(640, dtype=np.int32)
    a = hash_request_tokens(tokens, 128, algo="sha256", salt="user-1")
    b = hash_request_tokens(tokens, 128, algo="sha256", salt="user-2")
    assert a != b
    assert all(x != y for x, y in zip(a, b, strict=False))


def test_sha256_seed_isolation() -> None:
    """Different seeds also produce different chains."""
    tokens = np.arange(640, dtype=np.int32)
    a = hash_request_tokens(tokens, 128, algo="sha256", seed=b"\x00" * 32)
    b = hash_request_tokens(tokens, 128, algo="sha256", seed=b"\xab" * 32)
    assert a != b


def test_sha256_seed_and_salt_combine() -> None:
    """seed XOR sha256(salt) == seed XOR sha256(salt). Idempotent / consistent."""
    tokens = np.arange(640, dtype=np.int32)
    a = hash_request_tokens(
        tokens, 128, algo="sha256", seed=b"\x11" * 32, salt="x"
    )
    b = hash_request_tokens(
        tokens, 128, algo="sha256", seed=b"\x11" * 32, salt="x"
    )
    assert a == b


def test_sha256_no_salt_no_seed_is_deterministic() -> None:
    """Without salt or seed, behavior is reproducible (good for benchmarks)."""
    tokens = np.arange(640, dtype=np.int32)
    a = hash_request_tokens(tokens, 128, algo="sha256")
    b = hash_request_tokens(tokens, 128, algo="sha256")
    assert a == b


# --- sha256_64 truncated path ---------------------------------------------


def test_sha256_64_returns_bytes() -> None:
    tokens = np.arange(640, dtype=np.int32)
    out = hash_request_tokens(tokens, 128, algo="sha256_64")
    assert all(isinstance(h, bytes) and len(h) == 8 for h in out)


def test_sha256_64_truncates_full_sha256() -> None:
    """sha256_64[i] must equal the first 8 bytes of sha256_full[i]."""
    tokens = np.arange(640, dtype=np.int32)
    full = cast(list[bytes], hash_request_tokens(tokens, 128, algo="sha256"))
    short = cast(
        list[bytes], hash_request_tokens(tokens, 128, algo="sha256_64")
    )
    assert short == [d[:8] for d in full]
    # Equivalent to the legacy int truncation, now expressed as bytes.
    assert short == [
        _truncate_to_signed64(d).to_bytes(8, "big", signed=True) for d in full
    ]


def test_sha256_64_uses_full_chain_internally() -> None:
    """The internal chain must be full 256-bit SHA-256, not its truncation.

    If we (wrongly) truncated the chain, sha256_64 of a long sequence
    would diverge after a single truncation collision. Since the truncated
    output is just the low 8 bytes of each full-width chained digest,
    chaining the full and truncated paths must produce identical
    truncations everywhere -- which is what sha256_64 already returns.
    The previous test exercises this; this one is a redundancy guard.
    """
    tokens = np.arange(640 * 4, dtype=np.int32)
    full = cast(list[bytes], hash_request_tokens(tokens, 128, algo="sha256"))
    short = cast(
        list[bytes], hash_request_tokens(tokens, 128, algo="sha256_64")
    )
    for f, s in zip(full, short, strict=False):
        assert s == f[:8]


# --- helpers ---------------------------------------------------------------


def test_make_root_parent_hash_no_args_is_zero() -> None:
    assert _make_root_parent_hash(None, None) == _ZERO_SEED


def test_make_root_parent_hash_xor() -> None:
    seed = b"\xff" * 32
    salt = "abc"
    expected = bytes(
        b ^ s
        for b, s in zip(
            seed, hashlib.sha256(salt.encode()).digest(), strict=False
        )
    )
    assert _make_root_parent_hash(seed, salt) == expected


def test_make_root_parent_hash_rejects_wrong_seed_length() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        _make_root_parent_hash(b"\x00" * 16, None)


def test_truncate_signed64_signed_boundary() -> None:
    """High bit set -> negative int (matches Py_ssize_t convention)."""
    digest = bytes([0x80]) + bytes(31)
    assert _truncate_to_signed64(digest) == -(1 << 63)
    digest = bytes([0x7F] + [0xFF] * 7) + bytes(24)
    assert _truncate_to_signed64(digest) == (1 << 63) - 1
