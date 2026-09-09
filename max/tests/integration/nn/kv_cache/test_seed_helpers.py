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

"""Unit tests for the KV-cache seed resolver.

Covers all behavioral branches of ``resolve_kv_hash_seed``:

- ``ahash64`` returns ``None`` when no seed is configured, and never
  auto-generates one (unlike ``sha256``) -- default deployments keep
  deterministic, cross-restart-stable behavior.
- ``ahash64`` decodes and uses an explicitly configured seed_hex, the
  same as ``sha256`` / ``sha256_64`` do.
- ``sha256`` / ``sha256_64`` accept a valid 64-char hex seed.
- ``sha256`` / ``sha256_64`` cache a generated random seed across
  multiple calls within the same process.
- Invalid hex strings raise ``ValueError`` with a clear message.
"""

from __future__ import annotations

import logging

import pytest
from max.pipelines.kv_cache.paged_kv_cache import _seed_helpers
from max.pipelines.kv_cache.paged_kv_cache._seed_helpers import (
    resolve_kv_hash_seed,
)


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    """Each test starts with no cached random seed and an unlogged seed.

    The resolver caches both at module scope to make production behavior
    deterministic across replicas in the same process. Tests reset
    explicitly to keep cases independent.
    """
    _seed_helpers._cached_random_seed = None
    _seed_helpers._seed_logged = False


def test_ahash64_returns_none_when_no_seed() -> None:
    assert resolve_kv_hash_seed("ahash64", None) is None


def test_ahash64_decodes_valid_hex() -> None:
    """An explicitly configured seed_hex is decoded and used, not dropped."""
    seed_hex = "ab" * 32  # 64 chars -> 32 bytes
    result = resolve_kv_hash_seed("ahash64", seed_hex)
    assert result == bytes.fromhex(seed_hex)


def test_ahash64_rejects_short_hex() -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        resolve_kv_hash_seed("ahash64", "ab" * 16)  # only 16 bytes


def test_sha256_decodes_valid_hex() -> None:
    seed_hex = "ab" * 32  # 64 chars -> 32 bytes
    result = resolve_kv_hash_seed("sha256", seed_hex)
    assert result == bytes.fromhex(seed_hex)


def test_sha256_64_decodes_valid_hex() -> None:
    seed_hex = "cd" * 32
    result = resolve_kv_hash_seed("sha256_64", seed_hex)
    assert result == bytes.fromhex(seed_hex)


def test_sha256_random_seed_is_cached_across_calls() -> None:
    """Calling twice with seed_hex=None returns the same random bytes."""
    a = resolve_kv_hash_seed("sha256", None)
    b = resolve_kv_hash_seed("sha256", None)
    assert isinstance(a, bytes)
    assert isinstance(b, bytes)
    assert len(a) == 32
    assert a == b, "random seed must be cached for the lifetime of the process"


def test_sha256_random_seed_logs_hex_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="max.pipelines.kv_cache"):
        seed = resolve_kv_hash_seed("sha256", None)
        # Call again - should NOT emit a second log line.
        resolve_kv_hash_seed("sha256", None)

    assert isinstance(seed, bytes)
    info_records = [r for r in caplog.records if r.levelname == "INFO"]
    seed_records = [
        r for r in info_records if "Active KV-cache hash seed" in r.message
    ]
    assert len(seed_records) == 1, (
        f"expected exactly one seed-log line, got {len(seed_records)}: "
        f"{[r.message for r in seed_records]}"
    )
    assert seed.hex() in seed_records[0].message
    assert "auto-generated" in seed_records[0].message


def test_sha256_explicit_seed_logs_hex(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When operator supplies a seed, log identifies it as 'from config'."""
    seed_hex = "ef" * 32
    with caplog.at_level(logging.INFO, logger="max.pipelines.kv_cache"):
        resolve_kv_hash_seed("sha256", seed_hex)
    matching = [
        r
        for r in caplog.records
        if r.levelname == "INFO" and "Active KV-cache hash seed" in r.message
    ]
    assert len(matching) == 1, (
        f"expected exactly one seed-log line, got {len(matching)}"
    )
    assert "from config" in matching[0].message
    assert seed_hex in matching[0].message


def test_sha256_rejects_short_hex() -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        resolve_kv_hash_seed("sha256", "ab" * 16)  # only 16 bytes


def test_sha256_rejects_long_hex() -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        resolve_kv_hash_seed("sha256", "ab" * 64)  # 64 bytes, too long


def test_sha256_rejects_non_hex_string() -> None:
    with pytest.raises(ValueError, match="hex string"):
        resolve_kv_hash_seed(
            "sha256",
            "not-hex-at-all-but-64-chars-long-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        )


def test_sha256_64_rejects_non_hex_string() -> None:
    with pytest.raises(ValueError, match="hex string"):
        resolve_kv_hash_seed("sha256_64", "xx" * 32)
