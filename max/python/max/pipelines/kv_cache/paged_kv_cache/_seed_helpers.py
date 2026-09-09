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

"""Resolve the kv_cache_hash_seed from operator config.

The seed is an optional 64-character hex string, decoded here to 32 raw
bytes. If none is configured, `sha256`/`sha256_64` get a random one
(cached for the process lifetime); `ahash64` does not, so unconfigured
deployments stay deterministic across restarts. The active seed hex is
logged once.
"""

from __future__ import annotations

import logging
import secrets
import threading

from .block_utils import KVHashAlgo

logger = logging.getLogger("max.pipelines.kv_cache")

_lock = threading.Lock()
_cached_random_seed: bytes | None = None
_seed_logged: bool = False


def resolve_kv_hash_seed(
    algo: KVHashAlgo,
    seed_hex: str | None,
) -> bytes | None:
    """Resolve the operator-supplied kv_cache_hash_seed.
    Args:
        algo: The selected hash algorithm.
        seed_hex: Optional 64-character hex string (32 bytes after decode).
    Returns:
        - ``None`` if no seed is configured. ``ahash64`` never
          auto-generates one; ``sha256``/``sha256_64`` do (random,
          cached for the process lifetime).
        - 32 raw bytes when ``seed_hex`` is set, for any algo.
    Raises:
        ValueError: If ``seed_hex`` is not a valid 64-character hex
            string decoding to exactly 32 bytes.
    """
    if seed_hex is None:
        if algo == "ahash64":
            # Unlike sha256, never auto-randomize the seed for the default
            # algo: doing so would silently change every existing ahash64
            # deployment's cache behavior on every restart.
            return None
        return _get_or_create_random_seed()

    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as exc:
        raise ValueError(
            f"kv_cache_hash_seed must be a hex string; got {seed_hex!r}"
        ) from exc
    if len(seed) != 32:
        raise ValueError(
            f"kv_cache_hash_seed must decode to exactly 32 bytes; "
            f"got {len(seed)} bytes from {seed_hex!r}"
        )
    _log_active_seed_once(seed, generated=False)
    return seed


def _get_or_create_random_seed() -> bytes:
    global _cached_random_seed
    with _lock:
        if _cached_random_seed is None:
            _cached_random_seed = secrets.token_bytes(32)
    _log_active_seed_once(_cached_random_seed, generated=True)
    return _cached_random_seed


def _log_active_seed_once(seed: bytes, *, generated: bool) -> None:
    global _seed_logged
    with _lock:
        if _seed_logged:
            return
        _seed_logged = True
    logger.info(
        "Active KV-cache hash seed: %s (%s)",
        seed.hex(),
        "auto-generated" if generated else "from config",
    )
