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

import os
import sys
from typing import Any

import mojo.importer
import numpy as np
import numpy.typing as npt

# Imports from 'mojo_module.mojo'
from .mojo_module import (  # type: ignore
    mojo_block_hasher,
    mojo_block_hasher_sha256,
    mojo_sha256_oneshot,
)


def block_hasher(
    tokens: npt.NDArray[np.integer[Any]],
    block_size: int,
    parent_hash: bytes = b"\x00" * 8,
    seed: bytes = b"\x00" * 32,
) -> list[bytes]:
    """Hash tokens into blocks for prefix caching.

    The token list is partitioned into blocks of size `block_size`. The tokens in
    each block are hashed together with the hash of the previous block.

    This calls into the `mojo_block_hasher` function defined in `mojo_module.mojo`.
    The Mojo ABI (`mojo_block_hasher`) operates on signed 64-bit ints; this
    wrapper decodes the 8-byte parent to a signed int and encodes the int
    results back to the canonical 8-byte big-endian signed form.

    Args:
        tokens: A 1D numpy array of token IDs.
        block_size: The number of tokens per block. Must be greater than 0.
        parent_hash: The hash value of the parent block, as 8 big-endian
            signed bytes.
        seed: 32 bytes mixed into the hasher's keyed state, e.g. a
            per-tenant or per-request salt. All-zero (the default)
            reproduces the unseeded hash exactly.

    Returns:
        A list of block hash values, each 8 big-endian signed bytes.
    """
    if tokens.ndim != 1:
        raise ValueError(
            f"tokens must be a 1D array, found {tokens.ndim}D array"
        )
    if block_size <= 0:
        raise ValueError(
            f"block_size must be greater than 0, found {block_size}"
        )
    if len(parent_hash) != 8:
        raise ValueError(
            f"parent_hash must be exactly 8 bytes, got {len(parent_hash)}"
        )
    if len(seed) != 32:
        raise ValueError(f"seed must be exactly 32 bytes, got {len(seed)}")
    ph_int = int.from_bytes(parent_hash, "big", signed=True)
    # Cast the array to int32 as that is what the mojo block hasher expects.
    if tokens.dtype != np.int32:
        tokens = tokens.astype(np.int32)
    seed_arr = np.frombuffer(seed, dtype=np.uint8)
    result = mojo_block_hasher(tokens, block_size, ph_int, seed_arr)
    return [h.to_bytes(8, "big", signed=True) for h in result]


def block_hasher_sha256(
    tokens: npt.NDArray[np.integer[Any]],
    block_size: int,
    parent_hash: bytes = b"\x00"
    * 32,  # Default to all zeros if no parent hash is provided
) -> list[bytes]:
    """Hash tokens into blocks for prefix caching using SHA-256.

    The token list is partitioned into blocks of size `block_size`. The tokens in
    each block are hashed together with the hash of the previous block.

    This calls into the `mojo_block_hasher_sha256` function defined in `mojo_module.mojo`.

    Args:
        tokens: A 1D numpy array of token IDs.
        block_size: The number of tokens per block. Must be greater than 0.
        parent_hash: The hash value of the parent block.

    Returns:
        A list of block hash values.
    """
    if tokens.ndim != 1:
        raise ValueError(
            f"tokens must be a 1D array, found {tokens.ndim}D array"
        )
    if block_size <= 0:
        raise ValueError(
            f"block_size must be greater than 0, found {block_size}"
        )
    if len(parent_hash) != 32:
        raise ValueError(
            f"parent_hash must be exactly 32 bytes, got {len(parent_hash)}"
        )
    if tokens.dtype != np.int32:
        tokens = tokens.astype(np.int32)

    num_blocks = tokens.size // block_size
    out = np.empty((num_blocks, 32), dtype=np.uint8)
    parent_arr = np.frombuffer(parent_hash, dtype=np.uint8)
    mojo_block_hasher_sha256(tokens, block_size, parent_arr, out)
    return [bytes(out[i]) for i in range(num_blocks)]


def sha256_oneshot(data: bytes) -> bytes:
    """Compute the SHA-256 (FIPS 180-4) digest of ``data``.

    Thin wrapper around the Mojo ``sha256()`` primitive in
    ``sha256.mojo``, exposed for known-answer-test validation. Production
    callers should use :func:`block_hasher_sha256` instead.

    Args:
        data: Input bytes to hash (any length, including empty).

    Returns:
        32-byte SHA-256 digest.
    """
    if data:
        arr = np.frombuffer(data, dtype=np.uint8)
    else:
        arr = np.empty(0, dtype=np.uint8)
    out = np.empty(32, dtype=np.uint8)
    mojo_sha256_oneshot(arr, out)
    return bytes(out)
