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

import numpy as np
import pytest
from max._core_mojo import block_hasher
from pytest_benchmark.fixture import BenchmarkFixture


def test_seed_default_matches_unseeded() -> None:
    """Omitting `seed` must match passing an explicit all-zero seed."""
    block_size = 32
    tokens = np.arange(256, dtype=np.int32)

    default = block_hasher(tokens, block_size, b"\x00" * 8)
    explicit_zero = block_hasher(tokens, block_size, b"\x00" * 8, b"\x00" * 32)
    assert default == explicit_zero


def test_seed_isolation() -> None:
    """Different seeds on the same tokens must produce disjoint hash lists."""
    block_size = 32
    tokens = np.arange(256, dtype=np.int32)

    a = block_hasher(tokens, block_size, b"\x00" * 8, b"\x00" * 32)
    b = block_hasher(tokens, block_size, b"\x00" * 8, b"\x01" * 32)
    assert a != b
    assert all(x != y for x, y in zip(a, b, strict=False))


def test_invalid_seed_length_raises() -> None:
    tokens = np.arange(128, dtype=np.int32)
    with pytest.raises(ValueError):
        block_hasher(tokens, 32, b"\x00" * 8, b"\x00" * 16)


def test_block_hasher() -> None:
    block_size = 128
    num_tokens = 3000
    tokens = np.arange(num_tokens, dtype=np.int32)

    hashes = block_hasher(tokens, block_size, b"\x00" * 8)

    assert isinstance(hashes, list)
    assert isinstance(hashes[0], bytes)
    assert len(hashes) == num_tokens // block_size

    # It is very unlikely (but not impossible) that a valid hasher will return 0
    # Usually a 0 is indicative of a bug of some sort.
    assert b"\x00" * 8 not in hashes

    # It is unlikely (but not impossible) that a hasher will return the same value twice.
    # Usually a duplicate is indicative of a bug of some sort.
    seen = set()
    for h in hashes:
        assert h not in seen
        seen.add(h)


def mojo_block_hasher(tokens: np.ndarray, block_size: int) -> list[bytes]:
    return block_hasher(tokens, block_size, b"\x00" * 8)


def tensor_block_hasher(tokens: np.ndarray, block_size: int) -> list[int]:
    num_elts = tokens.size
    num_hashes = num_elts // block_size

    # Initial hash seed value
    prev_hash = 0

    results = []
    for i in range(num_hashes):
        block = tokens[i * block_size : (i + 1) * block_size]
        digest = hashlib.blake2b(block.tobytes()).hexdigest()
        pair_to_hash = (prev_hash, digest)
        curr_hash = hash(pair_to_hash)

        results.append(curr_hash)

    return results


def naive_block_hasher(tokens: np.ndarray, block_size: int) -> list[int]:
    num_elts = tokens.size
    num_hashes = num_elts // block_size

    # Initial hash seed value
    prev_hash = 0

    results = []
    for i in range(num_hashes):
        block = tokens[i * block_size : (i + 1) * block_size]
        pair_to_hash = (prev_hash, tuple(block))
        curr_hash = hash(pair_to_hash)
        results.append(curr_hash)
        prev_hash = curr_hash

    return results


def test_benchmark_mojo(benchmark: BenchmarkFixture) -> None:
    block_size = 128
    tokens = np.arange(30000, dtype=np.int32)

    _ = benchmark.pedantic(
        mojo_block_hasher,
        args=(tokens, block_size),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )


def test_benchmark_tensor(benchmark: BenchmarkFixture) -> None:
    block_size = 128
    tokens = np.arange(30000, dtype=np.int32)

    _ = benchmark.pedantic(
        tensor_block_hasher,
        args=(tokens, block_size),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )


def test_benchmark_naive(benchmark: BenchmarkFixture) -> None:
    block_size = 128
    tokens = np.arange(30000, dtype=np.int32)

    _ = benchmark.pedantic(
        naive_block_hasher,
        args=(tokens, block_size),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )
