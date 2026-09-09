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
from max._core_mojo import block_hasher_sha256, sha256_oneshot
from pytest_benchmark.fixture import BenchmarkFixture


def _py_reference(
    tokens: np.ndarray, block_size: int, parent_hash: bytes
) -> list[bytes]:
    """Pure-Python reference implementation of the chained SHA-256 block hasher."""
    assert tokens.dtype == np.int32
    n = tokens.size // block_size
    prev = parent_hash
    out: list[bytes] = []
    for i in range(n):
        block_bytes = tokens[i * block_size : (i + 1) * block_size].tobytes()
        local = hashlib.sha256(block_bytes).digest()
        seq = hashlib.sha256(local + prev).digest()
        out.append(seq)
        prev = seq
    return out


@pytest.mark.parametrize("block_size", [1, 4, 64, 128])
@pytest.mark.parametrize("num_blocks", [1, 2, 32])
def test_matches_python_reference(block_size: int, num_blocks: int) -> None:
    tokens = np.arange(block_size * num_blocks, dtype=np.int32)
    parent = b"\x00" * 32
    got = block_hasher_sha256(tokens, block_size, parent)
    expected = _py_reference(tokens, block_size, parent)
    assert got == expected


def test_chaining_matches_split() -> None:
    """Hashing N+M blocks in one shot equals hashing N then M with chained parent."""
    block_size = 16
    tokens = np.arange(block_size * 5, dtype=np.int32)
    parent = b"\xab" * 32

    one_shot = block_hasher_sha256(tokens, block_size, parent)
    first_two = block_hasher_sha256(
        tokens[: 2 * block_size], block_size, parent
    )
    last_three = block_hasher_sha256(
        tokens[2 * block_size :], block_size, first_two[-1]
    )
    assert one_shot == first_two + last_three


def test_parent_hash_isolation() -> None:
    """Different parent_hash with the same tokens produces different seq hashes."""
    tokens = np.arange(128, dtype=np.int32)
    a = block_hasher_sha256(tokens, 32, b"\x00" * 32)
    b = block_hasher_sha256(tokens, 32, b"\x01" * 32)
    assert a != b
    assert all(x != y for x, y in zip(a, b, strict=False))


def test_each_hash_is_32_bytes() -> None:
    tokens = np.arange(640, dtype=np.int32)
    out = block_hasher_sha256(tokens, 128, b"\x00" * 32)
    assert all(isinstance(x, bytes) and len(x) == 32 for x in out)


def test_invalid_parent_hash_length_raises() -> None:
    tokens = np.arange(128, dtype=np.int32)
    with pytest.raises(ValueError):
        block_hasher_sha256(tokens, 32, b"\x00" * 16)


def test_dtype_coerced_to_int32() -> None:
    """Non-int32 inputs should be cast (matches existing block_hasher behavior)."""
    tokens_int64 = np.arange(128, dtype=np.int64)
    tokens_int32 = tokens_int64.astype(np.int32)
    a = block_hasher_sha256(tokens_int64, 32)
    b = block_hasher_sha256(tokens_int32, 32)
    assert a == b


# FIPS 180-4 Appendix B / NIST CAVP known-answer test (KAT) vectors for
# SHA-256. These lock the raw `sha256.mojo` primitive against drift; the
# digests are constants from the published standard.
_FIPS_KAT_VECTORS: list[tuple[bytes, str]] = [
    (
        b"",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    (
        b"abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    ),
    (
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    ),
]


@pytest.mark.parametrize(
    ("data", "expected_hex"), _FIPS_KAT_VECTORS, ids=["empty", "abc", "fips_b2"]
)
def test_sha256_fips_kat(data: bytes, expected_hex: str) -> None:
    """Mojo `sha256()` must match the FIPS 180-4 published digests."""
    assert sha256_oneshot(data).hex() == expected_hex


def test_sha256_one_million_a() -> None:
    """Mojo `sha256()` of 1,000,000 'a' bytes must match the FIPS digest."""
    one_million_a = b"a" * 1_000_000
    expected = (
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    )
    assert sha256_oneshot(one_million_a).hex() == expected


def test_sha256_matches_hashlib_random() -> None:
    """Cross-check the Mojo primitive against hashlib for arbitrary inputs."""
    rng = np.random.default_rng(seed=0xC0FFEE)
    for n in (1, 55, 56, 63, 64, 65, 127, 128, 129, 1024, 8193):
        buf = rng.bytes(n)
        assert sha256_oneshot(buf) == hashlib.sha256(buf).digest()


def test_block_hasher_chained_block_layout() -> None:
    """Lock the `local_hash || prev_seq_hash` layout in `mojo_module.mojo`.
    Mirrors the algorithm described in the plan: hash two adjacent blocks
    of distinct token bytes against a zero parent and compare to a
    hand-rolled hashlib computation that asserts the exact chain order.
    """
    block_size = 8
    tokens = np.concatenate(
        [
            np.zeros(block_size, dtype=np.int32),
            np.ones(block_size, dtype=np.int32),
        ]
    )
    parent = b"\x00" * 32

    got = block_hasher_sha256(tokens, block_size, parent)

    block0_bytes = tokens[:block_size].tobytes()
    local0 = hashlib.sha256(block0_bytes).digest()
    seq0 = hashlib.sha256(local0 + parent).digest()

    block1_bytes = tokens[block_size:].tobytes()
    local1 = hashlib.sha256(block1_bytes).digest()
    seq1 = hashlib.sha256(local1 + seq0).digest()

    assert got == [seq0, seq1]


def test_benchmark_mojo_sha256(benchmark: BenchmarkFixture) -> None:
    block_size = 8
    tokens = np.arange(30000, dtype=np.int32)

    _ = benchmark.pedantic(
        block_hasher_sha256,
        args=(tokens, block_size, b"\x00" * 32),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )
