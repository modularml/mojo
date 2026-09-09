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
"""Unit tests for hash_image function."""

import numpy as np
import pytest
from max._core import xxhash
from max.support.image import hash_image, hash_video


def test_hash_image_contiguous() -> None:
    """Test hash_image with C-contiguous arrays."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert arr.flags["C_CONTIGUOUS"]

    h1 = hash_image(arr)
    h2 = hash_image(arr)
    assert h1 == h2, "Same array should produce same hash"

    # Different array should produce different hash
    arr2 = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    h3 = hash_image(arr2)
    assert h1 != h3, "Different arrays should produce different hashes"


def test_hash_image_non_contiguous() -> None:
    """Test hash_image with non-contiguous arrays (transposed)."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr_t = arr.T
    assert not arr_t.flags["C_CONTIGUOUS"]

    h1 = hash_image(arr_t)
    h2 = hash_image(arr_t)
    assert h1 == h2, "Same non-contiguous array should produce same hash"


def test_hash_image_contiguous_vs_non_contiguous_same_data() -> None:
    """Test that contiguous copy produces same hash as non-contiguous original."""
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    arr_t = arr.T  # Non-contiguous
    arr_t_copy = np.ascontiguousarray(arr_t)  # Contiguous copy of transposed

    assert not arr_t.flags["C_CONTIGUOUS"]
    assert arr_t_copy.flags["C_CONTIGUOUS"]
    assert np.array_equal(arr_t, arr_t_copy)

    # Both should produce the same hash since they represent the same data
    h1 = hash_image(arr_t)
    h2 = hash_image(arr_t_copy)
    assert h1 == h2, "Contiguous and non-contiguous with same data should match"


def test_hash_image_bytes_matches_encoder_content_key() -> None:
    """Pin hash_image's raw-bytes mode to the Rust encoder's ``content_key``.

    These goldens were produced by the encoder's
    ``multimodal::content_key(bytes, tier)`` (mach crate). Byte-for-byte parity
    is the whole point: the engine and the encoder must derive the same per-image
    key so cache-aware routing sends a request to the worker already holding it.
    A change here means the two sides have silently diverged.
    """
    assert hash_image(b"", 0) == -4072596861322023719
    assert (
        hash_image(b"modular-multimodal-routing", 280) == -1621788365911394505
    )
    assert hash_image(bytes(range(256)), 2016) == 441527499587478968


def test_hash_image_bytes_tier_and_bytes_participate() -> None:
    """The size tier and the raw bytes both change the key."""
    data = b"some encoded image bytes"
    assert hash_image(data, 0) == hash_image(data, 0)
    assert hash_image(data, 280) != hash_image(data, 560)
    assert hash_image(data, 280) != hash_image(data + b"!", 280)


def test_hash_image_bytes_requires_and_bounds_tier() -> None:
    """Bytes mode requires size_tier; out-of-range tiers raise OverflowError."""
    # Bytes without a size_tier is a usage error, not a silent pixel-mode hash.
    with pytest.raises(ValueError, match="size_tier is required"):
        hash_image(b"x")
    # size_tier is appended as a little-endian u64, so to_bytes itself rejects
    # negatives and values that do not fit in 64 bits.
    with pytest.raises(OverflowError):
        hash_image(b"x", -1)
    with pytest.raises(OverflowError):
        hash_image(b"x", 2**64)
    # The largest valid u64 is accepted, not rejected off-by-one.
    assert isinstance(hash_image(b"x", 2**64 - 1), int)


def test_hash_image_bytes_accepts_bytearray() -> None:
    """bytearray input hashes identically to the equivalent bytes."""
    ba = bytearray(b"modular")
    assert hash_image(ba, 280) == hash_image(b"modular", 280)


def test_hash_image_pixels_reject_size_tier() -> None:
    """Pixel mode rejects a size_tier instead of silently ignoring it."""
    arr = np.arange(8, dtype=np.float32)
    # A pixel array hashes fine with no tier...
    assert isinstance(hash_image(arr), int)
    # ...but passing a tier (which pixel mode cannot fold in) is a usage error
    # that would otherwise yield a subtly wrong, tier-independent key.
    with pytest.raises(ValueError, match="size_tier is only folded"):
        hash_image(arr, 123)


def _concatenated_digest(data: bytes, tier: int) -> int:
    """The raw-bytes key computed the way it was before streaming.

    Builds ``data ++ tier`` in one mutable buffer and hashes it in a single
    shot, which is what ``hash_image`` did when it copied the payload.
    """
    buf = bytearray(len(data) + 8)
    buf[: len(data)] = data
    buf[len(data) :] = int(tier).to_bytes(8, "little", signed=False)
    hash_val = xxhash.xxh3_64_intdigest(np.frombuffer(buf, dtype=np.uint8))
    return int(np.uint64(hash_val).astype(np.int64))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"x",
        b"modular",
        b"\x00" * 8,
        bytes(range(256)),
        b"\xff\xd8\xff\xe0" + bytes(range(256)) * 40,  # JPEG-ish, spans blocks
    ],
)
@pytest.mark.parametrize("tier", [0, 1, 280, 70 << 16 | 32, 2**64 - 1])
def test_streamed_key_matches_the_concatenated_key(
    data: bytes, tier: int
) -> None:
    """Streaming the tier must not change a single key.

    ``hash_image`` hashes the payload and the tier as two pieces instead of
    copying the payload to append eight bytes to it. XXH3 defines that as
    identical to hashing the concatenation, and this pins it: a separate
    encoder process reproduces these keys, so a digest change here would
    silently split the two sides' view of image identity.
    """
    assert hash_image(data, tier) == _concatenated_digest(data, tier)


def test_raw_byte_key_golden_values() -> None:
    """Hard-coded digests, so both implementations changing together is caught.

    The parity test above compares two code paths in this process; these pin the
    actual numbers a cache-aware router has to agree with.
    """
    assert hash_image(b"modular", 280) == 8862388020968666451
    assert hash_image(b"", 0) == -4072596861322023719
    assert hash_image(b"x", 2**64 - 1) == 6386758147624513512


def test_hash_video_depends_on_pixels_and_grid() -> None:
    pixels = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    grid = np.array([2, 2, 2], dtype=np.int32)

    same = hash_video(pixels.copy(), grid.copy())
    changed_pixels = pixels.copy()
    changed_pixels[0, 0] += 1
    changed_grid = np.array([1, 4, 2], dtype=np.int32)

    assert hash_video(pixels, grid) == same
    assert hash_video(pixels, grid) != hash_video(changed_pixels, grid)
    assert hash_video(pixels, grid) != hash_video(pixels, changed_grid)
