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

"""Image-related utilities."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max._core import xxhash

__all__ = [
    "find_contiguous_ranges",
    "hash_image",
    "hash_video",
]


def find_contiguous_ranges(
    arr: npt.NDArray[np.integer[Any]], targets: Sequence[int]
) -> list[tuple[int, int]]:
    """Find the contiguous ranges of the given tokens in a 1D array.

    For example::

        find_contiguous_ranges([1, 2, 99, 99, 3, 99, 98, 99], [98, 99])
        # -> [(2, 4), (5, 8)]

    Args:
        arr: The 1D array to scan.
        targets: The token values that delimit a range.

    Returns:
        A list of ``(start, end)`` half-open ranges ``[start, end)``, one per
        maximal run of positions where ``arr`` holds a value in ``targets``.

    Raises:
        ValueError: If ``arr`` is not 1D.
    """

    if arr.ndim != 1:
        raise ValueError(f"Array must be 1D, found {arr.shape}")

    # Boolean mask where arr == x
    mask = np.isin(arr, targets)
    # Find where mask changes value (True <-> False)
    diff = np.diff(mask.astype(int))
    # Start indices are where it changes from 0 -> 1
    starts = np.where(diff == 1)[0] + 1
    # End indices are where it changes from 1 -> 0
    ends = np.where(diff == -1)[0] + 1

    # Handle if the sequence starts or ends with True
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    # Cast values from int64 -> int
    starts = starts.tolist()
    ends = ends.tolist()

    return list(zip(starts, ends, strict=True))


def hash_image(
    image: npt.NDArray[Any] | bytes | bytearray,
    size_tier: int | None = None,
) -> int:
    """Compute the hash of an image.

    Two input modes are supported:

    - Raw encoded bytes (preferred for prefix-cache / routing keys): pass the
      raw encoded image container bytes (the JPEG/PNG/WebP bytes as received by
      the router, before any decode or resize) as ``image``, together with a
      ``size_tier`` capturing the resolution class the image is processed at.
      Because it never touches decoded float pixels, the result is
      byte-identical across torch/numpy/BLAS versions and CPU-vs-GPU, so it is
      safe as a key that a separate encoder must reproduce. This mirrors that
      encoder's ``content_key``: ``xxh3_64(bytes ++ size_tier)`` with
      ``size_tier`` appended as a little-endian unsigned 64-bit integer.

    - Decoded pixels (legacy): pass a numpy pixel array as ``image`` and omit
      ``size_tier``. Supports any dtype (float32, uint16 for bfloat16 bits,
      etc.) and ensures a C-contiguous layout before hashing. The digest
      depends on the post-resize float pixels, so it is not reproducible across
      torch/BLAS/device; prefer the raw-bytes mode where the encoded bytes are
      available.

    In both modes the unsigned 64-bit digest is reinterpreted as a signed
    64-bit int for numpy int64 token-hash compatibility.

    Args:
        image: Either the raw encoded image bytes or a decoded numpy pixel
            array.
        size_tier: The resolution size class folded into the key, so identical
            bytes processed at different resolutions key distinctly. Required
            when ``image`` is bytes; must be omitted for a pixel array.

    Returns:
        The signed 64-bit image hash.

    Raises:
        ValueError: If ``image`` is bytes and ``size_tier`` is not provided, or
            if ``image`` is a pixel array and ``size_tier`` is provided (the
            tier is only folded into the raw-bytes key, so passing it with a
            pixel array is a silent no-op we reject rather than swallow).
        OverflowError: If ``size_tier`` is negative or does not fit in an
            unsigned 64-bit integer, since the key appends it as a
            little-endian ``u64``.
    """
    if isinstance(image, (bytes, bytearray)):
        if size_tier is None:
            raise ValueError(
                "size_tier is required when hashing raw image bytes"
            )
        # Hash the bytes and the tier as two pieces rather than building
        # ``image ++ tier`` in one buffer. The payload here is a whole encoded
        # image or video, and copying it to append eight bytes costs about
        # twenty times the hash it feeds -- 0.26ms of copy against 0.009ms of
        # hashing for a 409KB JPEG. XXH3 defines the streamed digest as
        # identical to the one-shot digest over the same byte sequence, so this
        # produces the same key the concatenation did.
        #
        # The range check keeps ``OverflowError`` as the failure for a tier that
        # cannot be a u64, which is what ``int.to_bytes`` raised when this built
        # the suffix in Python.
        tier = int(size_tier)
        if not 0 <= tier < 1 << 64:
            raise OverflowError(
                f"size_tier {tier} does not fit in an unsigned 64-bit integer"
            )
        hash_val = xxhash.xxh3_64_intdigest_with_u64_suffix(
            # ``bytes`` passes through without a copy; a ``bytearray`` has to be
            # converted, which no caller does on the serving path.
            image if isinstance(image, bytes) else bytes(image),
            tier,
        )
    else:
        if size_tier is not None:
            raise ValueError(
                "size_tier is only folded into the raw-bytes key; it must be "
                "omitted when hashing a decoded pixel array"
            )
        hash_val = xxhash.xxh3_64_intdigest(np.ascontiguousarray(image).data)  # type: ignore[arg-type]
    # xxh3_64_intdigest returns an unsigned 64-bit int; reinterpret as signed to
    # match the numpy int64 token-hash contract.
    return int(np.uint64(hash_val).astype(np.int64))


def hash_video(
    pixel_values: npt.NDArray[Any], grid_thw: npt.NDArray[np.integer[Any]]
) -> int:
    """Compute the hash of preprocessed video pixels and grid metadata.

    The input must already be the sampled, resized, normalized, model-ready
    video tensor. This helper does not decode or preprocess video frames.

    Args:
        pixel_values: The preprocessed, model-ready video pixel tensor.
        grid_thw: The temporal/height/width grid metadata for the video.

    Returns:
        The signed 64-bit video hash.
    """
    pixel_hash = hash_image(pixel_values)
    grid_hash = hash_image(np.asarray(grid_thw, dtype=np.int64))
    shape = np.asarray(pixel_values.shape, dtype=np.int64)
    metadata = np.concatenate(
        (
            np.array(
                [pixel_hash, grid_hash, np.dtype(pixel_values.dtype).num],
                dtype=np.int64,
            ),
            shape,
        )
    )
    return hash_image(metadata)
