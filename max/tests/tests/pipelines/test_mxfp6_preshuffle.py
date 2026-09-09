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
"""Pins the host MXFP6 B preshuffle to the layout the AMD preb kernel reads.

The vectorized reshape-and-transpose in ``block_scaled_preshuffle`` has to agree with
``Shuffler.b_plane_byte_off`` in
``max/kernels/src/linalg/matmul/gpu/amd/block_scaled_preshuffle_layouts.mojo``, which
is what the kernel addresses each lane fragment through. This module
transcribes that offset formula scalar-for-scalar and checks the fast path
against it, so a change to either side that breaks the agreement fails here
rather than as wrong logits on MI355.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.weights.block_scaled_preshuffle import (
    MXFP6_LANE_BYTES,
    _shuffle_b_5d,
    _shuffle_b_planes,
)

_MFMA_MN_LANES = 16
_MFMA_K_LANES = 4
_MFMA_LANE_BYTES = 16


def b_plane_byte_off(
    n: int, k_byte: int, plane: int, *, k_bytes: int, lane_bytes: int
) -> int:
    """Scalar transcription of ``Shuffler.b_plane_byte_off`` for one expert."""
    mfma_k_bytes = _MFMA_K_LANES * lane_bytes
    tile_bytes = _MFMA_MN_LANES * mfma_k_bytes
    plane_width = min(_MFMA_LANE_BYTES, lane_bytes - plane * _MFMA_LANE_BYTES)
    plane_base = (
        _MFMA_MN_LANES
        * _MFMA_K_LANES
        * (lane_bytes - max(0, lane_bytes - plane * _MFMA_LANE_BYTES))
    )
    k0_count = k_bytes // mfma_k_bytes

    n0, nlane = divmod(n, _MFMA_MN_LANES)
    k0, k_in_tile = divmod(k_byte, mfma_k_bytes)
    klane = k_in_tile // lane_bytes

    return (
        n0 * (k0_count * tile_bytes)
        + k0 * tile_bytes
        + plane_base
        + klane * (_MFMA_MN_LANES * plane_width)
        + nlane * plane_width
    )


def reference_shuffle(src: np.ndarray) -> np.ndarray:
    """Permutes ``[N, K_BYTES]`` one lane fragment at a time, via the formula."""
    n_rows, k_bytes = src.shape
    dst = np.zeros(n_rows * k_bytes, dtype=np.uint8)
    for n in range(n_rows):
        for k_byte in range(0, k_bytes, MXFP6_LANE_BYTES):
            for plane in range(2):
                width = min(
                    _MFMA_LANE_BYTES,
                    MXFP6_LANE_BYTES - plane * _MFMA_LANE_BYTES,
                )
                offset = b_plane_byte_off(
                    n,
                    k_byte,
                    plane,
                    k_bytes=k_bytes,
                    lane_bytes=MXFP6_LANE_BYTES,
                )
                start = k_byte + plane * _MFMA_LANE_BYTES
                dst[offset : offset + width] = src[n, start : start + width]
    return dst


# K_BYTES must be a whole number of MFMA K tiles: 4 lanes * 24 bytes = 96.
@pytest.mark.parametrize(
    ("n_rows", "k_bytes"),
    [(16, 96), (16, 384), (32, 192), (48, 96), (64, 288), (128, 96)],
)
def test_plane_shuffle_matches_kernel_offsets(
    n_rows: int, k_bytes: int
) -> None:
    rng = np.random.default_rng(n_rows * 10_000 + k_bytes)
    src = rng.integers(0, 256, (n_rows, k_bytes), dtype=np.uint8)

    dst = np.zeros_like(src)
    _shuffle_b_planes(src, dst)

    np.testing.assert_array_equal(dst.reshape(-1), reference_shuffle(src))


def test_plane_shuffle_is_a_permutation() -> None:
    """No byte may be dropped or duplicated: the two buffers are the same size,
    so a layout bug shows up as a changed multiset, not a size mismatch."""
    rng = np.random.default_rng(11)
    src = rng.integers(0, 256, (64, 288), dtype=np.uint8)
    dst = np.zeros_like(src)
    _shuffle_b_planes(src, dst)
    np.testing.assert_array_equal(
        np.bincount(dst.reshape(-1), minlength=256),
        np.bincount(src.reshape(-1), minlength=256),
    )


def test_plane_zero_is_byte_identical_to_the_fp4_layout() -> None:
    """Plane 0 holds the first 16 bytes of each fragment in the FP4 order.

    The two layouts coincide there because a 16-byte plane is exactly one FP4
    lane fragment; only the 8-byte remainder is new in FP6. Comparing them
    catches a plane-ordering mistake that a self-consistent round trip cannot.
    """
    rng = np.random.default_rng(3)
    n_rows, tiles = 32, 3
    src = rng.integers(
        0,
        256,
        (n_rows, tiles * _MFMA_K_LANES * MXFP6_LANE_BYTES),
        dtype=np.uint8,
    )

    planes = np.zeros_like(src)
    _shuffle_b_planes(src, planes)

    # The same lane fragments truncated to 16 bytes, laid out by the FP4 path.
    truncated = src.reshape(n_rows, -1, MXFP6_LANE_BYTES)[
        ..., :_MFMA_LANE_BYTES
    ].reshape(n_rows, -1)
    fp4_style = np.zeros_like(truncated)
    _shuffle_b_5d(np.ascontiguousarray(truncated), fp4_style)

    tile_bytes = _MFMA_MN_LANES * _MFMA_K_LANES * MXFP6_LANE_BYTES
    plane0_bytes = _MFMA_MN_LANES * _MFMA_K_LANES * _MFMA_LANE_BYTES
    plane0 = planes.reshape(-1, tile_bytes)[:, :plane0_bytes]

    np.testing.assert_array_equal(plane0.reshape(-1), fp4_style.reshape(-1))
