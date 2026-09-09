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
"""The three weight transforms the vocoder cannot run without.

MAX has no ``ConvTranspose1d`` GPU kernel and no weight-norm fusion, so the
vocoder's kernels are rewritten as they load: weight norm is folded into a plain
kernel, every convolution is permuted to the tap-major layout the port's
convolutions read, and each transposed convolution becomes a pair of sub-pixel
matmul operands. Each is checked here against an independent reference written
in the source layout, so a transposition error cannot pass by agreeing with
itself.

Synthetic arrays only: no GPU, no checkpoint, no network.
"""

from __future__ import annotations

import numpy as np
from max.pipelines.architectures.minimax_music3.weight_adapters import (
    _conv_to_max_layout,
    _fold_weight_norm,
    _split_subpixel,
)


def test_conv_layout_is_tap_major() -> None:
    """``(out, in, k)`` becomes ``(k, in, out)``: one matmul per tap."""
    kernel = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    permuted = _conv_to_max_layout(kernel)

    assert permuted.shape == (5, 3, 2)
    assert permuted.flags.c_contiguous
    for tap in range(5):
        for inp in range(3):
            for out in range(2):
                assert permuted[tap, inp, out] == kernel[out, inp, tap]


def test_fold_weight_norm_reconstructs_the_kernel() -> None:
    """``weight_norm(dim=0)`` reduces every axis but the first, so the norm is
    taken over axes (1, 2) whatever axis 0 happens to mean."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((4, 3, 5)).astype(np.float32)
    g = rng.standard_normal((4, 1, 1)).astype(np.float32)

    folded = _fold_weight_norm(g, v)

    norm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=(1, 2), keepdims=True))
    np.testing.assert_allclose(folded, g * v / norm, rtol=1e-6)
    assert folded.dtype == v.dtype


def test_fold_weight_norm_scales_each_row_independently() -> None:
    """A per-row magnitude must not leak across rows -- the failure a norm taken
    over the wrong axes would produce."""
    v = np.ones((2, 1, 4), np.float32)
    g = np.array([[[1.0]], [[10.0]]], np.float32)

    folded = _fold_weight_norm(g, v)

    np.testing.assert_allclose(folded[0], 0.5, rtol=1e-6)
    np.testing.assert_allclose(folded[1], 5.0, rtol=1e-6)


def test_split_subpixel_shapes() -> None:
    """A stride-*s* transposed kernel is stored as ``2 * s`` taps, and becomes
    two operands that each emit every output phase in one matmul."""
    in_channels, out_channels, stride = 3, 4, 8
    kernel = np.zeros((in_channels, out_channels, 2 * stride), np.float32)

    low, high = _split_subpixel(kernel)

    assert low.shape == high.shape == (in_channels, stride * out_channels)
    assert low.flags.c_contiguous and high.flags.c_contiguous


def test_split_subpixel_places_every_tap() -> None:
    """The halves are the two tap sets, laid out phase-major: element
    ``(i, phase * out + o)`` of the low half is tap ``phase`` of ``(i, o)``."""
    in_channels, out_channels, stride = 2, 3, 4
    kernel = np.arange(
        in_channels * out_channels * 2 * stride, dtype=np.float32
    ).reshape(in_channels, out_channels, 2 * stride)

    low, high = _split_subpixel(kernel)

    for i in range(in_channels):
        for phase in range(stride):
            for o in range(out_channels):
                column = phase * out_channels + o
                assert low[i, column] == kernel[i, o, phase]
                assert high[i, column] == kernel[i, o, stride + phase]


def test_split_subpixel_matches_a_transposed_convolution() -> None:
    """The rewrite is exact, not approximate.

    A stride-*s* ``ConvTranspose1d`` with a ``2 * s`` kernel writes each input
    frame across two output blocks; summing the low half at a frame with the
    high half at its predecessor reproduces the same signal, which is the
    identity the whole substitution rests on.
    """
    rng = np.random.default_rng(0)
    in_channels, out_channels, stride, frames = 3, 2, 4, 6
    kernel = rng.standard_normal(
        (in_channels, out_channels, 2 * stride)
    ).astype(np.float32)
    signal = rng.standard_normal((frames, in_channels)).astype(np.float32)

    # The reference, straight from the definition of a transposed convolution.
    expected = np.zeros((frames * stride + stride, out_channels), np.float32)
    for frame in range(frames):
        for tap in range(2 * stride):
            expected[frame * stride + tap] += signal[frame] @ kernel[:, :, tap]

    low, high = _split_subpixel(kernel)
    blocks = (signal @ low).reshape(frames, stride, out_channels)
    tails = (signal @ high).reshape(frames, stride, out_channels)
    produced = blocks.copy()
    produced[1:] += tails[:-1]

    np.testing.assert_allclose(
        produced.reshape(-1, out_channels),
        expected[: frames * stride],
        rtol=1e-5,
        atol=1e-5,
    )
