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
"""The vocoder's convolutions, expressed as matmuls instead of cuDNN calls.

Both layers here are exact rewrites of a convolution, and both exist because the
cuDNN paths behind ``max.nn.Conv1D`` and ``max.nn.ConvTranspose1d`` are
unsuitable for this decoder on sm_86. Measured on an A10G:

* Forward convolution is 6x to 470x slower than the same arithmetic as matmuls
  (``probe/probe_conv_perf.py``): a 7-tap 768-channel convolution over 512
  positions is 8.5 GFLOP and takes 210 ms, about 0.04 TFLOP/s, against 0.66 ms
  and 12.9 TFLOP/s for seven shifted matmuls. Across the decoder's 37
  convolutions that is the difference between 0.5x and roughly 10x realtime.
* Transposed convolution hardcodes ``CUDNN_CONVOLUTION_BWD_DATA_ALGO_0``
  (``nn/conv/conv_transpose.mojo:1855``) and its workspace request fails with
  ``CUDNN_STATUS_ALLOC_FAILED`` past ~64k output samples, capping the decoder at
  96 latent frames against a 689-frame denoising window.

Neither rewrite changes the arithmetic: a ``k``-tap convolution is a sum of ``k``
shifted matmuls, and a ``2s``-tap stride-``s`` transposed convolution is two
matmuls interleaved. Both also keep the data channels-last, so the whole decoder
runs without a single transpose.

Every matmul here goes through :func:`split_matmul`, which recovers the mantissa
bits TF32 discards.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor


def split_halves(t: Tensor) -> tuple[Tensor, Tensor]:
    """Splits a tensor into a bfloat16-exact part and its remainder."""
    hi = t.cast(DType.bfloat16).cast(t.dtype)
    return hi, t - hi


def split_product(x: tuple[Tensor, Tensor], w: tuple[Tensor, Tensor]) -> Tensor:
    """Multiplies two already-split operands, dropping the ``lo * lo`` term."""
    x_hi, x_lo = x
    w_hi, w_lo = w
    return (x_hi @ w_hi) + (x_hi @ w_lo) + (x_lo @ w_hi)


def split_matmul(x: Tensor, w: Tensor) -> Tensor:
    """Multiplies two float32 tensors without losing them to TF32.

    MAX runs wide float32 matmuls on TF32 tensor cores, keeping 10 mantissa bits
    of each operand instead of 24, and offers no way to opt out
    (``probe/probe_fp32_precision.py``). Left alone that costs the vocoder three
    orders of magnitude: 5e-3 from a float64 reference where torch reaches
    1.8e-6.

    Splitting each operand into a part the narrow format holds exactly plus a
    remainder, ``x = x_hi + x_lo``, expands the product into four terms of which
    the last is negligible:

        x @ w = x_hi @ w_hi + x_hi @ w_lo + x_lo @ w_hi + x_lo @ w_lo

    Measured at the vocoder's real shapes, the three retained terms give 3e-6
    against 7.7e-4 for the plain matmul, at 2.4x the time
    (``probe/probe_split_matmul.py``) -- still an order of magnitude faster than
    the cuDNN convolution this replaces.

    Narrower dtypes are passed straight through, since there are no bits to
    recover.
    """
    if x.dtype != DType.float32:
        return x @ w
    return split_product(split_halves(x), split_halves(w))


class ShiftedConv1d(Module[[Tensor], Tensor]):
    """A length-preserving dilated convolution as a sum of shifted matmuls.

    The weight keeps the ``(kernel, in, out)`` layout that ``max.nn.Conv1D``
    declares for channels-last data, so checkpoints converted for either layer
    are interchangeable.
    """

    weight: Tensor
    """The kernel, shaped ``(kernel_size, in_channels, out_channels)``."""

    bias: Tensor
    """The per-output-channel bias."""

    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
    ) -> None:
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.dilation = dilation
        # "Same" padding: the reference derives it as (k - 1) * dilation // 2,
        # which keeps the length unchanged for every dilation used here.
        self.padding = (kernel_size - 1) // 2 * dilation
        self.weight = Tensor.zeros([kernel_size, in_channels, out_channels])
        self.bias = Tensor.zeros([out_channels])

    def forward(self, x: Tensor) -> Tensor:
        """Convolves a channels-last signal, preserving its length.

        Args:
            x: Shape ``(batch, length, in_channels)``.

        Returns:
            Shape ``(batch, length, out_channels)``.
        """
        weight = self.weight.cast(x.dtype)
        bias = self.bias.cast(x.dtype)
        if self.kernel_size == 1:
            return split_matmul(x, weight[0]) + bias

        # `F.zeros` broadcasts a scalar rather than materializing the pad, which
        # this decoder needs: it peaks near the card's limit, and real zeros at
        # every one of its 37 convolutions is enough to exhaust an A10G.
        zeros = F.zeros(
            [x.shape[0], self.padding, x.shape[2]],
            dtype=x.dtype,
            device=x.device,
        )
        padded = F.concat([zeros, x, zeros], axis=1)
        # Split once, then slice both halves per tap. Splitting each tap's slice
        # instead would materialize two copies of the activation seven times
        # over, which at the last block's 271 MB tensors is enough to exhaust an
        # A10G.
        split = x.dtype == DType.float32
        halves = split_halves(padded) if split else (padded, padded)

        length = x.shape[1]
        acc = None
        for tap in range(self.kernel_size):
            window = slice(tap * self.dilation, tap * self.dilation + length)
            if split:
                term = split_product(
                    (halves[0][:, window, :], halves[1][:, window, :]),
                    split_halves(weight[tap]),
                )
            else:
                term = padded[:, window, :] @ weight[tap]
            acc = term if acc is None else acc + term
        assert acc is not None
        return acc + bias


class SubPixelConvTranspose1d(Module[[Tensor], Tensor]):
    """Upsamples by ``stride`` using a ``2 * stride``-tap transposed kernel.

    Writing PyTorch's definition

        out[o] = sum_i inp[i] @ W[:, :, k],   o = i * stride - pad + k

    and substituting ``m = o + pad``, ``q = m // stride``, ``r = m % stride``, the
    constraint ``0 <= m - i * stride < 2 * stride`` admits exactly two taps,
    ``i = q`` with ``k = r`` and ``i = q - 1`` with ``k = r + stride``:

        out[q * stride + r - pad]
            = inp[q] @ W[:, :, r] + inp[q - 1] @ W[:, :, r + stride]

    So the operator is two dense matmuls covering all ``stride`` output phases at
    once, one shifted a position against the other, then interleaved.

    Restricted to ``kernel == 2 * stride`` and ``padding == stride // 2`` with an
    even stride, which is what the vocoder's four upsampling blocks use and what
    makes the output exactly ``stride`` times longer with nothing to crop.
    """

    weight_lo: Tensor
    """Taps ``[0, stride)``, folded to ``(in, stride * out)``."""

    weight_hi: Tensor
    """Taps ``[stride, 2 * stride)``, folded to ``(in, stride * out)``."""

    bias: Tensor
    """The per-output-channel bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        if stride % 2:
            raise ValueError(f"stride must be even, got {stride}")
        self.out_channels = out_channels
        self.stride = stride
        # The two phase-major weight halves: taps [0, stride) and
        # [stride, 2 * stride), each folded to (in, stride * out) so one matmul
        # produces every phase at once.
        self.weight_lo = Tensor.zeros([in_channels, stride * out_channels])
        self.weight_hi = Tensor.zeros([in_channels, stride * out_channels])
        self.bias = Tensor.zeros([out_channels])

    def forward(self, x: Tensor) -> Tensor:
        """Upsamples a channels-last signal.

        Args:
            x: Shape ``(batch, length, in_channels)``.

        Returns:
            Shape ``(batch, length * stride, out_channels)``.
        """
        batch, length = x.shape[0], x.shape[1]
        stride, out_channels = self.stride, self.out_channels

        lo = split_matmul(x, self.weight_lo.cast(x.dtype))
        hi = split_matmul(x, self.weight_hi.cast(x.dtype))
        # `hi` is the tap set fed by the *previous* input position, so shifting it
        # forward one place lines both halves up on the same output position. The
        # extra row at each end covers q = length (needed because the crop below
        # starts at pad > 0) and q = 0 (which has no predecessor).
        pad = F.broadcast_to(
            F.constant(0, x.dtype, x.device),
            [batch, 1, stride * out_channels],
        )
        acc = F.concat([lo, pad], axis=1) + F.concat([pad, hi], axis=1)

        dense = acc.reshape((batch, (length + 1) * stride, out_channels))
        # m = o + pad with pad = stride // 2, so the valid output window starts
        # `stride // 2` positions in and runs for exactly length * stride.
        start = stride // 2
        return dense[:, start : start + length * stride, :] + self.bias.cast(
            x.dtype
        )
