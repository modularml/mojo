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
"""The DAC-style Flow-VAE waveform decoder of MiniMax Music 3.

Every convolution here runs as a matmul rather than through cuDNN -- see
``layers/conv.py`` for the measurements that forced it, on both speed and on
cuDNN's workspace failure at long outputs. A consequence worth stating is that
the whole decoder stays channels-last, so there is not a single transpose in it.

This component runs in float32, not bfloat16: ``ConvTranspose1d`` produced
non-finite output in bfloat16 on sm_86 at all four upsampling shapes, and 25 Hz
latents expanded 512x give the vocoder the widest dynamic range in the pipeline.
Its weights are only 0.10 GiB, so float32 costs almost nothing.

Accuracy is bounded by the platform rather than by this code: MAX runs float32
matmul and convolution at TF32 on sm_86 with no opt-out
(``probe/probe_fp32_precision.py``), which is also PyTorch's own default on this
card. See ``FINDINGS.md``.
"""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module, ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorType

from ..layers.conv import ShiftedConv1d, SubPixelConvTranspose1d
from ..layers.snake import Snake1d
from ..model_config import VocoderConfig


class _ResidualUnit(Module[[Tensor], Tensor]):
    """Dilated 7-tap convolution plus a 1-tap mixer, added to its input."""

    def __init__(self, dim: int, dilation: int) -> None:
        self.snake1 = Snake1d(dim)
        self.conv1 = ShiftedConv1d(
            kernel_size=7,
            in_channels=dim,
            out_channels=dim,
            dilation=dilation,
        )
        self.snake2 = Snake1d(dim)
        self.conv2 = ShiftedConv1d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class _UpsampleBlock(Module[[Tensor], Tensor]):
    """One upsampling stage: transposed convolution then three residual units."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        self.snake1 = Snake1d(in_channels)
        # kernel = 2 * stride with padding = stride / 2 makes the output exactly
        # `stride` times longer, so the four blocks give 8*8*4*2 = 512 samples
        # per latent frame with no cropping anywhere.
        self.conv_t1 = SubPixelConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
        )
        self.res_unit1 = _ResidualUnit(out_channels, 1)
        self.res_unit2 = _ResidualUnit(out_channels, 3)
        self.res_unit3 = _ResidualUnit(out_channels, 9)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_t1(self.snake1(x))
        return self.res_unit3(self.res_unit2(self.res_unit1(x)))


class Vocoder(Module[[Tensor], Tensor]):
    """Decodes Flow-VAE latents into a stereo waveform.

    The two audio channels are decoded as two independent folded streams of
    ``latent_channels // 2``, so the batch axis carries ``2 * batch`` rows
    through the whole decoder and is unfolded only at the end.
    """

    def __init__(self, config: VocoderConfig) -> None:
        self.config = config

        half = config.latent_channels // 2
        self.dec_in_proj = ShiftedConv1d(
            kernel_size=1,
            in_channels=half,
            out_channels=config.decoder_input_dim,
        )
        self.conv_in = ShiftedConv1d(
            kernel_size=7,
            in_channels=config.decoder_input_dim,
            out_channels=config.decoder_hidden_dim,
        )

        blocks = []
        out_channels = config.decoder_hidden_dim
        for index, stride in enumerate(config.upsampling_ratios):
            in_channels = config.decoder_hidden_dim // (2**index)
            out_channels = config.decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(_UpsampleBlock(in_channels, out_channels, stride))
        self.blocks = ModuleList(blocks)
        self.snake_out = Snake1d(out_channels)
        self.conv_out = ShiftedConv1d(
            kernel_size=7,
            in_channels=out_channels,
            out_channels=1,
        )

    def input_types(self, frames: int) -> tuple[TensorType, ...]:
        """The graph input for a latent sequence of ``frames`` frames.

        The length is a compile-time choice the caller owns -- see
        :mod:`~max.pipelines.architectures.minimax_music3.vocode` for why the
        decoder is compiled at a fixed window rather than at the clip's length.
        """
        return (
            TensorType(
                self.conv_in.weight.dtype,
                [1, self.config.latent_channels, frames],
                device=self.device,
            ),
        )

    def forward(self, latents: Tensor) -> Tensor:
        """Decodes latents into a waveform.

        Args:
            latents: Flow-VAE latents of shape
                ``(batch, latent_channels, length)``, channel-first as the
                denoiser produces them.

        Returns:
            The stereo waveform, shape ``(batch, 2, length * 512)``, in
            ``[-1, 1]``.
        """
        batch = latents.shape[0]
        half = self.config.latent_channels // 2
        # Fold the stereo pair into the batch axis while still channel-first,
        # then transpose once into channels-last for the whole decoder body.
        x = latents.reshape([batch * 2, half, latents.shape[2]])
        x = F.transpose(x, 1, 2).cast(self.conv_in.weight.dtype)

        x = self.conv_in(self.dec_in_proj(x))
        for block in self.blocks:
            x = block(x)
        wave = F.tanh(self.conv_out(self.snake_out(x)))
        # (2 * batch, samples, 1) -> (batch, 2, samples)
        return wave.reshape([batch, 2, wave.shape[1]])
