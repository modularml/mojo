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

"""Downsampling utilities for MAX framework."""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Conv2d, Module
from max.experimental.nn.norm import LayerNorm, RMSNorm
from max.experimental.tensor import Tensor
from max.graph import DeviceRef


class Downsample2D(Module[[Tensor], Tensor]):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        kernel_size (`int`, default `3`):
            kernel size for the convolution.
        norm_type (`str`, optional):
            normalization type. Supported: "ln_norm" (LayerNorm), "rms_norm" (RMSNorm), or None.
        eps (`float`, optional):
            epsilon for normalization. Defaults to 1e-5 for LayerNorm, 1e-6 for RMSNorm.
        elementwise_affine (`bool`, optional):
            elementwise affine for normalization. Only used for LayerNorm. Defaults to True.
        bias (`bool`, default `True`):
            whether to use bias in the convolution.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        padding: int = 1,
        kernel_size: int = 3,
        norm_type: str | None = None,
        eps: float | None = None,
        elementwise_affine: bool | None = None,
        bias: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize 2D downsampling module.

        Args:
            channels: Number of input channels.
            use_conv: Whether to use convolution for downsampling.
            out_channels: Number of output channels. If None, uses channels.
            padding: Padding for the convolution.
            kernel_size: Kernel size for the convolution.
            norm_type: Normalization type ("ln_norm", "rms_norm", or None).
            eps: Epsilon for normalization. Defaults to 1e-5 for LayerNorm, 1e-6 for RMSNorm.
            elementwise_affine: Elementwise affine for normalization. Only used for LayerNorm.
            bias: Whether to use bias in the convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        self.norm: LayerNorm | RMSNorm | None = None
        if norm_type == "ln_norm":
            self.norm = LayerNorm(
                dim=channels,
                eps=eps or 1e-5,
                elementwise_affine=elementwise_affine
                if elementwise_affine is not None
                else True,
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(dim=channels, eps=eps or 1e-6)
        elif norm_type is None:
            pass
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        self.conv: Conv2d | None = None
        if use_conv:
            self.conv = Conv2d(
                kernel_size=kernel_size,
                in_channels=self.channels,
                out_channels=self.out_channels,
                dtype=dtype,
                stride=stride,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )
        else:
            if self.channels != self.out_channels:
                raise ValueError(
                    f"When use_conv=False, channels must equal out_channels. "
                    f"Got channels={self.channels}, out_channels={self.out_channels}"
                )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply 2D downsampling with optional convolution.

        Args:
            hidden_states: Input tensor of shape [N, C, H, W].

        Returns:
            Downsampled tensor of shape [N, C_out, H//2, W//2].
        """
        if self.norm is not None:
            hidden_states = F.permute(hidden_states, [0, 2, 3, 1])
            hidden_states = self.norm(hidden_states)
            hidden_states = F.permute(hidden_states, [0, 3, 1, 2])

        if self.use_conv and self.padding == 0:
            paddings = [0, 0, 0, 0, 0, 1, 0, 1]
            hidden_states = F.pad(
                hidden_states, paddings=paddings, mode="constant", value=0
            )

        if self.use_conv:
            assert self.conv is not None
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = F.permute(hidden_states, [0, 2, 3, 1])
            hidden_states = F.avg_pool2d(
                hidden_states,
                kernel_size=(2, 2),
                stride=2,
                padding=0,
            )
            hidden_states = F.permute(hidden_states, [0, 3, 1, 2])

        return hidden_states
