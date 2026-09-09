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

from max.driver import accelerator_api
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Conv2d, GroupNorm, Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef


class ResnetBlock2D(Module[[Tensor, Tensor | None], Tensor]):
    """Residual block for 2D VAE decoder.

    This module implements a residual block with two convolutional layers,
    group normalization, and optional shortcut connection. It supports
    time embedding conditioning and configurable activation functions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int | None,
        groups: int,
        groups_out: int,
        eps: float = 1e-6,
        non_linearity: str = "silu",
        use_conv_shortcut: bool = False,
        conv_shortcut_bias: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize ResnetBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            temb_channels: Number of time embedding channels (None if not used).
            groups: Number of groups for first GroupNorm.
            groups_out: Number of groups for second GroupNorm.
            eps: Epsilon value for GroupNorm layers.
            non_linearity: Activation function name (e.g., "silu").
            use_conv_shortcut: Whether to use convolutional shortcut.
            conv_shortcut_bias: Whether to use bias in shortcut convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
        )

        self.conv1 = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.norm2 = GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
        )

        self.conv2 = Conv2d(
            kernel_size=3,
            in_channels=out_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.conv_shortcut: Conv2d | None = None
        if self.use_conv_shortcut or in_channels != out_channels:
            self.conv_shortcut = Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=conv_shortcut_bias,
                device=device,
                permute=True,
            )

    def _use_fused_conv_residual(self) -> bool:
        """Check if fused conv2d + residual add should be used.

        Returns True when running on NVIDIA GPU with matching channel counts
        (so the shortcut is identity, not a projection conv).
        """
        return (
            self.in_channels == self.out_channels
            and self.conv_shortcut is None
            and isinstance(self.conv2.device, DeviceRef)
            and self.conv2.device.is_gpu()
            and accelerator_api() == "cuda"
        )

    def forward(self, x: Tensor, temb: Tensor | None = None) -> Tensor:
        """Apply ResnetBlock2D forward pass.

        Args:
            x: Input tensor of shape [N, C, H, W].
            temb: Optional time embedding tensor (currently unused).

        Returns:
            Output tensor of shape [N, C_out, H, W] with residual connection.
        """
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )

        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))

        if self._use_fused_conv_residual():
            # Fused conv2d + TMA residual add + bias in a single kernel.
            # Permute inputs to NHWC for the custom op.
            h_nhwc = F.permute(h, [0, 2, 3, 1])
            shortcut_nhwc = F.permute(shortcut, [0, 2, 3, 1])
            weight = self.conv2.weight
            bias = self.conv2.bias
            # conv2 is always created with has_bias=True, so bias
            # is always a Tensor (never the literal 0 fallback).
            assert isinstance(bias, Tensor), "conv2 bias must be a Tensor"

            pad = self.conv2.padding
            stride = self.conv2.stride

            result_nhwc = F.custom(
                "conv2d_residual_add",
                device=h.device,
                values=[h_nhwc, weight, shortcut_nhwc, bias],
                out_types=[shortcut_nhwc.type],
                parameters={
                    "stride_h": stride[0],
                    "stride_w": stride[1],
                    "pad_top": pad[0],
                    "pad_bottom": pad[1],
                    "pad_left": pad[2],
                    "pad_right": pad[3],
                    "has_bias": True,
                },
            )[0]

            # Permute back to NCHW.
            return F.permute(result_nhwc, [0, 3, 1, 2])
        else:
            h = self.conv2(h)
            return h + shortcut
