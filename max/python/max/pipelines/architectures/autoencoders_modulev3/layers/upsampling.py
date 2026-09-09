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

"""Upsampling utilities for MAX framework."""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Conv2d, Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorValue, TensorValueLike


def interpolate_2d_nearest(
    x: TensorValueLike,
    scale_factor: int = 2,
) -> TensorValue:
    """Upsamples a 2D tensor using nearest-neighbor interpolation.

    This is a workaround implementation because MAX framework's ops.resize
    does not support NEAREST mode (only BICUBIC is currently supported).
    The workaround uses reshape and broadcast operations to achieve
    nearest-neighbor upsampling by a factor of 2.

    This function works in both Graph context and eager execution contexts,
    compatible with Module API style.

    Note:
        This workaround can be removed once ops.resize supports NEAREST mode.

    Args:
        x: Input tensor of shape [N, C, H, W] in NCHW format.
            Can be Tensor or TensorValue.
        scale_factor: Upsampling factor. Currently only 2 is supported.
            Default: 2

    Returns:
        Upsampled tensor of shape [N, C, H*scale_factor, W*scale_factor].

    Raises:
        ValueError: If input tensor doesn't have rank 4.
        NotImplementedError: If scale_factor is not 2.
    """
    x = TensorValue(x)

    if x.rank != 4:
        raise ValueError(f"Input tensor must have rank 4, got {x.rank}")

    if scale_factor != 2:
        raise NotImplementedError(
            f"Only scale_factor=2 is currently supported, got {scale_factor}"
        )

    n, c, h, w = x.shape
    target_shape = [n, c, h * scale_factor, w * scale_factor]

    # Reshape: [N, C, H, W] -> [N, C, H, 1, W, 1]
    x_reshaped = x.reshape([n, c, h, 1, w, 1])

    ones_scalar = F.constant(1.0, dtype=x.dtype, device=x.device)
    ones = ones_scalar.broadcast_to([1, 1, 1, scale_factor, 1, scale_factor])

    # Broadcast: [N, C, H, 1, W, 1] * [1, 1, 1, 2, 1, 2] -> [N, C, H, 2, W, 2]
    x_expanded = x_reshaped * ones

    # Reshape: [N, C, H, 2, W, 2] -> [N, C, H*2, W*2]
    return x_expanded.reshape(target_shape)


class Upsample2D(Module[[Tensor], Tensor]):
    """2D upsampling module with optional convolution.

    This module performs 2D upsampling using nearest-neighbor interpolation
    (via interpolate_2d_nearest function) followed by an optional convolution layer.

    This version uses Tensor instead of TensorValue
    """

    conv: Conv2d | None
    """Optional Conv2d layer applied after upsampling."""

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: int | None = None,
        name: str = "conv",
        kernel_size: int | None = None,
        padding: int = 1,
        bias: bool = True,
        interpolate: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize 2D upsampling module.

        Args:
            channels: Number of input channels.
            use_conv: Whether to apply a convolution after upsampling.
            use_conv_transpose: Whether to use transposed convolution (not supported yet).
            out_channels: Number of output channels. If None, uses channels.
            name: Name for the convolution layer (unused, kept for compatibility).
            kernel_size: Kernel size for the convolution.
            padding: Padding for the convolution.
            bias: Whether to use bias in the convolution.
            interpolate: Whether to perform interpolation upsampling.
            device: Device reference (optional).
            dtype: Data type (optional).
        """
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.interpolate = interpolate
        self.device = device
        self.dtype = dtype

        if use_conv_transpose:
            raise NotImplementedError(
                "Upsample2D does not support use_conv_transpose=True yet."
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            self.conv = Conv2d(
                kernel_size=kernel_size,
                in_channels=self.channels,
                out_channels=self.out_channels,
                dtype=dtype,
                stride=1,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )
        else:
            self.conv = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply 2D upsampling with optional convolution.

        Args:
            x: Input tensor of shape [N, C, H, W].

        Returns:
            Upsampled tensor, optionally convolved.
        """
        if self.interpolate:
            # ``interpolate_2d_nearest`` drops to a raw graph ``TensorValue``
            # (its ``.device`` is a ``DeviceRef``). Re-lift to an experimental
            # ``Tensor`` so a downstream ``Conv2d`` — which colocates its weight
            # via ``weight.to(x.device)`` — sees a concrete device like every
            # other conv in the decoder.
            x = Tensor.from_graph_value(
                interpolate_2d_nearest(x, scale_factor=2)
            )

        if self.use_conv and self.conv is not None:
            x = self.conv(x)

        return x
