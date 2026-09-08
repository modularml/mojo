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
"""A Module for convolutional layers."""

from __future__ import annotations

from typing import Literal

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.nn.module import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.graph.type import FilterLayout


class Conv2d(Module[[Tensor], Tensor]):
    """A 2D convolution layer.

    This is a Conv2d implementation that uses Tensor instead of Weight objects.

    Example:
        .. Skipped: an Apple GPU satisfies the ``accelerator_count()`` guard
           below but cannot run an FCRS-filter conv, whose only GPU path is
           cuDNN. Paired ``start``/``end`` rather than ``next`` so the skip
           also covers the invisible check, which reads ``result`` from this
           block. Remove both once KERN-3583 is fixed.
        .. skip: start if(__import__("sys").platform == "darwin", "FCRS conv is cuDNN-only (KERN-3583)")

        .. code-block:: python

            from max.driver import Accelerator, accelerator_count
            from max.dtype import DType
            from max.experimental.nn import Conv2d
            from max.experimental.tensor import Tensor

            conv = Conv2d(
                kernel_size=3,
                in_channels=3,
                out_channels=64,
                dtype=DType.float32,
                has_bias=True,
                permute=True,
            )
            x = Tensor.ones([1, 3, 32, 32], dtype=DType.float32)
            # permute=True uses the FCRS filter layout, which requires a GPU.
            if accelerator_count():
                result = conv(x.to(Accelerator()))

        .. invisible-code-block: python

            if accelerator_count():
                # permute=True: NCHW in -> NCHW out. 3x3 kernel, no padding,
                # so 32x32 -> 30x30 and channels go 3 -> 64.
                assert tuple(int(d) for d in result.shape) == (1, 64, 30, 30)

        .. skip: end

    """

    weight: Tensor
    """The weight tensor with shape [out_channels, in_channels // num_groups, kernel_height, kernel_width]."""

    bias: Tensor | Literal[0]
    """The bias tensor with shape [out_channels] (or 0 if bias is disabled)."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        in_channels: int,
        out_channels: int,
        dtype: DType | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        num_groups: int = 1,
        device: Device | DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ):
        """Initialize Conv2d layer.

        Args:
            kernel_size: Size of the convolving kernel. Can be a single int (square kernel) or tuple (height, width).
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for both weights and bias. In v3, this is optional as Tensor manages dtype automatically.
            stride: Stride of the convolution for height and width dimensions.
                Can be int (applied to both dimensions) or tuple (stride_h, stride_w). Default: 1
            padding: Padding added to input. Can be int (applied to all sides),
                tuple of 2 ints (pad_h, pad_w), or tuple of 4 ints (pad_top, pad_bottom, pad_left, pad_right) to support asymmetric padding. Default: 0
            dilation: Spacing between kernel elements for height and width dimensions.
                Can be int (applied to both dimensions) or tuple (dilation_h, dilation_w). Default: 1
            num_groups: Number of blocked connections from input channels to output channels.
                Input channels and output channels are divided into groups. Default: 1
            device: The target device for computation. In v3, this is optional as Tensor manages device automatically.
            has_bias: If true, adds a learnable bias vector to the layer.
                Defaults to :obj:`False`.
            permute: If true, permutes weights from PyTorch format to MAX format.
                PyTorch order: (out_channels, in_channels / num_groups, height, width).
                MAX API order: (height, width, in_channels / num_groups, out_channels).
                Defaults to :obj:`False`.
            name: Base name for weights. In v3, this is stored but not used for Weight naming.
                Defaults to :obj:`None`.
        """
        # Store configuration for easy reconstruction
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.num_groups = num_groups
        self.has_bias = has_bias
        self.name = name

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            kernel_height = kernel_width = kernel_size
            self.kernel_size = (kernel_size, kernel_size)
        else:
            kernel_height, kernel_width = kernel_size
            self.kernel_size = kernel_size

        self.weight = random.normal(
            [
                out_channels,
                in_channels // num_groups,
                kernel_height,
                kernel_width,
            ]
            if self.permute
            else [
                kernel_height,
                kernel_width,
                in_channels // num_groups,
                out_channels,
            ],
            dtype=self.dtype,
            device=self.device,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.bias = 0

        # Convert scalar parameters to tuples as needed
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            # Convert (pad_h, pad_w) to (pad_top, pad_bottom, pad_left, pad_right)
            pad_h, pad_w = padding
            padding = (pad_h, pad_h, pad_w, pad_w)

        self.padding = padding

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation

        if (
            isinstance(self.weight, Tensor)
            and hasattr(self.weight, "quantization_encoding")
            and self.weight.quantization_encoding is not None
        ):
            raise ValueError("Conv2d not implemented with weight quantization.")

    def forward(self, x: Tensor) -> Tensor:
        """Applies the 2D convolution to ``x``.

        Args:
            x: The input tensor, shaped
                ``[batch_size, in_channels, height, width]`` when
                ``permute`` is ``True`` and
                ``[batch_size, height, width, in_channels]`` when
                ``permute`` is ``False``.

        Returns:
            The output tensor, shaped
            ``[batch_size, out_channels, new_height, new_width]`` when
            ``permute`` is ``True`` and
            ``[batch_size, new_height, new_width, out_channels]`` when
            ``permute`` is ``False``.
        """
        # Move weight and bias to same device as input
        weight = self.weight.to(x.device)
        bias = self.bias.to(x.device) if isinstance(self.bias, Tensor) else None

        if self.permute:
            # Input: NCHW -> NHWC
            x = F.permute(x, [0, 2, 3, 1])

        output = F.conv2d(
            x,
            weight,
            self.stride,
            self.dilation,
            self.padding,
            self.num_groups,
            bias,
            filter_layout=FilterLayout.FCRS
            if self.permute
            else FilterLayout.RSCF,
        )

        if self.permute:
            # Output: NHWC -> NCHW
            output = F.permute(output, [0, 3, 1, 2])

        return output
