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

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.graph.type import ConvInputLayout, FilterLayout

from .layer import Module


@dataclass
class ConvTranspose1d(Module):
    """A 1D transposed convolution operator over an input image composed of several input planes.

    When called, ``ConvTranspose1d`` accepts a :class:`~max.graph.TensorValue`
    of shape ``(batch, length, in_channels)`` and returns a
    :class:`~max.graph.TensorValue` of shape ``(batch, new_length,
    out_channels)``. If ``permute=True``, the input and output follow PyTorch
    channel-first layout: ``(batch, in_channels, length)`` and ``(batch,
    out_channels, new_length)``.

    .. code-block:: python

        from max.driver import Accelerator, CPU, accelerator_count
        from max.dtype import DType
        from max.graph import DeviceRef
        from max.nn import ConvTranspose1d

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        conv = ConvTranspose1d(
            length=3,
            in_channels=64,
            out_channels=128,
            dtype=DType.float32,
            stride=1,
            padding=0,
            output_padding=0,
            has_bias=False,
            name="conv_transpose1d_weight",
            device=device_ref,
        )
    """

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    weight: Weight
    """The weight matrix stored on CPU with shape (kernel_length, out_channels, in_channels).
    Model init moves the weight to :obj:`device`."""

    stride: tuple[int, int]
    """Controls the stride for the cross-correlation."""

    padding: tuple[int, int, int, int]
    """Controls the amount of padding applied before and after the input for depth, height, and width dimensions."""

    dilation: tuple[int, int]
    """Not implemented yet. Assuming dilation = 1 for now."""

    output_padding: tuple[int, int]
    """Additional size added to one side of the output shape. Default: 0"""

    permute: bool
    """bool controls whether self.weight is permuted from PyTorch order to max order.
    PyTorch order is: (in_channels, out_channels, kernel_length)
    Max API order: (kernel_length, out_channels, in_channels). """

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_channels,).
    Model init moves the bias to :obj:`device` if present."""

    def __init__(
        self,
        length: int,
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int, int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        output_padding: int | tuple[int, int] = 0,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ) -> None:
        """Initializes the ConvTranspose1d layer with weights and optional bias.

        Args:
            length: The length of the convolution kernel.
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for weights and bias.
            stride: Stride of the convolution. Default: 1.
            padding: Padding added to input. Default: 0.
            dilation: Spacing between kernel elements. Default: 1.
            output_padding: Additional size added to output shape. Default: 0.
            device: The target device for computation.
            has_bias: When True, adds a bias vector. Default: False.
            permute: Whether to permute weights between PyTorch and MAX format.
            name: Base name for weights.
        """
        super().__init__()

        self.kernel_length = length
        self.device = device
        self.permute = permute

        if self.permute:
            self.weight = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[
                    in_channels,
                    out_channels,
                    length,
                ],
                device=self.device or DeviceRef.CPU(),
            )
        else:
            self.weight = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[
                    length,
                    out_channels,
                    in_channels,
                ],
                device=self.device or DeviceRef.CPU(),
            )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_channels,),
                device=self.device or DeviceRef.CPU(),
            )

        # These need to be casted as the underlying ops.conv3d call
        # expects them to only be tuple types.
        if isinstance(stride, int):
            stride = (1, stride)
        self.stride = stride

        if isinstance(output_padding, int):
            output_padding = (0, output_padding)
        self.output_padding = output_padding

        if isinstance(padding, int):
            padding = (
                0,
                0,
                padding,
                padding,
            )
        self.padding = padding

        if isinstance(dilation, int):
            dilation = (1, dilation)
        self.dilation = dilation

        if (
            isinstance(self.weight, Weight)
            and self.weight.quantization_encoding is not None
        ):
            raise ValueError(
                "ConvTranspose1d not implemented with weight quantization."
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applied ConvTranspose1d to input ``x``. Permutes pytorch weights to match MAX API if ``permute=True``.

        Args:
            x: a tensor of shape ``(batch_size, length, in_channels)``
            if self.permute, then input is of shape: ``(batch_size, in_channels, length)``
            and will be permuted to match max's expected input shape.
            Also, self.weight will be permuted from (kernel_length, in_channels, out_channels) to
            (in_channels, out_channels, kernel_length)

        Returns:
             a tensor of shape (batch_size, new_length, out_channels).
             if self.permute, then the output shape will be (batch_size, out_channels, new_length)
        """
        weight: TensorValue = self.weight

        if self.permute:
            # Use Pyotorch and CuDNN layout.
            # Reshape (batch_size, in_channels, length) to [batch_size, in_channels, height=1, length].
            x = ops.unsqueeze(x, 2)
            # Reshape (in_channels, out_channels, kernel_length) to [in_channels, out_channels, kernel_height=1, kernel_length,].
            weight = ops.unsqueeze(self.weight, 2)
            # # [batch_size, in_channels, height=1, length] to (batch_size, height, length, in_channels)
            # x = ops.permute(x, [0, 2, 3, 1])
            # # (in_channels, out_channels, kernel_height, kernel_length) to [kernel_height=1, kernel_length, out_channels, in_channels]
            # weight = ops.permute(weight, [2, 3, 1, 0])
        else:
            # Reshape (batch_size, length, in_channels) to [batch_size, height=1, length, in_channels].
            x = ops.unsqueeze(x, 1)
            # Reshape [kernel_length, in_channels, out_channels] to [kernel_height=1, kernel_length, out_channels, in_channels].
            weight = ops.unsqueeze(weight, 0)

        res = ops.conv2d_transpose(
            x=x,
            filter=weight,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            output_paddings=self.output_padding,
            bias=self.bias,
            input_layout=ConvInputLayout.NCHW
            if self.permute
            else ConvInputLayout.NHWC,
            filter_layout=FilterLayout.CFRS
            if self.permute
            else FilterLayout.RSCF,
        )

        if self.permute:
            # # permute output from [batch_size, height=1, new_length, out_channels] to (batch_size, out_channels, height=1, new_length).
            # res = ops.permute(res, [0, 3, 1, 2])
            # Reshape  (batch_size, out_channels, height=1, new_length). to [batch_size, out_channels, new_length].
            res = ops.squeeze(res, 2)
        else:
            # Reshape [batch_size, height=1, new_length, out_channels] to [batch_size, new_length, out_channels].
            res = ops.squeeze(res, 1)
        return res


class WeightNormConvTranspose1d(Module):
    """A 1D transposed convolution operator over an input image composed of several input planes.

    MAX implements weight normalization as described in `Weight
    Normalization <https://arxiv.org/abs/1602.07868>`_. Weight normalization
    reparameterizes weights in terms of a direction vector ``v`` and a
    magnitude scalar ``g``. This can help improve optimization by decoupling
    the length and direction of weight vectors.

    When called, ``WeightNormConvTranspose1d`` accepts a
    :class:`~max.graph.TensorValue` of shape ``(batch, length, in_channels)``
    and returns a :class:`~max.graph.TensorValue` of shape ``(batch,
    new_length, out_channels)``. If ``permute=True``, the input and output
    follow PyTorch channel-first layout: ``(batch, in_channels, length)`` and
    ``(batch, out_channels, new_length)``.

    .. code-block:: python

        from max.driver import Accelerator, CPU, accelerator_count
        from max.dtype import DType
        from max.graph import DeviceRef
        from max.nn import WeightNormConvTranspose1d

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        conv = WeightNormConvTranspose1d(
            length=3,
            in_channels=64,
            out_channels=128,
            dtype=DType.float32,
            stride=1,
            padding=0,
            output_padding=0,
            has_bias=False,
            device=device_ref,
        )
    """

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    conv: ConvTranspose1d
    """The underlying ConvTranspose1d layer."""

    weight_g: Weight
    """The magnitude parameter g for weight normalization."""

    weight_v: Weight
    """The direction parameter v for weight normalization."""

    def __init__(
        self,
        length: int,
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int, int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        output_padding: int | tuple[int, int] = 0,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ) -> None:
        """Initializes the WeightNormConvTranspose1d layer.

        Args:
            length: The length of the convolution kernel.
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for weights and bias.
            stride: Stride of the convolution. Default: 1.
            padding: Padding added to input. Default: 0.
            dilation: Spacing between kernel elements. Default: 1.
            output_padding: Additional size added to output shape. Default: 0.
            device: The target device for computation.
            has_bias: When True, adds a bias vector. Default: False.
            permute: Whether to permute weights between PyTorch and MAX format.
            name: Base name for weights.
        """
        super().__init__()

        # Initialize the conv layer
        self.conv = ConvTranspose1d(
            length,
            in_channels,
            out_channels,
            dtype,
            stride,
            padding,
            dilation,
            output_padding,
            device,
            has_bias,
            permute,
            name,
        )

        # Initialize g parameter (magnitude) - shape matches PyTorch's implementation
        self.weight_g = Weight(
            name=f"{name}.weight_g" if name else "weight_g",
            dtype=dtype,
            shape=(in_channels, 1, 1),  # Shape matches PyTorch's weight_g
            device=device or DeviceRef.CPU(),
        )

        # Initialize v parameter (direction)
        if permute:
            v_shape = (in_channels, out_channels, length)
        else:
            v_shape = (length, out_channels, in_channels)

        self.weight_v = Weight(
            name=f"{name}.weight_v" if name else "weight_v",
            dtype=dtype,
            shape=v_shape,
            device=device or DeviceRef.CPU(),
        )

        self.bias = self.conv.bias

        del self.conv.weight
        # ``bias`` is only set as an instance attribute when ``has_bias=True``;
        # otherwise it resolves to the dataclass class default and cannot be
        # deleted through the instance.
        if "bias" in vars(self.conv):
            del self.conv.bias

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply the weight normalized convolution to input ``x``.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying convolution with normalized weights.
        """
        if not hasattr(self.conv, "weight"):
            # Compute normalized weight sqrt(sum(x**2))
            in_channels = self.weight_v.shape[0]
            v_norm = ops.sqrt(
                ops.sum((self.weight_v**2).reshape([in_channels, -1]), axis=1)
            ).reshape([in_channels, 1, 1])
            w = self.weight_v / v_norm
            w = w * self.weight_g

            # Update conv layer weight and apply convolution
            self.conv.weight = w
            self.conv.bias = self.bias

        return self.conv(x)
