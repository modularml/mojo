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

"""Provides group normalization for experimental tensors."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn.module import Module
from max.experimental.tensor import Tensor
from max.graph import ops

#: Functional form of group normalization for experimental tensors.
#:
#: See :func:`max.graph.ops.group_norm` for the underlying op and the full
#: parameter description.
group_norm = F.functional(ops.group_norm)


class GroupNorm(Module[[Tensor], Tensor]):
    """Group normalization over the channel axis of the input.

    The input is expected to have shape ``(batch, channels, ...)`` where
    ``...`` is any number of trailing axes (typically spatial dimensions
    for convolutional features). The channel axis is split into
    ``num_groups`` groups and each group is normalized independently.
    Useful when the batch axis is small enough that batch normalization
    is unstable.

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental.nn.norm import GroupNorm
        from max.experimental.tensor import Tensor, default_device

        with default_device(CPU()):
            norm = GroupNorm(num_groups=32, num_channels=128)
            x = Tensor.ones([2, 128, 8, 8], dtype=DType.float32)
            y = norm(x)

    .. invisible-code-block: python

        import numpy as np

        assert y.shape == [2, 128, 8, 8]
        # A constant input has zero variance, so it normalizes to zeros.
        assert np.allclose(y.to_numpy(), 0.0, atol=1e-3)


    Args:
        num_groups: The number of groups to split the channel axis into.
            Must divide ``num_channels`` evenly.
        num_channels: The size of the channel axis of the input (axis 1).
        eps: A small positive constant added to the variance for numerical
            stability. Defaults to ``1e-5``.
        affine: Whether to learn a per-channel scale and bias. When
            ``False``, no parameters are created. Defaults to ``True``.

    Raises:
        ValueError: If ``num_channels`` is not divisible by ``num_groups``.
    """

    weight: Tensor | None
    """The learned per-channel scale of shape ``[num_channels]``, or
    ``None`` when ``affine`` is ``False``."""

    bias: Tensor | None
    """The learned per-channel bias of shape ``[num_channels]``, or
    ``None`` when ``affine`` is ``False``."""

    num_groups: int
    """The number of groups the channel axis is split into."""

    num_channels: int
    """The size of the channel axis of the input."""

    eps: float
    """The variance epsilon used for numerical stability."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels({num_channels}) should be divisible by "
                f"num_groups({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = Tensor.ones([num_channels])
            self.bias = Tensor.zeros([num_channels])
        else:
            self.weight = None
            self.bias = None

    def __rich_repr__(self):
        """Yields fields for the rich debug repr."""
        yield "num_groups", self.num_groups
        yield "num_channels", self.num_channels
        yield "eps", self.eps, 1e-5
        yield "affine", self.affine, True

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x`` normalized within each channel group.

        Args:
            x: The input tensor of shape ``(batch, channels, ...)``. The
                size of the channel axis must equal :attr:`num_channels`.

        Returns:
            A tensor with the same shape and dtype as ``x``.

        Raises:
            ValueError: If ``x`` has fewer than 2 dimensions, or if its
                channel axis does not match :attr:`num_channels`.
        """
        if len(x.shape) < 2:
            raise ValueError(
                f"Expected input tensor with >=2 dimensions, got shape {x.shape}"
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got shape {x.shape}"
            )

        if self.affine:
            if self.weight is None or self.bias is None:
                raise ValueError("weight and bias must be set when affine=True")
            weight = self.weight
            bias = self.bias
        else:
            # Create temporary tensors of ones and zeros when affine=False
            weight = Tensor.ones(
                [self.num_channels], dtype=x.dtype, device=x.device
            )
            bias = Tensor.zeros(
                [self.num_channels], dtype=x.dtype, device=x.device
            )

        return group_norm(x, weight, bias, self.num_groups, self.eps)
