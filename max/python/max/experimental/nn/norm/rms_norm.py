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

"""Provides root mean square normalization layers."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.sharding.rules import rms_norm_rule
from max.experimental.tensor import Tensor
from max.graph import Dim, ops

from ..module import Module

#: Functional form of RMS normalization for experimental tensors.
#:
#: See :func:`max.graph.ops.rms_norm` for the underlying op, including the
#: ``weight_offset`` and ``multiply_before_cast`` knobs used to switch
#: between Llama-style and Gemma-style normalization.
rms_norm = F.functional(ops.rms_norm, rule=rms_norm_rule)


class RMSNorm(Module[[Tensor], Tensor]):
    """Root mean square normalization over the last dimension of the input.

    Unlike :class:`LayerNorm`, the mean is not subtracted; only the
    root-mean-square is used to rescale. See `Root Mean Square Layer
    Normalization <https://arxiv.org/abs/1910.07467>`_ for the formulation.
    For the Gemma variant that uses ``1 + weight`` and multiplies before
    casting back, see :class:`GemmaRMSNorm`.

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental.nn.norm import RMSNorm
        from max.experimental.tensor import Tensor, default_device

        with default_device(CPU()):
            norm = RMSNorm(2048, eps=1e-6)
            x = Tensor.ones([2, 4, 2048], dtype=DType.float32)
            y = norm(x)

    .. invisible-code-block: python

        import numpy as np

        assert y.shape == [2, 4, 2048]
        # An all-ones input has RMS 1, so it normalizes back to ones.
        assert np.allclose(y.to_numpy(), 1.0, atol=1e-3)


    Args:
        dim: The size of the last dimension of the input.
        eps: A small positive constant added to the mean of squares for
            numerical stability. Defaults to ``1e-6``.
    """

    weight: Tensor
    """The learned per-element scale of shape ``[dim]``."""

    eps: float
    """The variance epsilon used for numerical stability."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.weight = Tensor.ones([dim])
        self.eps = eps

    @property
    def dim(self) -> Dim:
        """The size of the last dimension over which normalization runs."""
        return self.weight.shape[0]

    def __rich_repr__(self):
        """Yields fields for the rich debug repr."""
        yield "dim", self.dim
        yield "eps", self.eps, 1e-6

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x`` normalized by its root-mean-square over the last axis."""
        return rms_norm(x, self.weight, self.eps)


class GemmaRMSNorm(RMSNorm):
    """Gemma-style root mean square normalization.

    Subclasses :class:`RMSNorm` with two differences:

    - Scales by ``1 + weight`` rather than ``weight``.
    - Multiplies by the scale before casting back to the input dtype,
      instead of after.

    The constructor signature is identical to :class:`RMSNorm`. Used by
    the Gemma model family.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x`` normalized using the Gemma-style RMS variant."""
        return rms_norm(
            x,
            self.weight,
            self.eps,
            weight_offset=1.0,
            multiply_before_cast=True,
        )
