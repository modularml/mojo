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

"""Row- and column-parallel linear layers with pre-defined sharding intent."""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from max.experimental import functional as F
from max.experimental import random
from max.experimental.nn.linear import Linear
from max.experimental.nn.transparent_module import TransparentModule
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import DimLike

from .mesh_axis import TP


class _LinearProtocol(Protocol):
    weight: Tensor
    bias: Tensor | Literal[0]


_LinearLayer = TypeVar("_LinearLayer", bound=_LinearProtocol)


def col_parallel(
    layer: _LinearLayer, tp_axis: str | None = None
) -> _LinearLayer:
    """Parallelize the linear layer across the column dimension."""
    if tp_axis is None:
        tp_axis = TP
    # Note that the first dimension of the weight is sharded because MAX's
    # linear layer applies ``x @ W.T`` (the transpose of W)
    layer.weight._mapping = NamedMapping(layer.weight.mesh, (tp_axis, None))
    if isinstance(layer.bias, Tensor):
        layer.bias._mapping = NamedMapping(layer.bias.mesh, (tp_axis,))
    return layer


def row_parallel(
    layer: _LinearLayer, tp_axis: str | None = None
) -> _LinearLayer:
    """Parallelize the linear layer across the row dimension."""
    if tp_axis is None:
        tp_axis = TP
    layer.weight._mapping = NamedMapping(layer.weight.mesh, (None, tp_axis))
    if isinstance(layer.bias, Tensor):
        layer.bias._mapping = NamedMapping(layer.bias.mesh, (None, tp_axis))
    return layer


class QKVLinear(TransparentModule[[Tensor], Tensor]):
    """A fused q/k/v projection with two checkpoint layouts.

    With ``stacked=False`` (the default) it holds separate
    ``q_proj``/``k_proj``/``v_proj`` :class:`Linear` children and is
    name-transparent, so weights load under native ``<parent>.q_proj.weight``
    names. With ``stacked=True`` it holds one pre-fused
    ``[q_dim + 2*kv_dim, in_dim]`` weight and stays opaque, keeping it under
    ``<parent>.qkv_proj.weight``. Either way ``forward`` is a single matmul over
    the fused q||k||v weight.
    """

    def __init__(
        self,
        in_dim: DimLike,
        q_dim: int,
        kv_dim: int,
        *,
        bias: bool = False,
        stacked: bool = False,
    ) -> None:
        """Constructs a fused q/k/v linear projection.

        Args:
            in_dim: The input (hidden) dimension.
            q_dim: The Q projection output dimension.
            kv_dim: The K and V projection output dimension (each).
            bias: Whether to use a bias in the transformation.
            stacked: Store one pre-fused weight (opaque) instead of separate,
                name-transparent ``q_proj``/``k_proj``/``v_proj`` children.
        """
        self.q_dim = int(q_dim)
        self.kv_dim = int(kv_dim)
        self._has_bias = bias
        self._stacked = stacked
        self.name_transparent = not stacked
        if stacked:
            out_dim = q_dim + 2 * kv_dim
            self.weight = random.normal([out_dim, in_dim])
            self.bias = random.normal([out_dim]) if bias else 0
        else:
            self.q_proj = Linear(in_dim, q_dim, bias=bias)
            self.k_proj = Linear(in_dim, kv_dim, bias=bias)
            self.v_proj = Linear(in_dim, kv_dim, bias=bias)

    @property
    def fused_weight(self) -> Tensor:
        """The fused q||k||v weight ``[q_dim + 2*kv_dim, in_dim]``."""
        if self._stacked:
            return self.weight
        return F.concat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            axis=0,
        )

    @property
    def fused_bias(self) -> Tensor | Literal[0]:
        """The fused q||k||v bias, or ``0`` when bias is disabled."""
        if not self._has_bias:
            return 0
        if self._stacked:
            # _has_bias is set, so the pre-fused bias is a Tensor, not 0.
            assert isinstance(self.bias, Tensor)
            return self.bias
        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def __rich_repr__(self):
        """Repr matching the QKVLinear constructor."""
        yield "q_dim", self.q_dim
        yield "kv_dim", self.kv_dim
        yield "stacked", self._stacked, False
        yield "bias", self._has_bias, False

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        """Applies the fused q/k/v projection: ``x @ fused_weight.T + bias``."""
        return x @ self.fused_weight.T + self.fused_bias


class ColumnParallelLinear(Linear):
    """Linear layer with column-parallel weight sharding."""

    def __init__(self, *args, tp_axis: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        col_parallel(self, tp_axis)


class RowParallelLinear(Linear):
    """Linear layer with row-parallel weight sharding."""

    def __init__(self, *args, tp_axis: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        row_parallel(self, tp_axis)
