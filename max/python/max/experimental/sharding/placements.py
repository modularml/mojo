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

"""Placement types describing how tensor data is distributed across one mesh axis."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from max.graph.dim import AlgebraicDim, Dim, DimLike, StaticDim, SymbolicDim
from max.graph.shape import Shape

from .mesh import DeviceMesh
from .per_shard_dim import (
    is_per_shard_dim,
    local_shape_at,
    make_per_shard_dim,
)


class Collective(Enum):
    """The collectives the cost model understands.

    A :meth:`Placement.transition_to` return value must be one of these.
    Custom :class:`Placement` subclasses that return anything else are
    reported as infeasible by the cost model.
    """

    NOOP = "noop"
    LOCAL_SLICE = "local_slice"
    ALLGATHER = "allgather"
    ALL_TO_ALL = "all_to_all"
    ALLREDUCE = "allreduce"
    REDUCE_SCATTER = "reduce_scatter"
    INFEASIBLE = "infeasible"


class ShardingError(RuntimeError):
    """Raised when a sharding constraint cannot be satisfied."""


def _shard_sizes_along_axis(global_size: int, num_shards: int) -> list[int]:
    """Splits ``global_size`` across ``num_shards``; sizes differ by at most 1.

    The ``-1`` wildcard is preserved on every rank.
    """
    if global_size == -1:
        return [-1] * num_shards
    base, remainder = divmod(global_size, num_shards)
    return [base + 1 if i < remainder else base for i in range(num_shards)]


class Placement(ABC):
    """Abstract base for all placement types.

    Each placement describes what a single mesh axis does to a tensor:
    ``Replicated`` (full copy), ``Sharded`` (split along a tensor dim),
    or ``Partial`` (partial result needing reduction).
    """

    @abstractmethod
    def __repr__(self) -> str: ...

    def localized_axis(self) -> int | None:
        """Tensor axis this placement localizes, or ``None`` if it doesn't."""
        return None

    def transition_to(self, other: Placement) -> Collective:
        """Returns the :class:`Collective` for ``self -> other``."""
        return Collective.NOOP if self == other else Collective.INFEASIBLE

    def local_dim(
        self,
        parent: DimLike,
        mesh: DeviceMesh,
        mesh_axis: int,
        *,
        allow_symbolic_mint: bool = True,
    ) -> Dim:
        """Returns the per-shard local cells of ``parent`` along this mesh axis.

        The default returns ``parent`` unchanged; ``Sharded`` overrides to
        split. A wrapper ``parent`` is passed through verbatim.

        Args:
            parent: The global dim to localize.
            mesh: The device mesh.
            mesh_axis: The mesh axis index being localized.
            allow_symbolic_mint: When ``False``, refuse to mint fresh
                per-shard symbols from a bare :class:`SymbolicDim`. Reshape
                propagation passes ``False``.
        """
        return Dim(parent)

    def global_dim(self, cells: Dim) -> Dim:
        """Combines per-shard cells along this mesh axis into one global :class:`~max.graph.Dim`.

        The default returns ``cells`` unchanged (every shard holds the
        same dim). ``Sharded`` overrides to sum the cells.
        """
        if is_per_shard_dim(cells):
            first = cells.per_shard[0]
            for d in cells.per_shard[1:]:
                if d != first:
                    raise ValueError(
                        f"{type(self).__name__}.global_dim: per-shard cells "
                        f"{cells.per_shard!r} disagree along this mesh axis; "
                        "this placement requires shape-identical shards. "
                        "Override global_dim() to allow heterogeneous cells."
                    )
            return first
        return cells


@dataclass(frozen=True)
class Replicated(Placement):
    """Every device on this mesh axis holds the same copy of the data."""

    def __repr__(self) -> str:
        return "Replicated()"

    def transition_to(self, other: Placement) -> Collective:
        """Replicated-to-Sharded is a free local split."""
        if self == other:
            return Collective.NOOP
        if isinstance(other, Sharded):
            return Collective.LOCAL_SLICE
        return super().transition_to(other)


@dataclass(frozen=True)
class Sharded(Placement):
    """Every device on this mesh axis holds a slice along ``axis``.

    Args:
        axis: The tensor axis along which data is split.
    """

    axis: int

    def __repr__(self) -> str:
        return f"Sharded(axis={self.axis})"

    def localized_axis(self) -> int | None:
        """Returns the tensor axis this Sharded localizes."""
        return self.axis

    def transition_to(self, other: Placement) -> Collective:
        """Sharded-to-Replicated is allgather; Sharded-to-Sharded is all-to-all."""
        if self == other:
            return Collective.NOOP
        if isinstance(other, Replicated):
            return Collective.ALLGATHER
        if isinstance(other, Sharded):
            return Collective.ALL_TO_ALL
        return super().transition_to(other)

    def local_dim(
        self,
        parent: DimLike,
        mesh: DeviceMesh,
        mesh_axis: int,
        *,
        allow_symbolic_mint: bool = True,
    ) -> Dim:
        """Splits ``parent`` along ``mesh_axis`` into per-shard cells.

        ``StaticDim`` parents use uneven divmod; ``SymbolicDim`` parents
        mint fresh per-shard cells; ``AlgebraicDim`` parents raise.
        Wrapper parents pass through verbatim. When
        ``allow_symbolic_mint`` is ``False``, a bare ``SymbolicDim``
        raises instead of minting.
        """
        if is_per_shard_dim(parent):
            assert isinstance(parent, Dim)
            return parent
        parent_dim = Dim(parent)
        mesh_axis_size = mesh.mesh_shape[mesh_axis]
        axis_name = mesh.axis_names[mesh_axis]
        stride = math.prod(mesh.mesh_shape[mesh_axis + 1 :])
        if isinstance(parent_dim, StaticDim):
            chunks = _shard_sizes_along_axis(int(parent_dim), mesh_axis_size)
            cells: tuple[Dim, ...] = tuple(
                StaticDim(chunks[(device_idx // stride) % mesh_axis_size])
                for device_idx in range(mesh.num_devices)
            )
        elif isinstance(parent_dim, SymbolicDim):
            if not allow_symbolic_mint:
                raise ShardingError(
                    f"Sharded.local_dim cannot mint per-shard cells from "
                    f"bare symbolic {parent_dim!r}. Thread a per-shard "
                    "wrapper (``tensor.shape[k]`` from a sharded input) "
                    "into the target shape, or pin the axis Replicated "
                    "upstream."
                )
            cells = tuple(
                SymbolicDim(
                    f"{parent_dim.name}_{axis_name}_"
                    f"{(device_idx // stride) % mesh_axis_size}"
                )
                for device_idx in range(mesh.num_devices)
            )
        elif isinstance(parent_dim, AlgebraicDim):
            raise ShardingError(
                f"Sharded.local_dim cannot split algebraic {parent_dim!r}: "
                "shard the underlying operand instead, or thread a "
                "per-shard wrapper (``tensor.shape[k]`` from a sharded "
                "input) into the target shape."
            )
        else:
            cells = (parent_dim // mesh_axis_size,) * mesh.num_devices
        return make_per_shard_dim(cells)

    def global_dim(self, cells: Dim) -> Dim:
        """Sums per-shard cells along this mesh axis."""
        if not is_per_shard_dim(cells):
            return cells
        total: Dim = cells.per_shard[0]
        for d in cells.per_shard[1:]:
            total = total + d
        return total


class ReduceOp(str, Enum):
    """Reduction operations for partial placements.

    Only ``SUM`` is currently implemented in eager dispatch; the others
    are reserved for forward compatibility.
    """

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


@dataclass(frozen=True)
class Partial(Placement):
    """Every device holds a partial result that must be reduced.

    Args:
        reduce_op: The reduction operation to apply. Defaults to
            :attr:`ReduceOp.SUM`.
    """

    reduce_op: ReduceOp = ReduceOp.SUM

    def __repr__(self) -> str:
        return f"Partial(reduce_op={self.reduce_op.value!r})"

    def transition_to(self, other: Placement) -> Collective:
        """Partial-to-Replicated is allreduce; Partial-to-Sharded is reduce-scatter."""
        if self.reduce_op in (ReduceOp.MIN, ReduceOp.MAX):
            return Collective.INFEASIBLE
        if self == other:
            return Collective.NOOP
        if isinstance(other, Replicated):
            return Collective.ALLREDUCE
        if isinstance(other, Sharded):
            return Collective.REDUCE_SCATTER
        return super().transition_to(other)


def shard_shape(
    global_shape: Sequence[DimLike],
    placements: Sequence[Placement],
    mesh: DeviceMesh,
) -> list[Dim]:
    """One representative shard shape via ``parent // mesh_axis_size`` per localized axis.

    For per-shard-precise shapes use :func:`local_shard_shape_from_global`.
    """
    result: list[Dim] = [Dim(d) for d in global_shape]
    for mesh_axis, p in enumerate(placements):
        axis = p.localized_axis()
        if axis is not None:
            result[axis] = result[axis] // mesh.mesh_shape[mesh_axis]
    return result


def local_shard_shape_from_global(
    global_shape: Sequence[DimLike],
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> list[Shape]:
    """One :class:`Shape` per device, in row-major mesh order.

    Composes every placement's :meth:`Placement.local_dim` for the tensor
    axis it localizes, then projects per rank.

    Raises:
        ValueError: If ``placements`` length differs from ``mesh.ndim`` or
            any placement localizes an axis out of range for ``global_shape``.
    """
    global_list: list[Dim] = [Dim(d) for d in global_shape]
    rank = len(global_list)
    if len(placements) != mesh.ndim:
        raise ValueError(
            f"Need one placement per mesh axis. Mesh has {mesh.ndim} axes, "
            f"got {len(placements)} placements."
        )
    norm_axes: list[int | None] = []
    for p in placements:
        ax = p.localized_axis()
        if ax is None:
            norm_axes.append(None)
            continue
        if ax < 0:
            ax += rank
        if ax < 0 or ax >= rank:
            raise ValueError(
                f"{p!r} localizes axis {p.localized_axis()}, out of range "
                f"for tensor with rank {rank}."
            )
        norm_axes.append(ax)
    wrapped: list[Dim] = []
    for ti, d in enumerate(global_list):
        x: Dim = d
        for mesh_axis, p in enumerate(placements):
            if norm_axes[mesh_axis] == ti:
                x = p.local_dim(x, mesh, mesh_axis)
        wrapped.append(x)
    return [local_shape_at(wrapped, r) for r in range(mesh.num_devices)]
