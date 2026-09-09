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

"""Distributed type descriptors for graph compilation.

* :class:`DistributedTensorType`: symbolic type for a tensor distributed
  across a device mesh. Analogous to :class:`~max.graph.TensorType`.
* :class:`DistributedBufferType`: symbolic type for a mutable buffer
  distributed across a device mesh. Analogous to
  :class:`~max.graph.BufferType`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Shape,
    TensorType,
)
from max.graph.shape import ShapeLike

from .mappings import DeviceMapping
from .mesh import DeviceMesh
from .placements import Placement, Sharded, local_shard_shape_from_global

T = TypeVar("T", TensorType, BufferType)


class DistributedType(Generic[T], ABC):
    """Shared state and shard-shape logic for distributed type descriptors.

    Not intended for direct use. See :class:`DistributedTensorType` and
    :class:`DistributedBufferType`.
    """

    dtype: DType
    shape: Shape
    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    def __init__(
        self,
        dtype: DType,
        shape: ShapeLike,
        mesh: DeviceMesh,
        placements: Sequence[Placement],
    ) -> None:
        self.dtype = dtype
        self.shape = Shape(shape)
        self.mesh = mesh
        self.placements = tuple(placements)

        if len(self.placements) != mesh.ndim:
            raise ValueError(
                f"Need one placement per mesh axis. Mesh has {mesh.ndim} "
                f"axes, got {len(self.placements)} placements."
            )
        for placement in self.placements:
            if isinstance(placement, Sharded):
                axis = placement.axis
                if axis < 0 or axis >= len(self.shape):
                    raise ValueError(
                        f"Sharded(axis={axis}) is out of range for tensor "
                        f"with rank {len(self.shape)}."
                    )

    def _local_shard_shape(self, device_idx: int = 0) -> Shape:
        """Computes one device's local shard shape from the global shape.

        Delegates to :func:`local_shard_shape_from_global` (which dispatches
        per placement type) and returns the requested device's shape.
        """
        shapes = local_shard_shape_from_global(
            self.shape, self.mesh, self.placements
        )
        return shapes[device_idx]

    @property
    def rank(self) -> int:
        """The rank (number of dimensions) of the global tensor."""
        return len(self.shape)

    @property
    @abstractmethod
    def local_types(self) -> list[T]:
        """The per-device types in mesh order."""
        ...


class DistributedTensorType(DistributedType[TensorType]):
    """A symbolic type for a tensor distributed across a device mesh.

    Analogous to :class:`~max.graph.TensorType` for single-device tensors.
    Derives per-device :class:`~max.graph.TensorType` objects via
    :attr:`local_types`.

    When a :class:`~max.graph.SymbolicDim` is sharded along a mesh axis, the
    local shard dimension becomes a new ``SymbolicDim`` named
    ``"{original}_{axis_name}"``. This keeps symbolic names short and
    debuggable while ensuring that sharding the same global dim on different
    axes produces distinct names.

    Args:
        dtype: Element data type.
        shape: Global (logical) shape.
        mesh: The device mesh.
        placements: One :class:`Placement` per mesh axis.
    """

    @property
    def local_types(self) -> list[TensorType]:
        """The per-device :class:`~max.graph.TensorType` objects in mesh order."""
        return [
            TensorType(
                self.dtype,
                self._local_shard_shape(i),
                DeviceRef.from_device(device),
            )
            for i, device in enumerate(self.mesh.devices)
        ]

    def __repr__(self) -> str:
        placement_str = ", ".join(repr(p) for p in self.placements)
        shape_str = ", ".join(str(d) for d in self.shape)
        return (
            f"DistributedTensorType(dtype={self.dtype}, shape=[{shape_str}], "
            f"mesh={self.mesh}, placements=[{placement_str}])"
        )


class DistributedBufferType(DistributedType[BufferType]):
    """A symbolic type for a mutable buffer distributed across a device mesh.

    Analogous to :class:`~max.graph.BufferType` for single-device buffers.
    Derives per-device :class:`~max.graph.BufferType` objects via
    :attr:`local_types`.

    Args:
        dtype: Element data type.
        shape: Global (logical) shape.
        mesh: The device mesh.
        placements: One :class:`Placement` per mesh axis.
    """

    @property
    def local_types(self) -> list[BufferType]:
        """The per-device :class:`~max.graph.BufferType` objects in mesh order."""
        return [
            BufferType(
                self.dtype,
                self._local_shard_shape(i),
                DeviceRef.from_device(device),
            )
            for i, device in enumerate(self.mesh.devices)
        ]

    def __repr__(self) -> str:
        placement_str = ", ".join(repr(p) for p in self.placements)
        shape_str = ", ".join(str(d) for d in self.shape)
        return (
            f"DistributedBufferType(dtype={self.dtype}, shape=[{shape_str}], "
            f"mesh={self.mesh}, placements=[{placement_str}])"
        )


# ═════════════════════════════════════════════════════════════════════════
#  TensorLayout — metadata snapshot for rule evaluation
# ═════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TensorLayout(DeviceMapping):
    """Metadata snapshot of a distributed tensor for rule evaluation.

    Bundles the tensor's dtype, shape, and distribution mapping. The mapping
    stays abstract (:class:`DeviceMapping`) so rules work with any concrete
    mapping type, such as :class:`PlacementMapping` or :class:`NamedMapping`.

    The shape is a :class:`~max.graph.Shape` (``list[Dim]``), supporting both
    static and symbolic dimensions for graph compilation compatibility.

    This class implements DeviceMapping, so sharding rules can
    return input TensorLayouts directly.
    """

    dtype: DType
    """The element data type of the tensor."""

    shape: Shape
    """The global shape of the tensor."""

    mapping: DeviceMapping
    """The distribution mapping over the device mesh."""

    def __init__(
        self, dtype: DType, shape: ShapeLike, mapping: DeviceMapping
    ) -> None:
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", Shape(shape))
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "mesh", mapping.mesh)
        object.__setattr__(self, "placements", mapping.placements)

    @property
    def rank(self) -> int:
        """The number of dimensions."""
        return len(self.shape)

    def __repr__(self) -> str:
        shape_str = ", ".join(str(d) for d in self.shape)
        return (
            f"TensorLayout(dtype={self.dtype}, shape=[{shape_str}], "
            f"mapping={self.mapping})"
        )
