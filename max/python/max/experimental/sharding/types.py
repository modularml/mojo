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

"""Describes how a value is laid out, and whether a callable may write it.

Two layouts describe every tensor argument in :mod:`max.experimental`:

* :class:`TensorLayout`: a dtype, a global shape, and a device mapping. The
  default, and what a value reports about itself.
* :class:`BufferLayout`: the same, for a buffer a compiled callable may
  also store through.

They differ only in the :mod:`max.graph` type each lowers to per device, so
which one an argument is given is the one place mutability is spelled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Shape,
    TensorType,
)
from max.graph.shape import ShapeLike
from max.graph.type import _TensorTypeBase

from .mappings import DeviceMapping, as_device_mapping
from .mesh import DeviceMesh
from .placements import Placement, local_shard_shape_from_global


@dataclass(frozen=True)
class TensorLayout:
    """A tensor's dtype and global shape, distributed across a device mesh.

    A compiled callable reads an argument given this layout but may not store
    through it. See :class:`BufferLayout` for one it may write.

    A :class:`~max.graph.SymbolicDim` sharded along a mesh axis becomes one
    fresh local dim per shard, named ``"{original}_{axis_name}_{shard}"``, so
    sharding the same global dim on different axes stays distinguishable.

    Args:
        dtype: The element data type.
        shape: The global shape, not one shard's.
        device: The placement: a single device, a mesh to replicate over,
            or a :class:`DeviceMapping`.

    Raises:
        ValueError: If a placement shards an axis this shape does not have.
    """

    dtype: DType
    """The element data type."""

    shape: Shape
    """The global shape."""

    mapping: DeviceMapping
    """The distribution across the device mesh."""

    def __init__(
        self,
        dtype: DType,
        shape: ShapeLike,
        device: Device | DeviceRef | DeviceMesh | DeviceMapping,
    ) -> None:
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", Shape(shape))
        object.__setattr__(self, "mapping", as_device_mapping(device))
        self.mapping.check_shape(self.shape)

    @property
    def mesh(self) -> DeviceMesh:
        """The mesh this value is distributed over."""
        return self.mapping.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """One placement per mesh axis."""
        return self.mapping.placements

    @property
    def rank(self) -> int:
        """The number of dimensions."""
        return len(self.shape)

    @property
    def device(self) -> Device:
        """The single device this value sits on.

        Raises:
            ValueError: If it spans more than one device.
        """
        if self.mesh.num_devices > 1:
            raise ValueError(
                f"{type(self).__name__} spans {self.mesh.num_devices} "
                "devices and so has no single device. Use .mesh.devices."
            )
        return self.mesh.devices[0]

    @property
    def local_types(self) -> Sequence[_TensorTypeBase[Any]]:
        """One :class:`~max.graph.TensorType` per device, in mesh order."""
        return [
            TensorType(self.dtype, shape, DeviceRef.from_device(device))
            for shape, device in zip(
                self._local_shapes(), self.mesh.devices, strict=True
            )
        ]

    def as_buffer(self) -> BufferLayout:
        """Returns this layout as a buffer a callable may store through."""
        return BufferLayout(self.dtype, self.shape, self.mapping)

    def as_tensor(self) -> TensorLayout:
        """Returns this layout as a value a callable only reads."""
        return TensorLayout(self.dtype, self.shape, self.mapping)

    def _local_shapes(self) -> list[Shape]:
        """Returns one shard shape per device, in mesh order."""
        return local_shard_shape_from_global(
            self.shape, self.mesh, self.placements
        )

    def __repr__(self) -> str:
        shape = ", ".join(str(dim) for dim in self.shape)
        return f"{type(self).__name__}({self.dtype}, [{shape}], {self.mapping})"


class BufferLayout(TensorLayout):
    """A buffer a compiled callable may store through.

    An argument given this layout lowers to a :class:`~max.graph.BufferValue`
    at every boundary it crosses, so a write to it reaches the caller. It is a
    :class:`TensorLayout` in every other respect, which is why a sharding rule
    never has to know the difference; only the lowering does.
    """

    @property
    def local_types(self) -> Sequence[BufferType]:
        """One :class:`~max.graph.BufferType` per device, in mesh order."""
        return [
            BufferType(self.dtype, shape, DeviceRef.from_device(device))
            for shape, device in zip(
                self._local_shapes(), self.mesh.devices, strict=True
            )
        ]


def as_layout(layout: object) -> TensorLayout:
    """Normalizes a declared layout.

    Takes a declaration, never a value: a live tensor's dims are whatever it
    currently holds, so reading a layout off one would fix every dimension to
    that. Anything holding a ``layout`` of its own is refused, and told to
    pass it.

    A single-device :mod:`max.graph` type is accepted and coerced, so the
    graph types stay usable without appearing in any signature. A
    :class:`~max.graph.BufferType` coerces to a :class:`BufferLayout`,
    carrying its declaration across.

    Args:
        layout: A :class:`TensorLayout` or :class:`BufferLayout` to take as
            given, or a single-device :class:`~max.graph.TensorType` or
            :class:`~max.graph.BufferType`.

    Returns:
        The equivalent layout.

    Raises:
        TypeError: If ``layout`` is a value rather than a declaration, or
            describes no layout at all.
    """
    if isinstance(layout, TensorLayout):
        return layout
    # Order matters: a BufferType is what declares the write.
    if isinstance(layout, BufferType):
        return BufferLayout(layout.dtype, layout.shape, layout.device)
    if isinstance(layout, TensorType):
        return TensorLayout(layout.dtype, layout.shape, layout.device)
    # Duck-typed rather than an import: `tensor` imports this module.
    if isinstance(getattr(layout, "layout", None), TensorLayout):
        raise TypeError(
            f"a {type(layout).__name__} is a value, not a layout: its dims "
            "are whatever it currently holds, so every one of them would be "
            "fixed to that size. Pass its .layout to do that on purpose."
        )
    raise TypeError(
        f"expected a TensorLayout, BufferLayout, TensorType or BufferType "
        f"layout, got {type(layout).__name__}: {layout!r}"
    )
