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

"""Device mesh for organizing devices into an N-dimensional logical grid."""

from __future__ import annotations

import contextvars
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from max.driver import CPU, Device


@dataclass(frozen=True)
class DeviceMesh:
    """An N-dimensional logical grid of devices."""

    devices: tuple[Device, ...]
    """The devices in the mesh, in row-major order."""

    mesh_shape: tuple[int, ...]
    """The shape of the logical grid."""

    axis_names: tuple[str, ...]
    """The human-readable name for each mesh axis."""

    def __post_init__(self) -> None:
        if len(self.devices) == 0:
            raise ValueError("DeviceMesh requires at least one device.")
        expected = math.prod(self.mesh_shape)
        if len(self.devices) != expected:
            raise ValueError(
                f"mesh_shape {self.mesh_shape} expects {expected} devices, "
                f"got {len(self.devices)}"
            )
        if len(self.axis_names) != len(self.mesh_shape):
            raise ValueError(
                f"axis_names length ({len(self.axis_names)}) must match "
                f"mesh_shape rank ({len(self.mesh_shape)})"
            )
        if len(set(self.axis_names)) != len(self.axis_names):
            raise ValueError(
                f"axis_names must be unique, got {self.axis_names}"
            )
        if len(self.devices) > 1:
            unique = set(self.devices)
            n_unique = len(unique)
            if n_unique != 1 and n_unique != len(self.devices):
                raise ValueError(
                    f"DeviceMesh requires either all-same devices "
                    f"(simulated) or all-distinct devices (multi-device), "
                    f"but got {n_unique} unique devices for "
                    f"{len(self.devices)} slots."
                )

    @property
    def ndim(self) -> int:
        """The number of mesh dimensions."""
        return len(self.mesh_shape)

    @property
    def num_devices(self) -> int:
        """The total number of devices."""
        return len(self.devices)

    def axis_size(self, axis: str | int) -> int:
        """Returns the size of a mesh axis by name or index.

        Raises:
            ValueError: If ``axis`` is a name not on the mesh.
            IndexError: If ``axis`` is an integer outside ``[0, ndim)``.
        """
        idx = self._resolve_axis(axis)
        return self.mesh_shape[idx]

    def device_coord(self, device_idx: int, axis: str | int) -> int:
        """Returns *device_idx*'s coordinate along the given mesh axis.

        For a mesh shaped ``(2, 3)`` with row-major device ordering, the
        device at flat index ``4`` has coords ``(1, 1)`` — so
        ``mesh.device_coord(4, 0) == 1`` and
        ``mesh.device_coord(4, 1) == 1``.

        Args:
            device_idx: The flat device index in row-major order.
            axis: The mesh axis to query, by name or integer index.

        Returns:
            The device's coordinate along *axis*, in ``[0, axis_size)``.

        Raises:
            IndexError: If ``device_idx`` is out of range, or if ``axis``
                is an integer outside ``[0, ndim)``.
            ValueError: If ``axis`` is a name not present on the mesh.
        """
        if device_idx < 0 or device_idx >= self.num_devices:
            raise IndexError(
                f"device_idx {device_idx} out of range for mesh with "
                f"{self.num_devices} devices"
            )
        idx = self._resolve_axis(axis)
        stride = math.prod(self.mesh_shape[idx + 1 :])
        return (device_idx // stride) % self.mesh_shape[idx]

    def _resolve_axis(self, axis: str | int) -> int:
        """Converts an axis name or index to a validated integer index."""
        if isinstance(axis, str):
            if axis not in self.axis_names:
                raise ValueError(
                    f"Unknown axis name {axis!r}. Available: {self.axis_names}"
                )
            return self.axis_names.index(axis)
        if axis < 0 or axis >= self.ndim:
            raise IndexError(
                f"Axis index {axis} out of range for mesh with "
                f"{self.ndim} dimensions"
            )
        return axis

    def __repr__(self) -> str:
        shape_str = ", ".join(
            f"{name}={size}"
            for name, size in zip(
                self.axis_names, self.mesh_shape, strict=False
            )
        )
        return f"DeviceMesh({shape_str})"

    @staticmethod
    def single(device: Device) -> DeviceMesh:
        """Creates a trivial single-device mesh."""
        return DeviceMesh(devices=(device,), mesh_shape=(1,), axis_names=("_",))

    @staticmethod
    def default() -> DeviceMesh:
        """Returns a single-device mesh on the default device (CPU)."""
        return DeviceMesh.single(CPU())

    @property
    def is_single(self) -> bool:
        """Returns ``True`` if this mesh contains exactly one device."""
        return self.num_devices == 1

    @property
    def is_simulated(self) -> bool:
        """Returns ``True`` if all mesh slots reference the same device.

        A simulated mesh uses graph-level ops to emulate multi-device
        collectives on a single CPU or GPU.
        """
        return self.num_devices > 1 and len(set(self.devices)) == 1


_active_mesh: contextvars.ContextVar[DeviceMesh | None] = (
    contextvars.ContextVar("active_mesh", default=None)
)


def get_active_mesh() -> DeviceMesh | None:
    """Returns the mesh from the current :func:`mesh_context`, or ``None``."""
    return _active_mesh.get(None)


@contextmanager
def mesh_context(mesh: DeviceMesh) -> Iterator[DeviceMesh]:
    """Publishes ``mesh`` to spec-first :class:`NamedMapping` constructions.

    JAX-style: when a :class:`NamedMapping` is created without an explicit
    mesh inside this block, it picks up ``mesh`` from this context and
    resolves the spec against it.

    .. code-block:: python

        from max.driver import CPU
        from max.experimental.sharding import (
            DeviceMesh,
            get_active_mesh,
            mesh_context,
        )

        mesh = DeviceMesh(
            devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",)
        )

        with mesh_context(mesh):
            # Spec-first constructions inside this block resolve against
            # ``mesh`` without naming it explicitly.
            active = get_active_mesh()

    .. invisible-code-block: python

        assert active is mesh
        assert get_active_mesh() is None  # cleared on block exit
    """
    token = _active_mesh.set(mesh)
    try:
        yield mesh
    finally:
        _active_mesh.reset(token)
