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

"""Tensor-to-mesh bindings: :class:`DeviceMapping` and JAX-style :class:`NamedMapping` sugar."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .mesh import DeviceMesh, get_active_mesh
from .placements import (
    Partial,
    Placement,
    Replicated,
    Sharded,
)


class ConversionError(Exception):
    """Raised when a sharding spec conversion would lose information."""


SpecEntry = str | tuple[str, ...] | None
"""One entry in a named spec: a mesh axis name, a tuple of names
(multi-axis sharding), or ``None`` for replicated."""


@dataclass(frozen=True)
class DeviceMapping:
    """How a tensor is distributed across a device mesh.

    Args:
        mesh: The device mesh.
        placements: One :class:`Placement` per mesh axis, in mesh-axis
            order.
    """

    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    def __post_init__(self) -> None:
        if len(self.placements) != self.mesh.ndim:
            raise ValueError(
                f"Need one placement per mesh axis ({self.mesh.ndim}), "
                f"got {len(self.placements)}."
            )

    def to_placements(self) -> tuple[Placement, ...]:
        """Back-compat alias for :attr:`placements`."""
        return self.placements

    @property
    def is_fully_replicated(self) -> bool:
        """``True`` when every mesh axis is :class:`Replicated`."""
        return all(isinstance(p, Replicated) for p in self.placements)

    def to_mesh(self, mesh: DeviceMesh) -> DeviceMapping:
        """Rebinds this mapping onto ``mesh`` by axis-name correspondence.

        For each axis in ``mesh``: if its name exists in
        :attr:`self.mesh`, copy that axis's placement; otherwise the
        axis becomes :class:`Replicated`. Axes unique to the source
        mesh drop away.
        """
        old_by_name = dict(
            zip(self.mesh.axis_names, self.placements, strict=False)
        )
        return DeviceMapping(
            mesh,
            tuple(
                old_by_name.get(name, Replicated()) for name in mesh.axis_names
            ),
        )

    def __repr__(self) -> str:
        placement_str = ", ".join(repr(p) for p in self.placements)
        return f"{type(self).__name__}({self.mesh}, [{placement_str}])"


PlacementMapping = DeviceMapping
"""Back-compat alias for :class:`DeviceMapping`."""


def is_fully_replicated(mapping: DeviceMapping) -> bool:
    """``True`` when every mesh axis is :class:`Replicated`."""
    return all(isinstance(p, Replicated) for p in mapping.placements)


class NamedMapping(DeviceMapping):
    """Builds a :class:`DeviceMapping` from a JAX-style named spec.

    Each spec entry corresponds to a tensor dim and names the mesh
    axis that shards it (or ``None`` for replicated). Mesh-axis names
    not present on the target mesh resolve to :class:`Replicated`.
    ``unreduced`` names mesh axes carrying a pending reduction; each
    becomes a :class:`Partial` placement. After construction this is
    a regular :class:`DeviceMapping`; the spec is forgotten.

    Args:
        mesh: The target device mesh, or :obj:`None` to take the one published
            by :func:`~max.experimental.sharding.mesh_context`.
        spec: One entry per tensor dimension.
        unreduced: Mesh axes carrying pending reductions.

    Raises:
        ValueError: If ``mesh`` is :obj:`None` and no
            :func:`~max.experimental.sharding.mesh_context` is active, leaving
            the spec with nothing to resolve against.
    """

    # Kept so :meth:`_resolve` can rebind the spec against another mesh.
    _original_spec: tuple[SpecEntry, ...]
    _original_unreduced: tuple[str, ...]

    def __init__(
        self,
        mesh: DeviceMesh | None = None,
        spec: tuple[SpecEntry, ...] = (),
        *,
        unreduced: Iterable[str] = (),
    ) -> None:
        if mesh is None:
            if (mesh := get_active_mesh()) is None:
                raise ValueError(
                    "NamedMapping needs a mesh to resolve its spec against: "
                    "pass one, or construct inside a mesh_context()."
                )
        unreduced_t = tuple(unreduced)
        placements = _spec_to_placements(mesh, spec, unreduced_t)
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "_original_spec", tuple(spec))
        object.__setattr__(self, "_original_unreduced", unreduced_t)

    def _resolve(self, mesh: DeviceMesh) -> NamedMapping:
        """Rebinds this mapping's original spec against ``mesh``.

        Axis names in the spec that are not present in ``mesh`` are
        treated as :class:`Replicated`, matching the construction-time
        graceful-degradation behavior. Use this when a tensor with a
        stub-mesh :class:`NamedMapping` is being moved onto its real
        target mesh.
        """
        return NamedMapping(
            mesh, self._original_spec, unreduced=self._original_unreduced
        )

    def __repr__(self) -> str:
        """Renders the underlying placements in JAX-style named form."""
        names = self.mesh.axis_names
        spec_parts: list[str] = []
        unreduced: list[str] = []
        mesh_names_by_tensor_dim: dict[int, list[str]] = {}
        for axis, p in enumerate(self.placements):
            if isinstance(p, Sharded):
                mesh_names_by_tensor_dim.setdefault(p.axis, []).append(
                    names[axis]
                )
            elif isinstance(p, Partial):
                unreduced.append(names[axis])
        max_tensor_dim = max(mesh_names_by_tensor_dim, default=-1) + 1
        for tensor_dim in range(max_tensor_dim):
            mesh_names = mesh_names_by_tensor_dim.get(tensor_dim, [])
            if not mesh_names:
                spec_parts.append("None")
            elif len(mesh_names) == 1:
                spec_parts.append(repr(mesh_names[0]))
            else:
                spec_parts.append(
                    "(" + ", ".join(repr(n) for n in mesh_names) + ")"
                )
        spec_str = ", ".join(spec_parts) if spec_parts else ""
        parts = [repr(self.mesh), f"({spec_str})"]
        if unreduced:
            parts.append(
                f"unreduced={{{', '.join(repr(n) for n in unreduced)}}}"
            )
        return f"NamedMapping({', '.join(parts)})"


def _spec_to_placements(
    mesh: DeviceMesh,
    spec: tuple[SpecEntry, ...],
    unreduced: Iterable[str],
) -> tuple[Placement, ...]:
    """Resolves a JAX-style spec into mesh-axis-indexed placements."""
    mesh_axes = set(mesh.axis_names)
    placements: list[Placement] = [Replicated() for _ in range(mesh.ndim)]

    for tensor_dim, entry in enumerate(spec):
        if entry is None:
            continue
        names = (entry,) if isinstance(entry, str) else entry
        for name in names:
            if name not in mesh_axes:
                continue
            ax = mesh.axis_names.index(name)
            if not isinstance(placements[ax], Replicated):
                raise ConversionError(
                    f"Mesh axis {name!r} is already assigned to another "
                    f"tensor dimension; each mesh axis can shard at most one."
                )
            placements[ax] = Sharded(tensor_dim)

    for name in unreduced:
        if name not in mesh_axes:
            continue
        ax = mesh.axis_names.index(name)
        if not isinstance(placements[ax], Replicated):
            raise ConversionError(
                f"Mesh axis {name!r} is both unreduced and used for sharding."
            )
        placements[ax] = Partial()

    return tuple(placements)
