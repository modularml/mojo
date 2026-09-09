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

"""Collective operations for distributed :class:`~max.experimental.tensor.Tensor` inputs.

Each collective acts along a single mesh axis and is intended for use on
tensors that are sharded across a multi-device mesh. :func:`transfer_to`
is the universal entry point for moving a tensor between devices or
placements.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.driver import Accelerator, Device
from max.experimental import tensor as _experimental_tensor
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.sharding.per_shard_dim import global_dim
from max.experimental.tensor import Tensor
from max.graph import BufferValue, DeviceRef, Shape, TensorValue, ops
from max.graph.dim import Dim, StaticDim, SymbolicDim
from max.graph.ops.slice_tensor import SliceIndex


def _devices_are_unique(shards: list[TensorValue]) -> bool:
    """True when all shards live on distinct physical devices."""
    devices = [str(s.device) for s in shards]
    return len(set(devices)) == len(devices)


def _signal_buffers(mesh: DeviceMesh) -> list[BufferValue] | None:
    """Returns the active context's signal buffers for ``mesh``, if available."""
    ctx = _experimental_tensor.current_realization_context(None)
    if ctx is None:
        return None

    if hasattr(ctx, "signal_buffers") and ctx.signal_buffers is not None:
        if not any(isinstance(d, Accelerator) for d in mesh.devices):
            return None
        return ctx.signal_buffers

    if hasattr(ctx, "ensure_signal_buffers"):
        return ctx.ensure_signal_buffers(mesh)

    return None


def _mesh_axis_groups(mesh: DeviceMesh, mesh_axis: int) -> list[list[int]]:
    """Partitions device indices into groups communicating along ``mesh_axis``."""
    axis_size = mesh.mesh_shape[mesh_axis]
    stride = 1
    for k in range(mesh_axis + 1, len(mesh.mesh_shape)):
        stride *= mesh.mesh_shape[k]
    groups: list[list[int]] = []
    visited: set[int] = set()
    for base in range(mesh.num_devices):
        if base in visited:
            continue
        group = [base + i * stride for i in range(axis_size)]
        visited.update(group)
        groups.append(group)
    return groups


def _even_split_sizes(dim: int, n: int) -> list[int]:
    """Splits ``dim`` into ``n`` sizes that differ by at most 1."""
    base, rem = divmod(dim, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _even_split_along_axis(
    sv: TensorValue, axis: int, n: int
) -> list[TensorValue]:
    """Splits ``sv`` into ``n`` load-balanced chunks along ``axis``."""
    dim = sv.shape[axis]
    if isinstance(dim, StaticDim):
        return list(ops.split(sv, _even_split_sizes(int(dim), n), axis=axis))

    rank_ndim = len(sv.shape)
    chunks: list[TensorValue] = []
    for i in range(n):
        start = (i * dim) // n
        stop = ((i + 1) * dim) // n
        size = stop - start
        start_tv = ops.shape_to_tensor([start])
        stop_tv = ops.shape_to_tensor([stop])
        indices: list[SliceIndex] = [slice(None)] * rank_ndim
        indices[axis] = (slice(start_tv, stop_tv, 1), size)
        chunks.append(ops.slice_tensor(sv, indices))
    return chunks


def _rebind_axis(tv: TensorValue, axis: int, new_dim: Dim) -> TensorValue:
    """Rebinds only ``axis`` of ``tv`` to ``new_dim``; leaves every other axis as-is."""
    new_shape = list(tv.shape)
    new_shape[axis] = new_dim
    return ops.rebind(tv, Shape(new_shape))


def _make_split_symbolic_dim(
    dim: Dim,
    axis_name: str,
    coord: int,
    tensor_axis: int,
    fallback_prefix: str,
) -> SymbolicDim:
    """Mints a per-device symbolic dim for a split axis, severing it from ``dim``.

    The chunks of a dynamic axis are ordinarily algebraic functions of their
    parent, so their extents stay related to it. This instead gives each
    device an independent symbol, for the axis whose per-device extents
    nothing global relates.
    """
    dim = global_dim(dim)
    if isinstance(dim, SymbolicDim):
        return SymbolicDim(f"{dim.name}_{axis_name}_{coord}")
    return SymbolicDim(
        f"{fallback_prefix}_{axis_name}_{coord}_axis{tensor_axis}"
    )


def _apply_per_group(
    t: Tensor,
    mesh_axis: int,
    new_placement: Replicated | Sharded | Partial,
    *,
    hw_op: Callable[[list[TensorValue], list[BufferValue]], list[TensorValue]],
    sim_op: Callable[[list[TensorValue], tuple[int, ...]], list[TensorValue]],
) -> Tensor:
    """Applies a collective operation per group along a mesh axis."""
    mesh = t.mesh
    groups = _mesh_axis_groups(mesh, mesh_axis)
    new_p = list(t.placements)
    new_p[mesh_axis] = new_placement
    new_placements = tuple(new_p)

    with ensure_context():
        shards = [s.__tensorvalue__() for s in t.local_shards]
        result = list(shards)

        use_hw = _devices_are_unique(shards) and len(groups[0]) > 1
        signal_bufs = _signal_buffers(mesh) if use_hw else None

        for group in groups:
            group_inputs = [shards[idx] for idx in group]
            if signal_bufs is not None:
                group_signals = [signal_bufs[idx] for idx in group]
                group_result = hw_op(group_inputs, group_signals)
            else:
                group_result = sim_op(group_inputs, tuple(group))
            for i, idx in enumerate(group):
                result[idx] = group_result[i]

        # Per-rank IR after the collective carries the honest algebraic
        # form (e.g. ``batch_dp_0 + batch_dp_1``); :attr:`Tensor.shape`
        # reads back the same dim on every rank and collapses the wrapper.
        return Tensor.from_shard_values(
            result,
            PlacementMapping(mesh, new_placements),
        )


def _colocate_then_redistribute(
    inputs: list[TensorValue],
    op: Callable[[list[TensorValue]], TensorValue],
) -> list[TensorValue]:
    """Run a same-device op on per-rank inputs and redistribute the result."""
    target_device = inputs[0].device
    colocated = [
        v if v.device == target_device else v.to(target_device) for v in inputs
    ]
    result = op(colocated)
    return [
        result if v.device == target_device else result.to(v.device)
        for v in inputs
    ]


def _sim_reduce_scatter(
    inputs: list[TensorValue], scatter_axis: int
) -> list[TensorValue]:
    """Simulate reduce-scatter: sum the partial inputs, then scatter the chunks.

    Used when no signal buffers are available (single-device or simulated
    multi-device). Reduces on the first input's device, splits along
    ``scatter_axis`` into one chunk per rank, and ships each chunk to its
    rank's device.
    """
    n = len(inputs)
    target_device = inputs[0].device
    colocated = [
        v if v.device == target_device else v.to(target_device) for v in inputs
    ]
    reduced = functools.reduce(ops.add, colocated)
    # TODO(MXF-493): `_even_split_along_axis` splits uneven chunks and return
    # the smallest chunks first, while the actual reduce-scatter kernel splits
    # the chunks from largest to smallest.
    chunks = _even_split_along_axis(reduced, scatter_axis, n)
    return [
        chunk
        if inputs[i].device == target_device
        else chunk.to(inputs[i].device)
        for i, chunk in enumerate(chunks)
    ]


def allreduce_sum(t: Tensor, mesh_axis: int = 0) -> Tensor:
    """All-reduces a tensor by summing its shards across a mesh axis.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Partial` to
    :class:`~max.experimental.sharding.Replicated`. Every device on
    ``mesh_axis`` ends up holding the sum of all inputs along that axis.

    Args:
        t: The input distributed tensor.
        mesh_axis: The mesh axis along which to reduce.

    Returns:
        A tensor with the same per-device values everywhere along
        ``mesh_axis``.
    """
    return _apply_per_group(
        t,
        mesh_axis,
        Replicated(),
        hw_op=lambda inputs, sigs: ops.allreduce.sum(inputs, sigs),
        sim_op=lambda inputs, _group: _colocate_then_redistribute(
            inputs, lambda c: functools.reduce(ops.add, c)
        ),
    )


def allgather(
    t: Tensor,
    tensor_axis: int = 0,
    mesh_axis: int = 0,
) -> Tensor:
    """All-gathers a tensor's shards along a mesh axis.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Sharded` to
    :class:`~max.experimental.sharding.Replicated`. Each device gathers
    the shards from its peers and concatenates them along ``tensor_axis``.

    Args:
        t: The input distributed tensor.
        tensor_axis: The tensor axis along which the shards are concatenated.
        mesh_axis: The mesh axis whose placement changes from Sharded to
            Replicated.

    Returns:
        A tensor with the full data replicated across ``mesh_axis``.
    """
    return _apply_per_group(
        t,
        mesh_axis,
        Replicated(),
        hw_op=lambda inputs, sigs: ops.allgather(
            inputs, sigs, axis=tensor_axis
        ),
        sim_op=lambda inputs, _group: _colocate_then_redistribute(
            inputs, lambda c: ops.concat(c, tensor_axis)
        ),
    )


def reduce_scatter(
    t: Tensor,
    scatter_axis: int = 0,
    mesh_axis: int = 0,
) -> Tensor:
    """Reduces a tensor across a mesh axis and scatters the result.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Partial` to
    :class:`~max.experimental.sharding.Sharded`. Each device contributes
    to the sum and ends up with one shard of the reduced tensor along
    ``scatter_axis``.

    Args:
        t: The input distributed tensor.
        scatter_axis: The tensor axis along which the reduced result is
            sharded.
        mesh_axis: The mesh axis whose placement changes from Partial to
            Sharded.

    Returns:
        A tensor with the reduced and re-sharded result.
    """
    return _apply_per_group(
        t,
        mesh_axis,
        Sharded(scatter_axis),
        hw_op=lambda inputs, sigs: ops.reducescatter.sum(
            inputs, sigs, axis=scatter_axis
        ),
        sim_op=lambda inputs, _group: _sim_reduce_scatter(inputs, scatter_axis),
    )


def _local_split(t: Tensor, mesh_axis: int, target: Sharded) -> Tensor:
    """``Replicated -> Sharded``: each device slices its local copy with no communication."""
    tensor_axis = target.axis
    mesh = t.mesh
    n = mesh.mesh_shape[mesh_axis]
    groups = _mesh_axis_groups(mesh, mesh_axis)

    new_p = list(t.placements)
    new_p[mesh_axis] = target
    new_placements = tuple(new_p)

    with ensure_context():
        shards: list[TensorValue] = [
            s.__tensorvalue__() for s in t.local_shards
        ]
        result: list[TensorValue] = list(shards)
        for group in groups:
            for rank_in_group, idx in enumerate(group):
                split_chunks = _even_split_along_axis(
                    shards[idx], tensor_axis, n
                )
                result[idx] = split_chunks[rank_in_group]
        return Tensor.from_shard_values(
            result,
            PlacementMapping(mesh, new_placements),
        )


def _scatter(t: Tensor, target: DeviceMapping) -> Tensor:
    """Distributes a non-distributed tensor across a mesh."""
    assert not t.is_distributed, "_scatter expects a non-distributed tensor"
    mesh = target.mesh
    placements = target.to_placements()

    # Size-1 axes cannot host a shard.
    src_shape = t.shape
    for mesh_axis, p in enumerate(placements):
        ax = p.localized_axis()
        if ax is None or not 0 <= ax < len(src_shape):
            continue
        dim = src_shape[ax]
        if isinstance(dim, StaticDim) and dim.dim == 1:
            from max.experimental.sharding.mode import ShardingError

            raise ShardingError(
                f"_scatter: placement {p!r} on mesh axis "
                f"{mesh.axis_names[mesh_axis]!r} targets tensor axis {ax} "
                f"with static extent 1; cannot split a size-1 axis. Use "
                f"Replicated() on this mesh axis instead."
            )

    with ensure_context():
        tv = t.__tensorvalue__()

        shard_tvs = [tv]
        for mesh_axis in range(mesh.ndim):
            p = placements[mesh_axis]
            n = mesh.mesh_shape[mesh_axis]
            tensor_axis = p.localized_axis()
            if tensor_axis is not None:
                new_tvs: list[TensorValue] = []
                for sv in shard_tvs:
                    new_tvs.extend(_even_split_along_axis(sv, tensor_axis, n))
                shard_tvs = new_tvs
            elif isinstance(p, Replicated):
                shard_tvs = [sv for sv in shard_tvs for _ in range(n)]
            else:
                raise ValueError(
                    f"Cannot scatter with placement {type(p).__name__}; "
                    "scatter requires Replicated or a placement that localizes "
                    "a single tensor axis (override ``localized_axis()``)."
                )
        shard_tvs = [
            ops.transfer_to(sv, DeviceRef.from_device(mesh.devices[i]))
            for i, sv in enumerate(shard_tvs)
        ]
        return Tensor.from_shard_values(
            shard_tvs,
            PlacementMapping(mesh, placements),
        )


def distributed_broadcast(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Replicates a non-distributed tensor onto every device with one collective.

    Args:
        t: A non-distributed source tensor.
        mesh: The device mesh to replicate onto.

    Returns:
        A distributed tensor with :class:`Replicated` placement on every axis.
    """
    with ensure_context():
        signal_buffers = _signal_buffers(mesh)
        if signal_buffers is None:
            raise RuntimeError("No signal buffers available for broadcast.")
        if t.mesh.num_devices > 1:
            raise RuntimeError(
                "`F.distributed_broadcast` requires the source tensor to be non-distributed."
            )
        shards = ops.distributed_broadcast(TensorValue(t), signal_buffers)
        return Tensor.from_shard_values(
            shards,
            PlacementMapping(mesh, (Replicated(),) * mesh.ndim),
        )


def transfer_to(
    t: Tensor, target: Device | DeviceMapping | DeviceRef
) -> Tensor:
    """Moves a tensor to a target device or device mapping.

    Handles every kind of placement transition: single-device transfers,
    scattering an unsharded tensor onto a mesh, redistributing across
    placements on the same mesh, and gathering then re-distributing
    across different meshes.

    Args:
        t: The source tensor, distributed or single-device.
        target: A :class:`~max.driver.Device` to move to a single device,
            or a :class:`~max.experimental.sharding.DeviceMapping`
            describing the target mesh and placement.

    Returns:
        A tensor with the requested placement on the target device or mesh.
    """
    if isinstance(target, DeviceRef):
        target = target.to_device()
    if isinstance(target, Device):
        target = PlacementMapping(DeviceMesh.single(target), (Replicated(),))

    if t.real and not t.is_distributed and target.mesh.num_devices == 1:
        mesh_device = target.mesh.devices[0]
        if t.device == mesh_device:
            return t
        return Tensor(storage=t.driver_tensor.to(mesh_device))

    target_p = target.to_placements()

    if not t.is_distributed:
        return _scatter(t, target)

    # Cross-mesh: gather, transfer, scatter.
    if t.mesh != target.mesh:
        source_mesh = t.mesh
        target_mesh = target.mesh

        replicated_p = tuple(Replicated() for _ in range(source_mesh.ndim))
        if t.placements != replicated_p:
            t = transfer_to(t, PlacementMapping(source_mesh, replicated_p))

        single = t.local_shards[0]
        with ensure_context():
            if single.real:
                buf = single.driver_tensor.to(target_mesh.devices[0])
                single = Tensor(storage=buf)
            else:
                tv = ops.transfer_to(
                    single.__tensorvalue__(),
                    DeviceRef.from_device(target_mesh.devices[0]),
                )
                single = Tensor.from_graph_value(tv)

        if target_mesh.num_devices == 1:
            return single
        return _scatter(single, target)

    # Phase-ordered redistribution: reduce Partials first, then unwind
    # Sharded to Replicated, then anything remaining (R -> S, etc.). This
    # avoids double-sharded transit on one tensor dim when a tensor axis
    # is sharded along multiple mesh axes.
    if t.placements == target_p:
        return t

    mesh = t.mesh

    # Phase 0: resolve Partials first (allreduce / reduce_scatter).
    for ax in range(mesh.ndim):
        cp, tp = t.placements[ax], target_p[ax]
        if isinstance(cp, Partial) and cp != tp:
            t = _axis_transition(t, cp, tp, mesh_axis=ax)

    # Phase 1: allgather Sharded to Replicated; reverse mesh-axis order
    # preserves element ordering when one tensor axis is sharded along
    # multiple mesh axes.
    for ax in reversed(range(mesh.ndim)):
        cp, tp = t.placements[ax], target_p[ax]
        if isinstance(cp, Sharded) and cp != tp:
            t = _axis_transition(t, cp, Replicated(), mesh_axis=ax)

    # Phase 2: anything still mismatched (now Replicated -> {Sharded, Partial}).
    for ax in range(mesh.ndim):
        cp, tp = t.placements[ax], target_p[ax]
        if cp != tp:
            t = _axis_transition(t, cp, tp, mesh_axis=ax)

    return t


def _axis_transition(
    t: Tensor, source: Placement, target: Placement, *, mesh_axis: int
) -> Tensor:
    """Inserts the collective for ``source -> target`` on one mesh axis."""
    if source == target:
        return t
    if isinstance(source, Replicated) and isinstance(target, Sharded):
        return _local_split(t, mesh_axis=mesh_axis, target=target)
    if isinstance(source, Sharded) and isinstance(target, Replicated):
        return allgather(
            t,
            tensor_axis=source.axis,
            mesh_axis=mesh_axis,
        )
    if isinstance(source, Sharded) and isinstance(target, Sharded):
        if source == target:
            return t
        if source.axis == target.axis:
            return t
        t = allgather(
            t,
            tensor_axis=source.axis,
            mesh_axis=mesh_axis,
        )
        return _local_split(t, mesh_axis=mesh_axis, target=target)
    if isinstance(source, Partial):
        if source.reduce_op.value in ("min", "max"):
            raise NotImplementedError(
                f"Partial({source.reduce_op}) redistribution is not supported."
            )
        if isinstance(target, Replicated):
            return allreduce_sum(t, mesh_axis=mesh_axis)
        if isinstance(target, Sharded):
            already_sharded = any(
                i != mesh_axis and p.localized_axis() == target.axis
                for i, p in enumerate(t.placements)
            )
            if already_sharded:
                return allreduce_sum(t, mesh_axis=mesh_axis)
            return reduce_scatter(
                t,
                mesh_axis=mesh_axis,
                scatter_axis=target.axis,
            )
        if isinstance(target, Partial):
            raise NotImplementedError(
                f"Partial({source.reduce_op}) -> "
                f"Partial({target.reduce_op}) redistribution is not "
                "supported."
            )
    # Custom-placement hook for subclasses.
    if hasattr(source, "materialize_to"):
        return source.materialize_to(t, target, mesh_axis=mesh_axis)
    if hasattr(target, "materialize_from"):
        return target.materialize_from(t, source, mesh_axis=mesh_axis)
    raise NotImplementedError(
        f"No transition {type(source).__name__} -> {type(target).__name__}"
    )
