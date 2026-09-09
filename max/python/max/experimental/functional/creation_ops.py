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

"""Creation and random ops for distributed tensors.

``full``, ``ones``, ``zeros``, ``*_like``, ``uniform``, ``gaussian``.
"""

from __future__ import annotations

import builtins
import itertools
from collections.abc import Callable
from typing import Any

from max.driver import Device, DLPackArray
from max.dtype import DType
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    DistributedTensorType,
    Placement,
    PlacementMapping,
    Replicated,
)
from max.experimental.sharding.placements import local_shard_shape_from_global
from max.experimental.tensor import Tensor, TensorType, defaults
from max.graph import (
    DeviceRef,
    DimLike,
    Shape,
    ShapeLike,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.graph.ops.constant import NestedArray, Number

from .collective_ops import transfer_to


def _trace_only() -> bool:
    """Legacy shim — always ``False`` after the Trace/Op rip-out."""
    return False


def _normalized_device(
    device: Device | DeviceMapping | DeviceRef | None,
) -> DeviceMapping:
    """Coerce any device specification to a :class:`DeviceMapping`."""
    if isinstance(device, DeviceMapping):
        return device
    if isinstance(device, DeviceRef):
        device = device.to_device()
    if isinstance(device, Device):
        return PlacementMapping(DeviceMesh.single(device), (Replicated(),))
    _, resolved = defaults(None, None)
    return PlacementMapping(DeviceMesh.single(resolved), (Replicated(),))


def _device_from_like(
    like: Tensor | TensorType | DistributedTensorType,
) -> Device | DeviceMapping:
    """Extract device or mapping from a tensor-like for ``*_like`` factories."""
    if isinstance(like, Tensor):
        if like.is_distributed:
            return PlacementMapping(like.mesh, like.placements)
        return like.device
    if isinstance(like, DistributedTensorType):
        return PlacementMapping(like.mesh, like.placements)
    return like.device.to_device()


def _reject_sharded_creation(mapping: DeviceMapping, op_name: str) -> None:
    """Raises if ``mapping`` localizes any tensor axis (sharded creation is unsupported)."""
    for p in mapping.to_placements():
        if p.localized_axis() is not None:
            raise ValueError(
                f"{op_name}: cannot create with sharded placement {p!r}. "
                f"Use Replicated and shard after creation."
            )


def full(
    shape: ShapeLike,
    value: Number,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Creates a tensor filled with a single value.

    When ``device`` is a
    :class:`~max.experimental.sharding.DeviceMapping`, the result is
    distributed across that mesh according to its placements.

    Args:
        shape: The shape of the resulting tensor.
        value: The fill value.
        dtype: The data type of the tensor. Defaults to ``float32`` on
            CPU or ``bfloat16`` on accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement. Defaults to the current realization
            context's device.

    Returns:
        A tensor of the requested shape, dtype, and placement with
        every element set to ``value``.
    """
    mapping = _normalized_device(device)
    resolved_dtype, _ = defaults(dtype, mapping.mesh.devices[0])
    mesh = mapping.mesh
    placements = mapping.to_placements()
    shard_shapes = local_shard_shape_from_global(Shape(shape), mesh, placements)
    with ensure_context():
        tvs = [
            ops.broadcast_to(
                ops.constant(
                    value,
                    resolved_dtype,
                    DeviceRef.from_device(mesh.devices[i]),
                ),
                list(shard_shapes[i]),
            )
            for i in builtins.range(mesh.num_devices)
        ]
        return Tensor.from_shard_values(
            [TensorValue(tv) for tv in tvs], mapping
        )


def ones(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Creates a tensor filled with ones.

    Args:
        shape: The shape of the resulting tensor.
        dtype: The data type. Defaults to ``float32`` on CPU or
            ``bfloat16`` on accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A tensor of the requested shape, dtype, and placement with every
        element set to ``1``.
    """
    return full(shape, 1.0, dtype=dtype, device=device)


def zeros(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Creates a tensor filled with zeros.

    Args:
        shape: The shape of the resulting tensor.
        dtype: The data type. Defaults to ``float32`` on CPU or
            ``bfloat16`` on accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A tensor of the requested shape, dtype, and placement with every
        element set to ``0``.
    """
    return full(shape, 0.0, dtype=dtype, device=device)


def _full_like_distributed(like: Tensor, value: Number) -> Tensor:
    """Build a ``full`` tensor from *like*'s per-shard TV shapes directly."""
    mapping = PlacementMapping(like.mesh, like.placements)
    mesh = mapping.mesh
    resolved_dtype, _ = defaults(like.dtype, mesh.devices[0])
    with ensure_context():
        tvs = [
            TensorValue(
                ops.broadcast_to(
                    ops.constant(
                        value,
                        resolved_dtype,
                        DeviceRef.from_device(mesh.devices[i]),
                    ),
                    list(tv.shape),
                )
            )
            for i, tv in builtins.enumerate(like.graph_values)
        ]
        return Tensor.from_shard_values(tvs, mapping)


def full_like(
    like: Tensor | TensorType | DistributedTensorType, value: Number
) -> Tensor:
    """Creates a tensor filled with a single value, matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.
        value: The fill value.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with every element set to ``value``.
    """
    if isinstance(like, Tensor) and like.is_distributed and not _trace_only():
        # Eager path: preserves per-rank symbolic dim identity. Under trace_only
        # the skeleton has no graph values; route through ``full`` which has
        # its own dispatch + skeleton path.
        return _full_like_distributed(like, value)
    return full(
        like.shape, value, dtype=like.dtype, device=_device_from_like(like)
    )


def ones_like(like: Tensor | TensorType | DistributedTensorType) -> Tensor:
    """Creates a tensor filled with ones, matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with every element set to ``1``.
    """
    if isinstance(like, Tensor) and like.is_distributed and not _trace_only():
        return _full_like_distributed(like, 1.0)
    return full(
        like.shape, 1.0, dtype=like.dtype, device=_device_from_like(like)
    )


def zeros_like(like: Tensor | TensorType | DistributedTensorType) -> Tensor:
    """Creates a tensor filled with zeros, matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with every element set to ``0``.
    """
    if isinstance(like, Tensor) and like.is_distributed and not _trace_only():
        return _full_like_distributed(like, 0.0)
    return full(
        like.shape, 0.0, dtype=like.dtype, device=_device_from_like(like)
    )


# Base-seed stride 100 = max devices per group; group = same Replicated axes.
_BASE_SEED_COUNTER = 0


def _next_base_seed() -> int:
    """Returns a fresh base seed and advances the counter by 100."""
    global _BASE_SEED_COUNTER
    val = _BASE_SEED_COUNTER
    _BASE_SEED_COUNTER += 100
    return val


def _shard_group_ids(
    mesh_shape: tuple[int, ...],
    placements: tuple[Placement, ...],
) -> list[int]:
    """Assign a group ID to each device so devices that share data share a seed."""
    ndim = len(mesh_shape)
    ids = []
    for coords in itertools.product(*(builtins.range(s) for s in mesh_shape)):
        group_id = 0
        stride = 1
        for ax in reversed(builtins.range(ndim)):
            if placements[ax].localized_axis() is not None:
                group_id += coords[ax] * stride
                stride *= mesh_shape[ax]
        ids.append(group_id)
    return ids


def _distributed_random_op(
    op_fn: Callable[..., TensorValue],
    shape: ShapeLike,
    *,
    dtype: DType,
    device: DeviceMapping,
    **op_kwargs: Any,
) -> Tensor:
    """Shared body for distributed random sampling (``uniform``, ``gaussian``)."""
    with ensure_context():
        if device.mesh.num_devices == 1:
            tt = TensorType(
                dtype, shape, DeviceRef.from_device(device.mesh.devices[0])
            )
            return Tensor.from_graph_value(op_fn(tt, **op_kwargs))
        mesh = device.mesh
        placements = device.to_placements()
        shard_shapes = local_shard_shape_from_global(
            Shape(shape), mesh, placements
        )
        # Random ops support Replicated and any localizing placement; Partial is invalid.
        assert all(
            isinstance(p, Replicated) or p.localized_axis() is not None
            for p in placements
        )
        group_ids = _shard_group_ids(mesh.mesh_shape, placements)
        n_unique = builtins.max(group_ids) + 1
        base = _next_base_seed()
        shard_values = []
        for i, d in enumerate(mesh.devices):
            tt = TensorType(dtype, shard_shapes[i], DeviceRef.from_device(d))
            ops.random.set_seed(base + group_ids[i])
            shard_values.append(op_fn(tt, **op_kwargs))
        ops.random.set_seed(base + n_unique)
        return Tensor.from_shard_values(shard_values, device)


def uniform(
    shape: ShapeLike = (),
    range: tuple[float, float] = (0, 1),
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Samples values uniformly from the half-open interval ``[range[0], range[1])``.

    Values satisfy ``range[0] <= x < range[1]``. When ``device`` is a
    :class:`~max.experimental.sharding.DeviceMapping`, each Sharded
    axis draws an independent stream while shards on Replicated axes
    draw identical values.

    .. code-block:: python

        from max.experimental import functional as F

        result = F.uniform((2, 3), range=(0.0, 1.0))
        # result is a (2, 3) tensor sampled uniformly from [0.0, 1.0).

    Args:
        shape: The shape of the resulting tensor.
        range: A ``(low, high)`` pair giving the half-open interval to
            sample from. Defaults to ``(0, 1)``.
        dtype: The data type of the tensor.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A ``Tensor`` of the requested shape, dtype, and placement with
        values sampled uniformly from ``[range[0], range[1])``.
    """
    mapping = _normalized_device(device)
    resolved_dtype, _ = defaults(dtype, mapping.mesh.devices[0])
    return _distributed_random_op(
        ops.random.uniform,
        shape,
        dtype=resolved_dtype,
        device=mapping,
        range=range,
    )


def gaussian(
    shape: ShapeLike = (),
    mean: float = 0.0,
    std: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Samples values from a Gaussian (normal) distribution with the given mean and standard deviation.

    When ``device`` is a
    :class:`~max.experimental.sharding.DeviceMapping`, each Sharded
    axis draws an independent stream while shards on Replicated axes
    draw identical values.

    .. code-block:: python

        from max.experimental import functional as F

        result = F.gaussian((2, 3), mean=0.0, std=1.0)
        # result is a (2, 3) tensor sampled from a standard normal distribution.

    Args:
        shape: The shape of the resulting tensor.
        mean: The mean of the distribution. Defaults to ``0.0``.
        std: The standard deviation of the distribution. Defaults to
            ``1.0``.
        dtype: The data type of the tensor.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A ``Tensor`` of the requested shape, dtype, and placement with
        values sampled from ``Normal(mean, std**2)``.
    """
    mapping = _normalized_device(device)
    resolved_dtype, _ = defaults(dtype, mapping.mesh.devices[0])
    return _distributed_random_op(
        ops.random.gaussian,
        shape,
        dtype=resolved_dtype,
        device=mapping,
        mean=mean,
        std=std,
    )


normal = gaussian


def _random_like_distributed(
    like: Tensor,
    *,
    op: str,
    range: tuple[float, float] | None = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    """Per-shard random sampling that preserves *like*'s per-rank symbol names."""
    mapping = PlacementMapping(like.mesh, like.placements)
    mesh = mapping.mesh
    resolved_dtype, _ = defaults(like.dtype, mesh.devices[0])
    placements = mapping.to_placements()
    assert all(
        isinstance(p, Replicated) or p.localized_axis() is not None
        for p in placements
    )
    group_ids = _shard_group_ids(mesh.mesh_shape, placements)
    n_unique = builtins.max(group_ids) + 1
    base = _next_base_seed()
    with ensure_context():
        shard_values: list[TensorValue] = []
        for i, tv in builtins.enumerate(like.graph_values):
            tt = TensorType(
                resolved_dtype,
                tv.shape,
                DeviceRef.from_device(mesh.devices[i]),
            )
            ops.random.set_seed(base + group_ids[i])
            if op == "uniform":
                assert range is not None
                shard_values.append(ops.random.uniform(tt, range=range))
            else:
                shard_values.append(ops.random.gaussian(tt, mean=mean, std=std))
        ops.random.set_seed(base + n_unique)
        return Tensor.from_shard_values(shard_values, mapping)


def uniform_like(
    like: Tensor | TensorType | DistributedTensorType,
    range: tuple[float, float] = (0, 1),
) -> Tensor:
    """Samples uniform values matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.
        range: A ``(low, high)`` pair giving the half-open interval to
            sample from. Defaults to ``(0, 1)``.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with values sampled uniformly from ``[range[0], range[1])``.
    """
    if isinstance(like, Tensor) and like.is_distributed and not _trace_only():
        return _random_like_distributed(like, op="uniform", range=range)
    return uniform(
        like.shape,
        range=range,
        device=_device_from_like(like),
        dtype=like.dtype,
    )


def gaussian_like(
    like: Tensor | TensorType | DistributedTensorType,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    """Samples Gaussian values matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.
        mean: The mean of the distribution. Defaults to ``0.0``.
        std: The standard deviation of the distribution. Defaults to
            ``1.0``.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with values sampled from ``Normal(mean, std**2)``.
    """
    if isinstance(like, Tensor) and like.is_distributed and not _trace_only():
        return _random_like_distributed(like, op="gaussian", mean=mean, std=std)
    return gaussian(
        like.shape,
        mean=mean,
        std=std,
        device=_device_from_like(like),
        dtype=like.dtype,
    )


normal_like = gaussian_like


def hann_window(
    window_length: int,
    *,
    periodic: bool = True,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Computes a Hann window of a given length.

    For a symmetric window of ``N`` points with ``N > 1``, the value at index
    ``n`` is:

    .. code-block:: text

        w[n] = 0.5 * (1 - cos(2 * pi * n / (N - 1)))

    A window of length ``0`` is empty, and a window of length ``1`` is
    ``[1]``.

    A periodic window instead computes ``N + 1`` points and drops the last
    one, which makes it suitable for spectral analysis.

    .. code-block:: python

        from max.experimental import functional as F

        result = F.hann_window(4, periodic=True)
        # result holds [0.0, 0.5, 1.0, 0.5].

    Args:
        window_length: The number of points in the window.
        periodic: Whether to return a periodic window. When ``True``,
            computes a symmetric window of ``window_length + 1`` points and
            drops the last point, so that ``hann_window(L, periodic=True)``
            equals ``hann_window(L + 1, periodic=False)[:-1]``. Defaults to
            ``True``.
        dtype: The output tensor's data type. Defaults to ``float32`` on CPU
            or ``bfloat16`` on accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping`. Sharded
            placement is not supported.

    Returns:
        A 1-D ``Tensor`` of shape ``(window_length,)`` containing the
        window.

    Raises:
        ValueError: If ``window_length`` is negative, or if ``device``
            requests a sharded placement.
        TypeError: If ``window_length`` isn't an integer.
    """
    mapping = _normalized_device(device)
    _reject_sharded_creation(mapping, "hann_window")
    resolved_dtype, _ = defaults(dtype, mapping.mesh.devices[0])
    mesh = mapping.mesh
    with ensure_context():
        shard_values = [
            ops.hann_window(
                window_length,
                DeviceRef.from_device(d),
                periodic=periodic,
                dtype=resolved_dtype,
            )
            for d in mesh.devices
        ]
        return Tensor.from_shard_values(shard_values, mapping)


def range(
    start: TensorValueLike,
    stop: TensorValueLike,
    step: TensorValueLike = 1,
    out_dim: DimLike | None = None,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Creates a sequence of evenly spaced values from ``start`` to ``stop``.

    The sequence begins at ``start`` and increments by ``step``, stopping
    before ``stop`` (the upper bound is exclusive).

    ``stop - start`` must be zero or have the same sign as ``step``.
    Currently, graph compilation fails when ``stop - start`` isn't evenly
    divisible by ``step``. For example, ``range(0, 5, 2)`` should produce
    three values, ``[0, 2, 4]``, but shape inference declares an output length
    of 2. The generated values therefore don't fit the declared output shape.

    .. code-block:: python

        from max.dtype import DType
        from max.experimental import functional as F

        result = F.range(0, 5, 1, dtype=DType.float32)
        # result holds [0.0, 1.0, 2.0, 3.0, 4.0].

    Args:
        start: The first value in the sequence. Must be a scalar value.
        stop: The exclusive upper bound. The sequence stops before this
            value. Must be a scalar value.
        step: The spacing between consecutive values. Must be non-zero.
            Defaults to ``1``.
        out_dim: The expected length of the output. Required when dynamic
            scalar tensor inputs prevent static length inference. When omitted,
            it's computed from scalar literals.
        dtype: The element type of the result tensor. Defaults to
            ``float32`` on CPU or ``bfloat16`` on accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping`. Sharded
            placement is not supported.

    Returns:
        A ``Tensor`` containing the generated sequence.

    Raises:
        ValueError: If ``out_dim`` is omitted for dynamic scalar inputs, if
            any input isn't scalar, if any input isn't on the CPU, or if
            ``device`` requests a sharded placement.
        RuntimeError: If a statically known interval isn't evenly divisible
            by ``step``, causing the inferred output length to disagree with
            the number of generated values.
    """
    mapping = _normalized_device(device)
    _reject_sharded_creation(mapping, "range")
    resolved_dtype, _ = defaults(dtype, mapping.mesh.devices[0])
    mesh = mapping.mesh
    with ensure_context():
        shard_values = [
            ops.range(
                start,
                stop,
                step,
                out_dim,
                dtype=resolved_dtype,
                device=DeviceRef.from_device(d),
            )
            for d in mesh.devices
        ]
        return Tensor.from_shard_values(shard_values, mapping)


# Backward-compat alias: callers use both F.arange and F.range.
arange = range


def constant(
    value: DLPackArray | NestedArray | Number,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Creates a constant tensor from a Python literal or array-like value.

    For an array-like ``value``, the array's own dtype is preserved when
    ``dtype`` is :obj:`None`. For a Python scalar or sequence, ``dtype``
    defaults to ``float32`` on CPU or ``bfloat16`` on accelerators.

    .. code-block:: python

        from max.dtype import DType
        from max.experimental import functional as F

        result = F.constant([[1.0, 2.0], [3.0, 4.0]], DType.float32)
        # result holds [[1.0, 2.0], [3.0, 4.0]].

    .. caution::

        Loading a constant can lose precision. For example, loading
        ``16777217`` as a ``float32`` produces ``16777216.0``.

    .. caution::

        When ``device`` is a multi-device
        :class:`~max.experimental.sharding.DeviceMapping`, the full
        ``value`` is materialized on every device. A mapping that shards a
        tensor axis therefore duplicates the data and reports a global shape
        scaled by the number of devices, without raising. Create the tensor
        Replicated and shard it afterward.

    Args:
        value: The value to embed. A Python scalar, a (nested) sequence of
            numbers, or an array-like object that supports DLPack, such as a
            NumPy array.
        dtype: The constant tensor's element type. For an array-like
            ``value``, defaults to the array's dtype. For a Python scalar or
            sequence, defaults to ``float32`` on CPU or ``bfloat16`` on
            accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement. Defaults to an accelerator if available,
            otherwise the CPU.

    Returns:
        A ``Tensor`` containing the constant, with the same shape as
        ``value``. A scalar ``value`` produces a rank-0 tensor.

    Raises:
        TypeError: If ``dtype`` is a sub-byte type.
        ValueError: If ``value`` is a nested sequence that isn't rectangular,
            if an integer in ``value`` is out of range for ``dtype``, or if
            ``dtype`` doesn't match the dtype of an array-like ``value``.
    """
    mapping = _normalized_device(device)
    mesh = mapping.mesh
    if isinstance(value, DLPackArray):
        resolved_dtype = dtype
    else:
        resolved_dtype, _ = defaults(dtype, mesh.devices[0])
    with ensure_context():
        tvs = [
            ops.constant(value, resolved_dtype, DeviceRef.from_device(d))
            for d in mesh.devices
        ]
        return Tensor.from_shard_values(
            [TensorValue(tv) for tv in tvs],
            mapping,
        )


def constant_external(
    name: str,
    type: TensorType,
    device: Device | DeviceMapping | DeviceRef | None = None,
    align: int | None = None,
    is_placeholder: bool = False,
) -> Tensor:
    """Creates a constant tensor from external (weight) data.

    External constants are loaded at compile time from the named
    weight rather than being inlined into the graph.

    Args:
        name: The external symbol name to load (typically a weight
            identifier).
        type: The :class:`~max.graph.TensorType` describing the
            constant's shape and dtype.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.
        align: The alignment of the constant. If not provided,
            the default alignment for the type's dtype will be used.
        is_placeholder: When :obj:`True`, marks the constant as a placeholder
            whose name is resolved by the enclosing subgraph call's ``prefix``
            (see :func:`max.graph.ops.call`). Used to thread per-layer weights
            through a shared subgraph body under layer-relative names.

    Returns:
        A tensor on the requested placement initialized from the
        external data.
    """
    mapping = _normalized_device(device) if device is not None else None
    with ensure_context():
        tv = ops.constant_external(name, type, align, is_placeholder)
        t = Tensor.from_graph_value(tv)
    if mapping is not None:
        return transfer_to(t, mapping)
    return t
