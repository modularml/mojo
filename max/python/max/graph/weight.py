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

"""Defines :class:`Weight` and sharding strategies for distributed MAX graph execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property, partial

from max._core import Value as _Value
from max._core.dialects import mo
from max.dtype import DType

from . import graph, ops
from .quantization import QuantizationEncoding
from .type import DeviceRef, Shape, ShapeLike
from .value import TensorValue, Value


def _compute_shard_range(
    shard_dim: int, shard_idx: int, num_devices: int
) -> tuple[int, int]:
    base_size, remainder = divmod(shard_dim, num_devices)

    # Give first 'remainder' devices an extra column/row.
    if shard_idx < remainder:
        start = shard_idx * (base_size + 1)
        end = start + base_size + 1
    else:
        start = (
            remainder * (base_size + 1) + (shard_idx - remainder) * base_size
        )
        end = start + base_size

    return start, end


def col_sharding_strategy(
    weight: Weight, i: int, num_devices: int
) -> TensorValue:
    """Shards a weight tensor by column for a given device.

    Args:
        weight: The :class:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.

    Returns:
        A :class:`~max.graph.TensorValue` representing the sharded portion of the weight
        for the i-th device.
    """
    start, end = _compute_shard_range(
        shard_dim=int(weight.shape[1]), shard_idx=i, num_devices=num_devices
    )

    return weight[:, start:end]


def row_sharding_strategy(
    weight: Weight, i: int, num_devices: int
) -> TensorValue:
    """Shards a weight tensor by row for a given device.

    Args:
        weight: The :class:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.

    Returns:
        A :class:`~max.graph.TensorValue` representing the sharded portion of the weight
        for the i-th device.
    """
    start, end = _compute_shard_range(
        shard_dim=int(weight.shape[0]), shard_idx=i, num_devices=num_devices
    )

    return weight[start:end]


def replicate_sharding_strategy(
    weight: Weight, i: int, num_devices: int
) -> TensorValue:
    """Replicates the entire weight tensor for a given device.

    Args:
        weight: The :class:`Weight` to replicate.
        i: The index of the current device (unused in this strategy).
        num_devices: The total number of devices (unused in this strategy).

    Returns:
        A :class:`~max.graph.TensorValue` representing the full weight tensor.
    """
    return weight[:]


def head_aware_col_sharding_strategy(
    weight: Weight, i: int, num_devices: int, num_heads: int, head_dim: int
) -> TensorValue:
    """Shards a weight tensor by column according to attention head distribution.

    This strategy is designed for output projection weights in attention layers
    where the number of heads is not evenly divisible by the number of devices.
    It splits columns according to how heads are distributed across devices.

    Supports packed weights, where the stored columns are bytes rather than
    elements: NVFP4/FP4 stores in_dim // 2 columns and MXFP6 stores
    in_dim * 3 // 4. The packing ratio is inferred from the tensor's own width,
    so column indices are converted to packed positions for those formats and
    left alone otherwise.

    Args:
        weight: The :class:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.
        num_heads: Total number of attention heads.
        head_dim: Dimension per attention head.

    Returns:
        A :class:`~max.graph.TensorValue` representing the sharded portion for the i-th device.
    """
    # Compute the head range for this device
    head_start, head_end = _compute_shard_range(num_heads, i, num_devices)

    # Convert head indices to logical column indices
    col_start = head_start * head_dim
    col_end = head_end * head_dim

    # If the weight is packed, its columns are bytes rather than elements, so the
    # logical head boundaries have to be converted to packed columns or the slice
    # runs past the end (and, worse, lands on the wrong data). The ratio is read
    # off the tensor itself, so an unpacked weight -- and the derived scale
    # tensor, whose columns are already blocks -- falls through unchanged.
    logical_in_dim = num_heads * head_dim
    actual_cols = int(weight.shape[1])
    if actual_cols * 2 == logical_in_dim:
        # FP4: two 4-bit codes per byte.
        col_start = col_start // 2
        col_end = col_end // 2
    elif actual_cols * 4 == logical_in_dim * 3:
        # MXFP6: four 6-bit codes per three bytes. `head_dim` is a multiple of 4
        # for every supported head size, so head boundaries stay byte-aligned.
        col_start = col_start * 3 // 4
        col_end = col_end * 3 // 4

    return weight[:, col_start:col_end]


@dataclass(frozen=True)
class Segment:
    """One contiguous segment of a :func:`segmented_sharding_strategy` weight.

    A segmented strategy splits a weight along one axis into several contiguous
    regions ("segments"), shards each region independently across devices, and
    concatenates the per-device pieces back along that axis. Each segment
    declares how it should be split:

    - :meth:`even` splits the segment's ``size`` rows/cols evenly across devices
      via :func:`_compute_shard_range`. Used for gate/up halves of SwiGLU
      stacks, MoE intermediate dims, etc.
    - :meth:`head_aware` splits the segment by attention head; each device
      receives ``_compute_shard_range(num_heads, ...)`` heads of ``head_dim``
      slots. Handles uneven head distributions.

    Construct via the factory methods rather than the raw constructor so the
    invariants (``size == num_heads * head_dim`` for head-aware) hold.
    """

    size: int
    """Logical extent of this segment along the sharded axis."""

    num_heads: int | None
    """Number of attention heads if this is a head-aware segment, else ``None``.

    When set, the segment's ``size`` must equal ``num_heads * head_dim`` and
    the per-device range is computed in head units. When ``None`` the segment
    is split evenly in dim units.
    """

    @staticmethod
    def even(size: int) -> Segment:
        """Returns a segment of ``size`` slots split evenly across devices."""
        return Segment(size=size, num_heads=None)

    @staticmethod
    def head_aware(num_heads: int, head_dim: int) -> Segment:
        """Returns a segment of ``num_heads * head_dim`` slots split by head."""
        return Segment(size=num_heads * head_dim, num_heads=num_heads)

    def shard_range(self, i: int, num_devices: int) -> tuple[int, int]:
        """Returns the (start, end) slot range of this segment on device ``i``."""
        if self.num_heads is None:
            return _compute_shard_range(self.size, i, num_devices)
        head_dim = self.size // self.num_heads
        head_start, head_end = _compute_shard_range(
            self.num_heads, i, num_devices
        )
        return head_start * head_dim, head_end * head_dim


def segmented_sharding_strategy(
    weight: Weight,
    i: int,
    num_devices: int,
    *,
    axis: int,
    segments: Sequence[Segment],
) -> TensorValue:
    """Shards a weight along ``axis`` by per-segment slicing and re-concat.

    Each :class:`Segment` declares a contiguous region along ``axis`` plus how
    that region should be split across devices (evenly or head-aware). The
    region offsets are derived from each segment's ``size``; the segment list
    must therefore cover the entire axis without gaps.

    Generalises :func:`stacked_qkv_sharding_strategy` (3 head-aware segments
    along axis 0) and :func:`gate_up_sharding_strategy` (2 even segments along
    a configured axis), and supports arbitrary mixtures (for example, the
    flux2 fused QKV+MLP projection: three head-aware Q/K/V segments followed
    by two even gate/up segments along axis 0).

    Supports packed weights (NVFP4 columns stored as in_dim // 2): if the
    actual axis size is exactly half the sum of segment sizes, slot indices
    are halved. This mirrors :func:`head_aware_col_sharding_strategy`.

    Args:
        weight: The :class:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.
        axis: Axis along which segments are laid out.
        segments: Ordered segments covering the entire ``axis`` extent.

    Returns:
        A :class:`~max.graph.TensorValue` containing the per-segment slices for
        device ``i`` concatenated back along ``axis``.
    """
    rank = len(weight.shape)
    if axis < 0:
        axis += rank

    logical_total = sum(seg.size for seg in segments)
    actual = int(weight.shape[axis])
    if actual == logical_total:
        packed_factor = 1
    elif actual * 2 == logical_total:
        # NVFP4-style packed columns: indices map to the packed extent.
        packed_factor = 2
    else:
        raise ValueError(
            f"Segmented sharding: segments cover {logical_total} slots along "
            f"axis {axis}, but weight has {actual} (expected {logical_total} "
            f"or {logical_total // 2} for packed FP4)."
        )

    pieces = []
    offset = 0
    for seg in segments:
        seg_start, seg_end = seg.shard_range(i, num_devices)
        slot_start = (offset + seg_start) // packed_factor
        slot_end = (offset + seg_end) // packed_factor
        slices: list[slice] = [slice(None)] * rank
        slices[axis] = slice(slot_start, slot_end)
        pieces.append(weight[tuple(slices)])
        offset += seg.size

    return ops.concat(pieces, axis=axis)


def stacked_qkv_sharding_strategy(
    weight: Weight, i: int, num_devices: int, num_heads: int, head_dim: int
) -> TensorValue:
    """Shards a stacked QKV weight tensor for tensor parallel attention.

    This strategy is designed for weights with shape [3 * hidden_size, hidden_size]
    where the first hidden_size rows are Q, next hidden_size rows are K,
    and last hidden_size rows are V. It shards each section separately by
    attention heads.

    Thin wrapper over :func:`segmented_sharding_strategy` with three
    head-aware segments along axis 0.

    Args:
        weight: The stacked QKV :class:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.
        num_heads: Total number of attention heads.
        head_dim: Dimension per attention head.

    Returns:
        A :class:`~max.graph.TensorValue` representing the sharded QKV for the i-th device.
    """
    total_rows = int(weight.shape[0])
    if total_rows % 3 != 0:
        raise ValueError(
            f"Stacked QKV weight must have shape [3*hidden_size, hidden_size], "
            f"got shape {weight.shape}"
        )
    expected_hidden = num_heads * head_dim
    if total_rows // 3 != expected_hidden:
        raise ValueError(
            f"Weight hidden_size {total_rows // 3} doesn't match "
            f"num_heads * head_dim = {num_heads} * {head_dim} = {expected_hidden}"
        )

    head_aware_seg = Segment.head_aware(num_heads, head_dim)
    return segmented_sharding_strategy(
        weight,
        i,
        num_devices,
        axis=0,
        segments=(head_aware_seg, head_aware_seg, head_aware_seg),
    )


def tensor_parallel_sharding_strategy(
    weight: Weight, i: int, num_devices: int
) -> TensorValue:
    """Shards a :class:`~max.nn.Module` across multiple devices using tensor parallel.

    This strategy is designed for :class:`~max.nn.Module` that has multiple weights,
    for which a single weight sharding strategy is not sufficient.

    Args:
        weight: The :class:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.

    Returns:
        A :class:`~max.graph.TensorValue` representing the sharded portion of the weight
        for the i-th device.
    """
    raise NotImplementedError(
        "tensor_parallel_sharding_strategy is a placeholder and should not be called directly. "
        "Modules are expected to implement their own tensor parallel sharding strategy."
    )


def expert_parallel_sharding_strategy(
    weight: Weight, i: int, num_devices: int
) -> TensorValue:
    """Shards a :class:`~max.nn.Module` across multiple devices using expert parallel.

    This strategy is designed for :class:`~max.nn.Module` that has multiple weights,
    for which a single weight sharding strategy is not sufficient.

    Args:
        weight: The :class:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.

    Returns:
        A :class:`~max.graph.TensorValue` representing the sharded portion of the weight
        for the i-th device.
    """
    raise NotImplementedError(
        "expert_parallel_sharding_strategy is a placeholder and should not be called directly. "
        "Modules are expected to implement their own expert parallel sharding strategy."
    )


def gate_up_sharding_strategy(
    weight: Weight, i: int, num_devices: int, axis: int
) -> TensorValue:
    """Shards a combined gate/up projection weight.

    Thin wrapper over :func:`segmented_sharding_strategy` with two even
    segments along ``axis``.

    Args:
        weight: The :class:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.
        axis: The axis along which the gate and up projections are concatenated.

    Returns:
        A :class:`~max.graph.TensorValue` representing the sharded portion for the i-th device.
    """
    full_dim = int(weight.shape[axis])
    if full_dim % 2 != 0:
        raise ValueError(
            f"Dimension {axis} must be even for gate_up sharding, got {full_dim}"
        )
    half = full_dim // 2
    return segmented_sharding_strategy(
        weight,
        i,
        num_devices,
        axis=axis,
        segments=(Segment.even(half), Segment.even(half)),
    )


def axiswise_sharding_strategy(
    weight: Weight, i: int, num_devices: int, axis: int
) -> TensorValue:
    """Shards a weight evenly along a single axis."""
    rank = len(weight.shape)
    if axis < 0:
        axis += rank
    start, end = _compute_shard_range(
        shard_dim=int(weight.shape[axis]),
        shard_idx=i,
        num_devices=num_devices,
    )
    slices: list[slice] = [slice(None)] * rank
    slices[axis] = slice(start, end)
    return weight[tuple(slices)]


@dataclass(frozen=True)
class ShardingStrategy:
    """Specifies how a :class:`Weight` should be sharded across multiple devices.

    This class encapsulates a sharding function and the number of devices
    over which to shard. It provides static methods for common sharding
    patterns like row-wise, column-wise, and replication.
    """

    num_devices: int
    """The number of devices to shard the weight across."""

    shard: Callable[[Weight, int, int], TensorValue]
    """A callable that takes a :class:`Weight`, a device index, and the total
    number of devices, and returns the sharded :class:`~max.graph.TensorValue` for that
    device.
    """

    def __call__(self, weight: Weight, i: int) -> TensorValue:
        """Applies the sharding strategy to a given weight for a specific device.

        Args:
            weight: The :class:`Weight` to be sharded.
            i: The index of the device for which to get the shard.

        Returns:
            A :class:`~max.graph.TensorValue` representing the portion of the weight for the
            i-th device.
        """
        return self.shard(weight, i, self.num_devices)

    @property
    def is_rowwise(self) -> bool:
        """Whether the sharding strategy is row-wise."""
        return self.shard is row_sharding_strategy

    @property
    def is_colwise(self) -> bool:
        """Whether the sharding strategy is column-wise."""
        return self.shard is col_sharding_strategy

    @property
    def is_head_aware_colwise(self) -> bool:
        """Whether the sharding strategy is head-aware column-wise."""
        # Check if this is a partial function wrapping head_aware_col_sharding_strategy.
        if isinstance(self.shard, partial):
            return self.shard.func is head_aware_col_sharding_strategy
        return self.shard is head_aware_col_sharding_strategy

    @property
    def is_replicate(self) -> bool:
        """Whether the sharding strategy is replicate."""
        return self.shard is replicate_sharding_strategy

    @property
    def is_stacked_qkv(self) -> bool:
        """Whether the sharding strategy is stacked QKV."""
        # Check if this is a partial function wrapping stacked_qkv_sharding_strategy.
        if isinstance(self.shard, partial):
            return self.shard.func is stacked_qkv_sharding_strategy
        return self.shard is stacked_qkv_sharding_strategy

    @property
    def is_tensor_parallel(self) -> bool:
        """Whether the sharding strategy is tensor parallel."""
        return self.shard is tensor_parallel_sharding_strategy

    @property
    def is_expert_parallel(self) -> bool:
        """Whether the sharding strategy is expert parallel."""
        return self.shard is expert_parallel_sharding_strategy

    @property
    def is_gate_up(self) -> bool:
        """Whether the sharding strategy is gate_up."""
        # Check if this is a partial function wrapping gate_up_sharding_strategy.
        if isinstance(self.shard, partial):
            return self.shard.func is gate_up_sharding_strategy
        return self.shard is gate_up_sharding_strategy

    @property
    def is_segmented(self) -> bool:
        """Whether the sharding strategy is segmented."""
        if isinstance(self.shard, partial):
            return self.shard.func is segmented_sharding_strategy
        return self.shard is segmented_sharding_strategy

    @property
    def is_axiswise(self) -> bool:
        """Whether the sharding strategy is axiswise."""
        if isinstance(self.shard, partial):
            return self.shard.func is axiswise_sharding_strategy
        return self.shard is axiswise_sharding_strategy

    @property
    def sharded_axis(self) -> int | None:
        """Axis being sharded, or ``None`` for replicate / placeholder strategies.

        Lets consumers (notably :meth:`max.nn.Linear.shard`) decide whether
        the output dim, the input dim, or neither is being split, without
        enumerating every concrete strategy. Returns ``0`` for ``rowwise`` /
        ``stacked_qkv``, ``1`` for ``columnwise`` / ``head_aware_columnwise``,
        and the configured axis for ``gate_up`` / ``axiswise`` / ``segmented``.
        Replicate and the ``tensor_parallel`` / ``expert_parallel``
        placeholders return ``None``.
        """
        if self.is_rowwise or self.is_stacked_qkv:
            return 0
        if self.is_colwise or self.is_head_aware_colwise:
            return 1
        if self.is_gate_up or self.is_axiswise or self.is_segmented:
            assert isinstance(self.shard, partial)
            return self.shard.keywords["axis"]
        return None

    @staticmethod
    def rowwise(num_devices: int) -> ShardingStrategy:
        """Creates a row-wise sharding strategy.

        This strategy shards the weight along its first axis (axis=0).

        Args:
            num_devices: The number of devices to shard the weight across.

        Returns:
            A :class:`ShardingStrategy` instance configured for row-wise sharding.
        """
        return ShardingStrategy(
            num_devices=num_devices, shard=row_sharding_strategy
        )

    @staticmethod
    def columnwise(num_devices: int) -> ShardingStrategy:
        """Creates a column-wise sharding strategy.

        This strategy shards the weight along its second axis (axis=1).

        Args:
            num_devices: The number of devices to shard the weight across.

        Returns:
            A :class:`ShardingStrategy` instance configured for column-wise sharding.
        """
        return ShardingStrategy(
            num_devices=num_devices, shard=col_sharding_strategy
        )

    @staticmethod
    def axiswise(axis: int, num_devices: int) -> ShardingStrategy:
        """Creates an axis-wise sharding strategy.

        This strategy shards the weight along a given axis.

        Args:
            axis: The axis along which to shard the weight.
            num_devices: The number of devices to shard the weight across.
        """
        return ShardingStrategy(
            num_devices=num_devices,
            shard=partial(axiswise_sharding_strategy, axis=axis),
        )

    @staticmethod
    def replicate(num_devices: int) -> ShardingStrategy:
        """Creates a replication strategy.

        This strategy replicates the entire weight on each device.

        Args:
            num_devices: The number of devices (primarily for consistency, as
                the weight is replicated).

        Returns:
            A :class:`ShardingStrategy` instance configured for replication.
        """
        return ShardingStrategy(
            num_devices=num_devices, shard=replicate_sharding_strategy
        )

    @staticmethod
    def stacked_qkv(
        num_devices: int, num_heads: int, head_dim: int
    ) -> ShardingStrategy:
        """Creates a stacked QKV sharding strategy for tensor parallel attention.

        This strategy is designed for weights with shape [3 * hidden_size, hidden_size]
        where Q, K, and V weights are stacked together. It shards each section
        separately by attention heads, properly handling cases where num_heads
        is not evenly divisible by num_devices.

        Args:
            num_devices: The number of devices to shard the weight across.
            num_heads: Total number of attention heads.
            head_dim: Dimension per attention head.

        Returns:
            A :class:`ShardingStrategy` instance configured for stacked QKV sharding.
        """
        # Use partial to bind num_heads and head_dim to the sharding function
        shard_fn = partial(
            stacked_qkv_sharding_strategy,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        return ShardingStrategy(num_devices=num_devices, shard=shard_fn)

    @staticmethod
    def head_aware_columnwise(
        num_devices: int, num_heads: int, head_dim: int
    ) -> ShardingStrategy:
        """Creates a head-aware column-wise sharding strategy for attention output projection.

        This strategy shards weight columns according to attention head distribution,
        properly handling cases where num_heads is not evenly divisible by num_devices.
        It's designed for output projection weights in attention layers where each
        device processes a different number of heads.

        Args:
            num_devices: The number of devices to shard the weight across.
            num_heads: Total number of attention heads.
            head_dim: Dimension per attention head.

        Returns:
            A :class:`ShardingStrategy` instance configured for head-aware column sharding.
        """
        shard_fn = partial(
            head_aware_col_sharding_strategy,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        return ShardingStrategy(num_devices=num_devices, shard=shard_fn)

    @staticmethod
    def tensor_parallel(num_devices: int) -> ShardingStrategy:
        """Creates a tensor parallel sharding strategy.

        This strategy is designed for :class:`~max.nn.Module` that has multiple weights, for
        which a single weight sharding strategy is not sufficient. This strategy
        is a placeholder and should not be called directly. Modules are expected
        to implement their own tensor parallel sharding strategy.

        Args:
            num_devices: The number of devices to shard the :class:`~max.nn.Module` across.

        Returns:
            A :class:`ShardingStrategy` instance configured for tensor parallel sharding.
        """
        return ShardingStrategy(
            num_devices=num_devices, shard=tensor_parallel_sharding_strategy
        )

    @staticmethod
    def expert_parallel(num_devices: int) -> ShardingStrategy:
        """Creates an expert parallel sharding strategy.

        This strategy is designed for :class:`~max.nn.Module` that has multiple weights,
        for which a single weight sharding strategy is not sufficient. This strategy
        is a placeholder and should not be called directly. Modules are expected
        to implement their own expert parallel sharding strategy.

        Args:
            num_devices: The number of devices to shard the :class:`~max.nn.Module` across.

        Returns:
            A :class:`ShardingStrategy` instance configured for expert parallel sharding.
        """
        return ShardingStrategy(
            num_devices=num_devices, shard=expert_parallel_sharding_strategy
        )

    @staticmethod
    def gate_up(num_devices: int, axis: int = 2) -> ShardingStrategy:
        """Creates a gate_up sharding strategy.

        This strategy shards weights where gate and up projections are concatenated
        along a specific axis.

        Args:
            num_devices: The number of devices to shard the weight across.
            axis: The axis along which the gate and up projections are concatenated.
                Defaults to 2 (common for MoE experts).

        Returns:
            A :class:`ShardingStrategy` instance configured for gate_up sharding.
        """
        shard_fn = partial(gate_up_sharding_strategy, axis=axis)
        return ShardingStrategy(num_devices=num_devices, shard=shard_fn)

    @staticmethod
    def segmented(
        num_devices: int, axis: int, segments: Sequence[Segment]
    ) -> ShardingStrategy:
        """Creates a segmented sharding strategy.

        Splits the weight along ``axis`` into the given ordered segments and
        shards each segment independently before concatenating the per-device
        pieces back. Generalises :meth:`stacked_qkv` (3 head-aware segments
        along axis 0) and :meth:`gate_up` (2 even segments along ``axis``).

        Use this when a fused projection's output (or input) is laid out as a
        sequence of independently-typed blocks — for example a stacked
        ``[Q | K | V | gate | up]`` along axis 0, or a fused output projection
        whose input is ``[attn_out | mlp_out]`` along axis 1.

        Args:
            num_devices: The number of devices to shard the weight across.
            axis: The axis along which segments are laid out.
            segments: Ordered :class:`Segment` instances covering the entire
                ``axis`` extent. Use :meth:`Segment.even` and
                :meth:`Segment.head_aware` to construct.

        Returns:
            A :class:`ShardingStrategy` instance configured for segmented
            sharding.
        """
        shard_fn = partial(
            segmented_sharding_strategy,
            axis=axis,
            segments=tuple(segments),
        )
        return ShardingStrategy(num_devices=num_devices, shard=shard_fn)


@dataclass
class _ShardingStrategyContainer:
    """Container for the sharding strategy and the host weight."""

    host_weight: Weight
    shard_value: ShardingStrategy
    """Defines how to shard a weight onto multiple devices.

    Should be a callable that take the host weight, and the shard index, and
    returns the shard value.
    """


# TODO: Make Weight explicitly inherit from Shardable. We currently do not do this
# because importing Shardable from nn causes a circular dependency.
class Weight(TensorValue):
    """Represents a value in a :class:`~max.graph.Graph` that can be loaded at a later time.

    Weights can be initialized outside of a :class:`~max.graph.Graph` and are lazily-added
    to the parent graph when used. If there is no parent graph when a weight is
    used, an error will be raised.

    Args:
        name: The name of the weight.
        dtype: The data type of the weight.
        shape: The shape of the weight.
        device: The device where the weight resides.
        quantization_encoding: Optional quantization encoding for the weight.
        align: Optional alignment requirement in bytes.
        sharding_strategy: Optional sharding strategy for distributed execution.
    """

    _dtype: DType
    _shape: ShapeLike
    _device: DeviceRef
    quantization_encoding: QuantizationEncoding | None
    align: int | None
    _sharding_strategy: _ShardingStrategyContainer | None
    shard_idx: int | None

    def __new__(cls, *args, **kwargs):
        """Create a new Weight instance.

        Skips the ``Value.__new__`` method to avoid staging a ``TensorValue``.
        A ``Weight`` can be initialized outside of a graph, but must be located
        within a graph when operating on it.
        """
        return super(Value, Weight).__new__(cls)

    def __init__(
        self,
        name: str,
        dtype: DType,
        shape: ShapeLike,
        device: DeviceRef,
        quantization_encoding: QuantizationEncoding | None = None,
        align: int | None = None,
        sharding_strategy: ShardingStrategy | None = None,
        _placeholder: bool = False,
        _has_alias: bool = False,
    ) -> None:
        self.name = name
        self._dtype = dtype
        self._shape = shape
        self._device = device
        self.quantization_encoding = quantization_encoding
        self.align = align
        self._sharding_strategy = (
            _ShardingStrategyContainer(self, sharding_strategy)
            if sharding_strategy
            else None
        )
        self.shard_idx = None
        self._placeholder = _placeholder
        self._has_alias = _has_alias

    @property
    def dtype(self) -> DType:
        """The data type of the weight."""
        return self._dtype

    @property
    def shape(self) -> Shape:
        """The shape of the weight.

        For sharded weights, returns the shape of the shard. Otherwise,
        returns the original weight shape.
        """
        if self.shard_idx is not None:
            # If this is a weight shard, then the weight shape will not
            # match the actual shard shape. The correct shape is the shape of
            # the TensorValue computed from the sharding strategy.
            return super().shape
        return Shape(self._shape)

    @cached_property
    def original_dtype_and_shape(self) -> tuple[DType, Shape]:
        """The original dtype and shape of this weight.

        This property should be used to store the original weight's dtype and
        shape the quantization encoding forces the weight to be loaded as uint8.
        """
        return self.dtype, self.shape

    @property
    def device(self) -> DeviceRef:
        """The device where the weight resides."""
        return self._device

    @cached_property
    def _mlir_value(self) -> _Value[mo.TensorType]:  # type: ignore[override]
        if not self._sharding_strategy or self.shard_idx is None:
            return _add_weight_to_graph(self)._mlir_value
        host_weight = self._sharding_strategy.host_weight
        tensor_value = self._sharding_strategy.shard_value(
            host_weight, self.shard_idx
        )
        tensor_value = tensor_value.to(self._device)
        return tensor_value._mlir_value

    def __repr__(self) -> str:
        device_str = f", {self._device}"
        if self.quantization_encoding:
            return f"Weight({self.name}, {self.dtype}, {self.shape}{device_str}, {self.quantization_encoding})"
        else:
            return (
                f"Weight({self.name}, {self.dtype}, {self.shape}{device_str})"
            )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the weight sharding strategy."""
        return (
            self._sharding_strategy.shard_value
            if self._sharding_strategy
            else None
        )

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the weight sharding strategy.

        Args:
            strategy: A ShardingStrategy that defines how to shard the weight.
        """
        # Reset device to CPU to avoid implicit transfer behavior
        # from `force_initial_weight_on_host`.
        # TODO(GEX-2121): Remove this hack
        self._device = DeviceRef.CPU()
        self._sharding_strategy = _ShardingStrategyContainer(self, strategy)

    def shard(self, devices: Iterable[DeviceRef]) -> list[Weight]:
        """Creates sharded views of this Weight across multiple devices.

        This :class:`Weight` must have ``sharding_strategy`` defined. The shard objects
        returned are also :class:`Weight` objects, but cannot be sharded further.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded weights, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                f"Weight {self.name} cannot be sharded because no sharding strategy was provided."
            )
        if self.shard_idx is not None:
            raise ValueError(
                f"Weight {self.name} was already sharded. Use __getitem__ instead."
            )

        shards = []
        for shard_idx, device in enumerate(devices):
            weight = Weight(
                name=f"{self.name}[{shard_idx}]",
                dtype=self._dtype,
                shape=self._shape,
                device=device,
                quantization_encoding=self.quantization_encoding,
                align=self.align,
            )

            # Copy the sharding strategy container directly to preserve the original host_weight
            # reference. Using the property setter would create a new container with this shard
            # as the host_weight, causing infinite recursion when the shard tries to compute
            # its shape by calling the sharding function on itself.
            weight._sharding_strategy = self._sharding_strategy
            weight.shard_idx = shard_idx
            shards.append(weight)

        return shards


def _add_weight_to_graph(weight: Weight):  # noqa: ANN202
    try:
        current_graph = graph.Graph.current
    except LookupError:
        raise ValueError(  # noqa: B904
            "Cannot operate on a `max.graph.Weight` when there is no parent graph."
        )

    # If the weight doesn't exist on the graph, `Graph.add_weight` will
    # return a new `TensorValue`. Otherwise, this will return the existing
    # `TensorValue`.
    return current_graph.add_weight(
        weight,
        # Don't force the weight to be on host if it has an alias.
        # We expect these external constants to already be resident on the
        # device.
        force_initial_weight_on_host=not weight._has_alias,
    )
