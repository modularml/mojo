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
"""Op implementation for allgather."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TypeVar

from max._core.dialects import mo
from max._core.dialects.builtin import IntegerAttr, IntegerType

from ..graph import Graph
from ..type import TensorType, _ChainType
from ..value import BufferValueLike, TensorValue, TensorValueLike
from .concat import concat
from .utils import _buffer_values, _tensor_values

T = TypeVar("T")


# Added to itertools in 3.12
def _batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = tuple(itertools.islice(it, n))
        if not batch:
            return
        yield batch


def allgather(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    axis: int = 0,
    group_size: int | None = None,
) -> list[TensorValue]:
    """Collective allgather operation.

    This op is a collective op which takes in tensors from different devices and
    outputs tensors on different devices.
    In particular, this operation will gather the inputs across different
    devices and concatenates them along the specified dimension.
    The result is then broadcasted back to the same devices that the inputs
    came from.

    Args:
        inputs: The input tensors to gather.
        signal_buffers: Device buffer values used for synchronization.
        axis: Dimension to concatenate the input tensors. Defaults to 0.
        group_size: Optional number of contiguous devices per independent
            allgather group. Defaults to all devices.

    Returns:
        An iterable outputs which all hold the gathered output. Each output
        tensor contains the concatenation of the inputs in its group along the
        specified dimension.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)

    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    if not inputs:
        return inputs

    shape = inputs[0].shape
    dtype = inputs[0].dtype
    num_devices = len(inputs)
    group_size = group_size or num_devices
    if group_size < 1:
        raise ValueError(
            "allgather operation requires group_size to be at least 1. "
            f"Got: {group_size=}"
        )
    if num_devices % group_size != 0:
        raise ValueError(
            "allgather operation requires group_size to evenly divide the "
            f"number of input tensors. Got: {group_size=} and {num_devices=}"
        )

    if group_size == 1:
        return inputs

    # Check that all inputs have the same rank and are compatible for concatenation
    if not all(input.shape.rank == shape.rank for input in inputs[1:]):
        raise ValueError(
            "allgather operation must have the same rank across all"
            f" input tensors. Got: {inputs=}"
        )
    if not all(input.dtype == dtype for input in inputs[1:]):
        raise ValueError(
            "allgather operation must have the same dtype across all"
            f" input tensors. Got: {inputs=}"
        )

    devices = [input.device for input in inputs]
    if len(set(devices)) < len(devices):
        raise ValueError(
            "allgather operation must have unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    if not -shape.rank <= axis < shape.rank:
        raise IndexError(f"Dimension out of range {shape.rank}, {axis=}")
    if axis < 0:
        axis += shape.rank

    # Check that all dimensions except the concatenation dimension are the same
    # within each group. Different groups may gather different local shapes.
    for group_start in range(0, num_devices, group_size):
        group_inputs = inputs[group_start : group_start + group_size]
        for i, dim in enumerate(group_inputs[0].shape):
            if i == axis:
                continue
            if not all(input.shape[i] == dim for input in group_inputs):
                raise ValueError(
                    "allgather operation inputs must have the same shape in "
                    "all dimensions except the concatenation dimension within "
                    f"each group. {inputs=}"
                )

    # Prepare raw output types - one gathered input per group member per device.
    output_types: list[TensorType] = []
    for device_idx, device in enumerate(devices):
        group_start = (device_idx // group_size) * group_size
        group_end = group_start + group_size
        group_inputs = inputs[group_start:group_end]
        output_types.extend(
            TensorType(dtype, input.shape, device) for input in group_inputs
        )

    # Get the current chain for synchronization.
    graph = Graph.current
    in_chain = graph.device_chains.merge_for(devices)

    # Stage the allgather op with signal buffers and chain.
    group_size_attr = IntegerAttr(IntegerType(64), group_size)
    *results, out_chain = graph._add_op_generated(
        mo.DistributedAllgatherOp,
        # Output types: tensors + chain
        output_types,
        _ChainType(),
        inputs,
        signal_buffers,
        in_chain,
        group_size_attr,
    )

    # Update the chain.
    graph._update_chain(out_chain)

    # Update device chains.
    for device in devices:
        graph.device_chains[device] = out_chain

    # Convert results to TensorValues.
    outputs = [res.tensor for res in results]
    return [concat(batch, axis=axis) for batch in _batched(outputs, group_size)]
