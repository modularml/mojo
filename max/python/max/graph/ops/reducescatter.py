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
"""Op implementation for reducescatter."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max._core.dialects.builtin import IntegerAttr, IntegerType

from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValueLike, TensorType, TensorValue, TensorValueLike
from .utils import _buffer_values, _tensor_values


def sum(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    axis: int = 0,
    group_size: int | None = None,
) -> list[TensorValue]:
    """Collective reduce-scatter summation operation.

    This op is a collective op which takes in tensors from different devices
    and outputs tensors on different devices. Each device reduces (via
    summation) and stores a disjoint partition of the inputs from all devices.

    Args:
        inputs: The input tensors to reduce and scatter.
        signal_buffers: Device buffer values used for synchronization.
        axis: The axis along which to scatter the reduced result. Defaults to 0.
        group_size: Optional number of contiguous devices per independent
            reduce-scatter group. Defaults to all devices.

    Returns:
        An iterable of outputs where each device receives its portion of the
        scattered result. The output shape on each device is the input shape
        with dimension `axis` divided by the group size.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    devices = [input.device for input in inputs]
    num_devices = len(devices)
    if num_devices == 0:
        raise ValueError(
            "reducescatter.sum operation requires at least one input"
        )
    group_size = group_size or num_devices

    if group_size < 1:
        raise ValueError(
            "reducescatter.sum operation requires group_size to be at least 1. "
            f"Got: {group_size=}"
        )
    if num_devices % group_size != 0:
        raise ValueError(
            "reducescatter.sum operation requires group_size to evenly divide "
            f"the number of input tensors. Got: {group_size=} and {num_devices=}"
        )
    if not all(input.dtype == inputs[0].dtype for input in inputs[1:]):
        raise ValueError(
            "reducescatter.sum operation must have the same dtype across all "
            f"input tensors. Got: {inputs=}"
        )
    if not all(
        input.shape.rank == inputs[0].shape.rank for input in inputs[1:]
    ):
        raise ValueError(
            "reducescatter.sum operation must have the same rank across all "
            f"input tensors. Got: {inputs=}"
        )
    for group_start in range(0, num_devices, group_size):
        group_inputs = inputs[group_start : group_start + group_size]
        if not all(
            input.shape == group_inputs[0].shape for input in group_inputs[1:]
        ):
            raise ValueError(
                "reducescatter.sum operation must have the same shape across "
                f"all input tensors in each group. Got: {inputs=}"
            )
    if len(set(devices)) < len(devices):
        raise ValueError(
            "reducescatter.sum operation must have unique devices across its "
            f"input tensors. Got: {devices=}"
        )

    # Resolve negative axis before passing to MLIR.
    input_dtype = inputs[0].dtype
    if axis < 0:
        axis = inputs[0].shape.rank + axis
    if axis < 0 or axis >= inputs[0].shape.rank:
        raise ValueError(
            f"axis {axis} is out of bounds for tensor with rank {inputs[0].shape.rank}"
        )

    graph = Graph.current

    # Compute output types for each device's portion.
    output_types = []
    for dev_idx, device in enumerate(devices):
        input_shape = inputs[dev_idx].shape
        scatter_dim = input_shape[axis]
        output_shape_list = list(input_shape)
        local_rank = dev_idx % group_size
        # Ragged binning across this device's group.
        output_shape_list[axis] = (
            scatter_dim + (group_size - local_rank - 1)
        ) // group_size
        output_types.append(
            TensorType(
                dtype=input_dtype,
                shape=output_shape_list,
                device=device,
            )
        )

    # Merge all device chains into one input chain.
    in_chain = graph.device_chains.merge_for(devices)

    # Stage a single reducescatter op across all devices.
    axis_attr = IntegerAttr(IntegerType(64), axis)
    group_size_attr = IntegerAttr(IntegerType(64), group_size)
    *results, out_chain = graph._add_op_generated(
        mo.DistributedReducescatterSumOp,
        output_types,
        _ChainType(),
        inputs,
        signal_buffers,
        in_chain,
        axis_attr,
        group_size_attr,
    )

    # Update all chains.
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return [res.tensor for res in results]
