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
"""Op implementation for distributed scatter.

Distributes different data chunks from a root GPU to multiple device groups.
Each group (DP replica) gets a different chunk, and all devices within a group
(TP devices) get the same chunk via P2P pull.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from max._core.dialects import mo
from max._core.dialects.builtin import IntegerAttr, IntegerType

from ..graph import Graph
from ..type import TensorType, _ChainType
from ..value import BufferValueLike, TensorValue, TensorValueLike
from .utils import _buffer_values, _tensor_values


def distributed_scatter(
    input_chunks: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
) -> list[TensorValue]:
    """Scatters different chunks from a root GPU to device groups.

    Each data-parallel replica group receives a different input chunk. All
    tensor-parallel devices within the same replica get the same chunk. Uses a
    pull-based approach where each GPU reads its chunk from the root GPU over
    peer-to-peer transfers.

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops
        from max.nn import Signals

        # Data-parallel size 2, tensor-parallel size 2, across 4 GPUs
        devices = [DeviceRef.GPU(id=i) for i in range(4)]
        signals = Signals(devices)

        with Graph(
            "distributed_scatter",
            input_types=[
                # One input chunk per data-parallel replica on the root GPU
                TensorType(dtype=DType.uint32, shape=[5], device=devices[0]),
                TensorType(dtype=DType.uint32, shape=[5], device=devices[0]),
                *signals.input_types(),
            ],
        ) as graph:
            input_chunks = [graph.inputs[0].tensor, graph.inputs[1].tensor]
            signal_buffers = [inp.buffer for inp in graph.inputs[2:]]
            # Returns one output per GPU.
            outputs = ops.distributed_scatter(input_chunks, signal_buffers)
            graph.output(*outputs)

    Args:
        input_chunks: The input tensors to scatter, one per data-parallel
            replica. All must reside on the same root device. The number of
            chunks determines ``dp_size``.
        signal_buffers: The device buffer values used for synchronization. The
            number of signal buffers determines the number of participating
            GPUs (``ngpus``).

    Returns:
        A list of output tensors, one per device. Each output tensor has the
        same shape and dtype as its replica's input chunk.

    Raises:
        ValueError: If any input is invalid. This includes when there are no
            input chunks, the input chunks aren't on the same device, the
            signal buffer devices aren't unique, or the root device isn't among
            the signal buffer devices.
    """
    input_chunks = _tensor_values(input_chunks)
    signal_buffers = _buffer_values(signal_buffers)
    dp_size = len(input_chunks)
    ngpus = len(signal_buffers)

    if dp_size < 1:
        raise ValueError(
            "distributed_scatter requires at least 1 input chunk. "
            f"Got: {dp_size}"
        )

    # All input chunks must be on the same root device.
    root_device = input_chunks[0].device
    for i, chunk in enumerate(input_chunks):
        if chunk.device != root_device:
            raise ValueError(
                f"All input chunks must be on the same device. "
                f"Chunk 0 is on {root_device}, but chunk {i} is on "
                f"{chunk.device}"
            )

    devices = [buf.device for buf in signal_buffers]
    if len(set(devices)) < len(devices):
        raise ValueError(
            "distributed_scatter requires unique devices across signal "
            f"buffers. Got: {devices=}"
        )

    # Infer root from where the input chunks live.
    if root_device not in devices:
        raise ValueError(
            f"input chunk device {root_device} not found in signal buffer "
            f"devices: {devices}"
        )
    root = devices.index(root_device)

    tp_size = math.ceil(ngpus / dp_size)

    # Build ngpus-sized padded chunk list so every GPU sees all chunk sizes
    # and computes the same grid dimensions (avoiding barrier deadlocks).
    # padded_chunks[i] is the chunk that GPU i should read.
    padded_chunks = [
        input_chunks[min(i // tp_size, dp_size - 1)] for i in range(ngpus)
    ]

    # Build output types: each GPU gets its replica's chunk shape.
    out_types = [
        TensorType(
            dtype=padded_chunks[i].dtype,
            shape=padded_chunks[i].shape,
            device=device,
        )
        for i, device in enumerate(devices)
    ]

    graph = Graph.current

    # Merge all device chains into one input chain.
    in_chain = graph.device_chains.merge_for(devices)

    # Stage a single scatter op across all devices.
    root_attr = IntegerAttr(IntegerType(64), root)
    *results, out_chain = graph._add_op_generated(
        mo.DistributedScatterOp,
        out_types,
        _ChainType(),
        padded_chunks,
        signal_buffers,
        in_chain,
        root_attr,
    )

    # Update all chains.
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return [res.tensor for res in results]
