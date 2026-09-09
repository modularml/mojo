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
"""Shared device-adaptive checks for ``ops.buffer_create`` with an init value.

A buffer created with an ``init_value`` becomes persistent state: it is
allocated and initialized once when the model is loaded, and the same buffer
(with its mutations preserved) is reused across executions, rather than being
re-created per call.

The checks run on the accelerator when one is present and on the CPU otherwise,
so the ``test_buffer_create_init`` (CPU) and ``test_buffer_create_init_gpu``
(GPU) targets both exercise them -- covering the host and device paths of the
memset primitive that performs the one-time initialization.

Each check increments the buffer by a value supplied as a graph input (always 1)
rather than a compile-time constant. That keeps the read-modify-write dependent
on an execute-time argument so it runs on every call, matching how a real model
uses such a buffer (its consumer depends on execution inputs).
"""

import max.driver as md
import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.graph.ops import buffer_load, buffer_store


def make_session() -> InferenceSession:
    """Creates a session over a single accelerator (when present) plus the CPU.

    The checks only ever use the first accelerator (see ``_device``) and read
    results back on the CPU, so the session needs just those two devices.
    Spanning *every* accelerator made the first ``session.load`` do per-device
    setup for GPUs the graph never touches, so its cost scaled with the GPU
    count (~10s on 2 GPUs, worse on an 8-GPU CI host). Under load that pushed
    the test past its timeout -- the flake that disabled it (GEX-3994/QUA-786).
    """
    devices: list[md.Device] = []
    if md.accelerator_count() > 0:
        devices.append(md.Accelerator(0))
    devices.append(md.CPU())
    return InferenceSession(devices=devices)


def _device() -> DeviceRef:
    """Runs on the accelerator when one is present, otherwise the CPU."""
    return DeviceRef.GPU() if md.accelerator_count() > 0 else DeviceRef.CPU()


def run_persists_across_calls(session: InferenceSession) -> None:
    """A buffer created with an init value is initialized once, not per call.

    The buffer is created with ``init_value=0`` and each execution adds the
    graph input (1) to it in place. If the buffer were re-created and re-zeroed
    on every call, every execution would return 1. Because it is initialized
    once when the model is loaded and the same buffer is reused across
    executions, the mutation persists and the k-th call returns k.
    """
    device = _device()
    increment_type = TensorType(DType.int32, [1], device=device)
    with Graph(
        "buffer_create_init_persist", input_types=[increment_type]
    ) as graph:
        buffer = ops.buffer_create(
            BufferType(DType.int32, [1], device=device),
            init_value=0,
        )
        updated = buffer_load(buffer) + graph.inputs[0].tensor
        buffer_store(buffer, updated)
        graph.output(updated)
        graph._mlir_op.verify()
        compiled = session.load(graph)

    one = md.Buffer.from_numpy(np.array([1], dtype=np.int32)).to(
        compiled.input_devices[0]
    )
    for expected in range(1, 6):
        output = compiled.execute(one)[0]
        assert output.to(md.CPU()).to_numpy().item() == expected


def run_nonzero_init_value(session: InferenceSession) -> None:
    """A non-zero, non-integer init value is applied exactly once and persists.

    Starting from ``init_value=2.5`` and adding the graph input (1.0) per call,
    the first call returns 3.5 (proving the init value was applied, not 0), and
    subsequent calls accumulate (4.5, 5.5), proving the buffer persists across
    executions.
    """
    device = _device()
    increment_type = TensorType(DType.float32, [4], device=device)
    with Graph(
        "buffer_create_init_value", input_types=[increment_type]
    ) as graph:
        buffer = ops.buffer_create(
            BufferType(DType.float32, [4], device=device),
            init_value=2.5,
        )
        updated = buffer_load(buffer) + graph.inputs[0].tensor
        buffer_store(buffer, updated)
        graph.output(updated)
        graph._mlir_op.verify()
        compiled = session.load(graph)

    ones = md.Buffer.from_numpy(np.ones([4], dtype=np.float32)).to(
        compiled.input_devices[0]
    )
    for i in range(3):
        output = compiled.execute(ones)[0]
        assert np.allclose(output.to(md.CPU()).to_numpy(), 2.5 + (i + 1))
