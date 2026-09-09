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

"""End-to-end tests for the device graph *synthesis* pathway.

A graph marked ``is_device_graph`` is lowered to ``mgp.device_graph.create`` /
``execute``: the first execution records the work into a device graph, and later
executions with a matching cache key replay it.

Distinct from ``test_engine_device_graph.py``, which covers the separate record
and replay API (``model.capture`` / ``model.replay``) over the same driver-level
device graphs.

These tests exist to cover what neither the lit tests nor the Mojo unit tests
can express: whether a replayed graph reads the input passed to *this* execution.
``mt`` allocates its inputs once, and the Mojo cache tests record ``memset`` with
a literal, so neither notices a graph replaying a stale address.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def test_replay_reads_the_current_input() -> None:
    """A cache hit must read the buffer passed to *this* execution.

    A recorded device graph bakes in the addresses it was built with, so
    replaying it is only correct if the live input is first copied into the
    stable location the graph owns. Both executions use the same shape, so they
    share a cache key and the second one replays the graph the first one built.

    The two inputs are allocated up front and both held alive for the duration,
    so the allocator cannot hand out the same address twice: the second
    execution genuinely reads from a different pointer than the one recorded.
    Without the input copy the replay would read the first buffer and return
    2.0 instead of 14.0.
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    input_type = TensorType(DType.float32, [4], device=DeviceRef.GPU(0))

    with Graph(
        "device_graph_input_binding",
        input_types=[input_type],
        is_device_graph=True,
    ) as graph:
        # The output has to depend on the input's *contents*, or reading the
        # wrong buffer would still produce the right answer.
        x = graph.inputs[0].tensor
        graph.output(x + x)

    model = session.load(graph)

    first = Buffer.from_numpy(np.full([4], 1.0, dtype=np.float32)).to(
        model.input_devices[0]
    )
    second = Buffer.from_numpy(np.full([4], 7.0, dtype=np.float32)).to(
        model.input_devices[0]
    )

    # Read the first result before executing again: a replayed graph writes its
    # recorded output buffer, so the two executions may hand back the same
    # storage.
    (first_output,) = model.execute(first)
    np.testing.assert_allclose(
        first_output.to(CPU()).to_numpy(), np.full([4], 2.0, dtype=np.float32)
    )

    (second_output,) = model.execute(second)
    np.testing.assert_allclose(
        second_output.to(CPU()).to_numpy(), np.full([4], 14.0, dtype=np.float32)
    )


def test_replay_is_stable_across_repeats() -> None:
    """Repeated executions with one buffer keep producing the same answer.

    Guards the copy itself: copying into the stable location must overwrite it
    rather than accumulate, and must not corrupt the recorded graph.
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    input_type = TensorType(DType.float32, [8], device=DeviceRef.GPU(0))

    with Graph(
        "device_graph_repeat",
        input_types=[input_type],
        is_device_graph=True,
    ) as graph:
        x = graph.inputs[0].tensor
        graph.output(x + x)

    model = session.load(graph)

    all_device_inputs = []
    for i in range(128):
        values = np.arange(8, dtype=np.float32) * float(i)
        device_input = Buffer.from_numpy(values).to(model.input_devices[0])
        all_device_inputs.append(device_input)

        (output,) = model.execute(device_input)
        np.testing.assert_allclose(output.to(CPU()).to_numpy(), values * 2)


def test_mutable_buffer_input_is_read_and_written_in_place() -> None:
    """A mutable buffer input is recorded at its live address, never copied.

    A ``BufferType`` input is the KV-cache shape: the graph reads it and writes
    it in place, so it must NOT be given a graph-private stable location — the
    in-place writes would land in the private copy and be discarded. Instead
    the buffer's address is part of the graph cache key, so a same-shape buffer
    at a different address misses and rebuilds rather than replaying the first
    buffer's recorded address.

    Three probes, each of which fails under a different regression:

    - The in-place write must be visible in the caller's buffer after
      execution (fails if a copy to a stable twin sneaks back in).
    - A same-shape buffer at a different address must produce a correct result
      for *its* contents (fails if the address is missing from the cache key:
      the hit would replay against the first buffer's address).
    - Re-executing with the first buffer must reflect its *current* contents
      (fails if replay snapshots contents rather than reading the live
      address).
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    buffer_type = BufferType(DType.float32, [4], device=DeviceRef.GPU(0))

    with Graph(
        "device_graph_mut_buffer",
        input_types=[buffer_type],
        is_device_graph=True,
    ) as graph:
        buf = graph.inputs[0].buffer
        # The output must depend on the buffer's contents, and the buffer must
        # be written in place, or the probes below cannot distinguish reading
        # or writing the wrong memory from correct behavior.
        doubled = ops.buffer_load(buf) + ops.buffer_load(buf)
        ops.buffer_store(buf, doubled)
        graph.output(doubled)

    model = session.load(graph)

    # Both buffers are allocated up front and held alive, so the second cannot
    # reuse the first's address.
    first = Buffer.from_numpy(np.full([4], 1.0, dtype=np.float32)).to(
        model.input_devices[0]
    )
    second = Buffer.from_numpy(np.full([4], 7.0, dtype=np.float32)).to(
        model.input_devices[0]
    )

    (first_output,) = model.execute(first)
    np.testing.assert_allclose(
        first_output.to(CPU()).to_numpy(), np.full([4], 2.0, dtype=np.float32)
    )
    # The in-place write reached the caller's buffer.
    np.testing.assert_allclose(
        first.to(CPU()).to_numpy(), np.full([4], 2.0, dtype=np.float32)
    )

    # Different address, same shape: with the address in the key this is a
    # miss + rebuild against `second`, and `first` is left untouched.
    (second_output,) = model.execute(second)
    np.testing.assert_allclose(
        second_output.to(CPU()).to_numpy(), np.full([4], 14.0, dtype=np.float32)
    )
    np.testing.assert_allclose(
        second.to(CPU()).to_numpy(), np.full([4], 14.0, dtype=np.float32)
    )
    np.testing.assert_allclose(
        first.to(CPU()).to_numpy(), np.full([4], 2.0, dtype=np.float32)
    )

    # Back to the first buffer: its graph is cached under its address, and the
    # replay must read the buffer's *current* contents (2.0 from the first
    # execution), not a snapshot of what it held at build time.
    (third_output,) = model.execute(first)
    np.testing.assert_allclose(
        third_output.to(CPU()).to_numpy(), np.full([4], 4.0, dtype=np.float32)
    )
    np.testing.assert_allclose(
        first.to(CPU()).to_numpy(), np.full([4], 4.0, dtype=np.float32)
    )


def test_host_input_contents_key_the_graph() -> None:
    """A CPU input's contents key the graph: changed bytes re-record.

    ``fill_from_host_scalar`` reads the host scalar on the host at enqueue
    time (the dispatch-metadata pattern), so the value is frozen into the
    recorded kernel. A replay is only sound for the exact bytes that were
    recorded: executing with a different value must miss the cache and
    re-record, and returning to a previous value must reuse its graph.
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    host_type = TensorType(DType.float32, [1], device=DeviceRef.CPU())
    out_type = TensorType(DType.float32, [4], device=DeviceRef.GPU(0))

    with Graph(
        "device_graph_host_input",
        input_types=[host_type],
        custom_extensions=[
            Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])
        ],
        is_device_graph=True,
    ) as graph:
        h = graph.inputs[0].tensor
        result = ops.custom(
            "fill_from_host_scalar",
            device=DeviceRef.GPU(0),
            values=[h],
            out_types=[out_type],
        )[0]
        graph.output(result)

    model = session.load(graph)

    three = Buffer.from_numpy(np.array([3.0], dtype=np.float32))
    nine = Buffer.from_numpy(np.array([9.0], dtype=np.float32))

    (out,) = model.execute(three)
    np.testing.assert_allclose(
        out.to(CPU()).to_numpy(), np.full([4], 3.0, dtype=np.float32)
    )

    # Changed bytes: a stale replay would still fill 3.0.
    (out,) = model.execute(nine)
    np.testing.assert_allclose(
        out.to(CPU()).to_numpy(), np.full([4], 9.0, dtype=np.float32)
    )

    # Back to the first value: hits the original cache entry.
    (out,) = model.execute(three)
    np.testing.assert_allclose(
        out.to(CPU()).to_numpy(), np.full([4], 3.0, dtype=np.float32)
    )

    # The same value at a different address is the same key.
    three_again = Buffer.from_numpy(np.array([3.0], dtype=np.float32))
    (out,) = model.execute(three_again)
    np.testing.assert_allclose(
        out.to(CPU()).to_numpy(), np.full([4], 3.0, dtype=np.float32)
    )
