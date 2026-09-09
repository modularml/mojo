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
"""The global launch trace captures a compiled graph's internal work.

A compiled graph's kernels are enqueued by the engine, not by the caller: the
caller never receives the stream they run on and so cannot register it with the
per-stream trace API. ``max.driver.begin_launch_trace`` needs no stream handle,
so a single begin/take records both the driver-issued input copy and the
graph's internal kernel launches into one enqueue-ordered list. This pins that
graph-internal device work is captured at all, and that it is ordered after the
driver copy that produced the graph's input -- the mechanism-level complement to
the DRIV-311 black-box sync-contract tests.

The trace spans every stream, so it does not assume the copy and the graph share
one (in a single-device ``InferenceSession`` they do); the assertions below
depend only on capture and enqueue order, not on how many streams are involved.
"""

from __future__ import annotations

import numpy as np
import pytest
from max import driver
from max.driver import Accelerator, Buffer, LaunchTraceEntry, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def _build_add_one_graph(device_ref: DeviceRef, shape: list[int]) -> Graph:
    """Returns a graph computing ``input + 1`` on ``device_ref``."""
    with Graph(
        "add_one",
        input_types=[TensorType(DType.int32, shape, device=device_ref)],
    ) as graph:
        x = graph.inputs[0].tensor
        one = ops.constant(
            np.ones(shape, dtype=np.int32), dtype=DType.int32, device=device_ref
        )
        graph.output(x + one)
    return graph


@pytest.fixture
def accelerator() -> Accelerator:
    if accelerator_count() == 0:
        pytest.skip("requires a GPU")
    device = Accelerator()
    if device.api not in ("cuda", "hip"):
        pytest.skip("launch tracing records entries only on CUDA/HIP")
    return device


def test_global_trace_spans_driver_copy_and_graph_kernels(
    accelerator: Accelerator,
) -> None:
    """One trace captures the driver copy and the graph's kernels, in order."""
    shape = [4]
    written = np.arange(1, 5, dtype=np.int32)
    device_ref = DeviceRef.from_device(accelerator)

    session = InferenceSession(devices=[accelerator])
    model = session.load(_build_add_one_graph(device_ref, shape))

    graph_input = Buffer.from_numpy(np.zeros(shape, dtype=np.int32)).to(
        accelerator
    )
    pending = Buffer.from_numpy(written)

    with driver.launch_trace() as entries:
        # Driver-issued input copy, before the graph runs.
        graph_input.inplace_copy_from(pending)
        # The engine enqueues the graph's kernels; the caller holds no handle
        # to the stream they run on.
        outputs = model.execute(graph_input)

    result = outputs[0].to_numpy()
    np.testing.assert_array_equal(result, written + 1)

    copy_idx = next(
        (
            i
            for i, e in enumerate(entries)
            if e.kind == LaunchTraceEntry.OperationKind.MEMCPY
            and e.memcpy_kind == LaunchTraceEntry.MemcpyKind.HTOD
            and e.memcpy_byte_size == written.nbytes
        ),
        None,
    )
    assert copy_idx is not None, f"driver HtoD copy not in trace: {entries}"

    # The graph's internal kernel launches -- work the caller never enqueued --
    # are captured, and ordered after the driver copy that produced their input.
    kernels_after_copy = [
        e
        for e in entries[copy_idx + 1 :]
        if e.kind == LaunchTraceEntry.OperationKind.KERNEL_LAUNCH
    ]
    assert kernels_after_copy, (
        "no graph kernel launch captured after the driver copy; graph-internal"
        f" work was not traced: {entries}"
    )
