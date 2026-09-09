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
"""Graph execution observes pending driver work (DRIV-311 contract 3).

The pending copy and the graph run do not share a stream by construction:
``inplace_copy_from`` enqueues on the driver ``DeviceContext``'s stream, while
the graph executes through MGP's own ``DeviceContext``. Same-stream ordering
therefore cannot be assumed -- GraphRT force-syncs the device before execution
for exactly this reason ("until MGP shares the same DeviceContext as the
driver", ``GraphCompiler/lib/Driver/Device/Driver.cpp``
``mapDeviceRefsToDevices``). This test pins the observable contract, not that
mechanism: an async host-to-device copy enqueued on the driver before
``execute`` is observed by the run.

The graph input is written behind a host-signalled ``CompletionFlag`` gate on
the device stream, and the run must compute on the written value, not the
pre-copy zeros -- asserted on every backend. On CUDA the pre-execution
force-sync host-blocks, so ``execute`` cannot return while the gate is held;
AMD/HIP reaches the same result through stream ordering without host-blocking,
so the block itself is asserted only on CUDA.

The gate is released in a ``finally`` and the worker thread is joined with a
timeout so a regression fails the assertion instead of hanging CI.
"""

import threading

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, CompletionFlag, accelerator_count
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
        pytest.skip("stream host-value gating requires CUDA/HIP")
    return device


def test_execute_observes_pending_driver_copy(
    accelerator: Accelerator,
) -> None:
    """``execute`` observes a host-to-device copy enqueued before it runs."""
    shape = [4]
    written = np.arange(1, 5, dtype=np.int32)
    device_ref = DeviceRef.from_device(accelerator)

    session = InferenceSession(devices=[accelerator])
    model = session.load(_build_add_one_graph(device_ref, shape))

    graph_input = Buffer.from_numpy(np.zeros(shape, dtype=np.int32)).to(
        accelerator
    )
    pending = Buffer.from_numpy(written)

    flag = CompletionFlag(accelerator)
    gate_stream = accelerator.default_queue
    done = threading.Event()
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            box["outputs"] = model.execute(graph_input)
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    gate_stream.wait_for_host_value(flag, 1)
    completed_while_gated = True
    try:
        graph_input.inplace_copy_from(pending)
        worker_thread.start()
        completed_while_gated = done.wait(timeout=2.0)
    finally:
        flag.signal(1)
        if worker_thread.ident is not None:
            worker_thread.join(timeout=60.0)

    assert not worker_thread.is_alive(), "execute() thread did not finish"
    assert "error" not in box, f"execute() raised: {box.get('error')}"

    outputs = box["outputs"]
    assert isinstance(outputs, list)
    result = outputs[0].to_numpy()
    # The portable contract: the run observes the pending driver copy and
    # computes on the written value, not the pre-copy zeros.
    np.testing.assert_array_equal(result, written + 1)

    # On CUDA the run additionally host-blocks on GraphRT's pre-execution
    # force-sync, so execute() cannot return while the copy is gated. AMD/HIP
    # reaches the same result via stream ordering without host-blocking.
    if accelerator.api == "cuda":
        assert not completed_while_gated, (
            "execute() returned while the input copy was gated; the CUDA"
            " pre-execution force-sync did not run"
        )
