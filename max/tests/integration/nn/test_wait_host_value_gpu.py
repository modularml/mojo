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
"""Integration test for the ``mo.wait_host_value`` custom op.

Builds a Graph that contains both ``mo.launch_host_func`` and
``mo.wait_host_value``. The host callback (running on a CUDA driver
thread, after the launch_host_func node fires) sleeps briefly to ensure
the wait_value node has been issued on the stream, then atomic-stores 1
to the flag. The wait_value node observes the store and lets execution
continue.

The test asserts that:

  1. The graph executes to completion (no deadlock).
  2. The callback ran.
  3. The flag was set to 1 by the time ``execute()`` returns.
"""

import threading
import time

import pytest
from max.driver import (
    Accelerator,
    Buffer,
    CompletionFlag,
    __unsafe_pack_py_host_func,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph
from max.nn import kernels


def _build_launch_then_wait_graph(device_ref: DeviceRef) -> Graph:
    """Returns a graph: launch_host_func -> wait_host_value.

    Inputs:
      - kickoff_payload: int64[2] holding (trampoline_ptr, user_data_ptr)
        for the host callback (max._core.driver.__unsafe_pack_py_host_func).
      - wait_payload: int64[2] holding (device_ptr, expected_value=1).
    """
    with Graph(
        "launch_then_wait",
        input_types=[
            BufferType(DType.int64, [2], device=DeviceRef.CPU()),
            BufferType(DType.int64, [2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        kickoff_payload = graph.inputs[0].buffer
        wait_payload = graph.inputs[1].buffer
        kernels.launch_host_func(payload=kickoff_payload, device=device_ref)
        kernels.wait_host_value(payload=wait_payload, device=device_ref)
        graph.output()
    return graph


@pytest.fixture
def accelerator() -> Accelerator:
    return Accelerator()


def test_wait_host_value_releases_on_callback_store(
    accelerator: Accelerator,
) -> None:
    """The graph runs callback -> wait, observing the callback's flag store."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("mo.wait_host_value is only supported on CUDA/HIP devices")

    flag = CompletionFlag(accelerator)

    called: list[int] = []
    callback_thread_id: list[int] = []
    main_thread_id = threading.get_ident()

    def cb() -> None:
        # The callback runs on a CUDA driver thread. Sleep briefly to
        # ensure the wait_value node has been issued on the stream
        # before we publish, making the test exercise the wait path
        # rather than a fast-path "value already set" early return.
        # The host callback signals the flag directly because it runs
        # through `mo.launch_host_func` (synchronous), not the
        # async-host-func trampoline that auto-signals from its worker.
        # `mo.wait_host_value` observes the store via the device-mapped
        # alias.
        callback_thread_id.append(threading.get_ident())
        time.sleep(0.002)
        flag.signal(1)
        called.append(1)

    trampoline_ptr, user_data_ptr = __unsafe_pack_py_host_func(cb)

    session = InferenceSession(devices=[accelerator])
    model = session.load(
        _build_launch_then_wait_graph(DeviceRef.from_device(accelerator))
    )

    kickoff_payload = Buffer(shape=[2], dtype=DType.int64)
    kickoff_payload[0] = trampoline_ptr
    kickoff_payload[1] = user_data_ptr

    wait_payload = Buffer(shape=[2], dtype=DType.int64)
    # `_unsafe_ptr` is the raw `M::Driver::CompletionFlag *`; the op
    # rebuilds the typed Mojo `CompletionFlag` wrapper from it.
    wait_payload[0] = flag._unsafe_ptr
    wait_payload[1] = 1

    model.execute(kickoff_payload, wait_payload)
    accelerator.synchronize()

    assert called == [1], "callback should have run exactly once"
    assert callback_thread_id and callback_thread_id[0] != main_thread_id, (
        "callback should run on a CUDA driver thread, not the main thread"
    )
    assert flag.load() == 1, "flag should be 1 after wait_value passes"


def test_wait_host_value_passes_when_flag_already_set(
    accelerator: Accelerator,
) -> None:
    """If the flag is already at the expected value, wait_value is a no-op.

    Builds a wait-only graph (no launch_host_func), pre-sets the flag
    from the main thread, and confirms the graph executes without hang.
    """
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("mo.wait_host_value is only supported on CUDA/HIP devices")

    flag = CompletionFlag(accelerator)
    flag.signal(1)

    with Graph(
        "wait_only",
        input_types=[BufferType(DType.int64, [2], device=DeviceRef.CPU())],
    ) as graph:
        wait_payload = graph.inputs[0].buffer
        kernels.wait_host_value(
            payload=wait_payload, device=DeviceRef.from_device(accelerator)
        )
        graph.output()

    session = InferenceSession(devices=[accelerator])
    model = session.load(graph)

    wait_payload_buf = Buffer(shape=[2], dtype=DType.int64)
    wait_payload_buf[0] = flag._unsafe_ptr
    wait_payload_buf[1] = 1

    model.execute(wait_payload_buf)
    accelerator.synchronize()  # would hang if wait_value did not pass
