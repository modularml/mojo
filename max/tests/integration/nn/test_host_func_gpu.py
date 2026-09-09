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
"""Integration test for the ``mo.launch_host_func`` custom op.

The op enqueues a Python callback on the device's default stream via
``cuLaunchHostFunc``. The callback pointer and its user-data pointer are
produced by ``max._core.driver.__unsafe_pack_py_host_func`` and passed to the op
as a 1-D int64 buffer of shape ``(2,)``.
"""

import pytest
from max.driver import Accelerator, Buffer, __unsafe_pack_py_host_func
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph
from max.nn import kernels


def _build_launch_graph(device_ref: DeviceRef) -> Graph:
    """Returns a one-op graph that wraps ``mo.launch_host_func``."""
    with Graph(
        "host_func_launch",
        input_types=[BufferType(DType.int64, [2], device=DeviceRef.CPU())],
    ) as graph:
        payload = graph.inputs[0].buffer
        kernels.launch_host_func(payload=payload, device=device_ref)
        graph.output()
    return graph


@pytest.fixture
def accelerator() -> Accelerator:
    return Accelerator()


def test_launch_host_func_runs_callback(accelerator: Accelerator) -> None:
    """Enqueues a Python callback through the custom op and verifies it ran."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("mo.launch_host_func is only supported on CUDA/HIP devices")

    calls: list[str] = []

    def cb() -> None:
        calls.append("hit")

    trampoline_ptr, user_data_ptr = __unsafe_pack_py_host_func(cb)

    session = InferenceSession(devices=[accelerator])
    model = session.load(
        _build_launch_graph(DeviceRef.from_device(accelerator))
    )

    payload = Buffer(shape=[2], dtype=DType.int64)
    payload[0] = trampoline_ptr
    payload[1] = user_data_ptr

    model.execute(payload)
    accelerator.synchronize()

    assert calls == ["hit"]
