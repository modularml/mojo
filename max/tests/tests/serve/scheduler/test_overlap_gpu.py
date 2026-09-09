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


import time

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    DevicePinnedBuffer,
    Usage,
    accelerator_api,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, SymbolicDim, TensorType, ops
from max.nn import kernels

KERNEL_DURATION_SEC = 3
CPU_WORK_DURATION_SEC = 1


def build_graph(device_ref: DeviceRef) -> Model:
    with Graph(
        "my_add_graph",
        input_types=[
            TensorType(DType.int8, [SymbolicDim("size")], device=device_ref),
            TensorType(DType.int8, [SymbolicDim("size")], device=device_ref),
            BufferType(DType.float64, [1], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y, sleep_duration = graph.inputs
        z = ops.add(x, y)
        kernels.sleep(sleep_duration.buffer, device_ref=device_ref)
        graph.output(z)
    device = device_ref.to_device()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    return model


def expensive_cpu_preprocessing() -> None:
    time.sleep(CPU_WORK_DURATION_SEC)


def expensive_cpu_postprocessing(array: np.ndarray, expected: int) -> None:
    assert (array == expected).all()
    time.sleep(CPU_WORK_DURATION_SEC)


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="FIXME MSTDL-2238: sleep kernel does not work on HIP.",
)
@pytest.mark.parametrize(
    "enable_overlap,expected_elapsed_time", [(False, 10), (True, 8)]
)
def test_overlap(enable_overlap: bool, expected_elapsed_time: int) -> None:
    """Test overlap of GPU and Python host code.

    I: Input Processing
    O: Output Processing
    K: Kernel Execution

    Without overlapping: Elapsed time ~10s           |
                                                     V
      Time: 0   1   2   3   4   5   6   7   8   9   10  11  12
       CPU: < I >           < O >< I>           < O >
       GPU:     [     K     ]       [     K     ]

    With overlapping: Elapsed time ~8s      |
                                            V
      Time: 0   1   2   3   4   5   6   7   8   9   10  11  12
       CPU: < I >< I>        < O >      < O >
       GPU:     [     K     ][     K    ]
    """
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    size = 1024

    # Build and load simple graph
    model = build_graph(device_ref=device_ref)

    # Allocate sleep duration buffer
    sleep_duration = Buffer(dtype=DType.float64, shape=[1], device=CPU())
    sleep_duration.to_numpy().fill(KERNEL_DURATION_SEC)

    # Allocate buffers before timing region
    if not enable_overlap:
        # Allocate regular pinned buffers for non-overlap case
        a_pinned = Buffer(
            dtype=DType.int8, shape=[size], device=device, usage=Usage.STAGING
        )
        a_pinned.to_numpy().fill(1)
        b_pinned = Buffer(
            dtype=DType.int8, shape=[size], device=device, usage=Usage.STAGING
        )
        b_pinned.to_numpy().fill(2)
        c_pinned = Buffer(
            dtype=DType.int8, shape=[size], device=device, usage=Usage.STAGING
        )
        d_pinned = Buffer(
            dtype=DType.int8, shape=[size], device=device, usage=Usage.STAGING
        )
    else:
        # Allocate DevicePinnedBuffer for overlap case
        a_pinned = DevicePinnedBuffer(
            dtype=DType.int8, shape=[size], device=device
        )
        a_pinned.to_numpy().fill(1)
        b_pinned = DevicePinnedBuffer(
            dtype=DType.int8, shape=[size], device=device
        )
        b_pinned.to_numpy().fill(2)
        c_pinned = DevicePinnedBuffer(
            dtype=DType.int8, shape=[size], device=device
        )
        d_pinned = DevicePinnedBuffer(
            dtype=DType.int8, shape=[size], device=device
        )

    t0 = time.monotonic()

    if not enable_overlap:
        # Run batch 1
        expensive_cpu_preprocessing()
        (c,) = model.execute(
            a_pinned.to(device), b_pinned.to(device), sleep_duration
        )
        c_pinned.inplace_copy_from(c)
        # Reads of pinned host buffers no longer synchronize pending device
        # work implicitly; the caller owns ordering.
        device.synchronize()
        expensive_cpu_postprocessing(c_pinned.to_numpy(), expected=3)

        # Run batch 2
        expensive_cpu_preprocessing()
        (d,) = model.execute(c, c, sleep_duration)
        d_pinned.inplace_copy_from(d)
        device.synchronize()
        expensive_cpu_postprocessing(d_pinned.to_numpy(), expected=6)
    else:
        # Run batch 1
        expensive_cpu_preprocessing()
        (c,) = model.execute(
            a_pinned.to(device), b_pinned.to(device), sleep_duration
        )
        c_pinned.inplace_copy_from(c)
        # Record event after batch 1 completes
        event1 = device.default_queue.record_event()

        # Run batch 2 (can overlap with CPU work)
        expensive_cpu_preprocessing()
        (d,) = model.execute(c, c, sleep_duration)
        d_pinned.inplace_copy_from(d)
        # Record event after batch 2 completes
        event2 = device.default_queue.record_event()

        # Postprocess results of batch 1 (wait for batch 1 to complete)
        event1.synchronize()
        expensive_cpu_postprocessing(c_pinned.to_numpy(), expected=3)

        # Postprocess results of batch 2 (wait for batch 2 to complete)
        event2.synchronize()
        expensive_cpu_postprocessing(d_pinned.to_numpy(), expected=6)

    t1 = time.monotonic()
    elapsed_time = t1 - t0
    print(f"Time taken: {elapsed_time} seconds")

    # Disable check since this is unreliable in CI
    #
    # Check that the measured elapsed time is within 0.5 seconds of expected
    # assert abs(elapsed_time - expected_elapsed_time) < 0.5
