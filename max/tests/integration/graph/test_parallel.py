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

"""Tests for staging and executing mo.parallel from the Python Graph API.

Mirrors the MLIR integration test in
GraphCompiler/test/integration/gpu/MODialect/parallel.mlir.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def _gpu_device_configs() -> list[tuple[str, list[int]]]:
    """Parametrize configs: (test_id, list of gpu ids)."""
    configs = [("same_device", [0, 0])]
    if accelerator_count() >= 2:
        configs.append(("cross_device", [0, 1]))
    return configs


@pytest.mark.parametrize(
    "device_ids",
    [c[1] for c in _gpu_device_configs()],
    ids=[c[0] for c in _gpu_device_configs()],
)
def test_parallel_relu(
    session: InferenceSession, device_ids: list[int]
) -> None:
    """Test mo.parallel with relu dispatched across the given devices."""
    required_gpus = max(device_ids) + 1
    if accelerator_count() < required_gpus:
        pytest.skip(f"Test requires at least {required_gpus} GPU(s)")

    devices = [DeviceRef.GPU(id=i) for i in device_ids]
    input_data = [
        np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32),
        np.array([-5.0, 6.0, -7.0, 8.0], dtype=np.float32),
    ]

    with Graph(
        "parallel_relu",
        input_types=[
            TensorType(DType.float32, shape=[4], device=dev) for dev in devices
        ],
    ) as graph:
        inputs = [inp.tensor for inp in graph.inputs]
        bundles = ops.parallel(
            [inputs],
            lambda x: ops.relu(x),
            result_types=[[t.type for t in inputs]],
        )
        assert isinstance(bundles, list)
        [results] = bundles
        graph.output(*[r.to(DeviceRef.CPU()) for r in results])

    compiled = session.load(graph)

    buffers = [
        Buffer.from_numpy(d).to(Accelerator(gid))
        for d, gid in zip(input_data, device_ids, strict=True)
    ]
    output = compiled.execute(*buffers)

    host = CPU()
    for out_buf, inp in zip(output, input_data, strict=True):
        np.testing.assert_array_equal(
            out_buf.to(host).to_numpy(), np.maximum(inp, 0)
        )


@pytest.mark.parametrize(
    "device_ids",
    [c[1] for c in _gpu_device_configs()],
    ids=[c[0] for c in _gpu_device_configs()],
)
def test_parallel_relu_symbolic_dim(
    session: InferenceSession, device_ids: list[int]
) -> None:
    """Test mo.parallel with relu and a symbolic batch dimension."""
    required_gpus = max(device_ids) + 1
    if accelerator_count() < required_gpus:
        pytest.skip(f"Test requires at least {required_gpus} GPU(s)")

    devices = [DeviceRef.GPU(id=i) for i in device_ids]
    input_data = [
        np.array([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]], dtype=np.float32),
        np.array([[-1.0, 2.0], [-3.0, 4.0], [-5.0, 6.0]], dtype=np.float32),
    ]

    with Graph(
        "parallel_relu_symbolic",
        input_types=[
            TensorType(DType.float32, shape=["batch", 2], device=dev)
            for dev in devices
        ],
    ) as graph:
        inputs = [inp.tensor for inp in graph.inputs]
        bundles = ops.parallel(
            [inputs],
            lambda x: ops.relu(x),
            result_types=[[t.type for t in inputs]],
        )
        assert isinstance(bundles, list)
        [results] = bundles
        graph.output(*[r.to(DeviceRef.CPU()) for r in results])

    compiled = session.load(graph)

    buffers = [
        Buffer.from_numpy(d).to(Accelerator(gid))
        for d, gid in zip(input_data, device_ids, strict=True)
    ]
    output = compiled.execute(*buffers)

    host = CPU()
    for out_buf, inp in zip(output, input_data, strict=True):
        np.testing.assert_array_equal(
            out_buf.to(host).to_numpy(), np.maximum(inp, 0)
        )
