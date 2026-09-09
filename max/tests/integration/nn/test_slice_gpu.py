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

import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def test_slice_gpu_ideal_case() -> None:
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    x_input = TensorType(
        shape=("dynamic_dim", 10), dtype=DType.float32, device=device_ref
    )
    x_slice_input = TensorType(shape=(1,), dtype=DType.int64, device=device_ref)

    with Graph("slice_gpu", input_types=[x_input, x_slice_input]) as graph:
        x_tensor, x_slice_val = graph.inputs

        # TODO(MAXPLAT-363): Better error message
        with pytest.raises(TypeError):
            _ = x_tensor.tensor[x_slice_val.tensor :, :]


def test_slice_gpu_explicit_devices() -> None:
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    x_input = TensorType(
        shape=("dynamic_dim", 10), dtype=DType.float32, device=device_ref
    )
    x_slice_input = TensorType(shape=(1,), dtype=DType.int64, device=device_ref)

    with Graph("slice_gpu", input_types=[x_input, x_slice_input]) as graph:
        x_tensor, x_slice_val = graph.inputs
        # TODO(MAXPLAT-363): Support slice indices on GPU
        with pytest.raises(ValueError, match="must be on the host device"):
            ops.slice_tensor(
                x_tensor.tensor,
                [
                    (
                        slice(
                            x_slice_val.tensor,
                            ops.constant(-1, DType.int64, device=device_ref),
                            ops.constant(1, DType.int64, device=device_ref),
                        ),
                        "dynamic_slice_dim",
                    ),
                    (
                        slice(
                            ops.constant(0, DType.int64, device=device_ref),
                            ops.constant(10, DType.int64, device=device_ref),
                            ops.constant(1, DType.int64, device=device_ref),
                        ),
                        10,
                    ),
                ],
            )


def test_slice_gpu_scalar_slice_ideal() -> None:
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    x_input = TensorType(
        shape=("dynamic_dim"), dtype=DType.float32, device=device_ref
    )
    x_slice_input = TensorType(shape=(1,), dtype=DType.int64, device=device_ref)

    with Graph("slice_gpu", input_types=[x_input, x_slice_input]) as graph:
        x_tensor, x_slice_val = graph.inputs
        # TODO(MAXPLAT-363):
        with pytest.raises(ValueError, match="must be on the same device"):
            _ = x_tensor.tensor[x_slice_val.tensor]


def test_slice_gpu_scalar_slice() -> None:
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    x_input = TensorType(
        shape=("dynamic_dim"), dtype=DType.float32, device=device_ref
    )
    x_slice_input = TensorType(shape=(1,), dtype=DType.int64, device=device_ref)

    with Graph("slice_gpu", input_types=[x_input, x_slice_input]) as graph:
        x_tensor, x_slice_val = graph.inputs
        # TODO(MAXPLAT-363):
        with pytest.raises(ValueError, match="must be on the same device"):
            ops.slice_tensor(
                x_tensor.tensor,
                [
                    (
                        slice(
                            x_slice_val.tensor,
                            x_slice_val.tensor + 1,
                            ops.constant(1, DType.int64, device=device_ref),
                        ),
                        1,
                    )
                ],
            )
