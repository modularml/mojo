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
"""Test the max.graph Python bindings."""

import pytest
from conftest import tensor_types
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.nn import Signals

shared_types = st.shared(tensor_types())


def test_allgather_rep_device() -> None:
    """Test unique device error for allgather."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=(
            r"allgather operation must have unique devices across its input"
            r" tensors."
        ),
    ):
        with Graph(
            "allgather",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[3]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            allgather_outputs = ops.allgather(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            )
            graph.output(
                allgather_outputs[0],
                allgather_outputs[1],
                allgather_outputs[2],
                allgather_outputs[3],
            )


def test_allgather_wrong_shape() -> None:
    """Test wrong shape error for allgather when non-concat dimensions don't match."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=(
            "allgather operation inputs must have the same shape in all"
            " dimensions except the concatenation dimension"
        ),
    ):
        with Graph(
            "allgather",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 2], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[3]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            allgather_outputs = ops.allgather(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            )

            graph.output(
                allgather_outputs[0],
                allgather_outputs[1],
                allgather_outputs[2],
                allgather_outputs[3],
            )


def test_allgather_uneven_shapes() -> None:
    """Test allgather with uneven shapes along concatenation dimension."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    # Test with uneven shapes along dimension 0.
    with Graph(
        "allgather_uneven",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[37919, 4096], device=devices[0]
            ),
            TensorType(
                dtype=DType.float32, shape=[37919, 4096], device=devices[1]
            ),
            TensorType(
                dtype=DType.float32, shape=[37918, 4096], device=devices[2]
            ),
            TensorType(
                dtype=DType.float32, shape=[37918, 4096], device=devices[3]
            ),
            *signals.input_types(),
        ],
    ) as graph:
        allgather_outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
        )

        # Check output shapes - should be sum of input shapes along axis 0.
        expected_dim0 = 37919 + 37919 + 37918 + 37918  # 151674
        for output in allgather_outputs:
            assert len(output.shape) == 2
            assert int(output.shape[0]) == expected_dim0
            assert int(output.shape[1]) == 4096

        graph.output(*allgather_outputs)


def test_allgather_bad_dim() -> None:
    """Test wrong shape error for allgather."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with Graph(
        "allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[3]),
            *signals.input_types(),
        ],
    ) as graph:
        with pytest.raises(IndexError, match="Dimension out of range"):
            _ = ops.allgather(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
                axis=-3,
            )

        with pytest.raises(IndexError, match="Dimension out of range"):
            _ = ops.allgather(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
                axis=-5,
            )


def test_allgather_basic() -> None:
    """Test basic allgather use case."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with Graph(
        "allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[3]),
            *signals.input_types(),
        ],
    ) as graph:
        allgather_outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
        )
        graph.output(
            allgather_outputs[0],
            allgather_outputs[1],
            allgather_outputs[2],
            allgather_outputs[3],
        )
        for output, device in zip(allgather_outputs, devices, strict=True):
            assert device == output.device
            assert output.shape == Shape((24, 5))
            assert output.dtype == DType.float32


def test_allgather_nonzero_dim() -> None:
    """Test allgather with non-default axis concatenation."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with Graph(
        "allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[3]),
            *signals.input_types(),
        ],
    ) as graph:
        outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            axis=-2,
        )
        for output in outputs:
            assert output.shape == Shape((6, 20, 4))

        outputs_dim_1 = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            axis=1,
        )
        for output in outputs_dim_1:
            assert output.shape == Shape((6, 20, 4))

        outputs_dim_2 = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            axis=2,
        )
        for output in outputs_dim_2:
            assert output.shape == Shape((6, 5, 16))


def test_allgather_nonzero_dim_uneven() -> None:
    """Test allgather with non-default axis concatenation and uneven shapes."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    # Test with uneven shapes along dimension 1.
    with Graph(
        "allgather_uneven_dim1",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 4, 4], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 4, 4], device=devices[3]),
            *signals.input_types(),
        ],
    ) as graph:
        outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            axis=1,
        )
        for output in outputs:
            assert output.shape == Shape((6, 18, 4))  # 5+5+4+4=18

        graph.output(*outputs)

    # Test with uneven shapes along dimension 2.
    with Graph(
        "allgather_uneven_dim2",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[6, 5, 10], device=devices[0]
            ),
            TensorType(
                dtype=DType.float32, shape=[6, 5, 10], device=devices[1]
            ),
            TensorType(dtype=DType.float32, shape=[6, 5, 9], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5, 9], device=devices[3]),
            *signals.input_types(),
        ],
    ) as graph:
        outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            axis=2,
        )
        for output in outputs:
            assert output.shape == Shape((6, 5, 38))  # 10+10+9+9=38

        graph.output(*outputs)


def test_allgather_grouped_uneven_shapes() -> None:
    """Test grouped allgather with different per-group concat sizes."""
    num_gpus = 8
    group_size = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    signals = Signals(devices)
    input_shapes = [[5, 128]] * group_size + [[3, 128]] * group_size
    expected_shapes = [Shape((20, 128))] * group_size + [
        Shape((12, 128))
    ] * group_size

    with Graph(
        "grouped_allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=shape, device=device)
            for shape, device in zip(input_shapes, devices, strict=True)
        ]
        + list(signals.input_types()),
    ) as graph:
        outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[:num_gpus]),
            signal_buffers=(v.buffer for v in graph.inputs[num_gpus:]),
            axis=0,
            group_size=group_size,
        )
        graph.output(*outputs)
        for output, device, expected_shape in zip(
            outputs, devices, expected_shapes, strict=True
        ):
            assert output.device == device
            assert output.shape == expected_shape
            assert output.dtype == DType.float32


def test_allgather_group_size_must_divide_inputs() -> None:
    """Test grouped allgather requires complete contiguous groups."""
    num_gpus = 6
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=r"group_size to evenly divide the number of input tensors",
    ):
        with Graph(
            "grouped_allgather",
            input_types=[
                TensorType(dtype=DType.float32, shape=[6, 16], device=device)
                for device in devices
            ]
            + list(signals.input_types()),
        ) as graph:
            ops.allgather(
                inputs=(v.tensor for v in graph.inputs[:num_gpus]),
                signal_buffers=(v.buffer for v in graph.inputs[num_gpus:]),
                axis=0,
                group_size=4,
            )


def test_allgather_group_size_one_is_noop() -> None:
    """Tests that singleton allgather groups return the original tensors."""
    devices = [DeviceRef.GPU(id=0), DeviceRef.GPU(id=1)]
    signals = Signals(devices)

    with Graph(
        "allgather_group_size_one",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=device)
            for device in devices
        ]
        + list(signals.input_types()),
    ) as graph:
        outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            group_size=1,
        )
        assert outputs[0]._mlir_value == graph.inputs[0]._mlir_value
        assert outputs[1]._mlir_value == graph.inputs[1]._mlir_value


def test_allgather_noop() -> None:
    """Tests that allgather is a noop if the number of inputs is 0 or 1."""
    devices = [DeviceRef.GPU(id=0)]
    signals = Signals(devices)

    with Graph(
        "allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            *signals.input_types(),
        ],
    ) as graph:
        allgather_outputs = ops.allgather(
            inputs=[graph.inputs[0].tensor],
            signal_buffers=[graph.inputs[1].buffer],
        )
        assert allgather_outputs[0]._mlir_value == graph.inputs[0]._mlir_value

        allgather_outputs = ops.allgather(inputs=[], signal_buffers=[])
        assert not allgather_outputs


def test_allgather_signal_buffer_mismatch() -> None:
    """Test error when number of inputs != number of signal buffers."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices[:3])  # Only 3 signal buffers

    with pytest.raises(
        ValueError,
        match=(
            "expected number of inputs \\(4\\) and number of signal buffers \\(3\\) to match"
        ),
    ):
        with Graph(
            "allgather",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[3]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            allgather_outputs = ops.allgather(
                inputs=(v.tensor for v in graph.inputs[:4]),
                signal_buffers=(v.buffer for v in graph.inputs[4:]),
            )
            graph.output(*allgather_outputs)
