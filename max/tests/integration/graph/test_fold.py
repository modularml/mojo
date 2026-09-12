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

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.ops import fold


def test_fold(session: InferenceSession) -> None:
    input_shape = (1, 6, 15)
    output_size = (5, 6)
    kernel_size = (3, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)

    # Input and expected output values from test_fold.mojo.
    # fmt: off
    input_value = [24., 43., 47., 13., 27., 24., 16.,  1., 41.,  1., 45., 24.,  4.,  7., 36.,
                11., 13., 36., 14.,  1., 28.,  2., 20., 20., 45., 27., 44., 20., 40., 14.,
                36., 45., 12., 30., 35., 15., 34.,  7., 32., 18., 32., 13.,  4., 39., 4.,
                38., 36., 24., 27., 16., 11., 49., 30., 37.,  1., 46.,  6., 41., 31., 26.,
                47., 45.,  7., 36., 14., 40., 23., 27.,  4., 22., 11.,  9., 28., 19., 48.,
                26., 26.,  8., 32.,  4., 23., 11., 34., 46., 15., 31., 45., 33.,  3., 17.]
    input_tensor = np.array(input_value, dtype=np.float32).reshape(input_shape)
    expected_output = [24.,  54.,  60.,  49.,  41.,   1.,
                60., 127.,  51., 115.,  83.,  61.,
                107., 167., 137., 133., 177.,  19.,
                72., 105.,  48., 118., 103.,  41.,
                11.,  40.,  73.,  52.,  51.,  17.]
    expected = np.array(expected_output, dtype=np.float32).reshape(1, 1, 5, 6)
    # fmt: on
    device_ref = DeviceRef.from_device(session.devices[0])
    with Graph(
        "fold",
        input_types=(
            TensorType(DType.float32, ["batch", "x", "y"], device_ref),
        ),
    ) as graph:
        output = fold(
            graph.inputs[0].tensor,
            output_size,
            kernel_size,
            stride,
            dilation,
            padding,
        )
        graph.output(output)

    model = session.load(graph)
    input = Buffer.from_numpy(input_tensor).to(model.input_devices[0])
    model_output = model(input)[0]
    assert isinstance(model_output, Buffer)
    actual = model_output.to_numpy()
    np.testing.assert_equal(actual, expected)


def test_fold_dynamic_shape(session: InferenceSession) -> None:
    """Test with dynamic kernel size and output size."""
    input_shape = (1, 6, 15)
    output_size = (5, 6)
    kernel_size = (3, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)

    # Input and expected output values from test_fold.mojo.
    # fmt: off
    input_value = [24., 43., 47., 13., 27., 24., 16.,  1., 41.,  1., 45., 24.,  4.,  7., 36.,
                11., 13., 36., 14.,  1., 28.,  2., 20., 20., 45., 27., 44., 20., 40., 14.,
                36., 45., 12., 30., 35., 15., 34.,  7., 32., 18., 32., 13.,  4., 39., 4.,
                38., 36., 24., 27., 16., 11., 49., 30., 37.,  1., 46.,  6., 41., 31., 26.,
                47., 45.,  7., 36., 14., 40., 23., 27.,  4., 22., 11.,  9., 28., 19., 48.,
                26., 26.,  8., 32.,  4., 23., 11., 34., 46., 15., 31., 45., 33.,  3., 17.]
    input_tensor = np.array(input_value, dtype=np.float32).reshape(input_shape)
    expected_output = [24.,  54.,  60.,  49.,  41.,   1.,
                60., 127.,  51., 115.,  83.,  61.,
                107., 167., 137., 133., 177.,  19.,
                72., 105.,  48., 118., 103.,  41.,
                11.,  40.,  73.,  52.,  51.,  17.]
    expected = np.array(expected_output, dtype=np.float32).reshape(1, 1, 5, 6)
    # fmt: on
    device_ref = DeviceRef.from_device(session.devices[0])
    with Graph(
        "fold",
        input_types=(
            TensorType(DType.float32, ["batch", "x", "y"], device_ref),
            # Set up symbolic dimensions for output_size and kernel_size.
            TensorType(
                DType.float32, ["output_0", "output_1"], DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, ["kernel_0", "kernel_1"], DeviceRef.CPU()
            ),
        ),
    ) as graph:
        output_size_0 = graph.inputs[1].tensor.shape[0]
        output_size_1 = graph.inputs[1].tensor.shape[1]
        kernel_size_0 = graph.inputs[2].tensor.shape[0]
        kernel_size_1 = graph.inputs[2].tensor.shape[1]
        output = fold(
            graph.inputs[0].tensor,
            (output_size_0, output_size_1),
            (kernel_size_0, kernel_size_1),
            stride,
            dilation,
            padding,
        )
        graph.output(output)
    model = session.load(graph)
    input = Buffer.from_numpy(input_tensor).to(model.input_devices[0])
    model_output = model(
        input,
        # Dummy inputs for output_size and kernel_size
        np.zeros(output_size, dtype=np.float32),
        np.zeros(kernel_size, dtype=np.float32),
    )[0]
    assert isinstance(model_output, Buffer)
    actual = model_output.to_numpy()
    np.testing.assert_equal(actual, expected)
