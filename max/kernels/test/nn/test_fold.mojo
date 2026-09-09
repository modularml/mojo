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
"""Tests the fold op.

Script to generate random inputs and expected output values:

```python
import std.math
import torch

def run_fold(output_size, kernel_size, stride=1, dilation=1, padding=0, batch=1, channel=1) raises:
    # Compute dimension of input tensor.
    L = 1
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    padding = padding if isinstance(padding, tuple) else (padding, padding)
    stride = stride if isinstance(stride, tuple) else (stride, stride)
    for n, (o, k) in enumerate(zip(output_size, kernel_size)):
        L_d = int((o + 2* padding[n] - dilation[n] * ( k - 1) - 1) // stride[n] + 1)
        L *= L_d
    fold = torch.nn.Fold(output_size, kernel_size, stride=stride, dilation=dilation, padding=padding)
    dim = channel * math.prod(kernel_size)
    input_dim = (batch, dim, L)

    # Generate random input tensor and run fold op.
    input_tensor = torch.randint(1,50,size=input_dim, dtype=torch.float32)
    out = fold(input_tensor)

    # Print results.
    print("input_dim =", input_tensor.shape)
    print("output_dim =", out.shape)
    print("output_size =",output_size)
    print("kernel_size =", kernel_size)
    print("stride =", stride)
    print("dilation =", dilation)
    print("padding =", padding)
    print("Input values =",str(input_tensor).replace("[","").replace("]",""))
    print("Expected values =",str(out).replace("[","").replace("]",""))

run_fold((5,6), (3,2), stride=1, dilation=1, padding=0)

```
"""

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.fold import fold


from std.utils.index import Index, IndexList


# CHECK-LABEL: test_fold
def test[
    dtype: DType,
    //,
    input_shape: IndexList[3],
    output_shape: IndexList[4],
    stride: Tuple[Int, Int],
    dilation: Tuple[Int, Int],
    padding: Tuple[Int, Int],
](
    output_size: IndexList[2],
    kernel_size: IndexList[2],
    input_values: List[Scalar[dtype]],
    expected_output: List[Scalar[dtype]],
) raises:
    print("== test_fold")

    # Create input tensor with dynamic layout
    comptime input_layout = row_major(Coord(input_shape))
    var input_stack = List[Scalar[dtype]](
        capacity=input_shape.flattened_length()
    )
    input_stack.resize(input_shape.flattened_length(), 0)
    var input_stack_ptr = input_stack.unsafe_ptr()
    var input = TileTensor(input_stack_ptr, input_layout)
    _copy_values_to_tile_tensor(input, input_values)

    # Create expected tensor with dynamic layout
    comptime output_layout = row_major(Coord(output_shape))
    var expected_stack = List[Scalar[dtype]](
        capacity=output_shape.flattened_length()
    )
    expected_stack.resize(output_shape.flattened_length(), 0)
    var expected_stack_ptr = expected_stack.unsafe_ptr()
    var expected = TileTensor(expected_stack_ptr, output_layout)
    _copy_values_to_tile_tensor(expected, expected_output)

    # Create output tensor with dynamic layout
    var output_stack = List[Scalar[dtype]](
        capacity=output_shape.flattened_length()
    )
    output_stack.resize(output_shape.flattened_length(), 0)
    var output_stack_ptr = output_stack.unsafe_ptr()
    var output = TileTensor(output_stack_ptr, output_layout)

    fold[stride=stride, dilation=dilation, padding=padding, target="cpu"](
        input=input,
        output=output,
        output_size=output_size,
        kernel_size=kernel_size,
        ctx=DeviceContext(api="cpu"),
    )

    # Check results, return on the first failed comparison.
    for n in range(output_shape[0]):
        for c in range(output_shape[1]):
            for h in range(output_shape[2]):
                for w in range(output_shape[3]):
                    var expected_val = expected[n, c, h, w]
                    var output_val = output[n, c, h, w]
                    if expected_val != output_val:
                        print("Input shape: ", input_shape)
                        print("Output shape: ", output_shape)
                        print("Output size: ", output_size)
                        print("Kernel size: ", kernel_size)
                        print("Stride: ", stride[0], stride[1])
                        print("Dilation: ", dilation[0], dilation[1])
                        print("Padding: ", padding[0], padding[1])
                        print(
                            "Test failed at index: ",
                            Index(n, c, h, w),
                        )
                        print(
                            "Golden value: ",
                            expected_val,
                        )
                        print(
                            "Actual value: ",
                            output_val,
                        )
                        return

    # CHECK: Succeed
    print("Succeed")


def _copy_values_to_tile_tensor[
    dtype: DType,
](
    tensor: TileTensor[mut=True, dtype, ...],
    values: List[Scalar[dtype]],
) raises:
    var num_elements = tensor.num_elements()

    if num_elements != len(values):
        raise Error("Tensor size and values size mismatch")

    for i in range(num_elements):
        tensor.raw_store(i, values[i])


def main() raises:
    comptime dtype = DType.float32
    # fmt: off
    test[
        input_shape = IndexList[3](1, 6, 15),
        output_shape = IndexList[4](1, 1, 5, 6),
        stride=(1, 1),
        dilation=(1, 1),
        padding=(0, 0),
    ](
        output_size=Index(5, 6),
        kernel_size=Index(3, 2),
        input_values=[
            Scalar[dtype](24.), 43., 47., 13., 27., 24., 16.,  1., 41.,  1., 45., 24.,  4.,  7., 36.,
            11., 13., 36., 14.,  1., 28.,  2., 20., 20., 45., 27., 44., 20., 40., 14.,
            36., 45., 12., 30., 35., 15., 34.,  7., 32., 18., 32., 13.,  4., 39., 4.,
            38., 36., 24., 27., 16., 11., 49., 30., 37.,  1., 46.,  6., 41., 31., 26.,
            47., 45.,  7., 36., 14., 40., 23., 27.,  4., 22., 11.,  9., 28., 19., 48.,
            26., 26.,  8., 32.,  4., 23., 11., 34., 46., 15., 31., 45., 33.,  3., 17.
        ],
        expected_output=[
            Scalar[dtype](24.),  54.,  60.,  49.,  41.,   1.,
            60., 127.,  51., 115.,  83.,  61.,
            107., 167., 137., 133., 177.,  19.,
            72., 105.,  48., 118., 103.,  41.,
            11.,  40.,  73.,  52.,  51.,  17.
        ],
    )

    # Test with dilation.
    test[
        input_shape = IndexList[3](1, 6, 4),
        output_shape = IndexList[4](1, 1, 5, 6),
        stride=(1, 1),
        dilation=(2, 2),
        padding=(0, 0),
    ](
        output_size=Index(5, 6),
        kernel_size=Index(3, 2),
        input_values=[
            Scalar[dtype](49.), 24., 22.,  9.,
            48., 38., 32., 30.,
            8., 35.,  1.,  1.,
            5., 13., 16., 10.,
            2., 26., 14., 47.,
            14., 46., 38.,  7.
        ],
        expected_output=[
            Scalar[dtype](49.), 24., 70., 47., 32., 30.,
            0.,  0.,  0.,  0.,  0.,  0.,
            8., 35.,  6., 14., 16., 10.,
            0.,  0.,  0.,  0.,  0.,  0.,
            2., 26., 28., 93., 38.,  7.
        ],
    )

    # Test with stride and dilation.
    test[
        input_shape = IndexList[3](1, 6, 2),
        output_shape = IndexList[4](1, 1, 5, 6),
        stride=(2, 2),
        dilation=(2, 2),
        padding=(0, 0),
    ](
        output_size=Index(5, 6),
        kernel_size=Index(3, 2),
        input_values=[
            Scalar[dtype](6.),  8.,
            39., 43.,
            43., 32.,
            32., 12.,
            13., 12.,
            44., 27.
        ],
        expected_output=[
            Scalar[dtype](6.),  0., 47.,  0., 43.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,
            43.,  0., 64.,  0., 12.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,
            13.,  0., 56.,  0., 27.,  0.
        ],
    )

    # Test with stride, dilation and padding.
    test[
        input_shape = IndexList[3](1, 6, 12),
        output_shape = IndexList[4](1, 1, 5, 6),
        stride=(2, 2),
        dilation=(1, 1),
        padding=(1, 1),
    ](
        output_size=Index(5, 6),
        kernel_size=Index(3, 2),
        input_values=[
            Scalar[dtype](20.), 23., 46., 16., 22., 40.,  9.,  6., 17., 31., 31.,  7.,
            6.,  3., 26.,  6., 34., 15.,  2., 21., 10.,  8., 48., 37.,
            16., 49., 10., 49.,  1., 47., 40., 26., 26., 42.,  4., 34.,
            9., 26., 18., 35., 18.,  5.,  4., 20., 21., 29., 18., 22.,
            43., 21., 18., 23., 33.,  4.,  2., 26., 46., 43., 15., 46.,
            6., 34.,  8.,  1., 34., 46., 39., 14.,  3., 44., 12., 22.
        ],
        expected_output=[
            Scalar[dtype](9.), 49., 26., 10., 18., 49.,
            40., 61., 49., 27., 10., 29.,
            18., 47.,  5., 40.,  4., 26.,
            44., 35., 54., 33., 87., 33.,
            21., 42., 29.,  4., 18., 34.
        ],
    )

    # Test with batch > 1.
    test[
        input_shape = IndexList[3](2, 4, 2),
        output_shape = IndexList[4](2, 1, 2, 3),
        stride=(1, 1),
        dilation=(1, 1),
        padding=(0, 0),
    ](
        output_size=Index(2, 3),
        kernel_size=Index(2, 2),
        input_values=[
            Scalar[dtype](39.), 32.,
            42., 31.,
            36., 48.,
            36.,  7.,
            3., 12.,
            49., 47.,
            36., 32.,
            9.,  4.
        ],
        expected_output=[
            Scalar[dtype](39.), 74., 31.,
            36., 84.,  7.,
            3., 61., 47.,
            36., 41.,  4.
        ],
    )


    # Test with channel size > 1.
    test[
        input_shape = IndexList[3](1, 8, 2),
        output_shape = IndexList[4](1, 2, 2, 3),
        stride=(1, 1),
        dilation=(1, 1),
        padding=(0, 0),
    ](
        output_size=Index(2, 3),
        kernel_size=Index(2, 2),
        input_values=[
            Scalar[dtype](42.),  3.,
            39., 27.,
            37., 26.,
            25., 49.,
            6., 42.,
            29., 32.,
            38., 32.,
            18., 34.
        ],
        expected_output=[
            Scalar[dtype](42.), 42., 27.,
            37., 51., 49.,
            6., 71., 32.,
            38., 50., 34.
        ],
    )

    # fmt: on
