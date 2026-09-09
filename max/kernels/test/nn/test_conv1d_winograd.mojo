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

from std.math import isclose
from std.random import rand

from nn.conv.conv import Naive2dConvolution

from std.utils.index import Index


def winograd_1d_convolution_3[
    dtype: DType, //, filter_len: Int
](
    input: ImmPointer[Scalar[dtype], _],
    filter: ImmPointer[Scalar[dtype], _],
    output: MutPointer[Scalar[dtype], _],
    input_len: Int,
):
    # TODO: Current implementation requires input_len >= 4
    comptime assert filter_len == 3
    # TODO
    # I expected to have to reverse the filter, but
    # internal conv seems not to do that
    var N = input_len - filter_len + 1

    var b0 = filter[0] + filter[2]
    var b1 = 0.5 * (b0 + filter[1])
    var b2 = 0.5 * (b0 - filter[1])

    for i in range(0, N - 1, 2):
        var a0 = (input[i + 1] + input[i + 2]) * b1
        var a1 = (input[i + 2] - input[i + 1]) * b2

        output[i + 0] = a0 + (a1 + (input[i + 0] - input[i + 2]) * filter[0])
        output[i + 1] = a0 - (a1 + (input[i + 1] - input[i + 3]) * filter[2])

    if N % 2 != 0:
        output[N - 1] = (
            filter[0] * input[N - 1]
            + filter[1] * input[N + 0]
            + filter[2] * input[N + 1]
        )


# CHECK-LABEL: test_conv1d_winograd
def test[dtype: DType](C: Int):  # Input Len
    print("== test_conv1d_winograd")

    # TODO: make assert dynamic
    # comptime assert C >= 4
    comptime S: Int = 3  # Filter len

    var O: Int = C - S + 1  # Output len (method="same")
    var input_ptr = List(length=C, fill=Scalar[dtype](0))
    var filter_ptr = List(length=S, fill=Scalar[dtype](0))
    var output_ptr = List(length=O, fill=Scalar[dtype](0))
    var output_ref_ptr = List(length=O, fill=Scalar[dtype](0))

    rand(input_ptr)
    rand(filter_ptr)

    var output_shape = Index(1, 1, 1, O, 1)
    var input_shape = Index(1, 1, 1, C, 1)
    comptime filter_shape = Index(1, 1, S, 1, 1)
    comptime pad_d = Index(0, 0)
    comptime pad_h = Index(0, 0)
    comptime pad_w = Index(0, 0)
    comptime stride = Index(1, 1, 1)
    comptime dilation = Index(1, 1, 1)
    comptime num_groups = 1

    Naive2dConvolution[
        dtype,
        dtype,
        dtype,
    ].run(
        output_ref_ptr.unsafe_ptr(),
        input_ptr.unsafe_ptr(),
        filter_ptr.unsafe_ptr(),
        output_shape,
        input_shape,
        filter_shape,
        pad_d,
        pad_h,
        pad_w,
        stride,
        dilation,
        1,
    )

    winograd_1d_convolution_3[S](
        input_ptr.unsafe_ptr(),
        filter_ptr.unsafe_ptr(),
        output_ptr.unsafe_ptr(),
        C,
    )

    for idx in range(O):
        if not isclose(
            output_ref_ptr[idx],
            output_ptr[idx],
            atol=1e-6,  # absolute error tolerance
            rtol=1e-6,  # relative error tolerance
        ):
            print(
                "diff naive-winograd: ", output_ref_ptr[idx] - output_ptr[idx]
            )
            print("Mismatch!")
            return

    # CHECK: Succeed
    print("Succeed")
    _ = output_ref_ptr^
    _ = output_ptr^
    _ = filter_ptr^
    _ = input_ptr^


def main() raises:
    comptime dtype = DType.float32

    # Make sure to test both even and odd
    test[dtype](7)
    test[dtype](128)
    test[dtype](129)
    test[dtype](256)
    test[dtype](16000)
    test[dtype](3199)
