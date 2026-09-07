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

# Use `kgen --emit=asm %s -o %t.asm` to exam the assembly code.

from std.math import ceildiv
from std.sys.info import simd_width_of

from layout import Coord, IntTuple, Layout, LayoutTensor
from nn.conv.conv import ConvDirectNHWC, ConvInfoStatic
from nn.conv.conv_utils import ConvShape, get_micro_kernel_shape

from std.utils.index import Index

comptime N = 1
comptime H = 14
comptime W = 14
comptime C = 8
comptime R = 3
comptime S = 3
comptime F = 8
comptime stride_h = 1
comptime stride_w = 1
comptime pad_left = 1
comptime pad_right = 1
comptime pad_top = 1
comptime pad_bottom = 1
comptime dilation_h = 1
comptime dilation_w = 1
comptime HO = (
    H + pad_left + pad_right - dilation_h * (R - 1) - 1
) // stride_h + 1
comptime WO = (
    W + pad_top + pad_bottom - dilation_w * (S - 1) - 1
) // stride_w + 1
comptime num_groups = 1

comptime conv_attr = ConvInfoStatic[2](
    IntTuple(pad_bottom, pad_left, pad_top, pad_right),
    IntTuple(stride_h, stride_w),
    IntTuple(dilation_h, dilation_w),
    num_groups,
)

comptime value_type = DType.float32
comptime simd_size = simd_width_of[value_type]()
comptime micro_kernel_shape = get_micro_kernel_shape[
    2, WO, F, conv_attr, simd_size
]()
# alias micro_kernel_width = get_direct_conv_micro_kernel_width()
comptime micro_kernel_f_size = micro_kernel_shape[1] * simd_size
comptime num_micro_tile = ceildiv(F, micro_kernel_f_size)


def static_conv(
    output: LayoutTensor[
        mut=True, value_type, Layout.row_major(N, HO, WO, F), _
    ],
    input: LayoutTensor[value_type, Layout.row_major(N, H, W, C), _],
    filter: LayoutTensor[
        value_type,
        Layout.row_major(num_micro_tile, R, S, C, micro_kernel_f_size),
        _,
    ],
):
    var conv_shape = ConvShape[2](
        n=N,
        input_dims=Coord(Index(H, W)),
        output_dims=Coord(Index(HO, WO)),
        filter_dims=Coord(Index(R, S)),
        c=C,
        f=F,
        stride=Coord(Index(stride_h, stride_w)),
        dilation=Coord(Index(dilation_h, dilation_w)),
        pad_d=Coord(Index(0, 0)),
        pad_h=Coord(Index(pad_bottom, pad_top)),
        pad_w=Coord(Index(pad_left, pad_right)),
        num_groups=num_groups,
    )

    def direct_null_elementwise_epilogue(
        n: Int, ho: Int, wo: Int, f_offset: Int, f_size: Int
    ):
        pass

    try:
        ConvDirectNHWC[
            Layout.row_major(N, H, W, C),
            Layout.row_major(num_micro_tile, R, S, C, micro_kernel_f_size),
            Layout.row_major(N, HO, WO, F),
            value_type,
            value_type,
            value_type,
            True,
            conv_attr,
        ].run(output, input, filter, conv_shape)
    except e:
        print(e)


# CHECK-LABEL: test_static_conv
def test_static_conv() raises:
    print("== test_static_conv")

    var output_stack = Array[Scalar[value_type], N * HO * WO * F](fill=0.0)
    var output = LayoutTensor[value_type, Layout.row_major(N, HO, WO, F)](
        output_stack
    )
    var input_stack = Array[Scalar[value_type], N * H * W * C](fill=1.0)
    var input = LayoutTensor[value_type, Layout.row_major(N, H, W, C)](
        input_stack
    )
    var filter_stack = Array[
        Scalar[value_type], num_micro_tile * R * S * C * micro_kernel_f_size
    ](fill=1.0)
    var filter = LayoutTensor[
        value_type,
        Layout.row_major(num_micro_tile, R, S, C, micro_kernel_f_size),
    ](filter_stack)

    static_conv(output, input, filter)

    # CHECK: 32.0
    print(output[0, 0, 0, 0])


def main() raises:
    test_static_conv()
