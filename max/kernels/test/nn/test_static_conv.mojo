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

from std.math import ceildiv, isclose
from std.random import rand
from std.sys.info import simd_width_of

from std.itertools import product
from layout import Coord, IntTuple, Layout, LayoutTensor, RuntimeLayout
from layout import lt_to_tt
from nn.conv.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    pack_filter,
    pack_filter_lt,
)
from nn.conv.conv_utils import (
    ConvShape,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
)

from std.utils.index import Index, IndexList


def test[
    N: Int,
    H: Int,
    W: Int,
    C: Int,
    R: Int,
    S: Int,
    F: Int,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
]() raises:
    # Output Shape.
    # fmt: off
    comptime HO = (H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1) // stride[0] + 1
    comptime WO = (W + pad_w[0] + pad_w[1] - dilation[0] * (S - 1) - 1) // stride[1] + 1
    # fmt: on
    comptime type = DType.float32
    comptime simd_size = simd_width_of[type]()
    comptime num_groups = 1

    var conv_shape = ConvShape[2](
        n=N,
        input_dims=Coord(Index(H, W)),
        output_dims=Coord(Index(HO, WO)),
        filter_dims=Coord(Index(R, S)),
        c=C,
        f=F,
        stride=Coord(stride),
        dilation=Coord(dilation),
        pad_d=Coord(Index(0, 0)),
        pad_h=Coord(pad_h),
        pad_w=Coord(pad_w),
        num_groups=num_groups,
    )

    var input_ptr = List(length=N * H * W * C, fill=Scalar[type](0))
    var filter_ptr = List(length=R * S * C * F, fill=Scalar[type](0))

    # output from conv w/ dynamic and static shapes.
    var output_ptr_static = List(length=N * HO * WO * F, fill=Scalar[type](0))
    var output_ptr_dynamic = List(length=N * HO * WO * F, fill=Scalar[type](0))

    rand(input_ptr)
    rand(filter_ptr)

    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()
    var input = LayoutTensor[type, Layout.row_major(N, H, W, C)](input_ptr)
    var filter = LayoutTensor[type, layout_4d](
        filter_ptr, RuntimeLayout[layout_4d].row_major(Index(R, S, C, F))
    )
    var output_static = LayoutTensor[type, Layout.row_major(N, HO, WO, F)](
        output_ptr_static
    )
    var output_dynamic = LayoutTensor[type, layout_4d](
        output_ptr_dynamic,
        RuntimeLayout[layout_4d].row_major(Index(N, HO, WO, F)),
    )

    # Pre-packed filter for dynamic shapes.
    comptime micro_kernel_width_default = get_direct_conv_micro_kernel_width()
    comptime micro_kernel_f_size_default = micro_kernel_width_default * simd_size
    var rounded_F_dynamic = (
        ceildiv(F, micro_kernel_f_size_default) * micro_kernel_f_size_default
    )
    var packed_filter_ptr_dynamic = List(
        length=R * S * C * rounded_F_dynamic, fill=Scalar[type](0)
    )
    var packed_filter_dynamic = LayoutTensor[type, layout_5d](
        packed_filter_ptr_dynamic,
        RuntimeLayout[layout_5d].row_major(
            Index(
                ceildiv(F, micro_kernel_f_size_default),
                R,
                S,
                C,
                micro_kernel_f_size_default,
            )
        ),
    )

    pack_filter(lt_to_tt(filter), lt_to_tt(packed_filter_dynamic), num_groups)

    # Conv attributes.
    comptime conv_attr_dynamic = ConvInfoStatic[2]()

    ConvDirectNHWC[
        input.layout,  # input shape
        layout_5d,  # filter shape
        layout_4d,  # output shape
        type,  # input type
        type,  # filter type
        type,  # output type
        True,
        conv_attr_dynamic,
    ].run(
        output_dynamic,
        input,
        packed_filter_dynamic,
        conv_shape,
    )

    comptime conv_attr_static = ConvInfoStatic[2](
        IntTuple(pad_h[0], pad_w[0], pad_h[1], pad_w[1]),
        IntTuple(stride[0], stride[1]),
        IntTuple(dilation[0], dilation[1]),
        num_groups,
    )

    comptime micro_kernel_shape = get_micro_kernel_shape[
        2, WO, F, conv_attr_static, simd_size
    ]()
    comptime micro_kernel_f_size = micro_kernel_shape[1] * simd_size
    comptime num_f_micro_tiles = ceildiv(F, micro_kernel_f_size)
    comptime rounded_F_static = num_f_micro_tiles * micro_kernel_f_size
    comptime packed_filter_layout = Layout.row_major(
        num_f_micro_tiles, R, S, C, micro_kernel_f_size
    )
    var packed_filter_ptr_static = List(
        length=R * S * C * rounded_F_static, fill=Scalar[type](0)
    )
    var packed_filter_static = LayoutTensor[type, packed_filter_layout](
        packed_filter_ptr_static
    )

    pack_filter_lt[simd_size, micro_kernel_f_size](
        filter,
        packed_filter_static,
        num_groups,
    )

    ConvDirectNHWC[
        Layout.row_major(N, H, W, C),
        packed_filter_layout,
        Layout.row_major(N, HO, WO, F),
        type,  # input type
        type,  # filter type
        type,  # output type
        True,
        conv_attr_static,
    ].run(
        output_static,
        input,
        packed_filter_static,
        conv_shape,
    )

    # Check results, return on the first failed comparison.
    for n, ho, wo, f in product(range(N), range(HO), range(WO), range(F)):
        if not isclose(
            output_dynamic[n, ho, wo, f],
            output_static[n, ho, wo, f],
            atol=1e-4,  # absolute error tolerance
            rtol=1e-5,  # relative error tolerance
        ):
            var expected = output_dynamic[n, ho, wo, f]
            var actual = output_static[n, ho, wo, f]
            print("Input shape NHWC: ", Index(N, H, W, C))
            print("filter shape RSCF: ", Index(R, S, C, F))
            print(
                "Failed at",
                Index(n, ho, wo, f),
                "expected",
                expected,
                "actual",
                actual,
                "rerr",
                abs(actual - expected) / abs(expected + 1e-10),
            )
            return

    # CHECK: Succeed
    print("Succeed")
    _ = output_ptr_static^
    _ = output_ptr_dynamic^
    _ = packed_filter_ptr_static^
    _ = packed_filter_ptr_dynamic^
    _ = filter_ptr^
    _ = input_ptr^


def main() raises:
    test[
        1,  # N
        14,  # H
        14,  # W
        256,  # C
        3,  # R
        3,  # S
        256,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    ]()
    test[
        1,  # N
        2,  # H
        2,  # W
        64,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    ]()

    # Each test will build a specialization of the conv kernel.
    # Disable the following tests for now to monitor build time.

    # test[
    #     1,  # N
    #     56,  # H
    #     56,  # W
    #     64,  # C
    #     3,  # R
    #     3,  # S
    #     64,  # F
    #     Index(1, 1),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     28,  # H
    #     28,  # W
    #     128,  # C
    #     3,  # R
    #     3,  # S
    #     128,  # F
    #     Index(1, 1),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     7,  # H
    #     7,  # W
    #     512,  # C
    #     3,  # R
    #     3,  # S
    #     512,  # F
    #     Index(1, 1),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     224,  # H
    #     224,  # W
    #     3,  # C
    #     7,  # R
    #     7,  # S
    #     64,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(3, 3),  # pad_h
    #     Index(3, 3),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     56,  # H
    #     56,  # W
    #     128,  # C
    #     3,  # R
    #     3,  # S
    #     128,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     28,  # H
    #     28,  # W
    #     256,  # C
    #     3,  # R
    #     3,  # S
    #     256,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     14,  # H
    #     14,  # W
    #     512,  # C
    #     3,  # R
    #     3,  # S
    #     512,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     19,  # N
    #     7,  # H
    #     7,  # W
    #     1,  # C
    #     3,  # R
    #     3,  # S
    #     16,  # F
    #     Index(1, 1),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     13,  # N
    #     14,  # H
    #     14,  # W
    #     2,  # C
    #     3,  # R
    #     3,  # S
    #     32,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()
