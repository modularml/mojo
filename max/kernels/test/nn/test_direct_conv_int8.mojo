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
from std.sys.info import num_physical_cores, simd_width_of

from layout import Coord, Layout, LayoutTensor, RuntimeLayout
from nn.conv.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    pack_filter_lt,
)
from nn.conv.conv_utils import (
    ConvShape,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)

from std.utils.index import Index, IndexList

comptime input_type = DType.uint8
comptime filter_type = DType.int8
comptime output_type = DType.int32
comptime simd_size: Int = simd_width_of[output_type]()


# CHECK-LABEL: test_direct_conv
def test[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_packed: Bool,
](
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
) raises:
    print("== test_direct_conv")

    # fmt: off
    var HO = (H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    var WO = (W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1
    # fmt: on

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
        num_groups=1,
    )

    var input_ptr = List(length=N * H * W * C, fill=Scalar[input_type](0))
    var filter_ptr = List(length=R * S * C * F, fill=Scalar[filter_type](0))
    var output_ptr = List(length=N * HO * WO * F, fill=Scalar[output_type](0))
    var output_ref_ptr = List(
        length=N * HO * WO * F, fill=Scalar[output_type](0)
    )

    rand(input_ptr)
    rand(filter_ptr)

    # Find the tile size used in packing.
    comptime micro_kernel_height = get_direct_conv_micro_kernel_height()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()

    # Rounded C and F size for pre-packed filter.
    comptime micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    var rounded_F = ceildiv(F, micro_kernel_f_size) * micro_kernel_f_size
    var packed_filter_ptr = List(
        length=R * S * C * rounded_F, fill=Scalar[filter_type](0)
    )

    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()
    var input = LayoutTensor[input_type, layout_4d](
        input_ptr, RuntimeLayout[layout_4d].row_major(Index(N, H, W, C))
    )
    var filter = LayoutTensor[filter_type, layout_4d](
        filter_ptr, RuntimeLayout[layout_4d].row_major(Index(R, S, C, F))
    )
    var packed_filter = LayoutTensor[filter_type, layout_5d](
        packed_filter_ptr,
        RuntimeLayout[layout_5d].row_major(
            Index(
                ceildiv(F, micro_kernel_width * simd_size),
                R,
                S,
                C,
                micro_kernel_width * simd_size,
            ),
        ),
    )
    var output = LayoutTensor[output_type, layout_4d](
        output_ptr, RuntimeLayout[layout_4d].row_major(Index(N, HO, WO, F))
    )
    var output_ref = LayoutTensor[output_type, layout_4d](
        output_ref_ptr, RuntimeLayout[layout_4d].row_major(Index(N, HO, WO, F))
    )

    comptime if filter_packed:
        pack_filter_lt[simd_size, micro_kernel_f_size](
            filter, packed_filter, conv_shape.num_groups
        )

    # Reference: naive conv
    Naive2dConvolution[output_type, input_type, filter_type].run(
        output_ref_ptr.unsafe_ptr(),
        input_ptr.unsafe_ptr(),
        filter_ptr.unsafe_ptr(),
        Index(N, 1, HO, WO, F),
        Index(N, 1, H, W, C),
        Index(1, R, S, C, F),
        Index(0, 0),
        pad_h,
        pad_w,
        Index(1, stride[0], stride[1]),
        Index(1, dilation[0], dilation[1]),
        conv_shape.num_groups,
    )

    # Test direct conv
    comptime conv_attr = ConvInfoStatic[2]()

    comptime if filter_packed:
        ConvDirectNHWC[
            layout_4d,
            layout_5d,
            layout_4d,
            input_type,
            filter_type,
            output_type,
            True,
            conv_attr,
        ].run(
            output,
            input,
            packed_filter,
            conv_shape,
        )
    else:
        ConvDirectNHWC[
            layout_4d,
            layout_4d,
            layout_4d,
            input_type,
            filter_type,
            output_type,
            False,
            conv_attr,
        ].run(output, input, filter, conv_shape)

    # Check results, return on the first failed comparison.
    for n in range(N):
        for ho in range(HO):
            for wo in range(WO):
                for f in range(F):
                    comptime if output_type.is_floating_point():
                        if isclose(
                            output_ref[n, ho, wo, f],
                            output[n, ho, wo, f],
                            atol=1e-4,  # absolute error tolerance
                            rtol=1e-5,  # relative error tolerance
                        ):
                            continue
                    else:
                        if output_ref[n, ho, wo, f] == output[n, ho, wo, f]:
                            continue

                    print("Input shape NHWC: ", Index(N, H, W, C))
                    print("filter shape RSCF: ", Index(R, S, C, F))
                    print("filter packed", filter_packed)
                    print("Test failed at index: ", Index(n, ho, wo, f))
                    print("Golden value: ", output_ref[n, ho, wo, f])
                    print("Actual value: ", output[n, ho, wo, f])
                    return

    # CHECK: Succeed
    print("Succeed")
    _ = output_ref_ptr^
    _ = output_ptr^
    _ = packed_filter_ptr^
    _ = filter_ptr^
    _ = input_ptr^


def main() raises:
    """It only includes shapes where F is multiple simd_size."""

    # likely partition in n_ho_wo or sequential
    test[input_type, filter_type, output_type, False](
        1,  # N
        6,  # H
        5,  # W
        1,  # C
        3,  # R
        4,  # S
        4,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        1,  # N
        12,  # H
        12,  # W
        12,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        1,  # N
        13,  # H
        13,  # W
        16,  # C
        5,  # R
        5,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        1,  # N
        7,  # H
        7,  # W
        32,  # C
        3,  # R
        3,  # S
        16,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        1,  # N
        17,  # H
        17,  # W
        16,  # C
        5,  # R
        5,  # S
        32,  # F
        Index(3, 3),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        5,  # N
        7,  # H
        7,  # W
        8,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    # likely partition in F or both
    test[input_type, filter_type, output_type, False](
        1,  # N
        7,  # H
        7,  # W
        7,  # C
        3,  # R
        3,  # S
        256,  # F
        Index(3, 3),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        1,  # N
        7,  # H
        7,  # W
        5,  # C
        5,  # R
        5,  # S
        288,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    # Pre-packed test
    # Avoid using dispatch functions for now because pre-packed version
    # has more restrictions for F.

    test[input_type, filter_type, output_type, True](
        1,  # N
        12,  # H
        12,  # W
        12,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        13,  # H
        13,  # W
        16,  # C
        5,  # R
        5,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        7,  # H
        7,  # W
        32,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        17,  # H
        17,  # W
        16,  # C
        5,  # R
        5,  # S
        64,  # F
        Index(3, 3),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        5,  # N
        12,  # H
        12,  # W
        8,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    # likely partition in F or both
    test[input_type, filter_type, output_type, True](
        1,  # N
        7,  # H
        7,  # W
        11,  # C
        3,  # R
        3,  # S
        192,  # F
        Index(3, 3),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        7,  # H
        7,  # W
        5,  # C
        5,  # R
        5,  # S
        256,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    # Top resnet shapes, all pre-packed

    # likely to partition C
    test[input_type, filter_type, output_type, True](
        1,  # N
        16,  # H
        16,  # W
        256,  # C
        3,  # R
        3,  # S
        256,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        58,  # H
        58,  # W
        64,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        30,  # H
        30,  # W
        128,  # C
        3,  # R
        3,  # S
        128,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        9,  # H
        9,  # W
        512,  # C
        3,  # R
        3,  # S
        512,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        230,  # H
        230,  # W
        3,  # C
        7,  # R
        7,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        58,  # H
        58,  # W
        128,  # C
        3,  # R
        3,  # S
        128,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        30,  # H
        30,  # W
        256,  # C
        3,  # R
        3,  # S
        256,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        16,  # H
        16,  # W
        512,  # C
        3,  # R
        3,  # S
        512,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        56,  # H
        56,  # W
        256,  # C
        3,  # R
        3,  # S
        512,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        14,  # H
        14,  # W
        1024,  # C
        3,  # R
        3,  # S
        2048,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        28,  # H
        28,  # W
        512,  # C
        3,  # R
        3,  # S
        1024,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    # Test with padding
    # This is a fallback implementation assuming all shapes are dynamic.

    test[input_type, filter_type, output_type, False](
        1,  # N
        56,  # H
        56,  # W
        64,  # C
        3,  # R
        3,  # S
        1024,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    # Test with padding
    # This is a fallback implementation assuming all shapes are dynamic.

    test[input_type, filter_type, output_type, False](
        1,  # N
        5,  # H
        5,  # W
        3,  # C
        3,  # R
        3,  # S
        1,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        2,  # N
        12,  # H
        11,  # W
        5,  # C
        4,  # R
        3,  # S
        2,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        1,  # N
        8,  # H
        12,  # W
        6,  # C
        2,  # R
        5,  # S
        3,  # F
        Index(1, 3),  # stride
        Index(1, 1),  # dilation
        Index(1, 0),  # pad_h
        Index(2, 2),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        1,  # N
        9,  # H
        7,  # W
        1,  # C
        5,  # R
        4,  # S
        3,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(2, 2),  # pad_h
        Index(2, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, False](
        1,  # N
        10,  # H
        5,  # W
        2,  # C
        4,  # R
        3,  # S
        6,  # F
        Index(3, 2),  # stride
        Index(1, 1),  # dilation
        Index(2, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        224,  # H
        224,  # W
        3,  # C
        7,  # R
        7,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(3, 3),  # pad_h
        Index(3, 3),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        56,  # H
        56,  # W
        64,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        56,  # H
        56,  # W
        128,  # C
        3,  # R
        3,  # S
        128,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        28,  # H
        28,  # W
        256,  # C
        3,  # R
        3,  # S
        256,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
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
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        14,  # H
        14,  # W
        3,  # C
        3,  # R
        3,  # S
        16,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        7,  # H
        7,  # W
        512,  # C
        3,  # R
        3,  # S
        512,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        19,  # N
        7,  # H
        7,  # W
        1,  # C
        3,  # R
        3,  # S
        16,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        13,  # N
        14,  # H
        14,  # W
        2,  # C
        3,  # R
        3,  # S
        32,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    # Test with F not multiple of simd_size

    test[input_type, filter_type, output_type, True](
        1,  # N
        5,  # H
        5,  # W
        2,  # C
        3,  # R
        3,  # S
        7,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        7,  # H
        7,  # W
        2,  # C
        3,  # R
        3,  # S
        42,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        23,  # H
        23,  # W
        17,  # C
        3,  # R
        3,  # S
        90,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        5,  # H
        11,  # W
        2,  # C
        3,  # R
        5,  # S
        7,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(2, 2),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        7,  # H
        9,  # W
        2,  # C
        3,  # R
        3,  # S
        42,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    )

    test[input_type, filter_type, output_type, True](
        1,  # N
        11,  # H
        7,  # W
        17,  # C
        3,  # R
        5,  # S
        90,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(2, 2),  # pad_w
    )
