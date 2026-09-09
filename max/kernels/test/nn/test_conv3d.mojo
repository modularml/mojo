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

from layout import Coord, Layout, LayoutTensor, RuntimeLayout
from layout import lt_to_tt
from nn.conv.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    pack_conv_filter_shape,
    pack_filter,
)
from nn.conv.conv_utils import (
    ConvShape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)

from std.utils.index import Index, IndexList

comptime simd_size: Int = simd_width_of[DType.float32]()


# CHECK-LABEL: test_conv3d
def test[
    dtype: DType, filter_packed: Bool
](
    N: Int,
    DHW: IndexList[3],
    C: Int,
    QRS: IndexList[3],
    F: Int,
    stride: IndexList[3],
    dilation: IndexList[3],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    num_groups: Int,
) raises:
    print("test_conv3d: Testing 3D Convolution")

    var D = DHW[0]
    var H = DHW[1]
    var W = DHW[2]

    var Q = QRS[0]
    var R = QRS[1]
    var S = QRS[2]

    # fmt: off
    var DO = (D + pad_d[0] + pad_d[1] - dilation[0] * (Q - 1) - 1) // stride[0] + 1
    var HO = (H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1) // stride[1] + 1
    var WO = (W + pad_w[0] + pad_w[1] - dilation[2] * (S - 1) - 1) // stride[2] + 1
    # fmt: on

    var conv_shape = ConvShape[3](
        n=N,
        input_dims=Coord(DHW),
        output_dims=Coord(Index(DO, HO, WO)),
        filter_dims=Coord(QRS),
        c=C,
        f=F,
        stride=Coord(stride),
        dilation=Coord(dilation),
        pad_d=Coord(pad_d),
        pad_h=Coord(pad_h),
        pad_w=Coord(pad_w),
        num_groups=num_groups,
    )

    var C_per_group = C // num_groups

    var input_size = N * D * H * W * C
    var filter_size = Q * R * S * C_per_group * F
    var output_size = N * DO * HO * WO * F

    var input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    var filter_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    var output_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var output_ref_ptr = List(length=output_size, fill=Scalar[dtype](0))

    rand(input_ptr)
    rand(filter_ptr)

    # Find the tile size used in packing.
    comptime micro_kernel_height = get_direct_conv_micro_kernel_height()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()

    comptime layout_5d = Layout.row_major[5]()
    comptime layout_6d = Layout.row_major[6]()

    # Buffers for direct conv.
    var input = LayoutTensor[dtype, layout_5d](
        input_ptr, RuntimeLayout[layout_5d].row_major(Index(N, D, H, W, C))
    )
    var filter = LayoutTensor[dtype, layout_5d](
        filter_ptr,
        RuntimeLayout[layout_5d].row_major(Index(Q, R, S, C_per_group, F)),
    )
    var packed_filter_shape = pack_conv_filter_shape(
        lt_to_tt(filter), num_groups
    )

    var packed_filter_ptr = List(
        length=packed_filter_shape.flattened_length(), fill=Scalar[dtype](0)
    )
    var packed_filter = LayoutTensor[dtype, layout_6d](
        packed_filter_ptr,
        RuntimeLayout[layout_6d].row_major(packed_filter_shape),
    )
    var output = LayoutTensor[dtype, layout_5d](
        output_ptr, RuntimeLayout[layout_5d].row_major(Index(N, DO, HO, WO, F))
    )

    comptime if filter_packed:
        pack_filter(lt_to_tt(filter), lt_to_tt(packed_filter), num_groups)

    # Reference: naive conv
    Naive2dConvolution[
        dtype,
        dtype,
        dtype,
    ].run(
        output_ref_ptr.unsafe_ptr(),
        input_ptr.unsafe_ptr(),
        filter_ptr.unsafe_ptr(),
        Index(N, DO, HO, WO, F),
        Index(N, D, H, W, C),
        Index(Q, R, S, C // num_groups, F),
        pad_d,
        pad_h,
        pad_w,
        stride,
        dilation,
        num_groups,
    )

    # Test direct conv
    comptime conv_attr = ConvInfoStatic[3]()

    comptime if filter_packed:
        ConvDirectNHWC[
            layout_5d,
            layout_6d,
            layout_5d,
            dtype,
            dtype,
            dtype,
            True,
            conv_attr,
        ].run(output, input, packed_filter, conv_shape)
    else:
        ConvDirectNHWC[
            layout_5d,
            layout_5d,
            layout_5d,
            dtype,
            dtype,
            dtype,
            False,
            conv_attr,
        ].run(output, input, filter, conv_shape)

    # Check results, return on the first failed comparison.
    var total_elements = output_size
    var matching_elements = 0
    var max_diff = 0.0
    var max_diff_idx = -1

    var idx = 0
    for _ in range(N):
        for _ in range(DO):
            for _ in range(HO):
                for _ in range(WO):
                    for _ in range(F):
                        var gpu_val = Float64(output_ptr[idx])
                        var ref_val = Float64(output_ref_ptr[idx])
                        var diff = abs(gpu_val - ref_val)

                        if diff > max_diff:
                            max_diff = diff
                            max_diff_idx = idx

                        if isclose(gpu_val, ref_val, atol=1e-4, rtol=1e-4):
                            matching_elements += 1

                        idx += 1

    _ = max_diff_idx  # silence warning

    var match_percentage = (
        Float64(matching_elements) / Float64(total_elements) * 100.0
    )

    if matching_elements == total_elements:
        print("RESULT: PASS - All elements match within tolerance")
    else:
        print(
            "RESULT: FAIL - Elements do not match with a max percentage of ",
            match_percentage,
        )

    if matching_elements == total_elements:
        print("Succeed")
    _ = output_ref_ptr^
    _ = output_ptr^
    _ = packed_filter_ptr^
    _ = filter_ptr^
    _ = input_ptr^


def main() raises:
    comptime dtype = DType.float32

    test[.float32, False](  # dtype, filter_packed
        1,  # N: batch size
        IndexList[3](4, 4, 4),  # DHW: depth, height, width
        2,  # C: channels
        IndexList[3](2, 2, 2),  # QRS: depth filter, height filter, width filter
        3,  # F: out channels
        IndexList[3](1, 1, 1),  # stride
        IndexList[3](1, 1, 1),  # dilation
        IndexList[2](0, 0),  # pad_d: padding for depth dimension
        IndexList[2](0, 0),  # pad_h: padding for height dimension
        IndexList[2](0, 0),  # pad_w: padding for width dimension
        1,  # num_groups
    )

    test[dtype, False](
        1,  # batch size
        Index(2, 4, 5),  # input shape
        4,  # C
        Index(1, 2, 3),  # filter shape
        3,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(0, 0),  # pad_d
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )
    test[dtype, False](
        1,  # batch size
        Index(9, 8, 5),  # input shape
        1,  # C
        Index(2, 2, 3),  # filter shape
        32,  # F
        Index(1, 3, 2),  # stride
        Index(1, 1, 1),  # dilation
        Index(0, 0),  # pad_d
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # w/o packing, w/ padding.
    test[dtype, False](
        1,  # batch size
        Index(5, 7, 6),  # input shape
        7,  # C
        Index(3, 4, 3),  # filter shape
        24,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )
    test[dtype, False](
        1,  # batch size
        Index(10, 11, 6),  # input shape
        2,  # C
        Index(3, 4, 3),  # filter shape
        31,  # F
        Index(2, 3, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(2, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    # w/ packing, w/o padding.
    test[dtype, True](
        1,  # batch size
        Index(11, 13, 17),  # input shape
        9,  # C
        Index(7, 5, 3),  # filter shape
        3,  # F
        Index(1, 2, 4),  # stride
        Index(1, 1, 1),  # dilation
        Index(0, 0),  # pad_d
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )
    test[dtype, True](
        1,  # batch size
        Index(13, 9, 7),  # input shape
        4,  # C
        Index(4, 7, 3),  # filter shape
        17,  # F
        Index(2, 2, 2),  # stride
        Index(1, 1, 1),  # dilation
        Index(0, 0),  # pad_d
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # w/ packing, w/ padding.
    test[dtype, True](
        1,  # batch size
        Index(5, 5, 5),  # input shape
        4,  # C
        Index(3, 3, 3),  # filter shape
        64,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )
    test[dtype, True](
        1,  # batch size
        Index(11, 9, 14),  # input shape
        4,  # C
        Index(4, 7, 3),  # filter shape
        3,  # F
        Index(2, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(2, 1),  # pad_d
        Index(3, 3),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    # 3D-UNet shapes.
    # Leave large shapes in comments to save time for CI.
    test[dtype, True](
        1,  # batch size
        Index(8, 8, 8),  # input shape
        320,  # C
        Index(3, 3, 3),  # filter shape
        320,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[dtype, True](
        1,  # batch size
        Index(8, 8, 8),  # input shape
        320,  # C
        Index(3, 3, 3),  # filter shape
        320,  # F
        Index(2, 2, 2),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[dtype, True](
        1,  # batch size
        Index(4, 4, 4),  # input shape
        320,  # C
        Index(3, 3, 3),  # filter shape
        320,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[dtype, True](
        1,  # batch size
        Index(8, 8, 8),  # input shape
        640,  # C
        Index(3, 3, 3),  # filter shape
        320,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     1,  # C
    #     Index(3, 3, 3),  # filter shape
    #     32,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     32,  # C
    #     Index(3, 3, 3),  # filter shape
    #     32,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     32,  # C
    #     Index(3, 3, 3),  # filter shape
    #     64,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(64, 64, 64),  # input shape
    #     64,  # C
    #     Index(3, 3, 3),  # filter shape
    #     128,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(32, 32, 32),  # input shape
    #     128,  # C
    #     Index(3, 3, 3),  # filter shape
    #     128,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(32, 32, 32),  # input shape
    #     128,  # C
    #     Index(3, 3, 3),  # filter shape
    #     256,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(16, 16, 16),  # input shape
    #     256,  # C
    #     Index(3, 3, 3),  # filter shape
    #     256,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(16, 16, 16),  # input shape
    #     256,  # C
    #     Index(3, 3, 3),  # filter shape
    #     320,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(16, 16, 16),  # input shape
    #     512,  # C
    #     Index(3, 3, 3),  # filter shape
    #     256,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(16, 16, 16),  # input shape
    #     256,  # C
    #     Index(3, 3, 3),  # filter shape
    #     256,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(32, 32, 32),  # input shape
    #     256,  # C
    #     Index(3, 3, 3),  # filter shape
    #     128,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(32, 32, 32),  # input shape
    #     128,  # C
    #     Index(3, 3, 3),  # filter shape
    #     128,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(64, 64, 64),  # input shape
    #     128,  # C
    #     Index(3, 3, 3),  # filter shape
    #     64,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(64, 64, 64),  # input shape
    #     64,  # C
    #     Index(3, 3, 3),  # filter shape
    #     64,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     64,  # C
    #     Index(3, 3, 3),  # filter shape
    #     32,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[dtype, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     32,  # C
    #     Index(3, 3, 3),  # filter shape
    #     3,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(0, 0),  # pad_d
    #     Index(0, 0),  # pad_h
    #     Index(0, 0),  # pad_w
    #     1,  # num_groups
    # )
