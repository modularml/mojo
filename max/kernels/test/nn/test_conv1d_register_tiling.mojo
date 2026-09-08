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

from std.sys import simd_width_of

from layout import Coord, TileTensor, Idx, row_major
from nn.conv.conv import conv1d_update_wo_tile
from nn.conv.conv_utils import ConvShape
from std.testing import assert_equal

from std.utils.index import Index

comptime type = DType.float32
comptime micro_kernel_height = 2
comptime micro_kernel_width = 2
comptime simd_size = simd_width_of[type]()
comptime micro_kernel_f_size = micro_kernel_width * simd_size

comptime N = 1
comptime H = 1
comptime W = 14
comptime C = 2 * simd_size
comptime R = 1
comptime S = 3
comptime F = 2 * micro_kernel_f_size
comptime stride_h = 1
comptime stride_w = 1
comptime pad_left = 1
comptime pad_right = 1
comptime pad_top = 0
comptime pad_bottom = 0
comptime dilation_h = 1
comptime dilation_w = 1
# alias HO = (H + pad_top + pad_bottom - dilation_h * (R - 1) - 1) // stride_h + 1
comptime HO = 1
comptime WO = (
    W + pad_left + pad_right - dilation_w * (S - 1) - 1
) // stride_w + 1
comptime num_micro_tile = F // micro_kernel_f_size

comptime output_shape = row_major[N, WO, F]()
comptime input_shape = row_major[N, W, C]()
comptime filter_shape = row_major[num_micro_tile, S, C, micro_kernel_f_size]()


def conv1d_register_tiling(
    output: MutPointer[Scalar[type], _],
    input: ImmPointer[Scalar[type], _],
    filter: ImmPointer[Scalar[type], _],
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    wo: Int,
) abi("C"):
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
        num_groups=1,
    )

    conv1d_update_wo_tile[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        filter_packed=True,
        effected_by_padding=False,
        has_residual=False,
        last_c_tile=False,
    ](
        output,
        input,
        filter,
        True,
        c_tile_size,
        f_tile_offset,
        f_tile_size,
        conv_shape,
        0,
        wo,
    )


def test_conv1d_register_tiling() raises:
    var output_stack = Array[Scalar[type], Int(output_shape.product())](
        fill=0.0
    )
    var output = TileTensor(output_stack, output_shape)
    var input_stack = Array[Scalar[type], Int(input_shape.product())](fill=1.0)
    var input = TileTensor(input_stack, input_shape)
    var filter_stack = Array[Scalar[type], Int(filter_shape.product())](
        fill=1.0
    )
    var filter = TileTensor(filter_stack, filter_shape)

    var c_tile_offset = 0
    var c_tile_size = C
    var f_tile_offset = F // 2
    var f_tile_size = F // 2
    var wo = 2
    var w = wo * stride_w - pad_left

    # FRSCf
    var filter_ptr = filter._storage + f_tile_offset * R * S * C
    # NHWC
    var input_ptr = input._storage + c_tile_offset + C * w
    var output_ptr = output._storage + f_tile_offset + F * (wo)

    conv1d_register_tiling(
        output_ptr,
        input_ptr,
        filter_ptr,
        c_tile_size,
        f_tile_offset,
        f_tile_size,
        wo,
    )

    var actual = output.load[width=simd_size]((Idx[0], wo, f_tile_size))
    var expect = SIMD[type, simd_size](R * S * c_tile_size)
    assert_equal(expect, actual)

    actual = output.load[width=simd_size](
        (Idx[0], wo + micro_kernel_height - 1, f_tile_size)
    )

    assert_equal(expect, actual)

    actual = output.load[width=simd_size](
        (Idx[0], wo + micro_kernel_height, f_tile_size)
    )

    assert_equal(SIMD[type, simd_size](0), actual)


def main() raises:
    test_conv1d_register_tiling()
