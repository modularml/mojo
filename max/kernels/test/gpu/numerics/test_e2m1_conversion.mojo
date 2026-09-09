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

from linalg.fp4_utils import (
    cast_fp32_to_fp4e2m1,
    E2M1_TO_FLOAT32,
    cast_fp_to_fp4e2m1,
    cast_f4e2m1x2_to_fp16x2,
)
from max.gpu.host import DeviceContext
from std.math import nan, inf
from std.sys import bit_width_of


# CHECK-LABEL: test_simd_f32_to_e2m1
# CHECK: [0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0]
# CHECK: [2.0, 3.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0]
# CHECK: [0.0, -0.0, -0.5, -1.0, -1.0, -1.5, -2.0, -2.0]
# CHECK: [-2.0, -3.0, -4.0, -4.0, -4.0, -6.0, 6.0, -6.0]
def test_simd_f32_to_e2m1():
    print("== test_simd_f32_to_e2m1")

    comptime size = 32
    var f32_simd = SIMD[.float32, size](
        0.0,
        0.23,
        0.26,
        0.76,
        1.25,
        1.26,
        1.75,
        1.76,
        2.5,
        2.51,
        3.5,
        3.51,
        5.0,
        5.01,
        nan[.float32](),
        inf[.float32](),
        -0.0,
        -0.23,
        -0.26,
        -0.76,
        -1.25,
        -1.26,
        -1.75,
        -1.76,
        -2.5,
        -2.51,
        -3.5,
        -3.51,
        -5.0,
        -5.01,
        -nan[.float32](),
        -inf[.float32](),
    )

    var e2m1_simd = cast_fp_to_fp4e2m1(f32_simd)

    comptime for iter in range(size // 8):
        var x_slice = e2m1_simd.slice[8, offset=iter * 8]()
        print(
            x_slice,
        )


def test_simd_f32_to_e2m1_ptx_kernel[
    size: Int,
](x: SIMD[.float32, size]):
    comptime FP4_E2M1_WIDTH = 4
    comptime FP4_E2M1_MASK = pow(2, FP4_E2M1_WIDTH) - 1

    comptime for iter in range(size // 8):
        var x_slice = x.slice[8, offset=iter * 8]()
        var x_casted = cast_fp32_to_fp4e2m1(x_slice)

        comptime for shift in range(
            0, bit_width_of[DType.uint32](), FP4_E2M1_WIDTH
        ):
            comptime BitsType = type_of(x_casted.to_bits())
            var x = (x_casted.to_bits() >> BitsType(shift)) & BitsType(
                FP4_E2M1_MASK
            )
            print(E2M1_TO_FLOAT32[Int(x)], end=" ")
        print("")


# CHECK-LABEL: test_simd_f32_to_e2m1_ptx_path
# CHECK: 0.0 0.0 0.5 1.0 1.0 1.5 2.0 2.0
# CHECK: 2.0 3.0 4.0 4.0 4.0 6.0 6.0 6.0
# CHECK: -0.0 -0.0 -0.5 -1.0 -1.0 -1.5 -2.0 -2.0
# CHECK: -2.0 -3.0 -4.0 -4.0 -4.0 -6.0 6.0 -6.0
def test_simd_f32_to_e2m1_ptx_path(ctx: DeviceContext) raises:
    print("== test_simd_f32_to_e2m1_ptx_path")

    comptime size = 32
    var f32_simd = SIMD[.float32, size](
        0.0,
        0.23,
        0.26,
        0.76,
        1.25,
        1.26,
        1.75,
        1.76,
        2.5,
        2.51,
        3.5,
        3.51,
        5.0,
        5.01,
        nan[.float32](),
        inf[.float32](),
        -0.0,
        -0.23,
        -0.26,
        -0.76,
        -1.25,
        -1.26,
        -1.75,
        -1.76,
        -2.5,
        -2.51,
        -3.5,
        -3.51,
        -5.0,
        -5.01,
        -nan[.float32](),
        -inf[.float32](),
    )

    comptime kernel = test_simd_f32_to_e2m1_ptx_kernel[size,]
    ctx.enqueue_function[kernel](f32_simd, grid_dim=1, block_dim=1)
    ctx.synchronize()


def test_simd_f4e2m1x2_to_fp16x2_ptx_kernel[
    size: Int,
](x: SIMD[.uint8, size]):
    comptime for i in range(size // 4):
        for j in range(4):
            var x_casted = cast_f4e2m1x2_to_fp16x2(x[i * 4 + j])
            print(x_casted, end=" ")
        print("")


# CHECK-LABEL: test_simd_f4e2m1x2_to_fp16x2
# CHECK: [0.0, 0.0] [0.5, 0.0] [0.0, 0.5] [0.5, 0.5]
# CHECK: [4.0, -1.0] [1.0, 0.5] [-0.0, 1.5] [6.0, 1.0]
# CHECK: [0.0, 4.0] [1.5, -1.5] [0.5, 1.5] [1.5, 1.5]
# CHECK: [-6.0, 0.0] [3.0, 0.5] [-2.0, 2.0] [-4.0, 2.0]
def test_simd_f4e2m1x2_to_fp16x2(ctx: DeviceContext) raises:
    print("== test_simd_f4e2m1x2_to_fp16x2")

    comptime size = 16
    var e4m21_simd = SIMD[.uint8, size](
        0x00,
        0x01,
        0x10,
        0x11,
        0xA6,
        0x12,
        0x38,
        0x27,
        0x60,
        0xB3,
        0x31,
        0x33,
        0x0F,
        0x15,
        0x4C,
        0x4E,
    )

    comptime kernel = test_simd_f4e2m1x2_to_fp16x2_ptx_kernel[size,]
    ctx.enqueue_function[kernel](e4m21_simd, grid_dim=1, block_dim=1)
    ctx.synchronize()


def main() raises:
    test_simd_f32_to_e2m1()

    with DeviceContext() as ctx:
        test_simd_f32_to_e2m1_ptx_path(ctx)
        test_simd_f4e2m1x2_to_fp16x2(ctx)
