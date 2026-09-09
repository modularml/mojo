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

from std.math import inf, nan

from std.builtin.simd import _convert_f32_to_float8_ue8m0
from max.gpu.host import DeviceContext
from std.memory import bitcast


# CHECK-LABEL: test_simd_f32_to_ue8m0
# CHECK: 2**-127
# CHECK: 2**-127
# CHECK: 2**-126
# CHECK: nan
# CHECK: nan
# CHECK: 2**2
# CHECK: 2**2
# CHECK: 2**2
# CHECK: 2**-127
# CHECK: 2**0
# CHECK: 2**2
# CHECK: 2**12
# CHECK: nan
# CHECK: 2**-127
# CHECK: 2**12
# CHECK: 2**2
# CHECK: 2**0
# CHECK: 2**127
def test_simd_f32_to_ue8m0():
    print("== test_simd_f32_to_ue8m0")

    comptime M = 32

    var f32_simd = SIMD[.float32, M](0.0)

    var i = 0
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x3FFFFF))
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x400000))
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x400001))
    i += 1
    f32_simd[i] = inf[.float32]()
    i += 1
    f32_simd[i] = nan[.float32]()
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x403FFFFF))
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x40400000))
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x40400001))
    i += 1
    f32_simd[i] = Float32(0.0)
    i += 1
    f32_simd[i] = Float32(1.0)
    i += 1
    f32_simd[i] = Float32(4.0)
    i += 1
    f32_simd[i] = Float32(4096.0)
    i += 1
    f32_simd[i] = -inf[.float32]()
    i += 1
    f32_simd[i] = Float32(-0.0)
    i += 1
    f32_simd[i] = Float32(-4096.0)
    i += 1
    f32_simd[i] = Float32(-4.0)
    i += 1
    f32_simd[i] = Float32(-1.0)
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x7F000000))
    i += 1

    var f32_casted_ue8m0 = _convert_f32_to_float8_ue8m0[.float8_e8m0fnu](
        f32_simd
    )

    for i in range(i):
        print(
            f32_casted_ue8m0[i],
        )


# CHECK-LABEL: test_simd_ue8m0_to_f32
# CHECK: [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
def test_simd_ue8m0_to_f32():
    print("== test_simd_ue8m0_to_f32")

    var f32_simd = SIMD[.float32, 8](
        1.1, 2.23, 4.34, 8.45, 16.56, 32.67, 64.78, 128.89
    )

    var ue8m0_simd = f32_simd.cast[.float8_e8m0fnu]()

    var ue8m0_casted_f32 = ue8m0_simd.cast[.float32]()
    print(ue8m0_casted_f32)


def test_simd_f32_to_ue8m0_ptx_kernel[
    size: Int,
    target: DType,
    idx: Int,
](x: SIMD[.float32, size]):
    var x_casted = _convert_f32_to_float8_ue8m0[target](x)

    for i in range(idx):
        print(
            x_casted[i],
        )


# CHECK-LABEL: test_simd_f32_to_ue8m0_ptx_path
# CHECK: 2**-127
# CHECK: 2**-127
# CHECK: 2**-126
# CHECK: nan
# CHECK: nan
# CHECK: 2**2
# CHECK: 2**2
# CHECK: 2**2
# CHECK: 2**-127
# CHECK: 2**0
# CHECK: 2**2
# CHECK: 2**12
# CHECK: nan
# CHECK: 2**-127
# CHECK: 2**12
# CHECK: 2**2
# CHECK: 2**0
# CHECK: 2**127
def test_simd_f32_to_ue8m0_ptx_path(ctx: DeviceContext) raises:
    print("== test_simd_f32_to_ue8m0_ptx_path")

    comptime M = 32

    var f32_simd = SIMD[.float32, M](0.0)

    var i = 0
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x3FFFFF))
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x400000))
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x400001))
    i += 1
    f32_simd[i] = inf[.float32]()
    i += 1
    f32_simd[i] = nan[.float32]()
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x403FFFFF))
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x40400000))
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x40400001))
    i += 1
    f32_simd[i] = Float32(0.0)
    i += 1
    f32_simd[i] = Float32(1.0)
    i += 1
    f32_simd[i] = Float32(4.0)
    i += 1
    f32_simd[i] = Float32(4096.0)
    i += 1
    f32_simd[i] = -inf[.float32]()
    i += 1
    f32_simd[i] = Float32(-0.0)
    i += 1
    f32_simd[i] = Float32(-4096.0)
    i += 1
    f32_simd[i] = Float32(-4.0)
    i += 1
    f32_simd[i] = Float32(-1.0)
    i += 1
    f32_simd[i] = bitcast[.float32, 1](UInt32(0x7F000000))
    i += 1

    comptime kernel = test_simd_f32_to_ue8m0_ptx_kernel[
        M, DType.float8_e8m0fnu, 18
    ]
    ctx.enqueue_function[kernel](f32_simd, grid_dim=1, block_dim=1)
    ctx.synchronize()


def test_simd_ue8m0_to_f32_ptx_kernel[
    size: Int,
    target: DType,
](x: SIMD[.float8_e8m0fnu, size]):
    var x_casted_to_fp32 = x.cast[target]()
    print(x_casted_to_fp32)


# CHECK-LABEL: test_simd_ue8m0_to_f32_ptx_path
# CHECK: [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
def test_simd_ue8m0_to_f32_ptx_path(ctx: DeviceContext) raises:
    print("== test_simd_ue8m0_to_f32_ptx_path")

    var f32_simd = SIMD[.float32, 8](
        1.1, 2.23, 4.34, 8.45, 16.56, 32.67, 64.78, 128.89
    )

    var ue8m0_simd = f32_simd.cast[.float8_e8m0fnu]()

    comptime kernel = test_simd_ue8m0_to_f32_ptx_kernel[8, .float32]
    ctx.enqueue_function[kernel](ue8m0_simd, grid_dim=1, block_dim=1)
    ctx.synchronize()


def main() raises:
    test_simd_f32_to_ue8m0()
    test_simd_ue8m0_to_f32()

    with DeviceContext() as ctx:
        test_simd_f32_to_ue8m0_ptx_path(ctx)
        test_simd_ue8m0_to_f32_ptx_path(ctx)
