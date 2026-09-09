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

# https://github.com/PaddlePaddle/Paddle/blob/3862f8303d2723c03ffb42ce332d4c570906669f/paddle/phi/kernels/funcs/weight_only_gemv.cu#L795

# logic and shift instruction: lop3
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3

from std.sys.info import is_amd_gpu, is_apple_gpu

from max.gpu.host import DeviceContext
from max.gpu.intrinsics import lop
from std.memory.unsafe import bitcast
from std.testing import assert_equal
from layout import TileTensor, row_major


# 8xint4 -> 8xbfloat16 interleaved conversion
def int4tobf16[no_lop: Bool = False](i4: Int32) -> SIMD[.bfloat16, 8]:
    comptime MASK: Int32 = 0x000F000F
    comptime I4s_TO_BF16s_MAGIC_NUM: Int32 = 0x43004300

    # 0xc308 = -136.0, 0xc300 = -128.0
    comptime BF16_BIAS = SIMD[.bfloat16, 2](-128, -128)
    # 0x3f80 = 1.0
    comptime BF16_ONE = SIMD[.bfloat16, 2](1, 1)

    var i4s: Int32 = i4
    var v: SIMD[.int32, 4] = 0
    comptime lut: Int32 = (0xF0 & 0xCC) | 0xAA
    # This lut is operation: (A & B) | C

    comptime for i in range(0, 4):
        # The ternary operator isnot working.
        # The conditional is_amd_gpu() or no_lop appears to not be constant
        # var t = (i4s & MASK) | I4s_TO_BF16s_MAGIC_NUM if (is_amd_gpu() or no_lop) else lop[
        #    lut
        # ](i4s, MASK, I4s_TO_BF16s_MAGIC_NUM)
        var t: Int32

        comptime if is_apple_gpu() or is_amd_gpu() or no_lop:
            t = (i4s & MASK) | I4s_TO_BF16s_MAGIC_NUM
        else:
            t = lop[lut](i4s, MASK, I4s_TO_BF16s_MAGIC_NUM)

        v[i] = bitcast[.int32, 1](
            bitcast[.bfloat16, 2](t).fma(BF16_ONE, BF16_BIAS)
        )
        i4s >>= 4
    return bitcast[.bfloat16, 8](v)


def call_int4tobf16[
    no_lop: Bool
](i4: Int32, out_ptr: MutPointer[BFloat16, MutAnyOrigin],):
    var v = int4tobf16[no_lop](i4)
    out_ptr.bitcast[Int32]().store[alignment=16](0, bitcast[.int32, 4](v))


def test_int4tobfloat16[no_lop: Bool](ctx: DeviceContext) raises:
    var stack = Array[BFloat16, 8](fill={})
    var out_host = TileTensor(stack, row_major[8]())
    var out_device = ctx.enqueue_create_buffer[.bfloat16](8)

    comptime kernel = call_int4tobf16[no_lop]
    ctx.enqueue_function[kernel](
        Int32(0x76543210), out_device, grid_dim=1, block_dim=1
    )

    ctx.enqueue_copy(out_host._storage, out_device)
    ctx.synchronize()
    for i in range(4):
        assert_equal(out_host[2 * i + 0], BFloat16(i + 0))
        assert_equal(out_host[2 * i + 1], BFloat16(i + 4))


def main() raises:
    with DeviceContext() as ctx:
        test_int4tobfloat16[no_lop=False](ctx)
        test_int4tobfloat16[no_lop=True](ctx)
