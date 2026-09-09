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


from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.compute.mma import mma
from std.testing import *


def SM80_16x8x8_F16F16F16F16_TN(
    a: SIMD[.float16, 4],
    b: SIMD[.float16, 2],
    c: SIMD[.float16, 4],
) -> SIMD[.float16, 4]:
    var d = SIMD[.float16, 4]()
    mma(d, a, b, c)
    return d


def test_SM80_16x8x8_F16F16F16F16_TN() raises:
    var asm = _compile_code[SM80_16x8x8_F16F16F16F16_TN]().asm
    assert_true("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16" in asm)
    assert_true("{%r6, %r7}," in asm)
    assert_true("{%r1, %r2}," in asm)
    assert_true("{%r3}," in asm)
    assert_true("{%r4, %r5};" in asm)


def SM80_m16n8k4_F32TF32TF32F32_TN(
    a: SIMD[.float32, 2],
    b: Float32,
    c: SIMD[.float32, 4],
) -> SIMD[.float32, 4]:
    var d = SIMD[.float32, 4]()
    mma(d, a, b, c)
    return d


def test_SM80_m16n8k4_F32TF32TF32F32_TN() raises:
    var asm = _compile_code[SM80_m16n8k4_F32TF32TF32F32_TN]().asm
    assert_true("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32" in asm)
    assert_true("{%r8, %r9, %r10, %r11}," in asm)
    assert_true("{%r1, %r2}," in asm)
    assert_true("{%r7}," in asm)
    assert_true("{%r3, %r4, %r5, %r6};" in asm)


def SM80_m16n8k8_F32TF32TF32F32_TN(
    a: SIMD[.float32, 4],
    b: Float32,
    c: SIMD[.float32, 4],
) -> SIMD[.float32, 4]:
    var d = SIMD[.float32, 4]()
    mma(d, a, b.join(b), c)
    return d


def test_SM80_m16n8k8_F32TF32TF32F32_TN() raises:
    var asm = _compile_code[SM80_m16n8k8_F32TF32TF32F32_TN]().asm
    assert_true("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32" in asm)
    assert_true("{%r10, %r11, %r12, %r13}," in asm)
    assert_true("{%r1, %r2, %r3, %r4}," in asm)
    assert_true("{%r9, %r9}," in asm)
    assert_true("{%r5, %r6, %r7, %r8};" in asm)


def SM80_m16n8k32_F8E4M3F8E4M_TN(
    a: SIMD[.float8_e4m3fn, 16],
    b: SIMD[.float8_e4m3fn, 8],
    c: SIMD[.float32, 4],
) -> SIMD[.float32, 4]:
    var d = SIMD[.float32, 4]()
    mma(d, a, b, c)
    return d


def test_SM80_m16n8k8_F8E4M3F8E4M3_TN() raises:
    var asm = _compile_code[
        SM80_m16n8k32_F8E4M3F8E4M_TN, target=get_gpu_target["sm_90"]()
    ]().asm
    assert_true("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32" in asm)
    assert_true("{%r1, %r2, %r3, %r4}" in asm)
    assert_true("{%r5, %r6, %r7, %r8}" in asm)
    assert_true("{%r9, %r10}, {%r11, %r12, %r13, %r14}" in asm)


def main() raises:
    test_SM80_16x8x8_F16F16F16F16_TN()
    test_SM80_m16n8k4_F32TF32TF32F32_TN()
    test_SM80_m16n8k8_F32TF32TF32F32_TN()
    test_SM80_m16n8k8_F8E4M3F8E4M3_TN()
