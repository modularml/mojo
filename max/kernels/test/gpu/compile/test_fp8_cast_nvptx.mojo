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
"""Verify FP8 E4M3 → FP16 casts emit paired cvt.rn.f16x2.e4m3x2 on SM100."""

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import *


def fp8_e4m3_to_f16_2(
    a: SIMD[.float8_e4m3fn, 2],
) -> SIMD[.float16, 2]:
    return a.cast[.float16]()


def fp8_e4m3_to_f16_8(
    a: SIMD[.float8_e4m3fn, 8],
) -> SIMD[.float16, 8]:
    return a.cast[.float16]()


def fp8_e4m3_to_f32_8(
    a: SIMD[.float8_e4m3fn, 8],
) -> SIMD[.float32, 8]:
    return a.cast[.float32]()


def test_fp8_e4m3_to_f16_paired_cvt() raises:
    """FP8 E4M3 → FP16 must generate the paired cvt.rn.f16x2.e4m3x2
    instruction on SM100, which converts two packed E4M3 values to two
    FP16 values in one op and avoids per-element PRMT byte shuffles.
    """
    comptime target = get_gpu_target["sm_100"]()

    var asm_2 = _compile_code[fp8_e4m3_to_f16_2, target=target]()
    assert_true(
        "cvt.rn.f16x2.e4m3x2" in asm_2,
        "width=2: expected cvt.rn.f16x2.e4m3x2 for paired FP8→FP16 conversion",
    )

    var asm_8 = _compile_code[fp8_e4m3_to_f16_8, target=target]()
    assert_true(
        "cvt.rn.f16x2.e4m3x2" in asm_8,
        "width=8: expected cvt.rn.f16x2.e4m3x2 for paired FP8→FP16 conversion",
    )

    var asm_f32_8 = _compile_code[fp8_e4m3_to_f32_8, target=target]()
    assert_true(
        "cvt.rn.f16x2.e4m3x2" in asm_f32_8,
        "width=8: expected cvt.rn.f16x2.e4m3x2 for paired FP8→FP16 conversion",
    )


def main() raises:
    test_fp8_e4m3_to_f16_paired_cvt()
