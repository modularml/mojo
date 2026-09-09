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
from std.testing import assert_true, TestSuite

comptime A100_TARGET = get_gpu_target["sm_80"]()
comptime MI355X_TARGET = get_gpu_target["mi355x"]()


def test_abs() raises:
    def do_abs[
        dtype: DType, *, width: Int = 1
    ](val: SIMD[dtype, width]) -> type_of(val):
        return abs(val)

    # AMD GPU kernels cannot have a return value
    def do_abs_noreturn[
        dtype: DType, *, width: Int = 1
    ](val: SIMD[dtype, width], x: Pointer[mut=True, Scalar[dtype], _]):
        x.unsafe_store(0, abs(val))

    # Check the NVIDIA PTX.
    assert_true(
        "abs.f16 " in _compile_code[do_abs[.float16], target=A100_TARGET]()
    )
    assert_true(
        "abs.bf16 " in _compile_code[do_abs[.bfloat16], target=A100_TARGET]()
    )

    assert_true(
        "abs.f16x2 "
        in _compile_code[do_abs[.float16, width=2], target=A100_TARGET]()
    )
    assert_true(
        "abs.bf16x2 "
        in _compile_code[do_abs[.bfloat16, width=2], target=A100_TARGET]()
    )

    # Check the AMD CDNA assembly.

    # Set the sign bit to zero.
    assert_true(
        "s_and_b32 s0, s4, 0x7fffffff"
        in _compile_code[do_abs_noreturn[.float32], target=MI355X_TARGET]()
    )

    # Mask out the lower half sign bit.
    assert_true(
        "s_and_b32 s0, s4, 0x7fff"
        in _compile_code[
            do_abs_noreturn[.float16, width=1], target=MI355X_TARGET
        ]()
    )
    # Mask out the lower and upper half sign bit
    assert_true(
        "s_and_b32 s0, s4, 0x7fff7fff"
        in _compile_code[
            do_abs_noreturn[.float16, width=2], target=MI355X_TARGET
        ]()
    )
    # Mask out the sign bit.
    assert_true(
        "s_and_b32 s0, s4, 0x7fff"
        in _compile_code[
            do_abs_noreturn[.bfloat16, width=1], target=MI355X_TARGET
        ]()
    )
    # Mask out the lower and upper half sign bit.
    assert_true(
        "s_and_b32 s0, s4, 0x7fff7fff"
        in _compile_code[
            do_abs_noreturn[.bfloat16, width=2], target=MI355X_TARGET
        ]()
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
