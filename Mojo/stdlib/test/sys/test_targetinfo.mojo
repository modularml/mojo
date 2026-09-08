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

from std.sys import (
    CompilationTarget,
    align_of,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    size_of,
)
from std.sys.info import stdlib_plugin

from std.testing import assert_equal, assert_false, assert_true
from std.testing import TestSuite


comptime _target_default_plugin = __mlir_attr[
    `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
    `arch = "sm_80", `,
    `features = "+ptx81"`,
    `> : !kgen.target`,
]

comptime _target_named_plugin = __mlir_attr[
    `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
    `arch = "sm_80", `,
    `stdlib_plugin = "cuda", `,
    `features = "+ptx81"`,
    `> : !kgen.target`,
]


def test_size_of() raises:
    assert_equal(size_of[__mlir_type.i16](), 2)

    assert_equal(size_of[__mlir_type.ui16](), 2)

    assert_equal(size_of[DType.int16](), 2)

    assert_equal(size_of[DType.uint16](), 2)

    assert_equal(size_of[SIMD[.int16, 2]](), 4)


def test_align_of() raises:
    assert_true(align_of[__mlir_type.i16]() > 0)

    assert_true(align_of[__mlir_type.ui16]() > 0)

    assert_true(align_of[DType.int16]() > 0)

    assert_true(align_of[DType.uint16]() > 0)

    assert_true(align_of[SIMD[.int16, 2]]() > 0)


def test_cores() raises:
    assert_true(num_logical_cores() > 0)
    assert_true(num_physical_cores() > 0)
    assert_true(num_performance_cores() > 0)


def test_target_is_apple() raises:
    # Consistency: ensure predicates exist + returns Bool
    var _is_apple_mx: Bool = CompilationTarget.is_apple_m1()
    _is_apple_mx |= CompilationTarget.is_apple_m2()
    _is_apple_mx |= CompilationTarget.is_apple_m3()
    _is_apple_mx |= CompilationTarget.is_apple_m4()
    _is_apple_mx |= CompilationTarget.is_apple_m5()

    # Invariant: M series implies apple silicon
    if _is_apple_mx:
        assert_true(
            CompilationTarget.is_apple_silicon(),
            "Target is Apple M-Series but is_apple_silicon() returned False",
        )


def test_target_arch() raises:
    # This runs on the host, which is always x86 or ARM, and the two are
    # mutually exclusive. RISC-V is a cross-compilation target only, so
    # `test_arch_predicates.mojo` covers it.
    assert_true(
        CompilationTarget.is_x86() != CompilationTarget.is_arm(),
        "exactly one of is_x86() and is_arm() must hold",
    )
    assert_false(CompilationTarget.is_riscv())

    # Neon implies ARM, but not the converse: 32-bit ARM can lack Neon.
    if CompilationTarget.has_neon():
        assert_true(
            CompilationTarget.is_arm(),
            "Target has Neon but is_arm() returned False",
        )

    if CompilationTarget.has_sse4():
        assert_true(
            CompilationTarget.is_x86(),
            "Target has SSE4 but is_x86() returned False",
        )


def test_target_has_feature() raises:
    # Ensures target feature check functions exist and return a boolable value.
    var _has_feature: Bool = CompilationTarget.has_avx()
    _has_feature = CompilationTarget.has_avx2()
    _has_feature = CompilationTarget.has_avx512f()
    _has_feature = CompilationTarget.has_fma()
    _has_feature = CompilationTarget.has_intel_amx()
    _has_feature = CompilationTarget.has_neon()
    _has_feature = CompilationTarget.has_neon_int8_dotprod()
    _has_feature = CompilationTarget.has_neon_int8_matmul()
    _has_feature = CompilationTarget.has_sse4()
    _has_feature = CompilationTarget.has_vnni()


def test_stdlib_plugin() raises:
    # The plugin name defaults to "default" when omitted from the target
    # attribute.
    assert_equal(stdlib_plugin[_target_default_plugin](), "default")

    # A plugin name is extracted from the target attribute when present.
    assert_equal(stdlib_plugin[_target_named_plugin](), "cuda")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
