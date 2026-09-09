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

# Negative tests for Apple 8x8 MMA type validation.
# These are expected to fail.

# Mixed float/int inputs (i8 x f16) — falls through to _unsupported_mma_op.
# RUN: not %mojo-build \
# RUN:   --target-triple arm64-apple-darwin25.3.0 \
# RUN:   --target-cpu apple-m5 \
# RUN:   --target-accelerator metal:5:4 \
# RUN:   -D TEST_MIXED_FLOAT_INT=1 \
# RUN:   %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-MIXED

# Non-F32 accumulator with float inputs (f16 x f16 -> f16).
# RUN: not %mojo-build \
# RUN:   --target-triple arm64-apple-darwin25.3.0 \
# RUN:   --target-cpu apple-m5 \
# RUN:   --target-accelerator metal:5:4 \
# RUN:   -D TEST_BAD_FLOAT_ACCUM=1 \
# RUN:   %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-ACCUM

# Unsupported float type (f64 x f64 -> f32).
# RUN: not %mojo-build \
# RUN:   --target-triple arm64-apple-darwin25.3.0 \
# RUN:   --target-cpu apple-m5 \
# RUN:   --target-accelerator metal:5:4 \
# RUN:   -D TEST_F64_INPUT=1 \
# RUN:   %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-F64

# Integer inputs (i8 x i8 -> i32): no 8x8 widening intrinsic exists.
# RUN: not %mojo-build \
# RUN:   --target-triple arm64-apple-darwin25.3.0 \
# RUN:   --target-cpu apple-m5 \
# RUN:   --target-accelerator metal:5:4 \
# RUN:   -D TEST_INT_INPUT=1 \
# RUN:   %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-INT

# Wrong fragment size (8 elements instead of 2).
# RUN: not %mojo-build \
# RUN:   --target-triple arm64-apple-darwin25.3.0 \
# RUN:   --target-cpu apple-m5 \
# RUN:   --target-accelerator metal:5:4 \
# RUN:   -D TEST_WRONG_SHAPE=1 \
# RUN:   %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-SHAPE

from std.sys import get_defined_bool
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8


# CHECK-MIXED: no valid implementation of mma
def test_mixed_float_int():
    comptime if get_defined_bool["TEST_MIXED_FLOAT_INT", False]():
        var a = SIMD[.int8, 2](0)
        var b = SIMD[.float16, 2](0)
        var c = SIMD[.float32, 2](0)
        var d = SIMD[.float32, 2](0)
        _mma_apple_8x8(d, a, b, c)


# CHECK-ACCUM: Apple 8x8 MMA accumulator (C and D) must be F32
def test_bad_float_accum():
    comptime if get_defined_bool["TEST_BAD_FLOAT_ACCUM", False]():
        var a = SIMD[.float16, 2](0)
        var b = SIMD[.float16, 2](0)
        var c = SIMD[.float16, 2](0)
        var d = SIMD[.float16, 2](0)
        _mma_apple_8x8(d, a, b, c)


# CHECK-F64: no valid implementation of mma
def test_f64_input():
    comptime if get_defined_bool["TEST_F64_INPUT", False]():
        var a = SIMD[.float64, 2](0)
        var b = SIMD[.float64, 2](0)
        var c = SIMD[.float32, 2](0)
        var d = SIMD[.float32, 2](0)
        _mma_apple_8x8(d, a, b, c)


# CHECK-INT: no valid implementation of mma
def test_int_input():
    comptime if get_defined_bool["TEST_INT_INPUT", False]():
        var a = SIMD[.int8, 2](0)
        var b = SIMD[.int8, 2](0)
        var c = SIMD[.float32, 2](0)
        var d = SIMD[.float32, 2](0)
        _mma_apple_8x8(d, a, b, c)


# CHECK-SHAPE: Apple 8x8 MMA requires 2-element fragments
def test_wrong_shape():
    comptime if get_defined_bool["TEST_WRONG_SHAPE", False]():
        var a = SIMD[.float16, 8](0)
        var b = SIMD[.float16, 8](0)
        var c = SIMD[.float32, 8](0)
        var d = SIMD[.float32, 8](0)
        _mma_apple_8x8(d, a, b, c)


def main():
    test_mixed_float_int()
    test_bad_float_accum()
    test_f64_input()
    test_int_input()
    test_wrong_shape()
