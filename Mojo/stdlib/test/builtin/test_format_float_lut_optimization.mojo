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

# Test that float formatting lookup tables (cache_f32 and cache_f64) are
# optimized to use global constants instead of stack allocation.
#
# This verifies that the global_constant optimization is working correctly.
# Before the optimization, these lookup tables would be materialized on the
# stack (9,904 bytes for cache_f64, 624 bytes for cache_f32) every time they
# were accessed. With the optimization, they live as global constants in the
# read-only data section.

# RUN: mkdir -p %t
# RUN: %mojo-build %s -o %t/test_format_float_lut.ll --emit llvm
# RUN: FileCheck %s --check-prefix=GLOBAL-CONSTANTS --input-file=%t/test_format_float_lut.ll
# RUN: FileCheck %s --check-prefix=NO-STACK-ALLOC --input-file=%t/test_format_float_lut.ll

# Verify that cache_f32 (78 x i64) exists as a global constant
# GLOBAL-CONSTANTS-DAG: @{{.*}}global_constant{{.*}} = {{.*}}constant {{.*}}[78 x i64]

# Verify that cache_f64 (619 x i128) exists as a global constant
# GLOBAL-CONSTANTS-DAG: @{{.*}}global_constant{{.*}} = {{.*}}constant {{.*}}[619 x i128]

# Verify that cache_f32 is NOT allocated on the stack
# NO-STACK-ALLOC-NOT: alloca {{.*}}[78 x i64]

# Verify that cache_f64 is NOT allocated on the stack
# NO-STACK-ALLOC-NOT: alloca {{.*}}[619 x i128]

from std.builtin._format_float import _write_float


def test_float32_formatting() -> String:
    """Test Float32 formatting that will use cache_f32 lookup table."""
    var result = String()

    # These values will trigger different code paths in the formatter
    # and exercise the cache_f32 lookup table
    var f1 = Float32(1.23e-15)  # Small number requiring scientific notation
    var f2 = Float32(1.23e20)  # Large number requiring scientific notation
    var f3 = Float32(3.14159)  # Regular decimal

    _write_float(result, f1)
    result += ","
    _write_float(result, f2)
    result += ","
    _write_float(result, f3)

    return result


def test_float64_formatting() -> String:
    """Test Float64 formatting that will use cache_f64 lookup table."""
    var result = String()

    # These values will trigger different code paths in the formatter
    # and exercise the cache_f64 lookup table
    var f1 = Float64(1.23e-100)  # Very small number
    var f2 = Float64(1.23e100)  # Very large number
    var f3 = Float64(2.718281828)  # Regular decimal

    _write_float(result, f1)
    result += ","
    _write_float(result, f2)
    result += ","
    _write_float(result, f3)

    return result


def main():
    var f32_results = test_float32_formatting()
    var f64_results = test_float64_formatting()

    # We don't actually need to print the results for the test,
    # but we need to use them so they don't get optimized away
    if f32_results.byte_length() > 0 and f64_results.byte_length() > 0:
        pass
