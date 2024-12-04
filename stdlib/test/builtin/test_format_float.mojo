# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from random import random_float64

from builtin._format_float import _write_float
from python import Python, PythonObject
from testing import assert_equal


def test_float64():
    var test_floats = List[Float64](
        # Zero values
        0.0,
        -0.0,
        # Integer-like values
        1.0,
        -1.0,
        42.0,
        # Simple decimals
        0.5,
        -0.5,
        1.23,
        -1.23,
        # Very small numbers
        0.000123,
        -0.000123,
        1e-10,
        -1e-10,
        # Very large numbers
        1e10,
        -1e10,
        1.23e15,
        -1.23e15,
        # Numbers requiring scientific notation
        1.23e-15,
        -1.23e-15,
        1.23e20,
        -1.23e20,
        # Numbers near scientific notation threshold (typically e±16)
        9.9999e14,
        -9.9999e14,
        1e15,
        -1e15,
        # Repeating decimals
        1 / 3,  # 0.3333...
        -1 / 3,  # -0.3333...
        2 / 3,  # 0.6666...
        # Numbers with many decimal places
        3.141592653589793,
        -3.141592653589793,
        # Numbers that might trigger rounding
        1.999999999999999,
        -1.999999999999999,
        2.0000000000000004,
        -2.0000000000000004,
        # Subnormal numbers
        2.2250738585072014e-308,  # Near minimum subnormal
        -2.2250738585072014e-308,
        # Numbers near system limits
        1.7976931348623157e308,  # Near maximum float
        -1.7976931348623157e308,
        2.2250738585072014e-308,  # Near minimum normal float
        -2.2250738585072014e-308,
        # Numbers that might trigger special formatting
        1000000.0,  # Could be formatted as 1e6 or 1000000
        0.0000001,  # Could be formatted as 1e-7 or 0.0000001
        # Numbers with trailing zeros
        1.100,
        -1.100,
        1.0010,
        -1.0010,
        # Numbers that might affect alignment
        999999.999999,
        -999999.999999,
        0.000000999999,
        -0.000000999999,
    )

    for f in test_floats:
        # Float64
        var mojo_f64_str = String()
        _write_float(mojo_f64_str, f[])

        var py_f64_str = str(PythonObject(f[]))

        assert_equal(py_f64_str, mojo_f64_str)


def test_float32():
    var test_floats = List[Float32](
        # Zero values
        Float32(0.0),
        Float32(-0.0),
        # Integer-like values
        Float32(1.0),
        Float32(-1.0),
        Float32(42.0),
        # Simple decimals
        Float32(0.5),
        Float32(-0.5),
        Float32(1.23),
        Float32(-1.23),
        # Very small numbers
        Float32(1.18e-38),  # Near minimum normal float32
        Float32(-1.18e-38),
        Float32(1e-35),
        Float32(-1e-35),
        # Very large numbers
        Float32(1e35),
        Float32(-1e35),
        Float32(3.4e38),  # Near maximum float32
        Float32(-3.4e38),
        # Numbers requiring scientific notation
        Float32(1.23e-35),
        Float32(-1.23e-35),
        Float32(1.23e35),
        Float32(-1.23e35),
        # Numbers near scientific notation threshold
        Float32(9.9999e14),
        Float32(-9.9999e14),
        Float32(1e15),
        Float32(-1e15),
        # Repeating decimals
        Float32(0.3333),
        Float32(-0.3333),
        Float32(0.6666),
        # Numbers with precision near float32 limit (~7 decimal digits)
        Float32(3.141593),  # Pi
        Float32(-3.141593),
        # Numbers that might trigger rounding
        Float32(1.9999999),
        Float32(-1.9999999),
        Float32(2.0000002),
        Float32(-2.0000002),
        # Subnormal numbers for float32
        Float32(1.4e-45),  # Near minimum subnormal float32
        Float32(-1.4e-45),
        # Numbers near system limits for float32
        Float32(3.4028234e38),  # Max float32
        Float32(-3.4028234e38),
        Float32(1.1754944e-38),  # Min normal float32
        Float32(-1.1754944e-38),
        # Numbers that might trigger special formatting
        Float32(100000.0),
        Float32(0.000001),
        # Numbers with trailing zeros
        Float32(1.100),
        Float32(-1.100),
        Float32(1.001),
        Float32(-1.001),
        # Numbers that might affect alignment
        Float32(99999.99),
        Float32(-99999.99),
        Float32(0.0000999),
        Float32(-0.0000999),
        # Powers of 2 (important for binary floating-point)
        Float32(2.0),
        Float32(4.0),
        Float32(8.0),
        Float32(16.0),
        Float32(32.0),
        Float32(64.0),
        Float32(128.0),
        # Numbers that demonstrate float32 precision limits
        Float32(
            16777216.0
        ),  # 2^24, last integer that can be represented exactly
        Float32(16777217.0),  # 2^24 + 1, demonstrates precision loss
        # Numbers that demonstrate mantissa precision
        Float32(1.000000119),  # Smallest number > 1.0 in float32
        Float32(0.999999881),  # Largest number < 1.0 in float32
    )

    for f in test_floats:
        var np = Python.import_module("numpy")
        var mojo_f32_str = String()
        _write_float(mojo_f32_str, f[])

        var py_f32_str = str(np.float32(f[]))

        assert_equal(py_f32_str, mojo_f32_str)


def test_random_floats():
    for _ in range(10000):
        var f64 = random_float64()
        var mojo_f64_str = String()
        _write_float(mojo_f64_str, f64)
        var py_f64_str = str(PythonObject(f64))
        assert_equal(py_f64_str, mojo_f64_str)


def main():
    test_float64()
    test_float32()
    test_random_floats()
