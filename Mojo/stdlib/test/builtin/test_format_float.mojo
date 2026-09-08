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

from std.builtin._format_float import _write_float
from std.testing import assert_equal, TestSuite


def test_float64() raises:
    var cases = Dict[String, Float64]()
    # Zero values
    cases["0.0"] = Float64(0.0)
    cases["-0.0"] = Float64(-0.0)
    # Integer-like values
    cases["1.0"] = Float64(1.0)
    cases["-1.0"] = Float64(-1.0)
    cases["42.0"] = Float64(42.0)
    # Simple decimals
    cases["0.5"] = Float64(0.5)
    cases["-0.5"] = Float64(-0.5)
    cases["1.23"] = Float64(1.23)
    cases["-1.23"] = Float64(-1.23)
    # Very small numbers
    cases["1.18e-38"] = Float64(1.18e-38)  # Near minimum normal float64
    cases["-1.18e-38"] = Float64(-1.18e-38)
    cases["1e-35"] = Float64(1e-35)
    cases["-1e-35"] = Float64(-1e-35)
    # Very large numbers
    cases["1e+35"] = Float64(1e35)
    cases["-1e+35"] = Float64(-1e35)
    cases["1230000000000000.0"] = Float64(1.23e15)
    cases["-1230000000000000.0"] = Float64(-1.23e15)
    # Numbers requiring scientific notation
    cases["1.23e-15"] = Float64(1.23e-15)
    cases["-1.23e-15"] = Float64(-1.23e-15)
    cases["1.23e+20"] = Float64(1.23e20)
    cases["-1.23e+20"] = Float64(-1.23e20)
    # Numbers near scientific notation threshold (typically e±16)
    cases["999990000000000.0"] = Float64(9.9999e14)
    cases["-999990000000000.0"] = Float64(-9.9999e14)
    cases["1000000000000000.0"] = Float64(1e15)
    cases["-1000000000000000.0"] = Float64(-1e15)
    # Repeating decimals
    cases["0.3333"] = Float64(0.3333)
    cases["-0.3333"] = Float64(-0.3333)
    cases["0.6666"] = Float64(0.6666)
    # Numbers with many decimal places
    cases["3.141592653589793"] = Float64(3.141592653589793)
    cases["-3.141592653589793"] = Float64(-3.141592653589793)
    # Numbers that might trigger rounding
    cases["1.999999999999999"] = Float64(1.999999999999999)
    cases["-1.999999999999999"] = Float64(-1.999999999999999)
    cases["2.0000000000000004"] = Float64(2.0000000000000004)
    cases["-2.0000000000000004"] = Float64(-2.0000000000000004)
    # Subnormal numbers
    cases["2.2250738585072014e-308"] = Float64(2.2250738585072014e-308)
    cases["-2.2250738585072014e-308"] = Float64(-2.2250738585072014e-308)
    # Numbers near system limits
    cases["1.7976931348623157e+308"] = Float64(1.7976931348623157e308)
    cases["-1.7976931348623157e+308"] = Float64(-1.7976931348623157e308)
    cases["2.2250738585072014e-308"] = Float64(2.2250738585072014e-308)
    cases["-2.2250738585072014e-308"] = Float64(-2.2250738585072014e-308)
    # Numbers that might trigger special formatting
    cases["1000000.0"] = Float64(
        1000000.0
    )  # Could be formatted as 1e6 or 1000000
    cases["1e-06"] = Float64(
        0.000001
    )  # Could be formatted as 1e-7 or 0.0000001
    # Numbers with trailing zeros
    cases["1.1"] = Float64(1.100)
    cases["-1.1"] = Float64(-1.100)
    cases["1.001"] = Float64(1.0010)
    cases["-1.001"] = Float64(-1.0010)
    # Numbers that might affect alignment
    cases["999999.999999"] = Float64(999999.999999)
    cases["-999999.999999"] = Float64(-999999.999999)
    cases["9.99999e-07"] = Float64(0.000000999999)
    cases["-9.99999e-07"] = Float64(-0.000000999999)

    for entry in cases.items():
        var mojo_f64_str = String()
        _write_float(mojo_f64_str, entry.value)
        assert_equal(entry.key, mojo_f64_str)


def test_float32() raises:
    var cases = Dict[String, Float32]()
    # Zero values
    cases["0.0"] = Float32(0.0)
    cases["-0.0"] = Float32(-0.0)
    # Integer-like values
    cases["1.0"] = Float32(1.0)
    cases["-1.0"] = Float32(-1.0)
    cases["42.0"] = Float32(42.0)
    # Simple decimals
    cases["0.5"] = Float32(0.5)
    cases["-0.5"] = Float32(-0.5)
    cases["1.23"] = Float32(1.23)
    cases["-1.23"] = Float32(-1.23)
    # Very small numbers
    cases["1.18e-38"] = Float32(1.18e-38)  # Near minimum normal float32
    cases["-1.18e-38"] = Float32(-1.18e-38)
    cases["1e-35"] = Float32(1e-35)
    cases["-1e-35"] = Float32(-1e-35)
    # Very large numbers
    cases["1e+35"] = Float32(1e35)
    cases["-1e+35"] = Float32(-1e35)
    cases["3.4e+38"] = Float32(3.4e38)  # Near maximum float32
    cases["-3.4e+38"] = Float32(-3.4e38)
    # Numbers requiring scientific notation
    cases["1.23e-35"] = Float32(1.23e-35)
    cases["-1.23e-35"] = Float32(-1.23e-35)
    cases["1.23e+35"] = Float32(1.23e35)
    cases["-1.23e+35"] = Float32(-1.23e35)
    # Numbers near scientific notation threshold
    cases["999990000000000.0"] = Float32(9.9999e14)
    cases["-999990000000000.0"] = Float32(-9.9999e14)
    cases["1000000000000000.0"] = Float32(1e15)
    cases["-1000000000000000.0"] = Float32(-1e15)
    # Repeating decimals
    cases["0.3333"] = Float32(0.3333)
    cases["-0.3333"] = Float32(-0.3333)
    cases["0.6666"] = Float32(0.6666)
    # Numbers with precision near float32 limit (~7 decimal digits)
    cases["3.141593"] = Float32(3.141593)  # Pi
    cases["-3.141593"] = Float32(-3.141593)
    # Numbers that might trigger rounding
    cases["1.9999999"] = Float32(1.9999999)
    cases["-1.9999999"] = Float32(-1.9999999)
    cases["2.0000002"] = Float32(2.0000002)
    cases["-2.0000002"] = Float32(-2.0000002)
    # Subnormal numbers for float32
    # TODO(MSTDL-1610): use Float32(1.4e-45) and Float32(-1.4e-45)
    cases["1e-45"] = Float32(1.4e-44) / 10  # Near minimum subnormal float32
    cases["-1e-45"] = Float32(-1.4e-44) / 10
    # Numbers near system limits for float32
    cases["3.4028235e+38"] = Float32(3.4028234e38)  # Rounds to max float32
    cases["-3.4028235e+38"] = Float32(-3.4028234e38)
    cases["1.1754944e-38"] = Float32(1.1754944e-38)  # Min normal float32
    cases["-1.1754944e-38"] = Float32(-1.1754944e-38)
    # Numbers that might trigger special formatting
    cases["100000.0"] = Float32(100000.0)
    cases["1e-06"] = Float32(0.000001)
    # Numbers with trailing zeros
    cases["1.1"] = Float32(1.100)
    cases["-1.1"] = Float32(-1.100)
    cases["1.001"] = Float32(1.0010)
    cases["-1.001"] = Float32(-1.0010)
    # Numbers that might affect alignment
    cases["99999.99"] = Float32(99999.99)
    cases["-99999.99"] = Float32(-99999.99)
    cases["9.99e-05"] = Float32(0.0000999)
    cases["-9.99e-05"] = Float32(-0.0000999)
    # Powers of 2 (important for binary floating-point)
    cases["2.0"] = Float32(2.0)
    cases["4.0"] = Float32(4.0)
    cases["8.0"] = Float32(8.0)
    cases["16.0"] = Float32(16.0)
    cases["32.0"] = Float32(32.0)
    cases["64.0"] = Float32(64.0)
    cases["128.0"] = Float32(128.0)
    # Numbers that demonstrate float32 precision limits
    cases["16777216.0"] = Float32(
        16777216.0
    )  # 2^24, last integer that can be represented exactly
    cases["16777216.0"] = Float32(
        16777217.0
    )  # 2^24 + 1, demonstrates precision loss
    # Numbers that demonstrate mantissa precision
    cases["1.0000001"] = Float32(
        1.000000119
    )  # Smallest number > 1.0 in float32
    cases["0.9999999"] = Float32(0.999999881)  # Largest number < 1.0 in float32

    for entry in cases.items():
        var mojo_f32_str = String()
        _write_float(mojo_f32_str, entry.value)
        assert_equal(entry.key, mojo_f32_str)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
