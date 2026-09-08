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
"""Tests for internal random number generation implementation details.

This test file validates the internal implementation of the random module,
specifically the _PhiloxWrapper adapter layer that sits between the raw
Philox SIMD generator and the public scalar API.

These tests may break during internal refactoring and that's expected.
They verify specific implementation contracts that are important for
correctness but not part of the public API.
"""

from std.random._rng import _PhiloxWrapper

from std.testing import assert_equal, assert_true, TestSuite


def test_philox_wrapper_basic() raises:
    """Test basic functionality of the Philox wrapper.

    This verifies that the wrapper correctly generates random values
    in the expected range and maintains consistent state.
    """
    # Test that values are generated in expected range
    var rng = _PhiloxWrapper(seed=42)
    for _ in range(100):
        var u32 = rng.next_uint32()
        # Just verify it's a valid uint32 (will always be true, but documents expectation)
        assert_true(u32 <= UInt32.MAX, "Should generate valid uint32")

    # Test next_uint64 produces consistent results
    var rng1 = _PhiloxWrapper(seed=42)
    var val64 = rng1.next_uint64()
    # Ensure next_uint64 calls next_uint32 twice
    var rng2 = _PhiloxWrapper(seed=42)
    var high = UInt64(rng2.next_uint32())
    var low = UInt64(rng2.next_uint32())
    assert_equal(val64, (high << 32) | low)

    # Test next_float64 is in range [0, 1)
    var rng3 = _PhiloxWrapper(seed=99)
    for _ in range(100):
        var f = rng3.next_float64()
        assert_true(f >= 0.0, "Float64 should be >= 0")
        assert_true(f < 1.0, "Float64 should be < 1")


def test_float64_precision() raises:
    """Test precision properties of next_float64().

    The Philox wrapper implementation should:
    1. Use exactly 53 bits of precision (Float64 mantissa size)
    2. Never return exactly 1.0 (should be in [0, 1))
    3. Generate values across the full range [0, 1)
    """

    # Test 1: Verify it never returns exactly 1.0
    var rng = _PhiloxWrapper(seed=42)
    for _ in range(10000):
        var value = rng.next_float64()
        assert_true(value >= 0.0, "Should be >= 0.0")
        assert_true(value < 1.0, "Should be < 1.0 (never equal)")

    # Test 2: Check the implementation uses 53 bits
    # next_float64() should use: (next_uint64() >> 11) * (1.0 / 2^53)
    # This gives us exactly 53 bits of precision
    var rng2 = _PhiloxWrapper(seed=100)
    var raw_bits = rng2.next_uint64() >> 11  # Top 53 bits
    var expected = Float64(raw_bits) * (1.0 / 9007199254740992.0)  # 2^53

    var rng3 = _PhiloxWrapper(seed=100)
    var actual = rng3.next_float64()
    assert_equal(actual, expected, "Should use exactly 53 bits")

    # Test 3: Verify the full range [0, 1) is covered
    # With 53 bits, smallest non-zero value is 2^-53 ≈ 1.11e-16
    # We should be able to generate values across the entire range
    var rng4 = _PhiloxWrapper(seed=0)
    var found_small = False  # < 0.1
    var found_large = False  # > 0.9
    for _ in range(10000):
        var val = rng4.next_float64()
        if val < 0.1:
            found_small = True
        if val > 0.9:
            found_large = True
        if found_small and found_large:
            break
    assert_true(found_small, "Should generate values < 0.1")
    assert_true(found_large, "Should generate values > 0.9")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
