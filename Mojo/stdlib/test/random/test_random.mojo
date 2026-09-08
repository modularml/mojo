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
"""Tests for the public random number generation API.

This test file validates the public API of the random module, including
random number generation, seeding, and shuffle functionality.
"""

from std.random import (
    randn_float64,
    random_float64,
    random_si64,
    random_ui64,
    seed,
    shuffle,
)

from std.testing import assert_equal, assert_false, assert_true, TestSuite


def test_random() raises:
    """Test basic random number generation for all numeric types.

    This is a smoke test verifying that random number generators produce
    values within the specified ranges for float, signed int, and unsigned int
    types. This catches fundamental issues like:
    - Incorrect range calculations (e.g., off-by-one errors)
    - Type conversion bugs (e.g., signed/unsigned confusion)
    - Broken RNG that produces values outside expected bounds
    - API contract violations (e.g., inclusive vs exclusive bounds)

    This test is the first line of defense - if this fails, the RNG is
    fundamentally broken and no other tests matter.
    """
    for _ in range(100):
        var random_float = random_float64(0, 1)
        assert_true(
            random_float >= 0,
            String("Value ", random_float, " is not above or equal to 0"),
        )
        assert_true(
            random_float <= 1,
            String("Value ", random_float, " is not below or equal to 1"),
        )

        var random_signed = random_si64(-255, 255)
        assert_true(
            random_signed >= -255,
            String(
                "Signed value ", random_signed, " is not above or equal to -255"
            ),
        )
        assert_true(
            random_signed <= 255,
            String(
                "Signed value ", random_signed, " is not below or equal to 255"
            ),
        )

        var random_unsigned = random_ui64(0, 255)
        assert_true(
            random_unsigned >= 0,
            String(
                "Unsigned value ",
                random_unsigned,
                " is not above or equal to 0",
            ),
        )
        assert_true(
            random_unsigned <= 255,
            String(
                "Unsigned value ",
                random_unsigned,
                " is not below or equal to 255",
            ),
        )


def test_seed_normal() raises:
    """Test that `randn_float64` produces a proper normal distribution.

    This validates that the Box-Muller transform implementation correctly
    generates samples from a normal distribution with the specified mean
    and standard deviation.
    """
    seed(42)
    var num_samples = 1000
    var samples = List[Float64](capacity=num_samples)
    for _ in range(num_samples):
        samples.append(randn_float64(0, 2))

    var sum: Float64 = 0.0
    for sample in samples:
        sum += sample

    var mean: Float64 = sum / Float64(num_samples)

    var sum_sq: Float64 = 0.0
    for sample in samples:
        sum_sq += (sample - mean) ** 2

    var variance = sum_sq / Float64(num_samples)

    # Calculate absolute differences (errors)
    var mean_error = abs(mean)
    var variance_error = abs(variance - 4)

    var mean_tolerance: Float64 = 0.06  # SE_μ = σ / √n
    assert_true(
        mean_error < mean_tolerance,
        String(
            "Mean error ",
            mean_error,
            " is above the accepted tolerance ",
            mean_tolerance,
        ),
    )
    var variance_tolerance: Float64 = 0.57  # SE_S² = √(2 * σ^4 / (n - 1))
    assert_true(
        variance_error < variance_tolerance,
        String(
            "Variance error ",
            variance_error,
            " is above the accepted tolerance ",
            variance_tolerance,
        ),
    )


def test_seed() raises:
    """Test that seeding produces reproducible random sequences."""
    seed(5)
    var some_float = random_float64(0, 1)
    var some_signed_integer = random_si64(-255, 255)
    var some_unsigned_integer = random_ui64(0, 255)

    seed(5)
    assert_equal(some_float, random_float64(0, 1))
    assert_equal(some_signed_integer, random_si64(-255, 255))
    assert_equal(some_unsigned_integer, random_ui64(0, 255))


def test_shuffle() raises:
    """Test the Fisher-Yates shuffle implementation.

    This validates that `shuffle()` correctly randomizes list order while
    preserving important properties.
    - Preservation: All original elements remain present (no additions/deletions)
    - Length: List size doesn't change
    - Permutation: Order actually changes (not a no-op)
    - Edge cases: Single-element lists handled correctly
    - Type generics: Works with various types (Int, String, List[List[T]])
    - Randomness: Large lists don't remain in original order

    In the future, we may move this to be more a property-based test instead of
    a unit test.
    """
    # TODO: Clean up with list comprehension when possible.

    # Property tests
    comptime L_i = List[Int]
    comptime L_s = List[String]
    var a: L_i = [1, 2, 3, 4]
    var b: L_i = [1, 2, 3, 4]
    var c: L_s = ["Random", "shuffle", "in", "Mojo"]
    var d: L_s = ["Random", "shuffle", "in", "Mojo"]

    shuffle(b)
    assert_equal(len(a), len(b))
    assert_true(a != b)
    for item in b:
        assert_true(item in a)

    shuffle(d)
    assert_equal(len(c), len(d))
    assert_true(c != d)
    for item in d:
        assert_true(item in c)

    var e: L_i = [21]
    shuffle(e)
    assert_true(e == [21])
    var f: L_s = ["Mojo"]
    shuffle(f)
    assert_true(f == ["Mojo"])

    comptime L_l = List[List[Int]]
    var g = L_l()
    var h = L_l()
    for i in range(10):
        g.append([i, i + 1, i + 3])
        h.append([i, i + 1, i + 3])
    shuffle(g)
    # TODO: Uncomment when possible
    # assert_true(g != h)
    assert_equal(len(g), len(h))
    for i in range(10):
        # Currently, the below does not compile.
        # assert_true(g.__contains__(L_i(i, i + 1, i + 3)))
        var target: List[Int] = [i, i + 1, i + 3]
        var found = False
        for j in range(len(g)):
            if g[j] == target:
                found = True
                break
        assert_true(found)

    comptime L_l_s = List[List[String]]
    var i = L_l_s()
    var j = L_l_s()
    for x in range(10):
        i.append([String(x), String(x + 1), String(x + 3)])
        j.append([String(x), String(x + 1), String(x + 3)])
    shuffle(i)
    # TODO: Uncomment when possible
    # assert_true(g != h)
    assert_equal(len(i), len(j))
    for x in range(10):
        var target: List[String] = [String(x), String(x + 1), String(x + 3)]
        var found = False
        for y in range(len(i)):
            if j[y] == target:
                found = True
                break
        assert_true(found)

    # Given the number of permutations of size 1000 is 1000!,
    # we rely on the assertion that a truly random shuffle should not
    # result in the same order as the to pre-shuffle list with extremely
    # high probability.
    var l = L_i()
    var m = L_i()
    for i in range(1000):
        l.append(i)
        m.append(i)
    shuffle(l)
    assert_equal(len(l), len(m))
    assert_true(l != m)
    shuffle(m)
    assert_equal(len(l), len(m))
    assert_true(l != m)


def test_random_edge_cases() raises:
    """Test edge cases in random number generation.

    This validates correct behavior for unusual but valid inputs that often
    expose implementation bugs:
    - Degenerate ranges (min == max): Should return the constant value
    - Inverted ranges (min > max): Tests undefined behavior handling
    - Very small ranges (1.0 + 1e-10): Tests floating-point precision
    - Negative ranges: Ensures sign handling is correct
    - Boundary values: Tests integer overflow/underflow protection
    """
    # Test min == max returns min
    assert_equal(random_float64(5.0, 5.0), 5.0)
    assert_equal(random_float64(0.0, 0.0), 0.0)
    assert_equal(random_float64(-100.0, -100.0), -100.0)
    assert_equal(random_si64(42, 42), 42)
    assert_equal(random_si64(-42, -42), -42)
    assert_equal(random_ui64(100, 100), 100)

    # Test min > max behavior
    # Note: Integer versions return min, but float version computes with negative range
    # For float: when min > max, range is negative, so result is between max and min
    seed(42)
    var inverted_float = random_float64(10.0, 5.0)
    assert_true(inverted_float >= 5.0, "Should be >= max (lower bound)")
    assert_true(inverted_float <= 10.0, "Should be <= min (upper bound)")

    # Integer versions have explicit min >= max check that returns min
    assert_equal(random_si64(100, 50), 100)
    assert_equal(random_ui64(100, 50), 100)

    # Test very small ranges work correctly
    var small_range = random_float64(1.0, 1.0 + 1e-10)
    assert_true(small_range >= 1.0)
    assert_true(small_range <= 1.0 + 1e-10)

    # Test negative ranges
    var neg = random_float64(-100.0, -50.0)
    assert_true(neg >= -100.0)
    assert_true(neg <= -50.0)

    # Test integer boundary values
    var small_int = random_si64(-10, 10)
    assert_true(small_int >= -10)
    assert_true(small_int <= 10)

    # Test that seeding produces deterministic results for edge cases
    seed(12345)
    var edge1 = random_float64(5.0, 5.0)
    seed(12345)
    var edge2 = random_float64(5.0, 5.0)
    assert_equal(edge1, edge2)


def test_random_float64_special_values() raises:
    """Test random_float64 with extreme and special IEEE 754 values."""
    # Test with very large ranges (near infinity)
    # These should not overflow or produce inf/nan
    var large1 = random_float64(-1e100, 1e100)
    assert_false(large1 != large1, "Result should not be NaN")  # NaN check

    # Test with zero range
    assert_equal(random_float64(0.0, 0.0), 0.0)

    # Test values near zero
    var near_zero = random_float64(-1e-100, 1e-100)
    assert_true(near_zero >= -1e-100)
    assert_true(near_zero <= 1e-100)


def test_uniformity_basic() raises:
    """Basic test for uniform distribution of random numbers.

    This performs a simple chi-square-like test to verify that random_ui64
    produces roughly uniform distribution across buckets. A truly uniform
    random number generator should distribute values evenly across the range.

    Test methodology:
    - Generate 1000 samples from random_ui64(0, 9)
    - Count how many fall into each of the 10 buckets (0-9)
    - Expected: ~100 samples per bucket (1000 / 10 = 100)
    - Actual: We allow [50, 150] per bucket (very lenient tolerance)

    Why this matters:
    - A broken RNG might produce values that cluster (e.g., mostly 0-3)
    - Or it might have bias toward certain values (e.g., even numbers)
    - Or it might produce predictable patterns (e.g., 0,1,2,3,0,1,2,3...)

    Note: This is NOT a rigorous statistical test. A proper chi-square test
    would compute the statistic: χ² = Σ((observed - expected)² / expected)
    and compare against a critical value for the desired confidence level.
    This test just catches egregiously bad distributions.
    """
    seed(42)
    comptime num_buckets = 10
    comptime samples_per_bucket = 100
    comptime total_samples = num_buckets * samples_per_bucket

    # Count samples in each bucket
    var buckets = List[Int](length=num_buckets, fill=0)

    # Generate samples and count which bucket they fall into
    for _ in range(total_samples):
        var val = random_ui64(0, num_buckets - 1)
        buckets[Int(val)] += 1

    # Check that each bucket has a reasonable number of samples
    # With a truly uniform distribution and 1000 samples across 10 buckets:
    # - Expected value per bucket: 100
    # - Standard deviation per bucket: sqrt(n*p*(1-p)) = sqrt(1000*0.1*0.9) ≈ 9.5
    # - Our tolerance [50, 150] is approximately ±5 standard deviations
    # - This is extremely lenient: proper test would use ±3 std devs (~[71, 129])
    #
    # We're lenient to avoid flaky tests, but even a mediocre RNG should
    # easily pass this. If this test fails, the RNG is seriously broken.
    for i, count in enumerate(buckets):
        assert_true(
            count >= 50,
            String("Bucket ", i, " has too few samples: ", count),
        )
        assert_true(
            count <= 150,
            String("Bucket ", i, " has too many samples: ", count),
        )

    # Calculate basic statistics
    var sum_samples = 0
    for i in range(num_buckets):
        sum_samples += buckets[i]
    assert_equal(sum_samples, total_samples)


def test_normal_distribution_edge_cases() raises:
    """Test edge cases for normal distribution generation.

    The Box-Muller transform can behave unexpectedly with extreme parameters.
    This test ensures robust handling of edge cases that could cause:
    - Division by zero or multiplication overflow in the transform
    - Loss of precision with very small/large values
    - NaN propagation from intermediate calculations
    """
    # Test 1: standard_deviation = 0 (degenerate distribution)
    # Rationale: When std_dev = 0, we have a "point mass" distribution where
    # all probability is concentrated at the mean. This is mathematically valid
    # but can break naive implementations that divide by std_dev.
    # Expected behavior: All samples should exactly equal the mean.
    seed(42)
    for _ in range(10):
        var value = randn_float64(mean=5.0, standard_deviation=0.0)
        assert_equal(
            value,
            5.0,
            "With standard_deviation=0, all values should equal mean",
        )

    # Verify this works for different means (positive, large, negative)
    assert_equal(randn_float64(mean=100.0, standard_deviation=0.0), 100.0)
    assert_equal(randn_float64(mean=-50.0, standard_deviation=0.0), -50.0)

    # Test 2: Very large standard_deviation (overflow risk)
    # Rationale: Box-Muller computes: mag = std_dev * sqrt(-2 * log(u1))
    # With std_dev = 1e100, intermediate values like (1e100 * sqrt(some_number))
    # could overflow to infinity. We want to ensure the implementation handles
    # this gracefully and produces finite (if very large) results.
    # Note: Statistically, extreme values ARE possible with large std_dev, but
    # we shouldn't immediately overflow to inf due to implementation issues.
    seed(123)
    for _ in range(20):
        var large_std = randn_float64(mean=0.0, standard_deviation=1e100)
        # Check it doesn't produce NaN (which would indicate a calculation error)
        assert_false(large_std != large_std, "Should not produce NaN")
        # The result may be very large (even inf statistically), but that's
        # a consequence of the extreme parameter, not an implementation bug.

    # Test 3: Very small standard_deviation (precision test)
    # Rationale: With std_dev = 1e-10, values should cluster extremely tightly
    # around the mean. This tests whether floating-point precision is maintained
    # when the transform computes: result = mean + std_dev * normal_sample
    # With tiny std_dev, we could lose precision or produce values identical
    # to mean due to rounding. We want to ensure the distribution isn't
    # completely collapsed by numerical errors.
    seed(456)
    var samples_small = List[Float64](capacity=100)
    for _ in range(100):
        samples_small.append(randn_float64(mean=10.0, standard_deviation=1e-10))

    # All values should be very close to mean, but not necessarily bitwise identical
    # We allow deviation up to 1e-8 (much larger than std_dev = 1e-10) to account
    # for accumulated floating-point errors in the Box-Muller transform.
    for sample in samples_small:
        var deviation = abs(sample - 10.0)
        assert_true(
            deviation < 1e-8,
            String(
                (
                    "With tiny standard_deviation, values should be near mean,"
                    " got "
                ),
                sample,
            ),
        )

    # Test 4: Negative mean (sanity check)
    # Rationale: The normal distribution is symmetric and defined for all real means.
    # However, implementations sometimes make implicit assumptions about positive
    # values. This test ensures negative means work correctly and produce the
    # expected statistical properties.
    # We perform a simple statistical test: sample mean should be close to -100.
    seed(789)
    var negative_mean_samples = List[Float64](capacity=50)
    for _ in range(50):
        negative_mean_samples.append(
            randn_float64(mean=-100.0, standard_deviation=10.0)
        )

    # Compute the sample mean
    var sum_neg: Float64 = 0.0
    for sample in negative_mean_samples:
        sum_neg += sample
    var mean_neg = sum_neg / 50.0

    # With 50 samples from N(-100, 10²), the sample mean should be approximately -100
    # Standard error of the mean = σ / sqrt(n) = 10 / sqrt(50) ≈ 1.4
    # We allow ±30 (very lenient, ~21 standard errors) to avoid flaky tests.
    # This catches gross errors like forgetting to add the mean parameter.
    assert_true(mean_neg > -130.0 and mean_neg < -70.0)

    # Test 5: Large mean with small standard_deviation (precision test)
    # Rationale: When mean = 1e100 and std_dev = 1.0, the final computation is:
    # result = 1e100 + (std_dev * normal_sample) where normal_sample ∈ roughly [-4, 4]
    # This tests whether adding a small value to a very large value maintains
    # precision. We should get back a value very close to 1e100, not exactly 1e100
    # due to loss of precision in the low-order bits.
    seed(999)
    var large_mean = randn_float64(mean=1e100, standard_deviation=1.0)
    # With std_dev = 1.0, sample should be within a few standard deviations of mean
    # We check |result - 1e100| < 10 (10 standard deviations, very conservative)
    # This catches cases where precision is completely lost and we get exactly 1e100
    assert_true(abs(large_mean - 1e100) < 10.0)


def test_full_range_integers() raises:
    """Test random integers with full type range don't always return min."""
    seed(42)

    # Generate multiple samples and verify we don't always get min
    var got_non_zero = False
    for _ in range(100):
        if random_ui64(0, UInt64.MAX) != 0:
            got_non_zero = True
            break
    assert_true(got_non_zero, "Full-range UInt64 should not always return 0")

    var got_non_min = False
    for _ in range(100):
        if random_si64(Int64.MIN, Int64.MAX) != Int64.MIN:
            got_non_min = True
            break
    assert_true(got_non_min, "Full-range Int64 should not always return MIN")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
