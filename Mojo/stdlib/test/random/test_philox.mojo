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

"""Tests for the Philox random number generator."""

from std.random.philox import Random, NormalRandom
from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_false, TestSuite


def test_philox_basic() raises:
    """Test basic Philox RNG functionality."""
    var rng = Random(seed=42, subsequence=0, offset=0)

    # Generate random integers
    var random_ints = rng.step()
    # Verify we get 4 values
    assert_equal(len(random_ints), 4, "step() should return 4 values")

    # Generate uniform random floats
    var random_floats = rng.step_uniform()
    # Verify we get 4 values
    assert_equal(len(random_floats), 4, "step_uniform() should return 4 values")

    # Verify all floats are in [0, 1) range
    for i in range(4):
        assert_true(
            random_floats[i] >= 0.0,
            String("Value at index ", i, " should be >= 0.0"),
        )
        assert_true(
            random_floats[i] < 1.0,
            String("Value at index ", i, " should be < 1.0"),
        )


def test_philox_reproducibility() raises:
    """Test that the same seed produces the same sequence."""
    # Create two RNGs with the same seed
    var rng1 = Random(seed=42, subsequence=0, offset=0)
    var rng2 = Random(seed=42, subsequence=0, offset=0)

    # Generate values from both
    var vals1_1 = rng1.step()
    var vals1_2 = rng1.step()
    var vals2_1 = rng2.step()
    var vals2_2 = rng2.step()

    # Verify sequences are identical
    for i in range(4):
        assert_equal(
            vals1_1[i],
            vals2_1[i],
            String("First step values should match at index ", i),
        )
        assert_equal(
            vals1_2[i],
            vals2_2[i],
            String("Second step values should match at index ", i),
        )


def test_philox_different_seeds() raises:
    """Test that different seeds produce different sequences."""
    var rng1 = Random(seed=42, subsequence=0, offset=0)
    var rng2 = Random(seed=123, subsequence=0, offset=0)

    var vals1 = rng1.step()
    var vals2 = rng2.step()

    # At least one value should be different (with overwhelming probability)
    var all_equal = True
    for i in range(4):
        if vals1[i] != vals2[i]:
            all_equal = False
            break

    assert_false(
        all_equal, "Different seeds should produce different sequences"
    )


def test_philox_subsequences() raises:
    """Test that different subsequences produce different values."""
    var rng1 = Random(seed=42, subsequence=0, offset=0)
    var rng2 = Random(seed=42, subsequence=1, offset=0)

    var vals1 = rng1.step()
    var vals2 = rng2.step()

    # Different subsequences should produce different values
    var all_equal = True
    for i in range(4):
        if vals1[i] != vals2[i]:
            all_equal = False
            break

    assert_false(
        all_equal, "Different subsequences should produce different sequences"
    )


def test_philox_offset() raises:
    """Test that offset parameter skips ahead in the sequence."""
    var rng1 = Random(seed=42, subsequence=0, offset=0)
    var rng2 = Random(seed=42, subsequence=0, offset=2)

    # rng1 should generate first values, then match rng2 after 2 steps
    _ = rng1.step()  # offset 0
    _ = rng1.step()  # offset 1
    var vals1 = rng1.step()  # offset 2
    var vals2 = rng2.step()  # starts at offset 2

    # These should be equal since both are at offset 2
    for i in range(4):
        assert_equal(
            vals1[i],
            vals2[i],
            String("Values at same offset should match at index ", i),
        )


def test_philox_uniform_range() raises:
    """Test that uniform values are in the correct range."""
    var rng = Random(seed=42, subsequence=0, offset=0)

    # Generate many values and verify they're all in [0, 1)
    for _ in range(100):
        var vals = rng.step_uniform()
        for i in range(4):
            assert_true(
                vals[i] >= 0.0,
                String("Uniform value should be >= 0.0, got ", vals[i]),
            )
            assert_true(
                vals[i] < 1.0,
                String("Uniform value should be < 1.0, got ", vals[i]),
            )


def test_philox_uniform_distribution() raises:
    """Test that uniform values are reasonably distributed."""
    var rng = Random(seed=42, subsequence=0, offset=0)

    # Generate many values and compute mean
    var num_samples = 1000
    var sum: Float64 = 0.0
    for _ in range(num_samples):
        var vals = rng.step_uniform()
        for i in range(4):
            sum += Float64(vals[i])

    var mean = sum / Float64((num_samples * 4))

    # For uniform distribution in [0, 1), mean should be close to 0.5
    # Use a generous tolerance since we're not generating that many samples
    var tolerance = 0.05
    var mean_error = abs(mean - 0.5)
    assert_true(
        mean_error < tolerance,
        String(
            "Mean of uniform distribution should be close to 0.5, got ",
            mean,
            " (error: ",
            mean_error,
            ")",
        ),
    )


def test_philox_different_rounds() raises:
    """Test that different rounds parameter works."""
    var rng6 = Random[rounds=6](seed=42, subsequence=0, offset=0)
    var rng10 = Random[rounds=10](seed=42, subsequence=0, offset=0)

    var vals6 = rng6.step()
    var vals10 = rng10.step()

    # Different rounds should produce different values (with same seed)
    var all_equal = True
    for i in range(4):
        if vals6[i] != vals10[i]:
            all_equal = False
            break

    assert_false(
        all_equal, "Different rounds should produce different sequences"
    )


def test_normal_basic() raises:
    """Test basic NormalRandom functionality."""
    var rng = NormalRandom(seed=42, subsequence=0, offset=0)

    # Generate normal random values
    var vals = rng.step_normal(mean=0.0, stddev=1.0)

    # Verify we get 8 values
    assert_equal(len(vals), 8, "step_normal() should return 8 values")


def test_normal_mean_stddev() raises:
    """Test that NormalRandom respects mean and stddev parameters."""
    var rng = NormalRandom(seed=42, subsequence=0, offset=0)

    # Test with specific mean and stddev
    comptime test_mean: Float32 = 10.0
    comptime test_stddev: Float32 = 2.0

    # Generate many samples
    var num_samples = 500
    var sum: Float64 = 0.0
    var sum_sq: Float64 = 0.0

    for _ in range(num_samples):
        var vals = rng.step_normal(mean=test_mean, stddev=test_stddev)
        for i in range(8):
            var val = Float64(vals[i])
            sum += val
            sum_sq += val * val

    var total_count = num_samples * 8
    var observed_mean = sum / Float64(total_count)
    var observed_variance = (sum_sq / Float64(total_count)) - (
        observed_mean * observed_mean
    )
    var observed_stddev = sqrt(observed_variance)

    # Check mean is close to expected
    var mean_tolerance = 0.3
    var mean_error = abs(observed_mean - Float64(test_mean))
    assert_true(
        mean_error < mean_tolerance,
        String(
            "Mean should be close to ",
            test_mean,
            ", got ",
            observed_mean,
            " (error: ",
            mean_error,
            ")",
        ),
    )

    # Check stddev is close to expected
    var stddev_tolerance = 0.3
    var stddev_error = abs(observed_stddev - Float64(test_stddev))
    assert_true(
        stddev_error < stddev_tolerance,
        String(
            "Stddev should be close to ",
            test_stddev,
            ", got ",
            observed_stddev,
            " (error: ",
            stddev_error,
            ")",
        ),
    )


def test_normal_reproducibility() raises:
    """Test that NormalRandom is reproducible with the same seed."""
    var rng1 = NormalRandom(seed=42, subsequence=0, offset=0)
    var rng2 = NormalRandom(seed=42, subsequence=0, offset=0)

    var vals1_1 = rng1.step_normal(mean=0.0, stddev=1.0)
    var vals1_2 = rng1.step_normal(mean=0.0, stddev=1.0)
    var vals2_1 = rng2.step_normal(mean=0.0, stddev=1.0)
    var vals2_2 = rng2.step_normal(mean=0.0, stddev=1.0)

    # Verify sequences are identical
    for i in range(8):
        assert_equal(
            vals1_1[i],
            vals2_1[i],
            String("First step normal values should match at index ", i),
        )
        assert_equal(
            vals1_2[i],
            vals2_2[i],
            String("Second step normal values should match at index ", i),
        )


def test_philox_sequence_independence() raises:
    """Test that consecutive calls produce independent values."""
    var rng = Random(seed=42, subsequence=0, offset=0)

    var vals1 = rng.step()
    var vals2 = rng.step()
    var vals3 = rng.step()

    # All three sets should be different
    var vals1_eq_vals2 = True
    var vals2_eq_vals3 = True
    var vals1_eq_vals3 = True

    for i in range(4):
        if vals1[i] != vals2[i]:
            vals1_eq_vals2 = False
        if vals2[i] != vals3[i]:
            vals2_eq_vals3 = False
        if vals1[i] != vals3[i]:
            vals1_eq_vals3 = False

    assert_false(
        vals1_eq_vals2,
        "Consecutive step() calls should produce different values",
    )
    assert_false(
        vals2_eq_vals3,
        "Consecutive step() calls should produce different values",
    )
    assert_false(
        vals1_eq_vals3,
        "Consecutive step() calls should produce different values",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
