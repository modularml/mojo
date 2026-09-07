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
"""Provides utilities for testing hash function quality and distribution."""

from std.hashlib.hasher import Hasher

from std.bit import pop_count
from std.reflection import call_location
from std.testing import assert_true


def dif_bits(i1: UInt64, i2: UInt64) raises -> Int:
    """Computes the number of differing bits between two integers.

    Args:
        i1: First integer.
        i2: Second integer.

    Returns:
        The number of bits that differ between the two integers.

    Raises:
        Error: if an error occurs during computation.
    """
    return Int(pop_count(i1 ^ i2))


@always_inline
def assert_dif_hashes(hashes: List[UInt64], upper_bound: Int) raises:
    """Asserts that all pairs of hashes differ by more than the upper bound.

    Args:
        hashes: List of hash values to compare.
        upper_bound: Minimum number of differing bits required between hashes.

    Raises:
        Error: if any pair of hashes differs by fewer than `upper_bound` bits.
    """
    var min_diff = 64
    var max_diff = 0
    var total_diff = 0
    var comparisons = 0

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            var diff = dif_bits(hashes[i], hashes[j])
            min_diff = min(min_diff, diff)
            max_diff = max(max_diff, diff)
            total_diff += diff
            comparisons += 1

            if diff <= upper_bound:
                var avg_diff = Float64(total_diff) / Float64(comparisons)
                print(
                    "Hash difference check failed!\n"
                    "  Expected: > {} differing bits\n"
                    "  Got: {} differing bits at indices {}:{}\n"
                    "  Hash[{}] = {} ({})\n"
                    "  Hash[{}] = {} ({})\n"
                    "  XOR = {}\n"
                    "  Statistics over {} comparisons:\n"
                    "    Min diff: {} bits\n"
                    "    Max diff: {} bits\n"
                    "    Avg diff: {} bits".format(
                        upper_bound,
                        diff,
                        i,
                        j,
                        i,
                        hashes[i],
                        hex(hashes[i]),
                        j,
                        hashes[j],
                        hex(hashes[j]),
                        hex(hashes[i] ^ hashes[j]),
                        comparisons,
                        min_diff,
                        max_diff,
                        avg_diff,
                    )
                )
                assert_true(
                    False,
                    "Hash difference threshold violated (see details above)",
                    location=call_location(),
                )


@always_inline
def assert_fill_factor[
    label: String, HasherType: Hasher
](words: List[String], num_buckets: Int, lower_bound: Float64) raises:
    """Asserts that the hash function achieves a minimum fill factor.

    Parameters:
        label: Label for the test output.
        HasherType: Type of hasher to use.

    Args:
        words: List of strings to hash.
        num_buckets: Number of buckets to distribute hashes into.
        lower_bound: Minimum required fill factor (0.0 to 1.0).

    Raises:
        Error: if the achieved fill factor is below `lower_bound`.
    """
    # A perfect hash function is when the number of buckets is equal to number of words
    # and the fill factor results in 1.0
    var buckets = List([0]) * num_buckets
    var hash_samples = List[UInt64]()

    for idx, w in enumerate(words):
        var h = hash[HasherType](w)
        buckets[Int(h % UInt64(num_buckets))] += 1

        # Collect first 5 hash samples for debugging
        if idx < 5:
            hash_samples.append(h)

    var unfilled = 0
    var max_collisions = 0
    var total_items = 0

    for v in buckets:
        if v == 0:
            unfilled += 1
        else:
            max_collisions = max(max_collisions, v)
            total_items += v

    var filled = num_buckets - unfilled
    var fill_factor = 1.0 - Float64(unfilled) / Float64(num_buckets)

    if fill_factor < lower_bound:
        print(
            "Fill factor check failed for {}!\n"
            "  Expected fill factor: >= {}\n"
            "  Actual fill factor: {}\n"
            "  Total words: {}\n"
            "  Bucket stats:\n"
            "    Total buckets: {}\n"
            "    Filled buckets: {} ({}%)\n"
            "    Unfilled buckets: {} ({}%)\n"
            "    Max collisions in a bucket: {}\n"
            "    Avg items per filled bucket: {}\n"
            "  Sample hash values (first 5):\n"
            "    [0]: {} ({})\n"
            "    [1]: {} ({})\n"
            "    [2]: {} ({})\n"
            "    [3]: {} ({})\n"
            "    [4]: {} ({})".format(
                label,
                lower_bound,
                fill_factor,
                len(words),
                num_buckets,
                filled,
                Float64(filled) / Float64(num_buckets) * 100.0,
                unfilled,
                Float64(unfilled) / Float64(num_buckets) * 100.0,
                max_collisions,
                Float64(total_items) / Float64(filled) if filled > 0 else 0.0,
                hash_samples[0],
                hex(hash_samples[0]),
                hash_samples[1],
                hex(hash_samples[1]),
                hash_samples[2],
                hex(hash_samples[2]),
                hash_samples[3],
                hex(hash_samples[3]),
                hash_samples[4],
                hex(hash_samples[4]),
            )
        )

    assert_true(
        fill_factor >= lower_bound,
        "Fill factor threshold violated (see details above)",
        location=call_location(),
    )
