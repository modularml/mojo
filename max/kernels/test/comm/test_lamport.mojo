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
"""Host (CPU) unit tests for the Lamport negative-zero sentinel primitives.

These exercise pure SIMD/integer logic and require no GPU: the sentinel map,
the readiness poll, the all-sentinel pack, and the generation rotation.
"""

from std.math import inf, nan
from std.memory import bitcast
from std.sys import bit_width_of
from std.builtin.dtype import _unsigned_integral_type_of

from comm.lamport import (
    LamportGeneration,
    has_neg_zero,
    remove_neg_zero,
    set_neg_zero,
)

from std.testing import assert_equal, assert_false, assert_true


@always_inline
def _bits[
    dtype: DType, width: Int
](v: SIMD[dtype, width]) -> SIMD[_unsigned_integral_type_of[dtype](), width]:
    """Raw unsigned-integer bits of a float pack, for bit-exact comparison."""
    return bitcast[_unsigned_integral_type_of[dtype](), width](v)


@always_inline
def _neg_zero[dtype: DType]() -> Scalar[dtype]:
    """A single `-0.0` scalar of `dtype`, built from raw bits."""
    comptime uint = _unsigned_integral_type_of[dtype]()
    var sentinel = Scalar[uint](1) << Scalar[uint](bit_width_of[uint]() - 1)
    return bitcast[dtype, 1](SIMD[uint, 1](sentinel))[0]


def _assert_bits_equal[
    dtype: DType, width: Int
](a: SIMD[dtype, width], b: SIMD[dtype, width]) raises:
    """Asserts two packs are bit-for-bit identical (NaN-safe, signed-zero-aware).
    """
    assert_equal(_bits(a), _bits(b))


# ===----------------------------------------------------------------------=== #
# set_neg_zero
# ===----------------------------------------------------------------------=== #


def test_set_neg_zero[dtype: DType, width: Int]() raises:
    var pack = set_neg_zero[dtype, width]()

    assert_true(has_neg_zero(pack))
    assert_false(has_neg_zero(remove_neg_zero(pack)))


# ===----------------------------------------------------------------------=== #
# remove_neg_zero — sentinel -> +0.0
# ===----------------------------------------------------------------------=== #


def test_remove_neg_zero_maps_to_plus_zero[dtype: DType, width: Int]() raises:
    var pack = set_neg_zero[dtype, width]()
    var sanitized = remove_neg_zero(pack)

    # Bit pattern must be exactly +0.0 (all-zero bits), not just any zero.
    var plus_zero = SIMD[dtype, width](0)
    _assert_bits_equal(sanitized, plus_zero)
    # And numerically equal to zero in every lane.
    assert_true(Bool(sanitized.eq(plus_zero).reduce_and()))

    # Sanitized data can never be mistaken for the sentinel.
    assert_false(has_neg_zero(sanitized))


# ===----------------------------------------------------------------------=== #
# remove_neg_zero — no-op on data with no -0.0
# ===----------------------------------------------------------------------=== #


def test_remove_neg_zero_noop[dtype: DType, width: Int]() raises:
    # A pack with +0.0, NaN, Inf, and normal values, but no -0.0.
    var data = SIMD[dtype, width](Scalar[dtype](0))  # start at +0.0
    # Seed a few distinctive lanes; pattern repeats harmlessly for small width.
    data[0] = Scalar[dtype](0)  # +0.0
    data[width - 1] = Scalar[dtype](3.5)
    comptime if width >= 4:
        data[1] = nan[dtype]()
        data[2] = inf[dtype]()
        data[3] = Scalar[dtype](-1.25)

    # No -0.0 present -> has_neg_zero false, remove is bit-for-bit identity.
    assert_false(has_neg_zero(data))
    var sanitized = remove_neg_zero(data)
    _assert_bits_equal(sanitized, data)


# ===----------------------------------------------------------------------=== #
# remove_neg_zero — idempotent
# ===----------------------------------------------------------------------=== #


def test_remove_neg_zero_idempotent[dtype: DType, width: Int]() raises:
    var pack = set_neg_zero[dtype, width]()
    var once = remove_neg_zero(pack)
    var twice = remove_neg_zero(once)
    _assert_bits_equal(twice, once)


# ===----------------------------------------------------------------------=== #
# Mixed / torn pack — only sentinel lanes change
# ===----------------------------------------------------------------------=== #


def test_remove_neg_zero_mixed[dtype: DType, width: Int]() raises:
    comptime assert width >= 2, "mixed test needs at least 2 lanes"

    var neg_zero = _neg_zero[dtype]()
    var data = SIMD[dtype, width](Scalar[dtype](2.0))
    # Make lane 0 a sentinel (the "torn read" / partial-write case).
    data[0] = neg_zero

    # has_neg_zero must detect the partial/torn case.
    assert_true(has_neg_zero(data))

    var sanitized = remove_neg_zero(data)

    # Expected: lane 0 becomes +0.0, every real lane bit-preserved.
    var expected = data
    expected[0] = Scalar[dtype](0)
    _assert_bits_equal(sanitized, expected)

    # After sanitize, no sentinel remains.
    assert_false(has_neg_zero(sanitized))


# ===----------------------------------------------------------------------=== #
# LamportGeneration rotation
# ===----------------------------------------------------------------------=== #


def test_lamport_generation() raises:
    # For flag = 0..7: data = flag%3, clear = (flag+2)%3, and data != clear.
    for flag in range(8):
        assert_equal(LamportGeneration.data_index(flag), flag % 3)
        assert_equal(LamportGeneration.clear_index(flag), (flag + 2) % 3)
        # One-generation-of-slack invariant: never clear what we read/write now.
        assert_true(
            LamportGeneration.data_index(flag)
            != LamportGeneration.clear_index(flag)
        )


# ===----------------------------------------------------------------------=== #
# Driver — run every check across bf16/fp16/fp32 and a couple of widths,
# including the natural 128-bit pack width for each dtype.
# ===----------------------------------------------------------------------=== #


def run_all_for[dtype: DType, width: Int]() raises:
    test_set_neg_zero[dtype, width]()
    test_remove_neg_zero_maps_to_plus_zero[dtype, width]()
    test_remove_neg_zero_noop[dtype, width]()
    test_remove_neg_zero_idempotent[dtype, width]()
    test_remove_neg_zero_mixed[dtype, width]()


def main() raises:
    # 16-bit dtypes: 128-bit pack == 8 lanes; also exercise width 4.
    run_all_for[.bfloat16, 4]()
    run_all_for[.bfloat16, 8]()
    run_all_for[.float16, 4]()
    run_all_for[.float16, 8]()

    # 32-bit dtype: 128-bit pack == 4 lanes; also exercise width 8.
    run_all_for[.float32, 4]()
    run_all_for[.float32, 8]()

    test_lamport_generation()

    print("All Lamport sentinel primitive tests passed.")
