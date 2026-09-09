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

# Issue #31111 -- run this test with -O0 also.

# These tests aren't _great_. They're platform specific, and implementation
# specific. But for now they test behavior and reproducibility.

from std.hashlib import default_comp_time_hasher

from std.testing import assert_equal, assert_not_equal, assert_true
from std.testing import TestSuite

from test_utils.reflection import SimplePoint, NestedStruct, EmptyStruct


def same_low_bits(i1: UInt64, i2: UInt64, bits: Int = 5) raises -> UInt8:
    var mask = (1 << bits) - 1
    return UInt8(Int(not (i1 ^ i2) & UInt64(mask)))


def test_hash_byte_array() raises:
    # Test that values hash deterministically
    assert_equal(hash("a".ptr(), 1), hash("a".ptr(), 1))
    assert_equal(hash("b".ptr(), 1), hash("b".ptr(), 1))
    assert_equal(hash("c".ptr(), 1), hash("c".ptr(), 1))
    assert_equal(hash("d".ptr(), 1), hash("d".ptr(), 1))

    # Test that low bits are different
    var num_same: UInt8 = 0
    num_same += same_low_bits(hash("a".ptr(), 1), hash("b".ptr(), 1))
    num_same += same_low_bits(hash("a".ptr(), 1), hash("c".ptr(), 1))
    num_same += same_low_bits(hash("a".ptr(), 1), hash("d".ptr(), 1))
    num_same += same_low_bits(hash("b".ptr(), 1), hash("c".ptr(), 1))
    num_same += same_low_bits(hash("b".ptr(), 1), hash("d".ptr(), 1))
    num_same += same_low_bits(hash("c".ptr(), 1), hash("d".ptr(), 1))

    # This test is just really bad. We really need to re-evaluate the
    # right way to test these. Hash function behavior varies a bit  based
    # on architecture, so these tests as-is end up being really flaky.
    # Making this _much_ more relaxed for now, but at least still testing
    # that at least the hash function returns _some_ different things.

    # TODO(MSTDL-472): fix this flaky check
    # assert_true(num_same < 6, "too little entropy in hash def low bits")


def _test_hash_int_simd[
    dtype: DType
](bits: Int = 4, max_num_same: Int = 2) raises:
    var a = Scalar[dtype](0)
    var b = Scalar[dtype](1)
    var c = Scalar[dtype](2)
    var d = Scalar[dtype](-1)

    # Test that values hash deterministically
    assert_equal(hash(a), hash(a))
    assert_equal(hash(b), hash(b))
    assert_equal(hash(c), hash(c))
    assert_equal(hash(d), hash(d))

    # Test that low bits are different
    var num_same: UInt8 = 0
    num_same += same_low_bits(hash(a), hash(b), bits)
    num_same += same_low_bits(hash(a), hash(c), bits)
    num_same += same_low_bits(hash(a), hash(d), bits)
    num_same += same_low_bits(hash(b), hash(c), bits)
    num_same += same_low_bits(hash(b), hash(d), bits)
    num_same += same_low_bits(hash(c), hash(d), bits)

    assert_true(
        num_same < UInt8(max_num_same),
        "too little entropy in hash def low bits",
    )


def test_hash_simd() raises:
    _test_hash_int_simd[.int8]()
    _test_hash_int_simd[.int16]()
    _test_hash_int_simd[.int32]()
    _test_hash_int_simd[.int64]()
    # float32 currently has low entropy in the low bits for these test examples.
    # this could affect performance of small dicts some. Let's punt and see
    # if this is an issue in practice, if so we can specialize the float
    # hash implementation.
    _test_hash_int_simd[.float32](max_num_same=7)
    # TODO: test hashing different NaNs.

    # Test a couple other random things
    assert_not_equal(
        hash(Float32(3.14159)),
        hash(Float32(1e10)),
    )
    assert_equal(
        hash(Scalar[.bool](True)),
        hash(Scalar[.bool](True)),
    )
    assert_equal(
        hash(Scalar[.bool](False)),
        hash(Scalar[.bool](False)),
    )
    assert_not_equal(
        hash(Scalar[.bool](True)),
        hash(Scalar[.bool](False)),
    )
    assert_equal(
        hash(SIMD[.bool, 2](fill=True)),
        hash(SIMD[.bool, 2](fill=True)),
    )
    assert_equal(
        hash(SIMD[.bool, 2](fill=False)),
        hash(SIMD[.bool, 2](fill=False)),
    )
    assert_not_equal(
        hash(SIMD[.bool, 2](fill=True)),
        hash(SIMD[.bool, 2](fill=False)),
    )


def _test_hash_signed_zero[dtype: DType]() raises:
    var pos = Scalar[dtype](0.0)
    var neg = Scalar[dtype](-0.0)

    # Keep the check honest: it is vacuous unless the two really do differ in
    # their bit patterns.
    assert_not_equal(pos.to_bits(), neg.to_bits())

    assert_true(pos == neg)
    assert_equal(hash(pos), hash(neg))


def test_hash_signed_zero() raises:
    _test_hash_signed_zero[DType.float8_e3m4]()
    _test_hash_signed_zero[DType.float8_e4m3fn]()
    _test_hash_signed_zero[DType.float8_e5m2]()
    _test_hash_signed_zero[DType.float16]()
    _test_hash_signed_zero[DType.bfloat16]()
    _test_hash_signed_zero[DType.float32]()
    _test_hash_signed_zero[DType.float64]()

    # The sign of zero is normalized elementwise, not just for scalars.
    assert_equal(
        hash(SIMD[DType.float64, 4](0.0, -0.0, 1.5, -1.5)),
        hash(SIMD[DType.float64, 4](-0.0, 0.0, 1.5, -1.5)),
    )

    # Normalizing zero must not fold the sign of anything else away.
    assert_not_equal(hash(Float64(1.5)), hash(Float64(-1.5)))

    # The comptime hasher takes its own `_update_with_simd` path.
    comptime pos = hash[default_comp_time_hasher](Float64(0.0))
    comptime neg = hash[default_comp_time_hasher](Float64(-0.0))
    assert_equal(pos, neg)


def test_hash_exotic_float_vectors() raises:
    """Test that every 8-bit float encoding still lowers, including those with
    no negative zero (`fnuz`) and no zero at all (`e8m0fnu`)."""
    _ = hash(SIMD[DType.float8_e3m4, 4](0))
    _ = hash(SIMD[DType.float8_e4m3fn, 4](0))
    _ = hash(SIMD[DType.float8_e4m3fnuz, 4](0))
    _ = hash(SIMD[DType.float8_e5m2, 4](0))
    _ = hash(SIMD[DType.float8_e5m2fnuz, 4](0))
    _ = hash(SIMD[DType.float8_e8m0fnu, 4](1))


def test_issue_31111() raises:
    _ = hash(Int(1))


def test_hash_comptime() raises:
    comptime hash_123 = hash[default_comp_time_hasher](StaticString("123"))
    assert_equal(hash_123, hash[default_comp_time_hasher](StaticString("123")))

    comptime hash_22 = hash[default_comp_time_hasher](22)
    assert_equal(hash_22, hash[default_comp_time_hasher](22))


@fieldwise_init
struct TestStruct(Hashable):
    var x: StaticString
    var y: Int
    var z: Float32


def test_default_conformance() raises:
    # Test that two instances with the same values hash to the same value
    var a = TestStruct("hello", 42, 3.14)
    var b = TestStruct("hello", 42, 3.14)
    assert_equal(hash(a), hash(b))

    # Test that instances with different values hash to different values
    var c = TestStruct("world", 42, 3.14)
    var d = TestStruct("hello", 43, 3.14)
    var e = TestStruct("hello", 42, 2.71)

    assert_not_equal(hash(a), hash(c))  # different x
    assert_not_equal(hash(a), hash(d))  # different y
    assert_not_equal(hash(a), hash(e))  # different z


def test_default_conformance_deterministic() raises:
    var a = TestStruct("hello", 42, 3.14)
    var b = TestStruct("world", 42, 3.14)
    # Test that hash is deterministic across multiple calls
    assert_equal(hash(a), hash(a))
    assert_equal(hash(b), hash(b))


def test_default_hash_simple() raises:
    """Test the reflection-based default __hash__ with a simple struct."""
    var p1 = SimplePoint(1, 2)
    var p2 = SimplePoint(1, 2)
    var p3 = SimplePoint(1, 3)
    var p4 = SimplePoint(2, 2)

    # Equal values must have equal hashes
    assert_equal(hash(p1), hash(p2))

    # Different values should (usually) have different hashes
    assert_not_equal(hash(p1), hash(p3))
    assert_not_equal(hash(p1), hash(p4))

    # Verify deterministic hashing
    assert_equal(hash(p1), hash(p1))


def test_default_hash_nested() raises:
    """Test the reflection-based default __hash__ with nested structs."""
    var s1 = NestedStruct(SimplePoint(1, 2), "hello")
    var s2 = NestedStruct(SimplePoint(1, 2), "hello")
    var s3 = NestedStruct(SimplePoint(1, 2), "world")
    var s4 = NestedStruct(SimplePoint(3, 4), "hello")

    # Equal values must have equal hashes
    assert_equal(hash(s1), hash(s2))

    # Different values should have different hashes
    assert_not_equal(hash(s1), hash(s3))
    assert_not_equal(hash(s1), hash(s4))


def test_default_hash_empty() raises:
    """Test the reflection-based default __hash__ with an empty struct."""
    var e1 = EmptyStruct()
    var e2 = EmptyStruct()

    # Empty structs should hash to the same value
    assert_equal(hash(e1), hash(e2))


def test_default_hash_equatable_consistency() raises:
    """Test that default __hash__ is consistent with default __eq__."""
    var p1 = SimplePoint(42, 99)
    var p2 = SimplePoint(42, 99)
    var p3 = SimplePoint(42, 100)

    # If p1 == p2, then hash(p1) == hash(p2)
    assert_true(p1 == p2)
    assert_equal(hash(p1), hash(p2))

    # If p1 != p3, hashes should differ (not guaranteed but expected)
    assert_true(p1 != p3)
    assert_not_equal(hash(p1), hash(p3))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
