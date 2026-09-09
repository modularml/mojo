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

"""Tests for the UnsafeUnion type."""

from std.sys import align_of, size_of
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from std.ffi import UnsafeUnion


def test_basic_int_storage() raises:
    var u = UnsafeUnion[Int32, Int64](Int32(42))
    assert_equal(u.unsafe_get[Int32](), 42)

    # Store a different value
    u.unsafe_set(Int32(-100))
    assert_equal(u.unsafe_get[Int32](), -100)


def test_basic_float_storage() raises:
    var u = UnsafeUnion[Float32, Float64](Float32(3.14))
    assert_almost_equal(u.unsafe_get[Float32](), 3.14, atol=0.01)


def test_type_punning() raises:
    # IEEE 754: 1.0f = 0x3F800000 = 1065353216
    var u = UnsafeUnion[Int32, Float32](Float32(1.0))
    assert_equal(u.unsafe_get[Int32](), 1065353216)

    # Reverse: store int, read as float
    var u2 = UnsafeUnion[Int32, Float32](Int32(1065353216))
    assert_almost_equal(u2.unsafe_get[Float32](), 1.0, atol=0.01)


def test_union_size() raises:
    # Union uses !pop.union directly, so size is max of element sizes
    # with no discriminant overhead - exactly like C unions.

    # Single byte type
    assert_equal(size_of[UnsafeUnion[Int8]](), 1)

    # Union of same-size types
    comptime U1 = UnsafeUnion[Int32, Float32]
    assert_equal(size_of[U1](), 4)
    var u1 = U1(Int32(42))
    assert_equal(u1.unsafe_get[Int32](), 42)

    # Union where Int64 is largest
    comptime U2 = UnsafeUnion[Int8, Int16, Int32, Int64]
    assert_equal(size_of[U2](), 8)
    var u2 = U2(Int64.MAX_FINITE)
    assert_equal(u2.unsafe_get[Int64](), Int64.MAX_FINITE)

    # Single element union
    comptime U3 = UnsafeUnion[Int8]
    assert_equal(size_of[U3](), 1)
    var u3 = U3(Int8(127))
    assert_equal(u3.unsafe_get[Int8](), 127)

    # Two-byte type
    assert_equal(size_of[UnsafeUnion[Int16]](), 2)
    assert_equal(size_of[UnsafeUnion[Int16, Int8]](), 2)


def test_union_alignment() raises:
    # Int64 has 8-byte alignment - union should have at least this
    comptime U1 = UnsafeUnion[Int8, Int64]
    assert_true(align_of[U1]() >= 8)

    # All 4-byte aligned types
    comptime U2 = UnsafeUnion[Int32, Float32]
    assert_true(align_of[U2]() >= 4)


def test_unsafe_ptr() raises:
    var u = UnsafeUnion[Int32, Float32](Int32(0))

    # Write through pointer
    var ptr = u.unsafe_ptr[Int32]()
    ptr[] = 999
    assert_equal(u.unsafe_get[Int32](), 999)

    # Read through pointer
    u.unsafe_set(Int32(123))
    assert_equal(ptr[], 123)


def test_unsafe_get() raises:
    var u = UnsafeUnion[Int32, Float32](Int32(42))

    # Read via unsafe_get
    assert_equal(u.unsafe_get[Int32](), 42)

    # Modify via unsafe_get
    u.unsafe_get[Int32]() = 100
    assert_equal(u.unsafe_get[Int32](), 100)


def test_copy() raises:
    var u1 = UnsafeUnion[Int32, Float32](Int32(42))
    var u2 = u1  # Copy via copy ctor
    assert_equal(u2.unsafe_get[Int32](), 42)

    # Modify original, copy should be unchanged
    u1.unsafe_set(Int32(100))
    assert_equal(u2.unsafe_get[Int32](), 42)
    assert_equal(u1.unsafe_get[Int32](), 100)


def test_move() raises:
    var u1 = UnsafeUnion[Int32, Float32](Int32(42))
    var u2 = u1^  # Move via move constructor
    assert_equal(u2.unsafe_get[Int32](), 42)


def test_unsafe_take() raises:
    var u = UnsafeUnion[Int32, Float32](Int32(42))
    var val = u.unsafe_take[Int32]()
    assert_equal(val, 42)

    # After take, must reinitialize before use
    u.unsafe_set(Float32(3.14))
    assert_almost_equal(u.unsafe_get[Float32](), 3.14, atol=0.01)


def test_multiple_types() raises:
    comptime BigUnion = UnsafeUnion[Int8, Int16, Int32, Int64, Float32, Float64]

    # Test storing and retrieving each type
    var u = BigUnion(Int8(127))
    assert_equal(u.unsafe_get[Int8](), 127)

    u.unsafe_set(Int64.MAX_FINITE)
    assert_equal(u.unsafe_get[Int64](), Int64.MAX_FINITE)

    u.unsafe_set(Float64(2.718281828))
    assert_almost_equal(u.unsafe_get[Float64](), 2.718281828, atol=0.01)


def test_explicit_construction() raises:
    # UnsafeUnion requires explicit construction (no @implicit)
    # This is intentional for safety - users must be explicit about unsafe types
    var u = UnsafeUnion[Int32, Float32](Int32(42))
    assert_equal(u.unsafe_get[Int32](), 42)


def test_uninitialized() raises:
    var u = UnsafeUnion[Int32, Float32](unsafe_uninitialized=())
    # Set before reading
    u.unsafe_set(Int32(42))
    assert_equal(u.unsafe_get[Int32](), 42)


def test_simd_types() raises:
    """Test union with SIMD types as members.

    SIMD types are common in FFI for vectorized operations.
    """
    # SIMD[DType.float32, 4] is a 128-bit vector (16 bytes)
    comptime SimdUnion = UnsafeUnion[SIMD[.float32, 4], Int32]

    # Size should be max(16, 4) = 16 bytes
    assert_equal(size_of[SimdUnion](), 16)

    # Create with SIMD value
    var vec = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    var u = SimdUnion(vec)

    # Retrieve and verify
    var result = u.unsafe_get[SIMD[.float32, 4]]()
    assert_equal(result[0], 1.0)
    assert_equal(result[1], 2.0)
    assert_equal(result[2], 3.0)
    assert_equal(result[3], 4.0)

    # Can also store scalar
    u.unsafe_set(Int32(42))
    assert_equal(u.unsafe_get[Int32](), 42)


def test_simd_type_punning() raises:
    # Store 4 floats and read as integers
    comptime SimdIntUnion = UnsafeUnion[SIMD[.float32, 4], SIMD[.int32, 4]]

    var floats = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    var u = SimdIntUnion(floats)

    # Type pun to int32x4 - IEEE 754 representation
    var ints = u.unsafe_get[SIMD[.int32, 4]]()
    # 1.0f = 0x3F800000 = 1065353216
    assert_equal(ints[0], 1065353216)


@fieldwise_init
struct TrivialPair(ImplicitlyCopyable, Movable):
    """A trivial struct for testing."""

    var a: Int32
    var b: Int32


@fieldwise_init
struct TrivialSingle(ImplicitlyCopyable, Movable):
    """A single-field trivial struct."""

    var value: Int64


def test_trivial_struct_types() raises:
    comptime TrivialUnion = UnsafeUnion[TrivialPair, TrivialSingle]

    # TrivialPair is 8 bytes (2x Int32), TrivialSingle is 8 bytes (1x Int64)
    assert_equal(size_of[TrivialUnion](), 8)

    # Store and retrieve TrivialPair
    var u = TrivialUnion(TrivialPair(10, 20))
    var pair = u.unsafe_get[TrivialPair]()
    assert_equal(pair.a, 10)
    assert_equal(pair.b, 20)

    # Store and retrieve TrivialSingle
    u.unsafe_set(TrivialSingle(Int64.MAX_FINITE))
    var single = u.unsafe_get[TrivialSingle]()
    assert_equal(single.value, Int64.MAX_FINITE)


def test_writable() raises:
    var u = UnsafeUnion[Int32, Float32](Int32(42))
    var s = String(u)

    # Should contain the union prefix and size/alignment info
    assert_true("UnsafeUnion[" in s)
    assert_true("size=" in s)
    assert_true("align=" in s)
    # Verify size and alignment values are present (4 bytes for Int32/Float32)
    assert_true("size=4" in s)
    assert_true("align=4" in s)


def test_nested_unions() raises:
    # Two-level nesting
    comptime InnerUnion = UnsafeUnion[Int32, Float32]
    comptime OuterUnion = UnsafeUnion[InnerUnion, Int64]

    # Inner union is 4 bytes, Int64 is 8 bytes, so outer is 8 bytes
    assert_equal(size_of[InnerUnion](), 4)
    assert_equal(size_of[OuterUnion](), 8)

    # Store an inner union (use ^ to transfer ownership)
    var inner = InnerUnion(Int32(42))
    var outer = OuterUnion(inner^)

    # Retrieve inner union via unsafe_get (returns reference) and access value
    # Use ref to get a reference without copying
    ref retrieved_inner = outer.unsafe_get[InnerUnion]()
    assert_equal(retrieved_inner.unsafe_get[Int32](), 42)

    # Store Int64 directly in outer
    outer.unsafe_set(Int64(0x123456789ABCDEF0))
    assert_equal(outer.unsafe_get[Int64](), 0x123456789ABCDEF0)

    # Three-level nesting - sizes cascade: 2, 4, 8 bytes
    comptime Level1 = UnsafeUnion[Int8, Int16]
    comptime Level2 = UnsafeUnion[Level1, Int32]
    comptime Level3 = UnsafeUnion[Level2, Int64]

    assert_equal(size_of[Level1](), 2)
    assert_equal(size_of[Level2](), 4)
    assert_equal(size_of[Level3](), 8)

    # Store and retrieve through three levels
    var l1 = Level1(Int16(1000))
    var l2 = Level2(l1^)
    var l3 = Level3(l2^)

    ref retrieved_l2 = l3.unsafe_get[Level2]()
    ref retrieved_l1 = retrieved_l2.unsafe_get[Level1]()
    assert_equal(retrieved_l1.unsafe_get[Int16](), 1000)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
