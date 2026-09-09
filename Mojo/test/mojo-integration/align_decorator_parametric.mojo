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

# RUN: %mojo -debug-level full %s

# Integration tests for parametric @align decorator - verifies runtime behavior
# when alignment is specified via a struct parameter.

from std.sys import align_of
from std.memory import Pointer, alloc, dealloc
from std.testing import assert_equal, assert_true, TestSuite


@align(alignment)
struct AlignedBuffer[alignment: Int](Movable where False):
    var data: Int

    def __init__(out self):
        self.data = 0


@align(alignment)
struct AlignedTrivialParam[alignment: Int](TrivialRegisterPassable):
    var value: Int


def test_parametric_align() raises:
    """Test that align_of[T]() reflects parametric @align values."""
    assert_equal(align_of[AlignedBuffer[64]](), 64)
    assert_equal(align_of[AlignedBuffer[128]](), 128)
    assert_equal(align_of[AlignedBuffer[256]](), 256)
    assert_equal(align_of[AlignedBuffer[4096]](), 4096)


def test_parametric_align_stack() raises:
    """Test that stack allocations respect parametric @align."""
    # Test 64-byte alignment
    var buf64 = AlignedBuffer[64]()
    var addr64 = Int(Pointer(to=buf64))
    assert_true(
        (addr64 & 63) == 0, "AlignedBuffer[64] should be 64-byte aligned"
    )

    # Test 128-byte alignment
    var buf128 = AlignedBuffer[128]()
    var addr128 = Int(Pointer(to=buf128))
    assert_true(
        (addr128 & 127) == 0, "AlignedBuffer[128] should be 128-byte aligned"
    )

    # Test 256-byte alignment
    var buf256 = AlignedBuffer[256]()
    var addr256 = Int(Pointer(to=buf256))
    assert_true(
        (addr256 & 255) == 0, "AlignedBuffer[256] should be 256-byte aligned"
    )


def test_parametric_align_trivial() raises:
    """Test parametric alignment on RegisterPassable structs."""
    assert_equal(align_of[AlignedTrivialParam[32]](), 32)
    assert_equal(align_of[AlignedTrivialParam[64]](), 64)


def test_different_instantiations() raises:
    """Test that different instantiations have correct independent alignment."""
    # Create multiple instantiations with different alignments
    comptime A32 = AlignedBuffer[32]
    comptime A64 = AlignedBuffer[64]
    comptime A128 = AlignedBuffer[128]

    # Each should have its own alignment
    assert_equal(align_of[A32](), 32)
    assert_equal(align_of[A64](), 64)
    assert_equal(align_of[A128](), 128)


@align(alignment)
struct Outer[alignment: Int](Movable where False):
    """Struct that contains another parametrically-aligned struct."""

    var inner: AlignedBuffer[Self.alignment]

    def __init__(out self):
        self.inner = AlignedBuffer[Self.alignment]()


def test_nested_parametric_align() raises:
    """Test nested structs with parametric alignment."""
    # Outer struct should have the same alignment as its inner parametric struct
    assert_equal(align_of[Outer[64]](), 64)
    assert_equal(align_of[Outer[128]](), 128)

    # Verify stack allocation is aligned
    var outer = Outer[64]()
    var addr = Int(Pointer(to=outer))
    assert_true((addr & 63) == 0, "Outer[64] should be 64-byte aligned")


def test_parametric_align_default() raises:
    """Test parametric @align(1) which is the default alignment."""
    # @align(1) is the default and should use natural alignment (8 for Int).
    assert_equal(align_of[AlignedBuffer[1]](), 8)
    assert_equal(align_of[AlignedTrivialParam[1]](), 8)

    # Verify the struct can be instantiated and used normally.
    var buf = AlignedBuffer[1]()
    var addr = Int(Pointer(to=buf))
    # Natural alignment of Int (8 bytes) should be respected.
    assert_true((addr & 7) == 0, "AlignedBuffer[1] should be 8-byte aligned")


struct ContainsParametricAligned[alignment: Int](Movable where False):
    """Container with a parametrically-aligned field as the second member."""

    var first: Int
    var second: AlignedBuffer[Self.alignment]

    def __init__(out self, first: Int):
        self.first = first
        self.second = AlignedBuffer[Self.alignment]()


def test_parametric_field_offset_alignment() raises:
    """Test that fields with parametric @align are at correct offsets.

    When a struct has a field with parametric @align(N), that field should be
    placed at an offset that satisfies its alignment requirement. This exercises
    the padding insertion and remapped field indices with parametric alignment.
    """
    var container = ContainsParametricAligned[64](99)

    var base_ptr = Pointer(to=container)
    var second_ptr = Pointer(to=container.second)

    var base_addr = Int(base_ptr)
    var second_addr = Int(second_ptr)
    var offset = second_addr - base_addr

    # The AlignedBuffer[64] field should be at offset 64, not 8.
    # The first field (Int) takes 8 bytes, then 56 bytes of padding are needed
    # to align the second field to a 64-byte boundary.
    assert_equal(offset, 64, "AlignedBuffer[64] field should be at offset 64")

    # The second field address should be 64-byte aligned.
    assert_true(
        (second_addr & 63) == 0, "second field should be 64-byte aligned"
    )

    # Verify we can access the field values correctly.
    assert_equal(container.first, 99, "first field should have value 99")

    # Test with a different alignment parameter.
    var container128 = ContainsParametricAligned[128](42)
    var base128 = Int(Pointer(to=container128))
    var second128 = Int(Pointer(to=container128.second))
    assert_equal(
        second128 - base128,
        128,
        "AlignedBuffer[128] field should be at offset 128",
    )


def test_parametric_align_heap() raises:
    """Test that heap allocations respect parametric @align."""
    # Allocate on the heap with 64-byte alignment.
    var ptr64_alloc = alloc[AlignedBuffer[64]]({count = 1}).into_managed()
    var ptr64 = ptr64_alloc.unsafe_ptr()
    var addr64 = Int(ptr64)
    assert_true(
        (addr64 & 63) == 0, "Heap AlignedBuffer[64] should be 64-byte aligned"
    )
    dealloc(ptr64_alloc^)

    # Allocate on the heap with 128-byte alignment.
    var ptr128_alloc = alloc[AlignedBuffer[128]]({count = 1}).into_managed()
    var ptr128 = ptr128_alloc.unsafe_ptr()
    var addr128 = Int(ptr128)
    assert_true(
        (addr128 & 127) == 0,
        "Heap AlignedBuffer[128] should be 128-byte aligned",
    )
    dealloc(ptr128_alloc^)

    # Allocate on the heap with 256-byte alignment.
    var ptr256_alloc = alloc[AlignedBuffer[256]]({count = 1}).into_managed()
    var ptr256 = ptr256_alloc.unsafe_ptr()
    var addr256 = Int(ptr256)
    assert_true(
        (addr256 & 255) == 0,
        "Heap AlignedBuffer[256] should be 256-byte aligned",
    )
    dealloc(ptr256_alloc^)

    # Test multiple separate allocations to verify each respects alignment.
    for _ in range(4):
        var ptr_alloc = alloc[AlignedBuffer[64]]({count = 1}).into_managed()
        var ptr = ptr_alloc.unsafe_ptr()
        var addr = Int(ptr)
        assert_true(
            (addr & 63) == 0,
            "Each separate heap allocation should be 64-byte aligned",
        )
        dealloc(ptr_alloc^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
