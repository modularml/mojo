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

# Integration tests for @align decorator - verifies runtime behavior.

from std.sys import align_of, size_of
from std.memory import Pointer, alloc, dealloc
from std.testing import assert_equal, assert_true, TestSuite
from std.collections import Optional


# Basic aligned struct
@align(64)
struct CacheAligned(Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


# @align works on single-element trivial register structs. When @align is
# specified, the struct is NOT flattened to its element type during lowering,
# preserving the alignment metadata.
@align(32)
struct AlignedTrivial(TrivialRegisterPassable):
    var value: Int

    @always_inline
    def __init__(out self, value: Int):
        self.value = value


# Struct for testing compile-time constants with alignment padding.
# Uses TrivialRegisterPassable so it can be used in compile-time contexts.
struct ContainsAlignedTrivialSecond(TrivialRegisterPassable):
    var first: Int
    var second: AlignedTrivial  # @align(32) requires padding after first (8 bytes)

    @always_inline
    def __init__(out self, first: Int, second: AlignedTrivial):
        self.first = first
        self.second = second


# Large alignment
@align(4096)
struct PageAligned:
    var data: Int

    def __init__(out self, data: Int):
        self.data = data


# Struct containing an aligned struct (should inherit alignment)
struct ContainsAligned:
    var inner: CacheAligned
    var other: Int

    def __init__(out self, var inner: CacheAligned, other: Int):
        self.inner = inner^
        self.other = other


# Struct containing an aligned struct as second member (MOCO-3167 test case)
# The aligned field should be at offset 64, not offset 8
struct ContainsAlignedSecond:
    var other: Int
    var inner: CacheAligned

    def __init__(out self, other: Int, var inner: CacheAligned):
        self.other = other
        self.inner = inner^


# Generic struct with alignment
@align(128)
struct AlignedGeneric[T: TrivialRegisterPassable]:
    var value: Self.T

    def __init__(out self, value: Self.T):
        self.value = value


# Nested alignment - outer has smaller alignment than inner's requirement
@align(16)
struct OuterSmallAlign:
    var inner: CacheAligned  # CacheAligned requires 64-byte alignment

    def __init__(out self, var inner: CacheAligned):
        self.inner = inner^


# Test cross-struct references: UsesLaterStruct is defined before LaterAlignedStruct.
# This tests that alignment lookup works for structs not yet lowered.
struct UsesLaterStruct:
    """A struct whose method creates a local of a later-defined aligned struct.
    """

    @staticmethod
    def create_later() -> Int:
        """Create a local variable of LaterAlignedStruct and verify alignment.
        """
        var later = LaterAlignedStruct(123)
        var ptr = Pointer(to=later)
        var addr = Int(ptr)
        # Return 1 if aligned, 0 if not
        return 1 if (addr & 255) == 0 else 0


@align(256)
struct LaterAlignedStruct:
    """An aligned struct defined after UsesLaterStruct."""

    var value: Int

    def __init__(out self, value: Int):
        self.value = value


def test_align_of() raises:
    """Test that align_of[T]() reflects the @align decorator."""
    assert_equal(align_of[CacheAligned](), 64)
    assert_equal(align_of[AlignedTrivial](), 32)
    assert_equal(align_of[PageAligned](), 4096)
    assert_equal(align_of[AlignedGeneric[Int]](), 128)


def test_nested_alignment() raises:
    """Test that containing structs inherit alignment from aligned fields."""
    # ContainsAligned should have at least 64-byte alignment due to CacheAligned field
    assert_equal(align_of[ContainsAligned](), 64)

    # OuterSmallAlign specifies @align(16) but contains CacheAligned (64-byte)
    # The actual alignment should be max(16, 64) = 64
    assert_equal(align_of[OuterSmallAlign](), 64)


def test_heap_allocation_alignment() raises:
    """Test that heap-allocated aligned structs are actually aligned at runtime.

    `Layout[T]` uses `align_of[T]()` as the default alignment, so heap
    allocations should respect the @align decorator.
    """
    # Allocate on heap - should be 64-byte aligned
    var heap_ptr_alloc = alloc[CacheAligned]({count = 1}).into_managed()
    var heap_ptr = heap_ptr_alloc.unsafe_ptr()
    var heap_addr = Int(heap_ptr)
    assert_true(
        (heap_addr & 63) == 0, "CacheAligned should be 64-byte aligned on heap"
    )
    dealloc(heap_ptr_alloc^)

    # Large alignment on heap
    var page_ptr_alloc = alloc[PageAligned]({count = 1}).into_managed()
    var page_ptr = page_ptr_alloc.unsafe_ptr()
    var page_addr = Int(page_ptr)
    assert_true(
        (page_addr & 4095) == 0,
        "PageAligned should be 4096-byte aligned on heap",
    )
    dealloc(page_ptr_alloc^)


def test_stack_allocation_alignment() raises:
    """Test that stack-allocated aligned structs respect @align.

    The compiler propagates alignment from @align(N) decorator to the LLVM
    alloca instructions, ensuring stack allocations are properly aligned.
    """
    var cache_aligned = CacheAligned(42)
    var ptr = Pointer(to=cache_aligned)
    var addr = Int(ptr)

    # Stack allocation should respect @align(64)
    assert_true(
        (addr & 63) == 0, "CacheAligned should be 64-byte aligned on stack"
    )

    # Test large alignment on stack (4096-byte aligned)
    var page_aligned = PageAligned(99)
    var page_ptr = Pointer(to=page_aligned)
    var page_addr = Int(page_ptr)
    assert_true(
        (page_addr & 4095) == 0,
        "PageAligned should be 4096-byte aligned on stack",
    )


def test_generic_alignment() raises:
    """Test that alignment works correctly with generic types."""
    # Different instantiations should all have 128-byte alignment
    assert_equal(align_of[AlignedGeneric[Int8]](), 128)
    assert_equal(align_of[AlignedGeneric[Int64]](), 128)
    assert_equal(align_of[AlignedGeneric[SIMD[.float32, 4]]](), 128)


def test_generic_stack_allocation() raises:
    """Test that generic aligned structs are properly aligned on stack."""
    # Stack-allocate a generic aligned struct
    var generic_aligned = AlignedGeneric[Int](42)
    var ptr = Pointer(to=generic_aligned)
    var addr = Int(ptr)

    # Should be 128-byte aligned
    assert_true(
        (addr & 127) == 0,
        "AlignedGeneric[Int] should be 128-byte aligned on stack",
    )


def test_array_alignment() raises:
    """Test array allocation alignment behavior."""
    # Allocate array - base pointer should be 64-byte aligned
    var arr_alloc = alloc[CacheAligned]({count = 4}).into_managed()
    var arr: Pointer[
        CacheAligned, origin_of(arr_alloc)
    ] = arr_alloc.unsafe_ptr()
    var base_addr = Int(arr)
    assert_true(
        (base_addr & 63) == 0,
        "CacheAligned array base should be 64-byte aligned",
    )

    # Elements should also be 64-byte apart; each element is 64-byte aligned.
    var stride = Int(arr.unsafe_offset(1)) - Int(arr)
    assert_equal(stride, 64)

    dealloc(arr_alloc^)


def test_cross_struct_alignment() raises:
    """Test that alignment works when a struct uses a later-defined aligned struct.

    This exercises the code path where we look up alignment from the symbol
    table (struct not yet lowered) rather than from structDecls.
    """
    # This calls a method that creates a LaterAlignedStruct (256-byte aligned)
    # The alignment lookup must work even though UsesLaterStruct is defined
    # before LaterAlignedStruct.
    var result = UsesLaterStruct.create_later()
    assert_equal(result, 1, "LaterAlignedStruct should be 256-byte aligned")


def test_inherited_stack_alignment() raises:
    """Test that stack allocation respects alignment inherited from fields.

    This is the key test for MOCO-3165: a struct containing an @align(64) field
    should be allocated on the stack with 64-byte alignment, even if the
    containing struct has no explicit @align decorator.
    """
    # ContainsAligned has no @align decorator but contains CacheAligned which
    # has @align(64). The containing struct should inherit this alignment
    # requirement for stack allocation.
    var container = ContainsAligned(CacheAligned(42), 99)
    var ptr = Pointer(to=container)
    var addr = Int(ptr)

    # The struct should be 64-byte aligned due to the CacheAligned field
    assert_true(
        (addr & 63) == 0,
        (
            "ContainsAligned should inherit 64-byte alignment from CacheAligned"
            " field"
        ),
    )

    # Also verify the inner field is aligned
    var inner_ptr = Pointer(to=container.inner)
    var inner_addr = Int(inner_ptr)
    assert_true(
        (inner_addr & 63) == 0,
        "ContainsAligned.inner field should be 64-byte aligned",
    )

    # Test OuterSmallAlign which has @align(16) but contains CacheAligned (64)
    # The effective alignment should be max(16, 64) = 64
    var outer = OuterSmallAlign(CacheAligned(77))
    var outer_ptr = Pointer(to=outer)
    var outer_addr = Int(outer_ptr)
    assert_true(
        (outer_addr & 63) == 0,
        "OuterSmallAlign should use max(explicit=16, inherited=64) = 64",
    )


def test_field_offset_alignment() raises:
    """Test that fields within a struct are at correct offsets (MOCO-3167).

    When a struct has a field with @align(N), that field should be placed at
    an offset that satisfies its alignment requirement. Previously, LLVM would
    use natural alignment only, placing the inner field at offset 8 instead of
    offset 64.
    """
    var container = ContainsAlignedSecond(99, CacheAligned(42))

    var base_ptr = Pointer(to=container)
    var inner_ptr = Pointer(to=container.inner)

    var base_addr = Int(base_ptr)
    var inner_addr = Int(inner_ptr)
    var offset = inner_addr - base_addr

    # The CacheAligned field should be at offset 64, not 8
    assert_equal(offset, 64, "CacheAligned field should be at offset 64")

    # The inner field address should be 64-byte aligned
    assert_true((inner_addr & 63) == 0, "inner field should be 64-byte aligned")

    # Verify we can access the values correctly
    assert_equal(container.other, 99, "other field should have value 99")
    assert_equal(container.inner.x, 42, "inner.x field should have value 42")


def test_struct_replace_with_alignment() raises:
    """Test that StructReplace works correctly with padding (MOCO-3167).

    This exercises the ConvertKGENStructReplace pattern which must use
    remapped field indices when the struct has alignment padding.
    """
    var container = ContainsAlignedSecond(99, CacheAligned(42))

    # Modify the first field - exercises StructReplace at logical index 0
    container.other = 123
    assert_equal(container.other, 123, "other field should be updated to 123")

    # Verify inner field is still correct (wasn't corrupted by the replace)
    assert_equal(container.inner.x, 42, "inner.x should still be 42")

    # Verify alignment is preserved after modification
    var inner_ptr = Pointer(to=container.inner)
    var inner_addr = Int(inner_ptr)
    assert_true(
        (inner_addr & 63) == 0,
        "inner field should still be 64-byte aligned after replace",
    )

    # Also test replacing via the inner field (logical index 1, LLVM index 2)
    container.inner = CacheAligned(999)
    assert_equal(container.inner.x, 999, "inner.x should be updated to 999")
    assert_equal(container.other, 123, "other field should still be 123")


def _helper_with_const_default(
    val: ContainsAlignedTrivialSecond = ContainsAlignedTrivialSecond(
        42, AlignedTrivial(99)
    )
) -> ContainsAlignedTrivialSecond:
    """Helper that uses a compile-time constant struct as default parameter.

    The default parameter value exercises convertParameterToLLVM for struct
    constants with alignment padding.
    """
    return val


def test_compile_time_struct_constant() raises:
    """Test that compile-time struct constants work with alignment padding.

    This exercises convertParameterToLLVM which handles StructAttr lowering.
    The default parameter value is a compile-time constant that must be
    correctly lowered with remapped field indices.
    """
    # Use the default parameter (compile-time constant)
    var from_default = _helper_with_const_default()
    assert_equal(
        from_default.first, 42, "first field from default should be 42"
    )
    assert_equal(
        from_default.second.value, 99, "second.value from default should be 99"
    )

    # Verify the constant struct has correct layout when stored
    var ptr = Pointer(to=from_default)
    var base_addr = Int(ptr)
    var second_ptr = Pointer(to=from_default.second)
    var second_addr = Int(second_ptr)
    var offset = second_addr - base_addr

    # AlignedTrivial has @align(32), so second should be at offset 32
    assert_equal(offset, 32, "second field should be at offset 32")
    assert_true(
        (second_addr & 31) == 0, "second field should be 32-byte aligned"
    )


# Struct containing an Optional with an aligned type.
# Tests that Union types (which back Optional) respect @align on variants.
# Uses AlignedTrivial which is TrivialRegisterPassable and thus Copyable.
struct ContainsOptionalAligned:
    var other: Int
    var opt: Optional[AlignedTrivial]

    def __init__(out self, other: Int, opt: Optional[AlignedTrivial]):
        self.other = other
        self.opt = opt


def test_optional_alignment() raises:
    """Test that Optional[T] where T has @align(N) is properly aligned.

    This exercises UnionType::getTypeAlign which must return the max alignment
    of all variant types (AlignedTrivial has @align(32), NoneType has align 1).
    """
    # Verify align_of[Optional[AlignedTrivial]]() reports the correct alignment
    assert_equal(
        align_of[Optional[AlignedTrivial]](),
        32,
        "Optional[AlignedTrivial] should have 32-byte alignment",
    )

    # Create a struct containing the Optional and verify field alignment
    var container = ContainsOptionalAligned(99, AlignedTrivial(42))

    var base_ptr = Pointer(to=container)
    var opt_ptr = Pointer(to=container.opt)

    var base_addr = Int(base_ptr)
    var opt_addr = Int(opt_ptr)
    var offset = opt_addr - base_addr

    # The Optional field should be at offset 32 due to AlignedTrivial's @align(32)
    assert_equal(
        offset, 32, "Optional[AlignedTrivial] field should be at offset 32"
    )
    assert_true(
        (opt_addr & 31) == 0, "Optional field should be 32-byte aligned"
    )

    # Verify we can access the value correctly
    assert_equal(container.other, 99, "other field should have value 99")
    if container.opt:
        assert_equal(
            container.opt.value().value,
            42,
            "opt.value().value should be 42",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
