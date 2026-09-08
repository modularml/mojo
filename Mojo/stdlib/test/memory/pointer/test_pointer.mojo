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

from std.compile import compile_info
from std.ffi import external_call
from max.gpu.host import get_gpu_target
from std.memory import Allocation, MaybeUninit, alloc, dealloc
from std.memory.unsafe_pointer import pointer_to_int
from std.sys import align_of, bit_width_of, size_of
import std.memory.alloc

from test_utils import (
    ExplicitCopyOnly,
    MoveCounter,
    ObservableDel,
    ObservableMoveOnly,
    PinnedExplicitDelOnly,
    check_write_to,
)
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
    TestSuite,
)

# ---------------------------------------------------------------------------- #
# New tests for `Pointer`
# ---------------------------------------------------------------------------- #


def _mutable_pointer(p: MutPointer[Int, ...]) raises:
    assert_equal(p[], 42)


def _immutable_pointer(p: ImmPointer[Int, ...]) raises:
    assert_equal(p[], 42)


def _parameterized_pointer(p: Pointer[Int, ...]) raises:
    assert_equal(p[], 42)


def _named_origin[
    mut: Bool, //, origin: Origin[mut=mut]
](p: Pointer[Int, origin, ...]) raises:
    assert_equal(p[], 42)


def test_mutable_conversions() raises:
    var x = 42
    var p = Pointer(to=x)
    _named_origin[origin_of(x)](p)
    _mutable_pointer(p)
    _immutable_pointer(p)
    _parameterized_pointer(p)


def test_immutable_conversions() raises:
    var x = 42
    var p = Pointer(to=x).as_imm()
    _named_origin[mut=False, origin_of(x)](p)
    _immutable_pointer(p)
    _parameterized_pointer(p)


def test_mutable_any_conversions() raises:
    var x = 42
    var p = Pointer(to=x).as_unsafe_any_origin()
    _mutable_pointer(p)
    _immutable_pointer(p)
    _parameterized_pointer(p)


def test_immutable_any_conversions() raises:
    var x = 42
    var p = Pointer(to=x).as_imm().as_unsafe_any_origin()
    _immutable_pointer(p)
    _parameterized_pointer(p)


# ---------------------------------------------------------------------------- #
# Copied tests from `test_unsafepointer.mojo`
# ---------------------------------------------------------------------------- #


def test_unsafepointer_of_move_only_type() raises:
    var actions = List[String]()
    var actions_ptr = Pointer(to=actions).as_imm()

    comptime ObserveType = ObservableMoveOnly[actions_ptr.origin]

    var allocation = alloc[ObserveType]({count = 1})
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_write(ObserveType(42, actions_ptr))

    # `assert_equal` can raise, and an `Allocation` must be consumed on every
    # path (including the raising one), so capture every value the asserts
    # below need before tearing down the allocation.
    var len_after_write = len(actions_ptr[])
    var action0 = actions_ptr[][0]
    var action1 = actions_ptr[][1]
    var written_value = ptr[].value

    var value = ptr.unsafe_take_pointee()
    var len_after_take = len(actions_ptr[])
    var action2 = actions_ptr[][2]
    var taken_value = value.value

    dealloc(allocation^)

    assert_equal(len_after_write, 2)
    assert_equal(action0, "__init__")
    assert_equal(action1, "move ctor", msg="emplace_value")
    assert_equal(written_value, 42)

    assert_equal(len_after_take, 3)
    assert_equal(action2, "move ctor")
    assert_equal(taken_value, 42)

    assert_equal(len(actions_ptr[]), 4)
    assert_equal(actions_ptr[][3], "__deinit__")


def test_unsafepointer_move_pointee_move_count() raises:
    var ptr = alloc[MoveCounter[Int]]({count = 1}).unsafe_leak()

    var value = MoveCounter(5)
    assert_equal(0, value.move_count)
    ptr.unsafe_write(value^)

    # -----
    # Test that `Pointer.move_pointee` performs exactly one move.
    # -----

    assert_equal(1, ptr[].move_count)

    var ptr_2 = alloc[MoveCounter[Int]]({count = 1}).unsafe_leak()
    ptr_2.unsafe_write_move_from(ptr)

    assert_equal(2, ptr_2[].move_count)


def test_write() raises:
    var x = 0
    var ptr = Pointer(to=x)

    ptr.write(41)
    assert_equal(ptr[], 41)

    ptr.write(42)
    assert_equal(ptr[], 42)


def test_unsafepointer_unsafe_write() raises:
    var ptr = alloc[ExplicitCopyOnly]({count = 1}).unsafe_leak()

    var orig = ExplicitCopyOnly(5)
    assert_equal(orig.copy_count, 0)

    # Test initialize pointee from `Copyable` type
    ptr.unsafe_write(copy=orig)

    assert_equal(ptr[].value, 5)
    assert_equal(ptr[].copy_count, 1)


def test_unsafepointer_unsafe_write_closure() raises:
    var x = 66
    var ptr = Pointer(to=x)
    ptr.unsafe_write(init_with=lambda () -> Int: 42)
    assert_equal(ptr[], 42)


def test_unsafepointer_unsafe_write_non_movable() raises:
    var allocation = alloc[PinnedExplicitDelOnly]({count = 1})
    var ptr = allocation.unsafe_ptr()

    def make() -> PinnedExplicitDelOnly:
        return PinnedExplicitDelOnly(5)

    ptr.unsafe_write(init_with=make)
    var data = ptr[].data

    ptr.unsafe_deinit_pointee_with(PinnedExplicitDelOnly.destroy)
    dealloc(allocation^)
    assert_equal(data, 5)


def test_refitem() raises:
    var allocation = alloc[Int]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr[] = 0
    ptr[] += 1
    assert_equal(ptr[], 1)


def test_refitem_offset() raises:
    var allocation = alloc[Int]({count = 5}).into_managed()
    var ptr = allocation.unsafe_ptr()
    for i in range(5):
        ptr[unsafe_offset=i] = i
    for i in range(5):
        assert_equal(ptr[unsafe_offset=i], i)


def test_address_of() raises:
    var local = 1
    assert_not_equal(0, Int(Pointer[Int](to=local)))
    _ = local


def test_pointer_to() raises:
    var local = 1
    assert_not_equal(0, Pointer(to=local)[])


def test_explicit_copy_of_pointer_address() raises:
    var local = 1
    var ptr = Pointer[Int](to=local)
    var copy = Pointer.copy(ptr)
    assert_equal(Int(ptr), Int(copy))
    _ = local


def test_address_as_integer_scalar() raises:
    var local = 1
    var ptr = Pointer[Int](to=local)
    assert_equal(UInt(ptr), UInt(Int(ptr)))
    assert_equal(UInt64(ptr), UInt64(Int(ptr)))
    assert_equal(Int64(ptr), Int64(Int(ptr)))
    _ = local


def test_bitcast() raises:
    var local = 1
    var ptr = Pointer[Int](to=local)
    var aliased_ptr = ptr.unsafe_bitcast[SIMD[.uint8, 4]]()

    assert_equal(Int(ptr), Int(ptr.unsafe_bitcast[Int]()))

    assert_equal(Int(ptr), Int(aliased_ptr))

    _ = local


def test_unsafepointer_string() raises:
    var allocation = alloc[Int]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    assert_true(String(ptr).startswith("0x"))
    assert_not_equal(String(ptr), "0x0")


def test_eq() raises:
    var local = 1
    # FIXME(#5133): should just be Pointer[mut=False](to=local)
    var p1 = Pointer(to=local).as_imm()
    var p2 = p1
    assert_equal(p1, p2)

    var other_local = 2
    var p3 = Pointer(to=other_local).as_imm()
    assert_not_equal(Int(p1), Int(p3))

    var p4 = Pointer(to=local).as_imm()
    assert_equal(p1, p4)
    _ = local
    _ = other_local


def test_comparisons() raises:
    var allocation = alloc[Int]({count = 1}).into_managed()
    var p1 = allocation.unsafe_ptr()
    assert_true(p1.unsafe_offset(-1) < p1)
    assert_true(p1.unsafe_offset(-1) <= p1)
    assert_true(p1 <= p1)
    assert_true(p1.unsafe_offset(1) > p1)
    assert_true(p1.unsafe_offset(1) >= p1)
    assert_true(p1 >= p1)


def test_unsafepointer_address_space() raises:
    var p1 = (
        alloc[Int]({count = 1})
        .unsafe_leak()
        .unsafe_address_space_cast[AddressSpace(0)]()
    )
    dealloc(Allocation(unsafe_owned_ptr=p1, layout={count = 1}))

    var p2 = (
        alloc[Int]({count = 1})
        .unsafe_leak()
        .unsafe_address_space_cast[.GENERIC]()
    )
    dealloc(Allocation(unsafe_owned_ptr=p2, layout={count = 1}))


def test_unsafepointer_aligned_alloc() raises:
    comptime alignment_1 = 32
    var ptr_allocation = alloc[UInt8](
        {count = 1, alignment = alignment_1}
    ).into_managed()
    var ptr = ptr_allocation.unsafe_ptr()
    var ptr_uint64 = UInt64(Int(ptr))
    assert_equal(ptr_uint64 % alignment_1, 0)

    comptime alignment_2 = 64
    var ptr_2_allocation = alloc[UInt8](
        {count = 1, alignment = alignment_2}
    ).into_managed()
    var ptr_2 = ptr_2_allocation.unsafe_ptr()
    var ptr_uint64_2 = UInt64(Int(ptr_2))
    assert_equal(ptr_uint64_2 % alignment_2, 0)

    comptime alignment_3 = 128
    var ptr_3_allocation = alloc[UInt8](
        {count = 1, alignment = alignment_3}
    ).into_managed()
    var ptr_3 = ptr_3_allocation.unsafe_ptr()
    var ptr_uint64_3 = UInt64(Int(ptr_3))
    assert_equal(ptr_uint64_3 % alignment_3, 0)


# Test that `alloc` no longer artificially extends the lifetime
# of every local variable in methods where its used.
def test_unsafepointer_alloc_origin() raises:
    # -----------------------------------------
    # Test with MutAnyOrigin alloc() origin
    # -----------------------------------------

    var did_del_1 = False

    # Allocate pointer with MutAnyOrigin.
    var ptr_1 = alloc[Int]({count = 1}).unsafe_leak().as_unsafe_any_origin()

    var obj_1 = ObservableDel(Pointer(to=did_del_1))

    # Object has not been deleted, because MutAnyOrigin is keeping it alive.
    assert_false(did_del_1)

    dealloc(
        Allocation(
            unsafe_owned_ptr=ptr_1.unsafe_origin_cast[MutUntrackedOrigin](),
            layout={count = 1},
        )
    )

    # Now that `ptr` is out of scope, `obj_1` was destroyed as well.
    assert_true(did_del_1)

    # ----------------------------------------
    # Test with default (empty) alloc() origin
    # ----------------------------------------

    var did_del_2 = False

    # Allocate pointer with empty origin.
    var ptr_2 = alloc[Int]({count = 1}).unsafe_leak()

    # Note: Set ObservableDel origin explicitly since it otherwise contains a
    #   MutAnyOrigin pointer that interferes with this test.
    _ = ObservableDel[origin_of(did_del_2)](Pointer(to=did_del_2))

    # `obj_2` is ASAP destroyed, since `ptr_2` origin does not keep it alive.
    assert_true(did_del_2)

    dealloc(Allocation(unsafe_owned_ptr=ptr_2, layout={count = 1}))


# NOTE: Tests fails due to a `Pointer` size
# constraint failing to be satisfied.
#
# def test_unsafepointer_zero_size():
#     alias T = SIMD[DType.int32, 0]
#
#     var start_ptr = Pointer[T].alloc(10)
#     var dest_ptr = start_ptr + 5
#
#     assert_true(start_ptr < dest_ptr)
#     assert_true(start_ptr != dest_ptr)


def test_indexing() raises:
    var allocation = alloc[Int]({count = 4}).into_managed()
    var ptr = allocation.unsafe_ptr()
    for i in range(4):
        ptr[unsafe_offset=i] = i

    assert_equal(ptr[unsafe_offset=Int(1)], 1)
    assert_equal(ptr[unsafe_offset=3], 3)


def test_indexing_simd() raises:
    var allocation = alloc[Int]({count = 4}).into_managed()
    var ptr = allocation.unsafe_ptr()
    for i in range(4):
        ptr[unsafe_offset=UInt8(i)] = i

    assert_equal(ptr[unsafe_offset=UInt8(1)], 1)
    assert_equal(ptr[unsafe_offset=UInt8(3)], 3)
    assert_equal(ptr[unsafe_offset=UInt16(1)], 1)
    assert_equal(ptr[unsafe_offset=UInt16(3)], 3)
    assert_equal(ptr[unsafe_offset=UInt32(1)], 1)
    assert_equal(ptr[unsafe_offset=UInt32(3)], 3)
    assert_equal(ptr[unsafe_offset=UInt64(1)], 1)
    assert_equal(ptr[unsafe_offset=UInt64(3)], 3)
    assert_equal(ptr[unsafe_offset=Int8(1)], 1)
    assert_equal(ptr[unsafe_offset=Int8(3)], 3)
    assert_equal(ptr[unsafe_offset=Int16(1)], 1)
    assert_equal(ptr[unsafe_offset=Int16(3)], 3)
    assert_equal(ptr[unsafe_offset=Int32(1)], 1)
    assert_equal(ptr[unsafe_offset=Int32(3)], 3)
    assert_equal(ptr[unsafe_offset=Int64(1)], 1)
    assert_equal(ptr[unsafe_offset=Int64(3)], 3)


def test_alignment() raises:
    var ptr_allocation = alloc[Int64](
        {count = 8, alignment = 64}
    ).into_managed()
    var ptr = ptr_allocation.unsafe_ptr()
    assert_equal(Int(ptr) % 64, 0)

    var ptr_2_allocation = alloc[UInt8](
        {count = 32, alignment = 32}
    ).into_managed()
    var ptr_2 = ptr_2_allocation.unsafe_ptr()
    assert_equal(Int(ptr_2) % 32, 0)


def test_offset() raises:
    var ptr_allocation = alloc[Int]({count = 5}).into_managed()
    var ptr = ptr_allocation.unsafe_ptr()
    for i in range(5):
        ptr[unsafe_offset=i] = i
    var x = UInt(3)
    var y = Int(4)
    assert_equal(ptr.unsafe_offset(x)[], 3)
    assert_equal(ptr.unsafe_offset(y)[], 4)

    var ptr2 = alloc[Int]({count = 5}).unsafe_leak()
    var ptr3 = ptr2
    ptr2 = ptr2.unsafe_offset(UInt(3))
    assert_equal(ptr2, ptr3.unsafe_offset(3))
    ptr2 = ptr2.unsafe_offset(-Int(UInt(5)))
    assert_equal(ptr2, ptr3.unsafe_offset(-2))
    assert_equal(ptr2.unsafe_offset(UInt(1)), ptr3.unsafe_offset(-1))
    assert_equal(ptr2.unsafe_offset(-Int(UInt(4))), ptr3.unsafe_offset(-6))
    dealloc(Allocation(unsafe_owned_ptr=ptr2, layout={count = 5}))


def test_offset_from() raises:
    var ptr_allocation = alloc[Int32]({count = 8}).into_managed()
    var ptr = ptr_allocation.unsafe_ptr()
    var end = ptr.unsafe_offset(8)

    assert_equal(end.offset_from(ptr), 8)
    assert_equal(ptr.offset_from(end), -8)
    assert_equal(ptr.offset_from(ptr), 0)
    assert_equal(ptr.unsafe_offset(3).offset_from(ptr.unsafe_offset(1)), 2)

    assert_equal(end - ptr, 8)
    assert_equal(ptr - end, -8)

    var wide_allocation = alloc[SIMD[.int64, 4]]({count = 3}).into_managed()
    var wide = wide_allocation.unsafe_ptr()
    assert_equal(wide.unsafe_offset(2) - wide, 2)
    assert_equal(wide - wide.unsafe_offset(2), -2)

    # offset_from() and the `-` operator work on safe pointers too, and the
    # operands may mix pointer safety.
    var data = List[Int32](length=4, fill=0)
    var first: Pointer[Int32, origin_of(data)] = data.unsafe_ptr()
    var last = first.unsafe_offset(3)
    assert_equal(last.offset_from(first), 3)
    assert_equal(first.offset_from(last), -3)
    assert_equal(last - first, 3)
    assert_equal(first - last, -3)
    assert_equal(data.unsafe_ptr() - first, 0)


def test_load_and_store_simd() raises:
    var ptr_allocation = alloc[Int8]({count = 16}).into_managed()
    var ptr = ptr_allocation.unsafe_ptr()
    for i in range(16):
        ptr[unsafe_offset=i] = Int8(i)
    for i in range(0, 16, 4):
        var vec = ptr.unsafe_load[width=4](i)
        assert_equal(
            vec,
            SIMD[.int8, 4](Int8(i), Int8(i + 1), Int8(i + 2), Int8(i + 3)),
        )

    var ptr2_allocation = alloc[Int8]({count = 16}).into_managed()
    var ptr2 = ptr2_allocation.unsafe_ptr()
    for i in range(0, 16, 4):
        ptr2.unsafe_store(i, SIMD[.int8, 4](i))
    for i in range(16):
        assert_equal(ptr2[unsafe_offset=i], Int8(i // 4 * 4))


def test_load_and_store_simd_bool() raises:
    # Regression test: storing SIMD[DType.bool, N] with width > 1 then
    # loading element-wise should give correct results (github.com/modular/modular/issues/5875).
    var allocation = alloc[Scalar[.bool]]({count = 4}).into_managed()
    var p = allocation.unsafe_ptr()
    p.unsafe_store(0, SIMD[.bool, 2](True, False))
    assert_true(p[unsafe_offset=0])
    assert_false(p[unsafe_offset=1])
    for i in range(2):
        assert_equal(p.unsafe_load[width=2](0)[i], p[unsafe_offset=i])

    p.unsafe_store(0, SIMD[.bool, 4](False, True, True, False))
    assert_false(p[unsafe_offset=0])
    assert_true(p[unsafe_offset=1])
    assert_true(p[unsafe_offset=2])
    assert_false(p[unsafe_offset=3])
    for i in range(4):
        assert_equal(p.unsafe_load[width=4](0)[i], p[unsafe_offset=i])


def test_unsafe_methods_on_safe_pointer() raises:
    var data = List[Int32](length=8, fill=0)
    for i in range(len(data)):
        data[i] = Int32(i)

    var ptr: Pointer[Int32, origin_of(data)] = data.unsafe_ptr()

    assert_equal(ptr.unsafe_offset(2)[], Int32(2))
    assert_equal(ptr[unsafe_offset=3], Int32(3))

    ptr.unsafe_store(4, SIMD[.int32, 4](10, 11, 12, 13))
    assert_equal(ptr.unsafe_load[width=4](4), SIMD[.int32, 4](10, 11, 12, 13))

    assert_equal(Int(ptr.unsafe_address_space_cast[.GENERIC]()), Int(ptr))
    assert_equal(Int(ptr.unsafe_as_noalias()), Int(ptr))


def test_volatile_load_and_store_simd() raises:
    var ptr_allocation = alloc[Int8]({count = 16}).into_managed()
    var ptr = ptr_allocation.unsafe_ptr()
    for i in range(16):
        ptr[unsafe_offset=i] = Int8(i)
    for i in range(0, 16, 4):
        var vec = ptr.unsafe_load[width=4, volatile=True](i)
        assert_equal(
            vec,
            SIMD[.int8, 4](Int8(i), Int8(i + 1), Int8(i + 2), Int8(i + 3)),
        )

    var ptr2_allocation = alloc[Int8]({count = 16}).into_managed()
    var ptr2 = ptr2_allocation.unsafe_ptr()
    for i in range(0, 16, 4):
        ptr2.unsafe_store[volatile=True](i, SIMD[.int8, 4](i))
    for i in range(16):
        assert_equal(ptr2[unsafe_offset=i], Int8(i // 4 * 4))


# Test pointer merging with ternary operation.
def test_merge() raises:
    var a: List = [1, 2, 3]
    var b: List = [4, 5, 6]

    def inner(cond: Bool, x: Int, mut a: List[Int], mut b: List[Int]):
        var either = Pointer(to=a) if cond else Pointer(to=b)
        either[].append(x)

    inner(True, 7, a, b)
    inner(False, 8, a, b)

    assert_equal(a, [1, 2, 3, 7])
    assert_equal(b, [4, 5, 6, 8])


def test_swap_pointees_trivial_move() raises:
    var a = 42
    Pointer(to=a).as_unsafe_any_origin().swap_pointees(
        Pointer(to=a).as_unsafe_any_origin()
    )
    assert_equal(a, 42)

    var x = 1
    var y = 2
    Pointer(to=x).swap_pointees(Pointer(to=y))
    assert_equal(x, 2)
    assert_equal(y, 1)


def test_swap_pointees_non_trivial_move() raises:
    var counter = MoveCounter[Int](42)
    Pointer(to=counter).as_unsafe_any_origin().swap_pointees(
        Pointer(to=counter).as_unsafe_any_origin()
    )
    # Pointers point to the same object, so no move should be performed
    assert_equal(counter.value, 42)
    assert_equal(counter.move_count, 0)

    var counterA = MoveCounter[Int](1)
    var counterB = MoveCounter[Int](2)
    Pointer(to=counterA).swap_pointees(Pointer(to=counterB))

    assert_equal(counterA.value, 2)
    assert_equal(counterA.move_count, 1)

    assert_equal(counterB.value, 1)
    assert_equal(counterB.move_count, 2)


def test_as_unsafe_any_origin_mutable() raises:
    var deleted = False
    var observer = ObservableDel[origin_of(deleted)](Pointer(to=deleted))
    var x = 42

    var mutable = Pointer(to=x).as_unsafe_any_origin()
    assert_true(mutable.mut)
    assert_false(deleted)

    mutable[] = 55
    assert_true(deleted)  # AnyOrigin extends all lifetimes


def test_as_unsafe_any_origin_immutable() raises:
    var deleted = False
    var observer = ObservableDel[origin_of(deleted)](Pointer(to=deleted))
    var x = 42

    var immutable = Pointer(to=x).as_unsafe_any_origin().as_imm()
    assert_false(immutable.mut)
    assert_false(deleted)

    var _x = immutable[]
    assert_true(deleted)  # AnyOrigin extends all lifetimes


def test_as_imm() raises:
    var x = 42
    var mutable = Pointer(to=x)
    assert_true(mutable.mut)
    assert_false(mutable.as_imm().mut)


def test_unsafe_mut_cast() raises:
    var x = 42
    var ptr = Pointer(to=x)
    var immutable = ptr.unsafe_mut_cast[False]()
    assert_false(immutable.mut)
    var _mutable = immutable.unsafe_mut_cast[True]()
    assert_true(_mutable.mut)


def _ref_to[origin: ImmOrigin](ref[origin] to: String):
    pass


def test_unsafe_origin_cast() raises:
    var x = "hello"
    var y = "world"

    var ptr = Pointer(to=x)
    _ref_to[origin_of(x)](ptr[])
    _ref_to[origin_of(y)](ptr.unsafe_origin_cast[origin_of(y)]()[])


def _ptr_to_int(ptr: Pointer[Int, MutUntrackedOrigin]) -> Int:
    return Int(ptr)


def test_ptr_to_int_llvm_lowering() raises:
    var info = compile_info[_ptr_to_int, emission_kind="llvm-opt"]()
    # https://llvm.org/docs/LangRef.html#ptrtoint-to-instruction
    # We need to check `ptrtoint` is used instead of `ptrtoaddr` to ensure
    # pointer provenance is preserved for the default ptr -> int conversion.
    assert_true("ptrtoint" in info.asm)
    assert_false("ptrtoaddr" in info.asm)


def _from_address(x: Int, out result: Pointer[Int, MutUntrackedOrigin]):
    result = type_of(result)(unsafe_from_address=x)


def test_unsafe_from_address_llvm_lowering() raises:
    var info = compile_info[_from_address, emission_kind="llvm-opt"]()
    assert_true("inttoptr" in info.asm)


def test_unsafe_from_address() raises:
    var x = 42
    var ptr = Pointer(to=x)
    var ptr2 = type_of(ptr)(unsafe_from_address=Int(ptr))
    assert_equal(ptr2[], 42)


def test_unsafe_from_address_pointer_width() raises:
    # `unsafe_from_address`'s bound is address-space specific: the overflow
    # `debug_assert` is only compiled in when the pointer is narrower than
    # `Int`. On an AMDGPU target, GENERIC pointers are 64-bit (as wide as
    # `Int`, so the check is elided) while SHARED pointers are 32-bit.
    comptime AMD_TARGET = get_gpu_target["mi355x"]()

    comptime GenericPtr = Pointer[Int, MutUntrackedOrigin]
    comptime SharedPtr = Pointer[Int, MutUntrackedOrigin, address_space=.SHARED]

    assert_equal(
        bit_width_of[GenericPtr, target=AMD_TARGET](),
        bit_width_of[Int, target=AMD_TARGET](),
    )
    assert_equal(bit_width_of[SharedPtr, target=AMD_TARGET](), 32)


def test_write_to() raises:
    var x = 42
    check_write_to(Pointer(to=x), contains="0x", is_repr=False)

    var s = String("hello")
    check_write_to(Pointer(to=s), contains="0x", is_repr=False)


def test_write_repr_to() raises:
    var x = 42
    check_write_to(
        Pointer(to=x),
        contains=(
            "Pointer[mut=True, SIMD[DType.int, 1],"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )
    check_write_to(
        Pointer(to=x).as_imm(),
        contains=(
            "Pointer[mut=False, SIMD[DType.int, 1],"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )
    check_write_to(
        Pointer(to=x).unsafe_address_space_cast[.SHARED](),
        contains=(
            "Pointer[mut=True, SIMD[DType.int, 1],"
            " address_space=AddressSpace.SHARED](0x"
        ),
        is_repr=True,
    )

    var s = String("hello")
    check_write_to(
        Pointer(to=s),
        contains=(
            "Pointer[mut=True, String, address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )


def test_unsafe_pointer_niche() raises:
    var x = 42
    comptime UP = Pointer[Int, ImmOrigin(origin_of(x))]
    assert_equal(size_of[UP](), size_of[Optional[UP]]())

    var storage = MaybeUninit[UP]()
    UP.write_niche(Pointer(to=storage))
    assert_true(UP.isa_niche(Pointer(to=storage)))

    storage.write(UP(to=x))
    assert_false(UP.isa_niche(Pointer(to=storage)))


def test_unsafe_pointer_dangling() raises:
    var int_ptr = Pointer[Int, MutUntrackedOrigin].unsafe_dangling()
    assert_equal(Int(int_ptr) % align_of[Int](), 0)

    var str_ptr = Pointer[String, MutUntrackedOrigin].unsafe_dangling()
    assert_equal(Int(str_ptr) % align_of[String](), 0)


def test_optional_unsafe_pointer_across_c_ffi() raises:
    var string = "abc"
    comptime Result = Optional[Pointer[Int8, origin_of(string)]]

    var not_found = external_call[
        "strchr",
        Result,
    ](string.as_c_string_slice(), Int8(ord("z")))
    assert_false(not_found)

    var found = external_call[
        "strchr",
        Result,
    ](string.as_c_string_slice(), Int8(ord("a")))
    assert_true(found)
    assert_equal(Int(found[]), Int(string.as_bytes().unsafe_ptr()))


def _test_lower(pointer: Optional[Pointer[Int32, MutAnyOrigin]]) -> Bool:
    return Bool(pointer)


def test_optional_unsafe_pointer_llvm_lowering() raises:
    var info = String(compile_info[_test_lower, emission_kind="llvm-opt"]())

    for line in info.splitlines():
        if "define" in line and "::_test_lower" in line:
            assert_true("ptr" in line, info)
            assert_false("[1 x ptr]" in line)
            return

    raise Error("did not find _test_lower function")


def test_pointer_to_int() raises:
    var x = 42
    comptime P = Pointer[Int, ImmOrigin(origin_of(x))]

    var present = Optional[P](P(to=x))
    assert_equal(pointer_to_int(present), Int(P(to=x)))

    var absent = Optional[P]()
    assert_equal(pointer_to_int(absent), 0)


def test_alloc_free_single_zst() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = std.memory.alloc.Layout[ZST](count=1)
    var ptr = alloc(layout).unsafe_leak()

    assert_equal(0, len(ptr[]))  # dereference the pointer

    std.memory.alloc.dealloc(
        std.memory.alloc.ThinAllocation(
            unsafe_owned_ptr=ptr
        ).unsafe_with_layout(layout)
    )


def test_alloc_free_many_zst() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = std.memory.alloc.Layout[ZST](count=Int.MAX)
    var ptr = alloc(layout).unsafe_leak()

    assert_equal(0, len(ptr[]))  # dereference the pointer
    assert_equal(0, len(ptr.unsafe_offset(Int.MAX)[]))

    std.memory.alloc.dealloc(
        std.memory.alloc.ThinAllocation(
            unsafe_owned_ptr=ptr
        ).unsafe_with_layout(layout)
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
