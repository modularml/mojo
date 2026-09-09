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

from std.memory import unsafe_destroy_n
from std.memory.alloc import (
    alloc,
    dealloc,
    Allocation,
    ManagedAllocation,
    ThinAllocation,
    Layout,
)
from std.sys import align_of, size_of

from test_utils import ObservableDel, check_write_to
from std.testing import (
    assert_equal,
    assert_not_equal,
    assert_false,
    assert_true,
    TestSuite,
)


def test_layout_with_default_alignment_uses_type_alignment() raises:
    var layout = Layout[Int32](count=4)
    assert_equal(layout.count(), 4)
    assert_equal(layout.alignment(), align_of[Int32]())


def test_layout_with_runtime_alignment() raises:
    var layout = Layout[Int32](count=4, alignment=64)
    assert_equal(layout.count(), 4)
    assert_equal(layout.alignment(), 64)


def test_layout_aligned_uses_comptime_alignment() raises:
    var layout = Layout[Int32].aligned[128](count=4)
    assert_equal(layout.count(), 4)
    assert_equal(layout.alignment(), 128)


def test_layout_single_has_count_one_and_default_alignment() raises:
    var layout = Layout[Int64].single()
    assert_equal(layout.count(), 1)
    assert_equal(layout.alignment(), align_of[Int64]())


def test_layout_as_byte_layout_scales_count_and_preserves_alignment() raises:
    var layout = Layout[Int32](count=4, alignment=64)
    var byte_layout = layout.as_byte_layout()
    assert_equal(byte_layout.count(), 4 * size_of[Int32]())
    assert_equal(byte_layout.alignment(), 64)


def test_layout_write_to_and_repr() raises:
    var layout = Layout[Int](count=8, alignment=64)
    check_write_to(
        layout,
        expected="Layout[SIMD[DType.int, 1]](count=8, alignment=64)",
        is_repr=False,
    )
    check_write_to(
        layout,
        expected="Layout[SIMD[DType.int, 1]](count=8, alignment=64)",
        is_repr=True,
    )


def test_alloc_and_free_round_trip_reads_and_writes_values() raises:
    var layout = Layout[Int](count=5)
    var a = alloc(layout)
    var ptr = a.unsafe_ptr()
    for i in range(5):
        ptr.unsafe_offset(i).write(i)
    # `assert_equal` can raise, and an `Allocation` must be consumed on
    # every path (including the raising one). So accumulate into a plain `Bool`,
    # `dealloc` the handle, and only then assert.
    var all_match = True
    for i in range(5):
        all_match = all_match and (ptr[unsafe_offset=i] == i)
    dealloc(a^)
    assert_true(all_match)


def test_alloc_returns_pointer_meeting_layout_alignment() raises:
    var layout = Layout[UInt8](count=1, alignment=128)
    var a = alloc(layout)
    var addr = Int(a.unsafe_ptr())
    dealloc(a^)
    assert_equal(addr % 128, 0)


def test_alloc_with_layout_single_supports_one_element() raises:
    var layout = Layout[Int64].single()
    var a = alloc(layout)
    var ptr = a.unsafe_ptr()
    ptr.write(42)
    var value = ptr[]
    dealloc(a^)
    assert_equal(value, 42)


def test_allocation_unsafe_span_covers_layout_count() raises:
    var a = alloc(Layout[Int](count=3))
    var ptr = a.unsafe_ptr()
    for i in range(3):
        ptr.unsafe_offset(i).write(i * 10)
    var span = a.unsafe_span()
    var span_len = len(span)
    var first = span[0]
    var last = span[2]
    dealloc(a^)
    assert_equal(span_len, 3)
    assert_equal(first, 0)
    assert_equal(last, 20)


def test_allocation_count_matches_layout() raises:
    var a = alloc(Layout[Int32](count=7))
    var element_count = a.layout().count()
    dealloc(a^)
    assert_equal(element_count, 7)


def test_allocation_unsized_then_unsafe_with_layout_round_trip() raises:
    var layout = Layout[Int](count=4)
    var a = alloc(layout)
    a.unsafe_ptr().write(99)
    var thin = a^.into_thin()
    var value = thin.unsafe_ptr()[]
    dealloc(thin^.unsafe_with_layout(layout))
    assert_equal(value, 99)


def test_allocation_unsafe_leak_then_reconstruct() raises:
    var layout = Layout[Int](count=2)
    var ptr = alloc(layout).unsafe_leak()
    ptr.write(5)
    ptr.unsafe_offset(1).write(6)
    var total = ptr[unsafe_offset=0] + ptr[unsafe_offset=1]
    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))
    assert_equal(total, 11)


def test_thin_allocation_unsafe_with_layout_and_unsafe_ptr() raises:
    var layout = Layout[Int](count=1)
    var thin = ThinAllocation(unsafe_owned_ptr=alloc(layout).unsafe_leak())
    thin.unsafe_ptr().write(7)
    var value = thin.unsafe_ptr()[]
    dealloc(thin^.unsafe_with_layout(layout))
    assert_equal(value, 7)


def test_deletable_allocation_into_allocation_round_trip() raises:
    var layout = Layout[Int](count=3)
    var allocation = alloc(layout)
    var alloc_addr = Int(allocation.unsafe_ptr())

    # Allocation -> ManagedAllocation -> Allocation (address stays stable).
    var deletable = allocation^.into_managed()
    var deletable_addr = Int(deletable.unsafe_ptr())

    var recovered = Allocation(deletable^)
    var recovered_addr = Int(recovered.unsafe_ptr())

    # `into_allocation` hands back an non-implicitly-deletable handle: dealloc
    # it before the (raising) asserts so it can't be abandoned on a throw.
    dealloc(recovered^)

    assert_equal(deletable_addr, alloc_addr)
    assert_equal(recovered_addr, alloc_addr)


def test_deletable_allocation_layout_matches() raises:
    var deletable = ManagedAllocation(alloc(Layout[Int32](count=7)))
    assert_equal(deletable.layout().count(), 7)
    assert_equal(deletable.layout().alignment(), align_of[Int32]())


def test_deletable_allocation_auto_deallocs_at_last_use() raises:
    var total = 0
    for i in range(3):
        var deletable = alloc(Layout[Int](count=1)).into_managed()
        deletable.unsafe_ptr().write(i)
        total += deletable.unsafe_ptr()[]
    assert_equal(total, 0 + 1 + 2)


def test_dealloc_does_not_run_pointee_destructors() raises:
    var deleted = False
    var obs = ObservableDel(Pointer(to=deleted).as_unsafe_any_origin())
    var a = alloc(Layout[type_of(obs)](count=1))
    a.unsafe_ptr().unsafe_write(obs^)
    dealloc(a^)
    assert_false(deleted)


def test_destroy_n_runs_pointee_destructors_before_dealloc() raises:
    var deleted = False
    var obs = ObservableDel(Pointer(to=deleted).as_unsafe_any_origin())
    var a = alloc(Layout[type_of(obs)](count=1))
    a.unsafe_ptr().unsafe_write(obs^)
    unsafe_destroy_n(a.unsafe_ptr(), 1)
    var ran = deleted
    dealloc(a^)
    assert_true(ran)


def test_alloc_count_0() raises:
    comptime ZST = Array[Int, 0]
    var zst_alloc = alloc(Layout[ZST](count=0)).into_managed()
    assert_equal(zst_alloc.layout().count(), 0)
    dealloc(zst_alloc^)

    var int_alloc = alloc(Layout[Int](count=0)).into_managed()
    assert_equal(int_alloc.layout().count(), 0)
    dealloc(int_alloc^)


def test_alloc_free_single_zst() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = Layout[ZST](count=1)
    var ptr = alloc(layout).unsafe_leak()

    assert_equal(
        ptr.unsafe_bitcast[Int]().as_unsafe_any_origin(),
        ptr[].unsafe_ptr().as_unsafe_any_origin(),
    )

    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))


def test_single_zst_lifecycle() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = Layout[ZST](count=1)
    var alloc = alloc(layout)
    var ptr = alloc^.unsafe_leak()

    ptr.unsafe_write(ZST(fill=0))
    assert_equal(0, len(ptr[]))
    ptr.unsafe_deinit_pointee()
    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))


def test_alloc_free_many_zst() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = Layout[ZST](
        count=Int.MAX
    )  # It's a ZST, it doesn't take memory
    var ptr = alloc(layout).unsafe_leak()

    assert_equal(ptr.unsafe_bitcast[Int](), ptr[].unsafe_ptr())

    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))


def test_many_zst_lifecycle() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = Layout[ZST](count=Int.MAX)
    var ptr = alloc(layout).unsafe_leak()

    ptr.unsafe_bitcast[Array[ZST, Int.MAX]]().unsafe_write({fill = ZST(fill=0)})

    assert_equal(0, len(ptr[]))
    assert_equal(0, len(ptr[unsafe_offset=Int.MAX]))

    ptr.unsafe_bitcast[Array[ZST, Int.MAX]]().unsafe_deinit_pointee()

    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
