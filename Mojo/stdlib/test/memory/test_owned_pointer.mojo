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

from std.memory import OwnedPointer
from test_utils import (
    ExplicitCopyOnly,
    ExplicitDelOnly,
    ImplicitCopyOnly,
    MoveOnly,
    ObservableDel,
    check_write_to,
)
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
    TestSuite,
)


def test_basic_ref() raises:
    var b = OwnedPointer(1)
    assert_equal(1, b[])


def test_from_unsafe_pointer_constructor() raises:
    var deleted = False
    var unsafe_ptr = alloc[ObservableDel[]]({count = 1}).unsafe_leak()
    unsafe_ptr.unsafe_write(
        ObservableDel(Pointer(to=deleted).as_unsafe_any_origin())
    )

    var ptr = OwnedPointer(unsafe_from_raw_pointer=unsafe_ptr)
    _ = ptr

    assert_true(deleted)


def test_owned_pointer_copy_constructor() raises:
    var b = OwnedPointer(1)
    var b2 = OwnedPointer(other=b)

    assert_equal(1, b[])
    assert_equal(1, b2[])

    assert_false(b.ptr() == b2.ptr())


def test_copying_constructor() raises:
    var v = ImplicitCopyOnly(1)
    var b = OwnedPointer(v)

    assert_equal(b[].value, 1)
    assert_equal(b[].copy_count, 1)  # this should only ever require one copy


def test_explicitly_copying_constructor() raises:
    var v = ExplicitCopyOnly(1)
    var b = OwnedPointer(copy_value=v)

    assert_equal(b[].value, 1)
    assert_equal(b[].copy_count, 1)  # this should only ever require one copy


def test_moving_constructor() raises:
    var v = MoveOnly[Int](1)
    var b = OwnedPointer(v^)

    assert_equal(b[].data, 1)


def test_basic_ref_mutate() raises:
    var b = OwnedPointer(1)
    assert_equal(1, b[])

    b[] = 2

    assert_equal(2, b[])


def test_multiple_refs() raises:
    var b = OwnedPointer(1)

    var borrow1 = b[]
    var borrow2 = b[]

    assert_equal(2, borrow1 + borrow2)


def test_basic_del() raises:
    var deleted = False
    var b = OwnedPointer(ObservableDel(Pointer(to=deleted)))

    assert_false(deleted)

    _ = b^

    assert_true(deleted)


def test_into_inner() raises:
    var b = OwnedPointer(1)
    var v = b^.into_inner()
    assert_equal(1, v)


def test_moveinit() raises:
    var deleted = False
    var b = OwnedPointer(ObservableDel(Pointer(to=deleted)))
    var p1 = Int(b.ptr())

    var b2 = b^
    var p2 = Int(b2.ptr())

    assert_false(deleted)

    # move should reuse the allocation, having the same address
    assert_equal(p1, p2)

    _ = b2^


def test_unsafe_take_allocation() raises:
    var deleted = False

    var owned_ptr = OwnedPointer(ObservableDel(Pointer(to=deleted)))

    var ptr = owned_ptr^.unsafe_take_allocation().unsafe_leak()

    # Check that `Box` did not deinitialize its pointee.
    assert_false(deleted)

    _ = OwnedPointer(unsafe_from_raw_pointer=ptr)


def test_owned_pointer_linear_type() raises:
    # An `OwnedPointer` holding a linear (non-`Deinitable`) element
    # has no implicit destructor, so it is consumed explicitly with
    # `into_inner()`.
    # The linear value is destroyed before any raising assert runs (the
    # linear-in-`raises` idiom).
    var b = OwnedPointer(ExplicitDelOnly(5))
    var v = b^.into_inner()
    var data = v.data
    v^.destroy()
    assert_equal(data, 5)

    # `unsafe_take_allocation()` releases the storage without touching the
    # pointee, so it also works for a linear element type.
    var b2 = OwnedPointer(ExplicitDelOnly(7))
    var b3 = OwnedPointer(
        unsafe_from_raw_pointer=b2^.unsafe_take_allocation().unsafe_leak()
    )
    var v3 = b3^.into_inner()
    var data3 = v3.data
    v3^.destroy()
    assert_equal(data3, 7)


def test_write_to() raises:
    check_write_to(OwnedPointer(42), expected="42", is_repr=False)
    check_write_to(OwnedPointer("hello"), expected="hello", is_repr=False)


def test_write_repr_to() raises:
    check_write_to(
        OwnedPointer(42),
        expected="OwnedPointer[SIMD[DType.int, 1]](Int(42))",
        is_repr=True,
    )
    check_write_to(
        OwnedPointer("hello"),
        expected="OwnedPointer[String]('hello')",
        is_repr=True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
