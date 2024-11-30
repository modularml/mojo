# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from memory import OwnedPointer, UnsafePointer
from test_utils import (
    ExplicitCopyOnly,
    ImplicitCopyOnly,
    MoveOnly,
    ObservableDel,
)
from testing import assert_equal, assert_false, assert_not_equal, assert_true


def test_basic_ref():
    var b = OwnedPointer(1)
    assert_equal(1, b[])


def test_owned_pointer_copy_constructor():
    var b = OwnedPointer(1)
    var b2 = OwnedPointer(other=b)

    assert_equal(1, b[])
    assert_equal(1, b2[])

    assert_not_equal(b.unsafe_ptr(), b2.unsafe_ptr())


def test_copying_constructor():
    var v = ImplicitCopyOnly(1)
    var b = OwnedPointer(v)

    assert_equal(b[].value, 1)
    assert_equal(b[].copy_count, 1)  # this should only ever require one copy


def test_explicitly_copying_constructor():
    var v = ExplicitCopyOnly(1)
    var b = OwnedPointer(copy_value=v)

    assert_equal(b[].value, 1)
    assert_equal(b[].copy_count, 1)  # this should only ever require one copy


def test_moving_constructor():
    var v = MoveOnly[Int](1)
    var b = OwnedPointer(v^)

    assert_equal(b[].data, 1)


def test_basic_ref_mutate():
    var b = OwnedPointer(1)
    assert_equal(1, b[])

    b[] = 2

    assert_equal(2, b[])


def test_multiple_refs():
    var b = OwnedPointer(1)

    var borrow1 = b[]
    var borrow2 = b[]

    assert_equal(2, borrow1 + borrow2)


def test_basic_del():
    var deleted = False
    var b = OwnedPointer(ObservableDel(UnsafePointer.address_of(deleted)))

    assert_false(deleted)

    _ = b^

    assert_true(deleted)


def test_take():
    var b = OwnedPointer(1)
    var v = b^.take()
    assert_equal(1, v)


def test_moveinit():
    var deleted = False
    var b = OwnedPointer(ObservableDel(UnsafePointer.address_of(deleted)))
    var p1 = b.unsafe_ptr()

    var b2 = b^
    var p2 = b2.unsafe_ptr()

    assert_false(deleted)
    assert_equal(p1, p2)  # move should reuse the allocation

    _ = b2^


def test_steal_data():
    var deleted = False

    var owned_ptr = OwnedPointer(
        ObservableDel(UnsafePointer.address_of(deleted))
    )

    var ptr = owned_ptr^.steal_data()

    # Check that `Box` did not deinitialize its pointee.
    assert_false(deleted)

    ptr.destroy_pointee()
    ptr.free()


def main():
    test_basic_ref()
    test_owned_pointer_copy_constructor()
    test_moving_constructor()
    test_copying_constructor()
    test_explicitly_copying_constructor()
    test_basic_ref_mutate()
    test_basic_del()
    test_take()
    test_moveinit()
    test_steal_data()
