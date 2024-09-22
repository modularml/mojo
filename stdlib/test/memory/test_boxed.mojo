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
# RUN: %mojo --debug-level full %s

from testing import assert_equal, assert_false, assert_true
from memory import Boxed, UnsafePointer


@value
struct ObservableDel(CollectionElement):
    var target: UnsafePointer[Bool]

    fn __init__(inout self, *, other: Self):
        self = other

    fn __del__(owned self):
        self.target.init_pointee_move(True)


struct OnlyCopyable:
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __init__(inout self, *, other: Self):
        self.value = other.value

    fn __copyinit__(inout self, existing: Self):
        self.value = existing.value


struct OnlyMovable:
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __moveinit__(inout self, owned existing: Self):
        self.value = existing.value


def test_basic_ref():
    var b = Boxed(1)
    assert_equal(1, b[])


def test_box_copy_constructor():
    var b = Boxed(1)
    var b2 = Boxed(copy_box=b)

    assert_equal(1, b[])
    assert_equal(1, b2[])


def test_copying_constructor():
    var v = OnlyCopyable(1)
    var b = Boxed(copy_value=v)


def test_moving_constructor():
    var v = OnlyMovable(1)
    var b = Boxed(v^)


def test_basic_ref_mutate():
    var b = Boxed(1)
    assert_equal(1, b[])

    b[] = 2

    assert_equal(2, b[])


def test_multiple_refs():
    var b = Boxed(1)

    var borrow1 = b[]
    var borrow2 = b[]

    assert_equal(2, borrow1 + borrow2)


def test_basic_del():
    var deleted = False
    var b = Boxed(ObservableDel(UnsafePointer.address_of(deleted)))

    assert_false(deleted)

    _ = b^

    assert_true(deleted)


def test_take():
    var b = Boxed(1)
    var v = b^.take()
    assert_equal(1, v)


def test_moveinit():
    var deleted = False
    var b = Boxed(ObservableDel(UnsafePointer.address_of(deleted)))
    var b2 = b^

    assert_false(deleted)

    _ = b2^


def main():
    test_basic_ref()
    test_box_copy_constructor()
    test_moving_constructor()
    test_copying_constructor()
    test_basic_ref_mutate()
    test_basic_del()
    test_take()
    test_moveinit()
