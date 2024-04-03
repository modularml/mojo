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
# RUN: %mojo -debug-level full %s

from sys.ffi import _get_global

from memory.unsafe import Pointer
from testing import assert_equal, assert_false, assert_true

from utils.variant import Variant


struct TestCounter(CollectionElement):
    var copied: Int
    var moved: Int

    fn __init__(inout self):
        self.copied = 0
        self.moved = 0

    fn __copyinit__(inout self, other: Self):
        self.copied = other.copied + 1
        self.moved = other.moved

    fn __moveinit__(inout self, owned other: Self):
        self.copied = other.copied^
        self.moved = other.moved + 1


fn _poison_ptr() -> Pointer[Bool]:
    var ptr = _get_global[
        "TEST_VARIANT_POISON", _initialize_poison, _destroy_poison
    ]()
    return ptr.bitcast[Bool]()


fn assert_no_poison() raises:
    assert_false(_poison_ptr().load())


fn _initialize_poison(payload: Pointer[NoneType]) -> Pointer[NoneType]:
    var poison = Pointer[Bool].alloc(1)
    poison.store(False)
    return poison.bitcast[NoneType]()


fn _destroy_poison(p: Pointer[NoneType]):
    p.free()


struct Poison(CollectionElement):
    fn __init__(inout self):
        pass

    fn __copyinit__(inout self, other: Self):
        _poison_ptr().store(True)

    fn __moveinit__(inout self, owned other: Self):
        _poison_ptr().store(True)

    fn __del__(owned self):
        _poison_ptr().store(True)


alias TestVariant = Variant[TestCounter, Poison]


def test_basic():
    alias IntOrString = Variant[Int, String]
    var i = IntOrString(4)
    var s = IntOrString(String("4"))

    # isa
    assert_true(i.isa[Int]())
    assert_false(i.isa[String]())
    assert_true(s.isa[String]())
    assert_false(s.isa[Int]())

    # get
    assert_equal(4, i.get[Int]()[])
    assert_equal("4", s.get[String]()[])
    # we don't test what happens when you `get` the wrong type.
    # have fun!

    # set
    i.set[String]("i")
    assert_false(i.isa[Int]())
    assert_true(i.isa[String]())
    assert_equal("i", i.get[String]()[])


def test_copy():
    var v1 = TestVariant(TestCounter())
    var v2 = v1
    assert_true(
        v2.get[TestCounter]()[].copied > v1.get[TestCounter]()[].copied,
        "didn't call copyinit",
    )
    # test that we didn't call the other copyinit too!
    assert_no_poison()


def test_move():
    var v1 = TestVariant(TestCounter())
    var v2 = v1
    assert_true(
        v2.get[TestCounter]()[].moved > v1.get[TestCounter]()[].moved,
        "didn't call moveinit",
    )
    # test that we didn't call the other moveinit too!
    assert_no_poison()


@value
struct ObservableDel(CollectionElement):
    var target: Pointer[Bool]

    fn __del__(owned self):
        self.target.store(True)


def test_del():
    alias TestDeleterVariant = Variant[ObservableDel, Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(Pointer.address_of(deleted)))
    _ = v1^  # call __del__
    assert_true(deleted)
    # test that we didn't call the other deleter too!
    assert_no_poison()


def test_set_calls_deleter():
    alias TestDeleterVariant = Variant[ObservableDel, Poison]
    var deleted: Bool = False
    var deleted2: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(Pointer.address_of(deleted)))
    v1.set[ObservableDel](ObservableDel(Pointer.address_of(deleted2)))
    assert_true(deleted)
    assert_false(deleted2)
    _ = v1^
    assert_true(deleted2)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_take_doesnt_call_deleter():
    alias TestDeleterVariant = Variant[ObservableDel, Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(Pointer.address_of(deleted)))
    assert_false(deleted)
    var v2 = v1.take[ObservableDel]()
    assert_false(deleted)
    _ = v2
    assert_true(deleted)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_get_returns_mutable_reference():
    var v1: Variant[Int, String] = 42
    var x = v1.get[Int]()[]
    assert_equal(42, x)
    x = 100
    assert_equal(100, x)
    v1.set[String]("hello")
    assert_equal(100, x)  # the x reference is still valid

    var v2: Variant[Int, String] = String("something")
    v2.get[String]()[] = "something else"
    assert_equal(v2.get[String]()[], "something else")

    var v3: Variant[Int, String] = 4
    assert_equal(
        int(v3._get_ptr[Int]()),
        int(Reference(v3)[]._get_ptr[Int]()),
    )


def main():
    test_basic()
    test_get_returns_mutable_reference()
    test_copy()
    test_move()
    test_del()
    test_set_calls_deleter()
    test_take_doesnt_call_deleter()
