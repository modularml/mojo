# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from memory.unsafe import Pointer
from collections.vector import CollectionElement
from utils.variant import Variant
from testing import *

from sys.ffi import _get_global


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
        self.copied = other.copied ^
        self.moved = other.moved + 1


fn _poison_ptr() -> Pointer[Bool]:
    let ptr = _get_global[
        "TEST_VARIANT_POISON", _initialize_poison, _destroy_poison
    ]()
    return ptr.bitcast[Bool]()


fn assert_no_poison() raises:
    assert_false(_poison_ptr().load())


fn _initialize_poison(payload: Pointer[NoneType]) -> Pointer[NoneType]:
    let poison = Pointer[Bool].alloc(1)
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
    let s = IntOrString(String("4"))

    # isa
    assert_true(i.isa[Int]())
    assert_false(i.isa[String]())
    assert_true(s.isa[String]())
    assert_false(s.isa[Int]())

    # get
    assert_equal(4, i.get[Int]())
    assert_equal("4", s.get[String]())
    # we don't test what happens when you `get` the wrong type.
    # have fun!

    # set
    i.set[String]("i")
    assert_false(i.isa[Int]())
    assert_true(i.isa[String]())
    assert_equal("i", i.get[String]())


def test_copy():
    var v1 = TestVariant(TestCounter())
    var v2 = v1
    assert_true(
        v2.get[TestCounter]().copied > v1.get[TestCounter]().copied,
        "didn't call copyinit",
    )
    # test that we didn't call the other copyinit too!
    assert_no_poison()


def test_move():
    let v1 = TestVariant(TestCounter())
    let v2 = v1
    assert_true(
        v2.get[TestCounter]().moved > v1.get[TestCounter]().moved,
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
    let v1 = TestDeleterVariant(ObservableDel(Pointer.address_of(deleted)))
    _ = v1 ^  # call __del__
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
    _ = v1 ^
    assert_true(deleted2)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_take_doesnt_call_deleter():
    alias TestDeleterVariant = Variant[ObservableDel, Poison]
    var deleted: Bool = False
    let v1 = TestDeleterVariant(ObservableDel(Pointer.address_of(deleted)))
    assert_false(deleted)
    let v2 = v1.take[ObservableDel]()
    assert_false(deleted)
    _ = v2
    assert_true(deleted)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def main():
    test_basic()
    test_copy()
    test_move()
    test_del()
    test_set_calls_deleter()
    test_take_doesnt_call_deleter()
