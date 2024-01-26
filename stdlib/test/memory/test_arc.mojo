# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo --debug-level full %s

from memory._arc import Arc
from collections.vector import DynamicVector
from testing import *


def test_basic():
    let p = Arc(4)
    let p2 = p
    p2.set(3)
    assert_equal(3, p.get())


@value
struct ObservableDel(CollectionElement):
    var target: Pointer[Bool]

    fn __del__(owned self):
        self.target.store(True)


def test_deleter_not_called_until_no_references():
    var deleted = False
    let p = Arc(ObservableDel(Pointer.address_of(deleted)))
    let p2 = p
    _ = p ^
    assert_false(deleted)

    var vec = DynamicVector[Arc[ObservableDel]]()
    vec.push_back(p2)
    _ = p2 ^
    assert_false(deleted)
    _ = vec ^
    assert_true(deleted)


def main():
    test_basic()
    test_deleter_not_called_until_no_references()
