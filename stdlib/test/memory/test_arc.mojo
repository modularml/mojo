# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo --debug-level full %s

from collections import List

from memory._arc import Arc
from testing import *


def test_basic():
    var p = Arc(4)
    var p2 = p
    p2.set(3)
    assert_equal(3, p[])


@value
struct ObservableDel(CollectionElement):
    var target: Pointer[Bool]

    fn __del__(owned self):
        self.target.store(True)


def test_deleter_not_called_until_no_references():
    var deleted = False
    var p = Arc(ObservableDel(Pointer.address_of(deleted)))
    var p2 = p
    _ = p ^
    assert_false(deleted)

    var vec = List[Arc[ObservableDel]]()
    vec.append(p2)
    _ = p2 ^
    assert_false(deleted)
    _ = vec ^
    assert_true(deleted)


def test_arc_bitcast():
    var arc_f32 = Arc[Scalar[DType.float32]](16.0)

    var arc_i32 = arc_f32._bitcast[Scalar[DType.int32]]()

    assert_equal(arc_f32[], 16.0)
    assert_equal(arc_i32[], 1098907648)


def main():
    test_basic()
    test_deleter_not_called_until_no_references()
    test_arc_bitcast()
