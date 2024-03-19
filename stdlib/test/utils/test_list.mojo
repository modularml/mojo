# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from buffer.list import Dim, DimList
from testing import *

from utils.index import StaticIntTuple


# CHECK-LABEL: test_dim_list
fn test_dim_list():
    print("== test_dim_list")

    var lst0 = DimList(1, 2, 3, 4)
    var lst1 = DimList(Dim(), 2, 3, 4)

    # CHECK: [1, 2, 3, 4]
    print(lst0)

    # CHECK: 24
    print(lst0.product[4]().get())

    # CHECK: True
    print(lst0.all_known[4]())

    # CHECK: False
    print(lst1.all_known[4]())

    # CHECK: True
    print(lst1.all_known[1, 4]())

    # CHECK: False
    print(lst1.has_value[0]())

    # CHECK: True
    print(lst1.has_value[2]())


# CHECK-LABEL: test_dim
fn test_dim():
    print("== test_dim")

    var dim0 = Dim(8)
    # CHECK: True
    print(dim0.is_multiple[4]())

    var dim1 = Dim()
    # CHECK: False
    print(dim1.is_multiple[4]())

    var dim2 = dim0 // 2
    # CHECK: True
    print(dim2.has_value())
    # CHECK: 4
    print(dim2.value.value())

    var dim3 = dim1 // Dim()
    # CHECK: False
    print(dim3.has_value())


def test_dim_to_string():
    assert_equal(str(Dim()), "?")
    assert_equal(str(Dim(33)), "33")
    assert_equal(str(DimList(2, Dim(), 3)), "[2, ?, 3]")
    assert_equal(str(DimList.create_unknown[5]()), "[?, ?, ?, ?, ?]")


def main():
    test_dim_list()
    test_dim()
    test_dim_to_string()
