# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from testing import *

from utils._optional_param import OptionalParamInt, OptionalParamInts
from utils.index import StaticIntTuple
from utils.list import Dim, DimList


# CHECK-LABEL: test_dim_list
fn test_dim_list():
    print("== test_dim_list")

    var lst0 = DimList(1, 2, 3, 4)
    var lst1 = DimList(Dim(), 2, 3, 4)

    # CHECK: [1, 2, 3, 4]
    print[4](lst0)

    # CHECK: 24
    print(lst0.product[4]().get())

    # CHECK: True
    print(lst0.all_known[4]())

    # CHECK: False
    print(lst1.all_known[4]())

    # CHECK: True
    print(lst1.all_known[1, 4]())


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
    assert_equal(str(DimList(2, Dim(), 3)), "2, ?, 3")
    assert_equal(str(DimList.create_unknown[5]()), "?, ?, ?, ?, ?")


# CHECK-LABEL: test_opt_param_int
fn test_opt_param_int():
    print("=== test_opt_param_int")
    alias dp0 = Dim(0)
    # CHECK: 0
    print(OptionalParamInt[dp0](1).get())

    alias dp1 = Dim()
    # CHECK: 1
    print(OptionalParamInt[dp1](1).get())


# CHECK-LABEL: test_opt_param_ints
fn test_opt_param_ints():
    print("=== test_opt_param_ints")
    alias dp0 = DimList(0, 0)
    var d0 = OptionalParamInts[2, dp0](StaticIntTuple[2](1, 1))
    # CHECK: 0
    print(d0.at[0]())
    # CHECK: 0
    print(d0.at[1]())

    alias dp1 = DimList(Dim(), 0)
    var d1 = OptionalParamInts[2, dp1](StaticIntTuple[2](1, 1))
    # CHECK: 1
    print(d1.at[0]())
    # CHECK: 0
    print(d1.at[1]())


def main():
    test_dim_list()
    test_dim()
    test_dim_to_string()
    test_opt_param_int()
    test_opt_param_ints()
