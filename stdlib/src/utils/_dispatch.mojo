# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm import unroll
from memory.buffer import Buffer, DynamicRankBuffer, NDBuffer


@always_inline
fn rank_axis_dispatch[
    func: fn[rank: Int, axis: Int] () capturing -> None, max_rank: Int
](rank: Int, axis: Int) raises:
    if axis < 0 or axis >= rank:
        raise Error("axis must be be non-negative and less than rank")

    @always_inline
    @parameter
    fn _func_rank[rank: Int]():
        @always_inline
        @parameter
        fn _func_axis[axis_static: Int]():
            if axis == axis_static:
                func[rank, axis_static]()

        unroll[rank, _func_axis]()  # ensure that 0 <= axis < rank

    range_dispatch[_func_rank, 0, max_rank, 1](rank)


@always_inline
fn range_dispatch[
    func: fn[start: Int] () capturing -> None,
    start: Int,
    end: Int,
    step: Int,
](value: Int) raises:
    if value < start or value >= end:
        raise Error("out of range value in range_dispatch")

    _range_dispatch[func, start, end, step](value)


@always_inline
fn _range_dispatch[
    func: fn[start: Int] () capturing -> None,
    start: Int,
    end: Int,
    step: Int,
](value: Int):
    @parameter
    if start < end:
        if start == value:
            func[start]()
        _range_dispatch[func, start + step, end, step](value)


@always_inline
fn dispatch_3bool[
    func: fn[x: Bool, y: Bool, z: Bool] () capturing -> None,
](x: Bool, y: Bool, z: Bool):
    if z:

        @always_inline
        @parameter
        fn func_x[x: Bool, y: Bool]():
            func[x, y, True]()

        dispatch_2bool[func_x](x, y)
    else:

        @always_inline
        @parameter
        fn func_y[x: Bool, y: Bool]():
            func[x, y, False]()

        dispatch_2bool[func_y](x, y)


@always_inline
fn dispatch_2bool[
    func: fn[x: Bool, y: Bool] () capturing -> None
](x: Bool, y: Bool):
    if x and y:
        func[True, True]()
    elif not x and y:
        func[False, True]()
    elif x and not y:
        func[True, False]()
    else:
        func[False, False]()


@always_inline
fn dispatch_1bool[
    func: fn[x: Bool] () capturing -> None,
](x: Bool):
    if x:
        func[True]()
    else:
        func[False]()
