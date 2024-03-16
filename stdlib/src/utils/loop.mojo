# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Implements higher-order functions.

You can import these APIs from the `utils.loop` module. For example:

```mojo
from utils.loop import unroll
```
"""


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx: Int] () capturing -> None,
    count: Int,
]():
    """Repeatedly evaluates a function `count` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        count: A number of repetitions.
    """
    _unroll_impl[func, 0, count]()


@always_inline
fn unroll[
    func: fn[idx: Int] () raises capturing -> None,
    count: Int,
]() raises:
    """Repeatedly evaluates a function `count` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        count: A number of repetitions.
    """
    _unroll_impl[func, 0, count]()


@always_inline
fn _unroll_impl[
    func: fn[idx: Int] () capturing -> None,
    idx: Int,
    count: Int,
]():
    @parameter
    if idx < count:
        func[idx]()
        _unroll_impl[func, idx + 1, count]()


@always_inline
fn _unroll_impl[
    func: fn[idx: Int] () raises capturing -> None,
    idx: Int,
    count: Int,
]() raises:
    @parameter
    if idx < count:
        func[idx]()
        _unroll_impl[func, idx + 1, count]()


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx0: Int, idx1: Int] () capturing -> None,
    dim0: Int,
    dim1: Int,
]():
    """Repeatedly evaluates a 2D nested loop.

    Parameters:
        func: The function to evaluate. The function should take two `Int`
          arguments: the outer and inner loop index values.
        dim0: The first dimension size.
        dim1: The second dimension size.
    """

    @always_inline
    @parameter
    fn outer_func_wrapper[idx0: Int]():
        @always_inline
        @parameter
        fn inner_func_wrapper[idx1: Int]():
            func[idx0, idx1]()

        unroll[inner_func_wrapper, dim1]()

    unroll[outer_func_wrapper, dim0]()


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx0: Int, idx1: Int, idx2: Int] () capturing -> None,
    dim0: Int,
    dim1: Int,
    dim2: Int,
]():
    """Repeatedly evaluates a 3D nested loop.

    Parameters:
        func: The function to evaluate. The function should take three `Int`
          arguments: one for each nested loop index value.
        dim0: The first dimension size.
        dim1: The second dimension size.
        dim2: The second dimension size.
    """

    @always_inline
    @parameter
    fn func_wrapper[idx0: Int, idx1: Int]():
        alias _idx1 = idx1 // dim2
        alias _idx2 = idx1 % dim2
        func[idx0, _idx1, _idx2]()

    unroll[func_wrapper, dim0, dim1 * dim2]()
