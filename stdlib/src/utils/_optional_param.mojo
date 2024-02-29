# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils.index import StaticIntTuple
from utils.list import Dim, DimList


@value
@register_passable("trivial")
struct OptionalParamInt[dim_parametric: Dim](EqualityComparable):
    """A class to represent an optionally parametric Int.
    If dim_parametric is known, the get method can be evaluated at compile time.
    Otherwise, the get method will be evaluated at runtime using the dynamic
    value supplied to the constructor.

    Parameters:
        dim_parametric: The optional Int parameter.
    """

    var dim_dynamic: Int

    @always_inline("nodebug")
    fn __init__(dim_dynamic: Int) -> Self:
        return Self {dim_dynamic: dim_dynamic}

    @always_inline("nodebug")
    fn get(self) -> Int:
        @parameter
        if dim_parametric.has_value():
            return dim_parametric.get()
        else:
            return self.dim_dynamic

    @always_inline("nodebug")
    fn __eq__(self, rhs: OptionalParamInt[dim_parametric]) -> Bool:
        return self.get() == rhs.get()

    @always_inline("nodebug")
    fn __ne__(self, rhs: OptionalParamInt[dim_parametric]) -> Bool:
        return not self == rhs


@value
@register_passable("trivial")
struct OptionalParamInts[rank: Int, dim_list_parametric: DimList](
    EqualityComparable
):
    """A class to represent an optionally parametric list of Ints.
    If an entry in dim_parametric is known, the at method can be evaluated at
    compile time. Otherwise, the at method will be evaluated at runtime using the
    dynamic value supplied to the constructor.

    Parameters:
        rank: The rank of the dimension list.
        dim_list_parametric: The list of optional Ints.
    """

    var dim_list_dynamic: StaticIntTuple[rank]

    @always_inline("nodebug")
    fn __init__(
        dim_list_dynamic: StaticIntTuple[rank],
    ) -> Self:
        return Self {dim_list_dynamic: dim_list_dynamic}

    @always_inline("nodebug")
    fn at[i: Int](self) -> Int:
        @parameter
        if dim_list_parametric.at[i]().has_value():
            return dim_list_parametric.at[i]().get()
        else:
            return self.dim_list_dynamic[i]

    @always_inline("nodebug")
    fn __eq__(self, rhs: OptionalParamInts[rank, dim_list_parametric]) -> Bool:
        return self.dim_list_dynamic == rhs.dim_list_dynamic

    @always_inline("nodebug")
    fn __ne__(self, rhs: OptionalParamInts[rank, dim_list_parametric]) -> Bool:
        return not self == rhs

    @always_inline("nodebug")
    fn get(self) -> StaticIntTuple[rank]:
        @parameter
        if dim_list_parametric.all_known[rank]():
            return dim_list_parametric
        else:
            return self.dim_list_dynamic
