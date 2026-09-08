# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Implements the VariadicList, ParameterList and VariadicPack types.

These are Mojo built-ins, so you don't need to import them.
"""

from std.builtin.constrained import _constrained_conforms_to
from std.builtin.rebind import downcast
from std.format._utils import FormatStruct, TypeNames
from std.builtin.globals import global_constant


struct _MLIR:
    comptime KGENTypeType = __mlir_type.`!kgen.type`

    comptime POPArrayType[
        size: __mlir_type.index, elt_type: AnyType
    ] = __mlir_type[`!pop.array<`, size, `, `, elt_type, `>`]

    comptime KGENParamListType[elt_type: Self.KGENTypeType] = __mlir_type[
        `!kgen.param_list<`, elt_type, `>`
    ]


# ===-----------------------------------------------------------------------===#
# TypeList
# ===-----------------------------------------------------------------------===#


struct TypeList[
    Trait: type_of(AnyType), //, values: _MLIR.KGENParamListType[Trait]
](Sized, TrivialRegisterPassable):
    """A compile-time list of types conforming to a common trait.

    `TypeList` provides type-level operations on variadic sequences of types,
    such as reversing, slicing, mapping, and membership testing.

    Parameters:
        Trait: The trait that all types in the list must conform to.
        values: The types in the list.

    Examples:

    ```mojo
    from std.builtin.variadics import TypeList
    from std.testing import assert_equal

    # Create a type list
    comptime tl = TypeList.of[Trait=AnyType, Int, String, Float64]()

    def main() raises:
        # Query length
        assert_equal(tl.length, 3)

        # Check membership
        comptime assert tl.contains[Int]()
        comptime assert not tl.contains[Bool]()

        # Index into the list
        comptime assert tl[0] == Int
    ```
    """

    comptime _mlir_type = _MLIR.KGENParamListType[Self.Trait]
    """The low-level MLIR type of the type list."""

    comptime length = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
            Self._mlir_type,
            ` `,
            +Self.values,
            `> : index`,
        ]
    )
    """The number of types in the list."""

    comptime __getitem_param__[idx: SIMDLength] = __mlir_attr[
        `#kgen.param_list.get<:`,
        Self._mlir_type,
        ` `,
        +Self.values,
        `, `,
        idx._mlir_value,
        `> : `,
        +Self.Trait,
    ]
    """Gets a type at the given index.

    Parameters:
        idx: The index of the type to access.
    """

    comptime _get_type_at_index[idx: __mlir_type.index] = __mlir_attr[
        `#kgen.param_list.get<:`,
        Self._mlir_type,
        ` `,
        +Self.values,
        `, `,
        idx,
        `> : `,
        +Self.Trait,
    ]
    """Gets a type at the given raw `index`.

    Unlike `__getitem_param__`, this accepts a raw `!kgen.index` so callers can
    index without constructing a `SIMDLength` (and thus without pulling in
    `SIMDLength` comparison machinery). Used by the stdlib plugin router during
    bootstrap.

    Parameters:
        idx: The raw `index` of the type to access.
    """

    @implicit
    @always_inline("builtin")
    def __init__(
        existing: TypeList[...],
        out self: TypeList[
            Trait=Self.Trait,
            __mlir_attr[
                `#kgen.upcast<`,
                existing.values,
                `> : `,
                _MLIR.KGENParamListType[Self.Trait],
            ],
        ],
    ) where __mlir_attr[
        `#kgen.is_refined_type<`, existing.Trait, `, `, Self.Trait, `>`
    ]:
        """Upcasts a TypeList to a base trait.

        Args:
            existing: The TypeList to upcast from.

        Constraints:
            The existing.Trait is more refined than Self.Trait.
        """
        pass

    # ===-------------------------------------------------------------------===#
    # Constructors
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Constructs a TypeList."""
        pass

    comptime of[Trait: type_of(AnyType), //, *Ts: Trait] = TypeList[
        Trait=Trait, Ts.values
    ]
    """Form a compile-time list of types with some elements, uninstantiated.

    Parameters:
        Trait: The type of the elements in the list.
        Ts: The types in the list.

    Examples:
        ```mojo
        comptime Ts = TypeList.of[Trait=AnyType, Int, String, Float64, Bool]
        comptime Ms = TypeList.of[Trait=Movable, Int, String, Float64, Bool]
        ```
    """

    comptime _IndexToIntTypeTabulateWrap[
        Trait: type_of(AnyType),
        ToT: Trait,
        //,
        ToWrap: __generator_type[Idx: Int] Trait,
        idx: __mlir_type.index,
    ] = ToWrap[Int(mlir_value=idx)]

    comptime tabulate[
        Trait: type_of(AnyType),
        ToT: Trait,
        //,
        count: Int,
        Mapper: __generator_type[Idx: Int] Trait,
    ] = TypeList[
        Trait=Trait,
        __mlir_attr[
            `#kgen.param_list.tabulate<`,
            count.__mlir_index__(),
            `,`,
            Self._IndexToIntTypeTabulateWrap[Trait=Trait, ToT=ToT, Mapper, ...],
            `> : `,
            _MLIR.KGENParamListType[Trait],
        ],
    ]
    """Builds a type list by applying an index-to-type mapper `count` times.

    Parameters:
        Trait: The trait of the generated TypeList.
        ToT: The type of the values in the generated TypeList.
        count: The number of times to apply the generator, the length of the result..
        Mapper: The generator to apply, mapping from Int to ToT.
    """

    comptime _SplatTypeTabulator[
        Trait: type_of(AnyType), T: Trait, index: Int
    ]: Trait = T

    comptime splat[
        Trait: type_of(AnyType), //, count: Int, type: Trait
    ] = TypeList.tabulate[count, Self._SplatTypeTabulator[Trait, type, _]]
    """Splats a type a given number of times.

    Parameters:
        Trait: The trait that the types conform to.
        count: The number of times to splat the type.
        type: The type to splat.
    """

    # Note: this is _concat instead of concat because it takes MLIR typelists
    comptime _concat[
        Trait: type_of(AnyType), //, *Ts: _MLIR.KGENParamListType[Trait]
    ] = TypeList[
        __mlir_attr[
            `#kgen.param_list.concat<`,
            Ts.values,
            `> :`,
            _MLIR.KGENParamListType[Trait],
        ]
    ]
    """Form a TypeList from the concatenation of multiple MLIR type lists.

    Parameters:
        Trait: The trait that types in the variadic sequences must conform to.
        Ts: The variadic sequences to concatenate.
    """

    # ===-------------------------------------------------------------------===#
    # Reductions
    # ===-------------------------------------------------------------------===#

    comptime _DiscardIndexWrapper[
        FromAndTo: AnyType,
        ToWrap: __generator_type[Prev: FromAndTo, From: Self.Trait] FromAndTo,
        PrevV: FromAndTo,
        VA: _MLIR.KGENParamListType[Self.Trait],
        idx: Int,
    ] = ToWrap[PrevV, TypeList[VA].__getitem_param__[idx]]
    """Adapts a (prev, element) reducer to the variadic reduce index signature."""

    comptime _IndexToIntWrap[
        ReduceT: AnyType,
        ToWrap: __generator_type[
            Prev: ReduceT, From: _MLIR.KGENParamListType[Self.Trait], Idx: Int
        ] ReduceT,
        PrevV: ReduceT,
        VA: _MLIR.KGENParamListType[Self.Trait],
        idx: __mlir_type.index,
    ] = ToWrap[PrevV, VA, Int(mlir_value=idx)]
    """Wrapper for type -> value."""

    comptime reduce[
        FromAndTo: AnyType,
        //,
        BaseVal: FromAndTo,
        Reducer: __generator_type[Prev: FromAndTo, From: Self.Trait] FromAndTo,
    ] = __mlir_attr[
        `#kgen.param_list.reduce<`,
        BaseVal,
        `,`,
        Self.values,
        `,`,
        Self._IndexToIntWrap[
            FromAndTo,
            Self._DiscardIndexWrapper[FromAndTo, Reducer, ...],
            ...,
        ],
        `> : `,
        +FromAndTo,
    ]
    """Folds this type list to a single value using an associative step function.

    Parameters:
        FromAndTo: The type of the accumulator and the final result.
        BaseVal: The initial accumulator value.
        Reducer: A compile-time generator
            `[prev: FromAndTo, element: Self.Trait] -> FromAndTo`.
    """

    comptime _PassIndexReducerWrapper[
        FromAndTo: AnyType,
        ToWrap: __generator_type[
            Prev: FromAndTo, From: Self.Trait, Idx: Int
        ] FromAndTo,
        PrevV: FromAndTo,
        VA: _MLIR.KGENParamListType[Self.Trait],
        list_idx: Int,
    ] = ToWrap[
        PrevV,
        TypeList[VA].__getitem_param__[list_idx],
        list_idx,
    ]
    """Adapts a (prev, element, index) reducer to the variadic reduce index signature."""

    comptime reduce_idx[
        FromAndTo: AnyType,
        //,
        BaseVal: FromAndTo,
        Reducer: __generator_type[
            Prev: FromAndTo, From: Self.Trait, Idx: Int
        ] FromAndTo,
    ] = __mlir_attr[
        `#kgen.param_list.reduce<`,
        BaseVal,
        `,`,
        Self.values,
        `,`,
        Self._IndexToIntWrap[
            FromAndTo,
            Self._PassIndexReducerWrapper[FromAndTo, Reducer, ...],
            ...,
        ],
        `> : `,
        +FromAndTo,
    ]
    """Folds this type list to a single value using a step function of position.

    Like `reduce`, but the reducer receives each element's index in this list,
    from `0` through `length - 1`, as a third compile-time argument.

    Parameters:
        FromAndTo: The type of the accumulator and the final result.
        BaseVal: The initial accumulator value.
        Reducer: A compile-time generator
            `[prev: FromAndTo, element: Self.Trait, idx: Int] -> FromAndTo`.
    """

    comptime _AnySatisfiesReducer[
        predicate: __generator_type[Type: Self.Trait] Bool,
        last_value: Bool,
        this_element: Self.Trait,
    ] = last_value or predicate[this_element]

    @always_inline("builtin")
    @staticmethod
    def any[
        predicate: __generator_type[Type: Self.Trait] Bool,
    ]() -> Bool:
        """Returns true if `predicate` holds for at least one type in this list.

        Parameters:
            predicate: A compile-time generator `[T: Self.Trait] -> Bool`.

        Returns:
            True if `predicate` holds for at least one type in this list, False otherwise.
        """

        return Self.reduce[
            False,
            Self._AnySatisfiesReducer[predicate, ...],
        ]

    comptime _AllSatisfiesReducer[
        predicate: __generator_type[Type: Self.Trait] Bool,
        last_value: Bool,
        this_element: Self.Trait,
    ] = last_value and predicate[this_element]

    @always_inline("builtin")
    @staticmethod
    def all_conforms_to[_trait: type_of(AnyType)]() -> Bool:
        """Returns true if all types in this list conform to `Trait`.

        Parameters:
            _trait: The trait to check for conformance to.

        Returns:
            True if all types in this list conform to `Trait`, False otherwise.
        """
        return conforms_to(Self.values, _trait)

    @always_inline("builtin")
    @staticmethod
    def all[
        predicate: __generator_type[Type: Self.Trait] Bool,
    ]() -> Bool:
        """Returns true if `predicate` holds for every type in this list.

        For an empty list, returns true.

        Parameters:
            predicate: A compile-time generator `[T: Self.Trait] -> Bool`.

        Returns:
            True if `predicate` holds for every type in this list, False otherwise.
        """
        return Self.reduce[
            True,
            Self._AllSatisfiesReducer[predicate, ...],
        ]

    comptime _ContainsTypePredicate[
        search: AnyType,
        element: Self.Trait,
    ] = element == search

    @always_inline("builtin")
    @staticmethod
    def contains[type: AnyType]() -> Bool:
        """Checks if a type is contained in this type list.

        Parameters:
            type: The type to check for.

        Returns:
            True if the type is contained in this type list, False otherwise.
        """
        return Self.any[Self._ContainsTypePredicate[type, ...],]()

    # ===-------------------------------------------------------------------===#
    # Mappings
    # ===-------------------------------------------------------------------===#

    comptime _MapTabulator[
        ToTrait: type_of(AnyType),
        Mapper: __generator_type[From: Self.Trait] ToTrait,
        idx: Int,
    ]: ToTrait = Mapper[Self.__getitem_param__[idx]]

    comptime map[
        ToTrait: type_of(AnyType),
        //,
        Mapper: __generator_type[From: Self.Trait] ToTrait,
    ] = TypeList.tabulate[
        Trait=ToTrait,
        Self.length,
        Self._MapTabulator[ToTrait, Mapper, idx=_],
    ]
    """Maps types to new types using a mapper.

    Returns a new TypeList resulting from applying `Mapper[T]` to each element
    in this list.

    Parameters:
        ToTrait: The trait that the output types conform to.
        Mapper: A generator that maps a type to another type.
            The generator type is `[T: Trait] -> To`.
    """

    comptime _FilterIdxTabulator[
        Predicate: __generator_type[Elt: Self.Trait, Idx: Int] Bool,
        idx: Int,
    ]: _MLIR.KGENParamListType[Self.Trait] = TypeList.of[
        Trait=Self.Trait, Self.__getitem_param__[idx]
    ]().values if Predicate[
        Self.__getitem_param__[idx], idx
    ] else TypeList.of[
        Trait=Self.Trait
    ]().values

    comptime filter_idx[
        Predicate: __generator_type[Elt: Self.Trait, Idx: Int] Bool,
    ] = TypeList._concat[
        *ParameterList.tabulate[
            Self.length,
            Self._FilterIdxTabulator[Predicate, _],
        ]()
    ]
    """Returns a new `TypeList` containing only elements selected by a predicate.

    The predicate is evaluated at compile time for each `(element, index)` pair.
    Indices are the positions in this list, from `0` through `length - 1`.

    Parameters:
        Predicate: A compile-time generator
            `[element: Self.Trait, idx: Int] -> Bool`. When it returns `True`,
            `element` is kept in order; when `False`, the element is dropped.

    Returns:
        A `TypeList` of the same trait containing the kept elements in order.
    """

    comptime _MapToValuesIntTabulator[
        ValueType: AnyType,
        //,
        Mapper: __generator_type[Elt: Self.Trait] ValueType,
        idx: Int,
    ]: ValueType = Mapper[Self.__getitem_param__[idx]]

    comptime map_to_values[
        ValueType: AnyType,
        //,
        Mapper: __generator_type[Elt: Self.Trait] ValueType,
    ] = ParameterList.tabulate[
        type=ValueType,
        Self.length,
        Self._MapToValuesIntTabulator[
            ValueType=ValueType, Mapper=Mapper, idx=_
        ],
    ]
    """Convert each type in this list to a value, forming a ParameterList.

    This is the value analogue of `ParameterList.map_to_type`: each element
    type is passed to `Mapper`, and the resulting values share the homogeneous
    element type `ValueType`.

    Parameters:
        ValueType: The element type of the resulting `ParameterList`.
        Mapper: A compile-time generator that maps an element type to a value.
            The generator type is `[T: Self.Trait] -> ValueType`.
    """

    # ===-------------------------------------------------------------------===#
    # Other
    # ===-------------------------------------------------------------------===#

    comptime _ReverseTabulator[idx: Int]: Self.Trait = Self.__getitem_param__[
        Self.length - 1 - idx
    ]
    comptime reverse = TypeList.tabulate[Self.length, Self._ReverseTabulator[_]]
    """Returns this type list in reverse order."""

    comptime _SliceTabulator[
        start: Int,
        idx: Int,
    ]: Self.Trait = Self.__getitem_param__[start + idx]

    comptime slice[
        start: Int = 0,
        end: Int = Self.length,
    ] = TypeList.tabulate[
        max(end - start, 0),
        Self._SliceTabulator[start, _],
    ]
    """Extracts a contiguous subsequence from the type list.

    Returns a new variadic containing elements from index `start` (inclusive)
    to index `end` (exclusive). Similar to Python's slice notation [start:end].

    Parameters:
        start: The starting index (inclusive). Defaults to 0.
        end: The ending index (exclusive). Defaults to the list length.

    Constraints:
        0 <= start <= end <= length.
    """

    @always_inline
    def __len__(self) -> Int:
        """Gets the number of elements in the TypeList.

        Returns:
            The number of elements on the TypeList.
        """
        return Self.length


# ===-----------------------------------------------------------------------===#
# ParameterList
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _ParameterListIter[type: Copyable, //, *values: type](
    ImplicitlyCopyable, Iterable, Iterator, TrivialRegisterPassable
):
    """Const Iterator for ParameterList.

    Parameters:
        type: The type of the elements in the list.
        values: The values in the list.
    """

    comptime Element = Self.type
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> ref[ImmStaticOrigin] Self.type:
        var index = self.index

        if index >= Self.values.size:
            raise StopIteration()
        self.index = index + 1
        return Self.values[index]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = Self.values.size - self.index
        return (len, {len})


# TODO: Make this conform to Iterable when IteratorType can be conditionally
# defined only when 'type' is Copyable.
struct ParameterList[type: AnyType, //, values: _MLIR.KGENParamListType[type]](
    Sized,
    TrivialRegisterPassable,
    Writable where conforms_to(type, Writable),
):
    """A utility class to access homogeneous variadic parameters.

    `ParameterList` is used by homogenous variadic parameter lists. Unlike
    `VariadicPack` (which is heterogeneous), `ParameterList` requires all
    elements to have the same type.

    `ParameterList` is only used for parameter lists, `VariadicList` is
    used for function arguments.

    For example, in the following function signature, `*args: Int` creates a
    `ParameterList` because it uses a single type `Int` instead of a variadic type
    parameter. The `*` before `args` indicates that `args` is a variadic argument,
    which means that the function can accept any number of arguments, but all
    arguments must have the same type `Int`.

    ```mojo
    def sum_values[*args: Int]():

        # Iterate over elements at compile time
        comptime for i in range(args.size):  # can also use len() at comptime

            # print() below is a run-time call, so placing args[i] directly inside
            # it will invoke run-time access and not compile-time access; both share
            # the same syntax. For illustration, we place comptime access in a separate
            # step here.
            comptime arg = args[i]
            print(arg, end=" ")

        print()

        # Iterate over elements at run-time
        var total = 0
        for i in range(len(args)): # can also use the comptime args.size
            total += args[i]

        print(total)

    def main():
        sum_values[1, 2, 3, 4, 5]()

        # Output:
        #  1 2 3 4 5
        #  15
    ```

    Parameters:
        type: The type of the elements in the list.
        values: The values in the list.
    """

    comptime _mlir_type = _MLIR.KGENParamListType[Self.type]
    """The low-level MLIR type of the parameter list."""

    comptime size: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
            Self._mlir_type,
            ` `,
            +Self.values,
            `> : index`,
        ]
    )
    """The number of elements in the list."""

    # ===-------------------------------------------------------------------===#
    # Accessors
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """
        return Self.size

    @staticmethod
    def get_span() -> Span[Self.type, ImmStaticOrigin]:
        """Gets a span of the elements on the variadic list.

        Returns:
            A span of the elements on the variadic list.
        """

        # Convert 'values' to use a flat array representation.
        comptime array = __mlir_attr[
            `#pop.variadic_to_array<:`,
            Self._mlir_type,
            ` `,
            +Self.values,
            `>`,
        ]
        # Map it into a runtime constant.
        ref static_array = global_constant[array]()
        # Get a pointer to the first element, not the whole array.
        var first_elt = Pointer(to=static_array).unsafe_bitcast[Self.type]()
        return Span(unsafe_ptr=first_elt, length=Self.size)

    @always_inline
    def __getitem__(self, idx: Int) -> ref[ImmStaticOrigin] Self.type:
        """Gets a single element on the variadic list.

        Args:
            idx: The index of the element to access on the list.

        Returns:
            The element on the list corresponding to the given index.
        """
        return self.get_span()[idx]

    comptime __getitem_param__[idx: SIMDLength]: Self.type = __mlir_attr[
        `#kgen.param_list.get<:`,
        Self._mlir_type,
        ` `,
        +Self.values,
        `, `,
        idx._mlir_value,
        `> : `,
        +Self.type,
    ]
    """Gets a single element on the variadic list."""

    # ===-------------------------------------------------------------------===#
    # Constructors
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self):
        """Constructs a ParameterList."""
        pass

    comptime empty_of[type: AnyType] = Self.of[type=type]
    """Form an empty compile-time list of values some element type.

    Parameters:
        type: The type of the elements in the list.

    Examples:
        ```mojo
        comptime Ints = ParameterList.empty_of[Int]()
        ```
    """

    comptime of[type: AnyType, //, *values: type] = ParameterList[
        type=type, values.values
    ]
    """Form a compile-time list of values with some elements, uninstantiated.

    Parameters:
        type: The type of the elements in the list.
        values: The values in the list.

    Examples:
        ```mojo
        comptime Ints = ParameterList.of[4, 5, 6]
        comptime Strings = ParameterList.of["foo", "bar", "baz"]
        ```
    """

    comptime _IndexToIntTabulateWrap[
        ToT: AnyType,
        //,
        ToWrap: __generator_type[Idx: Int] ToT,
        idx: __mlir_type.index,
    ]: ToT = ToWrap[Int(mlir_value=idx)]

    comptime tabulate[
        type: AnyType,
        //,
        count: Int,
        Mapper: __generator_type[Idx: Int] type,
    ] = ParameterList[
        type=type,
        __mlir_attr[
            `#kgen.param_list.tabulate<`,
            count.__mlir_index__(),
            `,`,
            Self._IndexToIntTabulateWrap[Mapper, ...],
            `> : `,
            _MLIR.KGENParamListType[type],
        ],
    ]
    """Builds a parameter list by applying an index-to-value mapper `count` times.

    Parameters:
        type: The element type of the resulting list.
        count: The length of the result; the mapper is invoked for each index in
            `0..<count`.
        Mapper: Compile-time generator mapping `Int` index to a value of `type`.
    """

    comptime _splat_tabulator[value: Some[AnyType], idx: Int] = value
    comptime splat[type: AnyType, //, count: Int, value: type] = Self.tabulate[
        count, Self._splat_tabulator[value, _]
    ]
    """Builds a homogeneous parameter list by repeating `value` `count` times.

    Parameters:
        type: The element type.
        count: The number of copies of `value` in the result.
        value: The value to repeat at every index.
    """

    comptime _concat[
        type: AnyType, //, *values: _MLIR.KGENParamListType[type]
    ] = __mlir_attr[
        `#kgen.param_list.concat<`,
        values.values,
        `> :`,
        _MLIR.KGENParamListType[type],
    ]
    """Represents the concatenation of multiple variadic sequences of values.

    Parameters:
        type: The types of the values in the variadic sequences.
        values: The variadic sequences to concatenate.
    """

    # ===-------------------------------------------------------------------===#
    # Reductions
    # ===-------------------------------------------------------------------===#

    comptime _DiscardIndexWrapper[
        FromAndTo: AnyType,
        ToWrap: __generator_type[Prev: FromAndTo, From: Self.type] FromAndTo,
        PrevV: FromAndTo,
        VA: Self._mlir_type,
        idx: __mlir_type.index,
    ] = ToWrap[PrevV, Self.__getitem_param__[SIMDLength(mlir_value=idx)]]
    """Takes an index because kgen.variadic.reduce passes it but we don't want it"""

    # TODO: This isn't returning a ParamList, so it should really be a 'def' so
    # we get parens on the caller side. However, that requires the type to be
    # materializable from parameter space to runtime.  We could split this into
    # reduce_param and reduce() where the later does materialization, or we
    # could just always use materialize?
    comptime reduce[
        FromAndTo: AnyType,
        //,
        BaseVal: FromAndTo,
        Reducer: __generator_type[Prev: FromAndTo, From: Self.type] FromAndTo,
    ] = __mlir_attr[
        `#kgen.param_list.reduce<`,
        BaseVal,
        `,`,
        Self.values,
        `,`,
        Self._DiscardIndexWrapper[FromAndTo, Reducer, ...],
        `> : `,
        +FromAndTo,
    ]
    """Form a value by applying a function that merges each element into a
    starting value, then return the result.

    Parameters:
        FromAndTo: The type of the input and output result.
        BaseVal: The initial value to reduce on.
        Reducer: A `[BaseVal: FromAndTo, T: Self.type] -> FromAndTo` that does the reduction.
    """

    comptime _AnySatisfiesReducer[
        predicate: __generator_type[Elt: Self.type] Bool,
        last_value: Bool,
        this_element: Self.type,
    ]: Bool = last_value or predicate[this_element]

    @always_inline("builtin")
    @staticmethod
    def any[predicate: __generator_type[Elt: Self.type] Bool]() -> Bool:
        """'any' applies a function to each element and returns true if
        the function returns True for any element.

        Parameters:
            predicate: A `[elt: Self.Type] -> Bool` comptime expression to apply.

        Returns:
            True if the predicate returns True for any element, False otherwise.
        """
        return Self.reduce[
            False,
            Self._AnySatisfiesReducer[predicate, ...],
        ]

    comptime _AllSatisfiesReducer[
        predicate: __generator_type[Elt: Self.type] Bool,
        last_value: Bool,
        this_element: Self.type,
    ]: Bool = last_value and predicate[this_element]

    @always_inline("builtin")
    @staticmethod
    def all[predicate: __generator_type[Elt: Self.type] Bool]() -> Bool:
        """'all' applies a function to each element and returns true if
        the function returns True for all elements.

        Parameters:
            predicate: A `[elt: Self.Type] -> Bool` comptime expression to apply.

        Returns:
            True if the predicate returns True for all elements, False otherwise.
        """
        return Self.reduce[
            True,
            Self._AllSatisfiesReducer[predicate, ...],
        ]

    # TODO(MOCO-3531): Remove downcasts here.
    comptime _ContainsValuePredicate[
        search_value: Self.type,
        element_value: Self.type,
    ] where conforms_to(Self.type, Equatable) = rebind[
        downcast[Self.type, Equatable]
    ](
        search_value
    ) == rebind[
        downcast[Self.type, Equatable]
    ](
        element_value
    )

    @always_inline("builtin")
    @staticmethod
    def contains[
        value: Self.type,
    ]() -> Bool where conforms_to(Self.type, Equatable):
        """
        Check if a value is contained in a variadic sequence of values.

        Parameters:
            value: The value to search for.

        Returns:
            True if the value is contained in the list, False otherwise.
        """
        return Self.any[Self._ContainsValuePredicate[value, ...]]()

    # ===-------------------------------------------------------------------===#
    # Mappings
    # ===-------------------------------------------------------------------===#

    comptime _MapToTypeTabulator[
        Trait: type_of(AnyType),
        //,
        Mapper: __generator_type[Elt: Self.type] Trait,
        idx: Int,
    ]: Trait = Mapper[Self.__getitem_param__[idx]]

    comptime map_to_type[
        Trait: type_of(AnyType),
        //,
        Mapper: __generator_type[Elt: Self.type] Trait,
    ] = TypeList.tabulate[
        Trait=Trait,
        Self.size,
        Self._MapToTypeTabulator[Trait=Trait, Mapper=Mapper, idx=_],
    ]
    """Convert each element of this list into a type, forming a TypeList with
    the result.

    Parameters:
        Trait: The trait of the resulting TypeList.
        Mapper: A generator that maps an element of this list to a type.
    """

    # ===-------------------------------------------------------------------===#
    # Other
    # ===-------------------------------------------------------------------===#

    def _write_elements[
        is_repr: Bool = False
    ](self, mut writer: Some[Writer]) where conforms_to(Self.type, Writable):
        writer.write_string("(")
        for i in range(len(self)):
            if i > 0:
                writer.write_string(", ")

            comptime if is_repr:
                self[i].write_repr_to(writer)
            else:
                self[i].write_to(writer)
        writer.write_string(")")

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.type, Writable):
        """Writes the elements of this variadic list to a writer.

        Constraints:
            `type` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        self._write_elements(writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.type, Writable):
        """Writes the repr of this variadic list to a writer.

        Constraints:
            `type` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_elements[is_repr=True](w)

        FormatStruct(writer, "ParameterList").params(
            TypeNames[Self.type](),
        ).fields(write_fields)

    # We can only support iteration when the elements are Copyable, because
    # iterators currently need to return the elements by value.
    @always_inline
    def __iter__(
        ref self,
    ) -> _ParameterListIter[
        *ParameterList[
            rebind[_MLIR.KGENParamListType[downcast[Self.type, Copyable]]](
                Self.values
            )
        ]()
    ] where conforms_to(Self.type, Copyable):
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return {0}


# ===-----------------------------------------------------------------------===#
# VariadicList
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _VariadicListIter[
    elt_is_mutable: Bool,
    //,
    elt_type: AnyType,
    elt_origin: Origin[mut=elt_is_mutable],
    list_origin: ImmOrigin,
    is_owned: Bool,
](RegisterPassable):
    """Iterator for VariadicList.

    Parameters:
        elt_is_mutable: Whether the elements in the list are mutable.
        elt_type: The type of the elements in the list.
        elt_origin: The origin of the elements.
        list_origin: The origin of the VariadicList.
        is_owned: Whether the elements are owned by the list because they are
                  passed as an 'var' argument.
    """

    comptime variadic_list_type = VariadicList[
        origin=Self.elt_origin,
        Self.elt_type,
        Self.is_owned,
    ]

    comptime Element = Self.elt_type

    var index: Int
    var src: Pointer[Self.variadic_list_type, Self.list_origin]

    def __init__(
        out self,
        index: Int,
        ref[Self.list_origin] list: Self.variadic_list_type,
    ):
        self.index = index
        self.src = Pointer(to=list)

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> ref[self.src[][0]] Self.elt_type:
        var index = self.index
        if index == len(self.src[]):
            raise StopIteration()
        self.index = index + 1
        return self.src[][index]


struct VariadicList[
    elt_is_mutable: Bool,
    origin: Origin[mut=elt_is_mutable],
    //,
    element_type: AnyType,
    is_owned: Bool,
](
    RegisterPassable,
    Sized,
    Writable where conforms_to(element_type, Writable),
):
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes references to the elements in a
    way that can be enumerated.  Each element may be accessed with `elt[]`.

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument.
        origin: The origin of the underlying elements.
        element_type: The type of the elements in the list.
        is_owned: Whether the elements are owned by the list because they are
                  passed as an 'var' argument.
    """

    comptime _EltPointerType = Pointer[Self.element_type, Self.origin]
    # FIXME: This should be the origin of the container, not UntrackedOrigin.
    var _value: Span[Self._EltPointerType, ImmUntrackedOrigin]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @always_inline
    @implicit
    def __init__[
        size: __mlir_type.index, container_origin: ImmOrigin
    ](
        out self,
        value: Pointer[
            _MLIR.POPArrayType[size, Self._EltPointerType._mlir_lit_ref],
            container_origin,
        ]._mlir_lit_ref,
    ):
        """Constructs a VariadicList from a compiler-generated array of element
        pointers.

        Parameters:
            size: The number of elements in the variadic list.
            container_origin: The origin of the container.

        Args:
            value: The raw reference to the array of element pointers.
        """
        # Convert the !lit.ref to a pointer, then cast to a pointer to
        # the first element.
        var array_up = Pointer(
            to=Pointer(_mlir_value=value)[]
        ).unsafe_origin_cast[ImmUntrackedOrigin]()
        var elt_ptr = Pointer[_, ImmUntrackedOrigin](
            _mlir_value=__mlir_op.`pop.array.gep`(
                array_up._get_kgen_pointer(),
                Int(0).__mlir_index__(),
            )
        ).unsafe_bitcast[Self._EltPointerType]()
        self._value = Span(unsafe_ptr=elt_ptr, length=Int(mlir_value=size))

    # The destructor for this type is trivial if not an "owned" list.
    comptime __del__is_trivial: Bool = not Self.is_owned

    @always_inline
    def __deinit__(deinit self):
        """Destructor that releases elements if owned."""

        # Destroy each element if this variadic has owned elements, destroy
        # them.  We destroy in backwards order to match how arguments are
        # normally torn down when CheckLifetimes is left to its own devices.
        comptime if Self.is_owned:
            _constrained_conforms_to[
                conforms_to(Self.element_type, Deinitable),
                Parent=Self,
                Element=Self.element_type,
                ParentConformsTo="Deinitable",
            ]()
            comptime assert conforms_to(Self.element_type, Deinitable)

            for i in reversed(range(len(self))):
                # Safety: We own the elements in this list.
                Pointer(to=self[i]).unsafe_mut_cast[
                    True
                ]().unsafe_deinit_pointee()

    def consume_elements(
        deinit self,
        elt_handler: Some[def(idx: Int, var elt: Self.element_type)],
    ) where (
        Self.is_owned,
        "consume_elements may only be called on owned variadic lists",
    ):
        """Consume the variadic list by transferring ownership of each element
        into the provided closure one at a time.  This is only valid on 'owned'
        variadic lists.

        Args:
            elt_handler: A function that will be called for each element of the
                         list.
        """

        for i in range(len(self)):
            var ptr = Pointer(to=self[i])
            # TODO: Cannot use Pointer.unsafe_take_pointee because it requires
            # the element to be Movable, which is not required here.
            elt_handler(
                i, __get_address_as_owned_value(ptr._get_kgen_pointer())
            )

    # FIXME: This is a hack to work around a miscompile, do not use.
    def _annihilate(deinit self):
        pass

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """
        return len(self._value)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __getitem__[
        self_origin: ImmOrigin
    ](ref[self_origin] self, idx: Int) -> ref[
        # cast mutability of self to match the mutability of the element,
        # since that is what we want to use in the ultimate reference and
        # the union overall doesn't matter.
        origin_of(Self.origin, self_origin).unsafe_mut_cast[
            Self.elt_is_mutable
        ]()
    ] Self.element_type:
        """Gets a single element on the variadic list.

        Parameters:
            self_origin: The origin of the list.

        Args:
            idx: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return self._value.unsafe_ptr()[unsafe_offset=idx][]

    def _write_elements[
        is_repr: Bool = False
    ](self, mut writer: Some[Writer]) where conforms_to(
        Self.element_type, Writable
    ):
        writer.write_string("(")
        for i in range(len(self)):
            if i > 0:
                writer.write_string(", ")

            comptime if is_repr:
                self[i].write_repr_to(writer)
            else:
                self[i].write_to(writer)
        writer.write_string(")")

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.element_type, Writable):
        """Writes the elements of this variadic list to a writer.

        Constraints:
            `element_type` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        self._write_elements(writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.element_type, Writable):
        """Writes the repr of this variadic list to a writer.

        Constraints:
            `element_type` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_elements[is_repr=True](w)

        FormatStruct(writer, "VariadicList").params(
            TypeNames[Self.element_type](),
        ).fields(write_fields)

    def __iter__[
        self_origin: ImmOrigin
    ](
        ref[self_origin] self,
    ) -> _VariadicListIter[
        Self.element_type, Self.origin, self_origin, Self.is_owned
    ]:
        """Iterate over the list.

        Parameters:
            self_origin: The origin of the list.

        Returns:
            An iterator to the start of the list.
        """
        return {0, self}


# ===-----------------------------------------------------------------------===#
# VariadicPack
# ===-----------------------------------------------------------------------===#


struct VariadicPack[
    elt_is_mutable: Bool,
    origin: Origin[mut=elt_is_mutable],
    element_trait: type_of(AnyType),
    //,
    is_owned: Bool,
    *Ts: element_trait,
](
    Copyable where (not is_owned, "Cannot copy an owned variadic pack."),
    RegisterPassable,
    Sized,
):
    """A utility class to access heterogeneous variadic function arguments.

    `VariadicPack` is used when you need to accept variadic arguments where each
    argument can have a different type, but all types conform to a common trait.
    Unlike `ParameterList` (which is homogeneous), `VariadicPack` allows each
    element to have a different concrete type.

    `VariadicPack` is essentially a heterogeneous tuple that gets lowered to a
    struct at runtime. Because `VariadicPack` is a heterogeneous tuple (not an
    array), each element can have a different size and memory layout, which
    means the compiler needs to know the exact type of each element at compile
    time to generate the correct memory layout and access code.

    Therefore, indexing into `VariadicPack` requires compile-time indices using
    `comptime for` loops, whereas indexing into `ParameterList` uses runtime
    indices.

    For example, in the following function signature, `*args: *ArgTypes` creates a
    `VariadicPack` because it uses a variadic type parameter `*ArgTypes` instead
    of a single type. The `*` before `ArgTypes` indicates that `ArgTypes` is a
    variadic type parameter, which means that the function can accept any number
    of arguments, and each argument can have a different type. This allows each
    argument to have a different type while all types must conform to the
    `Intable` trait.

    ```mojo
    def count_many_things[*ArgTypes: Intable](*args: *ArgTypes) -> Int:
        var total = 0

        # Must use comptime for loop because args is a VariadicPack
        comptime for i in range(args.__len__()):
            # Each args[i] has a different concrete type from *ArgTypes
            # The compiler generates specific code for each iteration
            total += Int(args[i])

        return total

    def main():
        print(count_many_things(Int8(5), UInt32(11), Int(12)))  # Prints: 28
    ```

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument pack.
        origin: The origin of the underlying elements.
        element_trait: The trait that each element of the pack conforms to.
        is_owned: Whether the elements are owned by the pack. If so, the pack
                  will release the elements when it is destroyed.
        Ts: The list of types held by the argument pack.
    """

    @deprecated(use=Ts)
    comptime element_types = Self.Ts
    """The list of types held by the argument pack. Deprecated alias for `Ts`.
    """

    comptime _mlir_type = __mlir_type[
        `!lit.ref.pack<:param_list<`,
        Self.element_trait,
        `> `,
        Self.Ts.values,
        `, `,
        Self.origin._mlir_origin,
        `>`,
    ]

    var _value: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @always_inline("nodebug")
    # This disables nested origin exclusivity checking because it is taking a
    # raw variadic pack which can have nested origins in it (which this does not
    # dereference).
    @__unsafe_nested_origins_read_only
    def __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicPack from the internal representation.

        Args:
            value: The argument to construct the pack with.
        """
        self._value = value

    @always_inline("nodebug")
    def __init__(
        out self, *, copy: Self
    ) where (not Self.is_owned, "Cannot copy an owned variadic pack."):
        """Copy construct the variadic pack.

        Args:
            copy: The pack to copy from.

        Constraints:
            The variadic pack must not be owned.
        """

        self._value = copy._value

    # The destructor for this type is trivial if not an "owned" pack.
    comptime __del__is_trivial: Bool = not Self.is_owned

    @always_inline("nodebug")
    def __deinit__(deinit self):
        """Destructor that releases elements if owned."""

        comptime if Self.is_owned:
            comptime for i in reversed(range(Self.__len__())):
                comptime element_type = Self.Ts[i]
                _constrained_conforms_to[
                    conforms_to(element_type, Deinitable),
                    Parent=Self,
                    Element=element_type,
                    ParentConformsTo="Deinitable",
                ]()
                comptime assert conforms_to(element_type, Deinitable)

                # Safety: We own the elements in this pack.
                Pointer(to=self[i]).unsafe_mut_cast[
                    True
                ]().unsafe_deinit_pointee()

    def consume_elements[
        elt_handler: def[idx: Int](var elt: Self.Ts[idx]) capturing
    ](deinit self) where (
        Self.is_owned,
        "consume_elements may only be called on owned variadic packs",
    ):
        """Consume the variadic pack by transferring ownership of each element
        into the provided closure one at a time.  This is only valid on 'owned'
        variadic packs.

        Parameters:
            elt_handler: A function that will be called for each element of the
                         pack.
        """

        comptime for i in range(Self.__len__()):
            var ptr = Pointer(to=self[i])
            # TODO: Cannot use Pointer.unsafe_take_pointee because it requires
            # the element to be Movable, which is not required here.
            elt_handler[i](
                __get_address_as_owned_value(ptr._get_kgen_pointer())
            )

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    @staticmethod
    def __len__() -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """
        return Self.Ts.length

    @always_inline
    def __len__(self) -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """
        return Self.__len__()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __getitem_param__[index: Int](self) -> ref[Self.origin] Self.Ts[index]:
        """Return a reference to an element of the pack.

        Parameters:
            index: The element of the pack to return.

        Returns:
            A reference to the element.  The Pointer's mutability follows the
            mutability of the pack argument convention.
        """
        var litref_elt = __mlir_op.`lit.ref.pack.extract`[
            index=index.__mlir_index__()
        ](self._value)
        return __get_litref_as_mvalue(litref_elt)

    # ===-------------------------------------------------------------------===#
    # C Pack Utilities
    # ===-------------------------------------------------------------------===#

    # FIXME: bound by AnyType
    comptime _kgen_element_types = rebind[
        _MLIR.KGENParamListType[__mlir_type.`!kgen.type`]
    ](Self.Ts.values)
    """This is the element_types list lowered to `variadic<type>` type for kgen.
    """

    # FIXME: bound by AnyType
    comptime _variadic_pointer_types = __mlir_attr[
        `#kgen.param.expr<variadic_ptr_map, `,
        Self._kgen_element_types,
        `, 0: index>: `,
        _MLIR.KGENParamListType[__mlir_type.`!kgen.type`],
    ]
    """Use variadic_ptr_map to construct the type list of the !kgen.struct that
    the !lit.ref.pack will lower to.  It exposes the pointers introduced by the
    references.
    """
    comptime _kgen_pack_with_pointer_type = __mlir_type[
        `!kgen.struct<:param_list<type> `,
        Self._variadic_pointer_types,
        ` isParamPack>`,
    ]
    """This is the !kgen.struct type with pointer elements."""

    @doc_hidden
    @always_inline("nodebug")
    def get_as_kgen_pack(self) -> Self._kgen_pack_with_pointer_type:
        """This rebinds `in_pack` to the equivalent `!kgen.struct` with kgen
        pointers."""
        return rebind[Self._kgen_pack_with_pointer_type](self._value)

    # FIXME: bound by AnyType
    comptime _variadic_with_pointers_removed = __mlir_attr[
        `#kgen.param.expr<variadic_ptrremove_map, `,
        Self._variadic_pointer_types,
        `>: `,
        _MLIR.KGENParamListType[__mlir_type.`!kgen.type`],
    ]
    comptime _loaded_kgen_pack_type = __mlir_type[
        `!kgen.struct<:param_list<type> `,
        Self._variadic_with_pointers_removed,
        ` isParamPack>`,
    ]
    """This is the `!kgen.struct` type that happens if one loads all the elements
    of the pack.
    """

    # Returns all the elements in a kgen.pack.
    # Useful for FFI, such as calling printf. Otherwise, avoid this if possible.
    @doc_hidden
    @always_inline("nodebug")
    def get_loaded_kgen_pack(self) -> Self._loaded_kgen_pack_type:
        """This returns the stored KGEN pack after loading all of the elements.
        """
        return __mlir_op.`kgen.struct.load_indirect`(self.get_as_kgen_pack())

    def _write_to[
        O1: ImmOrigin,
        O2: ImmOrigin,
        O3: ImmOrigin,
        *,
        is_repr: Bool = False,
    ](
        self,
        mut writer: Some[Writer],
        start: StringSlice[O1] = StaticString(""),
        end: StringSlice[O2] = StaticString(""),
        sep: StringSlice[O3] = StaticString(", "),
    ) where Self.Ts.all_conforms_to[Writable]():
        """Writes a sequence of writable values from a pack to a writer with
        delimiters.

        This function formats a variadic pack of writable values as a delimited
        sequence, writing each element separated by the specified separator and
        enclosed by start and end delimiters.

        Parameters:
            O1: The origin of the open `StringSlice`.
            O2: The origin of the close `StringSlice`.
            O3: The origin of the separator `StringSlice`.
            is_repr: Whether to use repr formatting for elements.

        Args:
            writer: The writer to write to.
            start: The starting delimiter.
            end: The ending delimiter.
            sep: The separator between items.
        """
        writer.write_string(start)

        comptime for i in range(self.__len__()):
            comptime if i != 0:
                writer.write_string(sep)

            comptime if is_repr:
                self[i].write_repr_to(writer)
            else:
                self[i].write_to(writer)
        writer.write_string(end)

    @no_inline
    def write_to(
        self,
        mut writer: Some[Writer],
    ) where Self.Ts.all_conforms_to[Writable]():
        """Writes the elements of this pack to a writer.

        Args:
            writer: The object to write to.
        """
        self._write_to(
            writer,
            start=StaticString("("),
            end=StaticString(")"),
        )

    @no_inline
    def write_repr_to(
        self,
        mut writer: Some[Writer],
    ) where Self.Ts.all_conforms_to[Writable]():
        """Writes the repr of the elements of this pack to a writer.

        Args:
            writer: The object to write to.
        """
        self._write_to[is_repr=True](
            writer,
            start=StaticString("("),
            end=StaticString(")"),
        )


@always_inline
def _call_with_dynamic_pack_pointers[
    ArgTrait: type_of(AnyType),
    Args: TypeList[Trait=ArgTrait, ...],
    RetType: AnyType,
    RaiseType: AnyType,
    MakeElemPtr: def[idx: Int]() -> Pointer[Args[idx], MutUnsafeAnyOrigin],
    //,
    user_func: def(* args: * Args) thin raises RaiseType -> RetType,
](make_elem_ptr: MakeElemPtr) raises RaiseType -> RetType:
    """Call `user_func` using a `VariadicPack` whose elements are initialized
    by per-index calls to `make_elem_ptr`.

    Ordinarily, the compiler constructs `VariadicPack` values automatically and
    implicitly from the syntax of argument sequences when calling a function
    that takes a `*args: *Ts` argument pack. This is a language feature, and
    not directly exposed programmatically.

    However, in some uncommon situations, it is useful to be able to initialize
    the elements in such a pack from dynamic data at runtime.

    The `make_elem_ptr` closure will be invoked exactly once for each type in
    `Args`, with `idx` values in ascending order.

    Parameters:
        Args: The argument types accepted by `user_func`.
        RetType: Type returned by the closure to be called.
        MakeElemPtr: Element initialization function.
        user_func: The user function to call with the dynamically initialized
            argument pack.

    Args:
        make_elem_ptr: Closure returning a pointer to the pack argument value
            for a given index. The pointer origin must be valid for the duration
            of the call to `user_func`.
    """
    comptime ToPointer[T: ArgTrait]: ImplicitlyCopyable & Deinitable = Pointer[
        T, MutUnsafeAnyOrigin
    ]

    # The tuple cannot be default constructed because `Pointer` is not default
    # constructible. The tuple however is initialized in the loop below.
    var pointers: Tuple[*Args.map[ToPointer]()]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(pointers))

    # Get a pointer to each pack element value. It's up to the caller
    # to ensure that these pointers are valid for the duration of this function.
    comptime for i in range(Args.length):
        var element = make_elem_ptr[i]()
        # FIXME(MOCO-4635): This rebind is needed to work around a type folding bug
        pointers[i] = rebind[type_of(pointers[i])](element)

    # Construct a pack from the initialized locals
    comptime BorrowedPack = VariadicPack[
        origin=MutUnsafeAnyOrigin,
        element_trait=ArgTrait,
        False,
        *Args,
    ]
    var borrowed = BorrowedPack(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[
            _type=BorrowedPack._mlir_type
        ](pointers._mlir_value)
    )

    # Call the user primary function
    return user_func(*borrowed)
