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


struct _MLIR:
    comptime KGENTypeType = __mlir_type.`!kgen.type`
    comptime POPArrayType[
        size: __mlir_type.index, elt_type: AnyType
    ] = __mlir_type[`!pop.array<`, size, `, `, elt_type, `>`]

    comptime KGENParamListType[elt_type: Self.KGENTypeType] = __mlir_type[
        `!kgen.param_list<`, ` `, elt_type, `>`
    ]


comptime ImmOrigin = Origin[mut=False]
comptime MutOrigin = Origin[mut=True]

comptime AnyOrigin[*, mut: Bool] = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.any.origin : !lit.origin<`, +mut._mlir_value, `>`
    ]
]()
comptime ImmAnyOrigin = AnyOrigin[mut=False]
comptime MutAnyOrigin = AnyOrigin[mut=True]

comptime UntrackedOrigin[*, mut: Bool] = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.origin.union<> : !lit.origin<`,
        mut._mlir_value,
        `>`,
    ],
]()
comptime ImmUntrackedOrigin = UntrackedOrigin[mut=False]
comptime MutUntrackedOrigin = UntrackedOrigin[mut=True]
comptime ImmStaticOrigin = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.origin.field<`,
        `#lit.static.origin : !lit.origin<false>`,
        `, "__constants__"> : !lit.origin<false>`,
    ]
]()

comptime OriginSet = __mlir_type.`!lit.origin.set`
comptime Never = __mlir_type.`!kgen.never`
comptime EllipsisType = __mlir_type.`!lit.ellipsis`

comptime _lit_origin_type_of_mut[mut: Bool] = __mlir_type[
    `!lit.origin<`, mut._mlir_value, `>`
]


struct Origin[mut: Bool, _mlir_origin: _lit_origin_type_of_mut[mut], //](
    TrivialRegisterPassable
):
    @always_inline("builtin")
    def __init__(out self):
        pass

    @always_inline("builtin")
    @implicit
    def __init__(v: Origin) -> Origin[mut=False, _mlir_origin=v._mlir_origin]:
        return {}

    @always_inline("builtin")
    @staticmethod
    def unsafe_mut_cast[
        dest_mut: Bool
    ]() -> Origin[
        _mlir_origin=__mlir_attr[
            `#lit.origin.mutcast<`,
            Self._mlir_origin,
            `> : !lit.origin<`,
            dest_mut._mlir_value,
            `>`,
        ],
    ]:
        return {}

    comptime equals[rhs: Origin]: Bool = __mlir_attr[
        `#lit.origin.eq<`,
        Self._mlir_origin,
        `, `,
        rhs._mlir_origin,
        `> : i1`,
    ]
    comptime contains[element: Origin]: Bool = Self.equals[
        origin_of(Self._mlir_origin, element._mlir_origin)
    ]

    comptime _get_owned_interior[name: StringLiteral] = Origin[
        _mlir_origin=__mlir_attr[
            `#lit.interior.origin<`,
            Self._mlir_origin,
            `, `,
            name.value,
            `> : `,
            type_of(Self._mlir_origin),
        ]
    ]()

    comptime subtree = Origin[
        _mlir_origin=__mlir_attr[
            `#lit.origin.subtree<`,
            Self._mlir_origin,
            `> : `,
            type_of(Self._mlir_origin),
        ]
    ]()


trait Iterable:
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        ...


# Implement the 'Some' and 'SomeTypeList' helper.
comptime __SomeImpl[Trait: type_of(AnyType), T: Trait] = T
comptime Some[Trait: type_of(AnyType)] = __SomeImpl[Trait, ...]

comptime __SomeTypeListImpl[
    Trait: type_of(AnyType), values: _MLIR.KGENParamListType[Trait]
] = TypeList[Trait=Trait, values]()
comptime SomeTypeList[Trait: type_of(AnyType)] = __SomeTypeListImpl[Trait, ...]


# ===----------------------------------------------------------------------=== #
# Builtin Types
# ===----------------------------------------------------------------------=== #


comptime KeyElement = Copyable  # & Hashable & Equatable


# This trait is used to test types that auto-convert to Error.
trait ErrorConversionTrait:
    pass


struct Error(Copyable):
    def __init__(out self):
        pass

    @implicit
    def __init__(out self, value: StringLiteral):
        pass

    @implicit
    def __init__(out self, value: Some[ErrorConversionTrait]):
        pass

    def __deinit__(deinit self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # A method for testing.
    def use(self):
        pass


struct NoneType(TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type.`!kgen.none`
    """Raw MLIR type of the `None` value."""

    var _value: Self._mlir_type

    # FIXME: Fix representation of None literal to remove this.
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: __mlir_type.`!kgen.none`):
        self._value = value


@stable
@__nonmaterializable(Int)
struct IntLiteral[value: __mlir_type.`!pop.int_literal`](
    TrivialRegisterPassable
):
    comptime _zero = IntLiteral[
        __mlir_attr.`#pop.int_literal<0> : !pop.int_literal`
    ]()
    comptime _one = IntLiteral[
        __mlir_attr.`#pop.int_literal<1> : !pop.int_literal`
    ]()

    @stable
    @always_inline("builtin")
    def __init__(out self):
        """Constructor for any value."""
        pass

    @always_inline("builtin")
    def __ne__(self, rhs: IntLiteral[_]) -> Bool:
        return __mlir_attr[
            `#pop<int_literal_cmp<ne `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __le__(self, rhs: IntLiteral[_]) -> Bool:
        return __mlir_attr[
            `#pop<int_literal_cmp<le `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        return self != Self._zero

    @always_inline("builtin")
    def __mul__(
        self,
        rhs: IntLiteral[_],
        out result: IntLiteral[
            __mlir_attr[
                `#pop<int_literal_bin<mul `,
                self.value,
                `,`,
                rhs.value,
                `>> : !pop.int_literal`,
            ]
        ],
    ):
        result = type_of(result)()

    @always_inline("builtin")
    def __sub__(
        self,
        rhs: IntLiteral[_],
        out result: IntLiteral[
            __mlir_attr[
                `#pop<int_literal_bin<sub `,
                self.value,
                `,`,
                rhs.value,
                `>> : !pop.int_literal`,
            ]
        ],
    ):
        result = type_of(result)()

    @always_inline("builtin")
    def __neg__(self) -> type_of(0 - self):
        return 0 - self

    @always_inline("builtin")
    def __pos__(self) -> Self:
        return self

    @always_inline("builtin")
    def __pow__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<pow `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        return {}

    @always_inline("builtin")
    def __floordiv__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<floordiv `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        return {}

    @always_inline("builtin")
    def __xor__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<xor `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        return {}

    @always_inline("builtin")
    def __lshift__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<lshift `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        return {}


@__nonmaterializable(FloatDyn)
struct FloatLiteral[value: __mlir_type.`!pop.float_literal`](
    TrivialRegisterPassable
):
    @always_inline("builtin")
    def __init__(out self):
        pass

    @always_inline("builtin")
    @implicit
    def __init__(
        val: IntLiteral[_],
        out result: FloatLiteral[
            __mlir_attr[
                `#pop<int_to_float_literal<`,
                val.value,
                `>> : !pop.float_literal`,
            ]
        ],
    ):
        result = type_of(result)()

    @always_inline("builtin")
    def __neg__(self, out result: type_of(self * -1)):
        result = type_of(result)()

    @always_inline("builtin")
    def __mul__(
        self,
        rhs: FloatLiteral,
        out result: FloatLiteral[
            __mlir_attr[
                `#pop<float_literal_bin<mul `,
                Self.value,
                `,`,
                rhs.value,
                `>> : !pop.float_literal`,
            ]
        ],
    ):
        result = type_of(result)()

    @always_inline("builtin")
    def __truediv__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<truediv `,
            Self.value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        return {}


struct FloatDyn(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.`!kgen.scalar<f64>`

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: __mlir_type.`!kgen.scalar<f64>`):
        self._mlir_value = value

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: FloatLiteral):
        self = __mlir_attr[
            `#pop<float_literal_convert<`,
            +value.value,
            `>> : !kgen.scalar<f64>`,
        ]

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: IntLiteral):
        self = FloatLiteral(value)


# NOTE: `Int` is now defined as an alias for `Scalar[DType.int]` (see the
# `Scalar`/`Int` aliases near the `SIMD` definition below), mirroring the
# unification of `Int` with `SIMD` in the real standard library. The operators
# and constructors that used to live on the standalone `struct Int` now live on
# `SIMD` so that `Int` (i.e. `SIMD[DType.int, 1]`) keeps behaving the same.


struct UInt8(TrivialRegisterPassable):
    def __init__(out self):
        pass

    @implicit
    def __init__(out self, value: IntLiteral):
        pass


comptime Byte = UInt8


struct Span[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
](TrivialRegisterPassable):
    # Field
    var _data: Pointer[Self.T, Self.origin]
    var _len: Int

    def __init__(out self):
        self._data = Pointer[Self.T, Self.origin].unsafe_dangling()
        self._len = 0

    def __init__(
        out self,
        *,
        unsafe_ptr: Pointer[Self.T, Self.origin],
        length: Int,
    ):
        self._data = unsafe_ptr
        self._len = length

    @always_inline
    @implicit
    def __init__(
        out self: Span[Self.T, Self.origin],
        ref[Self.origin] array: Array[downcast[Self.T, Movable], _],
    ):
        pass

    @always_inline
    @implicit
    def __init__[U: Copyable](out self, ref[Self.origin] list: List[U]):
        self = {}  # Stub impl.

    def unsafe_ptr(
        self,
    ) -> Pointer[Self.T, Self.origin]:
        return self._data

    @always_inline
    def __getitem__(self, idx: Int) -> ref[Self.origin] Self.T:
        return self._data[normalized_idx]


@stable
@__nonmaterializable(String)
struct StringLiteral[value: __mlir_type.`!kgen.string`](
    TrivialRegisterPassable
):
    @stable
    @always_inline("builtin")
    def __init__(out self):
        pass

    @always_inline("nodebug")
    def __eq__(self, other: StringLiteral) -> Bool:
        return Bool()

    # TODO(MSTDL-1327): Reduce pain when string literals can't be
    # nonmaterializable by making them merge into StaticString.  They should
    # eventually merge into String through nonmaterialization.
    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(StringLiteral[_]),
    ](self) -> StaticString:
        return self

    @always_inline("nodebug")
    def __add__(
        self, rhs: StringLiteral
    ) -> StringLiteral[
        __mlir_attr[
            `#pop.string_concat<`,
            self.value,
            `,`,
            rhs.value,
            `> : !kgen.string`,
        ]
    ]:
        return {}


comptime StringSlice = StringSpan


struct StringSpan[mut: Bool, //, origin: Origin[mut=mut]](
    TrivialRegisterPassable
):
    var _slice: Span[Byte, Self.origin]

    @implicit
    def __init__[](out self, ref[Self.origin] value: String):
        self._slice = {}

    @implicit
    def __init__(out self: StaticString, lit: StringLiteral):
        self._slice = {}

    @always_inline
    def __init__(out self: StaticString, _kgen: __mlir_type.`!kgen.string`):
        pass

    @always_inline
    def unsafe_ptr(
        self,
    ) -> Pointer[Byte, origin]:
        return self._slice.unsafe_ptr()

    @always_inline
    def byte_length(self) -> Int:
        return self._slice._len


comptime StaticString = StringSlice[ImmStaticOrigin]


@always_inline("builtin")
def _get_kgen_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.values,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
def get_static_string[
    string: StaticString, *extra: StaticString
]() -> StaticString:
    return StaticString(_get_kgen_string[string, *extra]())


trait Stringable:
    def __str__(self) -> String:
        ...


struct String(ErrorConversionTrait, ImplicitlyCopyable, KeyElement):
    def __init__(out self):
        pass

    @implicit
    def __init__(out self, literal: StringLiteral):
        pass

    def __init__[T: Stringable](out self, value: T):
        self = value.__str__()

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: String):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def __deinit__(deinit self):
        pass

    def __len__(self) -> Int:
        return 0

    def __contains__(self, substr: StringSlice[mut=False, ...]) -> Bool:
        return True

    def __add__(self, other: StringSlice) -> String:
        pass

    def __iadd__(mut self, rhs: StringSlice[mut=False, ...]):
        pass

    def byte_length(self) -> Int:
        return 0

    def unsafe_ptr(
        self,
    ) -> Pointer[UInt8, origin_of(self)]:
        return {}


@stable
struct Bool(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.`!kgen.scalar<bool>`

    @stable
    @always_inline("builtin")
    def __init__(out self):
        self._mlir_value = __mlir_attr.`#kgen.simd<false> : !kgen.scalar<bool>`

    @stable
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: __mlir_type.i1):
        self._mlir_value = __mlir_op.`pop.cast_from_builtin`[
            _type=__mlir_type.`!kgen.scalar<bool>`
        ](value)

    @implicit
    @always_inline("builtin")
    def __init__(out self, mlir_value: __mlir_type.`!kgen.scalar<bool>`):
        """Construct a Bool value given a `!kgen.scalar<bool>` value.

        Args:
            mlir_value: The initial value.
        """
        self._mlir_value = mlir_value

    @always_inline("builtin")
    def __mlir_bool__(self) -> __mlir_type.`!kgen.scalar<bool>`:
        return self._mlir_value

    @always_inline("builtin")
    def __mlir_i1__(self) -> __mlir_type.i1:
        return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i1](
            self._mlir_value
        )

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        return self

    @always_inline("builtin")
    def __invert__(self) -> Bool:
        return __mlir_op.`pop.simd.xor`(
            self._mlir_value,
            __mlir_attr.`#kgen.simd<true> : !kgen.scalar<bool>`,
        )

    @always_inline("builtin")
    def __and__(self, rhs: Bool) -> Bool:
        return __mlir_op.`pop.simd.and`(self._mlir_value, rhs._mlir_value)

    @always_inline("nodebug")
    def __eq__(self, rhs: Bool) -> Bool:
        return True


struct Slice(TrivialRegisterPassable):
    @implicit
    def __init__(out self, end: Int):
        pass

    def __init__(out self, start: Int, end: Int):
        return

    def __init__[
        T0: TrivialRegisterPassable,
        T1: TrivialRegisterPassable,
        T2: TrivialRegisterPassable,
    ](
        out self,
        start: T0,
        end: T1,
        step: T2,
        __slice_literal__: NoneType = None,
    ):
        pass


@fieldwise_init
struct _ListIter[
    mut: Bool,
    //,
    T: Copyable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    comptime Element = Self.T  # FIXME(MOCO-2068): shouldn't be needed.

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int
    var src: Pointer[List[Self.Element], Self.origin]

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        abort()


struct List[T: Copyable](Copyable, Iterable):
    def __init__(out self):
        pass

    def __init__(
        out self, var *elements: Self.T, __list_literal__: NoneType = None
    ):
        pass

    def append(
        mut self, var value: Self.T
    ) where conforms_to(Self.T, Deinitable):
        pass

    def __getitem__(
        ref self, idx: Int
    ) -> ref[origin_of(self)._get_owned_interior["element"]] Self.T:
        abort()

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _ListIter[
        Self.T, iterable_origin._get_owned_interior["element"], True
    ]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        abort()

    def __len__(self) -> Int:
        return 0


@fieldwise_init
struct _ArrayIter[
    mut: Bool,
    //,
    T: Copyable,
    length: Int,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    comptime Element = Self.T  # FIXME(MOCO-2068): shouldn't be needed.

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int
    var src: Pointer[Array[Self.Element, Self.length], Self.origin]

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        abort()


struct Array[T: AnyType, length: Int](Copyable, Iterable):
    def __init__[
        *, __literal_size__: Int
    ](
        out self: Array[Self.T, __literal_size__],
        var *elements: Self.T,
        __list_literal__: NoneType,
    ):
        pass

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _ArrayIter[
        downcast[Self.T, Copyable], Self.length, iterable_origin, True
    ]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        abort()

    def __len__(self) -> Int:
        return 0


struct Set[T: AnyType]:
    def __init__(out self, *elements: Self.T, __set_literal__: NoneType = None):
        pass

    def add(mut self, var value: Self.T):
        pass


struct Dict[K: Copyable & Deinitable, V: Copyable & Deinitable]:
    def __init__(out self):
        pass

    def __init__(
        out self,
        var keys: List[Self.K],
        var values: List[Self.V],
        __dict_literal__: NoneType,
    ):
        pass

    def __setitem__(mut self, key: Self.K, value: Self.V):
        pass


# ===----------------------------------------------------------------------=== #
# Value Stubs
# ===----------------------------------------------------------------------=== #


@stable
trait AnyType:
    pass


@stable
trait Copyable(Movable):
    @stable
    def __init__(out self, *, copy: Self):
        ...

    @always_inline
    def copy(self) -> Self:
        return Self(copy=self)

    comptime __copy_ctor_is_trivial: Bool


trait ImplicitlyCopyable(Copyable):
    pass


trait TrivialRegisterPassable(
    Deinitable, ImplicitlyCopyable, Movable, RegisterPassable
):
    pass


trait RegisterPassable(Movable):
    pass


trait Defaultable:
    def __init__(out self):
        ...


def materialize[T: AnyType, //, value: T](out result: T):
    """Explicitly materialize a compile time parameter into a runtime value."""
    __mlir_op.`lit.materialize_into`[value=value](
        __get_mvalue_as_litref(result)
    )


@stable
trait Movable:
    def __init__(out self, *, deinit move: Self):
        ...

    comptime __move_ctor_is_trivial: Bool


trait Deinitable:
    def __deinit__(deinit self, /):
        ...

    comptime __del__is_trivial: Bool


# ===----------------------------------------------------------------------=== #
# Pattern matching support
# ===----------------------------------------------------------------------=== #

comptime KGENString = __mlir_type.`!kgen.string`


trait EnumLike:
    comptime _enum_case_length: Int
    comptime _enum_case_names: _MLIR.KGENParamListType[KGENString]
    comptime _enum_case_types: _MLIR.KGENParamListType[AnyType]

    comptime _enum_elt_type_for_case[id: Int]: AnyType = TypeList[
        Self._enum_case_types
    ]()[id]

    def _get_enum_discriminant(self) -> Int:
        ...

    # FIXME: Use an interior origin.
    # Return type uses TypeList directly (not `_enum_elt_type_for_case`) to
    # avoid a recursive alias cycle when specializing this method.
    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        ...


# ===-----------------------------------------------------------------------===#
# ParameterList
# ===-----------------------------------------------------------------------===#


struct ParameterList[type: AnyType, //, values: _MLIR.KGENParamListType[type]](
    TrivialRegisterPassable
):
    comptime size: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
            type_of(Self.values),
            ` `,
            +Self.values,
            `> : index`,
        ]
    )

    comptime of[type: AnyType, //, *values: type] = ParameterList[
        type=type, values.values
    ]

    def __init__(out self):
        pass

    @always_inline
    def __getitem__(self, idx: Int) -> Self.type:
        pass

    comptime __getitem_param__[idx: Int]: Self.type = __mlir_attr[
        `#kgen.param_list.get<:`,
        type_of(Self.values),
        ` `,
        +Self.values,
        `, `,
        idx.__mlir_index__(),
        `> : `,
        +Self.type,
    ]

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

    comptime splat[T: AnyType, //, count: Int, value: T] = ParameterList[
        T,
        __mlir_attr[
            `#kgen.param_list.tabulate<`,
            count.__mlir_index__(),
            `,`,
            Self._IndexToIntTabulateWrap[_SplatValueTabulator[value, _], ...],
            `> : `,
            _MLIR.KGENParamListType[T],
        ],
    ]


# ===-----------------------------------------------------------------------===#
# TypeList
# ===-----------------------------------------------------------------------===#


struct TypeList[
    Trait: type_of(AnyType), //, values: _MLIR.KGENParamListType[Trait]
](TrivialRegisterPassable):
    comptime length: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
            type_of(Self.values),
            ` `,
            +Self.values,
            `> : index`,
        ]
    )

    comptime __getitem_param__[idx: Int]: Self.Trait = __mlir_attr[
        `#kgen.param_list.get<:`,
        type_of(Self.values),
        ` `,
        +Self.values,
        `, `,
        idx.__mlir_index__(),
        `> : `,
        +Self.Trait,
    ]

    comptime _DiscardIndexWrapper[
        FromAndTo: AnyType,
        ToWrap: __generator_type[Prev: FromAndTo, From: Self.Trait] FromAndTo,
        PrevV: FromAndTo,
        VA: _MLIR.KGENParamListType[Self.Trait],
        idx: Int,
    ] = ToWrap[PrevV, TypeList[VA].__getitem_param__[idx]]

    comptime _IndexToIntWrap[
        ReduceT: AnyType,
        ToWrap: __generator_type[
            Prev: ReduceT, From: _MLIR.KGENParamListType[Self.Trait], Idx: Int
        ] ReduceT,
        PrevV: ReduceT,
        VA: _MLIR.KGENParamListType[Self.Trait],
        idx: __mlir_type.index,
    ] = ToWrap[PrevV, VA, Int(mlir_value=idx)]

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

    @always_inline("builtin")
    def __init__(out self):
        pass

    comptime of[Trait: type_of(AnyType), //, *values: Trait] = TypeList[
        Trait=Trait, values.values
    ]

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
        pass

    comptime _ContainsTypePredicate[
        search: AnyType,
        element: Self.Trait,
    ] = element == search

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
        return Self.reduce[
            False,
            Self._AnySatisfiesReducer[predicate, ...],
        ]

    @always_inline("builtin")
    @staticmethod
    def contains[type: AnyType]() -> Bool:
        return Self.any[Self._ContainsTypePredicate[type, ...],]()

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
            _IndexToIntTypeTabulateWrap[Trait=Trait, ToT=ToT, Mapper, ...],
            `> : `,
            _MLIR.KGENParamListType[Trait],
        ],
    ]

    comptime _SplatTypeTabulator[
        Trait: type_of(AnyType), T: Trait, index: Int
    ]: Trait = T

    comptime splat[
        Trait: type_of(AnyType), //, count: Int, type: Trait
    ] = TypeList.tabulate[
        Trait=Trait, ToT=type, count, Self._SplatTypeTabulator[Trait, type, _]
    ]

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


comptime _IndexToIntTypeTabulateWrap[
    Trait: type_of(AnyType),
    ToT: Trait,
    //,
    ToWrap: __generator_type[Idx: Int] Trait,
    idx: __mlir_type.index,
] = ToWrap[Int(mlir_value=idx)]


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
    """

    comptime variadic_list_type = VariadicList[
        origin=Self.elt_origin,
        Self.elt_type,
        Self.is_owned,
    ]

    var index: Int
    var src: Pointer[Self.variadic_list_type, Self.list_origin]

    def __init__(
        out self,
        index: Int,
        ref[Self.list_origin] list: Self.variadic_list_type,
    ):
        self.index = index
        self.src = Pointer(to=list)

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.elt_origin] Self.elt_type:
        raise StopIteration()


struct VariadicList[
    elt_is_mutable: Bool,
    # NOTE: origin._mlir_origin is here in the param list.
    origin: Origin[mut=elt_is_mutable],
    //,
    element_type: AnyType,
    is_owned: Bool,
](RegisterPassable):
    comptime _EltPointerType = Pointer[Self.element_type, Self.origin]
    # FIXME: This should be the origin of the container, not UntrackedOrigin.
    var value: Span[Self._EltPointerType, ImmUntrackedOrigin]

    # TODO: the origin of the vardecl is captured in the Self.origin set
    # by the compiler to make sure the container outlives all its elements.
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
        # Convert the !lit.ref to a Pointer, then cast to a pointer to
        # the first element.
        var array_up = Pointer(to=Pointer(value)[])
        var elt_ptr = Pointer[_, ImmUntrackedOrigin](
            __mlir_op.`pop.array.gep`(
                array_up._get_kgen_pointer(),
                Int(0).__mlir_index__(),
            )
        ).unsafe_bitcast[Self._EltPointerType]()
        var size_tmp = size  # FIXME: Weird MLIR syntax error?
        self.value = Span(unsafe_ptr=elt_ptr, length=Int(mlir_value=size_tmp))

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
        while True:
            pass

    def __iter__[
        self_origin: Origin[]
    ](ref[self_origin] self) -> _VariadicListIter[
        Self.element_type, Self.origin, self_origin, Self.is_owned
    ]:
        return {0, self}


struct VariadicPack[
    elt_is_mutable: Bool,
    origin: Origin[mut=elt_is_mutable],
    element_trait: type_of(AnyType),
    //,
    is_owned: Bool,
    *element_types: element_trait,
](RegisterPassable):
    comptime _mlir_pack_type = __mlir_type[
        `!lit.ref.pack<:`,
        type_of(Self.element_types.values),
        ` `,
        Self.element_types.values,
        `, `,
        Self.origin._mlir_origin,
        `>`,
    ]
    var _value: Self._mlir_pack_type

    # This disables nested origin exclusivity checking because it is taking a
    # raw variadic pack which can have nested origins in it (which this does not
    # dereference).
    @__unsafe_nested_origins_read_only
    def __init__(out self, value: Self._mlir_pack_type):
        self._value = value

    def __getitem_param__[
        index: Int
    ](self) -> ref[Self.origin] Self.element_types[index]:
        while True:
            pass


struct AddressSpace(TrivialRegisterPassable):
    """Address space of the pointer."""

    # Stored as `SIMDLength` (a raw `index` wrapper) so it folds to a constant
    # `index` when spliced into pointer/ref MLIR types.
    var _value: SIMDLength

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: SIMDLength):
        self._value = value

    # CPU address space
    comptime GENERIC = AddressSpace(0)

    # GPU address spaces
    comptime GLOBAL = AddressSpace(1)
    comptime SHARED = AddressSpace(3)
    comptime CONSTANT = AddressSpace(4)
    comptime LOCAL = AddressSpace(5)
    comptime SHARED_CLUSTER = AddressSpace(7)

    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        return self._value.__mlir_index__()


struct Pointer[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = .GENERIC,
](Defaultable, TrivialRegisterPassable):
    comptime type = Self.T

    comptime _mlir_lit_ref = __mlir_type[
        `!lit.ref<`,
        Self.T,
        `, `,
        Self.origin._mlir_origin,
        `, `,
        Self.address_space._value.__mlir_index__(),
        `>`,
    ]

    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.T,
        `,`,
        Self.address_space._value.__mlir_index__(),
        `>`,
    ]
    var _mlir_value: Self._mlir_type

    def _get_kgen_pointer(self) -> Self._mlir_type:
        return self._mlir_value

    def __init__(out self):
        self._mlir_value = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @implicit
    @always_inline("builtin")
    def __init__(out self, value: Self._mlir_type):
        self._mlir_value = value

    @always_inline("nodebug")
    @implicit
    def __init__(out self, _mlir_value: Self._mlir_lit_ref):
        self = Self(__mlir_op.`lit.ref.to_pointer`(_mlir_value))

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        ref[Self.origin, Self.address_space._value.__mlir_index__()] to: Self.T,
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

    @implicit
    @always_inline("nodebug")
    def __init__(
        out self, other: Pointer
    ) where Self.origin.contains[other.origin]:
        self._mlir_value = rebind[Self._mlir_type](other._mlir_value)

    @staticmethod
    def address_of(ref[Self.address_space] arg: Self.T) -> Self:
        return Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(arg)))

    @staticmethod
    def unsafe_dangling() -> Self:
        return Self()

    @__unsafe_nested_origins_read_only
    def __getitem__(self) -> ref[Self.origin, Self.address_space] Self.T:
        while True:
            pass

    @__unsafe_nested_origins_read_only
    def __getitem__(
        self, offset: Int
    ) -> ref[Self.origin, Self.address_space] Self.T:
        while True:
            pass

    def store(self, offset: Int, value: Self.T):
        pass

    # Returns a reference to the pointee but with the origin rebased to be a
    # interior origin derived from the specified base origin. This is used by
    # collections that need to vend owned interior references.
    @always_inline
    def _get_ref_with_unsafe_interior_origin[
        name: StringLiteral,
    ](self, ref base: Some[AnyType]) -> ref[
        origin_of(base)._get_owned_interior[name], Self.address_space
    ] Self.T:
        comptime res_origin = origin_of(base)._get_owned_interior[name]
        comptime ptr_type = Pointer[
            Self.T, res_origin, address_space=Self.address_space
        ]
        # Do this delicately since we're manufacturing an interior origin here.
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type=ptr_type._mlir_lit_ref](
                self._mlir_value
            )
        )

    @always_inline
    def unsafe_take_pointee[
        U: Movable, //
    ](self: Pointer[U, _]) -> U where type_of(self).mut:
        return __get_address_as_owned_value(self._mlir_value)

    @always_inline("builtin")
    def unsafe_bitcast[
        U: AnyType
    ](self) -> Pointer[U, Self.origin, address_space=Self.address_space]:
        return __mlir_op.`pop.pointer.bitcast`[
            _type=Pointer[
                U, Self.origin, address_space=Self.address_space
            ]._mlir_type,
        ](self._mlir_value)

    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = Pointer[Self.T, target_origin, address_space=Self.address_space]

    @always_inline("builtin")
    def unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        return __mlir_op.`pop.pointer.bitcast`[
            _type=Self._OriginCastType[
                Self.origin.unsafe_mut_cast[target_mut]()
            ]._mlir_type,
        ](self._mlir_value)

    @always_inline("builtin")
    def unsafe_origin_cast[
        target_origin: Origin[mut=Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        return __mlir_op.`pop.pointer.bitcast`[
            _type=Self._OriginCastType[target_origin]._mlir_type,
        ](self._mlir_value)

    def as_imm(
        self,
    ) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[False]()]:
        return self.unsafe_mut_cast[False]()

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __eq__(
        self, rhs: Pointer[Self.T, _, address_space=Self.address_space]
    ) -> Bool:
        return True

    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(
            Pointer[Self.T, _, address_space=Self.address_space]
        ),
    ](self) -> Pointer[
        mut=Self.mut & other_type.origin.mut,
        T=Self.T,
        origin=origin_of(Self.origin, other_type.origin),
        address_space=Self.address_space,
    ]:
        return {self._mlir_value}  # allow kgen.pointer to convert.


comptime UnsafePointer[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = .GENERIC,
] = Pointer[T, origin, address_space=address_space]


struct Tuple[*element_types: Movable](ImplicitlyCopyable):
    comptime _mlir_type = __mlir_type[
        `!kgen.struct<:`,
        _MLIR.KGENParamListType[Movable],
        Self.element_types.values,
        ` isParamPack>`,
    ]
    var _mlir_value: Self._mlir_type

    def __init__(out self: Tuple[]):
        pass

    @implicit
    def __init__(out self, *args: *Self.element_types):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    def __getitem_param__[i: Int](ref self) -> ref[self] Self.element_types[i]:
        while True:
            pass


comptime MutOpaquePointer[
    origin: Origin[mut=True],
    *,
    address_space: AddressSpace = .GENERIC,
] = Pointer[NoneType, origin, address_space=address_space]


struct _StridedRangeIterator(Iterator, TrivialRegisterPassable):
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    def __len__(self) -> Int:
        if self.step > 0 and self.start < self.end:
            return self.end - self.start
        elif self.step < 0 and self.start > self.end:
            return self.start - self.end
        else:
            return 0

    @always_inline
    def __next__(mut self) raises StopIteration -> Int:
        if self.__len__() <= 0:
            raise StopIteration()
        var result = self.start
        self.start += self.step
        return result


# ===-----------------------------------------------------------------------===#
# parameter_for
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct StopIteration(TrivialRegisterPassable):
    pass


@fieldwise_init
struct ListLengthError(ErrorConversionTrait, TrivialRegisterPassable):
    pass


def check_list_length(actual: Int, expected: Int) raises ListLengthError:
    if actual != expected:
        raise ListLengthError()


trait Iterator(Deinitable, Movable):
    comptime Element: Movable

    def __next__(mut self) raises StopIteration -> Self.Element:
        ...


def paramfor_has_next[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> Bool where conforms_to(
    IteratorType.Element,
    Movable & Deinitable,
):
    var result = it.copy()
    try:
        var elem = result.__next__()
        _ = elem^
        return True
    except:
        return False


def paramfor_next_iter[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> IteratorType where conforms_to(
    IteratorType.Element,
    Movable & Deinitable,
):
    # NOTE: This function is called by the compiler's elaborator only when
    # paramfor_has_next will return true. This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it.copy()
    # This intentionally discards the value, but this only happens at comptime,
    # so recomputing it in the body of the loop is fine.
    try:
        var elem = result.__next__()
        _ = elem^
    except:
        abort()
    return result^


def paramfor_next_value[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> IteratorType.Element:
    # NOTE: This function is called by the compiler's elaborator only when
    # paramfor_has_next will return true. This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it.copy()
    try:
        return result.__next__()
    except:
        abort()


struct Optional[T: Movable](Copyable, EnumLike):
    def __deinit__(deinit self):
        pass

    def __init__(out self):
        pass

    @implicit
    def __init__(out self, var value: Self.T):
        # This isn't correct impl, but silences an error.
        __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(value))

    @implicit
    def __init__(out self, value: NoneType):
        pass

    # FIXME: None literal should be of NoneType not !kgen.none.
    @implicit
    def __init__(out self, x: __mlir_type.`!kgen.none`):
        pass

    def value(ref self) -> ref[self] Self.T:
        while True:
            pass

    # Enable pattern matching on Optional.
    comptime _enum_case_length = 2
    comptime _enum_case_names = ParameterList.of[
        "None".value, "Some".value
    ].values
    comptime _enum_case_types: _MLIR.KGENParamListType[AnyType] = TypeList.of[
        NoneType, Self.T
    ].values

    def _get_enum_discriminant(self) -> Int:
        return 0

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        comptime assert id != 0, "cannot get payload for None case"
        comptime elt_type = TypeList[Self._enum_case_types]()[id]
        return rebind[elt_type](self.value())


# ===-----------------------------------------------------------------------===#
# rebind
# ===-----------------------------------------------------------------------===#


@always_inline("builtin")
def rebind[
    src_type: TrivialRegisterPassable,
    //,
    dest_type: TrivialRegisterPassable,
](src: src_type) -> dest_type:
    return __mlir_op.`kgen.rebind`[_type=dest_type](src)


@always_inline("nodebug")
def rebind[
    src_type: AnyType,
    //,
    dest_type: AnyType,
](ref src: src_type) -> ref[src] dest_type:
    var lit = __get_mvalue_as_litref(src)
    var rebound = rebind[Pointer[dest_type, origin_of(src)]._mlir_lit_ref](lit)
    return __get_litref_as_mvalue(rebound)


def rebind_var[
    src_type: Movable,
    //,
    dest_type: Movable,
](var src: src_type, out dest: dest_type):
    ref dest_ref = rebind[dest_type](src)
    dest = Pointer(to=dest_ref).unsafe_take_pointee()
    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(src))


# ===-----------------------------------------------------------------------===#
# trait downcast
# ===-----------------------------------------------------------------------===#

comptime downcast[T: AnyType, _Trait: type_of(AnyType)] = __mlir_attr[
    `#kgen.downcast<:`, type_of(T), +T, `> : `, _Trait
]


@always_inline
def trait_downcast[
    T: TrivialRegisterPassable, //, Trait: type_of(AnyType)
](var src: T) -> downcast[T, Trait]:
    return rebind[downcast[T, Trait]](src)


@always_inline
def trait_downcast[
    T: AnyType, //, Trait: type_of(AnyType)
](ref x: T) -> ref[x] downcast[T, Trait]:
    return rebind[downcast[T, Trait]](x)


# ===----------------------------------------------------------------------=== #
#  Intable
# ===----------------------------------------------------------------------=== #


trait Intable(Deinitable):
    def __int__(self) -> Int:
        ...


# ===----------------------------------------------------------------------=== #
#  DType
# ===----------------------------------------------------------------------=== #


struct DType(TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type.`!kgen.dtype`
    var _mlir_value: Self._mlir_type

    comptime int = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`
    )
    comptime float32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`
    )
    comptime float64 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`
    )
    comptime int32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`
    )
    comptime uint32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`
    )

    @always_inline("builtin")
    @implicit
    def __init__(out self, mlir_value: Self._mlir_type):
        self._mlir_value = mlir_value

    @always_inline("builtin")
    def is_floating_point(self) -> Bool:
        return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<ne>`](
            __mlir_op.`pop.simd.and`(
                __mlir_op.`pop.cast_from_builtin`[
                    _type=__mlir_type.`!kgen.scalar<ui8>`
                ](__mlir_op.`pop.dtype.to_ui8`(self._mlir_value)),
                __mlir_attr.`#kgen.simd<64> : !kgen.scalar<ui8>`,
            ),
            __mlir_attr.`#kgen.simd<0> : !kgen.scalar<ui8>`,
        )


# ===----------------------------------------------------------------------=== #
#  SIMDLength
# ===----------------------------------------------------------------------=== #


# `SIMDLength` wraps the MLIR `index` type and is used as the `size` parameter of
# `SIMD`. It exists to break the circular dependency between `SIMD` and `Int`
# (which is now an alias for `SIMD[DType.int, 1]`).
@stable
struct SIMDLength(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.index

    @always_inline("builtin")
    def __init__(out self, *, mlir_value: __mlir_type.index):
        self._mlir_value = mlir_value

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: IntLiteral[_]):
        self._mlir_value = __mlir_attr[
            `#kgen.cast_to_builtin<#pop.int_literal_convert<`,
            +value.value,
            `> : !kgen.scalar<index>> : index`,
        ]

    # Implicit construction from `Int` (i.e. `Scalar[DType.int]`). This is what
    # lets `SIMD[dtype, some_int]` bind an `Int` width to the `size` parameter.
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: Scalar[DType.int], /):
        self._mlir_value = __mlir_op.`pop.cast_to_builtin`[
            _type=__mlir_type.index
        ](value._mlir_value)

    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        return self._mlir_value

    @always_inline("builtin")
    def __mul__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`index.mul`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __add__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`index.add`(self._mlir_value, rhs._mlir_value)
        )


comptime Float32 = SIMD[.float32, 1]
comptime Float64 = SIMD[.float64, 1]
comptime Float4_e2m1fn = SIMD[.float4_e2m1fn, 1]
comptime Int32 = SIMD[.int32, 1]
comptime UInt32 = SIMD[.uint32, 1]

# ===----------------------------------------------------------------------=== #
#  SIMD
# ===----------------------------------------------------------------------=== #


@stable
struct SIMD[dtype: DType, size: SIMDLength](
    Intable, Stringable, TrivialRegisterPassable
):
    comptime _mlir_type = __mlir_type[
        `!kgen.simd<`, Self.size._mlir_value, `, `, Self.dtype._mlir_value, `>`
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the vector."""

    @always_inline("builtin")
    def __init__(out self, *, mlir_value: Self._mlir_type):
        self._mlir_value = mlir_value

    # `Int`-specific constructor from a raw MLIR `index` value.
    @always_inline("builtin")
    def __init__(out self: Int, *, mlir_value: __mlir_type.index):
        self._mlir_value = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[.int, 1]._mlir_type
        ](mlir_value)

    @stable
    @always_inline("nodebug")
    def __init__(out self):
        self = Self(0)

    @stable
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: IntLiteral[_], /):
        self._mlir_value = __mlir_attr[
            `#pop.int_literal_convert<`, +value.value, `> : `, Self._mlir_type
        ]

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: Int, /):
        var index = __mlir_op.`pop.cast_from_builtin`[
            _type=__mlir_type.`!kgen.scalar<index>`
        ](value.__mlir_index__())
        var s = __mlir_op.`pop.cast`[_type=SIMD[Self.dtype, 1]._mlir_type](
            index
        )

        self._mlir_value = __mlir_op.`pop.simd.splat`[_type=Self._mlir_type](s)

    @implicit
    def __init__(out self, value: FloatLiteral, /):
        comptime assert (
            Self.dtype.is_floating_point()
        ), "the SIMD type must be floating point"
        var res = __mlir_attr[
            `#pop<float_literal_convert<`, value.value, `>> : `, Self._mlir_type
        ]
        self = Self(mlir_value=res)

    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.index](
            __mlir_op.`pop.cast`[
                _type=SIMD[.int, 1]._mlir_type,
                fastmathFlags=__mlir_attr.`#pop.fmf<fast>`,
            ](rebind[SIMD[Self.dtype, SIMDLength(1)]](self)._mlir_value)
        )

    @always_inline("builtin")
    def __int__(self) -> Int:
        return Int(mlir_value=self.__mlir_index__())

    def __str__(self) -> String:
        return "[unimplemented]"

    @staticmethod
    def splat():
        pass

    # ===------------------------------------------------------------------=== #
    # Arithmetic operators.
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    def __add__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.add`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __sub__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.sub`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __mul__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.mul`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    def __truediv__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.div`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    def __rtruediv__(self, value: Self) -> Self:
        return value / self

    @always_inline("nodebug")
    def __floordiv__(self, rhs: Self) -> Self:
        pass

    @always_inline("nodebug")
    def __mod__(self, rhs: Self) -> Self:
        pass

    @always_inline("nodebug")
    def __pow__(self, exp: Self) -> Self:
        pass

    @always_inline("builtin")
    def __neg__(self) -> Self:
        return Self(mlir_value=__mlir_op.`pop.neg`(self._mlir_value))

    # ===------------------------------------------------------------------=== #
    # Bitwise and shift operators.
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    def __and__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.simd.and`(
                self._mlir_value, rhs._mlir_value
            )
        )

    @always_inline("builtin")
    def __or__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.simd.or`(
                self._mlir_value, rhs._mlir_value
            )
        )

    @always_inline("builtin")
    def __xor__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.simd.xor`(
                self._mlir_value, rhs._mlir_value
            )
        )

    @always_inline("builtin")
    def __lshift__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.shl`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __rshift__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.shr`(self._mlir_value, rhs._mlir_value)
        )

    # ===------------------------------------------------------------------=== #
    # Comparison operators (scalar semantics; used by `Int`).
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    def __eq__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<eq>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("builtin")
    def __ne__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<ne>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("builtin")
    def __lt__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<slt>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("builtin")
    def __le__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<sle>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("builtin")
    def __gt__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<sgt>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("builtin")
    def __ge__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<sge>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        return self != Self(0)

    # ===------------------------------------------------------------------=== #
    # In-place operators.
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def __iadd__(mut self, rhs: Self):
        self = self + rhs

    @always_inline("nodebug")
    def __isub__(mut self, rhs: Self):
        self = self - rhs

    @always_inline("nodebug")
    def __imul__(mut self, rhs: Self):
        self = self * rhs

    @always_inline("nodebug")
    def __ifloordiv__(mut self, rhs: Self):
        self = self // rhs

    @always_inline("nodebug")
    def __imod__(mut self, rhs: Self):
        self = self % rhs

    @always_inline("nodebug")
    def __ipow__(mut self, rhs: Self):
        self = self**rhs

    @always_inline("nodebug")
    def __ilshift__(mut self, rhs: Self):
        self = self << rhs

    @always_inline("nodebug")
    def __irshift__(mut self, rhs: Self):
        self = self >> rhs

    @always_inline("nodebug")
    def __iand__(mut self, rhs: Self):
        self = self & rhs

    @always_inline("nodebug")
    def __ixor__(mut self, rhs: Self):
        self = self ^ rhs

    @always_inline("nodebug")
    def __ior__(mut self, rhs: Self):
        self = self | rhs

    @always_inline("nodebug")
    def join(self, other: Self) -> SIMD[Self.dtype, Self.size * 2]:
        return SIMD[Self.dtype, Self.size * 2]()


# ===----------------------------------------------------------------------=== #
#  Scalar / Int aliases
# ===----------------------------------------------------------------------=== #


@stable
comptime Scalar = SIMD[
    _, size=__mlir_attr[`#lit.struct<{_mlir_value = 1}> : `, SIMDLength]
]
"""Represents a scalar dtype."""


@stable
comptime Int = Scalar[DType.int]
"""Represents a signed integer suitable for indexing."""


@no_inline
def abort() -> Never:
    __mlir_op.`llvm.intr.trap`()
    while True:
        pass
