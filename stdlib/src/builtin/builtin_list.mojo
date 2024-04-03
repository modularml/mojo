# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements the ListLiteral class.

These are Mojo built-ins, so you don't need to import them.
"""

from memory.unsafe import Reference, _LITRef

# ===----------------------------------------------------------------------===#
# ListLiteral
# ===----------------------------------------------------------------------===#


@register_passable
struct ListLiteral[*Ts: AnyRegType](Sized):
    """The type of a literal heterogenous list expression.

    A list consists of zero or more values, separated by commas.

    Parameters:
        Ts: The type of the elements.
    """

    var storage: __mlir_type[`!kgen.pack<`, Ts, `>`]
    """The underlying storage for the list."""

    @always_inline("nodebug")
    fn __init__(inout self, *args: *Ts):
        """Construct the list literal from the given values.

        Args:
            args: The init values.
        """
        self.storage = args

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the list length.

        Returns:
            The length of this ListLiteral.
        """
        return __mlir_op.`pop.variadic.size`(Ts)

    @always_inline("nodebug")
    fn get[i: Int, T: AnyRegType](self) -> T:
        """Get a list element at the given index.

        Parameters:
            i: The element index.
            T: The element type.

        Returns:
            The element at the given index.
        """
        return rebind[T](
            __mlir_op.`kgen.pack.get`[index = i.value](self.storage)
        )


# ===----------------------------------------------------------------------===#
# VariadicList / VariadicListMem
# ===----------------------------------------------------------------------===#


@value
struct _VariadicListIter[type: AnyRegType]:
    """Const Iterator for VariadicList.

    Parameters:
        type: The type of the elements in the list.
    """

    var index: Int
    var src: VariadicList[type]

    fn __next__(inout self) -> type:
        self.index += 1
        return self.src[self.index - 1]

    fn __len__(self) -> Int:
        return len(self.src) - self.index


@register_passable("trivial")
struct VariadicList[type: AnyRegType](Sized):
    """A utility class to access variadic function arguments. Provides a "list"
    view of the function argument so that the size of the argument list and each
    individual argument can be accessed.

    Parameters:
        type: The type of the elements in the list.
    """

    alias storage_type = __mlir_type[`!kgen.variadic<`, type, `>`]
    var value: Self.storage_type
    """The underlying storage for the variadic list."""

    alias IterType = _VariadicListIter[type]

    @always_inline
    fn __init__(inout self, *value: type):
        """Constructs a VariadicList from a variadic list of arguments.

        Args:
            value: The variadic argument list to construct the variadic list
              with.
        """
        self = value

    @always_inline
    fn __init__(inout self, value: Self.storage_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        self.value = value

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """

        return __mlir_op.`pop.variadic.size`(self.value)

    @always_inline
    fn __getitem__(self, index: Int) -> type:
        """Gets a single element on the variadic list.

        Args:
            index: The index of the element to access on the list.

        Returns:
            The element on the list corresponding to the given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.value)

    @always_inline
    fn __iter__(self) -> Self.IterType:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return Self.IterType(0, self)


@value
struct _VariadicListMemIter[
    elt_type: AnyType,
    elt_is_mutable: __mlir_type.i1,
    elt_lifetime: AnyLifetime[elt_is_mutable].type,
    list_lifetime: ImmLifetime,
]:
    """Iterator for VariadicListMem.

    Parameters:
        elt_type: The type of the elements in the list.
        elt_is_mutable: Whether the elements in the list are mutable.
        elt_lifetime: The lifetime of the elements.
        list_lifetime: The lifetime of the VariadicListMem.
    """

    alias variadic_list_type = VariadicListMem[
        elt_type, elt_is_mutable, elt_lifetime
    ]

    var index: Int
    var src: Reference[
        Self.variadic_list_type, __mlir_attr.`0: i1`, list_lifetime
    ]

    fn __next__(inout self) -> Self.variadic_list_type.reference_type:
        self.index += 1
        # TODO: Need to make this return a dereferenced reference, not a
        # reference that must be deref'd by the user.
        return self.src[].__getitem__(self.index - 1)

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


# Helper to compute the union of two lifetimes:
# TODO: parametric aliases would be nice.
struct _lit_lifetime_union[
    is_mutable: __mlir_type.i1,
    a: AnyLifetime[is_mutable].type,
    b: AnyLifetime[is_mutable].type,
]:
    alias result = __mlir_attr[
        `#lit.lifetime.union<`, a, `,`, b, `> : !lit.lifetime<`, is_mutable, `>`
    ]


struct _lit_mut_cast[
    is_mutable: __mlir_type.i1,
    operand: AnyLifetime[is_mutable].type,
    result_mutable: __mlir_type.i1,
]:
    alias result = __mlir_attr[
        `#lit.lifetime.mutcast<`,
        operand,
        `> : !lit.lifetime<`,
        +result_mutable,
        `>`,
    ]


struct VariadicListMem[
    element_type: AnyType,
    elt_is_mutable: __mlir_type.i1,
    lifetime: AnyLifetime[elt_is_mutable].type,
](Sized):
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes references to the elements in a
    way that can be enumerated.  Each element may be accessed with `elt[]`.

    Parameters:
        element_type: The type of the elements in the list.
        elt_is_mutable: True if the elements of the list are mutable for an
                        inout or owned argument.
        lifetime: The reference lifetime of the underlying elements.
    """

    alias reference_type = Reference[element_type, elt_is_mutable, lifetime]
    alias mlir_ref_type = _LITRef[element_type, elt_is_mutable, lifetime].type
    alias storage_type = __mlir_type[
        `!kgen.variadic<`, Self.mlir_ref_type, `, borrow_in_mem>`
    ]

    var value: Self.storage_type
    """The underlying storage, a variadic list of references to elements of the
    given type."""

    # This is true when the elements are 'owned' - these are destroyed when
    # the VariadicListMem is destroyed.
    var _is_owned: Bool

    # Provide support for borrowed variadic arguments.
    @always_inline
    fn __init__(inout self, value: Self.storage_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        self.value = value
        self._is_owned = False

    # Provide support for variadics of *inout* arguments.  The reference will
    # automatically be inferred to be mutable, and the !kgen.variadic will have
    # convention=byref.
    alias inout_storage_type = __mlir_type[
        `!kgen.variadic<`, Self.mlir_ref_type, `, byref>`
    ]

    @always_inline
    fn __init__(inout self, value: Self.inout_storage_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        var tmp = value
        # We need to bitcast different argument conventions to a consistent
        # representation.  This is ugly but effective.
        self.value = Pointer.address_of(tmp).bitcast[Self.storage_type]().load()
        self._is_owned = False

    # Provide support for variadics of *owned* arguments.  The reference will
    # automatically be inferred to be mutable, and the !kgen.variadic will have
    # convention=owned_in_mem.
    alias owned_storage_type = __mlir_type[
        `!kgen.variadic<`, Self.mlir_ref_type, `, owned_in_mem>`
    ]

    @always_inline
    fn __init__(inout self, value: Self.owned_storage_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        var tmp = value
        # We need to bitcast different argument conventions to a consistent
        # representation.  This is ugly but effective.
        self.value = Pointer.address_of(tmp).bitcast[Self.storage_type]().load()
        self._is_owned = True

    @always_inline
    fn __moveinit__(inout self, owned existing: Self):
        """Moves constructor.

        Args:
          existing: The existing VariadicListMem.
        """
        self.value = existing.value
        self._is_owned = existing._is_owned

    @always_inline
    fn __del__(owned self):
        """Destructor that releases elements if owned."""

        # Immutable variadics never own the memory underlying them,
        # microoptimize out a check of _is_owned.
        @parameter
        if not Bool(elt_is_mutable):
            return

        else:
            # If the elements are unowned, just return.
            if not self._is_owned:
                return

            # Otherwise this is a variadic of owned elements, destroy them.  We
            # destroy in backwards order to match how arguments are normally torn
            # down when CheckLifetimes is left to its own devices.
            for i in range(len(self), 0, -1):
                # This cannot use Reference(self[i - 1]) because the subscript
                # will return a BValue, not an LValue.  We need to maintain the
                # parametric mutability by keeping the Reference returned by
                # refitem exposed.
                self.__refitem__(i - 1).destroy_element_unsafe()

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """
        return __mlir_op.`pop.variadic.size`(self.value)

    # TODO: Fix for loops + _VariadicListIter to support a __nextref__ protocol
    # allowing us to get rid of this and make foreach iteration clean.
    @always_inline
    fn __getitem__(self, index: Int) -> Self.reference_type:
        """Gets a single element on the variadic list.

        Args:
            index: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return Self.reference_type(
            __mlir_op.`pop.variadic.get`(self.value, index.value)
        )

    @always_inline
    fn __refitem__(
        self, index: Int
    ) -> Reference[
        element_type,
        elt_is_mutable,
        _lit_lifetime_union[
            elt_is_mutable,
            lifetime,
            # cast mutability of self to match the mutability of the element,
            # since that is what we want to use in the ultimate reference and
            # the union overall doesn't matter.
            _lit_mut_cast[
                __mlir_attr.`0: i1`, __lifetime_of(self), elt_is_mutable
            ].result,
        ].result,
    ]:
        """Gets a single element on the variadic list.

        Args:
            index: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.value)

    fn __iter__(
        self,
    ) -> _VariadicListMemIter[
        element_type, elt_is_mutable, lifetime, __lifetime_of(self)
    ]:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return _VariadicListMemIter[
            element_type, elt_is_mutable, lifetime, __lifetime_of(self)
        ](0, self)


# ===----------------------------------------------------------------------===#
# VariadicPack
# ===----------------------------------------------------------------------===#

alias _AnyTypeMetaType = __mlir_type[`!lit.anytrait<`, AnyType, `>`]


@register_passable
struct VariadicPack[
    elt_is_mutable: __mlir_type.i1,
    lifetime: AnyLifetime[elt_is_mutable].type,
    element_trait: _AnyTypeMetaType,
    *element_types: element_trait,
    # TODO: Add address_space when Reference supports it.
](Sized):
    """A utility class to access variadic pack  arguments and provide an API for
    doing things with them.

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        inout or owned argument pack.
        lifetime: The reference lifetime of the underlying elements.
        element_trait: The trait that each element of the pack conforms to.
        element_types: The list of types held by the argument pack.
    """

    alias _mlir_pack_type = __mlir_type[
        `!lit.ref.pack<:variadic<`,
        element_trait,
        `> `,
        element_types,
        `, `,
        lifetime,
        `>`,
    ]

    var _value: Self._mlir_pack_type
    var _is_owned: Bool

    @always_inline
    fn __init__(inout self, value: Self._mlir_pack_type, is_owned: Bool):
        """Constructs a VariadicPack from the internal representation.

        Args:
            value: The argument to construct the pack with.
            is_owned: Whether this is an 'owned' pack or 'inout'/'borrowed'.
        """
        self._value = value
        self._is_owned = is_owned

    @always_inline
    fn __del__(owned self):
        """Destructor that releases elements if owned."""

        # Immutable variadics never own the memory underlying them,
        # microoptimize out a check of _is_owned.
        @parameter
        if not Bool(elt_is_mutable):
            return
        else:
            # If the elements are unowned, just return.
            if not self._is_owned:
                return

            alias len = Self.__len__()

            @parameter
            fn destroy_elt[i: Int]():
                # destroy the elements in reverse order.
                self.get_element[len - i - 1]().destroy_element_unsafe()

            unroll[destroy_elt, len]()

    @always_inline
    @staticmethod
    fn __len__() -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """

        @parameter
        fn variadic_size(
            x: __mlir_type[`!kgen.variadic<`, element_trait, `>`]
        ) -> Int:
            return __mlir_op.`pop.variadic.size`(x)

        alias result = variadic_size(element_types)
        return result

    @always_inline
    fn __len__(self) -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """
        return Self.__len__()

    @always_inline
    fn get_element[
        index: Int
    ](self) -> Reference[
        element_types[index.value],
        Self.elt_is_mutable,
        Self.lifetime,
    ]:
        """Return a reference to an element of the pack.

        Parameters:
            index: The element of the pack to return.

        Returns:
            A reference to the element.  The Reference's mutability follows the
            mutability of the pack argument convention.
        """
        var ref_elt = __mlir_op.`lit.ref.pack.get`[index = index.value](
            self._value
        )

        # Rebind the !lit.ref to agree on the element type.  This is needed
        # because we're getting a low level rebind to AnyType when the
        # element_types[index] expression is erased to AnyType for Reference.
        alias result_ref = Reference[
            element_types[index.value],
            Self.elt_is_mutable,
            Self.lifetime,
        ]
        return rebind[result_ref.mlir_ref_type](ref_elt)

    @always_inline
    fn each[func: fn[T: element_trait] (T) -> None](self):
        """Apply a function to each element of the pack in order.  This applies
        the specified function (which must be parametric on the element type) to
        each element of the pack, from the first element to the last, passing
        in each element as a borrowed argument.

        Parameters:
            func: The function to apply to each element.
        """

        @parameter
        fn unrolled[i: Int]():
            func(self.get_element[i]()[])

        unroll[unrolled, Self.__len__()]()
