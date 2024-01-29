# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
    fn __init__(*args: *Ts) -> Self:
        """Construct the list literal from the given values.

        Args:
            args: The init values.

        Returns:
            The constructed ListLiteral.
        """
        return Self {storage: args}

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
        return __mlir_op.`kgen.pack.get`[_type=T, index = i.value](self.storage)


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
    fn __init__(*value: type) -> Self:
        """Constructs a VariadicList from a variadic list of arguments.

        Args:
            value: The variadic argument list to construct the variadic list
              with.

        Returns:
            The VariadicList constructed.
        """
        return value

    @always_inline
    fn __init__(value: Self.storage_type) -> Self:
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.

        Returns:
            The VariadicList constructed.
        """
        return Self {value: value}

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
                Reference(self.__refitem__(i - 1)).destroy_element_unsafe()

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
    ) -> _LITRef[
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
    ].type:
        """Gets a single element on the variadic list.

        Args:
            index: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.value)

    # FIXME: This is horrible syntax to return an iterator whose lifetime is
    # bound to the VariadicListMem
    @always_inline
    fn __iter__[
        self_lifetime: ImmLifetime,
    ](
        self: _LITRef[Self, __mlir_attr.`0: i1`, self_lifetime].type
    ) -> _VariadicListMemIter[
        element_type, elt_is_mutable, lifetime, self_lifetime
    ]:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return _VariadicListMemIter[
            element_type, elt_is_mutable, lifetime, self_lifetime
        ](0, self)
