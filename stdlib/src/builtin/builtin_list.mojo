# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the ListLiteral class.

These are Mojo built-ins, so you don't need to import them.
"""


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
struct _VariadicListIter[
    type: AnyRegType,
    list_type: AnyRegType,
    getitem: fn (list_type, Int) -> type,
]:
    """Const Iterator for VariadicList(Mem).

    Parameters:
        type: The type of the elements in the list.
        list_type: The type of the variadic list (Mem or non-Mem).
        getitem: The callback for getting an element from the list.
    """

    var index: Int
    var size: Int
    var src: list_type

    fn __next__(inout self) -> type:
        self.index += 1
        return getitem(self.src, self.index - 1)

    fn __len__(self) -> Int:
        return self.size - self.index


@register_passable("trivial")
struct VariadicList[type: AnyRegType](Sized):
    """A utility class to access variadic function arguments. Provides a "list"
    view of the function argument so that the size of the argument list and each
    individual argument can be accessed.

    Parameters:
        type: The type of the elements in the list.
    """

    alias StorageType = __mlir_type[`!kgen.variadic<`, type, `>`]
    var value: Self.StorageType
    """The underlying storage for the variadic list."""

    alias IterType = _VariadicListIter[
        type, VariadicList[type], Self.__getitem__
    ]

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
    fn __init__(value: Self.StorageType) -> Self:
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
        return Self.IterType(0, len(self), self)


@register_passable("trivial")
struct VariadicListMem[type: AnyRegType, life: Lifetime](Sized):
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes pointers to the elements in a way
    that can be enumerated.  Each element may be accessed with
    `__get_value_from_ref`.

    Parameters:
        type: The type of the elements in the list.
        life: The reference lifetime of the underlying elements.
    """

    alias RefType = __mlir_type[`!lit.ref<`, type, `, `, life, `>`]
    alias StorageType = __mlir_type[
        `!kgen.variadic<`, Self.RefType, `, borrow_in_mem>`
    ]
    var value: Self.StorageType
    """The underlying storage, a variadic list of pointers to elements of the
    given type."""

    alias IterType = _VariadicListIter[
        Self.RefType,
        VariadicListMem[type, life],
        Self.__getitem__,
    ]

    @always_inline
    fn __init__(value: Self.StorageType) -> Self:
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
    fn __getitem__(self, index: Int) -> Self.RefType:
        """Gets a single element on the variadic list.

        Args:
            index: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.value)

    @always_inline
    fn __iter__(self) -> Self.IterType:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return Self.IterType(0, len(self), self)
