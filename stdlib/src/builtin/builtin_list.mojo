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

from memory import Reference, UnsafePointer
from python import PythonObject


from collections import Dict

# ===----------------------------------------------------------------------===#
# ListLiteral
# ===----------------------------------------------------------------------===#


struct ListLiteral[*Ts: CollectionElement](
    Sized, CollectionElement, Formattable, Representable, Stringable
):
    """The type of a literal heterogeneous list expression.

    A list consists of zero or more values, separated by commas.

    Parameters:
        Ts: The type of the elements.
    """

    var storage: Tuple[*Ts]
    """The underlying storage for the list."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(inout self, owned *args: *Ts):
        """Construct the list literal from the given values.

        Args:
            args: The init values.
        """
        self.storage = Tuple(storage=args^)

    @always_inline("nodebug")
    fn __copyinit__(inout self, existing: Self):
        """Copy construct the tuple.

        Args:
            existing: The value to copy from.
        """
        self.storage = existing.storage

    fn __moveinit__(inout self, owned existing: Self):
        """Move construct the list.

        Args:
            existing: The value to move from.
        """

        self.storage = existing.storage^

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the list length.

        Returns:
            The length of this ListLiteral.
        """
        return len(self.storage)

    fn __str__(self) -> String:
        """Returns a string representation of a ListLiteral.

        Returns:
            The string representation of the ListLiteral.

        Here is an example below:
        ```mojo
        var my_list = [1, 2, "Mojo"]
        print(str(my_list))
        ```.
        """
        var output = String()
        var writer = output._unsafe_to_formatter()
        self.format_to(writer)
        return output^

    fn __repr__(self) -> String:
        """Returns a string representation of a ListLiteral.

        Returns:
            The string representation of the ListLiteral.

        Here is an example below:
        ```mojo
        var my_list = [1, 2, "Mojo"]
        print(repr(my_list))
        ```.
        """
        return self.__str__()

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """Format the list literal.

        Args:
            writer: The formatter to use.
        """
        writer.write("[")

        @parameter
        for i in range(len(VariadicList(Ts))):
            alias T = Ts[i]
            var s: String
            if i > 0:
                writer.write(", ")

            @parameter
            if _type_is_eq[T, PythonObject]():
                s = str(rebind[PythonObject](self.storage[i]))
            elif _type_is_eq[T, Int]():
                s = repr(rebind[Int](self.storage[i]))
            elif _type_is_eq[T, Float64]():
                s = repr(rebind[Float64](self.storage[i]))
            elif _type_is_eq[T, Bool]():
                s = repr(rebind[Bool](self.storage[i]))
            elif _type_is_eq[T, String]():
                s = repr(rebind[String](self.storage[i]))
            elif _type_is_eq[T, StringLiteral]():
                s = repr(rebind[StringLiteral](self.storage[i]))
            else:
                s = String("unknown")
                constrained[
                    False, "cannot convert list literal element to string"
                ]()
            writer.write(s)
        writer.write("]")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn get[i: Int, T: CollectionElement](self) -> ref [self.storage] T:
        """Get a list element at the given index.

        Parameters:
            i: The element index.
            T: The element type.

        Returns:
            The element at the given index.
        """
        return self.storage.get[i, T]()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __contains__[
        T: EqualityComparableCollectionElement
    ](self, value: T) -> Bool:
        """Determines if a given value exists in the ListLiteral.

        Parameters:
            T: The type of the value to search for. Must implement the
              `EqualityComparable` trait.

        Args:
            value: The value to search for in the ListLiteral.

        Returns:
            True if the value is found in the ListLiteral, False otherwise.
        """
        return value in self.storage


# ===----------------------------------------------------------------------===#
# VariadicList / VariadicListMem
# ===----------------------------------------------------------------------===#


@value
struct _VariadicListIter[type: AnyTrivialRegType]:
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
struct VariadicList[type: AnyTrivialRegType](Sized):
    """A utility class to access variadic function arguments. Provides a "list"
    view of the function argument so that the size of the argument list and each
    individual argument can be accessed.

    Parameters:
        type: The type of the elements in the list.
    """

    alias _mlir_type = __mlir_type[`!kgen.variadic<`, type, `>`]
    var value: Self._mlir_type
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
    fn __init__(inout self, value: Self._mlir_type):
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
    fn __getitem__(self, idx: Int) -> type:
        """Gets a single element on the variadic list.

        Args:
            idx: The index of the element to access on the list.

        Returns:
            The element on the list corresponding to the given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, idx.value)

    @always_inline
    fn __iter__(self) -> Self.IterType:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return Self.IterType(0, self)


@value
struct _VariadicListMemIter[
    elt_is_mutable: Bool, //,
    elt_type: AnyType,
    elt_lifetime: Lifetime[elt_is_mutable].type,
    list_lifetime: ImmutableLifetime,
]:
    """Iterator for VariadicListMem.

    Parameters:
        elt_is_mutable: Whether the elements in the list are mutable.
        elt_type: The type of the elements in the list.
        elt_lifetime: The lifetime of the elements.
        list_lifetime: The lifetime of the VariadicListMem.
    """

    alias variadic_list_type = VariadicListMem[elt_type, elt_lifetime]

    var index: Int
    var src: Reference[Self.variadic_list_type, list_lifetime]

    fn __init__(
        inout self, index: Int, ref [list_lifetime]list: Self.variadic_list_type
    ):
        self.index = index
        self.src = Reference.address_of(list)

    fn __next__(inout self) -> Self.variadic_list_type.reference_type:
        self.index += 1
        # TODO: Need to make this return a dereferenced reference, not a
        # reference that must be deref'd by the user.
        return rebind[Self.variadic_list_type.reference_type](
            Reference.address_of(self.src[][self.index - 1])
        )

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


# Helper to compute the union of two lifetimes:
# TODO: parametric aliases would be nice.
struct _lit_lifetime_union[
    is_mutable: Bool, //,
    a: Lifetime[is_mutable].type,
    b: Lifetime[is_mutable].type,
]:
    alias result = __mlir_attr[
        `#lit.lifetime.union<`,
        a,
        `,`,
        b,
        `> : !lit.lifetime<`,
        is_mutable.value,
        `>`,
    ]


struct _lit_mut_cast[
    is_mutable: Bool, //,
    operand: Lifetime[is_mutable].type,
    result_mutable: Bool,
]:
    alias result = __mlir_attr[
        `#lit.lifetime.mutcast<`,
        operand,
        `> : !lit.lifetime<`,
        +result_mutable.value,
        `>`,
    ]


struct VariadicListMem[
    elt_is_mutable: Bool, //,
    element_type: AnyType,
    lifetime: Lifetime[elt_is_mutable].type,
](Sized):
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes references to the elements in a
    way that can be enumerated.  Each element may be accessed with `elt[]`.

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        inout or owned argument.
        element_type: The type of the elements in the list.
        lifetime: The reference lifetime of the underlying elements.
    """

    alias reference_type = Reference[element_type, lifetime]
    alias _mlir_ref_type = Self.reference_type._mlir_type
    alias _mlir_type = __mlir_type[
        `!kgen.variadic<`, Self._mlir_ref_type, `, borrow_in_mem>`
    ]

    var value: Self._mlir_type
    """The underlying storage, a variadic list of references to elements of the
    given type."""

    # This is true when the elements are 'owned' - these are destroyed when
    # the VariadicListMem is destroyed.
    var _is_owned: Bool

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # Provide support for borrowed variadic arguments.
    @always_inline
    fn __init__(inout self, value: Self._mlir_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        self.value = value
        self._is_owned = False

    # Provide support for variadics of *inout* arguments.  The reference will
    # automatically be inferred to be mutable, and the !kgen.variadic will have
    # convention=inout.
    alias _inout_variadic_type = __mlir_type[
        `!kgen.variadic<`, Self._mlir_ref_type, `, inout>`
    ]

    @always_inline
    fn __init__(inout self, value: Self._inout_variadic_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        var tmp = value
        # We need to bitcast different argument conventions to a consistent
        # representation.  This is ugly but effective.
        self.value = UnsafePointer.address_of(tmp).bitcast[Self._mlir_type]()[]
        self._is_owned = False

    # Provide support for variadics of *owned* arguments.  The reference will
    # automatically be inferred to be mutable, and the !kgen.variadic will have
    # convention=owned_in_mem.
    alias _owned_variadic_type = __mlir_type[
        `!kgen.variadic<`, Self._mlir_ref_type, `, owned_in_mem>`
    ]

    @always_inline
    fn __init__(inout self, value: Self._owned_variadic_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        var tmp = value
        # We need to bitcast different argument conventions to a consistent
        # representation.  This is ugly but effective.
        self.value = UnsafePointer.address_of(tmp).bitcast[Self._mlir_type]()[]
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
        if not elt_is_mutable:
            return

        else:
            # If the elements are unowned, just return.
            if not self._is_owned:
                return

            # Otherwise this is a variadic of owned elements, destroy them.  We
            # destroy in backwards order to match how arguments are normally torn
            # down when CheckLifetimes is left to its own devices.
            for i in reversed(range(len(self))):
                UnsafePointer.address_of(self[i]).destroy_pointee()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """
        return __mlir_op.`pop.variadic.size`(self.value)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(
        self, idx: Int
    ) -> ref [
        _lit_lifetime_union[
            lifetime,
            # cast mutability of self to match the mutability of the element,
            # since that is what we want to use in the ultimate reference and
            # the union overall doesn't matter.
            _lit_mut_cast[__lifetime_of(self), elt_is_mutable].result,
        ].result
    ] element_type:
        """Gets a single element on the variadic list.

        Args:
            idx: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __get_litref_as_mvalue(
            __mlir_op.`pop.variadic.get`(self.value, idx.value)
        )

    fn __iter__(
        self,
    ) -> _VariadicListMemIter[element_type, lifetime, __lifetime_of(self),]:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return _VariadicListMemIter[
            element_type,
            lifetime,
            __lifetime_of(self),
        ](0, self)


# ===----------------------------------------------------------------------===#
# _LITRefPackHelper
# ===----------------------------------------------------------------------===#


alias _AnyTypeMetaType = __mlir_type[`!lit.anytrait<`, AnyType, `>`]


@value
struct _LITRefPackHelper[
    is_mutable: Bool, //,
    lifetime: Lifetime[is_mutable].type,
    address_space: __mlir_type.index,
    element_trait: _AnyTypeMetaType,
    *element_types: element_trait,
]:
    """This struct mirrors the !lit.ref.pack type, and provides aliases and
    methods that are useful for working with it."""

    alias _mlir_type = __mlir_type[
        `!lit.ref.pack<:variadic<`,
        element_trait,
        `> `,
        element_types,
        `, `,
        lifetime,
        `, `,
        address_space,
        `>`,
    ]

    var storage: Self._mlir_type

    # This is the element_types list lowered to `variadic<type>` type for kgen.
    alias _kgen_element_types = rebind[
        __mlir_type.`!kgen.variadic<!kgen.type>`
    ](Self.element_types)

    # Use variadic_ptr_map to construct the type list of the !kgen.pack that the
    # !lit.ref.pack will lower to.  It exposes the pointers introduced by the
    # references.
    alias _variadic_pointer_types = __mlir_attr[
        `#kgen.param.expr<variadic_ptr_map, `,
        Self._kgen_element_types,
        `, 0: index>: !kgen.variadic<!kgen.type>`,
    ]

    # This is the !kgen.pack type with pointer elements.
    alias kgen_pack_with_pointer_type = __mlir_type[
        `!kgen.pack<:variadic<type> `, Self._variadic_pointer_types, `>`
    ]

    # This rebinds `in_pack` to the equivalent `!kgen.pack` with kgen pointers.
    fn get_as_kgen_pack(self) -> Self.kgen_pack_with_pointer_type:
        return rebind[Self.kgen_pack_with_pointer_type](self.storage)

    alias _variadic_with_pointers_removed = __mlir_attr[
        `#kgen.param.expr<variadic_ptrremove_map, `,
        Self._variadic_pointer_types,
        `>: !kgen.variadic<!kgen.type>`,
    ]

    # This is the `!kgen.pack` type that happens if one loads all the elements
    # of the pack.
    alias loaded_kgen_pack_type = __mlir_type[
        `!kgen.pack<:variadic<type> `, Self._variadic_with_pointers_removed, `>`
    ]

    # This returns the stored KGEN pack after loading all of the elements.
    fn get_loaded_kgen_pack(self) -> Self.loaded_kgen_pack_type:
        return __mlir_op.`kgen.pack.load`(self.get_as_kgen_pack())


# ===----------------------------------------------------------------------===#
# VariadicPack
# ===----------------------------------------------------------------------===#


@register_passable
struct VariadicPack[
    elt_is_mutable: __mlir_type.i1, //,
    lifetime: Lifetime[Bool {value: elt_is_mutable}].type,
    element_trait: _AnyTypeMetaType,
    *element_types: element_trait,
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

    alias _mlir_type = __mlir_type[
        `!lit.ref.pack<:variadic<`,
        element_trait,
        `> `,
        element_types,
        `, `,
        lifetime,
        `>`,
    ]

    var _value: Self._mlir_type
    var _is_owned: Bool

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(inout self, value: Self._mlir_type, is_owned: Bool):
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
        if Bool(elt_is_mutable):
            # If the elements are unowned, just return.
            if not self._is_owned:
                return

            @parameter
            for i in reversed(range(Self.__len__())):
                UnsafePointer.address_of(self[i]).destroy_pointee()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

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

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__[
        index: Int
    ](self) -> ref [Self.lifetime] element_types[index.value]:
        """Return a reference to an element of the pack.

        Parameters:
            index: The element of the pack to return.

        Returns:
            A reference to the element.  The Reference's mutability follows the
            mutability of the pack argument convention.
        """
        litref_elt = __mlir_op.`lit.ref.pack.extract`[index = index.value](
            self._value
        )
        return __get_litref_as_mvalue(litref_elt)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn each[func: fn[T: element_trait] (T) capturing -> None](self):
        """Apply a function to each element of the pack in order.  This applies
        the specified function (which must be parametric on the element type) to
        each element of the pack, from the first element to the last, passing
        in each element as a borrowed argument.

        Parameters:
            func: The function to apply to each element.
        """

        @parameter
        for i in range(Self.__len__()):
            func(self[i])

    @always_inline
    fn each_idx[
        func: fn[idx: Int, T: element_trait] (T) capturing -> None
    ](self):
        """Apply a function to each element of the pack in order.  This applies
        the specified function (which must be parametric on the element type) to
        each element of the pack, from the first element to the last, passing
        in each element as a borrowed argument.

        Parameters:
            func: The function to apply to each element.
        """

        @parameter
        for i in range(Self.__len__()):
            func[i](self[i])
