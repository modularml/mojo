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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""None"""

import enum
from collections.abc import Callable, Sequence
from typing import Protocol, overload

import max._core
from max.mlir import Location

from . import passes as passes

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

DiagnosticHandler = Callable

class DenseElementsAttr(max._core.Attribute): ...
class DenseResourceElementsAttr(max._core.Attribute): ...
class FlatSymbolRefAttr(max._core.Attribute): ...
class AffineMap: ...
class IntegerSet: ...
class _ElementsAttrIndexer: ...

class SignednessSemantics(enum.Enum):
    signless = 0

    signed = 1

    unsigned = 2

class BoolAttr(max._core.Attribute):
    def __init__(self, arg: bool, /) -> None: ...
    @property
    def value(self) -> bool: ...

class ElementsAttr(Protocol):
    """
    This interface is used for attributes that contain the constant elements of
    a tensor or vector type. It allows for opaquely interacting with the
    elements of the underlying attribute, and most importantly allows for
    accessing the element values (including iteration) in any of the C++ data
    types supported by the underlying attribute.

    An attribute implementing this interface can expose the supported data types
    in two steps:

    * Define the set of iterable C++ data types:

    An attribute may define the set of iterable types by providing a definition
    of tuples `ContiguousIterableTypesT` and/or `NonContiguousIterableTypesT`.

    -  `ContiguousIterableTypesT` should contain types which can be iterated
       contiguously. A contiguous range is an array-like range, such as
       ArrayRef, where all of the elements are layed out sequentially in memory.

    -  `NonContiguousIterableTypesT` should contain types which can not be
       iterated contiguously. A non-contiguous range implies no contiguity,
       whose elements may even be materialized when indexing, such as the case
       for a mapped_range.

    As an example, consider an attribute that only contains i64 elements, with
    the elements being stored within an ArrayRef. This attribute could
    potentially define the iterable types as so:

    ```c++
    using ContiguousIterableTypesT = std::tuple<uint64_t>;
    using NonContiguousIterableTypesT = std::tuple<APInt, Attribute>;
    ```

    * Provide a `FailureOr<iterator> try_value_begin_impl(OverloadToken<T>) const`
      overload for each iterable type

    These overloads should return an iterator to the start of the range for the
    respective iterable type or fail if the type cannot be iterated. Consider
    the example i64 elements attribute described in the previous section. This
    attribute may define the value_begin_impl overloads like so:

    ```c++
    /// Provide begin iterators for the various iterable types.
    /// * uint64_t
    FailureOr<const uint64_t *>
    value_begin_impl(OverloadToken<uint64_t>) const {
      return getElements().begin();
    }
    /// * APInt
    auto value_begin_impl(OverloadToken<llvm::APInt>) const {
      auto it = llvm::map_range(getElements(), [=](uint64_t value) {
        return llvm::APInt(/*numBits=*/64, value);
      }).begin();
      return FailureOr<decltype(it)>(std::move(it));
    }
    /// * Attribute
    auto value_begin_impl(OverloadToken<mlir::Attribute>) const {
      mlir::Type elementType = getShapedType().getElementType();
      auto it = llvm::map_range(getElements(), [=](uint64_t value) {
        return mlir::IntegerAttr::get(elementType,
                                      llvm::APInt(/*numBits=*/64, value));
      }).begin();
      return FailureOr<decltype(it)>(std::move(it));
    }
    ```

    After the above, ElementsAttr will now be able to iterate over elements
    using each of the registered iterable data types:

    ```c++
    ElementsAttr attr = myI64ElementsAttr;

    // We can access value ranges for the data types via `getValues<T>`.
    for (uint64_t value : attr.getValues<uint64_t>())
      ...;
    for (llvm::APInt value : attr.getValues<llvm::APInt>())
      ...;
    for (mlir::IntegerAttr value : attr.getValues<mlir::IntegerAttr>())
      ...;

    // We can also access the value iterators directly.
    auto it = attr.value_begin<uint64_t>(), e = attr.value_end<uint64_t>();
    for (; it != e; ++it) {
      uint64_t value = *it;
      ...
    }
    ```

    ElementsAttr also supports failable access to iterators and ranges. This
    allows for safely checking if the attribute supports the data type, and can
    also allow for code to have fast paths for native data types.

    ```c++
    // Using `tryGetValues<T>`, we can also safely handle when the attribute
    // doesn't support the data type.
    if (auto range = attr.tryGetValues<uint64_t>()) {
      for (uint64_t value : *range)
        ...;
      return;
    }

    // We can also access the begin iterator safely, by using `try_value_begin`.
    if (auto safeIt = attr.try_value_begin<uint64_t>()) {
      auto it = *safeIt, e = attr.value_end<uint64_t>();
      for (; it != e; ++it) {
        uint64_t value = *it;
        ...
      }
      return;
    }
    ```
    """

    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def splat(self) -> bool: ...
    @property
    def shaped_type(self) -> ShapedType: ...
    def get_values_impl(
        self, arg: max._core.TypeID, /
    ) -> _ElementsAttrIndexer | None: ...

class TypedAttr(Protocol):
    """
    This interface is used for attributes that have a type. The type of an
    attribute is understood to represent the type of the data contained in the
    attribute and is often used as the type of a value with this data.
    """

    @property
    def type(self) -> max._core.Type | None: ...

class ArrayAttr(max._core.Attribute):
    """
    Syntax:

    ```
    array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
    ```

    An array attribute is an attribute that represents a collection of attribute
    values.

    Examples:

    ```mlir
    []
    [10, i32]
    [affine_map<(d0, d1, d2) -> (d0, d1)>, i32, "string attribute"]
    ```
    """

    def __init__(self, value: Sequence[max._core.Attribute]) -> None: ...
    @property
    def value(self) -> Sequence[max._core.Attribute]: ...

class DictionaryAttr(max._core.Attribute):
    """
    Syntax:

    ```
    dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
    ```

    A dictionary attribute is an attribute that represents a sorted collection of
    named attribute values. The elements are sorted by name, and each name must be
    unique within the collection.

    Examples:

    ```mlir
    {}
    {attr_name = "string attribute"}
    {int_attr = 10, "string attr name" = "string attribute"}
    ```
    """

    def __init__(
        self, value: Sequence[max._core.NamedAttribute] = []
    ) -> None: ...
    @property
    def value(self) -> Sequence[max._core.NamedAttribute]: ...

class FloatAttr(max._core.Attribute):
    """
    Syntax:

    ```
    float-attribute ::= (float-literal (`:` float-type)?)
                      | (hexadecimal-literal `:` float-type)
    ```

    A float attribute is a literal attribute that represents a floating point
    value of the specified [float type](#floating-point-types). It can be
    represented in the hexadecimal form where the hexadecimal value is
    interpreted as bits of the underlying binary representation. This form is
    useful for representing infinity and NaN floating point values. To avoid
    confusion with integer attributes, hexadecimal literals _must_ be followed
    by a float type to define a float attribute.

    Examples:

    ```
    42.0         // float attribute defaults to f64 type
    42.0 : f32   // float attribute of f32 type
    0x7C00 : f16 // positive infinity
    0x7CFF : f16 // NaN (one of possible values)
    42 : f32     // Error: expected integer type
    ```
    """

    @overload
    def __init__(self, type: max._core.Type, value: float) -> None: ...
    @overload
    def __init__(self, type: max._core.Type, value: float) -> None: ...
    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def value(self) -> float: ...

class IntegerAttr(max._core.Attribute):
    """
    Syntax:

    ```
    integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                          | `true` | `false`
    ```

    An integer attribute is a literal attribute that represents an integral
    value of the specified integer or index type. `i1` integer attributes are
    treated as `boolean` attributes, and use a unique assembly format of either
    `true` or `false` depending on the value. The default type for non-boolean
    integer attributes, if a type is not specified, is signless 64-bit integer.

    Examples:

    ```mlir
    10 : i32
    10    // : i64 is implied here.
    true  // A bool, i.e. i1, value.
    false // A bool, i.e. i1, value.
    ```
    """

    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def value(self) -> int: ...
    @overload
    def __init__(self, type: IntegerType, value: int = 0) -> None: ...
    @overload
    def __init__(self, type: IndexType, value: int = 0) -> None: ...
    @overload
    def __init__(self, value: int) -> None: ...

class StringAttr(max._core.Attribute):
    """
    Syntax:

    ```
    string-attribute ::= string-literal (`:` type)?
    ```

    A string attribute is an attribute that represents a string literal value.

    Examples:

    ```mlir
    "An important string"
    "string with a type" : !dialect.string
    ```
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, bytes: str, type: max._core.Type) -> None: ...
    @overload
    def __init__(self, bytes: str) -> None: ...
    @property
    def value(self) -> str: ...
    @property
    def type(self) -> max._core.Type | None: ...

class TypeAttr(max._core.Attribute):
    """
    Syntax:

    ```
    type-attribute ::= type
    ```

    A type attribute is an attribute that represents a
    [type object](#type-system).

    Examples:

    ```mlir
    i32
    !dialect.type
    ```
    """

    def __init__(self, type: max._core.Type) -> None: ...
    @property
    def value(self) -> max._core.Type | None: ...

class UnitAttr(max._core.Attribute):
    """
    Syntax:

    ```
    unit-attribute ::= `unit`
    ```

    A unit attribute is an attribute that represents a value of `unit` type. The
    `unit` type allows only one value forming a singleton set. This attribute
    value is used to represent attributes that only have meaning from their
    existence.

    One example of such an attribute could be the `swift.self` attribute. This
    attribute indicates that a function parameter is the self/context parameter.
    It could be represented as a [boolean attribute](#boolean-attribute)(true or
    false), but a value of false doesn't really bring any value. The parameter
    either is the self/context or it isn't.


    Examples:

    ```mlir
    // A unit attribute defined with the `unit` value specifier.
    func.func @verbose_form() attributes {dialectName.unitAttr = unit}

    // A unit attribute in an attribute dictionary can also be defined without
    // the value specifier.
    func.func @simple_form() attributes {dialectName.unitAttr}
    ```
    """

    def __init__(self) -> None: ...

class ModuleOp(max._core.Operation):
    """
    A `module` represents a top-level container operation. It contains a single
    [graph region](../LangRef.md#control-flow-and-ssacfg-regions) containing a single block
    which can contain any operations and does not have a terminator. Operations
    within this region cannot implicitly capture values defined outside the module,
    i.e. Modules are [IsolatedFromAbove](../Traits#isolatedfromabove). Modules have
    an optional [symbol name](../SymbolsAndSymbolTables.md) which can be used to refer
    to them in operations.

    Example:

    ```mlir
    module {
      func.func @foo()
    }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        name: str | None = None,
    ) -> None: ...
    @overload
    def __init__(self, location: Location, name: str | None = None) -> None: ...
    @property
    def sym_name(self) -> str | None: ...
    @sym_name.setter
    def sym_name(self, arg: StringAttr, /) -> None: ...
    @property
    def sym_visibility(self) -> str | None: ...
    @sym_visibility.setter
    def sym_visibility(self, arg: StringAttr, /) -> None: ...
    @property
    def body(self) -> max._core.Block: ...

class ShapedType(Protocol):
    """
    This interface provides a common API for interacting with multi-dimensional
    container types. These types contain a shape and an element type.

    A shape is a list of sizes corresponding to the dimensions of the container.
    If the number of dimensions in the shape is unknown, the shape is "unranked".
    If the number of dimensions is known, the shape "ranked". The sizes of the
    dimensions of the shape must be positive, or kDynamic (in which case the
    size of the dimension is dynamic, or not statically known).
    """

    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def shape(self) -> Sequence[int]: ...
    def clone_with(
        self, arg0: Sequence[int] | None, arg1: max._core.Type
    ) -> ShapedType: ...
    def has_rank(self) -> bool: ...

class FunctionType(max._core.Type):
    """
    Syntax:

    ```
    // Function types may have multiple results.
    function-result-type ::= type-list-parens | non-function-type
    function-type ::= type-list-parens `->` function-result-type
    ```

    The function type can be thought of as a function signature. It consists of
    a list of formal parameter types and a list of formal result types.

    #### Example:

    ```mlir
    func.func @add_one(%arg0 : i64) -> i64 {
      %c1 = arith.constant 1 : i64
      %0 = arith.addi %arg0, %c1 : i64
      return %0 : i64
    }
    ```
    """

    def __init__(
        self,
        inputs: Sequence[max._core.Type] = [],
        results: Sequence[max._core.Type] = [],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Type]: ...
    @property
    def results(self) -> Sequence[max._core.Type]: ...

class IndexType(max._core.Type):
    """
    Syntax:

    ```
    // Target word-sized integer.
    index-type ::= `index`
    ```

    The index type is a signless integer whose size is equal to the natural
    machine word of the target ( [rationale](../../Rationale/Rationale/#integer-signedness-semantics) )
    and is used by the affine constructs in MLIR.

    **Rationale:** integers of platform-specific bit widths are practical to
    express sizes, dimensionalities and subscripts.
    """

    def __init__(self) -> None: ...

class IntegerType(max._core.Type):
    """
    Syntax:

    ```
    // Sized integers like i1, i4, i8, i16, i32.
    signed-integer-type ::= `si` [1-9][0-9]*
    unsigned-integer-type ::= `ui` [1-9][0-9]*
    signless-integer-type ::= `i` [1-9][0-9]*
    integer-type ::= signed-integer-type |
                     unsigned-integer-type |
                     signless-integer-type
    ```

    Integer types have a designated bit width and may optionally have signedness
    semantics.

    **Rationale:** low precision integers (like `i2`, `i4` etc) are useful for
    low-precision inference chips, and arbitrary precision integers are useful
    for hardware synthesis (where a 13 bit multiplier is a lot cheaper/smaller
    than a 16 bit one).
    """

    def __init__(
        self,
        width: int,
        signedness: SignednessSemantics = SignednessSemantics.signless,
    ) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def signedness(self) -> SignednessSemantics: ...
