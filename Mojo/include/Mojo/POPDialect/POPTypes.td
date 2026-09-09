//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPTYPES_TD
#define KGEN_POPDIALECT_POPTYPES_TD

include "Mojo/KGENDialect/KGENTypeInterfaces.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "Mojo/POPDialect/POPDialect.td"
include "Support/MDialect/MTypeInterfaces.td"
include "Mojo/Interpreter/InterpreterInterface.td"
include "mlir/IR/AttrTypeBase.td"

class POPType<string typeName, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<POPDialect, typeName,
            [DeclareTypeInterfaceMethods<DataLayoutInterface>] # traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def POP_ArrayType : POPType<"Array", "array", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized array type";
  let description = [{
    This type represents a statically-sized array of elements. The size and
    element type are parameterized.

    Example:

    ```mlir
    // An array of four scalar floats.
    !pop.array<4, scalar<f32>>

    // A parameterized array.
    !pop.array<size, type>
    ```
  }];

  let parameters = (ins "TypedAttr":$size, "Type":$elementType);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` ` `
        custom<ParamType>($elementType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size,
                                        "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "int64_t":$size, "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "ValueRange":$elements)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the size expression to an integer.
    std::optional<int64_t> getResolvedSize() const;
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class ArrayElementType<string lhs, string rhs> : TypesMatchWith<
    "value type matches array element type", lhs, rhs,
    "llvm::cast<POP::ArrayType>($_self).getElementType()",
    "typeRangeMatches">;

def ArrayKnownSize : ConfinedType<POP_ArrayType,
    [CPred<"llvm::cast<POP::ArrayType>($_self).getResolvedSize()">],
    "array with known size">;

//===----------------------------------------------------------------------===//
// UnionType
//===----------------------------------------------------------------------===//

def POP_UnionType : POPType<"Union", "union", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "a parameterized union type";
  let description = [{
    The `!pop.union` type is a type that can contain any one of its union
    element types. Its size is large enough to contain the largest element and
    its alignment meets the strictest of its elements' alignment requirements.
    Unlike `!kgen.variant`, this union is not discriminated.

    Unions can be parameterized with a variadic type parameter, allowing the
    union element types to be determined at monomorphization time. Unresolved
    (parameterized) unions are resolved during the KGEN elaboration pass when
    the enclosing generator is specialized with concrete type arguments.

    **Lifecycle of variadic unions:**
    1. **Creation**: Variadic unions are created with a parameterized variadic
       reference (e.g., `!pop.union<[Ts]>`). Inline variadic expansion is also
       supported (e.g., `!pop.union<[[T0, T1]]>` which prints as `!pop.union<T0, T1>`).
    2. **Elaboration**: During KGEN elaboration/monomorphization, the variadic
       is substituted with concrete types, producing a resolved union.
    3. **Lowering**: Only resolved unions (where `getVariadic()` returns a
       `ParamListAttr`) can be lowered to LLVM IR or have debug info generated.

    Use `isResolved()` to check if a union is resolved.

    Example:

    ```mlir
    !pop.union<>                                   // empty union
    !pop.union<i32>                                // single-element union
    !pop.union<i32, i64>                           // multi-element union
    !pop.union<[Ts]>                               // variadic reference
    !pop.union<[rebind(:!kgen.param_list<type> Ts)]> // rebound variadic
    ```
  }];

  let parameters = (ins "TypedAttr":$variadic);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilder<(ins "ArrayRef<Type>":$types)>,
    TypeBuilderWithInferredContext<(ins "ArrayRef<Type>":$types)>
  ];

  let extraClassDeclaration = [{
    /// Returns whether this union type is resolved (element types are concrete).
    /// An unresolved union has a parameterized variadic that hasn't been
    /// specialized yet.
    bool isResolved() const;

    /// Get an element type as a parameterized type.
    /// Returns an empty Type if the union is unresolved (variadic not yet
    /// specialized). Callers should check `isResolved()` before calling this
    /// method if the union may be unresolved.
    Type getType(unsigned index) const;

    /// Return the number of types in the union.
    /// Returns 0 if the union is unresolved (variadic not yet specialized).
    size_t getNumTypes() const;

    /// Get the concrete union types. Returns an empty range if the types
    /// are not yet resolved (e.g., still parameterized). Callers should check
    /// `isResolved()` to distinguish between an empty union and an unresolved
    /// union.
    llvm::iterator_range<llvm::mapped_iterator<ArrayRef<TypedAttr>::iterator,
                                               Type (*)(TypedAttr)>>
    getTypes() const;
  }];
}

//===----------------------------------------------------------------------===//
// Type constraints that combine SIMDType (now in KGEN dialect) with POP types.
//===----------------------------------------------------------------------===//

def UnboundOrPointerOrSIMDAddressType : AnyTypeOf<
    [PointerType, AddressSIMD, UnboundSIMD]>;

//===----------------------------------------------------------------------===//
// IntLiteralType
//===----------------------------------------------------------------------===//

def POP_IntLiteralType : POPType<"IntLiteral", "int_literal",[
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "Arbitrary-precision integer type for literals.";
  let description = [{
    The `!pop.int_literal` type represents an infinite precision
    integer value.  It can't be materialized, but allows for arbitrary
    integer size for compile time computations.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// FloatLiteralType
//===----------------------------------------------------------------------===//

def POP_FloatLiteralType : POPType<"FloatLiteral", "float_literal", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "Arbitrary-precision float type for literals.";
  let description = [{
    The `!pop.float_literal` type is for use in arbitrary precision
    compile time computations.  They are implemented as infinite precision
    rational number values.  It can't be materialized, but allows for
    arbitrary precision for compile time computations.
  }];

  let assemblyFormat = "";
}

#endif // KGEN_POPDIALECT_POPTYPES_TD
