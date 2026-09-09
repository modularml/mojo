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

#ifndef DEBUGINFOTYPES_TD
#define DEBUGINFOTYPES_TD

include "Support/DebugInfoDialect/IR/DebugInfoDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// DIType
//===----------------------------------------------------------------------===//

// Base class for debug info dialect types.
class DIType<string name, string typeMnemonic, list<Trait> traits = [],
             string baseCppClass = "DIType">
    : TypeDef<DebugInfoDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`` `<` struct(params) `>` ``";
}

//===----------------------------------------------------------------------===//
// DIArrayType
//===----------------------------------------------------------------------===//

def DebugInfo_DIArrayType : DIType<"DIArray", "array"> {
  let summary = "Array type debug info description";
  let description = [{
    A `debuginfo.array` type represents the debug information for an array
    type. It contains the element type and the size of the array.

    Example:

    ```mlir
    !debuginfo.array<10 x !elementType>
    ```
  }];
  let parameters = (ins
    "DIType":$elementType,
    "uint64_t":$elementCount
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "DIType":$elementType, "uint64_t":$elementCount
    ), [{
      return $_get(elementType.getContext(), elementType, elementCount);
    }]>
  ];
  let assemblyFormat = "`<` $elementCount `x` custom<DIType>($elementType) `>`";
}

//===----------------------------------------------------------------------===//
// DIBasicType
//===----------------------------------------------------------------------===//

def DebugInfo_DIBasicType : DIType<"DIBasic", "basic"> {
  let summary = "Basic type debug info description";
  let description = [{
    The `debuginfo.basic` types represents the debug information for a basic
    type. A basic type is a type that is not defined in terms of other types.

    Example:

    ```mlir
    !debuginfo.basic<f32 {
      sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_float
    }>
    ```
  }];
  let parameters = (ins
    "StringAttr":$name,
    "uint64_t":$sizeInBits,
    "uint32_t":$alignInBits,
    DIEncodingParameter:$encoding
  );

  let builders = [
    TypeBuilder<(ins
      "const Twine &":$name, "uint64_t":$sizeInBits, "uint32_t":$alignInBits,
      "unsigned":$encoding
    ), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, name), sizeInBits,
                   alignInBits, encoding);
    }]>
  ];

  let assemblyFormat = [{
    `<`
      custom<Name>($name) ` `
      `{` struct($sizeInBits, $alignInBits, $encoding) `}`
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// DIMemberType
//===----------------------------------------------------------------------===//

def DebugInfo_DIMemberType : DIType<"DIMember", "member"> {
  let summary = "Data member type debug info description";
  let description = [{
    The `debuginfo.member` type represents the debug information for a data
    member, e.g. an element within a struct.

    Example:

    ```mlir
    !debuginfo.member<x: !type>
    ```
  }];
  let parameters = (ins
    "StringAttr":$name,
    "DIType":$type
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins "StringAttr":$name, "DIType":$type), [{
      return $_get(name.getContext(), name, type);
    }]>,
    TypeBuilderWithInferredContext<(ins "const Twine &":$name, "DIType":$type), [{
      MLIRContext *ctx = type.getContext();
      return $_get(ctx, StringAttr::get(ctx, name), type);
    }]>
  ];
  let assemblyFormat = [{
    `<` custom<Name>($name) `` `:` custom<DIType>($type) `>`
  }];
}

//===----------------------------------------------------------------------===//
// DIPointerType
//===----------------------------------------------------------------------===//

def DebugInfo_PointerType : DIType<"DIPointer", "ptr"> {
  let summary = "Pointer type debug info description";
  let description = [{
    The `debuginfo.ptr` type represents the debug information for a pointer
    type.

    Example:

    ```mlir
    !debuginfo.ptr<!elementType {sizeInBits = 64, alignInBits = 64}>
    ```
  }];
  let parameters = (ins
    "DIType":$elementType,
    "unsigned":$sizeInBits,
    "unsigned":$alignInBits,
    OptionalParameter<"std::optional<unsigned>">:$addressSpace
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "DIType":$elementType, "unsigned":$sizeInBits,
      "unsigned":$alignInBits, "std::optional<unsigned>":$addressSpace
    ), [{
      return $_get(elementType.getContext(), elementType, sizeInBits,
                   alignInBits, addressSpace);
    }]>
  ];
  let assemblyFormat = [{
    `<` custom<DIType>($elementType) ` ` `{` `sizeInBits` `=` $sizeInBits `` `,`
    `alignInBits` `=` $alignInBits
    (`` `,` `addressSpace` `=` $addressSpace^)?
    `}` `>`
  }];
}

//===----------------------------------------------------------------------===//
// DIStructType
//===----------------------------------------------------------------------===//

def DebugInfo_DIStructType : DIType<"DIStruct", "struct"> {
  let summary = "Composite struct type debug info description";
  let description = [{
    The `debuginfo.struct` type represents the debug information for a struct
    type.

    Example:

    ```mlir
    !debuginfo.struct<Foo(!member1, !member2)>
    ```
  }];
  let parameters = (ins
    "StringAttr":$name,
    OptionalArrayRefParameter<"DIMemberType">:$members
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "StringAttr":$name, CArg<"ArrayRef<DIMemberType>", "{}">:$members
    ), [{
      return $_get(name.getContext(), name, members);
    }]>
  ];
  let extraClassDeclaration = [{
    /// Return the size of the type in bits.
    uint64_t getSizeInBits() const;

    /// Return the alignment of the type in bits.
    uint32_t getAlignInBits() const;
  }];
  let assemblyFormat = "`<` custom<Name>($name) `(` (`)`):($members^ `)`)? `>`";
}

//===----------------------------------------------------------------------===//
// DISubroutineType
//===----------------------------------------------------------------------===//

def DebugInfo_DISubroutineType : DIType<"DISubroutine", "subroutine"> {
  let summary = "Subroutine type debug info description";
  let description = [{
    The `debuginfo.subroutine` type represents the debug information for a
    function type.

    Example:

    ```mlir
    !debuginfo.subroutine<() -> (): DW_CC_normal>
    ```
  }];
  let parameters = (ins
    DICallingConventionParameter:$callingConvention,
    OptionalArrayRefParameter<"DIType">:$argumentTypes,
    OptionalArrayRefParameter<"DIType">:$resultTypes
  );
  let builders = [
    TypeBuilder<(ins "ArrayRef<DIType>":$argTypes,
                     "ArrayRef<DIType>":$resultTypes), [{
      return $_get($_ctxt, llvm::dwarf::DW_CC_normal, argTypes, resultTypes);
    }]>
  ];
  let assemblyFormat = [{
    `<`
      custom<DITypes>($argumentTypes) `->` custom<DITypes>($resultTypes) ``
       `:` $callingConvention
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// DITargetIndependentPointerType
//===----------------------------------------------------------------------===//

def DebugInfo_TargetIndependentPointerType : DIType<
    "DITargetIndependentPointer", "ti.ptr"
  > {
  let summary = "Target-Independent Pointer type debug info description";
  let description = [{
    The `debuginfo.ti.ptr` type represents the debug information for a partially
    resolved pointer type, where we have the element type resolved, but not any
    target-specific information resolved yet.

    This will eventually resolve into a DIPointerType.

    Example:

    ```mlir
    !debuginfo.ti.ptr<!elementType>
    ```

    TODO: We will eventually find a better way to represent all the partially
    resolved types that we want to keep track of during lowering, and that may
    be dialect-specific. Putting this here until more types are needed.
  }];
  let parameters = (ins "DIType":$elementType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "DIType":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
  let assemblyFormat = "`<` custom<DIType>($elementType) `>`";
}

//===----------------------------------------------------------------------===//
// DIUnresolvedMLIRType
//===----------------------------------------------------------------------===//

def DebugInfo_DIUnresolvedMLIRType : DIType<
    "DIUnresolvedMLIR", "unresolved"
  > {
  let summary = "Unresolved MLIR type debug info description";
  let description = [{
    The `debuginfo.unresolved` type represents the debug information for an
    unresolved MLIR type. This allows for carrying forward MLIR types that have
    yet to reach a point where the debug information can be fully defined. For
    example, some types, such as `index`, do not have a proper representation
    defined until a certain stage of lowering. Allow unresolved types within
    the debug info type system allows for progressively lowering of debug
    information alongside the IR it represents.

    Example:

    ```mlir
    !debuginfo.unresolved<index>
    ```
  }];
  let parameters = (ins "Type":$type);
  let assemblyFormat = "`<` $type `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$type), [{
      // Don't wrap a DIUnresolvedMLIRType again - just return the type itself.
      if (auto t = llvm::dyn_cast<DebugInfo::DIUnresolvedMLIRType>(type))
        return t;
      return $_get(type.getContext(), type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DIUnspecifiedType
//===----------------------------------------------------------------------===//

def DebugInfo_DIUnspecifiedType : DIType<"DIUnspecified", "unspecified"> {
  let summary = "Unspecified type debug info description";
  let description = [{
    The `debuginfo.unspecified` type represents the debug information for an
    unspecified type. Some languages have constructs in which a type may be
    left unspecified or the absence of a type may be explicitly indicated.

    Example:

    ```mlir
    !debuginfo.unspecified<"void">
    ```
  }];
  let parameters = (ins "StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "StringAttr":$name), [{
      return $_get(name.getContext(), name);
    }]>,
    TypeBuilder<(ins "StringRef":$name), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, name));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DIVariantType
//===----------------------------------------------------------------------===//

def DebugInfo_DIVariantType : DIType<"DIVariant", "variant"> {
  let summary = "Variant type debug info description";
  let description = [{
    The `debuginfo.variant` type represents the debug information for a variant
    type.

    It is emitted as a DWARF composite type with DW_TAG_variant_part, and with
    each of the variant types as members of the composite type.

    A variant type can optionally reference a discriminator. However, the
    discriminator entry is not a member of the emitted composite type. This
    allows the frontend to compose this with DIStructType to freely arrange the
    discriminator in memory.

    Example:

    ```mlir
    !debuginfo.variant<Foo(!variant1, !variant2), !discriminator,
                       {sizeInBits = 64, alignInBits = 64}>
    ```
  }];
  let parameters = (ins
    "StringAttr":$name,
    "uint64_t":$sizeInBits,
    "uint32_t":$alignInBits,
    OptionalArrayRefParameter<"DIMemberType">:$variants,
    OptionalParameter<"DIMemberType">:$discriminatorRef
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "StringAttr":$name,
      "uint64_t":$sizeInBits,
      "uint32_t":$alignInBits,
      CArg<"ArrayRef<DIMemberType>", "{}">:$variants,
      CArg<"DIMemberType", "{}">:$discriminatorRef
    ), [{
      return $_get(name.getContext(), name, sizeInBits, alignInBits, variants,
                   discriminatorRef);
    }]>
  ];
  let assemblyFormat = [{
    `<`
      custom<Name>($name) `(` (`)`):($variants^ `)`)? (`,` $discriminatorRef^)?
      `{` struct($sizeInBits, $alignInBits) `}`
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// DIVectorType
//===----------------------------------------------------------------------===//

def DebugInfo_DIVectorType : DIType<"DIVector", "vector"> {
  let summary = "Vector type debug info description";
  let description = [{
    The `debuginfo.vector` type represents the debug information for a vector
    type. It contains the element type and the size of the vector.

    Example:

    ```mlir
    !debuginfo.vector<10 x !elementType>
    !debuginfo.vector<"pop.simd": 10 x !elementType>
    ```
  }];
  let parameters = (ins
    "DIType":$elementType,
    "uint64_t":$elementCount,
    OptionalParameter<"StringAttr">:$name
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "DIType":$elementType, "uint64_t":$elementCount, CArg<"StringAttr", "{}">:$name
    ), [{
      return $_get(elementType.getContext(), elementType, elementCount, name);
    }]>
  ];
  let assemblyFormat =
    "`<` $elementCount `x` custom<DIType>($elementType) (` ` `{` struct($name)^ `}`)? `>`";
}

#endif // DEBUGINFOTYPES_TD
