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
//
// This defines attributes in the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_TD
#define KGEN_LITDIALECT_LITATTRS_TD

include "Mojo/KGENDialect/KGENAttrInterfaces.td"
include "Mojo/KGENDialect/KGENAttrs.td"
include "Mojo/LITDialect/LITEnums.td"
include "Mojo/LITDialect/LITDialect.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

/// An attribute in the LIT dialect.
class LITAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LITDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// A typed attribute in the LIT dialect.
class LITTypedAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : LITAttr<attrName, attrMnemonic, [TypedAttrInterface] # traits>;

/// An array of DenseResourceElementsAttr's typically used on a
/// lit.package op to store the extern LLVM bitcode.
def LIT_DenseResourceElementsArrayAttr : ArrayOfAttr<LITDialect,
    "DenseResourceElementsArray", "dense_resource_elements_array",
    "DenseResourceElementsAttr"> {
  let builders = [
    AttrBuilderWithInferredContext<(ins "DenseResourceElementsAttr":$bitcode), [{
      return get(bitcode.getContext(), bitcode);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// SymbolAttr
//===----------------------------------------------------------------------===//

def LIT_SymbolAttr : LITAttr<"Symbol", "symbol"> {
  let summary = "a type symbol reference";
  let description = [{
    This attribute wraps a `SymbolRefAttr` in another attribute type so that
    aliases can be generated for it selectively.
  }];

  let parameters = (ins "SymbolRefAttr":$value);
  let assemblyFormat = "`` $value";

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// FnMetaOriginDataAttr
//===----------------------------------------------------------------------===//

def LIT_FnMetaOriginDataAttr : LITAttr<"FnMetaOriginData",
    "fn_meta_origin_data",
    [DeclareAttrInterfaceMethods<FnMetadataAttrInterface>]> {
  let summary = "Function origin/lifetime metadata.";
  let description = [{
    The `#lit.fn_meta_origin_data` attribute carries LIT-specific origin and
    lifetime information for a function: the number of implicit origin
    declarations, the captured origin set, and whether nested-origin
    exclusivity checking is disabled. The parameter (POG) list lives on
    [[kgen.func]]'s `argListAttrs` field, not here.

    Example:

    ```mlir
    // Metadata for a function with two implicit origin declarations and
    // no captured origins.
    #lit.fn_meta_origin_data<2>
    ```
  }];

  let parameters = (ins
    "size_t":$numImplicitOriginDecls,
    "TypedAttr":$captureOrigins,
    // TODO: Generalize this to a set of origin parameters that are guaranteed
    // to not be dereferenced.
    OptionalParameter<"bool">:$isNestedOriginsReadOnly,
    OptionalParameter<"bool">:$definesInteriorOrigins
  );
  let assemblyFormat = [{
    `<` $numImplicitOriginDecls
    `` custom<CaptureOrigins>($captureOrigins)
    (`,` $isNestedOriginsReadOnly^ (`,` $definesInteriorOrigins^)?)? `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins)>,
    AttrBuilder<(ins "size_t":$numImplicitOriginDecls,
                     "TypedAttr":$captureOrigins,
                     "bool":$isNestedOriginsReadOnly,
                     "bool":$definesInteriorOrigins)>,
  ];

  let extraClassDeclaration = [{
    /// Reconstruct the metadata with the union of the current capture origins
    /// with the new capture origins.
    FnMetaOriginDataAttr addCaptureOrigins(TypedAttr origins);
  }];
}

//===----------------------------------------------------------------------===//
// DeprecationInfoAttr
//===----------------------------------------------------------------------===//

def LIT_DeprecationInfoAttr : LITAttr<"DeprecationInfo", "deprecation"> {
  let summary = "Deprecation information for a declaration";
  let description = [{
    The `#lit.deprecation` attribute contains deprecation information for a
    declaration marked with the `@deprecated` decorator. It stores the warning
    message and optionally the replacement identifier (from `@deprecated(use=X)`).

    Example:

    ```mlir
    #lit.deprecation<"use new_func instead">
    #lit.deprecation<"'old_func' is deprecated, use 'new_func' instead", "new_func">
    ```
  }];

  let parameters = (ins
    "StringAttr":$warning,
    OptionalParameter<"StringAttr">:$replacement
  );
  let assemblyFormat = "`<` $warning (`,` $replacement^)? `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "StringAttr":$warning,
      CArg<"StringAttr", "{}">:$replacement
    ), [{
      return get(warning.getContext(), warning, replacement);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// UnavailableInfoAttr
//===----------------------------------------------------------------------===//

def LIT_UnavailableInfoAttr : LITAttr<"UnavailableInfo", "unavailable"> {
  let summary = "Unavailability information for a declaration";
  let description = [{
    The `#lit.unavailable` attribute contains unavailability information for a
    declaration marked with the `@unavailable` decorator. It stores the
    diagnostic message and optionally the replacement identifier (from
    `@unavailable(use=X)`).

    Unlike `@deprecated`, references to declarations marked `@unavailable`
    produce an error rather than a warning.

    Example:

    ```mlir
    #lit.unavailable<"use byte_length() or codepoint_length() instead">
    #lit.unavailable<"use 'new_func' instead", "new_func">
    ```
  }];

  let parameters = (ins
    "StringAttr":$reason,
    OptionalParameter<"StringAttr">:$replacement
  );
  let assemblyFormat = "`<` $reason (`,` $replacement^)? `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "StringAttr":$reason,
      CArg<"StringAttr", "{}">:$replacement
    ), [{
      return get(reason.getContext(), reason, replacement);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DocStringAttr
//===----------------------------------------------------------------------===//

def LIT_DocStringAttr : LITAttr<"DocString", "doc.string"> {
  let summary = "A source documentation string";
  let description = [{
    The `#lit.doc.string` attribute represents a documentation string within
    Mojo source. It contains the raw string, and optionally the source location
    of the start of the string.

    Example:

    ```mlir
    #lit.doc.string<"This is a doc string">
    #lit.doc.string<"This is a doc string", loc("file.mojo":10:8)>
    ```
  }];

  let parameters = (ins
    "StringAttr":$string,
    OptionalParameter<"FileLineColLoc">:$location
  );
  let assemblyFormat = "`<` $string (`,` $location^)? `>`";

  let builders = [AttrBuilderWithInferredContext<(ins
    "StringAttr":$string, CArg<"FileLineColLoc", "{}">:$location),
  [{
    return get(string.getContext(), string, location);
  }]>];
}

//===----------------------------------------------------------------------===//
// ImportPathAttr
//===----------------------------------------------------------------------===//

def LIT_ImportPathAttr : LITAttr<"ImportPath", "import_path"> {
  let summary = "A structured module import path";
  let description = [{
    The `#lit.import_path` attribute is the structured form of a module import
    path: the raw name components plus the number of leading dots of a
    relative import (0 for an absolute import). Components carry no separators
    or escaping, so a component may itself contain periods (an escaped
    identifier such as `package.with.dots`).

    Example:

    ```mlir
    // import a.b.c
    #lit.import_path<0, ["a", "b", "c"]>
    // from ..utils import x
    #lit.import_path<2, ["utils"]>
    // from . import x
    #lit.import_path<1>
    ```
  }];

  let parameters = (ins
    "unsigned":$relativeLevel,
    OptionalArrayRefParameter<"StringAttr">:$components
  );
  let assemblyFormat = "`<` $relativeLevel (`,` `[` $components^ `]`)? `>`";

  let builders = [
    AttrBuilder<(ins "unsigned":$relativeLevel,
                     "ArrayRef<StringRef>":$components), [{
      SmallVector<StringAttr> componentAttrs;
      componentAttrs.reserve(components.size());
      for (StringRef component : components)
        componentAttrs.push_back(StringAttr::get($_ctxt, component));
      return $_get($_ctxt, relativeLevel, componentAttrs);
    }]>
  ];

  // An absolute path must name at least one component; import resolution
  // relies on this invariant.
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// UnboundMLIROperationAttr
//===----------------------------------------------------------------------===//

def UnboundMLIROperationAttr
  : LITTypedAttr<"UnboundMLIROperation", "unbound_mlir_operation"> {
  let summary = "An unbound mlir operation";
  let description = [{
    This attribute is used by the Lit parser as a reference to an MLIR
    operation by name: it may have bound attributes as well.  When applied to
    value operands, it materializes into a specific MLIR operation instance
  }];

  let parameters = (ins "StringAttr":$name, "DictionaryAttr":$attrs);
  let assemblyFormat = "$name $attrs";
  let extraClassDeclaration = [{ Type getType() const; }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        "DictionaryAttr":$attrs), [{
      return $_get(name.getContext(), name, attrs);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// UnpackedAttr
//===----------------------------------------------------------------------===//

def LIT_UnpackedAttr : LITTypedAttr<"Unpacked", "unpacked"> {
  let summary = "An attribute representing a `*x` or `**x` value in param list";
  let description = [{
    Represents *x and **x expressions that unpack a variadic list.
  }];

  let parameters = (ins "TypedAttr":$value, "bool":$kwOnly,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = [{
    `<` custom<ColonTypeParamValue>($value) `,` custom<UnpackKind>($kwOnly) `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "bool":$kwOnly,
                                        "Type":$type), [{
      return $_get(type.getContext(), value, kwOnly, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// EllipsisAttr
//===----------------------------------------------------------------------===//

def LIT_EllipsisAttr : LITTypedAttr<"Ellipsis", "ellipsis"> {
  let summary = "An attribute representing a `...` value";
  let description = [{
    Represents ... value, a general literal expression in Mojo.  It also occurs
    in a parameter list, indicating that default parameter values should not be
    bound.
  }];

  let parameters = (ins);

  let extraClassDeclaration = [{ Type getType() const; }];
}


//===----------------------------------------------------------------------===//
// ModuleAttr
//===----------------------------------------------------------------------===//

def ModuleAttr : LITTypedAttr<"Module", "module"> {
  let summary = "A module attribute.";
  let description = [{
    The `#lit.module` attribute represents an unresolved module value.

    Example:

    ```mlir
    #lit.module : !lit.module<@some_module>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "LIT::ModuleType">:$type
  );
  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "LIT::ModuleType":$type),
                                   [{ return get(type.getContext(), type); }]>
  ];
}

//===----------------------------------------------------------------------===//
// AnyOriginAttr
//===----------------------------------------------------------------------===//

def AnyOriginAttr : LITTypedAttr<"AnyOrigin", "any.origin"> {
  let summary = "A !lit.origin<mut> value that could refer to anything";
  let description = [{
    This attribute is a singleton (for each mutability) which represents
    origin that might alias anything.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "OriginType">:$type
  );

  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return get(type.getContext(), ::cast<OriginType>(type));
    }]>,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$isMutable), [{
      return get(isMutable.getContext(), OriginType::get(isMutable));
    }]>,
    AttrBuilder<(ins "bool":$isMutable), [{
      return get($_ctxt, OriginType::get($_ctxt, isMutable));
    }]>
 ];
}

//===----------------------------------------------------------------------===//
// ComptimeOriginAttr
//===----------------------------------------------------------------------===//

def ComptimeOriginAttr : LITTypedAttr<"ComptimeOrigin", "comptime.origin", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface>
  ]> {
  let summary = "A !lit.origin<mut> value that refers to comptime memory";
  let description = [{
    This attribute is a singleton (for each mutability) which represents the
    origin of memory that is interpreted at comptime. This allows the compiler
    to reason locally about references within the interpreter to compile-time
    only memory without references leaking to runtime.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "OriginType">:$type
  );

  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return get(type.getContext(), ::cast<OriginType>(type));
    }]>,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$isMutable), [{
      return get(isMutable.getContext(), OriginType::get(isMutable));
    }]>,
    AttrBuilder<(ins "bool":$isMutable), [{
      return get($_ctxt, OriginType::get($_ctxt, isMutable));
    }]>
 ];
}


//===----------------------------------------------------------------------===//
// StaticOriginAttr
//===----------------------------------------------------------------------===//

def StaticOriginAttr : LITTypedAttr<"StaticOrigin", "static.origin", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface>
  ]> {
  let summary = "A !lit.origin<mut> value that refers to static memory";
  let description = [{
    This attribute is a singleton (for each mutability) which represents the
    origin of global static memory, used for static strings and global
    variables.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "OriginType">:$type
  );

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// OriginUnionAttr
//===----------------------------------------------------------------------===//

def OriginUnionAttr : LITTypedAttr<"OriginUnion", "origin.union", [
      DeclareAttrInterfaceMethods<ParameterAttrInterface>
  ]> {
  let summary = "A #lit.origin.union set";
  let description = [{
    This attribute represents a union of symbolic origin attributes.  It is
    canonicalized on construction, and always has !lit.origin type.
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);
  let assemblyFormat = [{
    custom<ParameterValues>($operands)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilderWithInferredContext<(ins
      "ArrayRef<TypedAttr>":$operands,
      "OriginType":$type
    ), "", "TypedAttr">,
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands, "OriginType":$type), [{
      return get(operands, type);
    }], "TypedAttr">,
    AttrBuilder<(ins CArg<"ArrayRef<TypedAttr>", "{}">:$operands),
                [{}], "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return get({}, ::cast<OriginType>(type));
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    size_t getNumOperands() const {
      return getOperands().size();
    }
    TypedAttr getOperand(size_t i) const {
      return getOperands()[i];
    }

    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginUnionAttr getFromBytecode(
        ArrayRef<TypedAttr> operands, OriginType type);
  }];
}

//===----------------------------------------------------------------------===//
// OriginEqAttr
//===----------------------------------------------------------------------===//

def OriginEqAttr : LITTypedAttr<"OriginEq", "origin.eq",
   [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Check if two !lit.origin values are equal";
  let description = [{
    This attribute is a check if two !lit.origin values are equal. This is used
    at parser time in 'where' clauses, but cannot be used in things like
    "comptime if" that are evaluated at elaboration time (because origins are
    gone by then).
  }];

  let parameters = (ins "TypedAttr":$lhs, "TypedAttr":$rhs);

  let assemblyFormat = "`<` $lhs `,` $rhs `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // These builders fold the request automatically.
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$lhs, "TypedAttr":$rhs), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$lhs, "TypedAttr":$rhs), [{
      return get(lhs, rhs);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// OriginMutCastAttr
//===----------------------------------------------------------------------===//

def OriginMutCastAttr : LITTypedAttr<"OriginMutCast", "origin.mutcast", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
  ]> {
  let summary = "Cast a !lit.origin value to a different mutability";
  let description = [{
    This attribute is a cast of a origin value from one mutability to another.
    It is often folded away by the builder.
  }];

  let parameters = (ins "TypedAttr":$operand,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);

  let assemblyFormat = "`<` $operand `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // These builders fold the request automatically.
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "TypedAttr":$isMutable), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "Type":$type), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "bool":$isMutable), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$operand, "OriginType":$type), [{
      return get(operand, type);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// If the specified value has a OriginMutCastAttr on it, return the
    /// operand, otherwise return the value.
    static TypedAttr strip(TypedAttr value) {
      if (auto castAttr = llvm::dyn_cast<OriginMutCastAttr>(value))
        return castAttr.getOperand();
      return value;
    }

    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginMutCastAttr getFromBytecode(
        TypedAttr operand, OriginType type);
  }];
}

//===----------------------------------------------------------------------===//
// OriginFieldAttr
//===----------------------------------------------------------------------===//

def OriginFieldAttr : LITTypedAttr<"OriginField", "origin.field", [
          DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
  ]> {
  let summary = "Get a field-sensitive subfield of another origin";
  let description = [{
    This attribute refines a larger origin to that of a subfield.  This
    subfield is known to alias the larger origin, but not other sibling origins.
  }];

  let parameters = (ins "TypedAttr":$base, "StringAttr":$field,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);

  let assemblyFormat = "`<` $base `,` $field `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // These builders fold the request automatically.
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$base, "StringAttr":$field), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$base, "StringAttr":$field,
                "OriginType":$type), [{
      return get(base, field);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginFieldAttr getFromBytecode(TypedAttr base,
        StringAttr field, OriginType type);
  }];
}


//===----------------------------------------------------------------------===//
// InteriorOriginAttr
//===----------------------------------------------------------------------===//

def InteriorOriginAttr : LITTypedAttr<"InteriorOrigin", "interior.origin",  [
          DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
  ]> {
  let summary = "Access an interior origin";
  let description = [{
    This attribute specifies an interior memory location that is usually (but
    not necessarily) a pointer indirection away from the base.  The base origin
    controls invalidation of the interior origin.  The string name is used in
    error messages when the interior origin is used after invalidation, and can
    be used to identify the interior object when mojo gets the ability to
    specify fine-grained invalidation sets.
  }];

  let parameters = (ins "TypedAttr":$base, "TypedAttr":$userName,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);

  let assemblyFormat = "`<` $base `,` $userName `>`";

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let builders = [
    // These builders fold the request automatically.
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$base, "TypedAttr":$userName), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$base, "TypedAttr":$userName,
                "OriginType":$type), [{
      return get(base, userName);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    static TypedAttr get(TypedAttr baseOrigin, llvm::StringRef userName);
    static InteriorOriginAttr getFromBytecode(TypedAttr base,
        TypedAttr userName, OriginType type);
  }];
}

//===----------------------------------------------------------------------===//
// ImplicitOriginRefAttr
//===----------------------------------------------------------------------===//

def ImplicitOriginRefAttr
  : LITTypedAttr<"ImplicitOriginRef", "implicit.origin.ref", [
    DeclareAttrInterfaceMethods<IndexRefAttrInterface>,
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "Reference to an unbound implicit origin parameter.";
  let description = [{
    The `#lit.implicit.origin.ref` attribute is a reference to an unbound
    implicit origin parameter decl of an enclosing signature.

    This attribute can only be used inside a `GeneratorType`.

    This attribute is largely based on `ParamIndexRefAttr`, see
    `ParamIndexRefAttr`'s comments for background.

    It contains two fields:

    - depth: Which containing signature contains the parameter we're referring
      to. Non-negative integer. Zero means the nearest containing signature, one
      means the signature containing that one, etc.
      Note they cannot refer to any op's parameter-decls, and you cannot always
      use a depth to refer to surrounding scopes, see DCRTODS.
    - index: index of the parameter decl in that signature (non-negative
      integer).

    Examples:

    ```mlir
    // Second implicit origin parameter-decl of the nearest signature.
    #lit.implicit.origin.ref<0, 1>
    // First implicit origin parameter-decl of next enclosing signature.
    #lit.implicit.origin.ref<1, 0>
    ```

    The latter would appear in something like this:

    ```
    comptime bar: def[
      X: AnyType,
      f: def[Y: AnyType](X, Y)->None
    ] = ...
    ```

    That `def[Y: AnyType](X, Y)->None`'s mention of `X` is actually a lit.ref
    whose origin is a #lit.implicit.origin.ref<1, 0> to refer to an implicit origin on
    the outermost `def`.

    But, per DCRTODS, it can NOT be used inside a generator's body to refer to
    one of the generator's origin parameters, like this:

    ```
    def foo[X: AnyType](x: X):
        comptime zork: def[...(
          # Cannot have: #lit.implicit.origin.ref<1, 0>
        )->None = ...
    ```

    In this case, it needs to use a ParamDeclRefAttr to refer to an op's
    parameter-decl origin, in this case probably named x`. See DCRTODS for more.

    These depths must be carefully handled and adjusted when dealing with
    multiple signatures or scopes, see STCHDDDOS.
  }];

  let parameters = (ins "size_t":$depth, "size_t":$index,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);
  let assemblyFormat = "`<` $depth `,` $index `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "size_t":$depth, "size_t":$index, "Type":$type), [{
      return get(type.getContext(), depth, index, ::cast<OriginType>(type));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// OriginSubtreeAttr
//===----------------------------------------------------------------------===//

def OriginSubtreeAttr : LITTypedAttr<"OriginSubtree", "origin.subtree", [
          DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
  ]> {
  let summary = "A view over an origin and all origins derived from it";
  let description = [{
    This attribute represents the entire subtree rooted at the given origin.
    A reference whose origin is `#lit.origin.subtree<x>` may refer to storage
    governed by `x` itself or by any origin formed by applying field or interior
    origin projections beneath `x`.
  }];

  let parameters = (ins "TypedAttr":$base,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);

  let assemblyFormat = "`<` $base `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$base), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$base, "OriginType":$type), [{
      return get(base);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginSubtreeAttr getFromBytecode(TypedAttr base, OriginType type);
  }];
}

//===----------------------------------------------------------------------===//
// OriginSetAttr
//===----------------------------------------------------------------------===//

def OriginSetAttr : LITTypedAttr<"OriginSet", "origin.set"> {
  let summary = "a set of origins with different mutabilities";
  let description = [{
    The `#lit.origin.set` attribute contains a set of distinct origins with
    distinct mutabilities. This tracks the individual mutabilities of each
    element and propagates them. It can be converted to a single origin value
    via `#lit.origin.union`.

    Example:

    ```mlir
    #lit.origin.set<{mut a, imm b, (is_mut) c}>
    ```
  }];


  let parameters = (ins ArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"", "OriginSetType">:$type);
  let assemblyFormat = "`<` custom<OriginSet>($operands) `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands, "OriginSetType":$type),
                "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$operands,
                                        "OriginSetType":$type), "", "TypedAttr">,
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands), "", "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginSetAttr getFromBytecode(
        ArrayRef<TypedAttr> operands, OriginSetType type);
  }];
}

//===----------------------------------------------------------------------===//
// OriginSetUnionAttr
//===----------------------------------------------------------------------===//

def OriginSetUnionAttr : LITTypedAttr<"OriginSetUnion",
                                      "origin.set.union"> {
  let summary = "converts a origin set to a single coarse origin";
  let description = [{
    The `#lit.origin.set.union` attribute converts a `!lit.origin.set` value
    to a single coarse-grained origin value by converting passing the elements
    into a `#lit.origin.union`.
  }];

  let parameters = (ins "TypedAttr":$value,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);
  let assemblyFormat = [{
    `<` custom<ParamValue>($value, "OriginSetType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "TypedAttr":$value, "OriginType":$type), [{
      return get(value, type);
    }], "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value,
                                        "OriginType":$type),
    [{}], "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginSetUnionAttr getFromBytecode(
        TypedAttr value, OriginType type);
  }];
}

//===----------------------------------------------------------------------===//
// UnresolvedAliasValueAttr
//===----------------------------------------------------------------------===//

def UnresolvedAliasValueAttr
  : LITTypedAttr<"UnresolvedAliasValue", "unresolved.alias"> {
  let summary = "The value of an 'alias' before it is bound to a value";

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return get(type.getContext(), type);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

def LIT_StructAttr : LITTypedAttr<"LITStruct", "struct", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface>,
    DeclareAttrInterfaceMethods<DeclRefAttrInterface>]> {
  let summary = "A constant struct value";
  let description = [{
    The `#lit.struct` attribute represents a constant struct value. It stores
    the names of the fields and its values.

    Example:

    ```mlir
    #lit.struct<{foo = 5, bar: dtype = dt}> : !lit.struct<@Struct>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"std::tuple<StringAttr, TypedAttr>">:$values,
    AttributeSelfTypeParameter<"", "StructType">:$type);
  let assemblyFormat = "`<` custom<StructElements>($values) `>`";

  // Disable default builders so we can canonicalize on construction.
  let skipDefaultBuilders = 1;

  let builders = [
    // Returns TypedAttr instead of StructAttr because it canonicalizes on
    // construction.
    AttrBuilder<(ins "ArrayRef<std::tuple<StringAttr, TypedAttr>>":$values,
                     "StructType":$type), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins
        "ArrayRef<std::tuple<StringAttr, TypedAttr>>":$values,
        "StructType":$type), [{
      return get(type.getContext(), values, type);
    }], "TypedAttr">
  ];
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

def LIT_StructExtractAttr : LITTypedAttr<"StructExtract", "struct.extract", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
  ]> {
  let summary = "Extract an element from a struct value";
  let description = [{
    The `#lit.struct.extract` attribute represents a field reference from a
    constant struct, which may be parametric.

    Example:

    ```mlir
    #lit.struct.extract<p, "indexfield"> : index
    ```
  }];

  let parameters = (ins "TypedAttr":$structValue, "StringAttr":$field,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat =
    "`<` custom<ColonTypeParamValue>($structValue) `,` $field `>`";

  let skipDefaultBuilders = 1;

  let builders = [
    // Override the default builder used by the generated parser.  This computes
    // the result type correctly and performs inline folding.
    AttrBuilderWithInferredContext<
                (ins "TypedAttr":$structValue, "StructFieldOp":$fieldOp),
                "", "TypedAttr">,
    AttrBuilderWithInferredContext<
                (ins "TypedAttr":$structValue, "StringAttr":$field,
                     "Type":$resultType), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$structValue, "StringAttr":$field,
                     "Type":$resultType), "", "TypedAttr">

  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static StructExtractAttr getFromBytecode(
        TypedAttr structValue, StringAttr field, Type type);
  }];
}

//===----------------------------------------------------------------------===//
// RefPackAttr
//===----------------------------------------------------------------------===//

def LIT_RefPackAttr : LITTypedAttr<"RefPack", "ref.pack", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "heterogeneous list of constant elements";
  let description = [{
    The `#kgen.pack` attribute contains a heterogenously typed list of constant
    references.

    Example:

    ```mlir
    // A pack of 3 elements.
    %0 = kgen.param.constant: !lit.ref.pack<[i8, ui4, i32]> =
         <store_to_mem(3), store_to_mem(1), store_to_mem(4)>>
    // An empty pack.
    %1 = kgen.param.constant: !lit.ref.pack<[]> = <<>>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "RefPackType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<PackElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "RefPackType":$type), [{
      return get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// TypeConventionAttr
//===----------------------------------------------------------------------===//

def LIT_TypeConventionAttr : EnumAttr<LITDialect, LIT_TypeConvention,
    "type_convention">;

//===----------------------------------------------------------------------===//
// VarDeclKindAttr
//===----------------------------------------------------------------------===//

def LIT_VarDeclKindAttr : EnumAttr<LITDialect, LIT_VarDeclKind,
    "var.decl.kind">;

//===----------------------------------------------------------------------===//
// ImplicitConversionKindAttr
//===----------------------------------------------------------------------===//

def LIT_ImplicitConversionKindAttr : EnumAttr<LITDialect,
    LIT_ImplicitConversionKind, "implicit_conversion">;

#endif // KGEN_LITDIALECT_LITATTRS_TD
