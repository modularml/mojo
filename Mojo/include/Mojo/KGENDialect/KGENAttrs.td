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
// This file defines KGEN dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRS_TD
#define KGEN_KGENDIALECT_KGENATTRS_TD

include "Mojo/KGENDialect/KGENAttrInterfaces.td"
include "Mojo/KGENDialect/KGENDialect.td"
include "Mojo/KGENDialect/KGENEnums.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "Support/Compiler/AttrBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

/// An attribute in the KGEN dialect.
class KGENAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<KGENDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// The base class of all parameter expression attributes. Expression attributes
/// are all typed attributes.
class KGEN_ExprAttr<string attrName, string attrMnemonic,
                    list<Trait> traits = []>
    : KGENAttr<attrName, attrMnemonic,
               [TypedAttrInterface,
                DeclareAttrInterfaceMethods<ParameterAttrInterface>] # traits>;

// IndexAttr whose return value is an `int64_t`.
def UnsignedIndexAttr :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[CPred<"isa<IntegerAttr>($_self)">,
           CPred<"isa<IndexType>(cast<IntegerAttr>($_self).getType())">]>,
      "index attribute"> {
  let returnType = "unsigned";
  let convertFromStorage = "$_self.getInt()";
}

//===----------------------------------------------------------------------===//
// ParameterExprArrayAttr
//===----------------------------------------------------------------------===//

def KGEN_ParameterExprArrayAttr
    : ArrayOfAttr<KGENDialect, "ParameterExprArray", "exprs", "TypedAttr">;

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

/// Declaration of a named parameter.
def KGEN_ParamDeclAttr : KGENAttr<"ParamDecl", "param.decl"> {
  let summary = "Parameter declaration";
  let description = [{
    This is a declaration of a parameter in the meta-programming domain for
    generators and related infrastructure.  These are typically owned by
    kgen.generator instances or other things that produce new attributes.
  }];

  /// ParamDeclAttr's always have a `name` and `type`: an omitted type in the
  /// asmsyntax defaults to `index`.
  let parameters = (ins "StringAttr":$name, "Type":$type);

  let assemblyFormat = "custom<ParamName>($name) `:` $type";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "Type":$type),
      "return $_get(type.getContext(), name, type);">,
    AttrBuilderWithInferredContext<(ins "ParamDeclRefAttr":$ref),
      "return $_get(ref.getContext(), ref.getName(), ref.getType());">,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">
  ];
}

//===----------------------------------------------------------------------===//
// ParamDeclArrayAttr
//===----------------------------------------------------------------------===//

/// An array of ParamDeclAttr's typically used on a kgen.generator and other
/// ops that define parameters.
def KGEN_ParamDeclArrayAttr : ArrayOfAttr<KGENDialect, "ParamDeclArray",
    "param.decls", "ParamDeclAttr"> {
  let builders = [
    AttrBuilderWithInferredContext<(ins "ParamDeclAttr":$paramDecl), [{
      return get(paramDecl.getContext(), paramDecl);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ParamListAttr
//===----------------------------------------------------------------------===//

def KGEN_ParamListAttr : KGEN_ExprAttr<"ParamList", "param_list", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A value of !kgen.param_list types made from elements.";
  let description = [{
    The `#kgen.param_list` attribute contains a homogeneous list of elements of
    !kgen.param_list type.

    Example:

    ```mlir
    #kgen.param_list<1, 2> : !kgen.param_list<index>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "ParamListType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<SequenceElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "ParamListType":$type), [{
      return $_get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

def KGEN_ParamListReduceAttr : KGEN_ExprAttr<"ParamListReduce", "param_list.reduce",
  [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                               ["evaluateWithContext"]>]> {
  let summary = "Map a param_list of values to a new param_list of values";
  let description = [{
    The `#kgen.param_list.reduce` attribute is used to reduce a param_list of
    value to a value by repeatedly applying the provided reducer
    on each element of the list.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"">:$type,
    "TypedAttr":$base,
    "TypedAttr":$paramList,
    "TypedAttr":$generator
  );

  let assemblyFormat = "`<` $base `,` $paramList `,` $generator `>`";
  let genVerifyDecl = 1;
}

def KGEN_ParamListTabulateAttr : KGEN_ExprAttr<"ParamListTabulate", "param_list.tabulate",
  [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                               ["evaluateWithContext"]>]> {
  let summary = "Create a param_list by applying a function to each index 0..count-1";
  let description = [{
    The `#kgen.param_list.tabulate` attribute produces a param_list of values
    by invoking the provided generator function N times with indices 0, 1, ...,
    N-1, where N is the integer count. The generator is a function from index to
    value; each result is collected into the result param_list.

    Example:
    ```mlir
    #kgen.param_list.tabulate<:!kgen.param_list<f32> 3, fn(i: index) -> f32> : !kgen.param_list<f32>
    // ->
    #kgen.param_list<0, 1, 2> : !kgen.param_list<f32>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "ParamListType">:$type,
    "TypedAttr":$count,
    "TypedAttr":$generator
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
       "ParamListType":$type, "TypedAttr":$count, "TypedAttr":$generator), "", "TypedAttr">,
    AttrBuilder<(ins "ParamListType":$type, "TypedAttr":$count, "TypedAttr":$generator), [{
      return get(type, count, generator);
    }], "TypedAttr">,
  ];

  let assemblyFormat = "`<` $count `,` $generator `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

def KGEN_ParamListSizeAttr : KGEN_ExprAttr<"ParamListSize", "param_list.size",
    [ContextuallyEvaluatedAttrInterface]> {
  let summary = "return the size of a !kgen.param_list";

  let parameters = (ins "TypedAttr":$paramList);

  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an #kgen.param_list.size.
    AttrBuilderWithInferredContext<(ins
       "TypedAttr":$paramList), "", "TypedAttr">,

    // Verbose form with MLIRContext, needed for ODS constructors.
    AttrBuilder<(ins "IndexType":$type, "TypedAttr":$paramList), [{
      return get(paramList);
    }], "TypedAttr">,
  ];

  let assemblyFormat = "`<` custom<ColonTypeParamValue>($paramList) `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    Type getType() const { return IndexType::get(getContext()); }
  }];
}

def KGEN_ParamListGetAttr : KGEN_ExprAttr<"ParamListGet", "param_list.get",
    [ContextuallyEvaluatedAttrInterface]> {
  let summary = "return the element at index of a param_list";

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type,
    "TypedAttr":$paramList,
    "TypedAttr":$index
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
       "TypedAttr":$paramList, "TypedAttr":$index), "",
       "TypedAttr">,
    AttrBuilder<(ins "Type":$type, "TypedAttr":$paramList, "TypedAttr":$index),
      [{
      return get(paramList, index);
    }], "TypedAttr">,
  ];

  let assemblyFormat = [{
     `<` custom<ColonTypeParamValue>($paramList) `,`
         custom<ParamValue>($index, "IndexType::get($_ctxt)") `>`
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

def KGEN_ParamListConcatAttr : KGEN_ExprAttr<"ParamListConcat", "param_list.concat"> {
  let summary = "Concatenate a param_list of param_list values.";
  let description = [{
    The `#kgen.param_list.concat` attribute is used to concatenate a param_list of
    param_list values.

    Example:
    ```mlir
    #kgen.param_list.concat<[[Int, Int], [Float, Float]]> : !param_list<!AnyType>
    // ->
    #kgen.param_list<[Int, Int, Float, Float]> : !param_list<!AnyType>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "ParamListType">:$type,
    "TypedAttr":$paramLists
  );

  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an #kgen.param_list.concat.
    AttrBuilderWithInferredContext<(ins
       "ParamListType":$type, "TypedAttr":$paramLists), "", "TypedAttr">,

    // Concat the array of paramLists, the array must be non-empty.
    AttrBuilderWithInferredContext<(ins
        "ArrayRef<TypedAttr>":$paramLists), "", "TypedAttr">,

    // Verbose form with MLIRContext, needed for ODS constructors.
    AttrBuilder<(ins "ParamListType":$type, "TypedAttr":$paramLists), [{
      return get(type, paramLists);
    }], "TypedAttr">,
  ];

  let assemblyFormat = "`<` $paramLists `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// UnknownAttr
//===----------------------------------------------------------------------===//

def KGEN_UnknownAttr : KGEN_ExprAttr<"Unknown", "unknown"> {
  let summary = "An unknown parameter value.";
  let description = [{
    The `#kgen.unknown` attribute represents an unknown parameter value. It is
    a special placeholder that represents a parameter that may only be known
    dynamically.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// SingletonAttr
//===----------------------------------------------------------------------===//

def KGEN_SingletonAttr : KGEN_ExprAttr<"Singleton", "singleton"> {
  let summary = "The unique value of a type with a single inhabitant.";
  let description = [{
    The `#kgen.singleton` attribute is the value of a type that has exactly one
    inhabitant, so there is nothing left to represent once the type is known.
    This covers structs with no stored fields such as literal types.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// UnboundAttr
//===----------------------------------------------------------------------===//

def KGEN_UnboundAttr : KGEN_ExprAttr<"Unbound", "unbound"> {
  let summary = "An unbound parameter value.";
  let description = [{
    The `#kgen.unbound` attribute represents an unbound parameter value. It is
    a special placeholder that appears in collections of parameters that have
    been partially bound.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// NoneAttr
//===----------------------------------------------------------------------===//

def KGEN_NoneAttr : KGEN_ExprAttr<"None", "none"> {
  let summary = "A none value";
  let description = [{
    The `#kgen.none` attribute is a value of `!kgen.none` type.
  }];

  let assemblyFormat = "";
  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

/// This attribute models a reference to a named parameter.  The type of the
/// ParamDeclRefAttr must always be the same as the type of the parameter being
/// referenced.
def KGEN_ParamDeclRefAttr : KGEN_ExprAttr<"ParamDeclRef", "param.decl.ref",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "A reference to a parameter value.";
  let description = [{
    The `#kgen.param.decl.ref` attribute is a typed attribute that represents
    a reference to a declared parameter. It contains the type of the referenced
    parameter and its name.

    Example:

    ```mlir
    // A reference to parameter "p" with type "i1".
    #kgen.param.decl.ref<"p"> : i1
    ```

    There are special rules governing when these can appear, or when
    ParamIndexRefAttr must be used instead, see DCRTODS.
  }];

  let parameters = (ins "StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "Type":$type), [{
      return get(name.getContext(), name, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "Type":$type), [{
      return get(StringAttr::get(type.getContext(), name), type);
    }]>,
    AttrBuilderWithInferredContext<(ins "ParamDeclAttr":$decl), [{
      return get(decl.getName(), decl.getType());
    }]>
  ];

  let assemblyFormat = "`<` $name `>`";
}

//===----------------------------------------------------------------------===//
// ParamIndexRefAttr
//===----------------------------------------------------------------------===//

def KGEN_ParamIndexRefAttr : KGEN_ExprAttr<"ParamIndexRef", "param.index.ref",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "A reference to a signature parameter.";
  let description = [{
    The `#kgen.param.index.ref` attribute is a reference to an input
    parameter of an enclosing signature. This attribute can only be used inside
    a `GeneratorType`. This attribute contains two fields:

    - depth: Which containing signature contains the parameter we're referring
      to. Non-negative integer. Zero means the nearest containing signature, one
      means the signature containing that one, etc.
      Note they cannot refer to any op's parameter-decls, and you cannot always
      use a depth to refer to surrounding scopes, see DCRTODS.
    - index: index of the parameter decl in that signature (non-negative
      integer).

    Example:

    ```mlir
    // Second input parameter of the nearest signature.
    #kgen.param.index.ref<0, 1> : index
    // First input parameter of next enclosing signature.
    #kgen.param.index.ref<1, 0> : !lit.struct<@Int>
    ```

    The latter would appear in something like this:

    ```
    comptime bar: def[
      D: DType,
      N: Int,
      f: def[Y: AnyType](Y, SIMD[N, D])->None
    ](...) = ...
    ```

    The `SIMD[N, D]`'s `N` is a #kgen.param.index.ref<1, 0> : !lit.struct<@Int>.

    But, per DCRTODS, it can NOT be used inside a generator's body to refer to
    one of the generator's parameters, like this:

    ```
    def foo[X: AnyType](x: X):
        comptime zork: def[...(
          # Cannot have: #kgen.param.index.ref<1, 0> : !lit.struct<@Int>
        )->None = ...
    ```

    These depths must be carefully handled and adjusted when dealing with
    multiple signatures or scopes, see STCHDDDOS.
  }];

  let parameters = (ins "size_t":$depth,"size_t":$index,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $depth `,` $index `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "size_t":$depth, "size_t":$index,
                                        "Type":$type), [{
      return $_get(type.getContext(), depth, index, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "size_t":$index, "Type":$type), [{
      return $_get(type.getContext(), /*depth=*/0, index, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// QuoteAttr
//===----------------------------------------------------------------------===//

def KGEN_QuoteAttr : KGEN_ExprAttr<"Quote", "quote",
    [DeclareAttrInterfaceMethods<ParameterScopeAttrInterface>]> {
  let summary = "Quote an parameter expression";
  let description = [{
    This attribute "quote" a parameter expression, meaning that it freezes any
    param.index.ref within the expression it quoted. This allows us to pull out
    the parameter expression from its enclosing generator scope without creating
    invalid binding. The major use case for this is for FnTypeGeneratorTypeBuilder,
    which accept lists of "quoted" type expression, and assemble them into an
    actual FnTypeGeneratorType by later "unquoting" them.

    For example:
    ```mlir
    !fn.gen.builder<[#quote<AnyType, #type[*(0, 0)], [], NoneType>] >
    // will be folded into
    !kgen.generator<[AnyType, #type[*(0, 0)]() -> NoneType>
    ```

    Note that `#type[*(0, 0)]` would otherwise hold an invalid reference to (0, 0)
    without a quote, since it is not within a parameter scope.
  }];

  let parameters = (ins "TypedAttr":$quotedParam);

  let assemblyFormat = "`<` custom<TypeParamValue>($quotedParam) `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$quotedParam), [{
      return $_get(quotedParam.getContext(), quotedParam);
    }]>,
  ];

  let extraClassDeclaration = [{
    // TODO: do we need a !quote<type> here?
    Type getType() const { return getQuotedParam().getType(); }

    // A quote is a "deferred scope"
    ArrayRef<Type> getInputParamTypes() { return {};}
  }];
}

//===----------------------------------------------------------------------===//
// TypeParamAttr
//===----------------------------------------------------------------------===//

def KGEN_TypeParamAttr : KGEN_ExprAttr<"TypeParam", "type"> {
  let summary = "Convert an MLIR type to a parameter expression";
  let description = [{
    This represents a parameter whose value is an MLIR type.  It is similar to
    `TypeAttr` in that it is an attribute that refers to a type.  The difference
    is that it is a `TypedAttr` so it can be a parameter expression, and has a
    metatype.

    The `typeValue` field encodes the value-representation of a type, while the
    `mlirType` field encodes the type-representation of the type.

    Example:

    ```mlir
    // Default asm format.
    #kgen.type<!myTypeValue, !myMlirType> : !kgen.type

    // MlirType is omitted if same as typeValue.
    #kgen.type<!myTypeValue> : !kgen.type
    ```
  }];

  let parameters = (ins
    "Type":$typeValue,
    "Type":$mlirType,
    AttributeSelfTypeParameter<"", "Type">:$type
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "MLIRContext *":$ctx, "Type":$typeValue,
                                        "Type":$mlirType, "Type":$type),
                                   "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "Type":$typeValue, "Type":$mlirType,
                                        "Type":$type),
                                   "", "TypedAttr">,
    // Builders for identical typeValue & mlirType cases.
    // Only available temporarily until proper the type/value-domain split.
    AttrBuilderWithInferredContext<(ins "Type":$mlirType, "Type":$type),
                                   "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static TypeParamAttr getFromBytecode(
        Type typeValue, Type mlirType, Type type);

    /// Whether the type constant has identical value & type representation.
    bool hasIdenticalRepresentation();
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// TypeConformToTraitAttr
//===----------------------------------------------------------------------===//

def KGEN_TypeConformToTraitAttr : KGENAttr<"TypeConformsToTrait", "type_conforms_to_trait",
  [TypedAttrInterface,
   DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                               ["evaluateWithContext"]>]> {
  let summary = "check whether the checked types conform to specific traits";
  let description = [{
    This represents a flag to indicate that every type in `typeValue` conforms
    to the specified traits. The stored checked operand is normalized to a
    `param_list<!kgen.type>` value.

    Example:

    ```mlir
    #kgen.type_conforms_to_trait<
        #kgen.param_list<#kgen.param.decl.ref<"T"> : !kgen.type>,
        #kgen.type<typevalue<#kgen.trait_ref<[@Movable, @Copyable]>>, type> : !kgen.type>
    ```

    For the common case of a single checked type value, the operand is printed
    in sugared form: the 1-element `param_list` literal (and any outer upcast
    that simply retypes its element to `!kgen.type`) is stripped.
  }];

  let assemblyFormat =
    "`<` custom<TypeParamValue>($typeValue) `,` custom<TypeParamValue>($traitType) `>`";

  let parameters = (ins
    "TypedAttr":$typeValue,
    "TypedAttr":$traitType
  );

  let builders = [
    /// Build from a single TypedAttr — either a scalar type value or a
    /// param_list of type values — normalizing to the stored
    /// `param_list<!kgen.type>` representation.
    AttrBuilderWithInferredContext<(ins "TypedAttr":$typeValue,
                                        "TypedAttr":$traitType), "", "TypedAttr">,
    // Verbose form with MLIRContext, needed for ODS constructors.
    AttrBuilder<(ins "TypedAttr":$typeValue, "TypedAttr":$traitType), "", "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// Get the type of the attribute. It is always I1 type.
    Type getType() const;

    /// Extract the trait symbol references from the `traitType` value. The
    /// `traitType` wraps a type that implements the `TraitSymbolInterface`;
    /// this returns that type's trait symbols. Returns `std::nullopt` if the
    /// trait value is not yet concrete.
    std::optional<ArrayRef<TraitSymbolAttr>> getTraitSymbols() const;

    /// Fold the attribute given the operations that holds a symbol table for
    /// the trait it implemented, it should either be a `kgen.struct.generator`
    /// or a `lit.struct.decl` which contains symbol table for `conformance_op`.
    FailureOr<TypedAttr> simplify(StructDeclInterface traitTableOp,
                                  ParameterEvaluator &evaluator) const;
  }];

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// IsRefinedTraitAttr
//===----------------------------------------------------------------------===//

def KGEN_IsRefinedType : KGENAttr<"IsRefinedType", "is_refined_type",
  [TypedAttrInterface, ContextuallyEvaluatedAttrInterface]> {
  let summary = "check whether the source type is a more refined type of the target type";
  let description = [{
    This represents a flag to indicate the type, specified by `sourceType`,
    is a more refined type of the other target type, specified by `targetType`.

    It requires both `sourceType` and `targetType` to be at the same type depth.
  }];

  let assemblyFormat = [{
    `<` custom<TypeParamValue>($sourceType) `,`
        custom<TypeParamValue>($targetType) `>`
  }];

  let parameters = (ins
    "TypedAttr":$sourceType,
    "TypedAttr":$targetType
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$sourceType,
                                        "TypedAttr":$targetType), [{
      return $_get(sourceType.getContext(), sourceType, targetType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get the type of the attribute. It is always `!kgen.scalar<bool>`.
    Type getType() const {
      return SIMDType::get(getContext(), 1, KGENDType::kBool);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Upcast/DowncastAttr
//===----------------------------------------------------------------------===//

def KGEN_UpcastAttr : KGEN_ExprAttr<"Upcast", "upcast"> {
  let summary = "Rebind a type with metatype information to base trait.";
  let description = [{
    The `#kgen.upcast` attribute is used to convert from a (param_list of) typeValue
    to a (param_list of) typeValue of a less-derived trait.
    For example, this can represent a cast from Movable to AnyType, handling the
    rebind of the `__del__` member.

    This aggressively canonicalizes, e.g. when the operand is a simple type
    value like a struct, it will return a TypeParamAttr.

    Example:

    ```mlir
    #kgen.upcast<#kgen.param.decl.ref<"T"> : !lit.trait<Movable>> : !lit.trait<AnyType>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type,
    "TypedAttr":$inputTypeValue
  );
  let assemblyFormat = "`<` $inputTypeValue `>`";

  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an UpcastAttr.
    AttrBuilderWithInferredContext<(ins
       "Type":$type, "TypedAttr":$inputTypeValue), "", "TypedAttr">,

    // Verbose form with MLIRContext, needed for ODS constructors.
    AttrBuilder<(ins "Type":$type, "TypedAttr":$inputTypeValue), [{
      return get(type, inputTypeValue);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// If the specified attribute is an upcast, remove it.
    static TypedAttr strip(TypedAttr value) {
      if (auto upcast = llvm::dyn_cast<UpcastAttr>(value))
        return upcast.getInputTypeValue();
      return value;
    }
  }];

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}


def KGEN_DowncastAttr : KGEN_ExprAttr<"Downcast", "downcast",
  [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Rebind a type with metatype information to a new trait.";
  let description = [{
    The `#kgen.downcast` attribute is used to convert from a (param of) typeValue
    to a (param_list of) typeValue of a more-derived trait.
    For example, this can represent a cast from AnyType to Movable.

    Note that parser does not (also can not) verify whether the downcast is
    legal and a illegal downcast can lead to elaboration time error.


    Example:

    ```mlir
    #kgen.downcast<:AnyType T> : !lit.trait<Movable>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type,
    "TypedAttr":$inputTypeValue
  );
  let assemblyFormat = "`<` custom<TypeParamValue>($inputTypeValue) `>`";

  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an DowncastAttr.
    AttrBuilderWithInferredContext<(ins
       "Type":$type, "TypedAttr":$inputTypeValue), "", "TypedAttr">,

    // Verbose form with MLIRContext, needed for ODS constructors.
    AttrBuilder<(ins "Type":$type, "TypedAttr":$inputTypeValue), [{
      return get(type, inputTypeValue);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// If the specified attribute is an downcast, remove it.
    static TypedAttr strip(TypedAttr value) {
      if (auto downcast = llvm::dyn_cast<DowncastAttr>(value))
        return downcast.getInputTypeValue();
      return value;
    }

    /// Extract the type reference if the type value field is already resolved.
    /// This will return either a TypeGeneratorRefAttr / TypeInstanceRefAttr,
    /// or a null attribute if the type value is not resolved.
    TypedAttr getTypeRefIfResolved();
  }];

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}


//===----------------------------------------------------------------------===//
// ExtensionAttr
//===----------------------------------------------------------------------===//

def KGEN_ExtensionAttr : KGEN_ExprAttr<"Extension", "extension"> {
  let summary = "Augment a type value's trait view with extra conformances.";
  let description = [{
    The `#kgen.extension` attribute augments the trait view (metatype) of an
    `anchor` type value with additional trait conformances supplied by a list
    of "extension" struct type values, without changing the anchor's underlying
    physical type.

    Unlike `#kgen.downcast`, which merely re-views a conformance the anchor
    already possesses, an extension supplies the conformance: when
    `#kgen.get_witness` cannot find a trait entry on the anchor's own witness
    tables, it searches the conformance tables of the extension structs. Because
    the anchor's physical type is unchanged, a value typed as the anchor can be
    rebound to the extension type at zero cost.

    Example:

    ```mlir
    #kgen.extension<:!lit.trait<"def() -> T"> G, [!EXT1]>
      : !lit.trait<"def() -> V">
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type,
    "TypedAttr":$anchor,
    ArrayRefParameter<"TypedAttr">:$extensions
  );
  let assemblyFormat =
      "`<` custom<TypeParamValue>($anchor) `,` `[` "
      "custom<TypeParamValues>($extensions) `]` `>`";

  let extraClassDeclaration = [{
    /// If the specified attribute is an extension, return its anchor type
    /// value (the physical/underlying type value); otherwise return the value
    /// unchanged. Witness lookup and type-reference resolution key off the
    /// anchor, so extensions are physically transparent.
    static TypedAttr strip(TypedAttr value) {
      if (auto ext = llvm::dyn_cast<ExtensionAttr>(value))
        return ext.getAnchor();
      return value;
    }
  }];
}

//===----------------------------------------------------------------------===//
// TraitSymbolAttr
//===----------------------------------------------------------------------===//

def KGEN_TraitSymbolAttr : KGENAttr<"TraitSymbol", "trait_symbol"> {
  let summary = "A reference to a trait declaration.";
  let description = [{
    The `#kgen.trait_symbol` attribute names a trait by a reference to its
    declaration.

    TODO: it will be holding an array of parameters for closure traits in the
    future.

    Example:

    ```mlir
    #kgen.trait_symbol<@std::@builtin::@bool::@Boolable>
    ```
  }];

  let parameters = (ins
    "SymbolRefAttr":$symbol,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues
  );

  let assemblyFormat = "`<` $symbol `` custom<ParameterValues>($paramValues) `` `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol), [{
      return $_get(symbol.getContext(), symbol, ArrayRef<TypedAttr>{});
    }]>,

    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol, "ArrayRef<TypedAttr>":$paramValues), [{
      return $_get(symbol.getContext(), symbol, paramValues);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name that keys this trait's conformance tables
    ///
    /// TODO: the flattened name is still used for ASTDecls, what should we use
    /// there for potentially parametric traits?
    StringAttr getFlattenedName() const {
      return StringAttr::get(getContext(), getFlattenedSymbolName(getSymbol()));
    }

    /// Return true if the trait symbol is fully resolved, i.e. all its
    /// parameters are constant.
    bool isFullyResolved() const ;
  }];
}

//===----------------------------------------------------------------------===//
// TraitSymbolArrayAttr
//===----------------------------------------------------------------------===//

def KGEN_TraitSymbolArrayAttr : ArrayOfAttr<KGENDialect, "TraitSymbolArray",
    "trait_symbols", "TraitSymbolAttr"> {
  let summary = "List of trait symbols.";
  let description = [{
    The `#kgen.trait_symbols` attribute represents a list of trait symbols, the
    list is sorted by flattened name.

    Example:

    ```mlir
    #kgen.trait_symbols<[@std::@builtin::@Movable, @std::@builtin::@Copyable]>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// GetWitnessAttr
//===----------------------------------------------------------------------===//

def KGEN_GetWitnessAttr : KGEN_ExprAttr<"GetWitness", "get_witness",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get a witness entry from a witness table.";
  let description = [{
    The `#kgen.get_witness` attribute is used to lookup a witness entry
    from a witness table given a type value and a trait conformance.

    Since type value definitions are symbols, this attribute can only be folded
    when a global symbol table is provided.

    Example:

    ```mlir
    #kgen.get_witness<#Int, "Boolable", "__bool__">
      : !kgen.generator<("self": !Int) -> i1>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    "TraitSymbolAttr":$traitSymbol,
    "StringAttr":$witnessName,
    AttributeSelfTypeParameter<"", "Type">:$type
  );
  let assemblyFormat = [{
    `<` custom<TypeParamValue>($typeValue) `,`
    custom<TraitSymbol>($traitSymbol) `,` $witnessName `>`
  }];

  let builders = [
    // Convenience form with inferred context.
    AttrBuilderWithInferredContext<(ins "TypedAttr":$typeValue,
                                        "TraitSymbolAttr":$traitSymbol,
                                        "StringAttr":$witnessName,
                                        "Type":$type), [{
      return get(type.getContext(), typeValue, traitSymbol, witnessName, type);
    }], "GetWitnessAttr">,

    // Verbose form with MLIRContext, used by the generated parser.
    AttrBuilder<(ins "TypedAttr":$typeValue, "TraitSymbolAttr":$traitSymbol,
                     "StringAttr":$witnessName, "Type":$type), "",
                "GetWitnessAttr">,
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    /// Extract the type reference if the type value field is already resolved.
    /// This will return either a TypeGeneratorRefAttr / TypeInstanceRefAttr,
    /// or a null attribute if the type value is not resolved.
    TypedAttr getTypeRefIfResolved();

    /// Fold the attribute given the symbol table of the struct.
    /// An optional ParameterEvaluator can be provided to concretize the witness
    /// entry for comparison.
    /// Returns failure if the witness entry was not found.
    /// Returns an empty attribute if the evaluator failed to evaluate an
    /// expression.
    FailureOr<TypedAttr> simplify(ConformanceOp witnessTable,
                                  ParameterEvaluator *evaluator = nullptr) const;
  }];
}

//===----------------------------------------------------------------------===//
// CompileOffloadClosureAttr
//===----------------------------------------------------------------------===//

def KGEN_CompileOffloadClosureAttr : KGEN_ExprAttr<"CompileOffloadClosure",
    "compile_offload_closure", [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Compile offload closure for a given target.";
  let description = [{
    The `#kgen.compile_offload_closure` attribute is used to compile offload
    closures for a given target.

    Example:

    ```mlir
    #kgen.compile_offload_closure<
      #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target,
      #kgen.symbol.constant<@kernel> : !kgen.generator<() -> ()>
    > : !kgen.string
    ```
  }];

  let parameters = (ins
    "TypedAttr":$target,
    "TypedAttr":$func,
    AttributeSelfTypeParameter<"", "Type">:$type
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($target, "TargetType::get($_ctxt)") `,` $func `>`
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// GetLinkageNameAttr
//===----------------------------------------------------------------------===//

def KGEN_GetLinkageNameAttr : KGEN_ExprAttr<"GetLinkageName",
    "get_linkage_name", [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Get the linkage name of a function symbol.";
  let description = [{
    The `#kgen.get_linkage_name` attribute is used to get the linkage name of
    a function symbol for a given target.

    Example:

    ```mlir
    #kgen.get_linkage_name<
      #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target,
      #kgen.symbol.constant<@return_one> : !kgen.generator<() -> index>
    > : !kgen.string
    ```
  }];

  let parameters = (ins
    "TypedAttr":$target,
    "TypedAttr":$func,
    AttributeSelfTypeParameter<"", "Type">:$type
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($target, "TargetType::get($_ctxt)") `,` $func `>`
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// LinkageNameAttr
//===----------------------------------------------------------------------===//

def KGEN_LinkageNameAttr : KGEN_ExprAttr<"LinkageName", "linkage_name",
    [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Linkage name with an optional mangle flag.";
  let description = [{
    Holds a name expression (string literal or DataToStr) and a boolean `mangle`
    flag. The flag is stored but not yet acted upon — both `mangle=true` and
    `mangle=false` currently use the prefix verbatim as the symbol name
    (with target-specific name sanitization applied on top for offload
    targets).

    Intended future semantics: when `mangle=true`, the final symbol name will be
    derived from the prefix and a hash of the auto-mangled parameter values,
    guaranteeing uniqueness across instantiations while remaining human-readable
    (e.g. `my_kernel_a3f2c1b0`).
  }];
  let parameters = (ins "TypedAttr":$name, "BoolAttr":$mangle);
  let assemblyFormat = "`<` $name `,` $mangle `>`";
  let extraClassDeclaration = [{
    mlir::Type getType() const {
      return KGEN::StringType::get(getContext());
    }
    /// Create a LinkageNameAttr with a literal string name and an optional
    /// mangle flag.
    static LinkageNameAttr get(mlir::MLIRContext *ctx, llvm::StringRef name,
                               bool mangle = false);
  }];
  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$name, "bool":$mangle), [{
      return $_get(name.getContext(), name,
                   mlir::BoolAttr::get(name.getContext(), mangle));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// GetSourceNameAttr
//===----------------------------------------------------------------------===//

def KGEN_GetSourceNameAttr : KGEN_ExprAttr<"GetSourceName",
    "get_source_name", [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Get the source name of a function symbol.";
  let description = [{
    The `#kgen.get_source_name` attribute is used to get the source name of a
    function symbol.

    Example:

    ```mlir
    #kgen.get_source_name<
      #kgen.symbol.constant<@return_two> : !kgen.generator<() -> index>
    > : !kgen.string
    ```
  }];

  let parameters = (ins
    "TypedAttr":$func,
    AttributeSelfTypeParameter<"", "Type">:$type
  );

  let assemblyFormat = "`<` $func `>`";
}

//===----------------------------------------------------------------------===//
// GetFunctionParameterCountAttr
//===----------------------------------------------------------------------===//

def KGEN_GetFunctionParameterCountAttr : KGEN_ExprAttr<"GetFunctionParameterCount",
    "get_function_parameter_count",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get the compile-time parameter count of a function value.";
  let description = [{
    The `#kgen.get_function_parameter_count` attribute returns the number of
    compile-time parameters declared on a function generator, as an `index`.

    Example:

    ```mlir
    #kgen.get_function_parameter_count<
      #kgen.symbol.constant<@my_func> : !kgen.generator<...>
    > : index
    ```
  }];

  let parameters = (ins
    "TypedAttr":$func,
    AttributeSelfTypeParameter<"", "Type">:$type
  );

  let assemblyFormat = "`<` $func `>`";
}

//===----------------------------------------------------------------------===//
// GetFunctionParameterNamesAttr
//===----------------------------------------------------------------------===//

def KGEN_GetFunctionParameterNamesAttr : KGEN_ExprAttr<"GetFunctionParameterNames",
    "get_function_parameter_names",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get the compile-time parameter names of a function value.";
  let description = [{
    The `#kgen.get_function_parameter_names` attribute returns the names of
    the compile-time parameters declared on a function generator, as a
    `param_list` of strings, in declaration order.

    Example:

    ```mlir
    #kgen.get_function_parameter_names<
      #kgen.symbol.constant<@my_func> : !kgen.generator<...>
    > : !kgen.param_list<!kgen.string>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$func,
    AttributeSelfTypeParameter<"", "ParamListType">:$type
  );

  let assemblyFormat = "`<` $func `>`";
}

//===----------------------------------------------------------------------===//
// GetFunctionIsRaisingAttr
//===----------------------------------------------------------------------===//

def KGEN_GetFunctionIsRaisingAttr : KGEN_ExprAttr<"GetFunctionIsRaising",
    "get_function_is_raising",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Whether a function value can raise an error.";
  let description = [{
    The `#kgen.get_function_is_raising` attribute returns true if the function
    value's signature declares it as raising (Mojo's `raises` keyword, or any
    `def`), and false otherwise.

    Example:

    ```mlir
    #kgen.get_function_is_raising<
      #kgen.symbol.constant<@my_def> : !kgen.generator<...>
    > : i1
    ```
  }];

  let parameters = (ins
    "TypedAttr":$func,
    AttributeSelfTypeParameter<"", "IntegerType">:$type
  );

  let assemblyFormat = "`<` $func `>`";
}

//===----------------------------------------------------------------------===//
// CompileAssemblyAttr
//===----------------------------------------------------------------------===//

def KGEN_CompileAssemblyAttr : KGEN_ExprAttr<"CompileAssembly",
    "compile_assembly", [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Compile assembly for a given target and emission kind.";
  let description = [{
    The `#kgen.compile_assembly` attribute is used to model compiling a function
    to assembly code for a given target and emission format.

    Example:

    ```mlir
    kgen.param.declare some_target: target = #kgen.target<
      triple="", arch="", features="", data_layout="", simd_bit_width=128
    > : !kgen.target

    #kgen.compile_assembly<
      some_target, =llvm, "", false, :() -> () @kernel>
    > : !kgen.string
    ```
  }];

  let parameters = (ins
    "TypedAttr":$target,
    "TypedAttr":$emissionKind,
    "TypedAttr":$emissionOptions,
    "BoolAttr":$propagateError,
    "TypedAttr":$func,
    AttributeSelfTypeParameter<"", "Type">:$type
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($target, "TargetType::get($_ctxt)") `,`
    custom<EmissionKind>($emissionKind) `,`
    custom<ParamValue>($emissionOptions, "StringType::get($_ctxt)") `,`
    $propagateError `,`
    custom<ColonTypeParamValue>($func) `>`
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// GetTypeNameAttr
//===----------------------------------------------------------------------===//

def KGEN_GetTypeNameAttr : KGEN_ExprAttr<"GetTypeName", "get_type_name",
    [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Get the name of a type symbol.";
  let description = [{
    The `#kgen.get_type_name` attribute is used to get the name of a struct
    symbol.

    Example:

    ```mlir
    #kgen.get_type_name<#Int>: !kgen.string
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    "TypedAttr":$qualifiedBuiltins,
    AttributeSelfTypeParameter<"", "Type">:$type
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `,` $qualifiedBuiltins `>`";
}

//===----------------------------------------------------------------------===//
// StructFieldTypesAttr
//===----------------------------------------------------------------------===//

def KGEN_StructFieldTypesAttr : KGEN_ExprAttr<"StructFieldTypes",
    "struct_field_types",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get the field types of a struct as a param_list.";
  let description = [{
    The `#kgen.struct_field_types` attribute returns the types of all fields
    in a struct type as a param_list sequence.

    Example:

    ```mlir
    #kgen.struct_field_types<#MyStruct> : !kgen.param_list<!kgen.type>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    AttributeSelfTypeParameter<"", "ParamListType">:$type
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `>`";
}

//===----------------------------------------------------------------------===//
// StructFieldNamesAttr
//===----------------------------------------------------------------------===//

def KGEN_StructFieldNamesAttr : KGEN_ExprAttr<"StructFieldNames",
    "struct_field_names",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get the field names of a struct as a param_list of strings.";
  let description = [{
    The `#kgen.struct_field_names` attribute returns the names of all fields
    in a struct type as a param_list sequence of strings.

    Example:

    ```mlir
    #kgen.struct_field_names<#MyStruct> : !kgen.param_list<!kgen.string>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    AttributeSelfTypeParameter<"", "ParamListType">:$type
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `>`";
}

//===----------------------------------------------------------------------===//
// StructFieldIndexByNameAttr
//===----------------------------------------------------------------------===//

def KGEN_StructFieldIndexByNameAttr : KGEN_ExprAttr<"StructFieldIndexByName",
    "struct_field_index_by_name",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get the index of a struct field by name.";
  let description = [{
    The `#kgen.struct_field_index_by_name` attribute returns the index of a field
    in a struct type given the field name. Produces a compile error if the
    field name does not exist in the struct.

    The fieldName parameter should resolve to a StringAttr (kgen.string) after
    parameter evaluation.

    Example:

    ```mlir
    #kgen.struct_field_index_by_name<#MyStruct, "x"> : index
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    "TypedAttr":$fieldName,
    AttributeSelfTypeParameter<"", "IndexType">:$type
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `,` $fieldName `>`";
}

//===----------------------------------------------------------------------===//
// StructFieldTypeByNameAttr
//===----------------------------------------------------------------------===//

def KGEN_StructFieldTypeByNameAttr : KGEN_ExprAttr<"StructFieldTypeByName",
    "struct_field_type_by_name",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get the type of a struct field by name.";
  let description = [{
    The `#kgen.struct_field_type_by_name` attribute returns the type of a field
    in a struct given the field name. Produces a compile error if the field
    name does not exist in the struct.

    The fieldName parameter should resolve to a StringAttr (kgen.string) after
    parameter evaluation.

    Example:

    ```mlir
    #kgen.struct_field_type_by_name<#MyStruct, "x"> : !AnyType
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    "TypedAttr":$fieldName,
    AttributeSelfTypeParameter<"", "Type">:$type
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `,` $fieldName `>`";
}

//===----------------------------------------------------------------------===//
// StructFieldOffsetByIndexAttr
//===----------------------------------------------------------------------===//

def KGEN_StructFieldOffsetByIndexAttr : KGEN_ExprAttr<"StructFieldOffsetByIndex",
    "struct_field_offset_by_index",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get the byte offset of a struct field by index.";
  let description = [{
    The `#kgen.struct_field_offset_by_index` attribute returns the byte offset
    of a field in a struct type given the field index. Produces a compile error
    if the field index is out of bounds.

    The offset is computed using the target's data layout to determine field
    sizes and alignment requirements.

    Example:

    ```mlir
    #kgen.struct_field_offset_by_index<#MyStruct, 0 : index, #target> : index
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    "TypedAttr":$fieldIndex,
    "TypedAttr":$target
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `,` $fieldIndex `,` $target `>`";
  let extraClassDeclaration = [{ Type getType() const; }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// StructFieldOffsetByNameAttr
//===----------------------------------------------------------------------===//

def KGEN_StructFieldOffsetByNameAttr : KGEN_ExprAttr<"StructFieldOffsetByName",
    "struct_field_offset_by_name",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let summary = "Get the byte offset of a struct field by name.";
  let description = [{
    The `#kgen.struct_field_offset_by_name` attribute returns the byte offset
    of a field in a struct type given the field name. Produces a compile error
    if the field name does not exist in the struct.

    The fieldName parameter should resolve to a StringAttr (kgen.string) after
    parameter evaluation.

    The offset is computed using the target's data layout to determine field
    sizes and alignment requirements.

    Example:

    ```mlir
    #kgen.struct_field_offset_by_name<#MyStruct, "x", #target> : index
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    "TypedAttr":$fieldName,
    "TypedAttr":$target
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `,` $fieldName `,` $target `>`";
  let extraClassDeclaration = [{ Type getType() const; }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// IsStructTypeAttr
//===----------------------------------------------------------------------===//

def KGEN_IsStructTypeAttr : KGENAttr<"IsStructType", "is_struct_type",
    [TypedAttrInterface, ContextuallyEvaluatedAttrInterface]> {
  let summary = "Check if a type value refers to a struct type.";
  let description = [{
    The `#kgen.is_struct_type` attribute returns true if the given type value
    refers to a Mojo struct type, false otherwise. This is useful for guarding
    reflection code that needs to avoid compiler errors on non-struct types.

    Example:

    ```mlir
    #kgen.is_struct_type<#MyStruct> : i1
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    AttributeSelfTypeParameter<"", "IntegerType">:$type
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `>`";

  let extraClassDeclaration = [{
    /// Get the type of the attribute. It is always i1 type.
    static IntegerType getResultType(MLIRContext *ctx) {
      return IntegerType::get(ctx, 1);
    }
  }];
}

//===----------------------------------------------------------------------===//
// FnTypeIsCABIAttr
//===----------------------------------------------------------------------===//

def KGEN_FnTypeIsCABIAttr : KGENAttr<"FnTypeIsCABI", "fn_type_is_cabi",
    [TypedAttrInterface, ContextuallyEvaluatedAttrInterface]> {
  let summary = "Check if a type value refers to a function type with abi(\"C\").";
  let description = [{
    The `#kgen.fn_type_is_cabi` attribute returns true if the given type value
    refers to a Mojo function pointer type annotated with the `abi("C")` effect,
    and false otherwise (including for non-function types).

    This is used to enforce that `DLHandle.get_function` is always called with
    an explicit `abi("C")` function pointer type, ensuring dynamically-loaded
    symbols are called with the correct C ABI.

    Example:

    ```mlir
    #kgen.fn_type_is_cabi<#kgen.type<!kgen.generator<(f64) cabi -> f64>>> : i1
    // evaluates to true

    #kgen.fn_type_is_cabi<#kgen.type<!kgen.generator<(f64) -> f64>>> : i1
    // evaluates to false
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    AttributeSelfTypeParameter<"", "IntegerType">:$type
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `>`";

  let extraClassDeclaration = [{
    static IntegerType getResultType(MLIRContext *ctx) {
      return IntegerType::get(ctx, 1);
    }

    /// Build a FnTypeIsCABIAttr from a raw type value attribute.
    static FnTypeIsCABIAttr get(MLIRContext *ctx, TypedAttr typeValue);
  }];
}

//===----------------------------------------------------------------------===//
// GetBaseTypeNameAttr
//===----------------------------------------------------------------------===//

def KGEN_GetBaseTypeNameAttr : KGEN_ExprAttr<"GetBaseTypeName", "get_base_type_name",
    [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Get the unqualified name of a type's base type.";
  let description = [{
    The `#kgen.get_base_type_name` attribute extracts the unqualified name of
    the base (unparameterized) type from a parameterized type. For example,
    given `List[Int]`, it returns the string `"List"`. For non-parameterized
    types, it returns the type's simple name.

    This is useful for reflection-based code that needs to identify the base
    type of parameterized types.

    Example:

    ```mlir
    #kgen.get_base_type_name<#List[Int]> : !kgen.string
    // Returns "List"
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue
  );

  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `>`";

  let extraClassDeclaration = [{
    /// Get the type of the attribute. It is always StringType.
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// BindParams
//===----------------------------------------------------------------------===//

def KGEN_BindParamsAttr : KGEN_ExprAttr<"BindParams", "bind_params"> {
  let summary = "Bind params onto an expression with generator type.";
  let description = [{
    The `#kgen.bind_params` attribute is used to bind parameters onto a
    generator parameter (a parameter whose type is some GeneratorType).

    Partial & out-of-order parameter binding is allowed by specifying the
    unbound positions with UnboundAttr.

    Example:

    Given
    ```mlir
      @SIMD : !kgen.generator<<dtype, index> AnyStruct<@SIMD<*(0,0), *(0,1)>>>
    ```

    This attr binds the index parameter but not the dtype
    ```mlir
    #kgen.bind_param<@SIMD<?, 2>>
      : !kgen.generator<<dtype> AnyStruct<@SIMD<*(0,0), 2>>>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$generator,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    OptionalParameter<"DenseBoolArrayAttr">:$discharged
  );
  let assemblyFormat =
      "`<` custom<BindParams>($generator, $paramValues, $discharged) `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Optionally takes an evaluation context to use when simplifying/folding.
    // The evaluation context is never stored, only used temporarily prior to
    // construction.
    AttrBuilder<(ins
      "TypedAttr":$generator,
      "ArrayRef<TypedAttr>":$paramValues,
      "DenseBoolArrayAttr":$discharged,
      CArg<"ParameterEvaluationContext *", "nullptr">:$evaluationContext
    ), "", "TypedAttr">,
    AttrBuilder<(ins
      "TypedAttr":$generator,
      "ArrayRef<TypedAttr>":$paramValues,
      CArg<"ParameterEvaluationContext *", "nullptr">:$evaluationContext
    ), "", "TypedAttr">
  ];
  let extraClassDeclaration = [{
    /// Derive the result type from the generator and parameter values.
    /// Parameter values are normalized to the generator's input arity: missing
    /// trailing values are treated as unbound and extra trailing values are
    /// ignored. `discharged` is an optional bitmask over the generator's body
    /// constraints: a set bit drops the matching constraint from the result
    /// type. Null (the default) retains all body constraints.
    static Type inferResultType(
        TypedAttr generator, ArrayRef<TypedAttr> paramValues,
        DenseBoolArrayAttr discharged,
        ParameterEvaluationContext *evaluationContext);
    static Type inferResultType(
        TypedAttr generator, ArrayRef<TypedAttr> paramValues,
        ParameterEvaluationContext *evaluationContext);
    Type getType() const;

    /// A specific builder used to build the attribute from a bytecode context,
    /// bypassing simplification.
    static BindParamsAttr getFromBytecode(
        TypedAttr generator, ArrayRef<TypedAttr> paramValues,
        DenseBoolArrayAttr discharged = {});
  }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// TypeGeneratorRefAttr
//===----------------------------------------------------------------------===//

def KGEN_TypeGeneratorRefAttr : KGEN_ExprAttr<"TypeGeneratorRef", "genref", [
    ContextuallyEvaluatedAttrInterface,
    DeclareAttrInterfaceMethods<DeclRefAttrInterface>]> {
  let summary = "type generator reference";
  let description = [{
    This is a symbolic reference to a type-value generator. Its type is the
    metatype of the type-value. If the type-value is parametric, additional
    parameter values may be bound.

    TODO: Merge SymbolConstantAttr into this.
  }];

  let parameters = (ins "SymbolRefAttr":$symbol,
                        OptionalArrayRefParameter<"TypedAttr">:$paramValues,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "Type":$type), [{
      return get(type.getContext(), symbol, {}, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "ArrayRef<TypedAttr>":$paramValues,
                                        "Type":$type), [{
      return get(type.getContext(), symbol, paramValues, type);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $symbol `` custom<ParameterValues>($paramValues) `` `>`
  }];
}

//===----------------------------------------------------------------------===//
// TypeInstanceRefAttr
//===----------------------------------------------------------------------===//

def KGEN_TypeInstanceRefAttr : KGEN_ExprAttr<"TypeInstanceRef", "instref"> {
  let summary = "type instance reference";
  let description = [{
    This is a symbolic reference to a concrete type-value instance. Its type
    is the metatype of the type-value.
  }];

  let parameters = (ins "SymbolRefAttr":$symbol,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "Type":$type), [{
      return get(type.getContext(), symbol, type);
    }]>
  ];
  let assemblyFormat = "`<` $symbol `>`";
}

//===----------------------------------------------------------------------===//
// TypeInstanceRefAttr
//===----------------------------------------------------------------------===//

def KGEN_TraitInstanceRefAttr : KGEN_ExprAttr<"TraitInstanceRef", "trait_ref"> {
  let summary = "trait instance reference";
  let description = [{
    This is a symbolic reference to a trait instance. Its type is the metatype
    of the trait.
  }];

  let parameters = (ins ArrayRefParameter<"TraitSymbolAttr">:$symbols,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TraitSymbolAttr>":$symbols,
                                        "Type":$type), [{
      return get(type.getContext(), symbols, type);
    }]>
  ];
  // Print the traits as bare symbols rather than as nested attributes.
  let assemblyFormat = "`<` custom<TraitSymbols>($symbols) `>`";
}


//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

/// This is a value of !kgen.dtype with a constant value from 0...255.
def KGEN_DTypeConstantAttr : KGEN_ExprAttr<"DTypeConstant", "dtype.constant",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "Constant !kgen.dtype value";
  let description = [{
    This is constant value for a dtype, whose elements correspond to DType.
  }];

  /// The value of a DTypeConstantAttr is always signless i8 constant value
  /// and the type is always !kgen.dtype.
  let parameters = (ins "KGENDType":$dType);

  let assemblyFormat = "`<` $dType `>`";

  let extraClassDeclaration = [{
    /// Check if the DType is convertible to the MLIR type.
    bool isConvertibleTo(Type type);
    /// Check if the MLIR type can be converted from the DType.
    bool isConvertibleFrom(Type type);
    /// Get the type of the attribute. It is always DTypeType.
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// SymbolConstantAttr
//===----------------------------------------------------------------------===//

def KGEN_SymbolConstantAttr : KGEN_ExprAttr<"SymbolConstant", "symbol.constant",
    [CallableSymbolAttrInterface,
     DeclareAttrInterfaceMethods<DeclRefAttrInterface>]> {
  let summary = "kgen.func or kgen.generator reference";
  let description = [{
    This is a value of FuncTypeGenerator type, which refers to a func or
    generator.  This may optionally bind the input parameter values at time of
    formation - when this happens, the result type is non-parametric.
  }];

  let parameters = (ins
    "SymbolRefAttr":$symbol,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    AttributeSelfTypeParameter<"", "FuncTypeGeneratorType">:$type
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "SymbolRefAttr":$symbol, "FuncTypeGeneratorType":$type,
        CArg<"ArrayRef<TypedAttr>", "{}">:$paramValues), [{
      return get(type.getContext(), symbol, paramValues, type);
    }]>,
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name, "FuncTypeGeneratorType":$type,
        CArg<"ArrayRef<TypedAttr>", "{}">:$paramValues), [{
      return get(FlatSymbolRefAttr::get(name), type, paramValues);
    }]>,
    AttrBuilderWithInferredContext<(ins
        "GeneratorOp":$func, "FuncTypeGeneratorType":$type,
        CArg<"ArrayRef<TypedAttr>", "{}">:$paramValues), [{
      return get(func.getSymNameAttr(), type, paramValues);
    }]>,
    AttrBuilderWithInferredContext<(ins "GeneratorOp":$func), [{
      return get(func.getSymNameAttr(), func.getFuncTypeGenerator());
    }]>,
    AttrBuilderWithInferredContext<(ins "FuncOp":$func), [{
      return get(func.getSymNameAttr(), func.getFuncTypeGenerator());
    }]>,
  ];

  let assemblyFormat = [{
    `<` $symbol `` custom<ParameterValues>($paramValues) `` `>`
  }];
}

//===----------------------------------------------------------------------===//
// FuncPtrBitcastAttr
//===----------------------------------------------------------------------===//

def KGEN_FuncPtrBitcastAttr : KGEN_ExprAttr<"FuncPtrBitcast", "func_ptr_bitcast",
    [CallableSymbolAttrInterface]> {
  let summary = "reinterpret a func/generator reference to another signature";
  let description = [{
    The `#kgen.func_ptr_bitcast` attribute is the compile-time analogue of a
    `pop.pointer.bitcast` applied to a function pointer. It wraps a
    `SymbolConstantAttr` -- which keeps the callee's true, declared signature --
    and re-presents it under a different `FuncTypeGeneratorType`.

    Example:

    ```mlir
    #kgen.func_ptr_bitcast<#kgen.symbol.constant<@bar> : !kgen.generator<(i64) -> none>>
      : !kgen.generator<(...) -> none>
    ```
  }];

  let parameters = (ins
    "SymbolConstantAttr":$callee,
    AttributeSelfTypeParameter<"", "FuncTypeGeneratorType">:$type
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "SymbolConstantAttr":$callee, "FuncTypeGeneratorType":$type), [{
      return get(type.getContext(), callee, type);
    }]>,
  ];

  let assemblyFormat = "`<` custom<FuncPtrBitcastCallee>($callee) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Get the wrapped reference at its true (deep) signature.
    SymbolConstantAttr getSymbolConstant() const { return getCallee(); }

    SymbolRefAttr getSymbol() const { return getSymbolConstant().getSymbol(); }
    ArrayRef<TypedAttr> getParamValues() const {
      return getSymbolConstant().getParamValues();
    }
  }];
}

//===----------------------------------------------------------------------===//
// GeneratorAttr
//===----------------------------------------------------------------------===//

def KGEN_GeneratorAttr : KGEN_ExprAttr<"Generator", "gen", [
    DeclareAttrInterfaceMethods<ParameterScopeAttrInterface>,
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "generator constant attribute";
  let description = [{
    This is a generator constant attribute that represents a generator whose
    body is a parameter expression. The GeneratorAttr natively encodes the input
    parameter types and metadata, and computes the overall type on demand. This
    encoding ensures that the type and the value of the body are always at the
    same level of nesting. If we instead stored a GeneratorType in this
    attribute, the body of the GeneratorType would be at a deeper level of
    nesting than the body of the GeneratorAttr, leading to inconsistencies.

    Example:

    ```mlir
    #kgen.gen<*(0,0) + 1> : !kgen.generator<<index> index>
    ```
  }];

  let parameters = (ins "TypedAttr":$body,
                        ArrayRefParameter<"Type">:$inputParamTypes,
                        "PogListAttr":$metadata);

  let builders = [
    AttrBuilderWithInferredContext<
        (ins "ArrayRef<Type>":$inputParamTypes,
             "TypedAttr":$body,
             CArg<"Attribute", "{}">:$metadata), [{
      return get(body.getContext(), body, inputParamTypes,
                 ::cast_or_null<PogListAttr>(metadata));
    }]>
  ];
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the type of this `GeneratorAttr` (computed on demand).
    ///
    /// Defined out-of-line in `KGENAttrs.cpp` because it constructs a
    /// `GeneratorType`, which is generated in a separate header.
    Type getType() const;

    /// Get this GeneratorAttr with some parameters bound.
    GeneratorAttr getSpecializedGenerator(
      PartiallySpecializedInputParams &specialization,
      function_ref<InFlightDiagnostic()> emitErrorFn = {});
    GeneratorAttr getSpecializedGenerator(
      ArrayRef<TypedAttr> paramBindings,
      ParameterEvaluationContext *evaluationContext,
      function_ref<InFlightDiagnostic()> emitErrorFn = {});
    GeneratorAttr getSpecializedGenerator(
      ArrayRef<TypedAttr> paramBindings,
      ParameterEvaluationContext *evaluationContext, Location location);

    /// Returns true if the signature has no input params.
    bool isFullyBound() { return getInputParamTypes().empty(); };

    /// Returns the instantiated body value while asserting that the generator
    /// is fully bound. Index references inside the body are decremented to
    /// account for the fact that the body is now lifted outside one level of
    /// generator scope. See STCHDDDOS for more. This is STCHDDDOS-A.
    TypedAttr getInstantiatedValue();
  }];
}

//===----------------------------------------------------------------------===//
// TargetParamAttr
//===----------------------------------------------------------------------===//

def KGEN_TargetParamAttr : KGEN_ExprAttr<"TargetParam", "target"> {
  let summary = "Target configuration parameter attribute.";
  let description = [{
    The `#kgen.target` is an attribute representing a target of type
    `!kgen.target`. It contains the target configuration information.

    Example:

    ```mlir
    #kgen.target<triple="triple", cpu="cpu", features="features",
                 data_layout="p:32:32", simd_bit_width=128>
    ```

    Target types can be manipulated using target operators.

    Example:

    ```mlir
    kgen.generator @target_host<t0: target>()
        constraints <[eq(:target
            #kgen.target<triple="triple", cpu="cpu", features="features",
                         data_layout="", simd_bit_width=128>,
            t0),
          "Must support the target!"
        ]> {
      kgen.return
    }
    ```
  }];
  let parameters = (ins "TargetInfoAttr":$target);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TargetInfoAttr":$target), [{
      return $_get(target.getContext(), target);
    }]>
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// StructDefFieldAttr
//===----------------------------------------------------------------------===//

def KGEN_StructDefFieldAttr : KGENAttr<"StructDefField", "struct_def.field"> {
  let summary = "A single source struct field declaration.";
  let description = [{
    The `#kgen.struct_def.field` attribute represents a field declared in a
    mojo struct. It keeps track of the name and type of the field.

    Example:

    ```mlir
    #kgen.struct_def.field<"num" : Int>
    ```
  }];

  let parameters = (ins "StringAttr":$name, "TypedAttr":$typeValue);
  let assemblyFormat = " `<` $name `:` $typeValue `>` ";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        "TypedAttr":$typeValue), [{
      return get(name.getContext(), name, typeValue);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

def KGEN_StructAttr : KGENAttr<"Struct", "struct", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A heterogeneous list of elements.";
  let description = [{
    The `#kgen.struct` attribute contains a heterogenous list of elements of
    struct type. It is used to represent constant struct values.

    Example:

    ```mlir
    #kgen.struct<3, 3.5> : !kgen.struct<(index, f32)>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "StructType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<StructElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values)>,
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "StructType":$type), [{
      return get(type.getContext(), values, type);
    }]>,
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

def KGEN_StructExtractAttr : KGENAttr<"StructExtract", "struct.extract",
    [TypedAttrInterface]> {
  let summary = "Extract an element from a struct value";
  let description = [{
    The `#kgen.struct.extract` attribute represents a field reference from a
    constant struct, which may be parametric.

    Example:

    ```mlir
    #kgen.struct.extract<p, 4> : index
    ```
  }];

  let parameters = (ins "TypedAttr":$structValue, "TypedAttr":$fieldNo,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = [{
    `<` custom<ColonTypeParamValue>($structValue) `,`
        custom<IndexParamValue>($fieldNo) `>`
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    // Override the default builder used by the generated parser.  This computes
    // the result type correctly and performs inline folding.
    AttrBuilderWithInferredContext<
                (ins "TypedAttr":$structValue, "unsigned":$fieldNo),
                     "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$structValue,
                                        "TypedAttr":$fieldNo,
                                        "Type":$resultType), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$structValue, "TypedAttr":$fieldNo,
                     "Type":$resultType), [{
      return get(structValue, fieldNo, resultType);
    }], "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a different context from the normal builder.
    static StructExtractAttr getFromBytecode(
        TypedAttr structValue, TypedAttr fieldIdx, Type resultType);
  }];
}

//===----------------------------------------------------------------------===//
// VariantAttr
//===----------------------------------------------------------------------===//

def KGEN_VariantAttr : KGENAttr<"Variant", "variant", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A variant constant value.";
  let description = [{
    The `#kgen.variant` attribute represents a constant `!kgen.variant` value.
    It contains a single typed attribute where the typed of the value is
    expected to equal one of the corresponding variant types.

    Example:

    ```mlir
    #kgen.variant<3 : index> : !kgen.variant<index, i32, f32>
    ```
  }];

  let parameters = (ins "TypedAttr":$value, "unsigned":$index,
                        AttributeSelfTypeParameter<"", "VariantType">:$type);

  let assemblyFormat = "`<` custom<ColonTypeParamValue>($value) `,` $index `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "unsigned":$index,
                                        "VariantType":$type), [{
      return get(type.getContext(), value, index, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// EnvAttr
//===----------------------------------------------------------------------===//

def KGEN_EnvAttr : KGENAttr<"Env", "env"> {
  let summary = "Environment variable attribute.";
  let description = [{
    The `#kgen.env` attribute defines a generic dictionary of environment
    parameters that can be accessed through parameter operators. The values
    contained can be:
    - integers, represented with the `index` type
    - strings, represented as `!kgen.string` types
    - unit attributes, the presence of which indicates something

    Note that EnvAttr does not support storing BoolAttr in it.
    Instead, a boolean true is represented as a UnitAttr, and boolean false
    is not represented at all (absence of a value evaluates to false).

    Example:

    ```mlir
    #kgen<env{intVal = 1 : index, unitAttr, strVal = "hello" : !kgen.string}>
    ```
  }];

  let parameters = (ins "DictionaryAttr":$values);
  let assemblyFormat = "`<` $values `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "DictionaryAttr":$values), [{
      return $_get(values.getContext(), values);
    }]>
  ];

  let extraClassDeclaration = [{
    /// This is the attribute name of the `#kgen.env` attribute attached to the
    /// surrounding `ModuleOp`.
    static llvm::StringLiteral getEnvAttrName() { return "kgen.env"; }

    /// Parse a defines list and create the environment from that.
    static ErrorOr<EnvAttr> parseDefines(MLIRContext *ctx,
                                         ArrayRef<std::string> defines);

    /// Query an environment value by name and convert it to the requested output type.
    /// Supports conversion between int/bool/string/unit types as needed.
    /// Returns the converted TypedAttr on success, or an Error describing what went wrong.
    ErrorOr<TypedAttr> queryValue(StringRef name, Type outputType) const;

    /// Extends the EnvAttr with another one. If the keys conflict, then the
    /// extended value would take precedence.
    EnvAttr extend(EnvAttr attr);
  }];
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

// ParamOperatorAttr forms the core of the expression grammar for parameter
// expressions.  Its operators have been designed so that expression building
// force a deterministic canonical form on expression trees.  This ensures that
// `(x+1)*4` and `(x*4)+4` canonicalize into the same pointer value.
//
// For this reason, we should never have unary negate or subtract. `(x-y)` is
// modeled as `(x+-y)` => `(x + -1*y)`.  The ParamOperatorAttr class does
// provide convenience methods for these aggregate operations.
//
// The design of these operators assumes signful types like si64, and the fully
// associative operations are take arbitrary arity, also to make
// canonicalization simpler.
def KGEN_ParamOperatorAttr : KGEN_ExprAttr<"ParamOperator", "param.expr", [
    DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                               ["evaluateWithContext"]>,
    DeclareAttrInterfaceMethods<ParameterAttrInterface,
                                ["isLessThan", "validateForElaborator"]>
  ]> {
  let summary = "Parameter expression";
  let parameters = (ins "POC":$opcode,
                        OptionalArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"">:$type);

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "POC":$opcode, "ArrayRef<TypedAttr>":$operands,
                     "Type":$type), "", "TypedAttr">,

    // Build a parameter expression.  This automatically canonicalizes and
    // folds, so it may not necessarily return a ParamOperatorAttr.
    AttrBuilderWithInferredContext<(ins
        "POC":$opcode, "ArrayRef<TypedAttr>":$operands), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "POC":$opcode,
        "ArrayRef<TypedAttr>":$operands, "Type":$type), [{
      return get(type.getContext(), opcode, operands, type);
    }], "TypedAttr">,

    // Build a binary parameter expression for convenience.
    AttrBuilderWithInferredContext<(ins "POC":$opcode, "TypedAttr":$lhs,
                                        "TypedAttr":$rhs), [{
      TypedAttr operands[] = { lhs, rhs };
      return get(opcode, operands);
    }], "TypedAttr">
  ];

  let assemblyFormat = "`<` $opcode (`,` $operands^)? `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    size_t getNumOperands() const {
      return getOperands().size();
    }
    TypedAttr getOperand(size_t i) const {
      return getOperands()[i];
    }

    /// Return (ne x, y) which is the same as (not (eq x, y)).
    static TypedAttr getNE(TypedAttr lhs, TypedAttr rhs) {
      return getNot(get(POC::EQ, lhs, rhs));
    }

    /// Return (not x) which is the same as (xor x, true).  The `operand` value
    /// must have type `i1`.
    static TypedAttr getNot(TypedAttr operand);

    /// Return (neg x) which is the same as (mul x, -1).  The `operand` value
    /// must have `index` type.
    static TypedAttr getNeg(TypedAttr operand);

    /// Return (x-y) which is the same as (add x, (neg y)).  The `operand` value
    /// must have `index` type.
    static TypedAttr getSub(TypedAttr lhs, TypedAttr rhs);

    /// Helper to build a rebind operation.
    static TypedAttr getRebind(TypedAttr operand, Type resultType) {
      if (operand.getType() == resultType)
        return operand;
      return get(POC::Rebind, {operand}, resultType);
    }

    /// If the specified attribute is a rebind, return its operand, otherwise
    /// return the rebind itself.
    static TypedAttr stripRebind(TypedAttr src);

    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static ParamOperatorAttr getFromBytecode(
        POC opcode, ArrayRef<TypedAttr> operands, Type type);
  }];
}

//===----------------------------------------------------------------------===//
// ParamIdenticalAttr
//===----------------------------------------------------------------------===//

def KGEN_ParamIdenticalAttr
    : KGEN_ExprAttr<"ParamIdentical", "param.identical", [
    DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                ["evaluateWithContext"]>,
    DeclareAttrInterfaceMethods<ParameterAttrInterface,
                                ["validateForElaborator"]>
  ]> {
  let summary = "Parameter identity proposition";
  let description = [{
    The `#kgen.param.identical` attribute is the proposition that its operands
    all denote the same parameter value (as considered post-elaboration).
    Its result is always `!kgen.scalar<bool>`.

    Important: This is categorically different from `POC::EQ`, which is a
    *lane-wise numeric* comparison whose result inherits the operand lane count.

    Example:

    ```mlir
    #kgen.param.identical<#kgen.param.decl.ref<"T"> : !kgen.type,
                          #kgen.param.decl.ref<"U"> : !kgen.type>
    ```

    A class holds at least two operands, canonically ordered.
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$operands);

  let skipDefaultBuilders = 1;
  let builders = [
    // Build the proposition. This canonicalizes and folds, so it may not
    // return a ParamIdenticalAttr.
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$operands), "",
                                   "TypedAttr">,

    // Verbose form with MLIRContext, needed for the ODS constructors used by
    // the generated parser.
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands), "", "TypedAttr">,

    // Build a binary proposition for convenience.
    AttrBuilderWithInferredContext<(ins "TypedAttr":$lhs, "TypedAttr":$rhs), "",
                                   "TypedAttr">
  ];

  let assemblyFormat = "`<` $operands `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// The result is always `!kgen.scalar<bool>`.
    Type getType() const;

    size_t getNumOperands() const { return getOperands().size(); }
    TypedAttr getOperand(size_t i) const { return getOperands()[i]; }
  }];
}

//===----------------------------------------------------------------------===//
// MLIROpAttr
//===----------------------------------------------------------------------===//

def KGEN_MLIROpAttr : KGEN_ExprAttr<"MLIROp", "param.mlir_op"> {
  let summary = "an MLIR operation as a parameter expression";
  let description = [{
    The `#kgen.param.mlir_op` attribute represents an MLIR operation as a
    parameter expression. Its type is FuncTypeGeneratorType.

    Example:

    ```
    #kgen.param.mlir_op<"index.add", {}>
      : !kgen.generator<(index, index) -> index>
    ```

    Operation attributes can be specified using a dictionary attribute.

    Example:

    ```
    #kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate slt>}>
      : !kgen.generator<(index, index) -> i1>
    ```

    The operation can be parameterized on any of its attributes that are
    parametric -- that is, which are `TypedAttr`. These attributes are omitted
    from the attribute dictionary and are present in the signature.

    Example:

    ```
    #kgen.param.mlir_op<"pop.array.get", {}> : !kgen.generator<
      <*"index">(!pop.array<size, type>) -> !kgen.param<type>
    >
    ```

    Parametric operations can be bound using `bind_signature`.
  }];

  let parameters = (ins
    "StringAttr":$name,
    "DictionaryAttr":$attrs,
    AttributeSelfTypeParameter<"", "FuncTypeGeneratorType">:$type
  );

  let assemblyFormat = "`<` $name `,` $attrs `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name, "DictionaryAttr":$attrs,
        "FuncTypeGeneratorType":$type), [{
      return $_get(name.getContext(), name, attrs, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DecoratorsAttr
//===----------------------------------------------------------------------===//

def KGEN_DecoratorsAttr : ArrayOfAttr<KGENDialect, "Decorators", "decorators",
    "TypedAttr"> {
  let summary = "Compile-time decorator invocations.";
  let description = [{
    The `#kgen.decorators` attribute represents a list of decorator invocations
    attached to an operation. Decorators are closures where the first argument
    of the function will be the operation the decorator is attached to. The
    expected signature of the closure is:

    ```mlir
    (!pdl.operation) capturing -> !pdl.operation
    ```

    Decorators in the list are invoked on the operation from first to last.
    Decorators may be invoked at different points in KGEN pass pipeline. Each
    decorator contains a tag indicating when it should be invoked. Successive
    decorators must have later invocation points than previous ones.
    (TODO: Not implemented yet)
  }];
}

//===----------------------------------------------------------------------===//
// PreservedAttr
//===----------------------------------------------------------------------===//

def KGEN_PreservedAttr : KGENAttr<"Preserved", "preserved"> {
  let summary = "Preserved attribute that is not affected by rewrites.";
  let description = [{
    The `#kgen.preserved` attribute contains an attribute and isolates it from
    all rewrites and lowerings. This is useful for keeping higher-level
    information, like source information, around in the IR in case users need to
    inspect them.

    Example:

    ```mlir
    #kgen.preserved<!lit.generator<() -> ()>>
    ```
  }];

  let parameters = (ins "Attribute":$value);
  let assemblyFormat = "`<` $value `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$value), [{
      return $_get(value.getContext(), value);
   }]>
  ];

  let extraClassDeclaration = [{
    static auto getWalkImmediateSubElementsFn() {
      return [](auto, function_ref<void(Attribute)>,
                      function_ref<void(Type)>) {};
    }
    static auto getReplaceImmediateSubElementsFn() {
      return [](auto instance, ArrayRef<Attribute>, ArrayRef<Type>) {
        return instance;
      };
    }
  }];
}

//===----------------------------------------------------------------------===//
// ExportKindAttr
//===----------------------------------------------------------------------===//

def KGEN_ExportKindAttr : EnumAttr<KGENDialect, KGEN_ExportKind, "export"> {
  let summary = "Defines the export semantics of a symbol.";
  let description = [{
    The `#kgen.export` attribute defines the export semantics of a symbol. A
    symbol can be:

    - Not exported: its linkage is internal and its visibility is hidden.
    - Exported: its linkage is external and its visibility is public.
    - C exported: like `exported`, but with a C-compatible name and ABI.

    Example:

    ```mlir
    #kgen.export_kind<not_exported>
    #kgen.export_kind<exported>
    #kgen.export_kind<c_exported>
    ```
  }];

  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// LinkDependencyArrayAttr
//===----------------------------------------------------------------------===//

def KGEN_LinkDependencyArrayAttr : ArrayOfAttr<KGENDialect,
  "LinkDependencyArray", "link.dependencies", "FlatSymbolRefAttr"> {
  let summary = "List of references to link dependencies.";
  let description = [{
    The `#kgen.link.dependencies` attribute represents a list of link
    dependencies, which are flat symbol references.
  }];
}

//===----------------------------------------------------------------------===//
// TailKindAttr
//===----------------------------------------------------------------------===//

def KGEN_TailKindAttr : EnumAttr<KGENDialect, KGEN_TailKind, "tailkind"> {
  let summary = "Defines the tail call type.";
  let description = [{
    The `#kgen.tailkind` attribute defines the export semantics of a symbol. A
    symbol can be:

    - None: Unspecified.
    - Musttail: Compilation will fail if this call cannot be tail call
                optimized.
    - NoTail: Do not tail call optimize this call.

    Example:

    ```mlir
    #kgen.tailkind<none>
    #kgen.tailkind<musttail>
    #kgen.tailkind<notail>
    ```
  }];

  let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// ClosureMethodAttr
//===----------------------------------------------------------------------===//

def KGEN_ClosureMethodAttr : EnumAttr<KGENDialect, KGEN_ClosureMethod, "closure_method"> {
  let summary = "Reference to a closure method.";
  let description = [{
    The `#kgen.closure_method` attribute represents the symbol of a closure method.

    Example:

    ```mlir
    #kgen.closure_method<call>
    ```
  }];

  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ClosureSymbolAttr
//===----------------------------------------------------------------------===//

def KGEN_ClosureSymbolAttr : KGENAttr<"ClosureSymbol", "closure.symbol",
  [TypedAttrInterface]> {
  let summary = "Reference to a yet to be synthesized closure method";
  let description = [{
    We want to model function calls to functions that have not been generated
    yet. These functions are with respect to a capture struct and a nested
    function. This attribute must contain enough information to pair it with
    the generated methods which includes the symbol of the enclosing method.

    Example:
    ```mlir
    #kgen.closure.symbol<@foo,
                         "fn",
                         #kgen.closure_method<call>,
                         <:!kgen.param_capture<@foo "fn"> ?>
                        >
    ```
  }];

  let parameters = (ins
    "SymbolRefAttr": $parentSymbol,
    "StringAttr": $nestedFuncName,
    "ClosureMethodAttr": $method,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    AttributeSelfTypeParameter<"", "FuncTypeGeneratorType">:$type
  );
  let assemblyFormat = [{
    `<` custom<ClosureSymbolValue>($parentSymbol,
                               $nestedFuncName,
                               $method, $paramValues) `>`
  }];
}

//===----------------------------------------------------------------------===//
// FuncSymbolAttr
//===----------------------------------------------------------------------===//

def KGEN_FuncSymbolAttr : KGEN_ExprAttr<"FuncSymbol", "func.symbol",
    [DeclareAttrInterfaceMethods<DeclRefAttrInterface>]> {
  let summary = "a kgen.func reference";
  let description = [{
    This is a value of FuncType, which refers to a func, the `type` must
    match with the FuncType of the given `symbol` after instantiated with the
    `paramValues`.

    TODO: Delete SymbolConstantAttr after fully migrate to FuncLiteralType.
  }];

  let parameters = (ins
    "SymbolRefAttr":$symbol,
    ArrayRefParameter<"TypedAttr">:$paramValues,
    AttributeSelfTypeParameter<"", "FuncType">:$type
  );

  let builders = [
    AttrBuilderWithInferredContext<
      (ins "SymbolRefAttr":$symbol, "FuncType":$type,
           CArg<"ArrayRef<TypedAttr>">:$paramValues), [{
      return get(type.getContext(), symbol, paramValues, type);
    }]>,

    AttrBuilderWithInferredContext<
      (ins "StringAttr":$name, "FuncType":$type,
           CArg<"ArrayRef<TypedAttr>">:$paramValues), [{
      return get(FlatSymbolRefAttr::get(name), type, paramValues);
    }]>,

  ];

  let assemblyFormat = [{
    `<` $symbol `` custom<ParameterValues>($paramValues) `` `>`
  }];
}

//===----------------------------------------------------------------------===//
// DeferredAttr
//===----------------------------------------------------------------------===//

def KGEN_DeferredAttr : KGENAttr<"Deferred", "deferred", [
    TypedAttrInterface]> {
  let summary = "A wrapper for non-typed attribute";
  let description = [{
    The `#kgen.deferred` attribute holds a non-typed attribute to allow it
    to be created later.

    Example:

    ```mlir
    #kgen.deferred #index<cmp_predicate sle>> : !kgen.deferred
    ```
  }];

  let parameters = (ins "Attribute":$attr);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
  let assemblyFormat = "$attr";

  let extraClassDeclaration = [{ Type getType() const; }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// AttrCtorDeferredAttr
//===----------------------------------------------------------------------===//

def KGEN_AttrCtorDeferredAttr : KGENAttr<"AttrCtorDeferred",
                                         "attr_ctor_deferred",
     [TypedAttrInterface]> {
  let summary = "Deferred construction of an attribute";
  let description = [{
    The `#kgen.attr_ctor_deferred` attribute holds an array of StringAttr
    or `#kgen.to_string_deferred` attributes. In the elaborator, when
    attributes are concrete, the `#kgen.attr_ctor_deferred` concatenates them
    and builds and requested attribute.

    Example:

    ```mlir
    #kgen.attr_ctor_deferred<"#index<", "cmp_predicate", "sle>">>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$strings);

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$strings), [{
      return $_get(strings[0].getContext(), strings);
    }]>
  ];
  let assemblyFormat = "`(` $strings `)`";
  let extraClassDeclaration = [{ Type getType() const; }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// ToStringDeferredAttr
//===----------------------------------------------------------------------===//

def KGEN_ToStringDeferredAttr : KGENAttr<"ToStringDeferred",
                                         "to_string_deferred",
    [TypedAttrInterface]> {
  let summary = "Deferred string concatenation";
  let description = [{
    The `#kgen.to_string_deferred` attribute holds an array of StringAttr
    and concatenates them into a single StringAttr in Elaborator

    Example:

    ```mlir
    #kgen.to_string_deferred<"#index<", "cmp_predicate", "sle>">>
    ```
  }];

  // NOTE: Cannot have TypedAttr as it will make impossible to use attribute
  // replacer functions that do verify consistency of the replaced attribute.
  // Since in elaborator the attr can become non-typed, use generic Attribute
  // here.
  // Cannot add verification to make sure attribute is typed until elaborator,
  // as these replace functions do invoke verification to check consistency.
  let parameters = (ins "Attribute":$attr,
                        OptionalParameter<"UnitAttr">:$needElideType);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$attr, "bool":$needElideType), [{
      if (needElideType)
        return $_get(attr.getContext(), attr, UnitAttr::get(attr.getContext()));
      return $_get(attr.getContext(), attr, UnitAttr());
    }]>
  ];
  let assemblyFormat = [{
    `(` $attr `)` (`elide_type` $needElideType^ )?
  }];
  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// ConstraintAttr
//===----------------------------------------------------------------------===//

def KGEN_ConstraintAttr : KGENAttr<"Constraint", "constraint"> {
  let summary = "A constraint with proposition, location and optional message";
  let description = [{
    The `#kgen.constraint` attribute represents a proposition that should hold
    and a location for where this constraint was declared, which is useful for
    error reporting.

    The proposition is an i1-typed parameter expression.

    The optional message is a user-provided string (from
    `where (cond, "message")` syntax) that is surfaced in the diagnostic when
    the constraint fails. It is null for constraints without a user message
    (including all compiler-synthesized constraints).

    Example:

    ```mlir
    #kgen.constraint<1, loc("file.mojo":10:5)>
    #kgen.constraint<1, loc("file.mojo":10:5), "must be positive">
    ```
  }];

  let parameters = (ins
    "TypedAttr":$proposition,
    // FIXME(MOCO-4416): `loc` and `message` should be excluded from the
    // canonical form used for type equality, so two otherwise-equal constraints
    // that differ only in conformance location or message are not distinct types.
    "LocationAttr":$loc,
    OptionalParameter<"StringAttr">:$message
  );

  let assemblyFormat = [{
    `<` custom<ScalarBoolParamValue>($proposition) `,` $loc
    (`,` $message^)? `>`
  }];

  // Note: `message` is deliberately a required builder argument (no default).
  // ConstraintAttr is reconstructed at several places in the frontend; a
  // defaulted argument would let a reconstruction site silently drop the
  // user's message. Sites without a user message pass an explicit
  // `StringAttr{}`.
  let builders = [
    AttrBuilderWithInferredContext<(ins
        "TypedAttr":$proposition,
        "LocationAttr":$loc,
        "StringAttr":$message)>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// PogMetadataAttr
//===----------------------------------------------------------------------===//

def KGEN_PogMetadataAttr : KGENAttr<"PogMetadata", "pog_metadata"> {
  let summary = "Metadata for a single argument or parameter.";
  let description = [{
    The `#kgen.pog_metadata` attribute contains metadata about an argument or
    parameter of a function, including the name, passing kind, variadicness, and
    the default value (if present).

    Example:

    ```mlir
    #kgen.pog_metadata<"some_keyword_param", pos_or_kw, false, 42>
    #kgen.pog_metadata<"some_variadic_param", pos_or_kw, true>
    ```
  }];

  let parameters = (ins
    "StringAttr":$name,
    "PassingKind":$passingKind,
    "VariadicKind":$variadic,
    OptionalParameter<"TypedAttr">:$defaultValue
  );

  let assemblyFormat = [{
    `<` $name `,` $passingKind `,` $variadic
    (`,` `default` custom<ColonTypeParamValue>($defaultValue)^)?
    `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
      "PassingKind":$passingKind,
      CArg<"VariadicKind", "VariadicKind::None">:$variadic,
      CArg<"TypedAttr", "{}">:$defaultValue
    ), [{
      return $_get(name.getContext(), name, passingKind, variadic,
                   defaultValue);
    }]>,
    AttrBuilder<(ins), [{
      return get(StringAttr::get($_ctxt), PassingKind::PosOrKw);
    }]>
  ];

  let extraClassDeclaration = [{
    bool isAnyVarArg() const { return getVariadic() != VariadicKind::None; }
    bool isPosVarArg() const { return getVariadic() == VariadicKind::PosVarArg; }
    bool isPack() const { return getVariadic() == VariadicKind::PackVarArg; }
    bool isKwVarArg() const { return getVariadic() == VariadicKind::KwVarArg; }
  }];
}

//===----------------------------------------------------------------------===//
// FnMetadataAttr
//===----------------------------------------------------------------------===//

def KGEN_FnMetadataAttr : KGENAttr<"FnMetadata", "fn_metadata", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "The metadata of a `!kgen.func` type";
  let description = [{
    The `#kgen.fn_metadata` attribute aggregates everything a `!kgen.func` type
    describes on top of its value signature: the calling convention of each
    input argument, the effects of the function, and the dialect-specific
    metadata implementing `FnMetadataAttrInterface` (for example
    `#lit.fn_meta_origin_data`).

    Every `!kgen.func` type carries exactly one of these, so it is not printed
    on its own in the sugared function type syntax.

    Example:

    ```mlir
    #kgen.fn_metadata<[read, mut], "throws">
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"ArgConvention">:$argConventions,
    "FnEffects":$fnEffects,
    "FnMetadataAttrInterface":$metadata
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// The type of this attribute is always `!kgen.non_struct_type`.
    Type getType() const;

    /// Return this metadata with the function effects replaced.
    FnMetadataAttr getWithFnEffects(FnEffects effects) const;

    /// Return this metadata with the dialect-specific metadata replaced.
    FnMetadataAttr getWithMetadata(FnMetadataAttrInterface metadata) const;
  }];
}

//===----------------------------------------------------------------------===//
// PogListAttr
//===----------------------------------------------------------------------===//

def KGEN_OptionalConventionParameter : OptionalParameter<"ArgConvention"> {
  // Use ByRefError as a sentinel, since it is obviously invalid.
  let defaultValue = "ArgConvention::ByRefError";
}

def KGEN_PogListAttr : KGEN_ExprAttr<"PogList", "pog_list"> {
  let summary = "Argument or parameter list metadata.";
  let description = [{
    The `#kgen.pog_list` attribute contains metadata about an argument or
    parameter list of a function, including names, passing kinds, default
    values, and information about variadic arguments.
    The positional default values correspond to the trailing positional
    (i.e. pos-only or pos-or-kw) args/params, and the keyword-only default
    values similarly correspond to the trailing keyword-only args/params.

    This attribute is the `metadata` field of `GeneratorType`.

    Example:

    ```mlir
    #kgen.pog_list<
      ["a", "b", "c", "d"],
      [pos, pos_or_kw, kw, kw],
    >
    ```

    The `origVariadicConvention` indicates whether the original argument
    convention of a VariadicList or VariadicPack, e.g. "mut *args: Int".

    Optional `bodyConstraints` holds constraints that are enforced by the body
    of the generator type that carries this list as metadata.
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"PogMetadataAttr">:$pogs,
    OptionalArrayRefParameter<"ConstraintAttr">:$bodyConstraints,
    KGEN_OptionalConventionParameter:$origVariadicConvention
  );

  let assemblyFormat = [{
    `<` `[` (`]`) : ($pogs^ `]`)?
        (`{` $bodyConstraints^ `}`)?
        (`,` custom<ArgConvention>($origVariadicConvention)^)?
    `>`
  }];

  let builders = [
    AttrBuilder<(ins)>,
    AttrBuilder<(ins "size_t":$numPogs)>,
    AttrBuilder<(ins "size_t":$numPogs,
                     "ArrayRef<ConstraintAttr>":$bodyConstraints)>,
    AttrBuilder<(ins "ArrayRef<PogMetadataAttr>":$pogs)>,
    AttrBuilder<(ins "ArrayRef<StringAttr>":$names,
                     "ArrayRef<PassingKind>":$passingKinds)>,
    AttrBuilder<(ins "ArrayRef<StringAttr>":$names,
                     "ArrayRef<PassingKind>":$passingKinds,
                     "ArrayRef<VariadicKind>":$variadics,
                     "ArrayRef<TypedAttr>":$defaults,
                     "std::optional<ArgConvention>":$origVariadicConvention,
                     "ArrayRef<ConstraintAttr>":$bodyConstraints)>
  ];

  let extraClassDeclaration = [{
    /// TODO: probably a dedicate type would make more sense here.
    Type getType() const {return NonStructTypeType::get(getContext());}

    /// Create a new argument/parameter list attr with the given array of
    /// individual argument/parameter metadata attributes.
    PogListAttr cloneWith(ArrayRef<PogMetadataAttr> pogs) const;

    /// Return the number of parameter entries.
    size_t size() const { return getPogs().size(); }

    /// Return the variadic kind for the argument/parameter at the given index.
    VariadicKind getVariadicKind(size_t idx) const;

    /// Return true if the argument/param at `idx` is a variadic.
    bool isAnyVarArg(size_t idx) const;
    /// Return true if the argument/param at `idx` is a pack.
    bool isPack(size_t idx) const;
    /// Return true if the argument/param at `idx` is a positional variadic.
    bool isPosVarArg(size_t idx) const;
    /// Return true if the argument/param at `idx` is a keyword variadic.
    bool isKwVarArg(size_t idx) const;
    /// Return true if there is a variadic (pos or keyword) argument/param.
    bool hasAnyVarArg() const;
    /// Return true if there is a variadic pack argument/param.
    bool hasPackVarArg() const;
    /// Return true if there is a variadic keyword argument/param.
    bool hasKwVarArg() const;
    /// Return true if there are inferred parameters.
    bool hasInferredParams() const;

    /// Return the name for the argument/parameter at the given index.
    StringAttr getName(size_t idx) const;
    /// Return the passing kind for the argument/parameter at the given index.
    PassingKind getPassingKind(size_t idx) const;
    /// Return the default value for the argument/parameter at the given index
    /// or null if none. Returns null on an empty pog list (a `FuncType`
    /// constructed outside the Mojo parser with no source-level metadata).
    TypedAttr getDefault(size_t idx) const {
      ::llvm::ArrayRef<PogMetadataAttr> pogs = getPogs();
      if (pogs.empty())
        return {};
      return pogs[idx].getDefaultValue();
    }

    /// Return the number of leading "implicit" parameters by PassingKinds.
    size_t getNumImplicit() const;

    /// Create an array of PogMetadataAttr from the given names, passing kinds,
    /// and variadic indices.
    static SmallVector<PogMetadataAttr> toPogs(
      ArrayRef<StringAttr> names, ArrayRef<PassingKind> passingKinds,
      ArrayRef<VariadicKind> variadics, ArrayRef<TypedAttr> defaults);

    /// Prepend inferred parameters: `defaults` is either empty or the same
    /// length as `names` (use null/empty `TypedAttr` for slots with no
    /// default). When non-empty, defaults must already be remapped for the
    /// prepended `ParamDeclAttr` list by the caller. `bodyConstraints` are the
    /// constraints enforced by the body that follows the prepended parameters.
    PogListAttr prependAsInferredParams(
      ArrayRef<StringAttr> names, ArrayRef<TypedAttr> defaults,
      ArrayRef<ConstraintAttr> bodyConstraints = {}) const;

    //===------------------------------------------------------------------===//
    // GeneratorType metadata hooks
    //===------------------------------------------------------------------===//
    // `PogListAttr` is the metadata attached to `GeneratorType`. The hooks
    // below are invoked by the type framework during verification, printing,
    // and parameter specialization.

    /// Verify the generator using the current metadata.
    LogicalResult verifyGenerator(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<Type> inputParamTypes,
                                  Type body) const;

    /// Print the provided generator. The metadata owns the textual format.
    void printGenerator(AsmPrinter &p, GeneratorType generator) const;

    /// Get a new metadata attribute for a generator where the parameters
    /// indicated by set bits in `boundParams` have been removed by being
    /// explicitly bound to values. The `evaluator` substitutes bound values
    /// into any sub-attr/type of the metadata. Body constraints whose
    /// corresponding bit in `dischargedBodyConstraints` is set are removed
    /// from the result; an empty mask (the default) retains every body
    /// constraint.
    PogListAttr getSpecializedMetadata(
        ParameterEvaluator &evaluator, const llvm::BitVector &boundParams,
        function_ref<InFlightDiagnostic()> emitError,
        const llvm::BitVector &dischargedBodyConstraints = {}) const;

    /// Get a new metadata attribute with positional input parameters
    /// prepended, named by `newParams` and tagged as inferred.
    PogListAttr
    prependContextualParamsFromOps(ArrayRef<StringAttr> newParams,
                                   ArrayRef<Operation *> ops) const;
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// LLVMBitcodeLibAttr
//===----------------------------------------------------------------------===//

def KGEN_LLVMBitcodeLibAttr : KGENAttr<"LLVMBitcodeLib", "llvm.bitcode.lib"> {
  let summary = "LLVM bitcode library attribute.";
  let description = [{
    The `#kgen.llvm.bitcode.lib` attribute represents a single LLVM bitcode
    library with usage tracking. It contains:
    - `used`: A boolean flag indicating whether this library was used
    - `library`: The actual bitcode library, which can be either:
      - StringAttr: For bitcode libraries passed via command line
      - DenseResourceElementsAttr: For bitcode libraries from packages

    Example:
    ```mlir
    #kgen.llvm.bitcode.lib<used = false, library = "/path/to/lib.bc">
    #kgen.llvm.bitcode.lib<used = true, library = dense_resource<data> : ...>
    ```
  }];

  let parameters = (ins "BoolAttr":$used, "Attribute":$library);

  let assemblyFormat =
    "`<` `used` `=` $used `,` `library` `=` $library `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "bool":$used,
                                        "Attribute":$library), [{
      return $_get(library.getContext(),
                   BoolAttr::get(library.getContext(), used),
                   library);
    }]>,
    AttrBuilderWithInferredContext<(ins "BoolAttr":$used,
                                        "Attribute":$library), [{
      return $_get(used.getContext(), used, library);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// LLVMBitcodeLibArrayAttr
//===----------------------------------------------------------------------===//

def KGEN_LLVMBitcodeLibArrayAttr : ArrayOfAttr<KGENDialect,
  "LLVMBitcodeLibArray", "llvm.bitcode.libs", "LLVMBitcodeLibAttr"> {
  let summary = "Array of LLVM bitcode library attributes.";
  let description = [{
    The `#kgen.llvm.bitcode.libs` attribute represents an array of LLVM
    bitcode libraries, each with their own usage tracking. This is typically
    attached to ModuleOp to store all bitcode libraries that should be
    linked during compilation.
  }];

  let extraClassDeclaration = [{
    /// This is the attribute name of the `#kgen.llvm.bitcode.libs` attribute
    /// attached to ModuleOp.
    static llvm::StringLiteral getBitcodeLibsAttrName() {
      return "kgen.llvm.bitcode.libs";
    }

    /// Populate the given vector with this array attribute in native format.
    void externalize
      (llvm::SmallVector<std::pair<bool, mlir::Attribute>> &result) const;
  }];
}

//===----------------------------------------------------------------------===//
// SugarAttr
//===----------------------------------------------------------------------===//

def KGEN_SugarAttr : KGENAttr<"Sugar", "sugar",
                              [TypedAttrInterface, OpAsmAttrInterface]> {
  let summary = "Syntax sugar for some other attribute.";
  let description = [{
    The `#kgen.sugar` attribute represents a syntax sugar overlaid on some other
    value e.g. an alias or expanded builtin function call. It maintains the
    original sugared form as well as the "one level expanded" form, and
    fully expanded "canonical" version of the attribute.
  }];

  let parameters = (ins
    "SugarKind":$kind,
    "StringAttr":$memberName, // Member name if this is a MemberAlias.
    "TypedAttr":$sugared,     // Fully sugared form.
    "TypedAttr":$expanded,    // One level expanded form, other sugar may exist.
    "TypedAttr":$canonical    // Recursively desugared form.
  );

  let assemblyFormat =
    "custom<SugarAttr>($kind, $memberName, $sugared, $expanded, $canonical)";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "SugarKind":$kind, "StringAttr":$memberName,
                  "TypedAttr":$sugared, "TypedAttr":$expanded,
                  CArg<"TypedAttr", "{}">:$canonical), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// Remove any top-level sugar nodes from this type, but don't fully
    /// canonicalize it.  If keepOpaque is true, opaque sugar kinds
    /// (AlwaysInlineBuiltin, Preserved) are kept.
    static TypedAttr strip(TypedAttr value, bool keepOpaque = false);
    static Attribute strip(Attribute value, bool keepOpaque = false);
    static Type strip(Type value, bool keepOpaque = false);

    /// Return true if the specified value has top level SugarAttr.  Nested
    /// sugar isn't processed here.
    static bool hasTopLevelSugar(Type type);
    static bool hasTopLevelSugar(TypedAttr value) {
      return llvm::isa<KGEN::SugarAttr>(value);
    }

    Type getType() const;
    mlir::OpAsmAliasResult getAlias(raw_ostream &os) const;

    /// For a MemberAlias sugar, return the base type.
    Type getMemberAliasType() const;

    static TypedAttr getAlwaysInlineBuiltin(TypedAttr sugared,
                                            TypedAttr expanded) {
      return get(sugared.getContext(), SugarKind::AlwaysInlineBuiltin, {},
                 sugared, expanded);
    }
    static TypedAttr getPreserved(TypedAttr sugared, TypedAttr expanded) {
      return get(sugared.getContext(), SugarKind::Preserved, {}, sugared,
                 expanded);
    }
    static TypedAttr getMemberAlias(Type type, StringAttr memberName,
                                    TypedAttr expanded);
    static TypedAttr getAlias(TypedAttr sugared, TypedAttr expanded) {
      return get(sugared.getContext(), SugarKind::Alias, {}, sugared, expanded);
    }
  }];
}

//===----------------------------------------------------------------------===//
// CastFromBuiltin
//===----------------------------------------------------------------------===//

def KGEN_CastFromBuiltinAttr : KGEN_ExprAttr<"CastFromBuiltin",
                                             "cast_from_builtin"> {
  let summary = "Convert a builtin MLIR type to a POP type.";
  let description = [{
    The `#kgen.cast_from_builtin` attribute converts a builtin MLIR type to a
    POP type.
  }];

  let parameters = (ins "TypedAttr":$arg,
                        AttributeSelfTypeParameter<"", "SIMDType">:$type);

  let assemblyFormat = " `<` $arg `>` ";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an CastFromBuiltinAttr.
    AttrBuilder<(ins "TypedAttr":$arg, "SIMDType":$type), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$arg,
                                        "SIMDType":$type), [{
      return get(arg.getContext(), arg, type);
    }], "TypedAttr">,

    // A builder that infers the return type from the argument type.
    AttrBuilderWithInferredContext<(ins "TypedAttr":$arg), "", "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// Return the SIMD type that `cast_from_builtin(arg)` infers from `arg`'s
    /// builtin type alone. Returns null when no equivalent SIMD type exists.
    static SIMDType getInferredResultType(TypedAttr arg);
  }];
}

//===----------------------------------------------------------------------===//
// CastToBuiltin
//===----------------------------------------------------------------------===//

def KGEN_CastToBuiltinAttr : KGEN_ExprAttr<"CastToBuiltin", "cast_to_builtin"> {
  let summary = "Convert to a POP type to a builtin MLIR type.";
  let description = [{
    The `#kgen.cast_to_builtin` attribute converts a POP type to a builtin MLIR
    type.
  }];

  let parameters = (ins "TypedAttr":$arg,
                        AttributeSelfTypeParameter<"", "Type">:$type);

  let assemblyFormat = " `<` $arg `>` ";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an CastToBuiltinAttr.
    AttrBuilder<(ins "TypedAttr":$arg, "Type":$type), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$arg, "Type":$type), [{
      return get(arg.getContext(), arg, type);
    }], "TypedAttr">,

    // A builder that infers the return type from the argument type.
    AttrBuilderWithInferredContext<(ins "TypedAttr":$arg), "", "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// Return the builtin MLIR type that `cast_to_builtin(arg)` infers from
    /// `arg`'s SIMD type alone. Returns null when no equivalent type exists.
    static Type getInferredResultType(TypedAttr arg);
  }];
}

//===----------------------------------------------------------------------===//
// SIMDAttr
//===----------------------------------------------------------------------===//

def KGEN_SIMDAttr : KGEN_ExprAttr<"SIMD", "simd"> {
  let summary = "A SIMD vector attribute.";
  let description = [{
    The `#kgen.simd` attribute represents a constant SIMD vector value. It
    contains `N` values of a particular dtype. Only integer, floating point, and
    bool dtypes are supported.

    Example:

    ```mlir
    #kgen.simd<1, 2> : !kgen.simd<2, si32>
    #kgen.simd<1.5, 2.5> : !kgen.simd<2, f64>
    #kgen.simd<true, false> : !kgen.simd<2, bool>
    ```

    When all values of the SIMD vector are equal, the attribute has a special
    splat syntax:

    ```mlir
    #kgen<simd 0> : !kgen.simd<4, si32>
    #kgen<simd "1.5"> : !kgen.simd<4, f32>
    #kgen<simd false> : !kgen.simd<4, bool>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"DTypeValue">:$values,
                        AttributeSelfTypeParameter<"", "SIMDType">:$type);
  let assemblyFormat = "`` custom<SIMDValues>($values, ref($type))";

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<DTypeValue>":$values,
                                        "SIMDType":$type), [{
      return get(type.getContext(), values, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "DTypeValue":$value,
                                        "SIMDType":$type), [{
      return get(ArrayRef(value), type);
    }]>,
    AttrBuilderWithInferredContext<(ins "uint64_t":$intVal, "SIMDType":$type)>
  ];


  let extraClassDeclaration = [{
    /// Create a "zero" value initialized to 0, 0.0, false, etc.  NOTE: This can
    /// return null if the SIMD type is parametric or we don't know how to make
    /// a zero value of the specified dtype.
    static SIMDAttr getZeroValue(SIMDType type);

    /// Create a bool scalar simd attr.
    static SIMDAttr getScalarBool(MLIRContext *ctx, bool value);

    /// Returns true/false for the simd literal. It *asserts* the simd value has
    /// a scalar<bool> type.
    bool getAsBool() const;
  }];

  let genVerifyDecl = 1;

  /// Note: DTypeValue is not POD since APInt will allocate if its data exceeds
  /// 64 bits. Use a custom storage class to own the memory.
  let genStorageClass = 0;
}

//===----------------------------------------------------------------------===//
// SIMDSplat
//===----------------------------------------------------------------------===//

def KGEN_SIMDSplatAttr : KGEN_ExprAttr<"SIMDSplat", "simd_splat"> {
  let summary = "SIMD splat.";
  let description = [{
    The `#kgen.simd_splat` attribute takes a scalar value and replicates it
    across a SIMD vector.
  }];

  let parameters = (ins "TypedAttr":$arg,
                        AttributeSelfTypeParameter<"", "SIMDType">:$type);

  let assemblyFormat = " `<` $arg `>` ";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return a SIMDSplatAttr.
    AttrBuilder<(ins "TypedAttr":$arg, "SIMDType":$type), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$arg, "SIMDType":$type), [{
      return get(arg.getContext(), arg, type);
    }], "TypedAttr">
  ];
}

#endif // KGEN_KGENDIALECT_KGENATTRS_TD
