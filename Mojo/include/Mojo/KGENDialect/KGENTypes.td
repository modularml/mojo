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
// This defines types in the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPES_TD
#define KGEN_KGENDIALECT_KGENTYPES_TD

include "Mojo/KGENDialect/KGENDialect.td"
include "Mojo/KGENDialect/KGENTypeInterfaces.td"
include "Mojo/Interpreter/InterpreterInterface.td"
include "Support/MDialect/MTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class KGENType<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<KGENDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ParamType
//===----------------------------------------------------------------------===//

def KGEN_ParamType : KGENType<"Param", "param", [
  OpAsmTypeInterface,
  DeclareTypeInterfaceMethods<ParameterTypeInterface>
]> {
  let summary = "Type specified by a parameter expression (type domain)";
  let description = [{
    This is a symbolic type represented with a parameter expression that is
    resolved by the KGEN elaborator.  Once this parameter is substituted with a
    type constant, this ParamType is folded away and the MlirType inside
    the type constant is returned.
  }];

  let parameters = (ins "TypedAttr":$param);
  let assemblyFormat = "`<` custom<TypeParamValue>($param) `>`";

  // Our builders create ParamType but fold to the base type if not
  // parameterized.
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$param), [{}], "Type">,
     // This is needed for ODS generated uses.
    TypeBuilderWithInferredContext<(ins "MLIRContext *":$context,
                                        "TypedAttr":$param),
                                   [{ return get(param); }], "Type">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the type from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static ParamType getFromBytecode(TypedAttr param);

    mlir::OpAsmAliasResult getAlias(raw_ostream &os) const;

    Type getMetaType() { return getParam().getType(); }
  }];
}

//===----------------------------------------------------------------------===//
// TypeValueType
//===----------------------------------------------------------------------===//

def KGEN_TypeValueType : KGENType<"TypeValue", "typevalue",
    [DeclareTypeInterfaceMethods<TraitSymbolInterface>]> {
  let summary = "Type specified by a parameter expression (value domain)";
  let description = [{
    This is the value domain counterpart to KGEN ParamType. It allows
    losslessly using a parameter expression as a MLIR Type: Even when its
    type value parameter is substituted with a type constant, it does not fold
    into the MlirType itself (unless the type constant is a trivial mlir type).
  }];

  let parameters = (ins "TypedAttr":$typeValue);
  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `>`";

  // Our builders create TypeValueType but fold to the base type if not
  // parameterized.
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$typeValue), [{}], "Type">,
     // This is needed for ODS generated uses.
    TypeBuilderWithInferredContext<(ins "MLIRContext *":$context,
                                        "TypedAttr":$typeValue),
                                   [{ return get(typeValue); }], "Type">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the type from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static TypeValueType getFromBytecode(TypedAttr typeValue);
  }];
}

//===----------------------------------------------------------------------===//
// TypeType
//===----------------------------------------------------------------------===//

def KGEN_TypeType : KGENType<"Type", "type", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "Type of all types";
  let description = [{
    This kgen type represents an arbitrary type, used for parameterizing
    type-generic functions and structs.

    It cannot be materialized into an SSA value, because it has no runtime
    representation.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// NonStructTypeType
//===----------------------------------------------------------------------===//

def KGEN_NonStructTypeType : KGENType<"NonStructType", "non_struct_type", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Type of non-lit-struct types";
  let description = [{
    This kgen type presents the type of all L1 non-lit-struct types (which
    essentially means it is not a mojo struct nor a meta type expression).
    In particular, this categorizes types that need to be wrapped
    by `__MLIRType` in Mojo.

    ```mojo
    comptime mlir_i1 = __mlir_type.i1
    # type_of(mlir_i1) == !kgen.non_struct_type

    comptime def_type = def()->Int
    # type_of(def_type) == !kgen.non_struct_type

    # Notably:
    # type_of(type_of(mlir_i1)) == !kgen.type
    ```
  }];

  let assemblyFormat = "";

  let extraClassDeclaration = [{
    // If it is a non_struct_type, its has a kgen.type for metatype.
    Type getMetaType() { return TypeType::get(getContext()); }
  }];
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

def KGEN_DTypeType : KGENType<"DType", "dtype", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "8-bit integer DType type";
  let description = [{
    This type corresponds to the DType runtime class, representing an
    element type (aka "dtype") specifier for a data storage types.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def KGEN_StringType : KGENType<"String", "string", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "String type for parameters";
  let description = [{
    This kgen type represents an string value, used for parameterizing
    generators and working with string-returning expressions.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// GeneratorType
//===----------------------------------------------------------------------===//

def KGEN_GeneratorType : KGENType<"Generator", "generator", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    ParameterScopeTypeInterface]> {
  let summary = "The type of a generator (i.e. a parameterized value)";
  let description = [{
    This type describes a generator (i.e. a parameterized value). It describes
    the generator's parameter signature (the types of input parameters), and the
    generator's output type (potentially in terms of the input parameters).
  }];

  let parameters = (ins
    ArrayRefParameter<"Type">:$inputParamTypes,
    "Type":$body,
    "PogListAttr":$paramListAttrs
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "ArrayRef<Type>":$inputParamTypes, "Type":$body,
        CArg<"Attribute", "{}">:$paramListAttrs), [{
      return $_get(body.getContext(), inputParamTypes, body,
                   ::cast_or_null<PogListAttr>(paramListAttrs));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get this GeneratorType with a new body.
    GeneratorType getWithBody(Type newBody);

    /// Get this GeneratorType with some parameters bound.
    GeneratorType getSpecializedGenerator(
      PartiallySpecializedInputParams &specialization,
      function_ref<InFlightDiagnostic()> emitErrorFn = {});
    GeneratorType getSpecializedGenerator(
      ArrayRef<TypedAttr> paramBindings,
      ParameterEvaluationContext *evaluationContext,
      function_ref<InFlightDiagnostic()> emitErrorFn = {});
    GeneratorType getSpecializedGenerator(
      ArrayRef<TypedAttr> paramBindings,
      ParameterEvaluationContext *evaluationContext,
      Location location);

    /// Returns true if the signature has no input params.
    bool isFullyBound() { return getInputParamTypes().empty(); };

    /// Returns the body while asserting that the generator is fully bound.
    Type getInstantiatedBody() {
      assert(isFullyBound() && "cannot instantiate parameterized body");
      return getBody();
    }

    /// Return the name for the parameter at the specified index.
    StringAttr getParamName(size_t idx);

    /// Return this generator's body constraints, or empty if it has none.
    ArrayRef<ConstraintAttr> getBodyConstraints();

    /// Return this generator with all of its body constraints removed.
    GeneratorType getWithoutBodyConstraints();
  }];
}

//===----------------------------------------------------------------------===//
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//

def KGEN_FuncTypeGeneratorType : DialectType<KGENDialect,
    CPred<"llvm::isa<FuncTypeGeneratorType>($_self)">,
    "KGEN GeneratorType wrapping a FuncType", "FuncTypeGeneratorType">;

//===----------------------------------------------------------------------===//
// FuncGeneratorTypeBuilderType
//===----------------------------------------------------------------------===//

def KGEN_FuncGeneratorTypeBuilderType : KGENType<"FuncGeneratorTypeBuilder",
    "func_gen_type_builder"> {
  let summary = "A deferred construction of a `!kgen.generator<!kgen.func>`";
  let description = [{
    The `!kgen.func_gen_type_builder` type constructs a `FuncTypeGeneratorType`
    from its components: the parameters declared by the generator (an array of
    `fn_gen_builder.param.decl`s), the argument types (a `param_list` of
    `kgen.type`), the result type (a `kgen.type`), the function metadata (a
    `fn_metadata`), the generator's parameter pog list (a `pog_list` of the
    declared parameters' metadata), and the argument pog list (a `pog_list` of
    the arguments' metadata). It folds into the generator type itself once all
    of its components are constant.

    Every component is a parameter expression, so a still-symbolic piece (e.g.
    an argument pack referenced by name) can be represented before elaboration.

    The two pog lists are attached to the folded result: the parameter pog
    list becomes the `paramListAttrs` of the constructed `GeneratorType`, and
    the argument pog list becomes the `argListAttrs` of the wrapped `FuncType`.

    Example:

    ```mlir
    !kgen.func_gen_type_builder<
      #kgen<fn_gen_builder.param.decls[T : type]>,
      #kgen.param.decl.ref<"Ts"> : !kgen.param_list<!kgen.type>,
      #kgen.fn_gen_builder.param.decl.ref<"T", type>,
      #kgen.fn_metadata<[read], "none">,
      #kgen.pog_list<["T"], [pos]>,
      #kgen.pog_list<["x"], [pos_or_kw]>
    >
    ```
  }];

  let parameters = (ins
    "TypedAttr":$paramDecls,
    "TypedAttr":$argTypes,
    "TypedAttr":$resultType,
    "TypedAttr":$metadata,
    OptionalParameter<"TypedAttr">:$paramListAttrs,
    OptionalParameter<"TypedAttr">:$argListAttrs
  );

  let assemblyFormat = [{
    `<` $paramDecls `,` $argTypes `,` $resultType `,` $metadata
        (`,` $paramListAttrs^)? (`,` $argListAttrs^)? `>`
  }];

  let genVerifyDecl = 1;

  // Override the builder used by the generated parser: building this type
  // folds to the generator type it describes when the components are concrete,
  // so the result is not necessarily a !kgen.func_gen_type_builder.
  let skipDefaultBuilders = 1;
  let builders = [
    // Convenience form with inferred context.
    TypeBuilderWithInferredContext<(ins "TypedAttr":$paramDecls,
                                        "TypedAttr":$argTypes,
                                        "TypedAttr":$resultType,
                                        "TypedAttr":$metadata,
                                        "TypedAttr":$paramListAttrs,
                                        "TypedAttr":$argListAttrs), [{
      return get(paramDecls.getContext(), paramDecls, argTypes, resultType,
                 metadata, paramListAttrs, argListAttrs);
    }], "Type">,

    // Verbose form with MLIRContext, used by the generated parser.
    TypeBuilder<(ins "TypedAttr":$paramDecls, "TypedAttr":$argTypes,
                     "TypedAttr":$resultType, "TypedAttr":$metadata,
                     "TypedAttr":$paramListAttrs, "TypedAttr":$argListAttrs),
                "", "Type">
  ];
}

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

def KGEN_FuncType : KGENType<"Func", "func", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "The type of a KGEN function";
  let description = [{
    This type describes the type of a function KGEN, which can have input/output
    value arguments and results.

    The metadata attribute holds the argument calling conventions, the effects
    of the function itself, and the dialect-specific function metadata.
  }];

  let parameters = (ins
    "FunctionType":$values,
    "FnMetadataAttr":$metadataAttr,
    "PogListAttr":$argListAttrs
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    /// This builder allows everything other than the function type to be null.
    /// argConventions is sized to match `values.getNumInputs()` to satisfy the
    /// verifier. `argListAttrs` is the source-level pog metadata; it
    /// participates in FuncType uniqueness (MOCO-4022) so that
    /// semantically-distinct functions with otherwise-identical signatures
    /// remain distinct types.
    TypeBuilderWithInferredContext<(ins
        "FunctionType":$values,
        CArg<"ArrayRef<ArgConvention>", "{}">:$argConvs,
        CArg<"FnEffects", "{}">:$effects,
        CArg<"Attribute", "{}">:$metadata,
        CArg<"Attribute", "{}">:$argListAttrs), [{
      SmallVector<ArgConvention> convs = ::llvm::to_vector(argConvs);
      convs.resize(values.getNumInputs());
      auto argListAttrsAttr = ::cast_or_null<PogListAttr>(argListAttrs);
      if (!argListAttrsAttr)
        argListAttrsAttr = PogListAttr::get(values.getContext());
      return $_get(values.getContext(),
        values,
        FnMetadataAttr::get(values.getContext(),
          convs, effects, ::cast_or_null<FnMetadataAttrInterface>(metadata)),
        argListAttrsAttr);
    }]>,
    TypeBuilder<(ins "TypeRange":$inputs, "TypeRange":$results)>
  ];

  let extraClassDeclaration = [{
    /// Helper to return the argument and result types.
    ArrayRef<Type> getArguments() const;
    Type getArgument(size_t i) const { return getArguments()[i]; }
    ArrayRef<Type> getResults() const;

    /// The convention of each value argument, held by the metadata attribute.
    ArrayRef<ArgConvention> getArgConventions() const;
    /// The effects of this function, held by the metadata attribute.
    FnEffects getFnEffects() const;
    /// The dialect-specific function metadata, held by the metadata attribute.
    FnMetadataAttrInterface getMetadata() const;

    /// Return true if this signature's last argument is a result from the
    /// function returned through memory.
    bool hasMemoryOnlyResult();

    bool isThrows() const { return getFnEffects().isThrows(); }
    bool isAsync() const { return getFnEffects().isAsync(); }
    bool isCapturing() const { return getFnEffects().isCapturing(); }
    bool isRefResult() const { return getFnEffects().isRefResult(); }
    bool isCABI() const { return getFnEffects().isCABI(); }

    /// Return the convention for the specified value argument.
    ArgConvention getArgConvention(size_t inputNo) {
      return getArgConventions()[inputNo];
    }

    /// Set a function convention.
    FuncType getWithFnEffects(FnEffects effect);
    /// Return this signature type with the value signature replaced.
    FuncType getWithValuesReplaced(FunctionType fnType);
    /// Return this signature with a new metadata attribute and (optionally)
    /// a new argListAttrs. If `argListAttrs` is null, the existing
    /// argListAttrs are preserved.
    FuncType getWithMetadata(FnMetadataAttrInterface metadata,
                             PogListAttr argListAttrs);

    /// Build a func type with empty conventions and effects and the given
    /// metadata and argument pog list.
    static FuncType get(FunctionType values, FnMetadataAttrInterface metadata,
                        PogListAttr argListAttrs);

    /// Return the number of arguments.
    size_t getNumArguments() { return getArguments().size(); }
    /// Return the number of results.
    size_t getNumResults() { return getResults().size(); }

    /// Calls to async functions don't provide values for the `byref_error` and
    /// `byref_result` arguments. If this is an async function, return the
    /// number of result slots it has. Otherwise, return 0.
    size_t getNumAsyncReturnSlots();

    /// Return the name for the argument at the specified index.
    StringAttr getArgName(size_t idx);

    /// Get the origin set of the capture lifetimes.
    TypedAttr getCaptureOrigins();

    /// Get whether nested lifetimes are excluded from exclusivity checking.
    bool getIsNestedOriginsReadOnly();

    /// Get whether the function establishes new interior origins.
    bool getDefinesInteriorOrigins();

    /// Get the number of implicit origin decls this function type carries.
    size_t getNumImplicitOriginDecls();

    /// Return the single MLIR result type for this signature.
    Type getResultType() { return getResults().front(); }

    /// Get the user-visible result type of the signature.
    Type getUserResultType();

    /// Get the user-visible thrown type for a raising function.
    Type getUserThrownType();

    /// Returns true if the argument at this index is any vararg or a pack.
    bool isAnyVarArg(size_t index);

    /// Returns true if the argument at this index is a positional vararg.
    bool isPosVarArg(size_t index);

    /// For a PosVarArg/PackVarArg, return the declared ArgConvention of the
    /// elements. For example: def x(mut *args: Int) is declared 'mut'.
    ArgConvention getVariadicConvention(size_t index);

    /// Returns true if the argument at this index is a keyword vararg.
    bool isKwVarArg(size_t index);

    /// Returns true if the argument at this index is a pack vararg.
    bool isPack(size_t index);

    /// If the specified argument is a variadic list/pack, return the
    /// VariadicList/VariadicPack, stripping RefType, otherwise return null.
    Type getIfVariadicListOrPack(size_t index);

    /// Returns the index of the pack variadic arg, or std::nullopt if none.
    std::optional<size_t> findPackVarArgIndex();

    /// Returns true if the signature has keyword variadic arguments.
    bool hasKwVarArgs();

    /// Substitute the specified implicit origin references into the value
    /// signature, replacing them with `values` if they are at depth 0, or
    /// decrementing their depth if not. Returns the resultant FunctionType on
    /// success; invokes `emitError` and returns null on error.
    FunctionType substituteImplicitOriginsIntoValues(
        ArrayRef<TypedAttr> values,
        function_ref<InFlightDiagnostic()> emitError);

    /// Return this signature with the specified capture lifetimes.
    FuncType getWithCaptureOrigins(TypedAttr lifetimes);
  }];

  let genVerifyDecl = 1;
}

def CalleeArgumentTypes : TypesMatchWith<"operands match callee argument types",
    "callee", "operands",
    "cast<FuncTypeGeneratorType>($_self).getBody().getArguments()">;
def CalleeResultTypes : TypesMatchWith<"results match callee result types",
    "callee", "results",
    "cast<FuncTypeGeneratorType>($_self).getBody().getResults()">;


//===----------------------------------------------------------------------===//
// FuncLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

def KGEN_FuncLiteralTypeGeneratorType : DialectType<KGENDialect,
    CPred<"llvm::isa<FuncLiteralTypeGeneratorType>($_self)">,
    "KGEN GeneratorType wrapping a FuncLiteralType", "FuncLiteralTypeGeneratorType">;

//===----------------------------------------------------------------------===//
// FuncLiteralType
//===----------------------------------------------------------------------===//

def KGEN_FuncLiteralType : KGENType<"FuncLiteral", "func.literal", [
  DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "The type of a KGEN literal function";
  let description = [{
    This type describes the type of a literal function in KGEN, in additional to
    the FuncType, it uniquely identifies the function by storing its name.
  }];

  // Encodes parameter expression that evaluates to a function literal.
  let parameters = (ins "TypedAttr":$funcLiteral);
  let assemblyFormat = [{
    `<`custom<TypeParamValue>($funcLiteral) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$funcLiteral), [{
      return $_get(funcLiteral.getContext(), funcLiteral);
    }]>,
  ];

  let extraClassDeclaration = [{
    FuncSymbolAttr getTargetLiteral() const;
    FuncType getFuncType() const { return cast<FuncType>(getFuncLiteral().getType()); }
  }];
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def PointerType : KGENType<"Pointer", "pointer", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized pointer type";
  let description = [{
    This type represents a pointer. The pointee type is parameterized with a
    `!kgen.type` type. An optional `addressSpace` argument can be
    specified (default to 0). An optional `nonnull` flag can be specified
    to indicate the pointer is guaranteed to never be null.

    Example:

    ```mlir
    // A pointer to a scalar.
    !kgen.pointer<scalar<f32>>

    // A pointer to a SIMD vector.
    !kgen.pointer<simd<4, f32>>

    // A parameterized scalar pointer.
    !kgen.pointer<scalar<type>>

    // A completely parameterized pointer.
    !kgen.pointer<elementType>

    // The address space parameter is optional, but can be specified.
    !kgen.pointer<scalar<si32>, 2>

    // The address space also works on parametrized pointers.
    !kgen.pointer<elementType, 5>

    // A non-null pointer with default address space).
    !kgen.pointer<scalar<f32>, nonnull>

    // A non-null pointer with explicit address space).
    !kgen.pointer<scalar<f32>, 1, nonnull>
    ```
  }];

  let parameters = (ins
    "Type":$elementType,
    DefaultValuedParameter<"TypedAttr",
                           "$_builder.getIndexAttr(0)">:$addressSpace,
    DefaultValuedParameter<"bool", "false">:$isNonNull);
  let assemblyFormat = "`<` custom<ParamType>($elementType) `` custom<PointerOptionals>($addressSpace, $isNonNull) `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        CArg<"unsigned", "0">:$addressSpace,
                                        CArg<"bool", "false">:$isNonNull)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$addressSpace,
                                        CArg<"bool", "false">:$isNonNull)>,
  ];

  let extraClassDeclaration = [{
    /// Get the element type as the provided type.
    template <typename ElTy>
    ElTy getElementAs() { return llvm::cast<ElTy>(getElementType()); }

    /// Returns the address space, or std::nullopt if unresolved.
    std::optional<unsigned> getAddrSpace() const;

    /// Returns the address space, or zero if unresolved.
    unsigned getAddrSpaceOrZero() const { return getAddrSpace().value_or(0); }
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class PointerElementType<string lhs, string rhs> : TypesMatchWith<
    "pointer element type", lhs, rhs,
    "llvm::cast<PointerType>($_self).getElementType()">;

class PointerOf<Type type> : ConfinedType<PointerType,
    [CPred<"llvm::cast<PointerType>($_self).getElementType()">,
     SubstLeaves<"$_self",
      "llvm::cast<PointerType>($_self).getElementType()",
      type.predicate>],
    "pointer to " # type.summary> {
  let cppType = "::M::KGEN::PointerType";
  let builderCall = !if(!empty(type.builderCall), "",
                       "PointerType::get(" # type.builderCall #")");
}

//===----------------------------------------------------------------------===//
// ParamListType
//===----------------------------------------------------------------------===//

def KGEN_ParamListType : KGENType<"ParamList", "param_list", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized variadic type";
  let description = [{
    The `!kgen.param_list` type represents a homogeneously typed list
    of zero or more elements.  It also includes the original argument convention
    so clients know if the input argument is supposed to be owned, read,
    mut, etc.

    Example:

    ```mlir
    // A param_list of scalar floats.
    !kgen.param_list<scalar<f32>>

    // A parameterized param_list of types.
    !kgen.param_list<type>
    ```
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = [{
    `<`custom<ParamType>($elementType) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class ParamListElementType<string lhs, string rhs> : TypesMatchWith<
    "value type matches variadic element type", lhs, rhs,
    "llvm::cast<KGEN::ParamListType>($_self).getElementType()",
    "typeRangeMatches">;

//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

def KGEN_NoneType : KGENType<"None", "none",
    [DeclareTypeInterfaceMethods<DataLayoutInterface>,
     DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "Type used for 'void' function results";
  let description = [{
    The `!kgen.none` type represents a value with no contents. When used as a
    function result, it is equivalent to a void result or no results.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// TargetType
//===----------------------------------------------------------------------===//

def KGEN_TargetType : KGENType<"Target", "target"> {
  let summary = "The type of a target.";
  let description = [{
    A `!kgen.target` is the type of a target. It is used for parameterizing
    kernels by target.

    Example:
    ```mlir
    kgen.generator @target_params<t: !kgen.target>() {
      ...
    }
    ```
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// BuildInfoType
//===----------------------------------------------------------------------===//

def KGEN_BuildInfoType : KGENType<"BuildInfo", "build_info"> {
  let summary = "The type of a build info.";
  let description = [{
    A `!kgen.build_info` is the type of a build info. It is used for
    parameterizing kernels by how they are built.

    Example:
    ```mlir
    kgen.generator @target_params<bi: !kgen.build_info>() {
      ...
    }
    ```
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

def KGEN_StructType : KGENType<"Struct", "struct", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<ContextuallyEvaluatedTypeInterface,
                                ["evaluateWithContext"]>]> {
  let summary = "parameterized struct type";
  let description = [{
    This type represents a struct. A struct contains element types arranged in
    their order of declaration and a flag indicating register-passability.

    This is the "anonymous" version of `!kgen.concrete_source_struct` that
    strips away all information except for those necessary for understanding the
    memory layout of the data it describes.

    The element types are stored as a `TypedAttr` which can be either:
    - A concrete `ParamListAttr` with resolved types
    - A parametric expression (e.g., a variadic parameter reference) before elaboration

    Example:

    ```mlir
    // A struct with primitive element types.
    !kgen.struct<(scalar<f32>, simd<4, ui64>)>

    // A memory-only struct with primitive element types.
    !kgen.struct<(scalar<f32>, simd<4, ui64>) memoryOnly>

    // A struct with nested types.
    !kgen.struct<(
      !kgen.pointer<simd<4, si8>>,
      !kgen.array<24, scalar<si64>>,
      !kgen.struct<(
        !kgen.scalar<f32>,
        !kgen.scalar<f64>
      )>
    )>

    // A struct with parameterized element types.
    !kgen.struct<(type, array<size, scalar<dtype>>)>

    // A struct parameterized on a variadic sequence of element types.
    kgen.generator @example<Ts: param_list<!kgen.type>>(
      %0: !kgen.struct<(Ts)>,
    ) { kgen.return }
    ```
  }];

  let parameters = (ins
    "TypedAttr":$elementTypesVariadic,
    // Whether this struct is memory-only (not register-passable). Uses
    // TypedAttr: BoolAttr for concrete types, or an i1-typed constraint
    // proposition (e.g., NOT(conforms_to(T, RegisterPassable))) for
    // conditional RegisterPassable conformance that folds per-instantiation.
    "TypedAttr":$isMemoryOnly,

    // Explicit minimum alignment specified via @align(N). A value of 1 means
    // no explicit alignment (the default). Uses TypedAttr to support future
    // parametric alignment (e.g., @align(n) where n is a struct parameter).
    // For now, this will always be an IntegerAttr with index type.
    DefaultValuedParameter<"TypedAttr",
                           "$_builder.getIndexAttr(1)">:$minAlignment,

    // This is true when this corresponds to a variadic parameter pack. This is
    // looked at by ArgConvention lowering to know when it can flatten a pack.
    "bool":$isParamPack
  );
  // Use hasCustomAssemblyFormat since we need to handle empty struct specially.
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  // Skip default builders - we provide custom builders.
  let skipDefaultBuilders = 1;
  let builders = [
    // Main builder with TypedAttr isMemoryOnly - implemented in KGENTypes.cpp.
    // minAlignment defaults to 1 (no explicit alignment) when TypedAttr{} is passed.
    TypeBuilder<(ins "TypedAttr":$variadic,
                     CArg<"TypedAttr", "{}">:$isMemoryOnly,
                     CArg<"TypedAttr", "{}">:$minAlignment,
                     CArg<"bool", "false">:$isParamPack), [{}]>,
    // Convenience builder with bool isMemoryOnly (wraps in BoolAttr).
    TypeBuilder<(ins "TypedAttr":$variadic,
                     "bool":$isMemoryOnly,
                     CArg<"TypedAttr", "{}">:$minAlignment,
                     CArg<"bool", "false">:$isParamPack), [{
      return get($_ctxt, variadic,
                 BoolAttr::get($_ctxt, isMemoryOnly), minAlignment,
                 isParamPack);
    }]>,
    // Builder with TypedAttr variadic (infers context from variadic).
    TypeBuilderWithInferredContext<(ins "TypedAttr":$variadic,
                                        CArg<"bool", "false">:$isMemoryOnly), [{
      return get(variadic.getContext(), variadic,
                 BoolAttr::get(variadic.getContext(), isMemoryOnly),
                 /*minAlignment=*/TypedAttr{});
    }]>,
    // Builders with ArrayRef<Type> - implementations in KGENTypes.cpp.
    // These convert types to ParamListAttr and route through the main builder.
    TypeBuilderWithInferredContext<(ins "ArrayRef<Type>":$types,
                                        CArg<"bool", "false">:$isMemoryOnly), [{}]>
  ];

  let extraClassDeclaration = [{
    /// Return true if the specified type is a NoneType or an empty struct.
    static bool isNoneOrEmpty(Type type);

    /// If the struct type is concrete (not parametric), returns the attribute
    /// representing the variadic sequence. Otherwise, returns a null attribute.
    ParamListAttr getVariadicIfResolved() const;

    /// Returns whether this struct type is resolved (element types are concrete).
    bool isResolved() const;

    /// Get the element types. Returns std::nullopt if the struct is parametric.
    std::optional<SmallVector<Type>> getElementTypes() const;

    /// Get the number of struct elements. Returns std::nullopt if parametric.
    std::optional<size_t> getNumElements() const;

    /// True iff this struct's layout is known to use the memory-only ABI
    /// (not register-passable). Reads `isMemoryOnly`: a concrete `BoolAttr`
    /// means the answer is final; an i1 constraint proposition (conditional
    /// `RegisterPassable`) is not resolved here — this returns true so passes
    /// that cannot evaluate the proposition yet still choose the conservative
    /// (memory) path. After elaboration, propositions fold to `BoolAttr` and
    /// this reflects the instantiated convention.
    bool isDefinitelyMemoryOnly() const;

    /// Builders with explicit context.
    static StructType get(MLIRContext *context, ArrayRef<Type> types,
                          bool isMemoryOnly = false,
                          TypedAttr minAlignment = {});

    /// Builder with TypedAttr isMemoryOnly for constraint propositions.
    static StructType get(MLIRContext *context, ArrayRef<Type> types,
                          TypedAttr isMemoryOnly,
                          TypedAttr minAlignment = {});

    /// Checked builder with explicit context (for empty structs).
    static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                 MLIRContext *context, ArrayRef<Type> types,
                                 bool isMemoryOnly = false);
  }];
}

//===----------------------------------------------------------------------===//
// StructInstanceType
//===----------------------------------------------------------------------===//

def KGEN_StructInstanceType : KGENType<"StructInstance", "struct_inst"> {
  let summary = "concretized mojo struct type";
  let description = [{
    This type represents a concretized mojo source struct type. It is the result
    of flattening an AppliedStructType.

    Example:

    ```mlir
    // A trivially-concretized struct with no parameters and no fields.
    !kgen.struct_inst<"Foo">

    // A parameterized memory-only struct with primitive element types.
    !kgen.struct_inst<
      "Bar"
      [elemT]
      <:dtype f32>
      (first: !kgen.scalar<f32>, second: !kgen.scalar<f32>)
      memoryOnly
    >
    ```
  }];

  let parameters = (ins
    "StringAttr":$name,
    OptionalArrayRefParameter<"StringAttr">:$paramNames,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    OptionalArrayRefParameter<"StructDefFieldAttr">:$fields,
    // Whether this struct is memory-only. Uses TypedAttr: BoolAttr for
    // concrete types, or an i1-typed constraint proposition for conditional
    // RegisterPassable conformance.
    "TypedAttr":$isMemoryOnly
  );

  let assemblyFormat = [{
    `<` $name
        (`[` custom<ParamNames>($paramNames)^ `]`)?
        `` custom<ParameterValues>($paramValues)
        (`(` custom<StructDefFields>($fields)^ `)`)?
        `` custom<IsMemoryOnly>($isMemoryOnly)
    `>`
  }];

  let builders = [
    // Builder with TypedAttr isMemoryOnly.
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name,
        "ArrayRef<StringAttr>":$paramNames,
        "ArrayRef<TypedAttr>":$paramValues,
        "ArrayRef<StructDefFieldAttr>":$fields,
        "TypedAttr":$isMemoryOnly)>,
    // Convenience builder with bool isMemoryOnly (wraps in BoolAttr).
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name,
        "ArrayRef<StringAttr>":$paramNames,
        "ArrayRef<TypedAttr>":$paramValues,
        "ArrayRef<StructDefFieldAttr>":$fields,
        "bool":$isMemoryOnly), [{
      return get(name.getContext(), name, paramNames, paramValues, fields,
                 BoolAttr::get(name.getContext(), isMemoryOnly));
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// VariantType
//===----------------------------------------------------------------------===//

def KGEN_VariantType : KGENType<"Variant", "variant", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>
]> {
  let summary = "A parametric variant type.";
  let description = [{
    A `!kgen.variant` type is a structural variant type that represents a value
    that is one of a list of types (discriminated union).

    Variants of zero types or one type are allowed, e.g. `!kgen.variant<i32>`,
    and all variant operations are defined on the even though they aren't
    particularly useful types.

    Example:

    ```mlir
    !kgen.variant<i32, scalar<dtype>, T>
    ```
  }];

  let parameters = (ins "TypedAttr":$variadic);
  let assemblyFormat = "`<` custom<VariantTypes>($variadic) `>`";
  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilder<(ins "TypedAttr":$variadic)>,
    TypeBuilderWithInferredContext<(ins "ArrayRef<Type>":$types)>
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the type from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static VariantType getFromBytecode(TypedAttr variadic);

    /// Get an element type as a parameterized type.
    Type getType(unsigned index) const;

    /// Return the number of types in the variant.
    size_t getNumTypes() const;

    /// Get the number of bits allocated for the discriminator.
    size_t getDiscrSizeInBits() const;
    /// Get the size of the content section of the variant.
    std::optional<int64_t> getContentSize(TargetInfoAttr target) const;

    /// Get the concrete variant types.
    llvm::iterator_range<llvm::mapped_iterator<ArrayRef<TypedAttr>::iterator,
                                               Type (*)(TypedAttr)>>
    getTypes() const;
  }];
}

//===----------------------------------------------------------------------===//
// DeferredType
//===----------------------------------------------------------------------===//

def KGEN_DeferredType : KGENType<"Deferred", "deferred",
                          [ DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "Represents a deferred type";
  let description = [{
    A `!kgen.deferred` type represents a deferred attribute that will be
    concretized at elaborator

    Example:

    ```mlir
    #kgen<deferred #index<cmp_predicate sle>> : !kgen.deferred
    #kgen.param.decl.ref<"(lifted)apply_0"> : !kgen.deferred
    ```
  }];
  let assemblyFormat = "";
}


//===----------------------------------------------------------------------===//
// MLIRDeferredType
//===----------------------------------------------------------------------===//

def KGEN_MLIRDeferredType : KGENType<"MLIRDeferred", "deferred_type",
    [DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "Represents a type whose MLIR form is deferred until elaboration";
  let description = [{
    A `!kgen.deferred_type` wraps an `#kgen.attr_ctor_deferred` attribute that,
    once all parameters are concrete, is concatenated into a type string and
    parsed by the elaborator to produce the actual MLIR type.

    This type is produced by the Mojo parser when `__mlir_deferred_type[...]`
    is used as a function return type and the type string cannot be parsed at
    parse time (because it references not-yet-concrete parameters).

    Example:

    ```mlir
    !kgen.deferred_type<#kgen.attr_ctor_deferred("llvm.array<", ...)>
    ```
  }];
  let parameters = (ins "Attribute":$attr);
  let assemblyFormat = "`<` $attr `>`";
}

//===----------------------------------------------------------------------===//
// NeverType
//===----------------------------------------------------------------------===//

def KGEN_NeverType : KGENType<"Never", "never", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "Represents a type that cannot be constructed";
  let description = [{
    A `!kgen.never` type represents a type that cannot be
    constructed. This is used to represent functions that never return or
    generic thrown types that resolve to a concrete type of "not actually
    thrown".
    ```
  }];
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

def KGEN_SIMDType : KGENType<"SIMD", "simd", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized SIMD vector type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.
  }];

  let parameters = (ins "TypedAttr":$size, "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` ` `
        custom<DTypeParamValue>($dType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size, "TypedAttr":$dtype)>,
    TypeBuilderWithInferredContext<(ins "int64_t":$size, "TypedAttr":$dtype)>,
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size, "KGEN::KGENDType":$dtype)>,
    TypeBuilder<(ins "int64_t":$size, "KGEN::KGENDType":$dtype)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the dtype expression of the SIMD type to a value.
    std::optional<KGEN::KGENDType> getResolvedDType() const;
    /// Try to narrow the size expression of the SIMD type to an integer.
    std::optional<int64_t> getResolvedSize() const;
    /// Returns true if this is known to be a scalar type.
    bool isScalar() const { return getResolvedSize() == 1; }
    /// Returns the `!kgen.scalar<bool>` type (a SIMD of size 1 and bool dtype).
    static SIMDType getScalarBoolType(MLIRContext *ctx);
  }];
}

//===----------------------------------------------------------------------===//
// SIMD type constraints
//===----------------------------------------------------------------------===//

/// A DType constraint.
class DTypeConstraint<string value> {
  string pred = value;
}

class AndDType<list<DTypeConstraint> preds> :
    DTypeConstraint<"(" # !interleave(!foreach(p, preds, p.pred), " && ") # ")">;
class OrDType<list<DTypeConstraint> preds> :
    DTypeConstraint<"(" # !interleave(!foreach(p, preds, p.pred), " || ") # ")">;

def BoundDType : DTypeConstraint<"$_self">;
def UnboundDType : DTypeConstraint<"!$_self">;
def BoolDType : DTypeConstraint<"$_self->isBool()">;
def IndexDType : DTypeConstraint<"$_self->isIndex()">;
def UIndexDType : DTypeConstraint<"$_self->isUIndex()">;
def IntDType : DTypeConstraint<"$_self->isInt()">;
def SIntDType : DTypeConstraint<"$_self->isSInt()">;
def UIntDType : DTypeConstraint<"$_self->isUInt()">;
def FloatDType : DTypeConstraint<"$_self->isFloat()">;
def UI8DType : DTypeConstraint<"*$_self == KGEN::KGENDType::ui8">;
def SI8DType : DTypeConstraint<"*$_self == KGEN::KGENDType::si8">;
def SI64DType : DTypeConstraint<"*$_self == KGEN::KGENDType::si64">;
def AddressDType : DTypeConstraint<"$_self->isAddress()">;
def ArithmeticDType : DTypeConstraint<"$_self->isArithmetic()">;

class BoundAnd<DTypeConstraint pred> : AndDType<[BoundDType, pred]>;
class UnboundOr<DTypeConstraint pred> : OrDType<[UnboundDType, pred]>;

/// A SIMD type whose dtype conforms to a constraint.
class SIMDOfDType<DTypeConstraint constraint, string desc> :
  ConfinedType<KGEN_SIMDType,
    [CPred<!subst("$_self", "llvm::cast<KGEN::SIMDType>($_self).getResolvedDType()",
                 constraint.pred)>], desc> {
  let cppType = "::M::KGEN::SIMDType";
}

def BoolSIMD : SIMDOfDType<BoundAnd<BoolDType>, "!kgen.simd<N, bool>">;
def UnboundSIMD : SIMDOfDType<UnboundDType, "!kgen.simd<N, invalid>">;
def IndexSIMD : SIMDOfDType<BoundAnd<IndexDType>, "!kgen.simd<N, index>">;
def UIndexSIMD : SIMDOfDType<BoundAnd<UIndexDType>, "!kgen.simd<N, uindex>">;
def IntegerSIMD : SIMDOfDType<BoundAnd<IntDType>, "!kgen.simd<N, integral>">;
def FloatSIMD : SIMDOfDType<BoundAnd<FloatDType>, "!kgen.simd<N, float>">;
def AddressSIMD : SIMDOfDType<BoundAnd<AddressDType>, "!kgen.simd<N, address>">;

def UnboundOrIndexOrIntegerSIMD : SIMDOfDType<
    UnboundOr<OrDType<[IndexDType, UIndexDType, IntDType]>>,
    "SIMD type whose element type is either unbound, index, uindex, or integral dtype">;
def UnboundOrIndexOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<[IndexDType, UIndexDType, ArithmeticDType]>>,
    "SIMD type whose element type is either unbound, index, uindex, integer or float dtype">;

// TODO: the following two should be unified, but may be subtly different in the
// current implementation.
def UnboundOrBoolOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<[BoolDType, ArithmeticDType]>>,
    "SIMD type whose element type is either unbound, boolean, integral, or float dtype">;
def UnboundOrBoolOrIndexOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<[BoolDType, IndexDType, UIndexDType, IntDType, FloatDType]>>,
    "SIMD type whose element type is either unbound, boolean, index, uindex, integral, or float dtype">;

/// A "scalar" type is a SIMD of size 1.
def ScalarType : ConfinedType<KGEN_SIMDType,
    [CPred<"llvm::cast<KGEN::SIMDType>($_self).getResolvedSize() && "
           "*llvm::cast<KGEN::SIMDType>($_self).getResolvedSize() == 1">],
    "SIMD with size 1">;

/// A scalar type whose dtype conforms to a constraint.
class ScalarOfDType<DTypeConstraint constraint, string desc> :
  AllOfType<[SIMDOfDType<constraint, desc>, ScalarType], desc>;

def BoolScalar : ScalarOfDType<BoundAnd<BoolDType>, "!kgen.scalar<bool>">,
    BuildableType<"KGEN::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                  "$_builder.getContext(), KGEN::KGENDType::kBool))">;
def UnboundOrArithmeticScalar : ScalarOfDType<UnboundOr<ArithmeticDType>,
    "whose type is an arithmetic dtype">;

def UI8Scalar : ScalarOfDType<BoundAnd<UI8DType>, "!kgen.scalar<ui8>">,
    BuildableType<"KGEN::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGEN::KGENDType::ui8))">;
def SI8Scalar : ScalarOfDType<BoundAnd<SI8DType>, "!kgen.scalar<si8>">,
    BuildableType<"KGEN::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGEN::KGENDType::si8))">;

def SI64Scalar : ScalarOfDType<BoundAnd<SI64DType>, "!kgen.scalar<si64>">,
    BuildableType<"KGEN::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGEN::KGENDType::si64))">;

def UIndexScalar : ScalarOfDType<BoundAnd<UIndexDType>, "!kgen.scalar<uindex>">,
    BuildableType<"KGEN::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGEN::KGENDType::uindex))">;

def IndexScalar : ScalarOfDType<BoundAnd<IndexDType>, "!kgen.scalar<index>">,
    BuildableType<"KGEN::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGEN::KGENDType::index))">;

def UnboundOrIndexOrInt : AnyTypeOf<[
    ScalarOfDType<UnboundOr<OrDType<[IndexDType, UIndexDType, IntDType]>>,
                  "whose type is unbound, index, or integral dtype">]>;

//===----------------------------------------------------------------------===//
// SIMD Type Constraint Traits
//===----------------------------------------------------------------------===//

// Matches if Value has size=1 and same dtype as Operand
class SIMDSizeOneOfSameDType<string operand, string value> : TypesMatchWith<
    value # " type matches dtype of the " # operand # " operand", operand,
    value, "KGEN::SIMDType::get(1, llvm::cast<SIMDType>($_self).getDType())">;

class SIMDSize<string name> :
    StrFunc<"$" # name # ".getType().cast<KGEN::SIMDType>().getResolvedSize()">;

class SIMDOfSameSize<string lhs, string rhs> :
  AllMatchSameOperatorTrait<[lhs, rhs], SIMDSize<"_self">.result, "size">;

class AddressSIMDOfSameSize<string operand, string value> : TypesMatchWith<
    value # " type matches simd type of the " # operand # " operand", operand,
    value, "KGEN::SIMDType::get(llvm::cast<SIMDType>($_self).getSize(), "
    "DTypeConstantAttr::get($_self.getContext(), KGEN::KGENDType::address))">;

class BoolSIMDOfSameSize<string operand, string value> : TypesMatchWith<
    value # " type matches simd type of the " # operand # " operand", operand,
    value, "KGEN::SIMDType::get(llvm::cast<SIMDType>($_self).getSize(), "
    "DTypeConstantAttr::get($_self.getContext(), KGEN::KGENDType::kBool))">;

#endif // KGEN_KGENDIALECT_KGENTYPES_TD
