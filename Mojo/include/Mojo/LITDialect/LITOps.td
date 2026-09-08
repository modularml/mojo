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
// This defines operations in the LIT dialect.
//
//===----------------------------------------------------------------------===//

include "Mojo/LITDialect/LITAttrs.td"
include "Mojo/LITDialect/LITEnums.td"
include "Mojo/LITDialect/LITDialect.td"
include "Mojo/LITDialect/LITInterfaces.td"
include "Mojo/LITDialect/LITTypes.td"
include "Mojo/CODialect/COTypes.td"
include "Mojo/KGENDialect/KGENAttrs.td"
include "Mojo/KGENDialect/KGENEnums.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Mojo/POPDialect/POPTypes.td"
include "Mojo/HLCFDialect/HLCFAttrs.td"
include "Mojo/HLCFDialect/HLCFInterfaces.td"
include "Support/DebugInfoDialect/IR/DebugInfoAttrs.td"
include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.td"
include "Support/MDialect/MAttrs.td"
include "Support/Compiler/Properties.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class LITOp<string mnemonic, list<Trait> traits = []> :
    Op<LITDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FnOp
//===----------------------------------------------------------------------===//

def LIT_FnOp : LITOp<"fn", [
    FunctionLike, FunctionOpInterface, SymbolName, SymbolVisibility,
    SymbolTable, SingleBlock,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["collectParameterUses"]>,
    ASTDeclInterface, DeclInterface, FuncInterface, ExportInterface,
    StabilityDecoratorInterface]> {
  let summary = "Mojo function definition";
  let description = "The `lit.fn` operation models a fn/def declaration.";

  let arguments = (ins
    OptionalAttr<SymbolNameAttr>:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<KGEN_ParamDeclAttr>:$paramDecl,
    TypeAttrOf<LIT_FnTypeGeneratorType>:$funcTypeGenerator,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$params,
    KGEN_DecoratorsAttr:$decorators,
    UnitAttr:$isStatic,
    UnitAttr:$synthetic,
    DefaultValuedAttr<LIT_ImplicitConversionKindAttr,
      "ImplicitConversionKind::None">:$implicitConversion,
    UnitAttr:$external,
    UnitAttr:$defaultedTraitFn, // Present for trait methods with non-empty bodies.
    KGEN_ExportKindAttr:$exportKind,
    KGEN_InlineLevelAttr:$inlineLevel,
    DefaultValuedOptionalAttr<I8Attr, "0">:$specialFnKind,
    // The linkage name of this function within the compiled artifact. If null,
    // the flattened full symbol reference of the function will be used.
    OptionalAttr<KGEN_LinkageNameAttr>:$linkageName,
    // The name of this function in the pre-elaborated IR stage.
    OptionalAttr<StrAttr>:$sourceName,
    // The trait this function was inherited from, for both trait->trait and
    // trait->struct inheritance. None if self declared.
    OptionalAttr<KGEN_TraitSymbolAttr>:$inheritedFrom,
    // For trait->struct inheritance, the `lit.fn` this was inherited from,
    // which is the trait's default implementation the wrapper forwards to.
    // `inheritedFrom` only names the trait, which is not enough to resolve
    // that declaration.
    OptionalAttr<SymbolRefAttr>:$defaultFnRef,
    OptionalAttr<StrAttr>:$namedResult,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<LIT_DeprecationInfoAttr>:$deprecationInfo,
    OptionalAttr<LIT_UnavailableInfoAttr>:$unavailableInfo,
    // Present if the decl has an @stable decorator on it.
    UnitAttr:$hasStableDecorator,
    // Present if the decl has an @stable(since="version") decorator.
    OptionalAttr<StrAttr>:$stableSinceVersion,
    // To be able to store pre-elaborated names of the `llvm_metadata`,
    // decompose LLVM metadata into array of attributes of even size.
    // Each even attribute corresponds to the name of in `llvm_metadata`,
    // each odd corresponds to its value (zero index based)
    DefaultValuedAttr<ArrayAttr, "{}">:$LLVMMetadataArray,
    DefaultValuedAttr<ArrayAttr, "{}">:$LLVMArgMetadataArray,
    OptionalAttr<AnyAttr>:$thunkKey,
    UnitAttr:$noDocRequired
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    ArrayRef<ParamDeclAttr> getInputParams() { return getParams(); }
    Block *getBody() { return &getFunctionBody().back(); }
    Region *getCallableRegion() { return &getBodyRegion(); }
    bool isOptionalSymbol() { return getParamDecl().has_value(); }

    /// Returns the result type of the MLIR function (not the user defined
    /// result type).  This is guaranteed to be present.
    Type getMLIRResultType() { return getResultTypes().front(); }

    /// Returns the user-defined result type, looking through implicit memory
    /// results and stripping off the variant from error throwing results if
    /// needed.
    Type getUserResultType();

    /// Return a reference for this function, optionally bound to a set of
    /// parameter bindings.
    TypedAttr getBoundReference(ParameterEvaluationContext &evalContext,
                                ParameterExprArrayAttr bindings = {});
    SymbolConstantAttr
    getBoundSymbolRef(ParameterEvaluationContext &evalContext,
                      ParameterExprArrayAttr bindings = {});

    /// Return a GeneratorAttr that generates a FuncLiteralAttr for this
    /// function. `dischargedBodyConstraints` is an optional bitmask over the
    /// function's body constraints: a set bit drops the matching constraint
    /// from the result type. Empty (the default) retains all body constraints.
    TypedAttr
    getFuncLiteralGenerator(ParameterEvaluationContext &evalContext,
                            ParameterExprArrayAttr bindings = {},
                            const llvm::BitVector &dischargedBodyConstraints =
                                llvm::BitVector());

    /// Return the SpecialFunctionKind ID that indicates if this is a special
    /// function like __init__ or __radd__.
    SpecialFunctionKind getSpecialFunctionKind();
    const SpecialFunctionInfo &getSpecialFunctionInfo();

    /// Return the function name as it appears in the source code (not mangled).
    StringAttr getDeclName() { return getSourceNameAttr(); }

    /// Return the full signature of this declaration, including parameters from
    /// enclosing struct declarations.
    FnTypeGeneratorType getFullSignature();

    /// Return a collection of the ParamDeclAttr's that are declared on this
    /// function and any enclosing unit, ordered as will be required by a call.
    SmallVector<ParamDeclAttr> collectAllParams(bool includeImplOrigins);

    /// Return whether the function is compiler generated or not.
    bool isSynthetic();

    /// Override the default implementation of isExternal.
    bool isExternal() { return getExternal(); }

    /// Return true if this is a defaulted trait method.
    bool isDefaultedTraitFn() { return getDefaultedTraitFn(); }

    /// Return true if this is an implicit conversion function.
    bool isImplicitConversion() {
      return getImplicitConversion() != ImplicitConversionKind::None;
    }

    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr attrs){}
    void setResAttrsAttr(mlir::ArrayAttr attrs){}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];

  let builders = [
    // Create a func with a region and a body block, but no terminator.
    OpBuilder<(ins "StringAttr":$name, "StringAttr":$sourceName,
                   "FuncTypeGeneratorType":$funcTypeGenerator)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def LIT_CallOp : LITOp<"call", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>]> {
  let summary = "call a function";
  let description = [{
    The `lit.call` operation is a call to comptime parameter of function type.
    It contains optional parameter declarations for
    any callee result parameters and bindings for implicit origin parameters.

    Example:

    ```mlir
    lit.call @callee[*"a`"]<1, :dtype f32 -> out: type>(%arg0) : (!Int) -> ()
    ```
  }];

  let arguments = (ins
    AttrOfType<LIT_FnTypeGeneratorType>:$callee,
    KGEN_ParameterExprArrayAttr:$implicitOrigins,
    Variadic<AnyType>:$operands,
    DefaultValuedAttr<KGEN_TailKindAttr, "{}">:$tailKind
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `` custom<OptionalEnum>($tailKind, "symbolizeTailKind")
    `` custom<CallOp>($callee, $implicitOrigins, $operands,
                      type($operands), type($results)) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get the direct callee symbol if this is a direct call.
    SymbolRefAttr getDirectCallee();
    /// Get the callee signature type.
    FnTypeGeneratorType getCalleeType() {
      return cast<FnTypeGeneratorType>(getCallee().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def LIT_CallIndirectOp : LITOp<"call_indirect"> {
  let summary = "indirect call";
  let description = [{
    The `lit.call_indirect` operation is an indirect call to a signature-typed
    dynamic callee. It references its callee via an SSA operand. It contains
    optional bindings for implicit origin parameters.

    Example:

    ```mlir
    lit.call_indirect %callee[*"x`"](%x) : (!lit.ref<mut index, *"x`">) -> ()
    ```
  }];

  let arguments = (ins
    LIT_FnTypeGeneratorType:$callee,
    KGEN_ParameterExprArrayAttr:$implicitOrigins,
    Variadic<AnyType>:$arguments,
    DefaultValuedAttr<KGEN_TailKindAttr, "{}">:$tailKind
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `` custom<OptionalEnum>($tailKind, "symbolizeTailKind")
    $callee `` custom<OriginParams>($implicitOrigins) `(` $arguments `)` `:`
    custom<Generator>(type($callee)) attr-dict ``
    custom<CallOpTypes>(type($arguments), type($results), ref(type($callee)),
                        ref($implicitOrigins))
  }];

  let extraClassDeclaration = [{
    FnTypeGeneratorType getCalleeType() { return getCallee().getType(); }
  }];
}

//===----------------------------------------------------------------------===//
// BindParamsOp
//===----------------------------------------------------------------------===//

def LIT_BindParamsOp : LITOp<"bind_params", [Pure, InferTypeOpAdaptor]> {
  let summary = "Bind params onto an SSA value with function generator type.";
  let description = [{
    The `lit.bind_params` operation binds compile-time parameters onto an SSA
    value whose type is a function generator type. Partial and out-of-order
    parameter binding is allowed by specifying the unbound positions with
    `#kgen.unbound`.

    This is the SSA-value counterpart of `#kgen.bind_params`. All input
    parameters must be fully bound before the result can be used with
    `lit.call_indirect`.

    Example:

    Given `@SIMD : !lit.generator<<"dtype": !DType, "width": !Int>() -> !T>`,
    binding the width parameter but not the dtype:

    ```mlir
    %bound = lit.bind_params %fn : !lit.generator<<"dtype": !DType, "width": !Int>() -> !T>, ?, 2
      to !lit.generator<<"dtype": !DType>() -> !T>
    lit.call_indirect %bound(%arg) : ...
    ```
  }];

  let arguments = (ins
    LIT_FnTypeGeneratorType:$generator,
    KGEN_ParameterExprArrayAttr:$paramValues,
    OptionalAttr<DenseBoolArrayAttr>:$discharged
  );
  let results = (outs LIT_FnTypeGeneratorType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Derive the result type from the generator and parameter values.
    /// Parameter values are normalized to the generator's input arity: missing
    /// trailing values are treated as unbound and extra trailing values are
    /// ignored. `discharged` is an optional bitmask over the generator's body
    /// constraints: a set bit drops the matching constraint from the result
    /// type. Null (the default) retains all body constraints.
    static Type inferResultType(FnTypeGeneratorType generator,
                                ArrayRef<TypedAttr> paramValues,
                                DenseBoolArrayAttr discharged = {});
  }];
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

def LIT_FileModuleOp : LITOp<"file_module", [
    ASTDeclInterface, IsolatedFromAbove, NoRegionArguments, SymbolTable,
    SymbolName, SymbolVisibility, Symbol,
    DeclareOpInterfaceMethods<DeclInterface>,
  ] # GraphRegionNoTerminator.traits> {
  let summary = "source file module";
  let description = [{
    A `lit.file_module` models a file module at the lit source level. It
    contains any functions, modules, variables, etc. defined within the module.

    Example:

    ```mlir
    lit.file_module @bar {
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<LIT_DocStringAttr>:$docString
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    /// Return the file name as it appears in the source code (not mangled).
    StringAttr getDeclName() { return getSymNameAttr(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// PackageOp
//===----------------------------------------------------------------------===//

def LIT_PackageOp : LITOp<"package", [
    ASTDeclInterface, IsolatedFromAbove, NoRegionArguments, SymbolTable,
    SymbolName, SymbolVisibility, Symbol,
    DeclareOpInterfaceMethods<DeclInterface>,
  ] # GraphRegionNoTerminator.traits> {
  let summary = "mojo source package";
  let description = [{
    A `lit.package` models a mojo package. It contains a set of additional
    `lit.file_module` and `lit.package` operations.

    If the package was precompiled it may also contain:
    - The set of link dependencies for this package, `dependencies`.

    Example:

    ```mlir
    lit.package @foo {
      lit.file_module @bar {
      }
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    // References to the dependencies of this package.
    OptionalAttr<KGEN_LinkDependencyArrayAttr>:$dependencies,
    OptionalAttr<LIT_DenseResourceElementsArrayAttr>:$externLLVMBitcodeModules
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    /// Return the package name as it appears in the source code (not mangled).
    StringAttr getDeclName() { return getSymNameAttr(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

def LIT_StructDeclOp : LITOp<"struct.decl", [
    IsolatedFromAbove, SymbolName, SymbolVisibility, Symbol, SymbolTable,
    DeclInterface, SingleBlock,
    ASTDeclInterface, NoTerminator,
    DeclareOpInterfaceMethods<StructDeclInterface>,
    DeclareOpInterfaceMethods<DebugInfo_ScopedLocation>,
    StabilityDecoratorInterface
  ]> {
  let summary = "Struct declaration.";
  let description = [{
    The `lit.struct.decl` operation is a top-level operation used to declare
    parametric LIT struct types. Struct declarations are named with a symbol
    and have a parameter declaration list. The body contains
    `lit.struct.field` operations that declare the contents of the struct.

    Example:

    ```mlir
    lit.struct.decl @Buffer<size, dtype: dtype> {
      lit.struct.field size : index
      lit.struct.field ptr : !kgen.pointer<scalar<dtype>>
      lit.struct.field dtype : !kgen.dtype
    }
    ```

    The `registerPassable` is used to indicate structs that are preferably
    passed around as SSA values instead of being processed in memory.  This is
    true for simple values like F32 and Bool but not for complex aggregates like
    a SmallVector that can't be bitwise copied around or a Dictionary that
    shouldn't be.

    This includes a 'moveInit' field if the type implements the move ctor special
    functions if present.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<LIT_TypeSignatureType>:$signature,
    KGEN_ParamDeclArrayAttr:$params,
    KGEN_DecoratorsAttr:$decorators,
    TypeAttrOf<LIT_TraitType>:$canonicalTrait,
    UnitAttr:$synthetic,
    OptionalAttr<TypeAttr>:$nonmaterializableTarget,
    OptionalAttr<KGEN_SymbolConstantAttr>:$moveInit,
    OptionalAttr<KGEN_SymbolConstantAttr>:$copyInit,
    // This is the error message to produce if a non-destructible type needs an
    // implicit destructor call.
    OptionalAttr<StrAttr>:$linearTypeErrorMsg,
    OptionalAttr<TypeAttrOf<KGEN_FuncTypeGeneratorType>>:$closureSignature,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<LIT_DeprecationInfoAttr>:$deprecationInfo,
    OptionalAttr<LIT_UnavailableInfoAttr>:$unavailableInfo,
    // Present if the decl has an @stable decorator on it.
    UnitAttr:$hasStableDecorator,
    // Present if the decl has an @stable(since="version") decorator.
    OptionalAttr<StrAttr>:$stableSinceVersion,
    OptionalAttr<SourceNameAttr>:$sourceName,
    // Explicit minimum alignment specified via @align(N) decorator.
    OptionalAttr<AttrOfType<Index>>:$minAlignment,
    EnumProp<LIT_TypeConvention>:$convention,
    // If definesClosure attribute is present,
    // then this is a compiler generated structure
    // used to wrap a function as a closure.
    UnitAttr:$definesClosure,
    // Present when the struct has a conditional RegisterPassable conformance.
    // Stores the ConstraintAttr whose proposition governs whether the struct
    // is register-passable for a given instantiation.
    OptionalAttr<KGEN_ConstraintAttr>:$registerPassableConstraint
  );
  let regions = (region SizedRegion<1>:$fields);
  let assemblyFormat = [{
    $sym_name ``
    custom<StructParameterSpec>($params, $signature, $canonicalTrait) ``
    custom<TypeConvention>($convention) attr-dict-with-keyword
    (`\n` `move` custom<Symbol>($moveInit)^ ``)?
    (`\n` `copy` custom<Symbol>($copyInit)^ ``)?
    custom<OptionalDecorators>($decorators) $fields
  }];

  let extraClassDeclaration = [{
    /// Get the field declarations.
    // These are templates to defer the instantiation of op_iterator with
    // StructFieldOp until the call site, where StructFieldOp is complete.
    // This avoids an incomplete type error on macOS where libc++'s
    // invoke_result_t eagerly requires complete types.
    template <typename FieldOp = StructFieldOp>
    auto getFieldDecls() { return getFields().template getOps<FieldOp>(); }
    template <typename FieldOp = StructFieldOp>
    auto field_begin() { return getFieldDecls<FieldOp>().begin(); }
    template <typename FieldOp = StructFieldOp>
    auto field_end() { return getFieldDecls<FieldOp>().end(); }

    bool isRegisterPassable() {
      return getConvention() != TypeConvention::MemoryOnly;
    }
    bool isRegisterPassableTrivial() {
      return getConvention() == TypeConvention::RegisterPassableTrivial;
    }

    /// Return the struct name as it appears in the source code (not mangled).
    StringAttr getDeclName() { return getSymNameAttr(); }

    /// Get a bound reference to the struct type.
    StructType bindReference(ArrayRef<TypedAttr> paramValues = {});

    /// Get whether this struct is compiler generated.
    bool isSynthetic();
  }];

  let builders = [
    // Create a struct with a region but no body block. The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name)>,
  ];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructFieldOp
//===----------------------------------------------------------------------===//

def LIT_StructFieldOp : LITOp<"struct.field", [
    ASTDeclInterface, HasParent<"StructDeclOp">
  ]> {
  let summary = "Struct field declaration.";
  let description = [{
    The `lit.struct.field` operations declares a named struct field and type.

    Example:

    ```mlir
    // Declare a field named "foo".
    lit.struct.field foo : !kgen.scalar<f32>
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    TypeAttr:$type,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    // Set if the field has an @doc_hidden decorator on it.
    UnitAttr:$isDocHidden,
    // Set if the field allows legacy AnyOrigin usage.
    UnitAttr:$allowLegacyAnyOrigin
  );
  let assemblyFormat = "custom<KeywordAsString>($name) attr-dict `:` $type";

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "Type":$type)>,
    OpBuilder<(ins "const Twine &":$name, "Type":$type)>,
  ];

  let extraClassDeclaration = [{
    /// Given a struct base value type that may have bound parameters,
    /// substitute those bound parameters into our type and return the result.
    Type getReboundType(StructType structSelfType,
                        ParameterEvaluationContext *ctx = nullptr);
  }];
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

def LIT_StructInsertOp : LITOp<"struct.insert", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    AllTypesMatch<["container", "result"]>]> {
  let summary = "Insert a value into a struct.";
  let description = [{
    The `lit.struct.insert` operation inserts a value into a parametric struct.
    The original struct is immutable, so a new copy with the inserted value is
    returned.

    Example:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : !kgen.scalar<T>
    }

    %0 = lit.struct.insert %fieldVal, %struct[a]
      : !kgen.scalar<f64> into !lit.struct<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins StrAttr:$field, AnyType:$value,
                       LIT_StructType:$container);
  let results = (outs LIT_StructType:$result);
  let assemblyFormat = [{
    $value `,` $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `into` qualified(type($container))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

def LIT_StructExtractOp : LITOp<"struct.extract", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Extract a value from a struct.";
  let description = [{
    The `lit.struct.extract` operation extracts the value of a field by name
    from a parametric struct.

    Example:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : !kgen.scalar<T>
    }

    %0 = lit.struct.extract %struct[a]
      : !kgen.scalar<f64> from !lit.struct<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins LIT_StructType:$container, StrAttr:$field);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `from` qualified(type($container))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ExtensionDeclOp
//===----------------------------------------------------------------------===//

def LIT_ExtensionDeclOp : LITOp<"extension.decl", [
    IsolatedFromAbove, SymbolName, SymbolVisibility, Symbol, SymbolTable,
    DeclInterface, SingleBlock,
    ASTDeclInterface, NoTerminator,
    DeclareOpInterfaceMethods<DebugInfo_ScopedLocation>
  ]> {
  let summary = "Extension declaration.";
  let description = [{
    The `lit.extension.decl` operation is a top-level operation used to declare
    LIT extensions. Extension declarations are named with an auto-generated
    unique symbol.
    The body contains `lit.fn` operations that declare the methods/aliases/etc.
    that will be usable with the struct that's being extended.
    In the parser, this can be known to its parent ASTDecl twice, by different
    names. For example, an `__extension MyStruct` will be known to its parent
    ASTDecl as "extension:MyStruct" as well as "extension:" (the latter name is
    shared by all extensions).
    TODO(MOCO-522): Arcana docs on all the above, and on how importing works
    in general.

    Example:

    ```mlir
    lit.extension.decl @"extension:Spaceship_0" {
      lit.fn @foobar(%self: !lit.ref<!Spaceship, mut *"self`"> byref_result) -> i32 {
        lit.return %foo : i32
      }
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<LIT_TypeSignatureType>:$signature,
    KGEN_ParamDeclArrayAttr:$params,
    // String name of the target struct (e.g., the "MyStruct" in `extension MyStruct`).
    // This is here as well as targetStruct because we sometimes want to know the name
    // without signature resolving the whole extension.
    OptionalAttr<StrAttr>:$targetStructName,
    OptionalAttr<SymbolRefAttr>:$targetStruct,
    // Direct parent traits this extension adds (first level only).
    // Used for extension trait conformance checking.
    DefaultValuedAttr<KGEN_TraitSymbolArrayAttr,
                      "ArrayRef<TraitSymbolAttr>{}">:$immediateParents,
    // Canonical trait containing flattened hierarchy of all traits this extension adds.
    // Computed during extension signature resolution and used for conformance checking.
    OptionalAttr<TypeAttrOf<LIT_TraitType>>:$canonicalTrait
    // TODO(MOCO-522): Support doc strings, deprecation warnings.
  );

  let regions = (region SizedRegion<1>:$fields);

  let assemblyFormat = [{
    $sym_name ``
    custom<StructParameterSpec>($params, $signature)
    attr-dict-with-keyword $fields
  }];

  let extraClassDeclaration = [{
    ArrayRef<ParamDeclAttr> getInputParams() { return getParams(); }
  }];

  let builders = [
    // Create an extension with a region but no body block. The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name, CArg<"StringAttr", "{}">:$targetStructName)>,
  ];
}


//===----------------------------------------------------------------------===//
// TraitDeclOp
//===----------------------------------------------------------------------===//

def LIT_TraitDeclOp : LITOp<"trait.decl", [
    IsolatedFromAbove, SymbolName, SymbolVisibility, Symbol, SymbolTable,
    DeclInterface, SingleBlock,
    ASTDeclInterface, NoTerminator,
    DeclareOpInterfaceMethods<DebugInfo_ScopedLocation>,
    StabilityDecoratorInterface
  ]> {
  let summary = "Trait declaration.";
  let description = [{
    The `lit.trait.decl` operation is a top-level operation used to declare
    LIT trait types. Trait declarations are named with a symbol.
    The body contains `lit.trait.field` operations that declare the contents
    of the struct.

    Example:

    ```mlir
    lit.trait.decl @Trait<T: !Trait> {
      lit.fn @f() -> !kgen.none {
        kgen.unreachable
      }
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<LIT_TypeSignatureType>:$signature,
    KGEN_ParamDeclArrayAttr:$params,
    TypeAttrOf<LIT_TraitType>:$canonicalTrait,
    DefaultValuedAttr<KGEN_TraitSymbolArrayAttr,
                      "ArrayRef<TraitSymbolAttr>{}">:$immediateParents,
    EnumProp<LIT_TypeConvention>:$convention,
    // If definesClosure attribute is present,
    // then this is a compiler generated trait
    // used to implement a closure
    UnitAttr:$definesClosure,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<LIT_DeprecationInfoAttr>:$deprecationInfo,
    OptionalAttr<LIT_UnavailableInfoAttr>:$unavailableInfo,
    // Present if the decl has an @stable decorator on it.
    UnitAttr:$hasStableDecorator,
    // Present if the decl has an @stable(since="version") decorator.
    OptionalAttr<StrAttr>:$stableSinceVersion,
    // This is the error message to produce if a non-destructible type needs an
    // implicit destructor call.
    OptionalAttr<StrAttr>:$linearTypeErrorMsg,
    OptionalAttr<TypeAttrOf<KGEN_FuncTypeGeneratorType>>:$closureSignature,
    OptionalAttr<SourceNameAttr>:$sourceName
  );

  let regions = (region SizedRegion<1>:$fields);

  let assemblyFormat = [{
    $sym_name ``
    custom<StructParameterSpec>($params, $signature, $canonicalTrait)
    custom<TypeConvention>($convention) attr-dict-with-keyword $fields
  }];

  let extraClassDeclaration = [{
    /// Trait declarations only have input parameters.
    ArrayRef<ParamDeclAttr> getInputParams() { return getParams(); }

    bool isRegisterPassable() {
      TypeConvention c = getConvention();
      return c == TypeConvention::RegisterPassableTrivial ||
       c == TypeConvention::RegisterPassable;
    }

    TraitSymbolAttr bindReference(ArrayRef<TypedAttr> paramValues);

    bool isRegisterPassableTrivial() {
      return getConvention() == TypeConvention::RegisterPassableTrivial;
    }
  }];

  let builders = [
    // Create a trait with a region but no body block. The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name)>,
  ];
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

def LIT_TryOp : LITOp<"try",
  [DeclareOpInterfaceMethods<ControlFlowNode>,
   DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "A try-except-else block.";
  let description = [{
    The `lit.try` operation represents a try-except-else statement. When inside
    the 'try' region, any implicitly propagated errors interrupt execution and
    branch to the 'except' region. If the 'try' region completes execution
    without any exceptions, control-flow branches to the 'else' region.

    Example:

    ```mlir
    lit.try {
      %0 = kgen.call @maybeRaises() : () throws -> !lit.struct<@Int>
      lit.try.yield
    } except (%err: !lit.struct<@Error>) {
      lit.try.yield
    } else {
      lit.try.yield
    } finally {
      lit.try.yield
    }
    ```

    The 'finally' region is only present during semantic control-flow
    representation mode and is not modeled through the HLCF abstraction.

    The `label` becomes mandatory after lower-semantic-cf to support raising and
    transfer control flow to non-immediate try ops, which can happen after
    pasting a `finally` region that raises into inner scope.

    Example:
    ```mlir
    lit.try "OUTER" {
      lit.try "INNER" {
        lit.if ... {
          ...
          lit.try.raise "INNER"
        }

        # finally block pasted before `return`
        ...
        lit.try.raise "OUTER" # transfer control flow to outer scope.
        return
      } except {
        ...
      }
    } except (%err: !lit.struct<@Error>) {
      ...
    }
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$label,
    Optional<LIT_RefType>:$err,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$suppressWarnings);
  let regions = (region SizedRegion<1>:$tryRegion, SizedRegion<1>:$exceptRegion,
                        SizedRegion<1>:$elseRegion,
                        VariadicRegion<SizedRegion<1>>:$finallyRegions);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    ($label^)? ($err^ `:` type($err))? (`->` type($results)^)? $tryRegion
    `except` custom<RegionWithArgs>($exceptRegion)
    `else` custom<RegionWithArgs>($elseRegion)
    (`finally` $finallyRegions^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$err, CArg<"bool", "false">:$suppressWarnings),
      [{ build($_builder, $_state, {}, /*label=*/{}, err, suppressWarnings, 1); }]>,
  ];

  let extraClassDeclaration = [{
    // The region indices of the try operation.
    enum { kTRY, kEXCEPT, kELSE, kFINALLY };

    /// Get the finally region, assuming it is present.
    Region &getFinallyRegion() { return getFinallyRegions().front(); }

    /// Return true if the finally block is trivial.
    bool hasTrivialFinally();
  }];
}

//===----------------------------------------------------------------------===//
// TryYieldOp
//===----------------------------------------------------------------------===//

def LIT_TryYieldOp : LITOp<"try.yield", [Pure, Terminator,
    HasParent<"TryOp">, DeclareOpInterfaceMethods<ControlFlowTerminator>,
                        DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Try block terminator.";
  let description = [{
    The `lit.try.yield` operation is used as a terminator to `lit.try` regions.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];
}

//===----------------------------------------------------------------------===//
// TryRaiseOp
//===----------------------------------------------------------------------===//

def LIT_TryRaiseOp : LITOp<"try.raise", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Try block raise an error.";
  let description = [{
    The `lit.try.raise` operation is used within a `lit.try` try region to
    propagate an error to the except region.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    StrAttr:$label
  );
  let assemblyFormat = "$label ($operands^ `:` type($operands))? attr-dict";

  let builders = [
    OpBuilder<(ins "StringAttr": $label), [{
      build($_builder, $_state, {}, label);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// AliasDeclOp
//===----------------------------------------------------------------------===//

def LIT_AliasDeclOp : LITOp<"alias.decl", [
    ASTDeclInterface, DeclareOpInterfaceMethods<ParamOpInterface>,
    StabilityDecoratorInterface
  ]>, OneParamDecl {
  let summary = "`comptime x: T = value` or `comptime x: T` declaration";
  let description = [{
    This op represents an alias decl bound with the given initializer (unless
    it's in a trait, whose aliases don't need initializers).

    Examples:

    ```mlir
    lit.alias.decl A = <5>
    lit.alias.decl A_plus_one = <add(A, 1)>
    lit.alias.decl A
    ```
  }];

  let arguments = (ins
    KGEN_ParamDeclAttr:$paramDecl,
    OptionalAttr<TypedAttrInterface>:$value,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<LIT_DeprecationInfoAttr>:$deprecationInfo,
    OptionalAttr<LIT_UnavailableInfoAttr>:$unavailableInfo,
    // The trait symbol this alias is inherited from. None if self-declared.
    OptionalAttr<KGEN_TraitSymbolAttr>:$inheritedFrom,
    // Present for defaulted associated alias. This bit should be set before
    // parsing the alias declaration this allow use to probe whether the alias
    // decl is defaulted or not without body resolve it first.
    UnitAttr:$defaultedAssociatedAlias,
    // Present if the decl has an @stable decorator on it.
    UnitAttr:$hasStableDecorator,
    // Present if the decl has an @stable(since="version") decorator.
    OptionalAttr<StrAttr>:$stableSinceVersion,
    // Present if the alias has an @doc_hidden decorator on it.
    UnitAttr:$hasDocHiddenDecorator
  );
  let assemblyFormat = [{
    custom<AliasDeclOpValue>($paramDecl, $value) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Returns the (mangled) internal name for this alias.
    StringAttr getName() { return getParamDecl().getName(); }

    /// Return the alias name as it appears in the source code (not mangled).
    StringAttr getDeclName();

    Type getType() { return getParamDecl().getType(); }

    /// Return true if this is a defaulted trait method.
    bool isDefaultedAssociatedAlias() { return getDefaultedAssociatedAlias(); }
  }];

  let builders = [
    OpBuilder<(ins "ParamDeclAttr":$decl,
              CArg<"TypedAttr", "{}">:$value), [{
      build($_builder, $_state, decl, value, /*docString=*/{},
            /*deprecationInfo=*/{}, /*unavailableInfo=*/{},
            /*inheritedFrom=*/{}, /*defaultedAssociatedAlias=*/false,
            /*hasStableDecorator=*/false, /*stableSinceVersion=*/{},
            /*hasDocHiddenDecorator=*/false);
    }]>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VarDeclOp
//===----------------------------------------------------------------------===//

def LIT_VarDeclOp : LITOp<"var.decl", [
    ASTDeclInterface,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ParamOpInterface>]>, OneParamDecl {
  let summary = "Addressable local var declaration";
  let description = [{
    This declaration is used for var declarations with an exposed
    address.  `kind` encodes how this declaration was created, either via
    explicit `var` expressions, implicitly declared, or internally
    synthesized (which suppresses some warnings).  This type defines a origin
    and returns a mutable reference to its result in that origin.

    Example:

    ```mlir
    %foo = lit.var.decl "foo" var : !lit.ref<index, foo>
    ```
  }];


  let arguments = (ins
    StrAttr:$name,
    LIT_VarDeclKindAttr:$kind,
    KGEN_ParamDeclAttr:$paramDecl
  );
  let results = (outs LIT_RefType:$result);

  let assemblyFormat = [{
    $name enum($kind) `:`
    custom<VarDeclType>(type($result), $paramDecl) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$elementType, "StringRef":$name,
                   "StringRef":$originName,
                   "VarDeclKind":$kind)>
  ];
  let extraClassDeclaration = [{
    bool isSynthetic();

    /// Return true if this is non-synthetic variable, if its name starts with
    /// something other than an underscore, and is not an argument shadow.
    bool shouldWarnAboutUnused();

    // Change the element type of the var decl to the specified RValue type,
    // maintaining the origin of the vardecl.
    void changeElementType(Type newElementType);
  }];
}

//===----------------------------------------------------------------------===//
// VarLifetimeStartOp
//===----------------------------------------------------------------------===//

def LIT_VarLifetimeStartOp : LITOp<"var.lifetime.start"> {
  let arguments = (ins LIT_RefType:$arg);
  let assemblyFormat = "$arg attr-dict `:` type($arg)";
}

//===----------------------------------------------------------------------===//
// VarLifetimeEndOp
//===----------------------------------------------------------------------===//

def LIT_VarLifetimeEndOp : LITOp<"var.lifetime.end"> {
  let arguments = (ins LIT_RefType:$arg);
  let assemblyFormat = "$arg attr-dict `:` type($arg)";
}

//===----------------------------------------------------------------------===//
// AsyncCallOp
//===----------------------------------------------------------------------===//

def LIT_AsyncCallOp : LITOp<"async.call", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>]> {
  let summary = "Call to an async function.";
  let description = [{
    The `lit.async.call` operation calls an `async` parametric callable. It
    returns a `!co.routine`.

    Example:

    ```mlir
    %coro = lit.async.call[:() -> index: @foo]()
    ```
  }];

  let arguments = (ins
    AttrOfType<KGEN_FuncTypeGeneratorType>:$callee,
    KGEN_ParameterExprArrayAttr:$implicitOrigins,
    Variadic<AnyType>:$operands
  );
  let results = (outs CoroutineType:$coroutine);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee) ``
    custom<OriginParams>($implicitOrigins) ``
    custom<AsyncCallOpTypes>(type($operands), ref($callee),
                             ref($implicitOrigins))
    `(` $operands `)` attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// EndFnOp
//===----------------------------------------------------------------------===//

def LIT_EndFnOp : LITOp<"end_fn", [Terminator]> {
  let summary = "Function terminator.";
  let description = [{
    The `lit.end_fn` operation denotes the end of a `lit.fn` body. This
    terminator corresponds to the end of the function body suite in Lit. It is
    generated by the parser as the default terminator of a `lit.fn`.

    The `unresolved` attribute is used to indicate that the function is not
    body resolved - only the signature is resolved, but the body isn't parsed
    yet.

    Example:

    ```mlir
    lit.end_fn
    ```
  }];

  // The `unresolved` attribute helps keep track whether the containing lit.fn
  // op is body resolved or not. This is necessary as the Mojo language server
  // will run some of the initial LIT lowering passes and may be passed along
  // lit.fn ops that are not body resolved yet.
  let arguments = (ins UnitAttr:$unresolved);
  let assemblyFormat = "(`unresolved` $unresolved^)? attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def LIT_ReturnOp : LITOp<"return", []> {
  let summary = "Lexical return statement.";
  let description = [{
    The `lit.return` operation is not a terminator, but it corresponds to a
    return statement in Mojo. The operands are the desired return values at the
    return statement. This operation exists to model the fact that functions
    can have multiple returns and that those returns do not have to be the last
    statement in a suite.

    Example:

    ```mlir
    lit.fn @foobar(%foo: i32, %bar: i32) -> i32 {
      lit.return %foo : i32
      lit.return %bar : i32
      lit.end_fn
    }
    ```

    The operation also specifies the return parameters of the function.

    Example:

    ```mlir
    lit.fn @foobar<() -> index>() {
      lit.return<5>
      lit.end_fn
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict (`:` type($operands)^)?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ErrorReturnOp
//===----------------------------------------------------------------------===//

def LIT_ErrorReturnOp : LITOp<"error_return", [Terminator, ReturnLike,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Represents an early return due to failure.";
  let description = [{
    The `lit.error_return` operation is a terminator version
    of `lit.raise` that represents exit from function call due
    to failure.

    Example:

    ```mlir
    lit.fn @throwing_func() throws -> !kgen.variant<@Error, none> {
      %1 = ... : !lit.struct<@Error>
      %2 = kgen.variant.create %1 :
        !lit.struct<@Error> -> !kgen.variant<@Error, none>
      lit.error_return %2 : !kgen.variant<@Error, none>
    }
    ```
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

def LIT_RaiseOp : LITOp<"raise"> {
  let summary = "Lexical raise statement.";
  let description = [{
    The `lit.raise` operation is not a terminator. It corresponds to a raise
    statement in Lit. This operation is only allowed inside the 'try' region of
    a `lit.try` operation or within a `lit.fn` marked as `throws`.

    Example:

    ```mlir
    lit.raise
    ```
  }];

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoopContinueOp
//===----------------------------------------------------------------------===//

def LIT_LoopContinueOp : LITOp<"loop.continue", [
    HasParent<"LoopOp">, Terminator]> {

  let summary = "Loop continue.";
  let description = [{
    The `lit.loop.continue` operation is a terminator. The operation indicates
    that control-flow should branch to the beginning of the enclosing loop. It
    is only used as the terminator for the body of a `lit.loop`.

    Example:

    ```mlir
    lit.loop.continue
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// LoopYieldOp
//===----------------------------------------------------------------------===//

def LIT_LoopYieldOp : LITOp<"loop.yield", [
    HasParent<"LoopOp">, Terminator]> {
  let summary = "Loop yield.";
  let description = [{
    The `lit.loop.yield` operation is a terminator. The operation indicates
    that control-flow should branch to the end of the enclosing loop (after the
    else region finishes execution).

    Example:

    ```mlir
    lit.loop.yield
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// LoopBreakElseOp
//===----------------------------------------------------------------------===//

def LIT_LoopBreakElseOp : LITOp<"loop.break.else", [Terminator]> {
  let summary = "Loop break to the else block.";
  let description = [{
    The `lit.loop.break.else` operation is a terminator. The operation indicates
    that control-flow should branch to the else block of the enclosing loop,
    which is used for a "while" with a condition that is false or a for-each
    where the sequence is empty.

    In contrast, a lit.break will exit the loop without executing the 'else'.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

def LIT_BreakOp : LITOp<"break"> {
  let summary = "Lexical break statement.";
  let description = [{
    The `lit.break` operation is not a terminator, but it corresponds to a break
    statement in Lit. The operation indicates that control-flow should branch
    to after the enclosing loop.  This is lowered by LowerSemanticCF.

    Example:

    ```mlir
    lit.break
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

def LIT_ContinueOp : LITOp<"continue"> {
  let summary = "Lexical continue statement.";
  let description = [{
    The `lit.continue` operation is not a terminator, but it corresponds to a
    continue statement in Lit. The operation indicates that control-flow should
    branch to after the enclosing loop.  This is lowered by LowerSemanticCF.

    Example:

    ```mlir
    lit.continue
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// UnresolvedImportOp
//===----------------------------------------------------------------------===//

// TODO(LLVM Issue #59877): Why isn't this in BuiltinAttributes.td?
def Builtin_LocationAttr : Builtin_Attr<"Location", "location", []> {}

def LIT_UnresolvedImportOp : LITOp<"unresolved_import"> {
  let summary = "Lexical unresolved import.";
  let description = [{
    The `lit.unresolved_import` operation represents a module import that is yet
    to be resolved. If a decl name is provided, this operation represents an
    import of that decl, not the module itself.

    Example:

    ```mlir
    // Import a module as the given name.
    lit.unresolved_import #lit.import_path<0, ["module"]> as @newModule
    // Import a relative module as the given name.
    lit.unresolved_import #lit.import_path<1, ["module"]> as @newModule
    // Import a function within the module as the given name.
    lit.unresolved_import #lit.import_path<0, ["module"]> -> @func as @newFunc
    ```
  }];

  let arguments = (ins
    LIT_ImportPathAttr:$modulePath,
    SymbolNameAttr:$importName,
    OptionalAttr<SymbolNameAttr>:$declName,
    OptionalAttr<Builtin_LocationAttr>:$importNameLoc,
    OptionalAttr<Builtin_LocationAttr>:$declNameLoc
  );
  let assemblyFormat = [{
    $modulePath (`->` $declName^)? `as` $importName
    (`declNameLoc` `(` $declNameLoc^ `)`)?
    (`importNameLoc` `(` $importNameLoc^ `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnresolvedWildcardImportOp
//===----------------------------------------------------------------------===//

def LIT_UnresolvedWildcardImportOp : LITOp<"unresolved_wildcard_import"> {
  let summary = "Lexical unresolved wildcard import.";
  let description = [{
    The `lit.unresolved_wildcard_import` operation represents a wildcard module
    import that is yet to be resolved.

    Example:

    ```mlir
    // Import from the given module.
    lit.unresolved_wildcard_import from #lit.import_path<0, ["module"]>
    ```
  }];

  let arguments = (ins LIT_ImportPathAttr:$modulePath);
  let assemblyFormat = "`from` $modulePath attr-dict";
}

//===----------------------------------------------------------------------===//
// ImportOp
//===----------------------------------------------------------------------===//

def LIT_ImportOp : LITOp<"import", [NoRegionArguments, SymbolName,
    SymbolVisibility, Symbol, SymbolTable]
    # GraphRegionNoTerminator.traits> {
  let summary = "Resolved module import with scoped child access.";
  let description = [{
    Represents a resolved import that gates access to a package or module.
    Contains nested ImportOps for child modules accessible through dotted
    access. A child module is accessible only if it has a nested ImportOp here
    (it was imported) or the real package re-exports it. Non-module members
    (functions, structs, aliases) are always accessible.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    LIT_ImportPathAttr:$modulePath
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    $sym_name `path` $modulePath
    attr-dict-with-keyword $bodyRegion
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$sym_name, "ImportPathAttr":$modulePath)>
  ];
}

//===----------------------------------------------------------------------===//
// UnboundRegionOp
//===----------------------------------------------------------------------===//

def LIT_UnboundRegionOp : LITOp<"unbound_region"> {
  let summary = "A transient container for a region.";
  let description = [{
    The `lit.unbound_region` operation is a container for an MLIR region that
    has yet to be bound to an MLIR operation. This is used by the parser to
    attach regions to MLIR operations. The operation itself is never valid. It
    must be moved by the parser into an MLIR operation.
  }];

  let regions = (region SizedRegion<1>:$region);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoadConsumeOp
//===----------------------------------------------------------------------===//

def LIT_LoadConsumeOp : LITOp<"load.consume", [
    ReferenceElementType<"ref", "result">]> {
  let summary = "consuming pointer load operation";
  let description = [{
    Loads the data at the given pointer like `pop.load`, but consumes the
    ownership of the memory being accessed. This is used when performing MRValue
    to SRValue conversion (i.e. when consuming a `var` of `RegisterPassable`
    type), and is lowered to `pop.load`.

    Example:

    ```mlir
    %0 = lit.load.consume %ref : !lit.ref<scalar<f32>, #origin>
    ```
  }];

  let arguments = (ins Arg<LIT_RefType, "", [MemRead]>:$ref);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ref attr-dict `:` qualified(type($ref))";
}

//===----------------------------------------------------------------------===//
// MaterializeIntoOp
//===----------------------------------------------------------------------===//

def LIT_MaterializeIntoOp : LITOp<"materialize_into", [ReferenceElementType<"dest", "value">]> {
  let summary = "Materialize a parameter and store the value into a reference";
  let description = [{
    This operation materializes the value of a parameter expression and store
    the value into a reference. Unlike `kgen.param.materialize`, which produces
    an SSA value for the materialized value, this operation stores the value to
    memory so that it allows non-register-passable values to be explicitly
    materialized by users.

    The operation is lowered to `kgen.param.materialize` + `lit.ref.store`.

    Example:

    ```mlir
    lit.materialize_into %ref0 [v] : !lit.ref<scalar<f32>, mut origin>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value, LIT_RefType:$dest);
  let assemblyFormat = "$dest `[` $value `]` attr-dict `:` type($dest)";
}

//===----------------------------------------------------------------------===//
// OwnershipUseOp
//===----------------------------------------------------------------------===//

def LIT_OwnershipUseOp : LITOp<"ownership.use"> {
  let summary = "A use of a value to keep the value alive";
  let description = [{
    This op looks like a use of a value (or its pointee) which keeps the value
    live until at least this point.  This operation is a noop that is removed by
    the LowerLIT pass.

    ```mlir
    lit.ownership.use %0 : !somestring
    ```
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// OwnershipMarkInitialized
//===----------------------------------------------------------------------===//

def LIT_OwnershipMarkInitializedOp : LITOp<"ownership.mark_initialized"> {
  let summary = "Mark an uninitialized value as being fully initialized";
  let description = [{
    The `lit.ownership.mark_initialized` operation informs CheckLifetimes that
    the specified value is now fully initialized. Note that this marks all
    subelements of the value as being initialized too, not just the fully-object
    bit.

    ```mlir
    lit.ownership.mark_initialized %0 : !lit.ref<T, mut lt>
    ```

    The argument must be an indirectly origin-tracked value, because
    CheckLifetimes isn't field sensitive for register passable values.
  }];

  let arguments = (ins LIT_RefType:$value);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// OwnershipMarkConsumedOp
//===----------------------------------------------------------------------===//

def LIT_OwnershipMarkConsumedOp : LITOp<"ownership.mark_consumed"> {
  let summary = "Mark a value as completely consumed.";
  let description = [{
    This op informs CheckLifetimes that the specified value has had its whole
    value consumed, including all subfields.

    ```mlir
    lit.ownership.mark_consumed %0 : !somestring
    ```

    The argument must be an indirectly origin-tracked value.
  }];

  let arguments = (ins LIT_RefType:$value);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// OwnershipMarkDestroyedOp
//===----------------------------------------------------------------------===//

def LIT_OwnershipMarkDestroyedOp : LITOp<"ownership.mark_destroyed"> {
  let summary = "Mark a value as not need a dtor to run";
  let description = [{
    This op informs CheckLifetimes that the specified value does not need its
    full object destructor to run.  This corresponds to things like Rust's
    mem::forget and is synthesized in __del__ methods to indicate that self is
    destroyed.

    ```mlir
    lit.ownership.mark_destroyed %0 : !somestring
    ```

    The argument must be an indirectly origin-tracked value, because
    CheckLifetimes isn't field sensitive for register passable values.
  }];

  let arguments = (ins LIT_RefType:$value);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// RefImmutOp
//===----------------------------------------------------------------------===//

class RefImmutType<string lhs, string rhs> : TypesMatchWith<
    "reference with mutability removed", lhs, rhs,
    "llvm::cast<RefType>($_self).getWithMutability(false)">;

def LIT_RefImmutOp : LITOp<"ref.immut", [RefImmutType<"ref", "result">]> {
  let summary = "Drop mutability from a reference";
  let description = [{
    This op drops mutability from a reference type: given a
    `!lit.ref<X, mut life>`, it returns `!lit.ref<X, imm life>`.

    ```mlir
    lit.ref.immut %0 : !lit.ref<@Foo, mut life>
    ```
  }];

  let hasFolder = 1;
  let arguments = (ins LIT_RefType:$ref);
  let results = (outs LIT_RefType:$result);
  let assemblyFormat = "$ref `:` type($ref) attr-dict";


  let extraClassDeclaration = [{
    /// If the specified value is a RefImmutOp, look through it to its operand.
    /// Otherwise just return the value.
    static Value strip(Value value) {
      if (auto cast = value.getDefiningOp<RefImmutOp>())
        return cast.getOperand();
      return value;
    }

    /// If 'value' is defined by one or more rebind-like ops, look through them.
    static Value stripRebinds(Value value);
  }];
}

//===----------------------------------------------------------------------===//
// RefUpcastOp
//===----------------------------------------------------------------------===//

def LIT_RefUpcastOp : LITOp<"ref.upcast", [Pure]> {
  let summary = "Widen a reference's origin";
  let description = [{
    This op upcasts a reference's origin to a wider origin that covers the
    original, such as a subtree origin (`#lit.origin.subtree`) or an origin
    union that includes the source origin. The element type and address space
    are unchanged; the result origin must be specified explicitly because it
    is not uniquely determined by the operand.

    ```mlir
    %1 = lit.ref.upcast %0
      : !lit.ref<!lit.struct<@Foo>, mut o>
      -> !lit.ref<!lit.struct<@Foo>, mut #lit.origin.subtree<o>>
    ```
  }];

  let hasVerifier = 1;
  let arguments = (ins LIT_RefType:$ref);
  let results = (outs LIT_RefType:$result);
  let assemblyFormat = [{
    $ref attr-dict `:` type($ref) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// RefToPointerOp
//===----------------------------------------------------------------------===//

// Given a `!lit.ref<X, life>`, return `!kgen.pointer<X>`.
class RefPointerType<string lhs, string rhs> : TypesMatchWith<
    "reference as pointer type", lhs, rhs,
    "llvm::cast<RefType>($_self).getAsPointerType()">;

def LIT_RefToPointerOp : LITOp<"ref.to_pointer",
                              [RefPointerType<"ref", "ptr">]> {
  let summary = "Extract a low-level pointer out of a reference";
  let description = [{
    This op extracts a kgen pointer out of a reference, exposing an untracked
    and unsafe inner pointer.  This is used when lowering lifetimes.

    ```mlir
    lit.ref.to_pointer %0 : !lit.ref<@Foo, life>
    ```
  }];

  let arguments = (ins LIT_RefType:$ref);
  let results = (outs PointerType:$ptr);
  let assemblyFormat = "$ref `:` type($ref) attr-dict";
}

//===----------------------------------------------------------------------===//
// RefFromPointerOp
//===----------------------------------------------------------------------===//

def LIT_RefFromPointerOp : LITOp<"ref.from_pointer",
                                 [RefPointerType<"ref", "ptr">]> {
  let summary = "Create an immortal reference from a pointer value";
  let description = [{
    This op converts a raw pointer to a reference with specified mutability and
    origin.  When `startsUninit` is true, the reference is assumed to be
    uninitialized but needs to be initialized before exit.  When false it is
    assumed to be initialized and must be initialized on exit.

    ```mlir
    lit.ref.from_pointer %0, startUninit=true : !lit.ref<mut @Foo, life>
    ```
  }];

  let arguments = (ins PointerType:$ptr, UnitAttr:$startUninit,
                       UnitAttr:$endUninit);
  let results = (outs LIT_RefType:$ref);
  let assemblyFormat = [{
    $ptr (`start_uninit` $startUninit^)? (`end_uninit` $endUninit^)?
    `:` type($ref) attr-dict
  }];

  let builders = [
    // Create with a pointer and origin, to infer result type.
    OpBuilder<(ins "Value":$pointer, "TypedAttr":$origin,
                   "bool":$startsUninit, "bool":$endsUninit)>,
  ];
}


//===----------------------------------------------------------------------===//
// RefFromPointerREPLOp
//===----------------------------------------------------------------------===//

def LIT_RefFromPointerREPLOp : LITOp<"ref.from_pointer.repl",
                                     [RefPointerType<"ref", "ptr">]> {
  let summary = "Create an untracked reference from a pointer value";
  let description = [{
    This op is like `lit.ref.from_pointer` is a HACK that is only intended to be
    used in one place in the REPL as a stop-gap, do not perpetuate this!
  }];

  let arguments = (ins PointerType:$ptr, StrAttr:$name);
  let results = (outs LIT_RefType:$ref);
  let assemblyFormat = "$ptr `:` type($ref) attr-dict";
}


//===----------------------------------------------------------------------===//
// RefToKgenPtrOp
//===----------------------------------------------------------------------===//

def LIT_RefToKgenPtrOp : LITOp<"ref.to_kgen_ptr", [Pure]> {
  let summary = "Cast a lit.ref to a kgen.pointer with lowered element type";
  let description = [{
    Converts a `lit.ref` to a `kgen.pointer`, lowering the element type from
    LIT to KGEN (e.g., `lit.struct` → `kgen.struct`). The address space is
    preserved. Origin information is discarded.

    This enables using `kgen.struct.gep` on LIT references by first converting
    to a KGEN pointer with a KGEN struct element type.

    Example:
    ```mlir
    %ptr = lit.ref.to_kgen_ptr %ref : !lit.ref<!lit.struct<...>, origin>
                                    -> !kgen.pointer<!kgen.struct<...>>
    ```
  }];

  let arguments = (ins LIT_RefType:$ref);
  let results = (outs PointerType:$result);

  let assemblyFormat = [{
    $ref attr-dict `:` type($ref) `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RefFromKgenPtrOp
//===----------------------------------------------------------------------===//

def LIT_RefFromKgenPtrOp : LITOp<"ref.from_kgen_ptr", [Pure]> {
  let summary = "Cast a kgen.pointer to a lit.ref with origin";
  let description = [{
    Converts a `kgen.pointer` to a `lit.ref`, lifting the element type from
    KGEN to LIT (e.g., `kgen.struct` → `lit.struct`). The origin must be
    explicitly specified in the result type.

    This enables converting back to LIT references after pointer manipulations
    like `kgen.struct.gep`, preserving origin tracking.

    Example:
    ```mlir
    %ref = lit.ref.from_kgen_ptr %ptr : !kgen.pointer<!kgen.struct<...>>
                                      -> !lit.ref<!lit.struct<...>, origin>
    ```
  }];

  let arguments = (ins PointerType:$pointer);
  let results = (outs LIT_RefType:$result);

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer) `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RefStructGEROp
//===----------------------------------------------------------------------===//

def LIT_RefStructGEROp : LITOp<"ref.struct.ger", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Get a reference to a struct field.";
  let description = [{
    The `lit.ref.struct.ger` operation computes a reference to a struct field
    from a parametric struct. The result has the same origin as the source value
    (or field-sensitive origin when using field name access).

    The field can be specified either by name or by index:
    - By name: Uses `field` attribute, requires concrete struct type
    - By index: Uses `index` attribute, allows parametric types

    When using index access with a parametric index, the operation will be
    canonicalized to field name access once the index becomes concrete and the
    struct type is known.

    Examples:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : i32
      lit.struct.field b : i64
    }

    // Field name access
    %0 = lit.ref.struct.ger %struct[a] : <@MyStruct<...>, mut life> -> i32

    // Index access (concrete)
    %1 = lit.ref.struct.ger %struct[idx 0] : <@MyStruct<...>, mut life> -> i32

    // Index access (parametric - in generic context)
    %2 = lit.ref.struct.ger %struct[idx I] : <!kgen.param<T>, mut life> -> ...
    ```
  }];

  // Note: We use LIT_RefType instead of RefOf<LIT_StructType> because this op
  // needs to accept references to parametric types like `T` in generic
  // functions when using index access:
  //
  //   def foo[T: AnyType](ref s: T):
  //       __struct_field_ref(0, s)  # s has type !lit.ref<!kgen.param<T>, ...>
  //
  // At this point, T hasn't been instantiated to a concrete struct type yet.
  // The verifier allows both StructType and ParamType as the element type.
  // When T is instantiated, the op will canonicalize to use field name.
  let arguments = (ins OptionalAttr<StrAttr>:$field,
                       OptionalAttr<AttrOfType<Index>>:$index,
                       LIT_RefType:$container);
  let results = (outs LIT_RefType:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Infer the result type based on the result of lookup (field name access).
    OpBuilder<(ins "Value":$structBasePtr, "StructFieldOp":$field)>,
    // Build with index access.
    OpBuilder<(ins "Type":$resultType, "TypedAttr":$index, "Value":$container)>,
    // Build with explicit field name and result type (for compatibility).
    OpBuilder<(ins "Type":$resultType, "StringAttr":$field, "Value":$container)>
  ];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Return true if this operation uses field name access.
    bool usesFieldAccess() { return getField().has_value(); }

    /// Return true if this operation uses index access.
    bool usesIndexAccess() { return getIndex().has_value(); }

    /// Given a reference to a struct, return the reference type to the
    /// specified field, maintaining origin and mutability, and rebinding any
    /// generics from the base type through the field type.
    static RefType getFieldType(RefType structRefTy, StructFieldOp field,
                               ParameterEvaluationContext *ctx = nullptr);

    /// Given a reference to a struct, return the reference type to the
    /// specified field, maintaining origin and mutability, assuming the type
    /// is already rebound to its final type.
    static RefType getReboundFieldType(RefType structRefTy,
                                       StringAttr fieldName, Type reboundType);
  }];

}

//===----------------------------------------------------------------------===//
// RefLoadOp
//===----------------------------------------------------------------------===//

def LIT_RefLoadOp : LITOp<"ref.load", [ReferenceElementType<"ref", "result">]> {
  let summary = "reference load operation";
  let description = [{
    Loads the data at the given reference. Returns a scalar of the element type.
    The reference may either be mutable or immutable.

    Example:

    ```mlir
    %0 = lit.ref.load %ref0 : !lit.ref<scalar<f32>, origin>
    %1 = lit.ref.load %ref1 : !lit.ref<simd<4, type>, origin>
    %2 = lit.ref.load %ref0 : !lit.ref<scalar<f32>, mut origin>
    ```
  }];

  let arguments = (ins LIT_RefType:$ref);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ref attr-dict `:` type($ref)";
}

//===----------------------------------------------------------------------===//
// RefStoreOp
//===----------------------------------------------------------------------===//

def LIT_RefStoreOp : LITOp<"ref.store", [ReferenceElementType<"dest", "value">]> {
  let summary = "Reference store operation";
  let description = [{
    Store the given value to the given reference of the same element type.  The
    reference must be mutable.

    Example:

    ```mlir
    lit.ref.store %0, %ref0 : !lit.ref<scalar<f32>, mut origin>
    lit.ref.store %1, %ref1 : !lit.ref<simd<4, type>, mut origin>
    ```
  }];

  let arguments = (ins AnyType:$value, LIT_RefType:$dest);
  let assemblyFormat = "$value `,` $dest attr-dict `:` type($dest)";
}

//===----------------------------------------------------------------------===//
// MemcpyOp
//===----------------------------------------------------------------------===//

// Checks that all reference element types match.
class EqualPointeeTypes<list<string> names> :
    AllMatchSameOperatorTrait<
        names, "llvm::cast<RefType>($_self.getType()).getElementType()",
        "element types"
    >;

def LIT_MemcpyOp : LITOp<"memcpy", [EqualPointeeTypes<["src", "dst"]>]> {
  let summary = "memcpy operation";
  let description = [{
    Performs an in-memory copy operation between the source reference and
    destination reference. The source reference may either be mutable or
    immutable; the destination reference must be mutable. Unlike regular
    memcpy, this op doesn't work on bytes - it does a whole-value copy of the
    references' element type. The two references may have differing address
    space qualifiers. The destination is considered 'initialized' by this
    operation.

    Example:

    ```mlir
    lit.memcpy %src0, %dst0 : !lit.ref<scalar<f32>> -> !lit.ref<scalar<f32>>
    lit.memcpy %src1, %dst1 : !lit.ref<simd<4, type>> -> !lit.ref<simd<4, type>>
    lit.memcpy %src2, %dst2 : !lit.ref<scalar<f32>, immut> -> !lit.ref<scalar<f32>, mut>
    ```
  }];

  let arguments = (ins LIT_RefType:$src, LIT_RefType:$dst);
  let results = (outs);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `->` type($dst)";
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

def LIT_LoopOp: LITOp<"loop"> {
  let summary = "loop operation";
  let description = [{
    The `lit.loop` operation defines a lexical single scoped control-flow
    region. The operation has two regions:
      - body region
      - else region

    By itself, the loop has "infinite loop" semantics - when entered, it will
    continuously execute the body region (assuming it ends with lit.continue).
    However, typically the loop will contain control flow that causes execution
    to exit with a return, break, or continue.

    Like Python for-each/while loops, this loop may have an "else" block. When
    the normal iteration sequence ends, control transfers to the else block
    through a lit.loop.break.else terminator.

    ```mlir
    lit.loop { // Noop loop.
      lit.loop.break.else
    } else {
      lit.loop.yield
    }
  }];

  let results = (outs);
  let regions = (region SizedRegion<1>:$bodyRegion,
                        SizedRegion<1>:$elseRegion);

  let assemblyFormat = [{
    $bodyRegion
    `else` $elseRegion attr-dict
  }];

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];
}

//===----------------------------------------------------------------------===//
// RefPackCreateOp
//===----------------------------------------------------------------------===//

def LIT_RefPackCreateOp : LITOp<"ref.pack.create", [Pure]> {
  let summary = "create a reference pack";
  let description = [{
    The `kgen.ref.pack.create` operation creates a value of `!lit.ref.pack`
    type, populated with the given SSA values.  The elements should all have
    reference type with matching origin and address space.

    Example:

    ```mlir
      %0 = lit.ref.pack.create(%arg0, %arg1) : ...
    ```
  }];

  let arguments = (ins Variadic<LIT_RefType>:$elements);
  let results = (outs LIT_RefPackType:$result);
  let assemblyFormat = [{
    `(` $elements `)` attr-dict `:`
    custom<RefPackCreateType>(type($result), type($elements))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RefPackExtractOp
//===----------------------------------------------------------------------===//

def LIT_RefPackExtractOp : LITOp<"ref.pack.extract",
                      [Pure, InferTypeOpAdaptor]> {
  let summary = "reference pack get element";
  let description = [{
    The `lit.ref.pack.extract` operation returns the reference in the pack at the
    provided index.
  }];

  let arguments = (ins LIT_RefPackType:$pack, AttrOfType<Index>:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $pack `[` custom<IndexParamValue>($index) `]` attr-dict `:` type($pack)
  }];
}

//===----------------------------------------------------------------------===//
// RefPackFromPointerPackOp
//===----------------------------------------------------------------------===//

def LIT_RefPackFromPointerPackOp : LITOp<"ref.pack.from_pointer_pack", [Pure]> {
  let summary = "cast a pointer pack to a reference pack";
  let description = [{
    The `lit.ref.pack.from_pointer_pack` operation interprets a `!kgen.struct` of
    pointers as a `!lit.ref.pack`. The result type supplies the common origin and
    address space, and its variadic element list must match the operand pack's
    element types after applying `variadic_ptrremove_map`.

    Example:

    ```mlir
    %0 = lit.ref.pack.from_pointer_pack %pack
      : !kgen.struct<[pointer<index>, pointer<f32>]> isParamPack>
     -> !lit.ref.pack<:param_list<!kgen.type> [index, f32], imm #lit.any.origin>
    ```
  }];

  let arguments = (ins KGEN_StructType:$pack);
  let results = (outs LIT_RefPackType:$result);
  let assemblyFormat = [{
    $pack attr-dict `:` type($pack) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// MojoVersionMajorOp
//===----------------------------------------------------------------------===//

def LIT_MojoVersionMajorOp : LITOp<"mojo.version.major", [Pure]> {
  let summary = "The Mojo language version, major component";
  let description = [{
    returns current major version of the Mojo language
  }];

  let arguments = (ins);
  let results = (outs IndexScalar:$version);
}

//===----------------------------------------------------------------------===//
// MojoVersionMinorOp
//===----------------------------------------------------------------------===//

def LIT_MojoVersionMinorOp : LITOp<"mojo.version.minor", [Pure]> {
  let summary = "The Mojo language version, minor component";
  let description = [{
    returns current minor version of the Mojo language
  }];

  let arguments = (ins);
  let results = (outs IndexScalar:$version);
}

//===----------------------------------------------------------------------===//
// MojoVersionPatchOp
//===----------------------------------------------------------------------===//

def LIT_MojoVersionPatchOp : LITOp<"mojo.version.patch", [Pure]> {
  let summary = "The Mojo language version, patch component";
  let description = [{
    returns current patch version of the Mojo language
  }];

  let arguments = (ins);
  let results = (outs IndexScalar:$version);
}
