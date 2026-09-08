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
// This defines operations in the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENOPS_TD
#define KGEN_KGENDIALECT_KGENOPS_TD

include "Mojo/KGENDialect/KGENAttrs.td"
include "Mojo/KGENDialect/KGENDialect.td"
include "Mojo/KGENDialect/KGENEnums.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "Mojo/HLCFDialect/HLCFInterfaces.td"
include "Mojo/Interpreter/InterpreterInterface.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class KGENOp<string mnemonic, list<Trait> traits = []> :
    Op<KGENDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def KGEN_FuncOp : KGENOp<"func", [
    HasParent<"ModuleOp">, FunctionOpInterface, SymbolName, SymbolVisibility,
    Symbol, IsolatedFromAbove,
    SingleBlock, DeclInterface, FuncInterface, FunctionLike, ExportInterface,
    KGEN_InstantiatedOpInterface]> {
  let summary = "Concrete function definition.";
  let description = [{
    The `kgen.func` operation represents a concrete KGEN function. It has no
    input parameters, no results parameters, and cannot contain any parametric
    operations. A `kgen.func` represents an elaborated `kgen.generator` that can
    be lowered and compiled to an executable.

    The body of a `kgen.func` is a single block region terminated by a
    `kgen.return` whose operands represent the return values of the function.

    Example:

    ```mlir
    kgen.func @kernel(%arg0: index, %arg1: index) -> index {
      %0 = index.add %arg0, %arg1
      kgen.return %0 : index
    }
    ```

    There are cases where we might have a `kgen.func` with a
    `precompiledBodyRef` attribute, like so:

    ```mlir
    kgen.func @someFn(%arg0: index) -> index attributes {
      precompiledBodyRef = @importedLib
    } {
      ...
    }
    ```

    The presence of this attribute means that the function was already compiled
    all the way to an object file once, so we should avoid doing it again.

    The function can contain opaque metadata to pass onto LLVM in its
    `llvmMetadata` dictionary attribute. Note that this is not the same as LLVM
    function attributes, which is contained in the `llvmAttrs` dictionary
    attribute.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<KGEN_FuncTypeGeneratorType>:$funcTypeGenerator,
    KGEN_DecoratorsAttr:$decorators,
    KGEN_InlineLevelAttr:$inlineLevel,
    KGEN_ExportKindAttr:$exportKind,
    UnitAttr:$external, UnitAttr:$convergent,
    DefaultValuedAttr<DictionaryAttr, "{}">:$LLVMMetadata,
    DefaultValuedAttr<DictArrayAttr, "{}">:$LLVMArgMetadata,
    DefaultValuedAttr<StringArrayAttr, "ArrayRef<StringAttr>{}">:$crossDeviceCaptures,
    OptionalAttr<TypeAttr>:$coroutineType,
    OptionalAttr<KGEN_LinkageNameAttr>:$linkageName
  );
  let regions = (region MaxSizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    `` custom<FuncOp>($exportKind, $sym_name, $funcTypeGenerator, $inlineLevel,
                     $decorators, attr-dict, $bodyRegion)
  }];

  let extraClassDeclaration = [{
    using OpTrait::SingleBlock<FuncOp>::getBody;

    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }
    FunctionType getFunctionType() {
      return getFuncTypeGenerator().getBody().getValues();
    }
    Region *getCallableRegion() { return &getBodyRegion(); }

    ArrayAttr getArgAttrsAttr() { return nullptr; }
    ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(ArrayAttr attrs) {}
    void setResAttrsAttr(ArrayAttr attrs) {}
    Attribute removeArgAttrsAttr() { return nullptr; }
    Attribute removeResAttrsAttr() { return nullptr; }

    bool isExternal() { return getExternal(); }
    bool isConvergent() { return getConvergent(); }
  }];

  let builders = [
    // Builder to be used when there is no precompiled body.
    OpBuilder<(ins "StringAttr":$name, "FuncType":$type,
                   CArg<"InlineLevel",
                        "InlineLevel::Automatic">:$inlineLevel,
                   CArg<"ExportKind", "ExportKind::NotExported">:$exportKind,
                   CArg<"bool", "false">:$external,
                   CArg<"bool", "false">:$convergent,
                   CArg<"LinkageNameAttr", "{}">:$linkageName,
                   CArg<"ArrayRef<TypedAttr>", "{}">:$decorators,
                   CArg<"DictionaryAttr", "{}">:$llvmMetadata,
                   CArg<"ArrayAttr", "{}">:$llvmArgMetadata), [{
      if (!llvmMetadata)
        llvmMetadata = DictionaryAttr::get(name.getContext());
      if (!llvmArgMetadata)
        llvmArgMetadata = ArrayAttr::get(name.getContext(), {});
      auto sigGen = GeneratorType::get(/*inputParamTypes=*/{}, type);
      build($_builder, $_state, name, /*sym_visibility=*/nullptr, sigGen,
            decorators, inlineLevel,
            exportKind, external, convergent, llvmMetadata, llvmArgMetadata,
            /*crossDeviceCaptures=*/{}, /*coroutineType=*/{}, linkageName);
    }]>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_GeneratorOp : KGENOp<"generator", [
    HasParent<"ModuleOp">, FunctionOpInterface, SymbolName, SymbolVisibility,
    Symbol, IsolatedFromAbove,
    SingleBlock, DeclInterface, FuncInterface, FunctionLike, ExportInterface,
    KGEN_GeneratorOpInterface]> {
  let summary = "Function generator definition.";
  let description = [{
    The `kgen.generator` operation defines a function generator. A generator is
    a parametric template for a function. It can have input parameters and
    result parameters. Uses of parameters inside the function must obey the
    parameter use-def graph; there can be no cycles. Each input parameter as it
    is declared in the signature of the generator can be used in the declaration
    of subsequent parameters only and in the declaration of all function
    arguments and results.

    Example:

    ```mlir
    kgen.generator @add<rhs>(%lhs: index) -> index {
      %0 = kgen.param.constant = <rhs>
      %1 = index.add %lhs, %0
      kgen.return %1 : index
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<StrAttr>:$sourceName,
    TypeAttrOf<KGEN_FuncTypeGeneratorType>:$funcTypeGenerator,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_DecoratorsAttr:$decorators,
    KGEN_InlineLevelAttr:$inlineLevel,
    KGEN_ExportKindAttr:$exportKind,
    UnitAttr:$external,
    // Annotating a generator as if it is an "always_inline("builtin")", this
    // means to get a normalized form, the calls (POC::Apply) to the generators
    // which have the `inlinedForm` must further inlined in the same way.
    OptionalAttr<TypedAttrInterface>:$inlinedForm,
    OptionalAttr<KGEN_LinkageNameAttr>:$linkageName,
    // To be able to store pre-elaborated names of the `llvm_metadata`,
    // decompose LLVM metadata into array of attributes of even size.
    // Each even attribute corresponds to the name of in `llvm_metadata`,
    // each odd corresponds to its value (zero index based)
    DefaultValuedAttr<ArrayAttr, "{}">:$LLVMMetadataArray,
    DefaultValuedAttr<ArrayAttr, "{}">:$LLVMArgMetadataArray,
    // Source-declared parameter list (names + passing kinds + variadic
    // kinds). Captured during `LowerLIT` from the source
    // `FuncTypeGeneratorType`'s `PogListAttr` metadata, with default values
    // and constraints elided so the snapshot is verifiable outside a
    // parametric scope. The live `funcTypeGenerator` and `inputParams` may
    // be rewritten by later transformations (e.g. `RemoveUnusedParams`),
    // but this slot is left untouched and is the intended source of truth
    // for reflection queries such as `#kgen.get_function_parameter_names`.
    //
    // TODO: If future reflection clients need defaults or constraints,
    // expand the snapshot (or add a parallel slot) and arrange for the
    // op's attribute dictionary to establish a contextual parametric
    // signature so `ParamIndexRefAttr`s in those fields can be verified.
    OptionalAttr<KGEN_PogListAttr>:$sourceParamList,
    // Stable snapshot of the source `FuncTypeGeneratorType` (parameter types +
    // body func type), captured during `LowerLIT` wrapped in a `TypeParamAttr`
    // (`#kgen.type<...>`) so `LowerLITTypes` lowers it in the value domain: the
    // snapshot's argument/result types (the generator body) come out in the
    // value domain, which reflection clients need. Its POG metadata is stripped
    // along with the live signature's; `sourceParamList` carries the
    // names/kinds. Unlike the live `funcTypeGenerator`, which later transforms
    // (e.g. `RemoveUnusedParams`) rewrite, this slot is left untouched and
    // snapshots the original pre-pruning signature.
    OptionalAttr<TypedAttrInterface>:$sourceFuncTypeGenerator
  );
  let regions = (region MaxSizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    `` custom<GeneratorOp>($exportKind, $sym_name, $funcTypeGenerator,
                           $functionType, $inputParams, $inlineLevel,
                           $decorators, attr-dict, $bodyRegion)
  }];

  let extraClassDeclaration = [{
    using OpTrait::SingleBlock<GeneratorOp>::getBody;
    Region *getCallableRegion() { return &getBodyRegion(); }

    ArrayAttr getArgAttrsAttr() { return nullptr; }
    ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(ArrayAttr attrs) {}
    void setResAttrsAttr(ArrayAttr attrs) {}
    Attribute removeArgAttrsAttr() { return nullptr; }
    Attribute removeResAttrsAttr() { return nullptr; }

    bool isExternal() { return getExternal(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$symName, "StringAttr":$sourceName,
                   "FuncTypeGeneratorType":$type,
                   "FunctionType":$functionType,
                   "ArrayRef<ParamDeclAttr>":$inputParams,
                   CArg<"InlineLevel", "InlineLevel::Automatic">:$inlineLevel,
                   CArg<"TypedAttr", "{}">:$inlinedForm,
                   CArg<"LinkageNameAttr", "{}">:$linkageNameAttr,
                   CArg<"ArrayAttr", "{}">:$llvmMetadataArray,
                   CArg<"ArrayAttr", "{}">:$llvmArgMetadataArray), [{
      if (!llvmMetadataArray)
        llvmMetadataArray = ArrayAttr::get(symName.getContext(), {});
      if (!llvmArgMetadataArray)
        llvmArgMetadataArray = ArrayAttr::get(symName.getContext(), {});
      build($_builder, $_state,
            symName, /*sym_visibility=*/nullptr, sourceName, type,
            functionType, inputParams,
            /*decorators=*/{}, inlineLevel, ExportKind::NotExported,
            /*external=*/false, inlinedForm, linkageNameAttr, llvmMetadataArray,
            llvmArgMetadataArray, /*sourceParamList=*/PogListAttr(),
            /*sourceFuncTypeGenerator=*/TypedAttr());
    }]>,
    OpBuilder<(ins "StringAttr":$symName, "FuncTypeGeneratorType":$type), [{
      build($_builder, $_state, symName, StringAttr::get(symName.getContext()),
            type, type.getBody().getValues(), {}, {});
    }]>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExternGeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_ExternGeneratorOp : KGENOp<"extern.generator", [
    HasParent<"ModuleOp">, SymbolName, SymbolVisibility, Symbol, DeclInterface,
    FuncInterface,
    ExportInterface, DeclareOpInterfaceMethods<FunctionOpInterface>]> {
  let summary = "External generator declaration.";
  let description = [{
    The `kgen.extern.generator` operation declares a KGEN generator with a
    external definition. The definition and its dependencies must be made
    available prior to elaboration, but this operations allows manipulating
    sections of IR without requiring all transitive dependencies be present in
    the module.

    Example:

    ```mlir
    kgen.extern.generator @kernel<simd_width>(!kgen.simd<simd_width, f32>)
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<KGEN_FuncTypeGeneratorType>:$funcTypeGenerator,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_ExportKindAttr:$exportKind
  );
  let regions = (region SizedRegion<0>:$body);

  let extraClassDeclaration = [{
    InlineLevel getInlineLevel() { return InlineLevel::Never; }
    Region *getCallableRegion() { return &getBody(); }

    ArrayAttr getArgAttrsAttr() { return nullptr; }
    ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(ArrayAttr attrs) {}
    void setResAttrsAttr(ArrayAttr attrs) {}
    Attribute removeArgAttrsAttr() { return nullptr; }
    Attribute removeResAttrsAttr() { return nullptr; }
  }];

  let assemblyFormat = [{
    `` custom<SymbolExport>($exportKind) $sym_name ``
    custom<ExternGenerator>($funcTypeGenerator, $functionType, $inputParams)
    attr-dict-with-keyword ($body^)?
  }];
}

//===----------------------------------------------------------------------===//
// StructGeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_StructGeneratorOp : KGENOp<"struct.generator", [SymbolName, SymbolVisibility, Symbol, SymbolTable, DeclInterface, SingleBlock,
    NoTerminator, KGEN_GeneratorOpInterface,
    DeclareOpInterfaceMethods<ExportInterface>,
    DeclareOpInterfaceMethods<StructDeclInterface>]> {
  let summary = "Struct generator definition.";
  let description = [{
    The `kgen.struct.generator` operation defines a type-value generator. It is
    a parametric template for a type-value. It can have input parameters.

    Its body contains the definition for a potentially parametric struct type
    as a type-value.

    Example:

    ```mlir
    kgen.struct.generator @struct_SIMD<dt: dtype, size> : type
      = struct_inst<"struct_SIMD"[dt, size]<:dtype dt, size>(data: struct<()>)>
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       KGEN_ParamDeclArrayAttr:$inputParams,
                       TypeAttr:$valueDomainType,
                       TypeAttr:$metaType);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name ``
    custom<StructGeneratorSpec>(
      $inputParams, $valueDomainType, $metaType, $body)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// StructInstanceOp
//===----------------------------------------------------------------------===//

def KGEN_StructInstanceOp : KGENOp<"struct.instance", [
    HasParent<"ModuleOp">, SymbolName, SymbolVisibility, Symbol, SymbolTable,
    SingleBlock, NoTerminator,
    KGEN_InstantiatedOpInterface]> {
  let summary = "Concretized struct instance.";
  let description = [{
    The `kgen.struct.instance` operation defines a concretized struct type
    instance.

    Its body contains the definition for a non-parametric struct type as a
    type-value.

    Example:

    ```mlir
    kgen.struct.instance @"struct_SIMD,dt=f32,size=4" : type
      = struct_inst<"struct_SIMD"[dt, size]<:dtype f32, 4>(data: struct<()>)>
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       TypeAttr:$valueDomainType,
                       TypeAttr:$metaType);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name `` custom<StructInstanceSpec>($valueDomainType, $metaType, $body)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// ConformanceOp
//===----------------------------------------------------------------------===//

def KGEN_ConformanceOp : KGENOp<"conformance", [
    SymbolTable, SingleBlock, NoTerminator]> {
  let summary = "Conformance table of a struct type.";
  let description = [{
    The `kgen.conformance` operation defines the conformance table of a struct
    type for a trait.

    Its body contains the conformance table entries that map trait requirements
    to the struct type's definitions.

    - The `sym_name` parameter is the flattened name of the trait being
      conformed to, which is what a `#kgen.get_witness` looks this table up by.
    - The optional `traitSymbol` parameter references the trait declaration
      that `sym_name` is the flattened form of. It is present until `lower-lit`
      erases the trait declarations, after which nothing can resolve it and the
      flattened `sym_name` is the only identity that remains.
    - The `immediateParents` parameter names the conformance tables that this
      conformance table directly inherits from, sorted by flattened name. It
      only includes the first level of parents, not any further ancestors.
    - The `constraint` parameter specifies the condition under which this
      conformance applies. This is used for conditional trait conformance,
      where a struct only conforms to a trait when certain conditions are met.
      Unconditional conformances use a trivially true constraint (proposition =
      constant 1) and are not printed with a `where` clause. Note: the builder
      canonicalizes null constraints to the trivially true constraint.

    Logically, a ConformanceOp represents a witness table whose contents is a
    concatenation of each parent ConformanceOp's conformance table followed by
    the entries of the ConformanceOp itself. The parent conformance tables are
    ordered by the name of the ConformanceOps (also the order in
    `immediateParents`).

    Example:

    ```mlir
    kgen.struct.generator @SIMD<type: dtype, size> = ... {
      kgen.conformance @Boolable {
        kgen.witness @"__bool__" : (!kgen.simd<size, type>) -> i1
          = @"SIMD::__bool__(::SIMD[$0, $1])"<:dtype type, size>
      }
      ...
    }
    ```

    Example with conditional conformance:

    ```mlir
    kgen.struct.generator @List<T: type> = ... {
      kgen.conformance @Copyable where #kgen.constraint<conforms_to(T, Copyable), loc> {
        ...
      }
      ...
    }
    ```
  }];

  let arguments = (ins
    KGEN_TraitSymbolAttr:$traitSymbol,
    DefaultValuedAttr<KGEN_TraitSymbolArrayAttr,
                      "ArrayRef<TraitSymbolAttr>{}">:$immediateParents,
    KGEN_ConstraintAttr:$constraint
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  // Skip default builders to avoid conflict with our canonicalizing builders.
  let skipDefaultBuilders = 1;

  // Builders with constraint canonicalization - null constraints become
  // the trivially true constraint (proposition = constant 1).
  let builders = [
    // Primary builder that canonicalizes null constraints to the trivially
    // true constraint. This allows callers to pass a potentially-null
    // constraint.
    OpBuilder<(ins "TraitSymbolAttr":$traitSymbol,
                   "TraitSymbolArrayAttr":$immediateParents,
                   "ConstraintAttr":$constraint), [{
      $_state.addAttribute(getTraitSymbolAttrName($_state.name), traitSymbol);
      $_state.addAttribute(getImmediateParentsAttrName($_state.name),
                           immediateParents);
      $_state.addAttribute(getConstraintAttrName($_state.name),
                           constraint ? constraint
                                      : getUnconditionalConstraint(
                                            $_builder.getContext()));
      (void)$_state.addRegion();
    }]>,
    // Builder for unconditional conformance (no constraint argument).
    OpBuilder<(ins "TraitSymbolAttr":$traitSymbol,
                   "TraitSymbolArrayAttr":$immediateParents), [{
      build($_builder, $_state, traitSymbol, immediateParents,
            /*constraint=*/ConstraintAttr());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// WitnessOp
//===----------------------------------------------------------------------===//

def KGEN_WitnessOp : KGENOp<"witness", [
    SymbolName, SymbolVisibility, Symbol, HasParent<"ConformanceOp">]> {
  let summary = "Witness table entry of a struct type.";
  let description = [{
    The `kgen.witness` operation defines a witness table entry in a
    conformance table. It represents a single requirement satisfied by a struct
    type for the trait being conformed to.

    Example:

    ```mlir
    kgen.struct.generator @SIMD<type: dtype, size> = ... {
      kgen.conformance @Boolable {
        kgen.witness @"__bool__" : (!kgen.simd<size, type>) -> i1
          = @"SIMD::__bool__(::SIMD[$0, $1])"<:dtype type, size>
      }
      ...
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       TypedAttrInterface:$value);

  let assemblyFormat = [{
    custom<WitnessEntry>($sym_name, $value) attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// CallOpBase
//===----------------------------------------------------------------------===//

class KGEN_CallOpBase<string mnemonic, list<Trait> traits = []>
    : KGENOp<mnemonic, traits # [
        DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
        DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]>;

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def KGEN_CallOp : KGEN_CallOpBase<"call",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Call a func, generator, or generator interface";
  let description = [{
    The `kgen.call` operation invokes a parametric function. The callee must be
    a flat symbol reference to a KGEN function, generator, or interface. The
    operation must bind parameter values as required by the callee. The
    operation's operands are used as function arguments and the operation's
    results match the return values of the function.

    Example:

    ```mlir
    kgen.func @foo(%arg0: index) {
      kgen.return %arg0 : index
    }

    %0 = kgen.call @foo(%arg0) : (index) -> index
    ```

    Specify values for the input parameters to the callee using a parameter
    binding list and declarations for result parameters with a parameter
    declaration list.

    Example:

    ```mlir
    kgen.generator @add2<N -> index>() {
      kgem.param.result_bind<add(N, 2)>
      kgen.return
    }

    kgen.call @add2<N = 2 -> NPlus2>() : () -> ()
    ```
  }];

  let arguments = (ins
    KGEN_SymbolConstantAttr:$callee,
    Variadic<AnyType>:$operands,
    DefaultValuedAttr<KGEN_TailKindAttr, "{}">:$tailKind
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<OptionalEnum>($tailKind, "symbolizeTailKind")
    custom<CallOp>($callee, $operands, type($operands), type($results))
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "SymbolConstantAttr":$callee, "ValueRange":$operands), [{
      auto tailKind = TailKindAttr::get($_builder.getContext(), TailKind::None);
      return build($_builder, $_state,
        callee.getType().getBody().getValues().getResults(), callee,
        operands, tailKind);
    }]>
  ];

  let extraClassDeclaration = [{
    void concretizeCallee(IRRewriter &b, SymbolConstantAttr callee);

    /// Return the symbol reference to the callee.
    FlatSymbolRefAttr getCalleeSymbol() {
      return cast<FlatSymbolRefAttr>(getCallee().getSymbol());
    }

    FuncTypeGeneratorType getCalleeType() {
      return cast<FuncTypeGeneratorType>(getCallee().getType());
    }

    /// Return the bound input parameter values.
    ArrayRef<TypedAttr> getParamValues() {
      return getCallee().getParamValues();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

def KGEN_CallParamOp : KGEN_CallOpBase<"call_param",
    [CalleeArgumentTypes, CalleeResultTypes,
     DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Call declaration specified by a parameter expression";
  let description = [{
    The `kgen.call_param` operation invokes a parametric callee. The callee is
    a parameter expression with a signature type, which in practice is either a
    symbol constant (a KGEN function or generator), or a region
    body passed from higher up the call stack.

    Example:

    ```mlir
    // Symbol constant callee.
    %0 = kgen.call_param[(index) -> index: @someFn](%arg0)

    // Parameter reference callee.
    %1 = kgen.call_param[<N>() -> index: foo]<N = 4>()
    ```
  }];

  let arguments = (ins
    AttrOfType<KGEN_FuncTypeGeneratorType>:$callee,
    Variadic<AnyType>:$operands,
    DefaultValuedAttr<KGEN_TailKindAttr, "{}">:$tailKind
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<OptionalEnum>($tailKind, "symbolizeTailKind")
    `` custom<ParametricCallee>($callee) `(` $operands `)` attr-dict
  }];

  let extraClassDeclaration = [{
    void concretizeCallee(IRRewriter &b, SymbolConstantAttr callee);
    bool isAllowedInFunc() { return false; }

    FuncTypeGeneratorType getCalleeType() {
      return cast<FuncTypeGeneratorType>(getCallee().getType());
    }
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

def KGEN_ParamConstantOp : KGENOp<"param.constant", [ConstantLike,
    BytecodeOpInterface,
    // TODO(MOCO-1101): Replace two lines below with Pure
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>,
    NoParamDecls {

  let summary = "create a constant value from a parameter value";
  let description = [{
    The `kgen.param.constant` operation materializes the value of a parameter
    expression as a constant SSA value that may be used by other operations.
    Conceptually, it bridges the parameter/compile-time value domain to the
    dynamic/runtime value domain.

    Example:

    ```mlir
    %cst0 = kgen.param.constant: scalar<f32> = <"1.5">
    %cst1 = kgen.param.constant = <apply(:() -> index @makeIndex)>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `` custom<ParamConstantOpValue>($value, type($result)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$value),
              [{ build($_builder, $_state, value.getType(), value); }]>
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    struct alignas(8) Payload { bool containsPtr; Attribute value; };
    ErrorOrSuccess compile(Payload &payload, TargetInfoAttr target);
    ErrorOrSuccess parametric_compile(Payload &payload,
      TargetInfoAttr target,
      ArrayRef<Attribute> operands,
      ParametricInterpreterState& state);
    ErrorTreeOrSuccess interpret(ArrayRef<Attribute> operands,
      const Payload &payload, InterpreterState &state);
    ErrorTreeOrSuccess parametric_interpret(ArrayRef<Attribute> operands,
      const Payload &payload, ParametricInterpreterState &state);
  }];
}

//===----------------------------------------------------------------------===//
// ParamMaterializeOp
//===----------------------------------------------------------------------===//

def KGEN_ParamMaterializeOp : KGENOp<"param.materialize", [
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>,
    BytecodeOpInterface,]>,
    NoParamDecls {
  let summary = [{
    The `kgen.param.materialize` operation materializes the value of a parameter
    expression as an SSA value that may be used by other operations. It bridges
    the parameter domain to the runtime domain. This operation assumes the
    materialization of the parameter expression may have memory allocation
    side-effects.

    Example:

    ```mlir
    %value0 = kgen.param.materialize = <1>
    %value1 = kgen.param.materialize: pointer<f32> = <ptr_val>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Res<AnyType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ParamConstantOpValue>($value, type($result)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$value),
              [{ build($_builder, $_state, value.getType(), value); }]>
  ];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let extraClassDeclaration = [{
    struct alignas(8) Payload { bool containsPtr; Attribute value; };
    ErrorOrSuccess compile(Payload &payload, TargetInfoAttr target);
    ErrorOrSuccess parametric_compile(Payload &payload,
      TargetInfoAttr target,
      ArrayRef<Attribute> operands,
      ParametricInterpreterState& state);
    ErrorTreeOrSuccess interpret(ArrayRef<Attribute> operands,
      const Payload &payload, InterpreterState &state);
    ErrorTreeOrSuccess parametric_interpret(ArrayRef<Attribute> operands,
      const Payload &payload, ParametricInterpreterState &state);
  }];
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

def KGEN_ParamDeclareOp : KGENOp<"param.declare",
    [DeclareOpInterfaceMethods<ParamOpInterface>,
     DeclareOpInterfaceMethods<InterpreterOpInterface>]>,
    OneParamDecl {
  let summary = "Declare a single parameter value.";
  let description = [{
    The `kgen.param.declare` operation declares a single parameter and binds
    its value to a parameter expression. The parameter is visible within and
    below the scope of the enclosing `DeclInterface` operation.

    Example:

    ```mlir
    kgen.param.declare A = <5>
    kgen.param.declare A_plus_one = <add(A, 1)>
    ```
  }];

  let arguments = (ins KGEN_ParamDeclAttr:$paramDecl,
                       TypedAttrInterface:$value);
  let assemblyFormat = [{
    custom<ParamDeclareOpValue>($paramDecl, $value) attr-dict
  }];

  let extraClassDeclaration = [{
    StringAttr getName() { return getParamDecl().getName(); }
    Type getType() { return getParamDecl().getType(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamDeclareRegionOp
//===----------------------------------------------------------------------===//

def KGEN_ParamDeclareRegionOp : KGENOp<"param.declare.region", [
    DeclInterface, FuncInterface, FunctionOpInterface, FunctionLike,
    SymbolName, SymbolVisibility,
    DeclareOpInterfaceMethods<ParamOpInterface, ["collectParameterUses"]>,
    SingleBlock
  ]>, OneParamDecl {
  let summary = "declare a region parameter";
  let description = [{
    The `kgen.param.declare.region` operation declares a signature-type
    parameter whose value is an MLIR region (a closure). The region can be
    isolated from above, in which case it is treated by the elaborator as a
    stateless closure and inlined at wherever it ends up being called. The
    region can also capture values from above, in which case the elaborator
    treats it as a stateful closure and processes it by inlining every call that
    forwards this parameter down to all callsites.

    Parameters defined by a `kgen.param.declare.region` that are not isolated
    from above can only be used by function calls. They cannot be used, for
    example, by `kgen.addressof`.

    Example:

    ```mlir
    kgen.param.declare.region AddIt[my_add] = <N>(%arg0: index) -> index {
      %0 = kgen.param.constant = <N>
      %1 = index.add %0, %arg0
      kgen.return %1 : index
    }

    kgen.param.declare.region SubtractIt = <N>(%arg0: index) -> index {
      %0 = kgen.param.constant = <N>
      %1 = index.sub %0, %arg0
      kgen.return %1 : index
    }
    ```

    The operation carries an extra bit `isolated` to indicate that they are
    parametrically isolated from above.
  }];

  let arguments = (ins
    // `kgen.param.declare.region` is anonymous: it satisfies SymbolOpInterface
    // through FunctionOpInterface but never carries a name of its own.
    OptionalAttr<SymbolNameAttr>:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    KGEN_ParamDeclAttr:$paramDecl,
    StrAttr:$sourceName,
    TypeAttrOf<KGEN_FuncTypeGeneratorType>:$funcTypeGenerator,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_InlineLevelAttr:$inlineLevel,
    OptionalAttr<KGEN_LinkageNameAttr>:$linkageName,
    // To be able to store pre-elaborated names of the `llvm_metadata`,
    // decompose LLVM metadata into array of attributes of even size.
    // Each even attribute corresponds to the name of in `llvm_metadata`,
    // each odd corresponds to its value (zero index based)
    DefaultValuedAttr<ArrayAttr, "{}">:$LLVMMetadataArray,
    DefaultValuedAttr<ArrayAttr, "{}">:$LLVMArgMetadataArray,
    UnitAttr:$isolated
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    custom<RegionDeclaration>($paramDecl, $sourceName, $inputParams,
                              $functionType, $funcTypeGenerator, $inlineLevel,
                              $bodyRegion)
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);
    Region *getCallableRegion() { return &getBodyRegion(); }

    /// Region body op doesn't form a symbol.
    bool isOptionalSymbol() { return true; }

    ArrayAttr getArgAttrsAttr() { return nullptr; }
    ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(ArrayAttr attrs) {}
    void setResAttrsAttr(ArrayAttr attrs) {}
    Attribute removeArgAttrsAttr() { return nullptr; }
    Attribute removeResAttrsAttr() { return nullptr; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamApplyOp
//===----------------------------------------------------------------------===//

def KGEN_ParamApplyOp : KGENOp<"param.apply",
    [DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>,
     DeclareOpInterfaceMethods<ParamOpInterface>,
     DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Function call in parameter domain.";
  let description = [{
    The `kgen.param.apply` operation is a call operation entirely in the
    parameter domain. The operands to the call are parameter expressions and the
    results of the call are bound to parameter declarations. This is the 'apply'
    operator as an operation.

    The callee is limited to a single result value with no result parameters.

    Example:

    ```mlir
    kgen.param.declare sw: i1 = <1>
    kgen.param.apply A = [(i1) -> !kgen.simd<8, f32>: callee](sw)
    ```
  }];

  let arguments = (ins KGEN_ParamDeclAttr:$paramDecl,
                       AttrOfType<KGEN_FuncTypeGeneratorType>:$callee,
                       KGEN_ParameterExprArrayAttr:$operands);
  let assemblyFormat = [{
    custom<ParamApplyOp>($paramDecl, $callee, $operands) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CostOfOp
//===----------------------------------------------------------------------===//

def KGEN_CostOfOp : KGENOp<"cost_of", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]> {
  let summary = "Return the 'cost' of the function.";
  let description = [{
    The `kgen.cost_of` operation takes a parametric callee and returns
    characteristics of the function that can be used to develop heuristics for
    the cost of the function. This operation must be resolved at compile time.

    Currently, `kgen.cost_of` returns the number of loads, stores, additions,
    comparisons, divisions, multiplications, multiply-adds, and other
    operations, that is, operations that do not fall into any of the above
    categories. The cost is evaluated on the function at the output of
    elaboration, without running any post-elaboration passes.

    Example:

    ```mlir
    %loads, %stores, %additions, %comparisons, %divisions, %multiplications,
    %multiply_adds, %other = kgen.cost_of[(si8) -> si8: @foo]
    ```
  }];

  let arguments = (ins TypedAttrInterface:$callee);
  let results = (outs Index:$loads,
                      Index:$stores,
                      Index:$additions,
                      Index:$comparisons,
                      Index:$divisions,
                      Index:$multiplications,
                      Index:$multiplyAdds,
                      Index:$other);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def KGEN_ReturnOp : KGENOp<"return", [Terminator, ReturnLike,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "return operation for kgen declarations";
  let description = [{
    The `kgen.return` operation specifies the returned values for a `kgen.func`
    or `kgen.generator`.

    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the
    `kgen.generator` (or enclosing region) that contains the operation. For
    example:

    ```mlir
      kgen.generator @foo() : i32, f8 {
        ...
        kgen.return %0, %1 : i32, f8
      }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

def KGEN_UnreachableOp : KGENOp<"unreachable", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Unreachable operation";
  let description = [{
    The `kgen.unreachable` operation is a block terminator that indicates that
    the previous operations cannot continue control flow.  This is used after
    infinite loops and no-return functions to simplify dataflow analysis.

    Example:

    ```mlir
    hlcf.loop () {
      kgen.call @printHello()
      hlcf.continue
    }
    kgen.unreachable
    ```
  }];

  let arguments =
   (ins DefaultValuedOptionalAttr<BoolAttr, "false">:$isAfterUnreachableCall);

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

def KGEN_ParamAssertOp : KGENOp<"param.assert",
    [DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>,
     DeclareOpInterfaceMethods<InterpreterOpInterface>]>,
    NoParamDecls {
  let summary = "check a parameter expression at elaboration time";
  let description = [{
    The `kgen.param.assert` operation ensures that a parameter expression is
    true at elaboration time.
  }];

  let arguments = (ins AttrOfType<BoolScalar>:$cond,
                       AttrOfType<KGEN_StringType>:$message);
  let assemblyFormat = [{
    ` ` `<` custom<ScalarBoolParamValue>($cond) `>` `,`
    custom<StringParam>($message) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ParamForOp
//===----------------------------------------------------------------------===//

def KGEN_ParamForOp : KGENOp<"param.for", [DeclInterface,
    DeclareOpInterfaceMethods<ControlFlowNode>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric",
      "collectParameterUsesBelow"]>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]>, NoParamDecls {
  let summary = "compile-time for loop";
  let description = [{
    The `kgen.param.for` operation instantiates its body with values according
    to its iterator. It takes an initial iterator value, a 'hasNext' function
    that take an iterator and indicates whether more elements exist, and a
    'getNextIter' function that takes an iterator instance and returns the next
    iterator value.

    This operation can have loop-carried values - the "operands" inputs and
    results, which are values promoted within the loop by mem2reg.
  }];

  let arguments = (ins
    TypedAttrInterface:$initial,
    AttrOfType<KGEN_FuncTypeGeneratorType>:$hasNext,
    AttrOfType<KGEN_FuncTypeGeneratorType>:$getNextIter,
    KGEN_ParamDeclAttr:$paramDecl,
    Variadic<AnyType>:$operands,
    UnitAttr:$bodyIsolated,
    UnitAttr:$elseIsolated
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body, SizedRegion<1>:$elseRegion);

  let builders = [
    OpBuilder<(ins "TypedAttr":$initial, "TypedAttr":$hasNext,
                   "TypedAttr":$getNextIter, "ParamDeclAttr":$paramDecl), [{
      build($_builder, $_state, {}, initial, hasNext, getNextIter, paramDecl,
            {}, {}, {});
    }]>
  ];

  let assemblyFormat = [{
    custom<ParamDecl>($paramDecl) `in` custom<ColonTypeParamValue>($initial)`\n`
    `has_next` custom<ColonTypeParamValue>($hasNext) `\n`
    `get_next_iter` custom<ColonTypeParamValue>($getNextIter)
    custom<Loop>($operands, type($operands), type($results), $body)
    `else` custom<RegionWithArgs>($elseRegion) attr-dict
  }];

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);
    ArrayRef<ParamDeclAttr> getInputParams();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamForBreakOp
//===----------------------------------------------------------------------===//

def KGEN_ParamForBreakOp : KGENOp<"param.for.break", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "param loop exit operation";
  let description = [{
    The `kgen.param.for.break` operation represents an exit from a
    `kgen.param.for`, branching to the end of all generated iterations.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>,
  ];

  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// ParamForContinueOp
//===----------------------------------------------------------------------===//

def KGEN_ParamForContinueOp : KGENOp<"param.for.continue", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "param loop continue operation";
  let description = [{
    The `kgen.param.for.continue` operation branches to the next generated
    iteration of the loop.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>,
  ];

  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// ParamForGotoElseOp
//===----------------------------------------------------------------------===//

def KGEN_ParamForGotoElseOp : KGENOp<"param.for.goto.else", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "param loop 'goto else' operation";
  let description = [{
    The `kgen.param.for.goto.else` operation jumps to the 'else' block of a
    `kgen.param.for`.  It only exists to help the interface between the Mojo
    parser and the LowerSemanticCF pass work more easily.
  }];

  let arguments = (ins);

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ParamIfOp
//===----------------------------------------------------------------------===//

def KGEN_ParamIfOp : KGENOp<"param.if", [DeclInterface,
    DeclareOpInterfaceMethods<ControlFlowNode>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric",
      "collectParameterUsesBelow"]>]>, NoParamDecls {
  let summary = "compile-time constant if";
  let description = [{
    The `kgen.param.if` op provides a compile-time guarantee that if `cond`
    is false, only the `else` block will be elaborated, and if `cond` is true,
    only the `then` block will be elaborated. Whichever block is elaborated
    will be directly inlined into the scope. This op is fully removed
    during elaboration.

    The result parameter will be resolved by the parameter on the
    `param.yield` in the branch that is actually elaborated.

    ```mlir
    %0 = kgen.param.if <condition> -> index {
      %i0 = index.constant 0
      kgen.param.yield %i0
    } else {
      %i1 = index.constant 1
      kgen.param.yield %i1
    }

    kgen.param.if <condition -> result> {
      kgen.param.yield<1>
    } else {
      kgen.param.yield<0>
    }
    %0 = kgen.param.constant = <result> // == <condition ? 1 : 0>
    ```
  }];

  let arguments = (ins
    AttrOfType<BoolScalar>:$cond,
    UnitAttr:$thenIsolated,
    UnitAttr:$elseIsolated
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);

    Block::OpListType &getThenOps() {
      return getThenRegion().front().getOperations();
    }
    Block::OpListType &getElseOps() {
      return getElseRegion().front().getOperations();
    }

    /// Parametric ifs have no parameters.
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }
  }];

  let assemblyFormat = [{
    ` ` `<` custom<ScalarBoolParamValue>($cond) `` `>`
    (`->` type($results)^)?  $thenRegion `else` $elseRegion attr-dict
  }];

  let builders = [OpBuilder<(ins "TypedAttr":$cond), [{
    build($_builder, $_state, {}, cond, {});
  }]>];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ParamYieldOp
//===----------------------------------------------------------------------===//

def KGEN_ParamYieldOp : KGENOp<"param.yield", [Pure, Terminator,
    ParentOneOf<["ParamIfOp", "ParamForOp"]>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "conditional yield operation";
  let description = [{
    The `kgen.param.yield` operation is used to denote the terminator of a
    block in the `kgen.param.if` op. Conceptually, it branches to the next
    op after a `kgen.param.if`, but in reality it's to make sure the IR
    stays in a reasonable state and is possible to inline directly.

    This op defines the parameter that is used as the result parameter on the
    enclosing `kgen.param.if`.

    Example:

    ```mlir
    %0 = kgen.param.if <condition> -> index {
      kgen.param.yield %arg1 : index
    } else {
      kgen.param.yield %arg2 : index
    }
    ```

    This operation can always be marked pure because the control-flow edge from
    this operation leads only to after the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ return build($_builder, $_state, {}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

def KGEN_RebindOp : KGENOp<"rebind", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "rebind parametric types";
  let description = [{
    The `kgen.rebind` operation rebinds values with parametric types into other
    parameter domains. The operation can be used to convert between parameter
    domains or between concrete types. During elaboration, rebind operations
    must resolve to the same input and output types. Otherwise, elaboration will
    fail.

    Example:

    ```mlir
    // Rebind a parameterized type to a concrete type.
    %0 = kgen.rebind %arg0 : !kgen.param<type> to !kgen.scalar<f32>

    // Rebind between parameter domains.
    %1 = kgen.rebind %arg1 : !kgen.param<type> to !kgen.simd<size, dtype>

    // Unbind a concrete type to one with a parameter.
    %2 = kgen.rebind %arg2 : !kgen.scalar<f32> to !kgen.scalar<dtype>

    // ERROR: Cannot rebind between different concrete types.
    %3 = kgen.rebind %arg2 : !kgen.scalar<f32> to !kgen.scalar<si32>
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// If 'value' is defined by one or more rebinds, look through them.
    static Value strip(Value value);
  }];
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def KGEN_CallIndirectOp : KGENOp<"call_indirect", [
    TypesMatchWith<"operands match callee argument types", "callee",
    "arguments",
    "cast<FuncTypeGeneratorType>($_self).getBody().getArguments()">,
    CalleeResultTypes,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Call a signature typed instance";
  let description = [{
    The `kgen.call_indirect` operation takes an SSA value of `!kgen.generator`
    type (that wraps a `!kgen.func` type) and invokes it with the provided
    operands as arguments.

    Example:

    ```mlir
    %0 = kgen.call_indirect %closure(%arg0, %arg1)
      : (index, index) capturing -> index
    ```
  }];

  let arguments = (ins
    KGEN_FuncTypeGeneratorType:$callee,
    Variadic<AnyType>:$arguments,
    DefaultValuedAttr<KGEN_TailKindAttr, "{}">:$tailKind
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<OptionalEnum>($tailKind, "symbolizeTailKind") $callee
    `(` $arguments `)` `:` custom<Generator>(type($callee)) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StageClosureOp
//===----------------------------------------------------------------------===//

def KGEN_StageClosureOp : KGENOp<"stage_closure",
  [Pure, FunctionLike, DebugInfo_InlinedSubprogramScoped]
>{
  let summary = "Stage a closure declaring a concrete region";
  let description = [{
    The `kgen.stage_closure` operation declares a named concrete region. The
    region can capture values from its parent region. This operation models
    nested callable functions within their original scope before lifting them
    into functions so that we may perform transformations that depend on the
    original structure of the scopes.

    ```mlir
    %0 = kgen.stage_closure = () capturing -> index {
      kgen.return %arg0 : index
    }
    ```
  }];

  let results = (outs KGEN_FuncTypeGeneratorType:$result);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    custom<StageClosureOp>(type($result), $bodyRegion) attr-dict
  }];

  let extraClassDeclaration = [{
    ArrayRef<Type> getResultTypes() {
       return getResult().getType().getBody().getValues().getResults();
    }
  }];
}

//===----------------------------------------------------------------------===//
// CreateClosureOp
//===----------------------------------------------------------------------===//

def KGEN_CreateClosureOp : KGEN_CallOpBase<"create_closure", [Pure,
    InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Create a closure from variables and a function.";
  let description = [{
    The `kgen.create_closure` operation represents the instantiation of
    a closure. This operation models closure creation in terms of a function
    and local captured variables.

    ```mlir
    %idx0 = index.constant 0
    %0 = kgen.create_closure[(index) -> index: @h](%idx0)
    ```
  }];

  let arguments = (ins AttrOfType<KGEN_FuncTypeGeneratorType>:$callee,
                       Variadic<AnyType>:$captures);
  let results = (outs KGEN_FuncTypeGeneratorType:$result);
  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee) `(` $captures `)`
    custom<ClosureCaptureTypes>(ref($callee), ref($captures), type($captures))
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$callee), [{
      build($_builder, $_state, callee, {});
    }]>,
  ];

  let extraClassDeclaration = [{
    FuncTypeGeneratorType getCalleeType() {
      return cast<FuncTypeGeneratorType>(getCallee().getType());
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CreateRegStubOp
//===----------------------------------------------------------------------===//

def KGEN_CreateRegStubOp : KGENOp<"create_reg_stub", [Pure,
  DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]> {
  let summary = "Create a stub to enforce by-memory arguments for `callee`.";
  let description = [{

    This op is similar to CreateClosureOp (without captures).
    But during LowerArgConvention, some arguments are promoted from by-memory
    to by-value.
    This would also affect the signature type of CreateClosureOp.

    With CreateRegStubOp, during LowerArgConvention, the signature of `callee`
    might change, but the result signature always remains the same.
    The compiler also adds the correct load / stores around `callee`'s call.

    This is needed for interoperability. We can hold mojo register_passable
    types by pointer in C++, and create a function pointer with CreateRegStub.
    Even if the compiled mojo functions excepts value arguments, you can have
    a function pointer to a wrapper that takes arguments by pointer.

    For example:
    ```mlir
    kgen.create_reg_stub [
      (!kgen.pointer<index> owned_in_mem, !kgen.pointer<index> byref_result)
      -> !kgen.none: @regtype__moveinit__] :
      <(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem,
        !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none>
    ```

    The type is wrapped around a memory struct to preserve the signature,
    and also to indicate to LLVM that pointers don't alias.

    After LowerCallConvention, only `callee`'s signature change:
    ```mlir
    kgen.create_reg_stub [(index) -> index: @regtype__moveinit__] :
      <(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem,
        !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none>
    ```

    Callee can also take any other kind of arguments (eg memory mojo objects,
    scalars, raw pointers) that don't get promoted:
    ```mlir
    kgen.create_reg_stub [
      (!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem,
      !kgen.scalar<si16> borrow) -> !kgen.none: @foo] :
      <(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem,
      !kgen.scalar<si16> borrow) -> !kgen.none>
    ```
  }];

  let arguments = (ins AttrOfType<KGEN_FuncTypeGeneratorType>:$callee);
  let results = (outs KGEN_FuncTypeGeneratorType:$result);

  let assemblyFormat = [{
    custom<ParametricCallee>($callee) attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$callee)>,
  ];

  let extraClassDeclaration = [{
    /// Get the signature type of the output stub.
    /// `calleeSign` is the original callee with by-memory operands.
    static FuncType getStubSignatureType(FuncType calleeSign);

    /// The stub signature wraps some arguments to preserve the semantics.
    /// Returns the original (unwrapped) argument type.
    Type getOriginalArgType(unsigned index);

    /// Returns the type of the arg `index` in callee's signature.
    /// It takes into account that after callee is lowered some arguments are
    /// now outputs.
    Type getCalleeArgType(unsigned index);
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SourceLocOp
//===----------------------------------------------------------------------===//

def KGEN_SourceLocOp : KGENOp<"source_loc", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Track source location and update conditionally during inlining";
  let description = [{
    The `kgen.source_loc` operation enables implementing location capture of
    call stacks. When defined in an `@always_inline` or
    `@always_inline("nodebug")` function, it will return the capturing the call
    location of the enclosing function. By specifying an `inlineCount` larger
    than 0, it can generalize this behavior, e.g. for `inlineCount = 1`, it will
    return the call location of two steps up the call stack (as long as both
    calls are to `@always_inline` or `@always_inline("nodebug")` functions). In
    parameter contexts, no location context can be recovered, so the op will be
    interpreted as a constant of dummy location info.

    Example:
    ```mlir
     %line, %col, %fileName = kgen.source_loc[0]
    ```
  }];

  let arguments = (ins AttrOfType<Index>:$inlineCount);
  let results = (outs Index:$line, Index:$col, KGEN_StringType:$fileName);
  let assemblyFormat = [{
    `[` custom<IndexParamValue>($inlineCount) `]` attr-dict
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

def KGEN_StructCreateOp : KGENOp<"struct.create", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    TypesMatchWith<"operand types match struct element types",
                   "result", "elements",
                   "*llvm::cast<StructType>($_self).getElementTypes()">
]> {
  let summary = "create an instance of a struct";
  let description = [{
    The `kgen.struct.create` operation creates a struct of the given type
    from values corresponding to its element types.

    Example:

    ```mlir
    %0 = kgen.struct.create(%f32, %f64)
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    %1 = kgen.struct.create(%arr, %ptr, %v)
      : !kgen.struct<(array<size, type>, pointer<type>, type)>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs KGEN_StructType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

def KGEN_StructExtractOp : KGENOp<"struct.extract", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    InferTypeOpAdaptor]> {
  let summary = "struct extract value";
  let description = [{
    The `kgen.struct.extract` operation gets the struct element at the given
    index from the provided struct.

    Example:

    ```mlir
    // Extract the !kgen.scalar<f32> at index 0.
    %0 = kgen.struct.extract %struct[0]
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    // Extract the !kgen.scalar<f64> at index 1.
    %1 = kgen.struct.extract %struct[1]
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    // Extract an element at a parametric index.
    kgen.generator @example<I: index>(%struct: !kgen.struct<(i32, f32)>) {
      %0 = kgen.struct.extract %struct[I] : !kgen.struct<(i32, f32)>
      kgen.return
    }
    ```
  }];

  let arguments = (ins KGEN_StructType:$container, AttrOfType<Index>:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $container `[` custom<IndexParamValue>($index) `]` attr-dict `:` type($container)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;

  let builders = [
    // Builder with constant integer index (infers result type).
    OpBuilder<(ins "Value":$container, "unsigned":$index)>,
  ];

}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

def KGEN_StructReplaceOp : KGENOp<"struct.replace", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    AllTypesMatch<["container", "result"]>]> {
  let summary = "struct insert value";
  let description = [{
    The `kgen.struct.replace` operation inserts the given value into the struct
    at the given index. It returns a new struct with the inserted value.

    Example:

    ```mlir
    // Insert the !kgen.scalar<f32> at index 0.
    %0 = kgen.struct.replace %f32, %struct[0]
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    // Insert the !kgen.scalar<f64> at index 1.
    %1 = kgen.struct.replace %f64, %struct[1]
      : !kgen.struct<(scalar<f32>, scalar<f64>)>
    ```
  }];

  let arguments = (ins AnyType:$value, KGEN_StructType:$container,
                       UnsignedIndexAttr:$index);
  let results = (outs KGEN_StructType:$result);
  let assemblyFormat = [{
    $value `,` $container `[` $index `]` attr-dict `:`
    qualified(type($container)) ``
    custom<StructValueType>(type($value), ref(type($container)), ref($index))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

def KGEN_StructGEPOp : KGENOp<"struct.gep", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "struct get element pointer";
  let description = [{
    The `kgen.struct.gep` operation takes a pointer to a `!kgen.struct` and an
    index and returns a pointer to the struct element at that index. The index
    can be either a constant integer or a parametric expression that is resolved
    post-elaboration.

    Example:

    ```mlir
    %struct = pop.stack_allocation 1 : !kgen.struct<(i32, i64)>
    // Constant index - result type is inferred
    %i64Ptr = kgen.struct.gep %struct[1] : <struct<(i32, i64)>>
    ```

    With a parametric index:

    ```mlir
    kgen.generator @example<I: index>(%struct: !kgen.pointer<struct<(i32, i64)>>) {
      // Parametric index - result type must be specified
      %ptr = kgen.struct.gep %struct[I] : <struct<(i32, i64)>> -> <i64>
    }
    ```

    To get a pointer to a struct from a value-semantic struct, the struct value
    must be stored into a block of allocated memory first.

    ```mlir
    ^bb0(%struct: !kgen.struct<(i32, i64)>):
      %mem = pop.stack_allocation 1 : !kgen.struct<(i32, i64)>
      pop.store %struct, %mem : !kgen.pointer<struct<(i32, i64)>>
    ```
  }];

  let arguments = (ins PointerType:$container, AttrOfType<Index>:$index);
  let results = (outs PointerType:$result);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    // Builder with constant integer index (infers result type).
    OpBuilder<(ins "Value":$container, "unsigned":$index)>,
    // Builder with parametric index (requires explicit result type).
    OpBuilder<(ins "Type":$resultType, "Value":$container, "TypedAttr":$index)>
  ];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructLoadIndirectOp
//===----------------------------------------------------------------------===//

def KGEN_StructLoadIndirectOp : KGENOp<"struct.load_indirect", [Pure,
    InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "load all elements of a struct containing pointers";
  let description = [{
    The `kgen.struct.load_indirect` operation takes a struct of !kgen.pointer
    values and loads each one into a struct without the pointer type.  This
    requires elements with trivially loadable types supported by pop.load.

    When the operand struct is a variadic parameter pack (`isParamPack`), the
    result struct is also marked `isParamPack` so downstream ABI lowering can
    still recognize a pack.
  }];

  let arguments = (ins KGEN_StructType:$structValue);
  let results = (outs KGEN_StructType:$result);
  let assemblyFormat = "$structValue attr-dict `:` type($structValue)";
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

def KGEN_VariantCreateOp : KGENOp<"variant.create", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Create a parametric variant.";
  let description = [{
    The `kgen.variant.create` operation creates a variant of the referred type
    with a value provided for one of its possible types.

    Example:

    ```mlir
    // Create an `std::optional<T>` variant.
    %none = kgen.struct.create() : !kgen.struct<()>
    %0 = kgen.variant.create %none, 0 : !kgen.struct<()>
        -> !kgen.variant<struct<()>, T>

    // Create a variant of either a scalar float or integer.
    %1 = kgen.param.constant: scalar<f64> = <<"0.0">>
    %2 = kgen.variant.create %1, 1 : !kgen.scalar<f64>
        -> !kgen.variant<scalar<i64>, scalar<f64>>
    ```
  }];

  let arguments = (ins AnyType:$operand, UnsignedIndexAttr:$index);
  let results = (outs KGEN_VariantType:$result);
  let assemblyFormat = [{
    $operand `,` $index attr-dict `:` type($result)
    custom<VariantElementType>(type($operand), ref(type($result)), ref($index))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

def KGEN_VariantIsOp : KGENOp<"variant.is", [Pure]> {
  let summary = "Check the field contained by a variant.";
  let description = [{
    The `kgen.variant.is` operation checks whether the given variant contains
    a particular type. Returns a `!kgen.scalar<bool>` that indicates whether the
    variant is the particular type.

    Example:

    ```mlir
    %optional = ...
    %isNone = kgen.variant.is %optional, 0 : !kgen.variant<none, T>
    ```
  }];

  let arguments = (ins KGEN_VariantType:$variant, UnsignedIndexAttr:$index);
  let results = (outs BoolScalar:$result);
  let assemblyFormat = "$variant `,` $index attr-dict `:` type($variant)";

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

def KGEN_VariantGetOp : KGENOp<"variant.get", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    InferTypeOpAdaptor]> {
  let summary = "Take a register value out of a variant.";
  let description = [{
    The `kgen.variant.get` operation interprets the given variant as one of its
    possible types and gets it as that type. This operation does NOT check
    whether the variant is that type. If the result type does not match the
    actual type of the variant, the operation's result is a poison value.  From
    an ownership perspective, this consumes the input variant and returns an
    owned register value.

    Example:

    ```mlir
    %optional = ...
    %intVal = kgen.variant.get %optional, 1 : !kgen.variant<struct<()>, i32>
    ```
  }];

  let arguments = (ins KGEN_VariantType:$variant, UnsignedIndexAttr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$variant `,` $index attr-dict `:` type($variant)";

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CompileOffloadOp
//===----------------------------------------------------------------------===//
def KGEN_CompileOffloadOp: KGENOp<"compile_offload">  {
   let summary = "Compile functions to a different offload target from host.";
   let description = [{
     The `kgen.compile_offload` operation indicates compilation to a
     heterogenous target.

     `target_type` is target the offload function is compiled to.

     `emission_kind` is the output type for compile this offload function,
     i.e. asm, shared object, etc.

     `emission_option` is for extra compilation options for compiling
     this offload function.

     `emission_link_option` is for extra options passed through to the
     linker for compiling this offload function. The string will be passed
     as-is to the linker if a linker is involved in compiling the offload.

     `func` is the offload function.

     `kernelID` is an integer number to identify this op from compiled results
     where multiple functions of the same target are bundled together for
     compilation.

     Example:

     ```mlir
     %0 = kgen.compile_offload<target, 0, "", : ()->() @kernel> : !kgen.none
     ```
   }];

   let arguments = (ins AttrOfType<KGEN_TargetType>:$target_type,
                        AttrOfType<Index>:$emission_kind,
                        AttrOfType<KGEN_StringType>:$emission_option,
                        AttrOfType<KGEN_StringType>:$emission_link_option,
                        AttrOfType<AnyType>:$func,
                        OptionalAttr<AttrOfType<Index>>:$kernelID);

   let results = (outs AnyType:$result);

   let assemblyFormat = [{
    `<` custom<CompileOffloadOp>($target_type,
                                 $emission_kind,
                                 $emission_option,
                                 $emission_link_option,
                                 $func) attr-dict `>` `:` type($result)
   }];



}

//===----------------------------------------------------------------------===//
// IsRunInComptimeInterpreterOp
//===----------------------------------------------------------------------===//

def KGEN_IsRunInComptimeInterpreterOp: KGENOp<"is_run_in_comptime_interpreter",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Evaluated as true when running in the comptime interpreter and false otherwise.";
  let description = [{
    The `kgen.is_run_in_comptime_interpreter` represents a boolean value which
    is `true` when running in the comptime interpreter and `false` otherwise.
    When used as condition for control flow, for example,
    only the `true` branch will be evaluated during compile
    time, while the other branch will be compiled to generated code.
    This helps to have efficient compile time (interpreter) for generic
    program without loss of runtime generated code efficiency.

    Example:

    ```mlir
      kgen.is_run_in_comptime_interpreter : !kgen.scalar<bool>
    ```
  }];

  let results = (outs BoolScalar:$result);

  let assemblyFormat = [{
      attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// DeferredOp
//===----------------------------------------------------------------------===//

def KGEN_DeferredOp: KGENOp<"deferred"> {
  let summary = "Wraps an operations that will be constructed later";
  let description = [{
    The `kgen.deferred` is used to encapsulate operation that does have at least
    one !kgen.deferred typed attribute. That is, when operation cannot be
    constructed by the parser as it has non-typed attributes that require
    elaboration.
    It's expected that elaborator will replace `kgen.deferred` with the
    operation and attributes it holds.

    Example:

    ```mlir
    kgen.deferred "index.cmp"(%a, %b : !Int, !Int) {
      %pred = #kgen<deferred #index<cmp_predicate sle>> : !kgen.deferred } : i1

    kgen.deferred "index.cmp"(%a, %b : !Int, !Int) {
      pred = #kgen.param.expr<apply,
       #kgen.bind_params<:!lit.generator<<"cmp": !Bool>()
         -> !kgen.deferred> *"select_pred[::Bool]()", cmp> :
          !kgen.generator<!lit.generator<()
            -> !kgen.deferred>>> : !kgen.deferred} : i1
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    StrAttr:$opName,
    DictionaryAttr:$opAttrs,
    OptionalAttr<AnyAttr>:$opProperties
  );

  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $opName (`(` $operands^ `:` type($operands) `)`)? $opAttrs attr-dict
      (`:` type($results)^)? (`properties` qualified($opProperties)^)?
  }];
}

//===----------------------------------------------------------------------===//
// CodeGenReachable
//===----------------------------------------------------------------------===//

def KGEN_CodeGenReachableOp: KGENOp<"codegen.reachable",
  [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "check if codegen into runtime code is allowed";
  let description = [{
    The `kgen.codegen.reachable` operation checks if
    codegen into runtimne code is allowed or not.
  }];

  let arguments = (ins AttrOfType<I1>:$cond,
                       AttrOfType<KGEN_StringType>:$message);
  let assemblyFormat = [{
    ` ` `<` custom<ParamValue>($cond, "$_builder.getI1Type()") `>` `,`
    custom<StringParam>($message) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

#endif // KGEN_KGENDIALECT_KGENOPS_TD
