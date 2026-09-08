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

#ifndef KGEN_KGENDIALECT_KGENINTERFACES_TD
#define KGEN_KGENDIALECT_KGENINTERFACES_TD

include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Parameter Interfaces
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DeclInterface

def DeclInterface : OpInterface<"DeclInterface"> {
  let description = [{
    This interface represents an operation whose region defines a new
    parameter scope. A parameter scope contains a set of operations where a
    particular parameter can be defined at most once and where operations may
    only reference parameters declared in that scope or in a higher.

    Parameter scopes can be isolated from above, meaning that operations nested
    within and below a scope cannot reference parameters defined in higher
    scopes, if any. An operation that defines a parameter scope must be able
    to carry constraints and can optionally define new parameters. Circular
    references are only allowed between a `DeclInterface` and itself, which
    allows, for example, functions to define parameters and then use them in
    their signatures.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Returns whether the declaration scope is parametrically isolated from
      above, meaning that operations at and below the scope cannot reference
      parameters from higher scopes, if any.

      Note that `false` does not mean the scope has references to parameters in
      higher scopes but that it *can* have such references.
    }], "bool", "isIsolatedFromAbove", (ins "unsigned":$regionNumber), [{}],
      [{ return false; }]>,

    InterfaceMethod<[{
      Notify a declaration scope that allows references to parameters in higher
      scopes that it has no such references in its body. An implementing
      operation may store this knowledge in a state for optimizations.

      The region number is passed in because one region may be isolated from
      above while another may not be.
    }], "void", "notifyKnownIsolatedFromAbove", (ins "unsigned":$regionNumber),
      [{}], [{ return; }]>,

    InterfaceMethod<[{
      Get the input parameters to this parameter scope. These parameters are
      visible to the operations at or below the parameter scope defined by this
      operation. The `DeclInterface` is allowed to have a circular reference to
      these parameters.
    }], "ArrayRef<ParamDeclAttr>", "getInputParams">,

    InterfaceMethod<[{
      Get all of the parameters to this parameter scope. These parameters are
      visible to the operations at or below the parameter scope defined by this
      operation. The `DeclInterface` is allowed to have a circular reference to
      these parameters.
    }], "SmallVector<ParamDeclAttr>", "getAllParams",
    // (ins) - Specifies input parameters to the method. Empty parentheses
    // indicate no input parameters. If parameters were needed, they would
    // be listed like: (ins "Type":$paramName)
    (ins),
    // [{}] - Additional properties or constraints for the method. Empty braces
    // indicate no special properties. Could contain things like verification
    // requirements or other interface-specific properties.
    [{}],
    // This is the default implementation of the method. Individual types
    // derived from this interface can override this method to provide a
    // custom implementation.
    [{ return SmallVector<ParamDeclAttr>($_op.getInputParams()); }]>,
  ];
}

//===----------------------------------------------------------------------===//
// StructDeclInterface

def StructDeclInterface : OpInterface<"StructDeclInterface"> {
  let description = [{
    Interface for struct declaration operations. Provides common field access
    methods for both KGEN's StructGeneratorOp and LIT's StructDeclOp.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<
      "Find a field's index by name. Returns std::nullopt if not found.",
      "std::optional<size_t>", "findFieldIndex", (ins "StringRef":$name)
    >,
    InterfaceMethod<
      "Get a field's type by name. Returns nullptr if not found.",
      "TypedAttr", "getFieldType", (ins "StringRef":$name, "Type": $metaType)
    >,
    InterfaceMethod<
      "Get all field names.",
      "void", "getFieldNames",
      (ins "SmallVectorImpl<StringAttr> &":$names)
    >,
    InterfaceMethod<
      "Get all field types.",
      "void", "getFieldTypes",
      (ins "SmallVectorImpl<TypedAttr> &":$types, "Type": $metaType)
    >,
    InterfaceMethod<
      "Get the input parameter declarations for this struct.",
      "ArrayRef<ParamDeclAttr>", "getInputParams"
    >,
    InterfaceMethod< [{
      Lookup a conformance by trait symbol, when returning non-null, the type
      of the returned Operation must be a ConformanceOp, we declare it as
      `Operation *` here due to build dependencies issues.

      The evaluator provides parameter values of the struct that conforms to
      the querying trait.
    }], "Operation *", "lookupConformance", (ins "ParameterEvaluator&":$evaluator, "TraitSymbolAttr":$traitSymbol),
    [{}], [{
      return impl::lookupConformance($_op, evaluator, traitSymbol);
    }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// FuncInterface

def FuncInterface : OpInterface<"FuncInterface"> {
  let description = [{
    This interface represents an operation that defines a KGEN function. A KGEN
    function has a `FuncTypeGeneratorType`, meaning that it can accept
    arguments and input parameters and produce results and output parameters.
    Consequently, operations that implement `FuncInterface` must also implement
    `DeclInterface`.

    This interface also provides methods for accessing the signature metadata.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the function's FuncType generator type.
    }], "FuncTypeGeneratorType", "getFuncTypeGenerator", (ins)>,
    InterfaceMethod<[{
      Return the function's body argument and result types.
    }], "FunctionType", "getFunctionType", (ins)>,
    InterfaceMethod<[{
      Return whether the function should be always_inline'd, and whether it
      should maintain debug info when inlined.
    }], "InlineLevel", "getInlineLevel", (ins)>,
    InterfaceMethod<[{
      Return the decorators attached to this function.
    }], "ArrayRef<TypedAttr>", "getDecorators", (ins), [{}], [{ return {}; }]>,
  ];

  let extraSharedClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() {
      return $_op.getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() {
      return $_op.getFunctionType().getResults();
    }

    /// Return true if the function throws.
    bool isThrows() {
      return $_op.getFuncTypeGenerator().getBody().getFnEffects().isThrows();
    }
    /// Return true if the function is async.
    bool isAsync() {
      return $_op.getFuncTypeGenerator().getBody().getFnEffects().isAsync();
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Implementation
    //===------------------------------------------------------------------===//

    TypeAttr getFunctionTypeAttr() {
      return TypeAttr::get($_op.getFunctionType());
    }
    void setFunctionTypeAttr(TypeAttr type) {
      llvm_unreachable("can't opaquely set the function type");
    }
  }];

  let dependentTraits = [DeclInterface];
}


//===----------------------------------------------------------------------===//
// GeneratorOpInterface

def KGEN_GeneratorOpInterface : OpInterface<"GeneratorOpInterface",
    [DeclInterface, Symbol]> {
  let description = [{
    A parameterized symbol that can be multi-versioned by the elaborator based
    on its actual uses.
  }];
  let cppNamespace = "::M::KGEN";
  let methods = [
    InterfaceMethod<[{
      Return the body region of this generator.
    }], "Region &", "getBodyRegion", (ins)>,
  ];
}

//===----------------------------------------------------------------------===//
// InstantiatedOpInterface

def KGEN_InstantiatedOpInterface : OpInterface<"InstantiatedOpInterface",
    [Symbol]> {
  let description = [{
    A concrete symbol instantiated from a GeneratorOpInterface op.
  }];
  let cppNamespace = "::M::KGEN";
  let methods = [
    InterfaceMethod<[{
      Return the body region of this generator.
    }], "Region &", "getBodyRegion", (ins)>,
  ];
}

//===----------------------------------------------------------------------===//
// GeneratorUserOpInterface

def KGEN_GeneratorUserOpInterface : OpInterface<"GeneratorUserOpInterface"> {
  let description = [{
    This interface represents a reference to a parametric KGEN declaration:
    either a function or generator. The callee is resolved by the KGEN
    elaborator, and operations that implement this interface indicate that they
    can be multi-versioned if the referenced declaration is multi-versioned.

    Generator references must also declare result parameters that are bound to
    the resolved result parameters of the referenced declaration.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Get the parameter expression of the operation's callee.
    }], "TypedAttr", "getCallee", (ins)>,
    InterfaceMethod<[{
      Returns whether this call flavour is allowed to exist post-elaboration as
      a child of a `kgen.func` operation.
    }], "bool", "isAllowedInFunc", (ins), [{}], [{ return true; }]>,
    InterfaceMethod<[{
      This method plugs into the elaborator. It is invoked on the operation when
      its callee has been concretized into a symbol constant. The method should
      use the builder to create a new operation that invokes the callee.
    }], "void", "concretizeCallee", (ins "IRRewriter &":$b,
                                         "SymbolConstantAttr":$callee), [{}], [{
      $_op.setCalleeAttr(callee);
    }]>
  ];

  let extraSharedClassDeclaration = [{
    /// Get the signature type of the callee, if it is known.
    FuncTypeGeneratorType getCalleeSignature() {
      return dyn_cast<FuncTypeGeneratorType>($_op.getCallee().getType());
    }
  }];

  let verify = [{
    return impl::verifyGeneratorUser(cast<GeneratorUserOpInterface>($_op));
  }];
}

//===----------------------------------------------------------------------===//
// ParamOpInterface

def ParamOpInterface : OpInterface<"ParamOpInterface"> {
  let description = [{
    This interface specifies how an operation can declare and define parameters.
    Operations that implement this interface can declare and define parameters
    at their current scope.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Walk the parameters declared by this operation.
    }], "void", "walkDeclarations",
      (ins "function_ref<void(ParamDeclAttr)>":$walkDecl)>,
    InterfaceMethod<[{
      Walk the parameters defined by this operation and get an indication as to
      how they are defined: what expressions and regions the definitions depend
      on. These dependencies are used to build the parameter use-def graph.
    }], "void", "walkDefinitions",
      (ins "function_ref<void(ParamDeclAttr, const ParamDefValue &)>":$walkDef)>,
    InterfaceMethod<[{
      Rename the parameters being declared by this operation.
    }], "void", "renameDeclarations", (ins "ArrayRef<ParamDeclAttr>":$decls)>,
    InterfaceMethod<[{
      Some operations are implicitly parametric even when they contain no
      parameter declarations, definitions, references, or unresolved parameter
      expressions. This is often because the operation itself contains an
      implicit parameter operator in how it behaves.

      Two operations that are implicitly parameter are `kgen.call` and
      `kgen.param.if`.

      ```mlir
      kgen.call @foobar() : () -> ()
      ```

      Even though this operation uses and declares no parameters, it contains
      an implicit parameter operator: `concretize(@foobar)`, which resolves the
      symbol reference `@foobar` to a single concrete function.

      ```mlir
      kgen.param.if <1> {
        // ...
      } else {
        // ...
      }
      ```

      Likewise, this operation uses and declares no parameters, but it is
      implicitly parametric because it makes the body of the surrounding
      function parametric.
    }], "bool", "isImplicitlyParametric", (ins), [{}], [{ return false; }]>,
    InterfaceMethod<[{
      Scan the operation's types and attributes for parameter uses or unresolved
      parameter operators. By default, all of the operation's attributes are
      scanned and its operand, result, and block argument types are scanned.
      Operations can override this method to scan a subset of attributes and
      types if, for example, not all attributes are using parameters within the
      scope that contains the operation.
    }], "void", "collectParameterUses",
      (ins "function_ref<void(Attribute)>":$scanAttr,
           "function_ref<void(Type)>":$scanType), [{}], [{
      impl::scanAllAttrsAndTypes($_op, scanAttr, scanType);
    }]>,
    InterfaceMethod<[{
      For operations that can exist at the boundary of two parameter scopes,
      define an additional interface method that allows these operations to
      distinguish how parameter uses are collected depending on which scope is
      being scanned: the scope that contains the operation or the scope defined
      by the operation.

      Example:

      ```mlir
      kgen.param.declare.region Fn = <A>() -> ()
          constraints <[eq(A, 1), "A == 1"]> {
      }
      ```

      Here, the expression `eq(A, 1)` contains a use of the parameter `A` which
      is declared inside the scope defined by the `kgen.param.declare.region`.
      This use must be distinguished from the uses of parameters by the
      operation at the scope that contains the operation (i.e. the scope in
      which `Fn` is declared).
    }], "void", "collectParameterUsesBelow",
      (ins "function_ref<void(Attribute)>":$scanAttr,
           "function_ref<void(Type)>":$scanType), [{}], [{
      assert(isa<DeclInterface>(*$_op) && "should only be called on decl ops");
      impl::scanAllAttrsAndTypes($_op, scanAttr, scanType);
    }]>
  ];
}

class OneParamDecl {
  code extraDefinition = [{}];
  code extraClassDefinition = extraDefinition # [{
    void $cppClass::walkDeclarations(
        function_ref<void(ParamDeclAttr)> walkDecl) {
      walkDecl(getParamDecl());
    }
    void $cppClass::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
      assert(decls.size() == 1);
      setParamDeclAttr(decls.front());
    }
  }];
}

class NoParamDecls {
  code extraDefinition = [{}];
  code extraClassDefinition = extraDefinition # [{
    void $cppClass::walkDeclarations(
        function_ref<void(ParamDeclAttr)> walkDecl) {}
    void $cppClass::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
      assert(decls.empty());
    }
  }];
}

class CallLikeOperation {
  code extraDefinition = [{}];
  code extraClassDefinition = extraDefinition # [{
    void $cppClass::walkDeclarations(
        function_ref<void(ParamDeclAttr)> walkDecl) {}
    void $cppClass::walkDefinitions(
        function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}
    void $cppClass::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {}
    bool $cppClass::isImplicitlyParametric() { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// Compilation Model
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ExportInterface

def ExportInterface : OpInterface<"ExportInterface"> {
  let description = [{
    This interface allows a symbol-defining operation to specify its export
    semantics via a `#kgen.export_kind` attribute.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the op's (current) symbol name.
    }], "StringAttr", "getSymNameAttr", (ins), [{}], [{
      return $_op.getSymNameAttr();
    }]>,
    InterfaceMethod<[{
      Return the op's linkage name as an attribute. It may not yet be resolved
      if it is an expression.
    }], "TypedAttr", "getLinkageNameAttr", (ins), [{}], [{
      return $_op.getSymNameAttr();
    }]>,
    InterfaceMethod<[{
      Return the op's linkage name as a StringAttr. If the linkage name is
      unresolved, return a null attribute.
    }], "StringAttr", "getLinkageName", (ins), [{
      if (auto lna = dyn_cast_if_present<LinkageNameAttr>($_op.getLinkageNameAttr()))
        return dyn_cast_or_null<StringAttr>(lna.getName());
      return $_op.getSymNameAttr();
    }]>,
    InterfaceMethod<[{
      Return the export kind.
    }], "ExportKind", "getExportKind", (ins), [{}]>,
    InterfaceMethod<[{
      Set the export kind.
    }], "void", "setExportKind", (ins "ExportKind":$exportKind), [{}]>
  ];

  let dependentTraits = [Symbol];

  let extraSharedClassDeclaration = [{
    bool isExported() { return $_op.getExportKind() != ExportKind::NotExported; }

    void setNotExported() { $_op.setExportKind(ExportKind::NotExported); }
    void setExported() { $_op.setExportKind(ExportKind::Exported); }
  }];
}

//===----------------------------------------------------------------------===//
// Control Flow
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FunctionLike

def FunctionLike : OpInterface<"FunctionLike", [DebugInfo_SubprogramScoped]> {
  let description = [{
    The `FunctionLike` interface represents an operation whose regions behave
    like functions in terms of control-flow:

    - the region executes with a stack,
    - region arguments are passed into the region through the stack, and
    - region results are returned via the stack.

    This interface specifies the required result types of the region, allowing
    it to be terminated by a `kgen.return` or `kgen.unreachable`, if the
    function never returns.
  }];

  let cppNamespace = "::M::KGEN";
  let methods = [
    InterfaceMethod<[{
      Get the expected result types of the region.
    }], "ArrayRef<Type>", "getResultTypes", (ins)>,
  ];

  let verify = [{ return HLCF::verifyControlFlow(cast<FunctionLike>($_op)); }];
}

//===----------------------------------------------------------------------===//
// KGENCallOpInterface

def KGEN_CallOpInterface : OpInterface<"KGENCallOpInterface"> {
  let description = [{
    This interface represents a direct function call to a signature-typed
    callee, transferring the call arguments via some memory mechanism: stack,
    frame, or something else. The callee can be parametric, but the call itself
    is always direct.  Call operations that implement this interface could
    conceptually be inlined, unlike indirect calls.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Get the parameter expression of the operation's callee.
    }], "TypedAttr", "getCallee", (ins)>,
    InterfaceMethod<[{
    }], "void", "setCalleeAttr", (ins "::mlir::TypedAttr":$callee)>,
    InterfaceMethod<[{
      Get the call arguments.
    }], "ValueRange", "getArguments", (ins), [{}], [{
      return $_op.getOperands();
    }]>,
    InterfaceMethod<[{
      This interface method is used to inline the call operation. It should use
      the provided builder to create a "scope" operation. The scope operation
      will contained the inlined region of the callee. It should also return a
      callback used to process each `ReturnOp` of the callee.
    }], "FailureOr<InlineResult>", "prepInline",
    (ins "mlir::RewriterBase &":$b)>,
  ];

  let extraSharedClassDeclaration = [{
    /// Get the signature type of the callee.
    FuncTypeGeneratorType getCalleeType() {
      return cast<FuncTypeGeneratorType>($_op.getCallee().getType());
    }

    /// Get the symbol backing the callee.
    SymbolRefAttr getCalleeSymbol() {
      if (auto callee = dyn_cast<SymbolConstantAttr>($_op.getCallee()))
        return callee.getSymbol();
      return {};
    }
  }];
}

//===----------------------------------------------------------------------===//
// ComputeOpInterface
//===----------------------------------------------------------------------===//

def ComputeOpInterface : OpInterface<"ComputeOpInterface"> {
  let description = [{
    This interface defines an API to query the properties of compute operations.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Returns the kind of the compute operations.
    }], "ComputeKind", "getComputeKind">
  ];
}

class Addition {
  code extraDefinition = [{}];
  code extraClassDefinition = extraDefinition # [{
    ComputeKind $cppClass::getComputeKind() {
      return ComputeKind::Addition;
    }
  }];
}

class Comparison {
  code extraDefinition = [{}];
  code extraClassDefinition = extraDefinition # [{
    ComputeKind $cppClass::getComputeKind() {
      return ComputeKind::Comparison;
    }
  }];
}

class Division {
  code extraDefinition = [{}];
  code extraClassDefinition = extraDefinition # [{
    ComputeKind $cppClass::getComputeKind() {
      return ComputeKind::Division;
    }
  }];
}

class Multiplication {
  code extraDefinition = [{}];
  code extraClassDefinition = extraDefinition # [{
    ComputeKind $cppClass::getComputeKind() {
      return ComputeKind::Multiplication;
    }
  }];
}

class MultiplyAdd {
  code extraDefinition = [{}];
  code extraClassDefinition = extraDefinition # [{
    ComputeKind $cppClass::getComputeKind() {
      return ComputeKind::MultiplyAdd;
    }
  }];
}

//===----------------------------------------------------------------------===//
// UnifiedFolding Trait
//===----------------------------------------------------------------------===//

/// Unifies fold, interpret, and parametric_interpret into a single
/// target-aware fold function.  The C++ trait template
/// `detail::UnifiedFoldingOpInterfaceTrait<ConcreteOp>` (defined in
/// UnifiedFolding.h) provides default `interpret`, `parametric_interpret`,
/// and `foldTrait` implementations that delegate to a single `unifiedFold`
/// method.
///
/// Usage:
///   - List `UnifiedFolding` and `InterpreterOpInterface` in the op's traits.
///   - Set `let hasFolder = 0;` (fold is provided via `foldTrait`).
///   - Implement `unifiedFold` in the .cpp file.
///   - To override a defaulted hook (e.g. `parametric_interpret`), add a
///     declaration in the op's `extraClassDeclaration` — C++ name lookup will
///     shadow the trait's default.
def UnifiedFolding : NativeOpTrait<
    "UnifiedFoldingOpInterfaceTrait",
    /*traits=*/[],
    /*extraOpDeclaration=*/[{
  ::M::KGEN::FoldValue unifiedFold(::M::KGEN::FoldValues operands,
                                   ::M::TargetInfoAttr target);
}]> {
  let cppNamespace = "::M::detail";
}

#endif // KGEN_KGENDIALECT_KGENINTERFACES_TD
