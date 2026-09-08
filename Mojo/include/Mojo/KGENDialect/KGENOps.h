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
// This file declares the operation classes for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENOPS_H
#define KGEN_KGENDIALECT_KGENOPS_H

#include "Mojo/HLCFDialect/HLCFInterfaces.h"
#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M {

//===----------------------------------------------------------------------===//
// KGENModule
//===----------------------------------------------------------------------===//

/// A KGEN module wraps a `ModuleOp` and a `SymbolTableCollection` for
/// convenient nested symbol lookups across the module.
class KGENModule {
public:
  /// Create KGEN module with the provided module and symbol table collection.
  KGENModule(ModuleOp module, SymbolTableCollection &symbolTable)
      : module(module), symbolTable(symbolTable) {}

  /// Get the KGEN module from the provided operation.
  static KGENModule from(Operation *op, SymbolTableCollection &symbolTable) {
    return {op->getParentOfType<ModuleOp>(), symbolTable};
  }

  template <typename OpT>
  OpT lookup(SymbolRefAttr symbol) {
    return dyn_cast_or_null<OpT>(symbolTable.lookupSymbolIn(module, symbol));
  }

private:
  /// The top-level IR module.
  ModuleOp module;

  /// A collection of symbol tables.
  SymbolTableCollection &symbolTable;
};

//===----------------------------------------------------------------------===//
// Attribute keys
//===----------------------------------------------------------------------===//

/// Attribute written onto a transparent-thunk `GeneratorOp` to record the
/// parametric callee expression that names the wrapped function.
///
/// A "transparent" thunk forwards to a wrapped function while delegating its
/// public identity (linkage name, LLVM metadata) to that function. The
/// attribute value is a parametric expression — e.g. a `GetWitnessAttr` for
/// closure `__call__` thunks — that resolves to a fully-bound
/// `SymbolConstantAttr` once the thunk's paramDecls are substituted with the
/// callsite's paramValues. When `compile_offload` targets such a thunk the
/// elaborator must:
///   1. Substitute the thunk's paramDecls with the callsite's paramValues.
///   2. Evaluate any parameter operators in the expression (e.g.
///      `get_witness`) against the substituted values.
///   3. Redirect the offload to the resolved callee — surfacing its metadata
///      (e.g. `@__llvm_metadata`) on the emitted entry.
///
/// Resolution runs through `IREvaluatorContext::resolveTransparentThunkCallee`.
static constexpr llvm::StringLiteral kTransparentThunkCalleeExprAttr =
    "kgen.transparent_thunk_callee_expr";

} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Mojo/KGENDialect/KGEN.h.inc"

#endif // KGEN_KGENDIALECT_KGENOPS_H
