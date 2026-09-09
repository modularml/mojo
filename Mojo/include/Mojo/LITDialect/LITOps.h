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
// This file declares the operation classes for the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_LITOPS_H
#define KGEN_KGENDIALECT_LITOPS_H

#include "Mojo/CODialect/COTypes.h"
#include "Mojo/HLCFDialect/HLCFAttrs.h"
#include "Mojo/HLCFDialect/HLCFInterfaces.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/LITDialect/LITInterfaces.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M::KGEN {
class NoneType;
class PointerType;
class ReturnOp;

namespace LIT {
enum class SpecialFunctionKind : uint8_t;
class SpecialFunctionInfo;

/// Given an insertion point in a block, scan up the parent hierarchy to see if
/// this block is nested under the TryOp region that will handle a 'raise'd
/// error, or if this is in a function that is allowed to raise.  This returns
/// the TryOp or FuncOp if found, or null if raise is not valid.
Operation *findOpProcessingRaise(Block *currentBlock);

/// Given a call or indirect call, return the callee signature type.
FnTypeGeneratorType getCalleeType(Operation *op);
/// Given a call or indirect call, return the callee argument values.
ValueRange getCalleeArguments(Operation *op);

/// Return the fully resolved symbol reference for the given declaration,
/// including all scoping that may be needed, making it unique for every
/// declaration.
SymbolRefAttr getFullyResolvedSymbolRef(mlir::SymbolOpInterface op);

/// Get the full signature of a declaration in the given context.
FnTypeGeneratorType getFullSignature(Operation *container,
                                     FnTypeGeneratorType signature);

} // namespace LIT
} // namespace M::KGEN

namespace M::DebugInfo {
class DIFileAttr;
} // namespace M::DebugInfo

#define GET_OP_CLASSES
#include "Mojo/LITDialect/LIT.h.inc"

#endif // KGEN_KGENDIALECT_LITOPS_H
