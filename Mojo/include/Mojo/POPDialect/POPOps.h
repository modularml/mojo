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
// This file declares the operation classes for the Meta dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPOPS_H
#define KGEN_POPDIALECT_POPOPS_H

#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/UnifiedFolding.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPEnums.h"
#include "Mojo/POPDialect/POPInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::KGEN {
class DTypeType;
class PointerType;
class StringType;
class StructType;
class ParamListType;
class VariantType;
} // namespace M::KGEN

namespace M::KGEN::POP {
class CmpPredicateAttr;
class AtomicOrderingAttr;
class AtomicBinOpAttr;
class FastmathFlagsAttr;
class PrefetchTagAttr;
class PrefetchLocalityAttr;

class ArrayType;
class UnionType;
} // namespace M::KGEN::POP

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Mojo/POPDialect/POP.h.inc"

#endif // KGEN_POPDIALECT_POPOPS_H
