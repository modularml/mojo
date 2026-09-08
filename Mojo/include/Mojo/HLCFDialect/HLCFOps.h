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

#ifndef KGEN_HLCFDIALECT_HLCFOPS_H
#define KGEN_HLCFDIALECT_HLCFOPS_H

#include "Mojo/HLCFDialect/HLCFInterfaces.h"
#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFAttrs.h"

#define GET_OP_CLASSES
#include "Mojo/HLCFDialect/HLCF.h.inc"

#endif // KGEN_HLCFDIALECT_HLCFOPS_H
