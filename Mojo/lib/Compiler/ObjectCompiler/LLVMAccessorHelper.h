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

#ifndef KGEN_COMPILER_OBJECTCOMPILER_LLVMACCESSORHELPER_H
#define KGEN_COMPILER_OBJECTCOMPILER_LLVMACCESSORHELPER_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbolTableEntry.h"
#include "llvm/Target/TargetMachine.h"

namespace M::KGEN {
// A few helper functions to access LLVM private class/struct members:
// http://bloglitb.blogspot.com/2010/07/access-to-private-members-thats-easy.html

/// Wrapping accessing LLVM data structure's private filed accessor for
/// linking at MC-level where a few things need to be globalized such as:
/// - llvm::MachineFunction's numbering,
/// - all unique_ptrs of llvm::MachineFunctions in each split to be put
///   together for the final AsmPrint
/// - MCSymbol propagation for external global symbols to each split's
///   MCContext to avoid duplicates for X86's OrcJIT execution engine.

/// Get private field
/// DenseMap<const Function*, std::unique_ptr<MachineFunction>> MachineFunctions
/// from llvm::MachineModuleInfo.
llvm::DenseMap<const llvm::Function *, std::unique_ptr<llvm::MachineFunction>> &
getMachineFunctionsFromMachineModuleInfo(
    llvm::MachineModuleInfo &machineModuleInfo);

/// Set private field FunctionNumber in llvm::MachineFunction.
void setMachineFunctionNumber(llvm::MachineFunction &mf, unsigned number);

/// Set private field NextFnNum in llvm::MachineModuleInfo.
void setNextFnNum(llvm::MachineModuleInfo &mmi, unsigned value);

/// Call private member function
/// MCSymbolTableEntry &getSymbolTableEntry(StringRef Name)
/// from llvm::MCContext.
llvm::MCSymbolTableEntry &getMCContextSymbolTableEntry(llvm::StringRef name,
                                                       llvm::MCContext &);

/// Release MCSubTargetInfo.
void releaseTargetMachineConstants(llvm::TargetMachine &tm);

/// Clear SubtargetMap in SubtargetInfo.
void resetSubtargetInfo(llvm::TargetMachine &dst, llvm::MachineModuleInfo &mmi);

} // namespace M::KGEN

#endif // KGEN_COMPILER_OBJECTCOMPILER_LLVMACCESSORHELPER_H
