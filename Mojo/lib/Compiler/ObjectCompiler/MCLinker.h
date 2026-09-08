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

#ifndef KGEN_COMPILER_OBJECTCOMPILER_MCLINK_H
#define KGEN_COMPILER_OBJECTCOMPILER_MCLINK_H

#include "Mojo/Compiler/LLVMIRUtils.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Buffer.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace M::KGEN {
/// This file defines data structures to help linking LLVM modules
/// at MC level (right after codegen) and AsmPrint into one .o or .s file.
/// This linking is needed because we parallelize the llvm opt and
/// llc pipelines by splitting LLVMModule into multiple splits
/// with symbol linkage changes.
/// Linking at MC level helps to fix the temporary symbol linkage change,
/// deduplicate multiple symbols among the splits.
/// This allows mojo compilation to produce 1 .o file for each program
/// (instead of one .a file with multiple .o files in .a) with reduced
/// object file size (due to symbol dedup and linkage restoration), and
/// allows NVPTX compilation to share the same parallelized emitArchive
/// code path to produce an assembly output.
struct MCInfo {
  MCInfo(WriteableBufferRef moduleBuf,
         std::unique_ptr<llvm::MachineModuleInfo> &&machineModuleInfo,
         LLVMModuleAndContext &&moduleAndContext,
         llvm::StringMap<const llvm::Function *> &fnNameToFnPtr,
         std::unique_ptr<llvm::TargetMachine> &&targetMachine,
         std::unique_ptr<llvm::MCContext> &&mcContext,
         std::optional<int> splitIdx)
      : moduleBuf(std::move(moduleBuf)),
        moduleAndContext(std::move(moduleAndContext)),
        mcContext(std::move(mcContext)),
        machineModuleInfo(std::move(machineModuleInfo)),
        fnNameToFnPtr(std::move(fnNameToFnPtr)),
        targetMachine(std::move(targetMachine)), splitIdx(splitIdx) {}

  MCInfo(MCInfo &&other)
      : moduleBuf(std::move(other.moduleBuf)),
        moduleAndContext(std::move(other.moduleAndContext)),
        mcContext(std::move(other.mcContext)),
        machineModuleInfo(std::move(other.machineModuleInfo)),
        fnNameToFnPtr(std::move(other.fnNameToFnPtr)),
        targetMachine(std::move(other.targetMachine)),
        splitIdx(other.splitIdx) {}

  /// Serialize the llvm::Module in parallel and deserialize back to put into
  /// the same LLVMContext which is required for llvm::Linker.
  WriteableBufferRef moduleBuf;

  /// Keep original module split alive because llvm::Function is kept as
  /// reference in llvm::MachineFunctions and will be used during codegen.
  LLVMModuleAndContext moduleAndContext;

  /// ExternContext to MachineModuleInfo to work around the upstream bug
  /// with the move constructor of MachineModuleInfo.
  std::unique_ptr<llvm::MCContext> mcContext;

  /// This is where all the MachineFunction live that we need for AsmPrint.
  std::unique_ptr<llvm::MachineModuleInfo> machineModuleInfo;

  /// llvm::Function name to llvm::Function* map for concatenating the
  /// MachineFunctions map.
  llvm::StringMap<const llvm::Function *> fnNameToFnPtr;

  /// Keep targetMachine alive.
  std::unique_ptr<llvm::TargetMachine> targetMachine;

  /// parallel llvm module split id, mostly for debugging.
  std::optional<int> splitIdx;

  /// llvm::GlobalVariable name to the renamed llvm::GlobalVariable name, this
  /// is to handle llvm inserted local symbols after splitting the module.
  llvm::StringMap<const std::string> renamedLocalSymbols;
};

struct SymbolAndMCInfo {
  SymbolAndMCInfo() = default;

  SymbolAndMCInfo(SymbolAndMCInfo &&other)
      : symbolLinkageTypes(std::move(other.symbolLinkageTypes)),
        mcInfos(std::move(other.mcInfos)) {}

  /// Clear member variables explicitly.
  void clear();

  /// Book-keeping original symbol linkage type if they are changed due to
  /// splitting for parallel compilation.
  llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes;

  /// Vector of codegen results for each parallel split before AsmPrint.
  SmallVector<std::unique_ptr<MCInfo>> mcInfos;
};

class MCLinker {
public:
  MCLinker(SmallVectorImpl<SymbolAndMCInfo *> &symbolAndMCInfos,
           llvm::TargetMachine &targetMachine, CompilationOptions options,
           llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes,
           llvm::StringMap<unsigned> originalFnOrdering);

  /// Link multiple MC results and AsmPrint into one .o file.
  ErrorOr<WriteableBufferRef> linkAndPrint(StringRef moduleName,
                                           bool emitAssembly);

private:
  SmallVectorImpl<SymbolAndMCInfo *> &symbolAndMCInfos;
  llvm::TargetMachine &targetMachine;
  CompilationOptions options;
  SmallVector<MCInfo *> mcInfos;
  LLVMModuleAndContext linkedModule;

  llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes;
  llvm::StringMap<unsigned> originalFnOrdering;
  llvm::MachineModuleInfoWrapperPass *machineModInfoPass = nullptr;

  /// Link llvm::Modules from each split.
  ErrorOrSuccess linkLLVMModules(StringRef moduleName);

  /// Get llvm::Module and prepare MachineModuleInfoWrapperPass to print if
  /// there is only one split.
  llvm::Module *
  getModuleToPrintOneSplit(llvm::TargetMachine &llvmTargetMachine);

  /// Prepare MachineModuleInfo before AsmPrinting.
  void prepareMachineModuleInfo(llvm::TargetMachine &llvmTargetMachine);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_OBJECTCOMPILER_MCLINK_H
