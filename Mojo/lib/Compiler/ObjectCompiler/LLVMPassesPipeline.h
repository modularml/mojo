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

#ifndef KGEN_LLVMPASSESPIPELINE_H
#define KGEN_LLVMPASSESPIPELINE_H

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace M::KGEN {

class CompilationOptions;
struct MCInfo;

bool addPassesToEmitFile(CompilationOptions &options,
                         llvm::TargetMachine &targetMachine,
                         llvm::legacy::PassManagerBase &pm,
                         llvm::raw_pwrite_stream &out,
                         llvm::raw_pwrite_stream *dwoOut,
                         llvm::CodeGenFileType fileType, bool disableVerify,
                         llvm::MachineModuleInfoWrapperPass *mmiwp);

/// Build a pipeline that does machine specific codegen but stops before
/// AsmPrint.
bool addPassesToEmitMC(CompilationOptions &options,
                       llvm::TargetMachine &targetMachine,
                       llvm::legacy::PassManagerBase &pm,
                       llvm::raw_pwrite_stream &out, bool disableVerify,
                       llvm::MachineModuleInfoWrapperPass *mmiwp,
                       unsigned numFnBase);

/// Build a pipeline that does AsmPrint only.
bool addPassesToAsmPrint(CompilationOptions &options,
                         llvm::TargetMachine &targetMachine,
                         llvm::legacy::PassManagerBase &pm,
                         llvm::raw_pwrite_stream &out,
                         llvm::CodeGenFileType fileType, bool disableVerify,
                         llvm::MachineModuleInfoWrapperPass *mmiwp,
                         llvm::SmallVectorImpl<MCInfo *> &mcInfos);

} // namespace M::KGEN

#endif // KGEN_LLVMPASSESPIPELINE_H
