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
// This file contains MArchTarget declarations that include modular internal
// dependencies, and will not be included in the static version of
// KGENCompilerRT that links with user mojo object files.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MARCHTARGET_MARCHTARGET_H
#define SUPPORT_MARCHTARGET_MARCHTARGET_H

#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/Support/CodeGen.h"
#include <memory>

namespace llvm {
// Forward declare.
class TargetMachine;
} // namespace llvm

namespace M {

/// Returns a TargetMachine for the current host.
ErrorOr<std::unique_ptr<llvm::TargetMachine>> getTargetMachineForHost(
    bool isJIT = true,
    llvm::CodeGenOptLevel optLevel = llvm::CodeGenOptLevel::Aggressive);

/// As for `getMArchTargetInfo`, but returned as TargetInfoAttr. The `-mtune`
/// flag is captured in the result, and derived information such as for
/// data layout and SIMD width is filled in.
///
/// TODO: Split into separate MLIR-dependent library. All other functions
/// depend only on LLVMTarget and (unfortunately) clang.
ErrorOr<TargetInfoAttr>
getMArchFeatures(MLIRContext *ctx, StringRef targetTriple, StringRef march,
                 StringRef mcpu, StringRef mtune, StringRef acceleratorArch,
                 llvm::Reloc::Model relocModel, StringRef abi = "");

} // namespace M

#endif // SUPPORT_MARCHTARGET_MARCHTARGET_H
