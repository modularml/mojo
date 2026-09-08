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
// Rewrites abi("C") LLVM function definitions to use the platform C ABI
// (CABIInfo). Separated from LowerKGENToLLVM.cpp so that reviewers can read
// the definition-side logic in isolation from the call-site patterns.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H
#define KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H

namespace mlir::LLVM {
class LLVMFuncOp;
} // namespace mlir::LLVM

namespace M::KGEN {
class CABIInfo;

/// Rewrite a abi("C") LLVM function definition to use the platform C ABI.
/// Applies entry-block argument coercion (register/indirect/two-register) and
/// return-value coercion (register/sret) in place, then updates the function
/// type signature. No-ops for external declarations and identity-ABI functions.
/// Mojo direct callers are already patched by ConvertKGENCall.
void processCABIFunctionDefinition(mlir::LLVM::LLVMFuncOp func,
                                   CABIInfo &abiInfo);

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H
