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

#ifndef KGEN_LIB_KGENTOLLVM_LOWERPOPTOLLVMEXTERNALCALLS_H
#define KGEN_LIB_KGENTOLLVM_LOWERPOPTOLLVMEXTERNALCALLS_H

namespace mlir {
class RewritePatternSet;
class SymbolTable;
} // namespace mlir

namespace M::KGEN {

struct POPToLLVMTypeConverter;

/// Register the pattern that lowers POP external_call ops to LLVM calls,
/// applying platform-specific C ABI struct coercion when necessary.
///
/// The pattern handles struct argument/return coercion for x86-64 System V
/// and ARM64 AAPCS, falling back to pass-through for other targets.
void populateLowerPOPExternalCallPatterns(mlir::RewritePatternSet &patterns,
                                          POPToLLVMTypeConverter &typeConverter,
                                          mlir::SymbolTable &symtab);

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_LOWERPOPTOLLVMEXTERNALCALLS_H
