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

#ifndef SUPPORT_COMPILER_MLIRTOSTRING_H
#define SUPPORT_COMPILER_MLIRTOSTRING_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace M {
/// These mlirToString functions are mostly wrappers for MLIR's debugString
/// function, except that they are concrete, and guaranteed to be available in a
/// debugger, so you can get strings programmatically in the debugger.
/// Return the printed representation of the operation as a string.
LLVM_DUMP_METHOD std::string mlirToString(mlir::Operation *op);
/// Return the printed representation of the attribute as a string.
LLVM_DUMP_METHOD std::string mlirToString(mlir::Attribute attr);
/// Return the printed representation of the type as a string.
LLVM_DUMP_METHOD std::string mlirToString(mlir::Type type);

// Despite LLVM_DUMP_METHOD annotations, the above are not kept in the binary
// unless this header is included at some point AND the functions have uses,
// such as the below.
#ifndef NDEBUG
LLVM_DUMP_METHOD static void _keepMlirToStringFunctions() {
  llvm::nulls() << (void *)static_cast<std::string (*)(mlir::Operation *)>(
      M::mlirToString);
  llvm::nulls() << (void *)static_cast<std::string (*)(mlir::Attribute)>(
      M::mlirToString);
  llvm::nulls() << (void *)static_cast<std::string (*)(mlir::Type)>(
      M::mlirToString);
}
#endif
} // namespace M

#endif // SUPPORT_COMPILER_MLIRTOSTRING_H
