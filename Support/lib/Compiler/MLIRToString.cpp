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

#include "Support/Compiler/MLIRToString.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

LLVM_DUMP_METHOD std::string M::mlirToString(mlir::Operation *op) {
  // The implementation of debugString(op) prints a pointer, unhelpful.
  std::string outStr;
  llvm::raw_string_ostream out(outStr);
  out << *op;
  return out.str();
}

LLVM_DUMP_METHOD std::string M::mlirToString(mlir::Attribute attr) {
  return debugString(attr);
}

LLVM_DUMP_METHOD std::string M::mlirToString(mlir::Type type) {
  return debugString(type);
}
