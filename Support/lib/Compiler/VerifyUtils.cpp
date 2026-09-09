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

#include "Support/Compiler/VerifyUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/TypeUtilities.h"
#include <cstddef>

using namespace M;

LogicalResult M::checkOperandTypes(Operation *op, TypeRange expectedTypes) {
  if (op->getNumOperands() != expectedTypes.size()) {
    return op->emitOpError() << "expected " << expectedTypes.size()
                             << " operands, but given " << op->getNumOperands();
  }

  for (size_t i = 0, e = op->getNumOperands(); i != e; ++i) {
    auto t = op->getOperand(i).getType();
    if (t != expectedTypes[i]) {
      return op->emitOpError() << "operand #" << i << " has type " << t
                               << " but expected " << expectedTypes[i];
    }
  }
  return success();
}

LogicalResult M::checkArgumentTypes(Operation *op, StringRef blockName,
                                    Block *block, TypeRange expectedTypes) {
  if (block->getNumArguments() != expectedTypes.size()) {
    return op->emitOpError()
           << "expected " << expectedTypes.size() << " arguments to '"
           << blockName << "' block, but given " << block->getNumArguments();
  }

  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
    auto t = block->getArgument(i).getType();
    if (t != expectedTypes[i]) {
      return op->emitOpError()
             << "block '" << blockName << "' argument #" << i << " has type "
             << t << " but expected " << expectedTypes[i];
    }
  }
  return success();
}

LogicalResult M::checkMatchingTypes(Operation *op, StringRef context,
                                    TypeRange actualTypes,
                                    TypeRange expectedTypes) {
  if (actualTypes.size() != expectedTypes.size()) {
    return op->emitOpError()
           << "expected " << expectedTypes.size() << " " << context
           << " entries, but given " << actualTypes.size();
  }

  for (size_t i = 0, e = actualTypes.size(); i != e; ++i) {
    if (actualTypes[i] != expectedTypes[i]) {
      return op->emitOpError()
             << "actual " << context << " #" << i << " has type "
             << actualTypes[i] << " but expected " << expectedTypes[i];
    }
  }
  return success();
}
