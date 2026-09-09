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

#ifndef KGEN_INTERPRETER_UTILS_H
#define KGEN_INTERPRETER_UTILS_H

#include "Support/Compiler/ErrorTree.h"
#include "mlir/IR/SymbolTable.h"

namespace M {
/// Add a stack trace to an interpreter error.
template <typename StackFrame, typename GetOriginFnT>
ErrorTree addStackTraceImpl(ErrorTree error, ArrayRef<StackFrame> stack,
                            GetOriginFnT &&getOrigin) {
  for (const StackFrame &frame : llvm::reverse(stack)) {
    StringRef funcName = cast<mlir::SymbolOpInterface>(frame.func).getName();
    error = ErrorTree(frame.func->getLoc(),
                      Error("failed to interpret function @" + funcName),
                      std::move(error));
    if (Operation *origin = getOrigin(frame)) {
      error = ErrorTree(origin->getLoc(), Error("failed to evaluate call"),
                        std::move(error));
    }
  }
  return error;
}

/// Report an error with folding an operation.
inline ErrorTree reportFoldError(Operation *op, ArrayRef<Attribute> operands,
                                 const Twine &prefix,
                                 const Twine &suffix = "") {
  std::string note;
  llvm::raw_string_ostream os(note);
  os << prefix << op->getName();
  if (!op->getAttrs().empty()) {
    os << '{';
    llvm::interleaveComma(op->getAttrs(), os, [&](const NamedAttribute &attr) {
      os << attr.getName().getValue() << ": " << attr.getValue();
    });
    os << '}';
  }
  os << '(';
  llvm::interleaveComma(operands, os);
  os << ')' << suffix;

  // Provide a more helpful hint for inline assembly operations.
  if (op->getName().getStringRef() == "pop.inline_asm") {
    os << "\n\nNote: Inline assembly cannot be evaluated at compile time. "
       << "Inline assembly contains hardware-specific instructions that can "
       << "only execute at runtime on the target hardware.";
  }

  return {op->getLoc(), Error(os.str())};
}

} // namespace M

#endif // KGEN_INTERPRETER_UTILS_H
