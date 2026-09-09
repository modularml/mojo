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

#ifndef SUPPORT_COMPILER_BYTECODEREADERWRITER_H
#define SUPPORT_COMPILER_BYTECODEREADERWRITER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace M {
/// Read a single operation from the given bytecode file. Returns nullptr in the
/// case of failure.
OwningOpRef<Operation *>
readOpFromBytecodeFile(llvm::MemoryBufferRef buffer,
                       const mlir::ParserConfig &config);
template <typename T>
OwningOpRef<T> readOpFromBytecodeFile(llvm::MemoryBufferRef buffer,
                                      const mlir::ParserConfig &config) {
  OwningOpRef<Operation *> rawOp = readOpFromBytecodeFile(buffer, config);
  if (OwningOpRef<T> op = dyn_cast_if_present<T>(*rawOp)) {
    rawOp.release();
    return std::move(op);
  }
  return OwningOpRef<T>();
}

/// Read a single operation from the given bytecode blob. Returns nullptr in the
/// case of failure.
OwningOpRef<Operation *>
readOpFromBytecodeFile(DenseResourceElementsAttr bytecodeAttr,
                       const mlir::ParserConfig &config);
template <typename T>
OwningOpRef<T> readOpFromBytecodeFile(DenseResourceElementsAttr bytecodeAttr,
                                      const mlir::ParserConfig &config) {
  OwningOpRef<Operation *> rawOp = readOpFromBytecodeFile(bytecodeAttr, config);
  if (OwningOpRef<T> op = dyn_cast_if_present<T>(*rawOp)) {
    rawOp.release();
    return std::move(op);
  }
  return OwningOpRef<T>();
}
template <typename T>
OwningOpRef<T> readOpFromBytecodeFile(DenseResourceElementsAttr bytecodeAttr) {
  return readOpFromBytecodeFile<T>(bytecodeAttr, bytecodeAttr.getContext());
}

/// Write a single attribute to a bytecode file.
LogicalResult writeAttrToBytecodeFile(Attribute attr, raw_ostream &os);

/// Read a single attribute from a bytecode file.
Attribute readAttrFromBytecodeFile(llvm::MemoryBufferRef buffer,
                                   MLIRContext *ctx);
template <typename T>
T readAttrFromBytecodeFile(llvm::MemoryBufferRef buffer, MLIRContext *ctx) {
  return dyn_cast_if_present<T>(readAttrFromBytecodeFile(buffer, ctx));
}

/// Recursively and lazily read dependencies from the module contained by
/// `bytecodeSymTab` into `symTab`, rooted at `op`.
LogicalResult loadSymbolsFromBytecode(Operation *op,
                                      mlir::BytecodeReader &reader,
                                      SymbolTable &symTab,
                                      const SymbolTable &bytecodeSymTab);
/// Recursively and lazily read dependencies from the module contained by
/// `bytecodeSymTab` into `symTab`, rooted at `op`. This version abstracts the
/// insertion and lookup of symbols from the mutable symbol table.
LogicalResult
loadSymbolsFromBytecode(Operation *op, mlir::BytecodeReader &reader,
                        function_ref<bool(StringAttr)> existsFn,
                        function_ref<void(Operation *, Operation *)> insertFn,
                        const SymbolTable &bytecodeSymTab);

} // namespace M

#endif // SUPPORT_COMPILER_BYTECODEREADERWRITER_H
