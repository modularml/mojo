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

#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Buffer.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/xxhash.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace M;

OwningOpRef<Operation *>
M::readOpFromBytecodeFile(llvm::MemoryBufferRef buffer,
                          const mlir::ParserConfig &config) {
  Block b;
  if (failed(mlir::readBytecodeFile(buffer, &b, config)) ||
      !llvm::hasSingleElement(b))
    return nullptr;

  // Take ownership of the op from the block.
  Operation *op = &b.front();
  op->remove();
  return op;
}

OwningOpRef<Operation *>
M::readOpFromBytecodeFile(DenseResourceElementsAttr bytecodeAttr,
                          const mlir::ParserConfig &config) {
  mlir::AsmResourceBlob *blob = bytecodeAttr.getRawHandle().getBlob();
  if (!blob)
    return OwningOpRef<Operation *>();
  ArrayRef<char> bytecode = blob->getData();
  llvm::MemoryBufferRef bufferRef(StringRef(bytecode.begin(), bytecode.size()),
                                  "");

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  mlir::BytecodeReader reader(bufferRef, config, /*lazyLoad=*/false, sourceMgr);
  Block b;
  if (failed(reader.readTopLevel(&b)) || !llvm::hasSingleElement(b))
    return nullptr;

  // Take ownership of the op from the block.
  Operation *op = &b.front();
  op->remove();
  return op;
}

LogicalResult M::writeAttrToBytecodeFile(Attribute attr, raw_ostream &os) {
  // There isn't an easy way to do this other than create a temporary operation
  // and write it out.
  OwningOpRef<ModuleOp> tempBytecodeOp =
      ModuleOp::create(UnknownLoc::get(attr.getContext()));
  (*tempBytecodeOp)->setAttr("bytecode.attr", attr);
  return mlir::writeBytecodeToFile(tempBytecodeOp.get(), os);
}

//===----------------------------------------------------------------------===//
// loadSymbolsFromBytecode
//===----------------------------------------------------------------------===//

LogicalResult M::loadSymbolsFromBytecode(
    Operation *op, mlir::BytecodeReader &reader,
    function_ref<bool(StringAttr)> existsFn,
    function_ref<void(Operation *, Operation *)> insertFn,
    const SymbolTable &bytecodeSymTab) {

  // Process the dependencies using a worklist.
  std::vector<Operation *> worklist;
  worklist.push_back(op);
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();

    if (reader.isMaterializable(op)) {
      if (failed(reader.materialize(op, [&](Operation *op) { return true; })))
        return failure();
    }

    mlir::AttrTypeWalker walker;
    // Extract a dependency from the bytecode module and move it into the main
    // module, if it doesn't already exist there. If a symbol was moved, return
    // it.
    auto extractDependency = [&](StringAttr name) -> Operation * {
      // Don't move the symbol if it already exists in the main module.
      if (existsFn(name))
        return nullptr;
      Operation *symbol = bytecodeSymTab.lookup(name);
      assert(symbol && "expected valid symbol reference");

      // Move the symbol into the main module.
      insertFn(symbol, op);
      return symbol;
    };
    walker.addWalk([&](FlatSymbolRefAttr ref) {
      if (Operation *decl = extractDependency(ref.getAttr()))
        worklist.push_back(decl);
    });
    op->walk([&](Operation *op) {
      // Extract references to type declarations.
      walker.walk(op->getAttrDictionary());
      for (Type type : op->getResultTypes())
        walker.walk(type);
      for (Region &region : op->getRegions()) {
        for (Type type : region.getArgumentTypes())
          walker.walk(type);
      }
    });
  }

  return success();
}

LogicalResult M::loadSymbolsFromBytecode(Operation *op,
                                         mlir::BytecodeReader &reader,
                                         SymbolTable &symTab,
                                         const SymbolTable &bytecodeSymTab) {
  return loadSymbolsFromBytecode(
      op, reader, [&](StringAttr name) -> bool { return symTab.lookup(name); },
      [&](Operation *op, Operation *after) {
        op->moveAfter(after);
        symTab.insert(op);
      },
      bytecodeSymTab);
}
