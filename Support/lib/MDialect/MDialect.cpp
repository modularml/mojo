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

#include "Support/MDialect/MDialect.h"
#include "Support/AlignedAlloc.h"
#include "Support/Compiler/Bytecode.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

using namespace M;

//===----------------------------------------------------------------------===//
// MDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;

static LogicalResult parseTriple(DialectBytecodeReader &reader,
                                 llvm::Triple &triple) {
  StringRef tripleStr;
  if (failed(reader.readString(tripleStr)))
    return failure();
  triple = llvm::Triple(tripleStr);
  return success();
}

static void printTriple(MLIRContext *ctx, DialectBytecodeWriter &writer,
                        const llvm::Triple &triple) {
  writer.writeOwnedString(StringAttr::get(ctx, triple.str()));
}

static LogicalResult readDataLayout(DialectBytecodeReader &reader,
                                    DataLayout &dl) {
  StringRef dlStr;
  if (failed(reader.readString(dlStr)))
    return failure();
  ErrorOr<DataLayout> dlOr = DataLayout::parse(dlStr);
  if (dlOr.isError())
    return reader.emitError(dlOr.getError());
  dl = dlOr.takeValue();
  return success();
}

static void writeDataLayout(DialectBytecodeWriter &writer,
                            const DataLayout &dl) {
  writer.writeOwnedString(dl.toString());
}

static LogicalResult parseRelocModel(DialectBytecodeReader &reader,
                                     llvm::Reloc::Model &model) {
  StringRef modelStr;
  if (failed(reader.readString(modelStr)))
    return failure();

  ErrorOr<llvm::Reloc::Model> result = symbolizeRelocationModel(modelStr);
  if (result.isError())
    return failure();

  model = *result;
  return success();
}

static void printRelocModel(MLIRContext *ctx, DialectBytecodeWriter &writer,
                            llvm::Reloc::Model model) {
  writer.writeOwnedString(
      StringAttr::get(ctx, stringifyRelocationModel(model)));
}

namespace {
#include "Support/MDialect/MDialectBytecode.cpp.inc"

struct MDialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  MDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// MDialect
//===----------------------------------------------------------------------===//

void MDialect::initialize() {
  registerAttributes();
  registerTypes();
  injectTypeInterfaces();
  injectAttrInterfaces();

  addInterface<MDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.cpp.inc"
