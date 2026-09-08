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

#include "Mojo/Interpreter/InterpreterDialect.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Support/Compiler/Bytecode.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;

//===----------------------------------------------------------------------===//
// InterpreterDialectOpAsmDialectInterface
//===----------------------------------------------------------------------===//

namespace {
struct InterpreterDialectOpAsmDialectInterface
    : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<MemoryHandleAttr>(attr)) {
      os << "memory_handle";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// InterpreterDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;
using mlir::readResourceHandle;

static LogicalResult readAlignedBlob(DialectBytecodeReader &reader,
                                     AlignedBlob &blob) {
  if (failed(reader.readVarInt(blob.align)) ||
      failed(reader.readBlob(blob.data)) ||
      failed(reader.readBool(blob.isString)))
    return failure();
  return success();
}

static void writeAlignedBlob(DialectBytecodeWriter &writer, AlignedBlob blob) {
  writer.writeVarInt(blob.align);
  writer.writeOwnedBlob(blob.data);
  writer.writeOwnedBool(blob.isString);
}

static LogicalResult readOptionalIndex(DialectBytecodeReader &reader,
                                       std::optional<int64_t> &index) {
  int64_t value;
  if (failed(reader.readSignedVarInt(value)))
    return failure();
  // -1 encodes an absent index; anything below it is malformed.
  if (value < -1)
    return failure();
  index = value < 0 ? std::nullopt : std::optional<int64_t>(value);
  return success();
}

static void writeOptionalIndex(DialectBytecodeWriter &writer,
                               std::optional<int64_t> index) {
  writer.writeSignedVarInt(index.value_or(-1));
}

static LogicalResult
readPointerRegions(DialectBytecodeReader &reader,
                   SmallVectorImpl<PointerRegion> &regions) {
  auto readPointerRegion = [&](PointerRegion &region) {
    int64_t offset, blobIndex, blobOffset;
    if (failed(reader.readSignedVarInt(offset)) ||
        failed(reader.readSignedVarInt(blobIndex)) ||
        failed(reader.readSignedVarInt(blobOffset)))
      return failure();
    region = PointerRegion{offset, blobIndex, blobOffset};
    return LogicalResult::success();
  };

  if (failed(reader.readList(regions, readPointerRegion)))
    return failure();

  return success();
}

static void writePointerRegions(DialectBytecodeWriter &writer,
                                ArrayRef<PointerRegion> regions) {
  auto writePointerRegion = [&](const PointerRegion &region) {
    writer.writeSignedVarInt(region.offset);
    writer.writeSignedVarInt(region.blobIndex);
    writer.writeSignedVarInt(region.blobOffset);
  };

  writer.writeList(regions, writePointerRegion);
}

static LogicalResult readSymbolRegions(DialectBytecodeReader &reader,
                                       SmallVectorImpl<int64_t> &regions) {
  auto readSymbolRegion = [&](int64_t &offset) {
    if (failed(reader.readSignedVarInt(offset)))
      return failure();
    return LogicalResult::success();
  };

  if (failed(reader.readList(regions, readSymbolRegion)))
    return failure();

  return success();
}

static void writeSymbolRegions(DialectBytecodeWriter &writer,
                               ArrayRef<int64_t> regions) {
  auto writePointerRegion = [&](const int64_t &region) {
    writer.writeSignedVarInt(region);
  };
  writer.writeList(regions, writePointerRegion);
}

namespace {
#include "Mojo/Interpreter/InterpreterDialectBytecode.cpp.inc"

struct InterpreterDialectBytecodeInterface
    : public mlir::BytecodeDialectInterface {
  InterpreterDialectBytecodeInterface(Dialect *dialect)
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
// InterpreterDialect
//===----------------------------------------------------------------------===//

void InterpreterDialect::initialize() {
  registerAttributes();

  addInterfaces<InterpreterDialectOpAsmDialectInterface,
                InterpreterDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Mojo/Interpreter/InterpreterDialect.cpp.inc"
