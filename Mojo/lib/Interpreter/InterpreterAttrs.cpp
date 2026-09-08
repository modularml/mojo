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

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/Interpreter/InterpreterDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;

//===----------------------------------------------------------------------===//
// InterpreterDialect
//===----------------------------------------------------------------------===//

void InterpreterDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Mojo/Interpreter/InterpreterAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MemoryHandleAttr
//===----------------------------------------------------------------------===//

static ParseResult parseAlignedHex(AsmParser &p, OwnedAlignedBlob &blob) {
  // Parse the alignment and then hex data.
  std::string hex;
  llvm::SMLoc loc;
  if (p.parseInteger(blob.align) || p.parseComma() ||
      p.getCurrentLocation(&loc) || p.parseString(&hex))
    return failure();
  blob.isString = succeeded(p.parseOptionalKeyword("string"));
  if (blob.isString) {
    blob.data = std::move(hex);
    return success();
  }
  StringRef str = hex;
  if (!str.consume_front("0x") || !llvm::tryGetFromHex(str, blob.data))
    return p.emitError(loc, "expected a hex string blob");
  return success();
}

static void printAlignedHex(AsmPrinter &p, AlignedBlob blob) {
  p << blob.align << ", \"";
  StringRef str(blob.data.data(), blob.data.size());
  if (blob.isString)
    llvm::printEscapedString(str, p.getStream());
  else
    p << "0x" << llvm::toHex(str);
  p << '"';
  if (blob.isString)
    p << " string";
}

MemoryHandleAttr MemoryHandleAttr::get(MLIRContext *ctx, StringRef str) {
  return get(ctx, AlignedBlob(/*align=*/16, {str.data(), str.size()},
                              /*isString=*/true));
}

namespace M {
static llvm::hash_code hash_value(AlignedBlob blob) {
  return llvm::hash_combine(blob.align, blob.data, blob.isString);
}
} // namespace M

//===----------------------------------------------------------------------===//
// MemoryBlobAttr
//===----------------------------------------------------------------------===//

Attribute MemoryBlobAttr::parse(AsmParser &p, Type type) {
  MemoryHandleAttr hdl;
  if (p.parseLParen() || p.parseAttribute(hdl))
    return {};
  StringRef kindStr;
  SmallVector<PointerRegion> pointerRegions;
  unsigned addressSpace = 0;
  auto parsePointerRegion = [&] {
    PointerRegion &region = pointerRegions.emplace_back();
    return failure(p.parseLParen() || p.parseInteger(region.offset) ||
                   p.parseComma() || p.parseInteger(region.blobIndex) ||
                   p.parseComma() || p.parseInteger(region.blobOffset) ||
                   p.parseRParen());
  };
  SmallVector<int64_t> symbolRegions;
  auto parseSymbolRegion = [&] {
    int64_t &offset = symbolRegions.emplace_back();
    return p.parseInteger(offset);
  };
  if (p.parseComma() || p.parseKeyword(&kindStr) || p.parseComma() ||
      p.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                parsePointerRegion) ||
      p.parseComma() ||
      p.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                parseSymbolRegion))
    return {};
  if (succeeded(p.parseOptionalComma())) {
    if (p.parseInteger(addressSpace))
      return {};
  }
  if (p.parseRParen())
    return {};

  MemoryKind kind = llvm::StringSwitch<MemoryKind>(kindStr)
                        .Case("stack", MemoryKind::Stack)
                        .Case("heap", MemoryKind::Heap)
                        .Case("const_global", MemoryKind::ConstGlobal)
                        .Case("persistent", MemoryKind::Persistent);
  return get(hdl, kind, pointerRegions, symbolRegions, addressSpace);
}

void MemoryBlobAttr::print(AsmPrinter &p) const {
  p << '(' << getHandle() << ", ";
  switch (getKind()) {
  case MemoryKind::Stack:
    p << "stack";
    break;
  case MemoryKind::Heap:
    p << "heap";
    break;
  case MemoryKind::ConstGlobal:
    p << "const_global";
    break;
  case MemoryKind::Persistent:
    p << "persistent";
    break;
  }
  p << ", [";
  llvm::interleaveComma(getPointerRegions(), p,
                        [&](const PointerRegion &region) {
                          p << '(' << region.offset << ", " << region.blobIndex
                            << ", " << region.blobOffset << ')';
                        });
  p << "], [";
  llvm::interleaveComma(getSymbolRegions(), p,
                        [&](int64_t offset) { p << offset; });
  p << "]";
  if (getAddressSpace())
    p << ", " << getAddressSpace();
  p << ')';
}

namespace M {
static llvm::hash_code hash_value(const PointerRegion &region) {
  return llvm::hash_combine(region.offset, region.blobIndex, region.blobOffset);
}
static bool operator==(const PointerRegion &lhs, const PointerRegion &rhs) {
  return std::make_tuple(lhs.offset, lhs.blobIndex, lhs.blobOffset) ==
         std::make_tuple(rhs.offset, rhs.blobIndex, rhs.blobOffset);
}
} // namespace M

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

LogicalResult
MemorySpaceAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<MemoryBlobAttr> blobs) {
  for (auto [i, blob] : llvm::enumerate(blobs)) {
    if (blob.getKind() == MemoryKind::ConstGlobal &&
        !blob.getPointerRegions().empty()) {
      return emitError() << "const_global blob #" << i
                         << " cannot have pointer regions";
    }
    for (const PointerRegion &region : blob.getPointerRegions()) {
      if (region.blobIndex < 0 ||
          static_cast<size_t>(region.blobIndex) >= blobs.size()) {
        return emitError() << "blob #" << i << " pointer at offset "
                           << region.offset
                           << " has an out-of-bounds blob index: "
                           << region.blobIndex;
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MemRefAttr
//===----------------------------------------------------------------------===//

LogicalResult MemRefAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 MemoryModelAttr model, int64_t index,
                                 int64_t offset, Type type) {
  if (index < 0 || static_cast<size_t>(index) >= model.getMemory().size())
    return emitError() << "memref blob index " << index << " is out-of-bounds";
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Mojo/Interpreter/InterpreterAttrs.cpp.inc"

bool M::isGlobalBlob(MemoryBlobAttr blob) {
  return blob.getKind() == MemoryKind::ConstGlobal ||
         (blob.getKind() == MemoryKind::Persistent && blob.getAddressSpace());
}
