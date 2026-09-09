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

#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/Compiler/Bytecode.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cctype>
#include <cstdint>
#include <optional>

using namespace M;
using namespace M::DebugInfo;

namespace {

//===----------------------------------------------------------------------===//
// DebugInfoOpAsmDialectInterface
//===----------------------------------------------------------------------===//

struct DebugInfoOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (!attr)
      return AliasResult::NoAlias;

    // Always alias source name attributes. They tend to be long.
    if (auto sourceName = dyn_cast<SourceNameAttr>(attr)) {
      if (sourceName.getParamTypes().empty() &&
          sourceName.getArgTypes().empty() &&
          sourceName.getParamValues().empty() && !sourceName.getParent())
        return AliasResult::NoAlias;
      if (llvm::all_of(sourceName.getName(),
                       [](char c) { return std::isalnum(c) || c == '_'; })) {
        os << sourceName.getName().getValue() << "_name";
        return AliasResult::OverridableAlias;
      }
      return AliasResult::NoAlias;
    }

    // Essentially all of the debug info attributes are heavy syntax-wise, so
    // just print them all as aliases whenever we can.
    return TypeSwitch<Attribute, AliasResult>(attr)
        .Case<
#define GET_ATTRDEF_LIST
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.cpp.inc"
            >([&](auto attr) {
          os << decltype(attr)::getMnemonic();
          return AliasResult::OverridableAlias;
        })
        .Default([](Attribute) { return AliasResult::NoAlias; });
  }

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    // Essentially all of the debug info types are heavy syntax-wise, so
    // just print them all as aliases whenever we can.
    return TypeSwitch<Type, AliasResult>(type)
        .Case<
#define GET_TYPEDEF_LIST
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.cpp.inc"
            >([&](auto attr) {
          os << decltype(attr)::getMnemonic();
          return AliasResult::OverridableAlias;
        })
        .Default([](Type) { return AliasResult::NoAlias; });
  }
};

//===----------------------------------------------------------------------===//
// DebugInfoDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;
using OptionalUnsigned = std::optional<unsigned>;

static LogicalResult readOptionalUnsigned(DialectBytecodeReader &reader,
                                          std::optional<unsigned> &value) {
  bool hasValue;
  uint64_t val;
  if (failed(reader.readVarIntWithFlag(val, hasValue)))
    return failure();
  if (!hasValue)
    return success();
  value = val;
  return success();
}

static void writeOptionalUnsigned(DialectBytecodeWriter &writer,
                                  const std::optional<unsigned> &value) {
  if (!value)
    return writer.writeVarIntWithFlag(0, /*flag=*/false);
  return writer.writeVarIntWithFlag(value.value(), /*flag=*/true);
}

#include "Support/DebugInfoDialect/IR/DebugInfoDialectBytecode.cpp.inc"

struct DebugInfoDialectBytecodeInterface
    : public mlir::BytecodeDialectInterface {
  DebugInfoDialectBytecodeInterface(Dialect *dialect)
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
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::initialize() {
  registerAttributes();
  registerOperations();
  registerTypes();
  addInterfaces<DebugInfoOpAsmDialectInterface,
                DebugInfoDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoDialect.cpp.inc"
