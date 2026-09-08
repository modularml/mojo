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
//
// This file implements the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/Interpreter/InterpreterDialect.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/Compiler/Bytecode.h"
#include "Support/IPInt.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace M;
using namespace KGEN;

namespace {

//===----------------------------------------------------------------------===//
// KGENDialectFoldInterface
//===----------------------------------------------------------------------===//

struct KGENDialectFoldInterface : public mlir::DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Never hoist a constant out of a declaration scope. We could scan the
  /// parameters declarations to find the highest scope a constant could be
  /// hoisted into, but that is expensive to do. We also do not hoist constants
  /// out of ops that define a subprogram location scope, since the hoisted
  /// constant would carry incorrect scope information into their new scope.
  bool shouldMaterializeInto(Region *region) const override {
    if (DebugInfo::shouldMaterializeConstantsInto(*region))
      return true;
    return isa<DeclInterface>(region->getParentOp());
  }
};

//===----------------------------------------------------------------------===//
// KGENDialectOpAsmDialectInterface
//===----------------------------------------------------------------------===//

struct KGENDialectAliasOptions {
  llvm::cl::opt<bool> printInlineTypeValues{
      "kgen-print-inline-type-values",
      llvm::cl::desc("Print type values inline. Used for FileCheck testing."),
      llvm::cl::init(false)};
};

} // namespace

static llvm::ManagedStatic<KGENDialectAliasOptions> clOptions;

void KGEN::registerKGENCommandLineOptions() { *clOptions; }

namespace {

struct KGENDialectOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  //===--------------------------------------------------------------------===//
  // Aliases

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto typeCst = dyn_cast<TypeParamAttr>(attr)) {
      // Do not alias the type constant if it is a simple mlir Type.
      if (clOptions->printInlineTypeValues ||
          typeCst.hasIdenticalRepresentation())
        return AliasResult::NoAlias;

      os << "type_value";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    return AliasResult::NoAlias;
  }
};

//===----------------------------------------------------------------------===//
// KGENDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using WrappedParamRefType = WrappedAttrType<ParamType>;
using WrappedTypeValueType = WrappedAttrType<TypeValueType>;
using WrappedVariantType = WrappedAttrType<VariantType>;

using WrappedParamOperatorAttr = WrappedAttrType<ParamOperatorAttr>;
using WrappedTypeParamAttr = WrappedAttrType<TypeParamAttr>;
using WrappedStructExtractAttr = WrappedAttrType<StructExtractAttr>;

//===----------------------------------------------------------------------===//
// Utilities

using KGEN::NoneType;
using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;

static LogicalResult readFnEffects(DialectBytecodeReader &reader,
                                   FnEffects &effects) {
  impl::FnEffects impl;
  if (failed(M::readIntegral(reader, impl)))
    return failure();
  effects = impl;
  return success();
}

static void writeFnEffects(DialectBytecodeWriter &writer, FnEffects effects) {
  M::writeIntegral(writer, effects.getImpl());
}

static LogicalResult readKGENDType(DialectBytecodeReader &reader,
                                   KGENDType &dtype) {
  FailureOr<APInt> result = reader.readAPIntWithKnownWidth(8);
  if (failed(result))
    return failure();
  dtype = DType(static_cast<uint8_t>(result->getLimitedValue()));
  return success();
}

static void writeKGENDType(DialectBytecodeWriter &writer, KGENDType dtype) {
  writer.writeAPIntWithKnownWidth(APInt(8, dtype.getValue()));
}

static LogicalResult readDTypeValues(DialectBytecodeReader &reader,
                                     SmallVectorImpl<DTypeValue> &values) {
  uint64_t size;
  if (failed(reader.readVarInt(size)))
    return failure();
  values.reserve(size);
  for (unsigned i = 0; i < size; ++i) {
    uint64_t kind, width;
    if (failed(reader.readVarInt(kind)) || failed(reader.readVarInt(width)))
      return failure();
    FailureOr<APInt> value = reader.readAPIntWithKnownWidth(width);
    if (failed(value))
      return failure();
    values.emplace_back(std::move(*value), static_cast<KGENDType>(kind));
  }
  return success();
}

static void writeDTypeValues(DialectBytecodeWriter &writer,
                             ArrayRef<DTypeValue> values) {
  writer.writeVarInt(values.size());
  for (const DTypeValue &value : values) {
    writer.writeVarInt(value.getDType().getValue());
    writer.writeVarInt(value.getData().getBitWidth());
    writer.writeAPIntWithKnownWidth(value.getData());
  }
}

#include "Mojo/KGENDialect/KGENDialectBytecode.cpp.inc"

struct KGENDialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  KGENDialectBytecodeInterface(Dialect *dialect)
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
// Dialect specification.
//===----------------------------------------------------------------------===//

void KGENDialect::initialize() {
  registerAttributes();
  registerTypes();
  addInterfaces<KGENDialectFoldInterface, KGENDialectOpAsmDialectInterface,
                KGENDialectBytecodeInterface>();
  injectAttrInterfaces();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "Mojo/KGENDialect/KGEN.cpp.inc"
      >();
}

void KGENDialect::registerKeywordParser(StringRef keyword, TypeParseFn parse) {
  if (!typeParseFns.try_emplace(keyword, parse).second)
    llvm::report_fatal_error("duplicate pretty type keyword: " + keyword);
}

void KGENDialect::registerPrettyType(StringRef keyword, TypeParseFn parse,
                                     mlir::TypeID id, TypePrintFn print) {
  registerKeywordParser(keyword, parse);
  if (!typePrintFns.try_emplace(id, print).second)
    llvm::report_fatal_error("duplicate printer for: " + keyword);
  typeNames.try_emplace(id, keyword);
}

std::optional<StringRef> KGENDialect::getTypeName(mlir::TypeID id) {
  auto it = typeNames.find(id);
  if (it == typeNames.end())
    return {};
  return it->second;
}

Operation *KGENDialect::materializeConstant(OpBuilder &b, Attribute value,
                                            Type type, Location loc) {
  return ParamConstantOp::create(b, loc, type, cast<TypedAttr>(value));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "Mojo/KGENDialect/KGENDialect.cpp.inc"
