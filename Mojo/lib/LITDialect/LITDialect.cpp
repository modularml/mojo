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
// This file implements the LIT dialect.
//
//===----------------------------------------------------------------------===//

#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/CODialect/CODialect.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Support/Compiler/Bytecode.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN::LIT;
using KGEN::ArgConvention;
using KGEN::ConstraintAttr;
using KGEN::DeclInterface;

//===----------------------------------------------------------------------===//
// LITDialectFoldInterface
//===----------------------------------------------------------------------===//

namespace {
struct LITDialectFoldInterface : public mlir::DialectFoldInterface {
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
// LITOpAsmDialectInterface
//===----------------------------------------------------------------------===//

struct LITOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (!attr)
      return AliasResult::NoAlias;

    if (isa<DocStringAttr>(attr)) {
      // Doc strings are nearly always long, so make sure to print them as
      // aliases.
      os << "doc_string";
      return AliasResult::OverridableAlias;
    }

    if (auto symbol = dyn_cast<SymbolAttr>(attr)) {
      if (std::optional<StringRef> alias =
              StructType::getAliasName(symbol.getValue())) {
        os << *alias;
        return AliasResult::OverridableAlias;
      }
      return AliasResult::NoAlias;
    }

    return AliasResult::NoAlias;
  }
};

//===----------------------------------------------------------------------===//
// LITDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using WrappedStructExtractAttr = WrappedAttrType<StructExtractAttr>;
using WrappedOriginUnionAttr = WrappedAttrType<OriginUnionAttr>;
using WrappedOriginMutCastAttr = WrappedAttrType<OriginMutCastAttr>;
using WrappedOriginSetAttr = WrappedAttrType<OriginSetAttr>;
using WrappedOriginSetUnionAttr = WrappedAttrType<OriginSetUnionAttr>;
using WrappedOriginFieldAttr = WrappedAttrType<OriginFieldAttr>;
using WrappedInteriorOriginAttr = WrappedAttrType<InteriorOriginAttr>;
using WrappedOriginSubtreeAttr = WrappedAttrType<OriginSubtreeAttr>;

//===----------------------------------------------------------------------===//
// Utilities

using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;

static LogicalResult
readStructValues(DialectBytecodeReader &reader,
                 SmallVectorImpl<std::tuple<StringAttr, TypedAttr>> &values) {
  return reader.readList(values, [&](std::tuple<StringAttr, TypedAttr> &value) {
    if (failed(reader.readAttribute(std::get<0>(value))) ||
        failed(reader.readAttribute(std::get<1>(value))))
      return failure();
    return LogicalResult::success();
  });
}

static void
writeStructValues(DialectBytecodeWriter &writer,
                  ArrayRef<std::tuple<StringAttr, TypedAttr>> values) {
  writer.writeList(values, [&](auto &value) {
    writer.writeAttribute(std::get<0>(value));
    writer.writeAttribute(std::get<1>(value));
  });
}

#include "Mojo/LITDialect/LITDialectBytecode.cpp.inc"

struct LITDialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  LITDialectBytecodeInterface(Dialect *dialect)
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

// Pull in the dialect definition.
#include "Mojo/LITDialect/LITDialect.cpp.inc"

void LITDialect::initialize() {
  // Register attributes.
  registerAttributes();
  addInterfaces<LITDialectFoldInterface, LITOpAsmDialectInterface>();

  // Register types.
  registerTypes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "Mojo/LITDialect/LIT.cpp.inc"
      >();

  addInterface<LITDialectBytecodeInterface>();
}

Operation *LITDialect::materializeConstant(OpBuilder &b, Attribute value,
                                           Type type, Location loc) {
  return ParamConstantOp::create(b, loc, type, cast<TypedAttr>(value));
}
