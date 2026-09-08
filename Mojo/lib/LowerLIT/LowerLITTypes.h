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

#ifndef KGEN_LOWERLIT_LOWERLITTYPES_H
#define KGEN_LOWERLIT_LOWERLITTYPES_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"

namespace mlir {
class LockedSymbolTableCollection;
} // namespace mlir

namespace M::KGEN::LIT {

struct StructDecl {
  /// Return true if the struct should be flattened when lowered.
  /// Single-element register-passable structs are flattened to their element
  /// type, UNLESS @align is specified with a value > 1 - in that case we
  /// preserve the struct to maintain the alignment metadata.
  /// True when the struct is unconditionally register-passable (isMemoryOnly
  /// is a concrete false).
  bool isRegisterPassable() const {
    if (auto b = dyn_cast<BoolAttr>(isMemoryOnlyAttr))
      return !b.getValue();
    return false;
  }

  bool isSingleElement() const {
    if (!isRegisterPassable() || fields.size() != 1)
      return false;
    if (!minAlignment)
      return true;
    if (auto intAttr = dyn_cast<IntegerAttr>(minAlignment))
      return intAttr.getInt() == 1;
    return false;
  }

  /// The un-parameterized SourceNameAttr for the struct decl.
  DebugInfo::SourceNameAttr sourceName;
  /// The struct input parameters.
  ParamDeclArrayAttr decls;
  /// The isMemoryOnly attribute for the KGEN struct type. Can be a concrete
  /// BoolAttr or an i1-typed constraint proposition for conditional RP.
  TypedAttr isMemoryOnlyAttr;
  /// Explicit minimum alignment specified via @align(N), or null if
  /// unspecified. Uses TypedAttr to support future parametric alignment.
  TypedAttr minAlignment;
  /// The location of the decl, for emitting errors.
  LocationAttr loc;
  /// The field names and types of the struct in order.
  SmallVector<std::pair<StringAttr, Type>> fields;
  /// The symbol ref for the type-value generator.
  SymbolRefAttr symRef;

  /// True once this struct has been checked for illegal layout recursion.
  bool done = false;
  /// True if this struct lies on a lowering-recursion cycle and therefore needs
  /// a pre-generated shallow (erased) layout for the AsType cycle breaker.
  bool needsErasure = false;
};

struct StructDecls {
  LogicalResult process(ModuleOp module, SymbolTable &symtab);

  /// Lookup a struct decl.
  StructDecl &get(StringAttr name) {
    auto it = structDecls.find(name);
    assert(it != structDecls.end() && "struct decl not found");
    return it->second;
  }

  /// A map from struct name and field name to index. Used for lowering `insert`
  /// and `extract` ops.
  DenseMap<std::pair<StringAttr, StringAttr>, int> fieldIndices;
  /// Map from struct name to the lowering info.
  llvm::MapVector<StringAttr, StructDecl> structDecls;
};

LogicalResult lowerLITTypes(ModuleOp module, StructDecls &decls,
                            mlir::LockedSymbolTableCollection &symtab);

} // namespace M::KGEN::LIT

#endif // KGEN_LOWERLIT_LOWERLITTYPES_H
