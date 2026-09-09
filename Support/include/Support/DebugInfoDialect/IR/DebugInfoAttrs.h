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

#ifndef SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
#define SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H

#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/TypeSwitch.h"
#include <functional>
#include <optional>
#include <string>
#include <utility>

//===----------------------------------------------------------------------===//
// DIAttr
//===----------------------------------------------------------------------===//

namespace mlir {
class FunctionOpInterface;
}

namespace M::DebugInfo {
/// This class represents the base class of all DebugInfo attributes.
class DIAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents the base class of DebugInfo attributes that form
/// a scope.
class DIScopeAttr : public DIAttr {
public:
  using DIAttr::DIAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents the base class of DebugInfo attributes that form
/// a local scope.
class DILocalScopeAttr : public DIScopeAttr {
public:
  using DIScopeAttr::DIScopeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

} // namespace M::DebugInfo

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoEnums.h.inc"
#include "Support/DebugInfoDialect/IR/DebugInfoExprAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h.inc"

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {

/// This class provides utilities for prepending conversion expressions to an
/// existing conversion expression. As optimizations accumulate in the IR,
/// the conversion expression of a debuginfo needs to track new optimizations
/// by prepending conversions, i.e. replacing the leaves of expressions with
/// new subtrees.
/// All conversion thru the same prepender instance goes thru a shared attr/type
/// replacer cache.
class DIExprLeafReplacer {
public:
  DIExprLeafReplacer(std::function<ErrorOr<DIExprAttr>(Type)> conversionFunc);

  // Apply the leafReplacer to the input expression.
  // In practice, this means replacing the leaves of expr with the result of the
  // leafReplacer.
  ErrorOr<DIExprAttr> apply(DIExprAttr expr);

private:
  // Records any error message emitted during an `apply` call.
  // Always cleared before running the replacer.
  std::string currErrorMsg;

  std::function<ErrorOr<DIExprAttr>(Type)> leafReplacer;
  mlir::AttrTypeReplacer replacer;
};

/// Enhanced DIExprLeafReplacer that allows replacers to take an additional
/// opaque argument. To take advantage of caching, a replacer is created and
/// saved for each unique argument that is used with `apply`. The user is
/// responsible for making sure the number of unique arguments do not use up
/// too much memory for caching.
template <typename KeyT>
class DIExprParameterizedLeafReplacer {
public:
  DIExprParameterizedLeafReplacer(
      std::function<ErrorOr<DIExprAttr>(Type, KeyT)> conversionFunc)
      : leafReplacer(std::move(conversionFunc)) {}

  ErrorOr<DIExprAttr> apply(DIExprAttr expr, KeyT key) {
    mlir::AttrTypeReplacer &replacer = getOrCreateReplacer(std::move(key));
    currErrorMsg = {};
    auto newExpr = dyn_cast_or_null<DIExprAttr>(replacer.replace(expr));
    if (!currErrorMsg.empty())
      return Error(currErrorMsg);
    if (!newExpr)
      return Error("LeafReplacer failed to replace.");
    return newExpr;
  }

private:
  mlir::AttrTypeReplacer &getOrCreateReplacer(KeyT &&key) {
    auto [it, inserted] = replacers.try_emplace(key);
    if (inserted) {
      it->second.addReplacement(
          [&, key = std::move(key)](DIIRValueExprAttr irValue)
              -> std::optional<std::pair<Attribute, WalkResult>> {
            auto result = leafReplacer(irValue.getType(), key);
            if (failed(result)) {
              currErrorMsg = result.getError();
              return std::make_pair(nullptr, WalkResult::skip());
            }

            auto conversionResult = result.get();
            if (conversionResult.getType() != irValue.getType()) {
              currErrorMsg = "Converter result type differs from input type.";
              return std::make_pair(nullptr, WalkResult::skip());
            }
            return std::make_pair(conversionResult, WalkResult::skip());
          });
    }
    return it->second;
  }

  std::string currErrorMsg;

  std::function<ErrorOr<DIExprAttr>(Type, KeyT)> leafReplacer;
  DenseMap<KeyT, mlir::AttrTypeReplacer> replacers;
};

enum class LocWalkPolicy {
  CalleePriority, // Walk inlined call-stack inside-out.
  CallerPriority  // Walk inlined call-stack outside-in.
};

/// Pre-order walk on location tree. `policy` determines which child of a
/// CallSiteLoc to walk first.
WalkResult walkLocation(Location loc, LocWalkPolicy policy,
                        function_ref<WalkResult(Location)> walkFn);

/// Wrapper for `walkLocation` that visits fused debug scopes of locations.
WalkResult walkScope(Location loc, LocWalkPolicy policy,
                     function_ref<WalkResult(DIScopeAttr)> walkFn);

/// Extract the original source location from a call location, taking into
/// account debuginfo and other structure within locations. Returns null if a
/// FileLineColLoc cannot be found.
FileLineColLoc extractSourceLoc(Location callLoc);

/// Strip all debug scopes from this location recursively.
/// If a debug scope is fused with only one location, the fused location will be
/// collapsed. Otherwise (including the empty case), the fused location will
/// remain (just without any metadata).
Location stripDebugScopesRecursively(Location loc);

/// Extract the first scope found in a location based on a LocWalkPolicy.
template <typename ScopeT>
ScopeT extractScopeFrom(Location loc, LocWalkPolicy policy) {
  ScopeT result;
  walkScope(loc, policy, [&result](DIScopeAttr scope) {
    return scope.walk<mlir::WalkOrder::PreOrder>([&result](ScopeT innerScope) {
      result = innerScope;
      return WalkResult::interrupt();
    });
  });
  return result;
}

/// Extract the scope from the location of a function. Functions either have
/// a subprogram scope fused directly to the location, or we consider them
/// as not having any. Therefore this never requires a recursion, and
/// therefore can be done without a location cache.
DISubprogramAttr extractScope(mlir::FunctionOpInterface funcOp);

/// Extract the debug info scope from the location of the given operation.
DIScopeAttr extractScope(Operation *op);

/// Get the closest parent scope of a given type, or null if non-existent.
template <typename ScopeTy>
ScopeTy getParentScopeOfType(DIScopeAttr scope) {
  while (scope && !isa<ScopeTy>(scope))
    scope = TypeSwitch<DIScopeAttr, DIScopeAttr>(scope)
                .Case([](DILexicalBlockAttr block) { return block.getScope(); })
                .Case([](DISubprogramAttr sp) { return sp.getScope(); })
                .Default(DIScopeAttr());
  return llvm::cast_if_present<ScopeTy>(scope);
}

/// This class represents an attribute/type replacer with proper defaults for
/// updating debug information within operations.
class DIAttrTypeReplacer : public mlir::AttrTypeReplacer {
public:
  /// TODO: Upstream this templated version to AttrTypeReplacer.
  template <typename T, typename U>
  T replace(U value) {
    return dyn_cast_if_present<T>(replace(value));
  }
  using mlir::AttrTypeReplacer::replace;

  /// Replace elements within the given operation.
  void replaceElementsIn(Operation *op);

  /// Replace elements within the given operation, and any nested operations.
  void recursivelyReplaceElementsIn(Operation *op);
};

/// If the op has a subprogram scope, update it with the given linkage name
/// (and optionally the given name, if not null), as well as all references to
/// the scope recursively within the body.
void updateSubprogram(mlir::FunctionOpInterface op, StringAttr linkageName,
                      SourceNameAttr name = {});

/// Update the location of the op as if it was inlined at the given caller
/// location, handling special location interfaces.
void updateInlinedLoc(Operation *op, Location callerLoc);
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
