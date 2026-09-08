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
// This file implements logic for working with KGEN parameters.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include <algorithm>

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// IndexRefRemapper
//===----------------------------------------------------------------------===//

/// Populate the remapper with named input and result parameters.
IndexRefRemapper::IndexRefRemapper(ArrayRef<ParamDeclAttr> inputParams,
                                   size_t offset)
    : offset(offset) {
  for (auto [idx, param] : llvm::enumerate(inputParams))
    mapping.try_emplace(param.getName(), idx);
}
IndexRefRemapper::IndexRefRemapper(ArrayRef<ParamDeclRefAttr> inputParams,
                                   size_t offset)
    : offset(offset) {
  for (auto [idx, param] : llvm::enumerate(inputParams))
    mapping.try_emplace(param.getName(), idx);
}

/// Append a parameter declaration to the remapper.
void IndexRefRemapper::appendParamDecl(ParamDeclAttr paramDecl) {
  mapping.try_emplace(paramDecl.getName(), mapping.size());
}

// CRTP methods.
Attribute IndexRefRemapper::tryReplace(Attribute attr, size_t depth) {
  if (auto ref = dyn_cast<ParamDeclRefAttr>(attr)) {
    if (auto it = mapping.find(ref.getName()); it != mapping.end())
      return ParamIndexRefAttr::get(depth, it->second,
                                    Base::replaceImpl(ref.getType(), depth));
  }
  if (offset != 0) {
    if (auto ref = dyn_cast<ParamIndexRefAttr>(attr)) {
      if (ref.getDepth() == depth)
        return ParamIndexRefAttr::get(depth, ref.getIndex() + offset,
                                      Base::replaceImpl(ref.getType(), depth));
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// IndexDepthAdjuster
//===----------------------------------------------------------------------===//

Attribute IndexDepthAdjuster::tryReplace(Attribute attr, size_t depth) {
  // TODO(MOCO-2081): Change this to a dyn_cast<IndexRefAttrInterface>, per
  // https://github.com/modularml/modular/pull/62096#discussion_r2114820075.
  if (auto ref = dyn_cast<ParamIndexRefAttr>(attr)) {
    if (ref.getDepth() < depth)
      return ref;
    // Per STCHDDDOS, we adjust the depth by `adjustDepth` so it's now correct.
    return ParamIndexRefAttr::get(ref.getDepth() + adjustDepth, ref.getIndex(),
                                  replaceImpl(ref.getType(), depth));
  }
  if (auto itf = dyn_cast<IndexRefAttrInterface>(attr);
      itf && !onlyAdjustIndexRef) {
    if (itf.getDepth() < depth)
      return itf;
    // Replace subelements and then replace the whole reference with the
    // adjusted depth.
    SmallVector<Attribute, 2> attrs;
    SmallVector<Type, 2> types;
    attr.walkImmediateSubElements(
        [&](Attribute attr) { attrs.push_back(replaceImpl(attr, depth)); },
        [&](Type type) { types.push_back(replaceImpl(type, depth)); });
    return itf.replace(itf.getDepth() + adjustDepth, itf.getIndex(), attrs,
                       types);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ParameterCollector
//===----------------------------------------------------------------------===//

void ParameterCollector::collectUsesFromAttr(
    Attribute attr, SmallVectorImpl<ParamDeclRefAttr> &uses,
    bool &hasCtxEvalExpr, size_t &requiredSignatureDepth) {
  if (auto sig = dyn_cast<ParameterScopeAttrInterface>(attr)) {
    signatures.push_back(sig);
    collectUsesFromAttrImpl(attr, uses, hasCtxEvalExpr, requiredSignatureDepth);
    signatures.pop_back();
    // The result is intrinsic to `attr` itself; stepping out one signature
    // scope reduces the required surrounding depth by one.
    if (requiredSignatureDepth > 0)
      --requiredSignatureDepth;
    return;
  }
  collectUsesFromAttrImpl(attr, uses, hasCtxEvalExpr, requiredSignatureDepth);
}

/// Scan the specified attribute and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterCollector::collectUsesFromAttrImpl(
    Attribute attr, SmallVectorImpl<ParamDeclRefAttr> &uses,
    bool &hasCtxEvalExpr, size_t &requiredSignatureDepth) {
  // If we have already scanned it and know that it has no parameters in it,
  // return early.
  if (auto it = cache.parameterLess.find(attr.getAsOpaquePointer());
      it != cache.parameterLess.end()) {
    hasCtxEvalExpr |= it->second.hasCtxEvalExpr;
    requiredSignatureDepth =
        std::max(requiredSignatureDepth, it->second.requiredSignatureDepth);
    return;
  }

  // Look through any SugarAttr's we encounter.
  if (auto sugar = dyn_cast<SugarAttr>(attr)) {
    collectUsesFromAttr(sugar.getCanonical(), uses, hasCtxEvalExpr,
                        requiredSignatureDepth);
    return;
  }

  // Collect parameter references.
  if (auto paramRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    collectUsesFromType(paramRef.getType(), uses, hasCtxEvalExpr,
                        requiredSignatureDepth);
    uses.push_back(paramRef);
    return;
  }

  // Verify index parameter references.
  // TODO(MOCO-2080): Should this be dyn_cast<IndexRefAttrInterface>?
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(attr)) {
    collectUsesFromType(indexRef.getType(), uses, hasCtxEvalExpr,
                        requiredSignatureDepth);
    // Intrinsic requirement for this sub-expression: a ref with depth D needs
    // at least D+1 enclosing signature scopes around itself to be valid.
    size_t localRequiredDepth = indexRef.getDepth() + 1;
    requiredSignatureDepth =
        std::max(requiredSignatureDepth, localRequiredDepth);
    maybeVerify(
        [&](function_ref<InFlightDiagnostic()> emitError) -> LogicalResult {
          if (signatures.empty())
            return emitError() << "index reference has no contextual signature";
          if (indexRef.getDepth() >= signatures.size()) {
            // An index-based param-ref's depth is the number of signatures
            // between it and the param-decl it's pointing at. `signatures` has
            // all the (directly+indirectly) containing signatures, therefore
            // depth can't be more than signatures.size(). See IRAIDAI for more.
            return emitError()
                   << "index reference depth " << indexRef.getDepth()
                   << " exceeds depth of contextual signatures: "
                   << signatures.size();
          }
          auto sig = signatures[signatures.size() - 1 - indexRef.getDepth()];
          auto typesOrQuote = [sig]() -> std::optional<ArrayRef<Type>> {
            if (auto attrScope = dyn_cast<ParameterScopeAttrInterface>(sig)) {
              if (isa<QuoteAttr>(attrScope))
                return std::nullopt;
              return attrScope.getInputParamTypes();
            }
            return cast<ParameterScopeTypeInterface>(sig).getInputParamTypes();
          }();
          // A QuoteAttr forms a yet to be built scope, the index ref on the
          // "deferred scope" can not be verified (until the scope is formed).
          if (!typesOrQuote.has_value())
            return success();

          ArrayRef<Type> types = *typesOrQuote;
          if (indexRef.getIndex() >= types.size()) {
            return emitError() << "index reference " << indexRef.getIndex()
                               << " is out of bounds: referenced signature has "
                               << types.size() << " input parameters";
          }
          // The index parameter reference can exist in a different scope than
          // the one in which the referenced parameter was declared (see
          // PSTIAIRAID). This means the type of the reference and the type of
          // the declaration can also exist in different scopes, and thus have
          // different relative depths (see STCHDDDOS).
          // In order to correctly compare the types, we have to map the types
          // into the same scope. If there are currently N scopes, and the index
          // reference depth is M where M <= N, then the type of the declaration
          // is M scopes higher. To map the parameter type into the current
          // scope, adjust escaping index references on type by M.
          //
          // For example, consider the following scopes:
          //
          // ```
          // 0: <type,               // T: type
          // 1:  <*(1,0),            // :T a
          // 2:   <:*(2,0) *(1,0)>   // :a T
          // ```
          //
          // In the innermost scope, there is a reference to a parameter 1 scope
          // up. Thus, one scope up, the type of the referenced parameter
          // requires 1 less in the depths of index references in its type.
          Type type = types[indexRef.getIndex()];
          IndexDepthAdjuster adjuster(indexRef.getDepth());
          Type expectedType = adjuster.replace(type);
          if (!isEqualCanon(expectedType, indexRef.getType())) {
            return emitError()
                   << "type of index reference " << indexRef
                   << " does not match parameter type " << expectedType;
          }
          return success();
        });
    return;
  }

  // Check any SymbolConstantAttr's we encounter.
  if (auto ref = dyn_cast<DeclRefAttrInterface>(attr))
    verifyRefAttr(ref);

  // Save the number of nested parameters before recursing and check whether the
  // attribute has a nested constant expression.
  size_t oldSize = uses.size();
  // Parameterized type constants are by definition unresolved expressions.
  bool hasNestedCtxEvalExpr = false;
  size_t nestedRequiredDepth = 0;

  // Recursively check for any nested types/attributes, e.g. the elements of an
  // array attribute.
  attr.walkImmediateSubElements(
      [&](Attribute attr) {
        collectUsesFromAttr(attr, uses, hasNestedCtxEvalExpr,
                            nestedRequiredDepth);
      },
      [&](Type type) {
        collectUsesFromType(type, uses, hasNestedCtxEvalExpr,
                            nestedRequiredDepth);
      });

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == uses.size() && !nestedRequiredDepth) {
    // Check whether this is a parameterless expression.
    hasNestedCtxEvalExpr |= isa<ContextuallyEvaluatedAttrInterface>(attr);
    cache.parameterLess.try_emplace(
        attr.getAsOpaquePointer(),
        Analysis::ParameterlessInfo{hasNestedCtxEvalExpr, nestedRequiredDepth});
    hasCtxEvalExpr |= hasNestedCtxEvalExpr;
  }
  requiredSignatureDepth =
      std::max(requiredSignatureDepth, nestedRequiredDepth);
}

void ParameterCollector::collectUsesFromType(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &uses, bool &hasCtxEvalExpr,
    size_t &requiredSignatureDepth) {
  if (auto sig = dyn_cast<ParameterScopeTypeInterface>(type)) {
    signatures.push_back(sig);
    collectUsesFromTypesImpl(type, uses, hasCtxEvalExpr,
                             requiredSignatureDepth);
    signatures.pop_back();
    // The result is intrinsic to `type` itself; stepping out one signature
    // scope reduces the required surrounding depth by one.
    if (requiredSignatureDepth > 0)
      --requiredSignatureDepth;
    return;
  }
  return collectUsesFromTypesImpl(type, uses, hasCtxEvalExpr,
                                  requiredSignatureDepth);
}

void ParameterCollector::collectUsesFromTypesImpl(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &uses, bool &hasCtxEvalExpr,
    size_t &requiredSignatureDepth) {
  // Ignore types we have already scanned.
  if (auto it = cache.parameterLess.find(type.getAsOpaquePointer());
      it != cache.parameterLess.end()) {
    hasCtxEvalExpr |= it->second.hasCtxEvalExpr;
    requiredSignatureDepth =
        std::max(requiredSignatureDepth, it->second.requiredSignatureDepth);
    return;
  }

  // Check any StructType's we encounter.
  if (auto refType = dyn_cast<StructTypeInterface>(type))
    verifyRefType(refType);

  // Save the number of nested parameters before recursing and check whether the
  // attribute has a nested constant expression.
  size_t oldSize = uses.size();
  bool hasNestedConstExpr = false;
  size_t nestedRequiredDepth = 0;

  // Recursively check for any nested types, e.g. the input/outputs of a
  // function type, types like !kgen.scalar<ty> etc.
  type.walkImmediateSubElements(
      [&](Attribute attr) {
        collectUsesFromAttr(attr, uses, hasNestedConstExpr,
                            nestedRequiredDepth);
      },
      [&](Type type) {
        collectUsesFromType(type, uses, hasNestedConstExpr,
                            nestedRequiredDepth);
      });

  // If the type had parameter uses or constant expressions, don't consider it
  // "parameterless".  We want other operations using the same type to record
  // the uses as well.
  if (oldSize == uses.size() && !nestedRequiredDepth) {
    cache.parameterLess.try_emplace(
        type.getAsOpaquePointer(),
        Analysis::ParameterlessInfo{hasNestedConstExpr, nestedRequiredDepth});
    hasCtxEvalExpr |= hasNestedConstExpr;
  }
  requiredSignatureDepth =
      std::max(requiredSignatureDepth, nestedRequiredDepth);
}

//===----------------------------------------------------------------------===//
// VerifyingParameterCollector
//===----------------------------------------------------------------------===//

namespace {
class VerifyingParameterCollector : public ParameterCollector {
public:
  VerifyingParameterCollector(SymTabEvaluationContext *evaluationContext,
                              ParameterCollector::Analysis &cache)
      : ParameterCollector(cache), evaluationContext(evaluationContext) {}

  /// The first time we encounter an attribute with a reference to an
  /// out-of-line declaration, verify it.
  void verifyRefAttr(DeclRefAttrInterface refAttr) override;

  /// The first time we encounter a StructType, check to see if its parameter
  /// bindings agrees with the parameter declarations of the referred type
  /// declaration.
  void verifyRefType(StructTypeInterface refType) override;

  /// Invoke the verification function using the current operation's location.
  void maybeVerify(
      function_ref<LogicalResult(function_ref<InFlightDiagnostic()>)> verifyFn)
      override {
#ifndef MODULAR_PRODUCTION
    if (failed(verifyFn([&] { return mlir::emitError(op->getLoc()); })))
      hadError = true;
#endif
  }

  /// Whether a verification error occurred.
  bool hadError = false;
  /// The current operation where we are collecting parameters.
  Operation *op;

private:
  /// The symbol to use to verify symbol references.
  [[maybe_unused]] SymTabEvaluationContext *evaluationContext;
  /// Cached references that have already been verified.
  DenseSet<const void *> verifiedRefs;
};
} // namespace

void VerifyingParameterCollector::verifyRefAttr(DeclRefAttrInterface refAttr) {
#ifndef MODULAR_PRODUCTION
  // We only check this during the op verification phase.
  if (!evaluationContext)
    return;

  if (!verifiedRefs.insert(refAttr.getAsOpaquePointer()).second)
    return;

  if (failed(refAttr.verifySymbolUses(*evaluationContext, op->getLoc()))) {
    // If the attribute verifier failed, it will only have the location
    // source information we're passing down.  Include the full op dump now
    // for more context since this is an internal MLIR invariant violation.
    op->emitOpError("invalid symbol use within this operator");
    hadError = true;
  }
#endif
}

void VerifyingParameterCollector::verifyRefType(StructTypeInterface refType) {
#ifndef MODULAR_PRODUCTION
  VerboseCompilerTimeTraceScope traceScope("verifyRefType");

  // We only check this during the op verification phase.
  if (!evaluationContext)
    return;

  if (!verifiedRefs.insert(refType.getAsOpaquePointer()).second)
    return;

  if (failed(refType.verifySymbolUses(*evaluationContext, op->getLoc())))
    hadError = true;
#endif
}

//===----------------------------------------------------------------------===//
// ParamIndexRefAttrFinder
//===----------------------------------------------------------------------===//

template <typename T>
static std::optional<size_t> findFirstReferenceImpl(
    T value, size_t depth,
    DenseMap<std::pair<size_t, const void *>, ssize_t> &cache) {
  if (!value)
    return -1;

  // If we've already processed this value, just reuse the memoized result.
  std::pair<size_t, const void *> cacheKey(depth, value.getAsOpaquePointer());
  auto it = cache.find(cacheKey);
  if (it != cache.end())
    return it->second == -1 ? std::optional<size_t>()
                            : static_cast<size_t>(it->second);

  // Signatures push a parameter scope.
  if constexpr (std::is_base_of_v<Type, T>)
    if (isa<ParameterScopeTypeInterface>(value))
      ++depth;

  if constexpr (std::is_base_of_v<Attribute, T>)
    if (isa<ParameterScopeAttrInterface>(value))
      ++depth;

  ssize_t hasReference = -1;
  // Check to see if this is locally an index with the right depth.
  if constexpr (std::is_base_of_v<Attribute, T>)
    if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value))
      if (indexRef.getDepth() == depth)
        hasReference = indexRef.getIndex();

  if (hasReference == -1) {
    value.walkImmediateSubElements(
        [&](Attribute attr) {
          if (hasReference == -1)
            if (auto result = findFirstReferenceImpl(attr, depth, cache))
              hasReference = result.value();
        },
        [&](Type type) {
          if (hasReference == -1)
            if (auto result = findFirstReferenceImpl(type, depth, cache))
              hasReference = result.value();
        });
  }

  cache[cacheKey] = hasReference;
  return hasReference == -1 ? std::optional<size_t>()
                            : static_cast<size_t>(hasReference);
}

std::optional<size_t>
ParamIndexRefAttrFinder::findOneReference(TypedAttr value) {
  return findFirstReferenceImpl(value, 0, cached);
}

std::optional<size_t> ParamIndexRefAttrFinder::findOneReference(Type type) {
  return findFirstReferenceImpl(type, 0, cached);
}

//===----------------------------------------------------------------------===//
// ParameterUseDefGraph Implementation
//===----------------------------------------------------------------------===//

namespace {
struct ParameterUseDefGraphNodeIterator;

/// Each node in the parameter use-def graph is a parameter definition. An
/// outgoing edge represents a use of another parameter in the definition of the
/// parameter, and incoming edges are from other parameters that use this one in
/// their declarations.
///
/// A null parameter indicates a virtual root node that points to all other
/// nodes.
struct ParameterUseDefGraphNode {
  ParameterUseDefGraph *g;
  StringAttr param;

  /// Enable nodes to be check for equality.
  bool operator==(const ParameterUseDefGraphNode &rhs) const {
    return param == rhs.param;
  }
  bool operator!=(const ParameterUseDefGraphNode &rhs) const {
    return param != rhs.param;
  }

  ParameterUseDefGraphNodeIterator begin() const;
  ParameterUseDefGraphNodeIterator end() const;
};

/// An iterator for the parameter use-def graph. This class iterates through
/// the uses of a parameter.
struct ParameterUseDefGraphNodeIterator
    : public llvm::iterator_facade_base<ParameterUseDefGraphNodeIterator,
                                        std::forward_iterator_tag,
                                        ParameterUseDefGraphNode> {
  ParameterUseDefGraphNodeIterator(ParameterUseDefGraphNode node,
                                   size_t useNumber)
      : node(node), useNumber(useNumber) {}

  ParameterUseDefGraphNode node;
  size_t useNumber;

  /// Enable iterators to be checked for equality.
  bool operator==(const ParameterUseDefGraphNodeIterator &rhs) const {
    return node == rhs.node && useNumber == rhs.useNumber;
  }

  /// Enable iterators to be incremented.
  ParameterUseDefGraphNodeIterator operator++() {
    ++useNumber;
    return *this;
  }
  ParameterUseDefGraphNodeIterator operator++(int) {
    ParameterUseDefGraphNodeIterator tmp = *this;
    ++*this;
    return tmp;
  }

  /// For the virtual root node, deference into the parameter definition. For
  /// regular nodes, deference to the node that defines the used parameter.
  ParameterUseDefGraphNode operator*() const {
    if (!node.param)
      return {node.g, node.g->params[useNumber]};
    return {node.g, node.g->defs[node.param].uses[useNumber].getName()};
  }
};

ParameterUseDefGraphNodeIterator ParameterUseDefGraphNode::begin() const {
  return {*this, 0};
}

ParameterUseDefGraphNodeIterator ParameterUseDefGraphNode::end() const {
  // For the virtual root node, the end iterator is the last parameter.
  if (!param)
    return {*this, g->params.size()};
  // Do not traverse through to parameters in higher scopes.
  Region *scope = g->decls[param].scope;
  if (!scope || !g->scope->isAncestor(scope))
    return begin();
  // If the used parameter has no definition, this is a leaf node.
  auto it = g->defs.find(param);
  if (it == g->defs.end())
    return begin();
  // The end iterator is the last use.
  return {*this, it->second.uses.size()};
}
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<ParameterUseDefGraphNode> {
  static unsigned getHashValue(const ParameterUseDefGraphNode &node) {
    return DenseMapInfo<StringAttr>::getHashValue(node.param);
  }
  static bool isEqual(const ParameterUseDefGraphNode &lhs,
                      const ParameterUseDefGraphNode &rhs) {
    return lhs == rhs;
  }
};

template <>
struct GraphTraits<ParameterUseDefGraph *> {
  using NodeRef = ParameterUseDefGraphNode;
  using ChildIteratorType = ParameterUseDefGraphNodeIterator;

  static NodeRef getEntryNode(ParameterUseDefGraph *g) { return {g, nullptr}; }

  static ChildIteratorType child_begin(NodeRef node) { return node.begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node.end(); }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// ParameterUseDefGraph
//===----------------------------------------------------------------------===//

void impl::scanAllAttrsAndTypes(Operation *op,
                                function_ref<void(Attribute)> scanAttr,
                                function_ref<void(Type)> scanType) {
  llvm::for_each(op->getResultTypes(), scanType);
  for (Region &region : op->getRegions())
    llvm::for_each(region.getArgumentTypes(), scanType);

  // FIXME(#7743): Scan locations too when the elaborator has been updated to
  // handle the new parameter use-def graph.
  scanAttr(op->getAttrDictionary());
}

/// Collect parameter uses from the operation. If there are any uses or
/// otherwise unresolved parameter operators, indicate that the operation is
/// parametric.
static void collectUses(ParameterUseDefGraph &g, VerifyingParameterCollector &c,
                        Operation *op, bool isDefOrDecl) {
  // Track whether parameter uses or expressions were found.
  bool hasCtxEvalExpr = false;
  // Ignored. Free index refs are verified and errors in the collector.
  size_t requiredSignatureDepth = 0;
  SmallVector<ParamDeclRefAttr> uses;

  auto scanAttr = [&](Attribute attr) {
    c.collectUsesFromAttr(attr, uses, hasCtxEvalExpr, requiredSignatureDepth);
  };
  auto scanType = [&](Type type) {
    c.collectUsesFromType(type, uses, hasCtxEvalExpr, requiredSignatureDepth);
  };

  auto itf = dyn_cast<ParamOpInterface>(op);
  if (itf) {
    // If the parameter operation is the containing declaration, collect only
    // uses below the defined scope.
    if (op == g.scope->getParentOp())
      itf.collectParameterUsesBelow(scanAttr, scanType);
    else
      itf.collectParameterUses(scanAttr, scanType);

    // Otherwise, scan all attributes and types if the operation is not a
    // declaration or it is the containing declaration.
  } else if (!isa<DeclInterface>(op) || op == g.scope->getParentOp()) {
    impl::scanAllAttrsAndTypes(op, scanAttr, scanType);
  }

  // Operations that are implicitly parametric need to be visited even if they
  // don't contain parameter expressions or parameter uses. Defer to the
  // interface if it is implemented; otherwise, consider all generator users to
  // be parametric.
  // A rebind must be visited even when neither type mentions a parameter, since
  // its input and result types still have to be checked against each other.
  auto isImplicitlyParametric = [&] {
    return (itf && itf.isImplicitlyParametric()) ||
           isa<GeneratorUserOpInterface, DeferredOp, CodeGenReachableOp,
               RebindOp>(op);
  };

  // If the operation is parametric, add it to the list.
  if (hasCtxEvalExpr || !uses.empty()) {
    if (!isDefOrDecl)
      g.paramOps.push_back(op);
    g.opUses[op] = std::move(uses);
  } else if (!isDefOrDecl && isImplicitlyParametric()) {
    // Track implicitly parametric operations only when they don't already
    // declare parameters.
    g.paramOps.push_back(op);
  }
}

static LogicalResult recordDecl(ParameterUseDefGraph &g, ParamDeclAttr decl,
                                Operation *op, Region &scope) {
  ParamDeclaration &paramDecl = g.decls[decl.getName()];

  if (paramDecl.scope) {
    // If this parameter has already been declared by a different op, or by the
    // same op in the same scope, we have an error.
    // NOTE: Parameters can be redeclared by the same op in different scopes.
    // This is so that we don't run into redeclaration with ops that implement
    // both ParamOpInterface and DeclInterface.
    if (paramDecl.declOp != op || paramDecl.scope == &scope)
      return (emitError(op->getLoc(), "redeclaration of parameter ")
              << decl.getName())
                 .attachNote(paramDecl.declOp->getLoc())
             << "previous declaration here";
  }

  // Record the new declaration.
  paramDecl.declOp = op;
  paramDecl.type = decl.getType();
  paramDecl.scope = &scope;
  return success();
}

static FailureOr<ParamDefinition *>
recordDef(ParameterUseDefGraph &g, ParamDeclAttr decl, Operation *op) {
  ParamDefinition &paramDef = g.defs[decl.getName()];
  if (paramDef.defOp) {
    return (emitError(op->getLoc(), "redefinition of parameter ")
            << decl.getName())
               .attachNote(paramDef.defOp->getLoc())
           << "see previous definition here";
  }
  g.params.push_back(decl.getName());
  paramDef.defOp = op;
  return &paramDef;
}

/// Cycles detected in the definition of a parameter are always forbidden. When
/// that occurs, emit a nice error detailing the cycle.
static void emitCycleError(ParameterUseDefGraph &g,
                           ArrayRef<ParameterUseDefGraphNode> nodes) {
  // Build a set of the nodes in the SCC so we can do efficient queries.
  SmallPtrSet<StringAttr, 4> paramsInSCC;
  for (const ParameterUseDefGraphNode &node : nodes)
    paramsInSCC.insert(node.param);

  // Emit the error on the container operation with notes indicating the
  // problem.
  InFlightDiagnostic diag = emitError(
      g.scope->getParentOp()->getLoc(),
      "cyclic reference between expressions defining and using parameters");

  // An SCC may contain multiple different cyclic paths.  We diagnose the first
  // one we see by walking the graph - always staying within the SCC, until we
  // reach a node we've already seen.  Given this is an SCC, we know that we
  // will eventually reach one of the nodes in the path.
  SmallVector<ParameterUseDefGraphNodeIterator> path;
  SmallPtrSet<StringAttr, 4> paramsInPath;
  ParameterUseDefGraphNode nextNode = nodes.front();

  // Loop until we find a backreference.
  while (paramsInPath.insert(nextNode.param).second) {
    // Find an iterator from this node to another within this SCC.
    ParameterUseDefGraphNodeIterator it = nextNode.begin();
    while (!paramsInSCC.contains((*it).param)) {
      // Advance past edges to nodes outside the SCC.
      ++it;
      assert(it != nextNode.end() && "SCC means we should find an edge");
    }

    path.push_back(it);
    nextNode = *it;
  }

  // Okay, we found a path through the SCC that loops back to 'nextNode'.  Note
  // that it may not be a cycle though, because we may have found a path like
  // A->B->C->D->C.  In this case, we want to just diagnose C->D->C.  Handle
  // this by trimming off the beginning of the path until we find `C`.
  while (path.front().node != nextNode)
    path.erase(path.begin());

  // Okay, we found a path, diagnose it.
  for (ParameterUseDefGraphNodeIterator &edge : path) {
    const char *nextDiag = ", which references the expression:";
    if (path.size() == 1)
      nextDiag = ", which references itself";
    else if (&edge == &path.back())
      nextDiag = ", which references the first expression";

    StringAttr defParam = edge.node.param;
    diag.attachNote(g.defs[defParam].defOp->getLoc())
        << "parameter " << defParam << " is defined here" << nextDiag;
  }
}

LogicalResult ParameterUseDefGraph::calculateOrVerify(
    SymTabEvaluationContext *evaluationContext,
    ParameterCollector::Analysis &cache) {
  VerboseCompilerTimeTraceScope traceScope(
      "ParameterUseDefGraph::calculateOrVerify", [&] {
        if (auto symbol =
                dyn_cast<mlir::SymbolOpInterface>(scope->getParentOp()))
          return symbol.getName().str();
        return scope->getParentOp()->getName().getStringRef().str();
      });

  // Defer the processing of the use-def node for region declarations until
  // after nested scopes have been analyzed.
  SmallVector<std::pair<ParamDeclAttr, SmallVector<Region *, 0>>> regionValues;
  // The parameter collector to use.
  VerifyingParameterCollector c(evaluationContext, cache);

  auto processOp = [&](Operation *op) -> WalkResult {
    // Set the operation for which we are collecting parameters. It will be used
    // to report errors.
    c.op = op;

    // Track whether the operation declares or defines parameters. Operations
    // that declare or define parameters are treated differently than those that
    // simply use parameters.
    bool isDefOrDecl = false;

    // Check if this operation defines a parameter scope.
    auto result = WalkResult::advance();
    if (auto decl = dyn_cast<DeclInterface>(op)) {
      // Check if this is a nested scope.
      if (scope->getParentOp() != decl) {
        // Walk over nested scopes. Defer processing of nested scopes until
        // after this scope has been analyzed.
        for (Region &r : decl->getRegions())
          nestedDecls.push_back(&r);
        result = WalkResult::skip();
      } else {
        // Record parameter declarations for the top-level declaration.
        auto recordDeclWrapper = [&](ParamDeclAttr decl) -> LogicalResult {
          isDefOrDecl = true;
          return recordDecl(*this, decl, op, *scope);
        };
        // A declaration declares input and/or result parameters.
        for (ParamDeclAttr paramDecl : decl.getAllParams())
          if (failed(recordDeclWrapper(paramDecl)))
            return failure();
        // The input parameters are defined by the declaration.
        for (auto [idx, inputParam] : llvm::enumerate(decl.getInputParams())) {
          FailureOr<ParamDefinition *> def = recordDef(*this, inputParam, decl);
          if (failed(def))
            return failure();
          (*def)->index = idx;
        }
      }
    }

    // Check if this operation implements the parametric operation interface.
    if (auto itf = dyn_cast<ParamOpInterface>(op);
        itf && itf != scope->getParentOp()) {
      // Check declarations.
      bool hadError = false;
      itf.walkDeclarations([&](ParamDeclAttr decl) {
        isDefOrDecl = true;
        if (failed(recordDecl(*this, decl, op, *scope)))
          hadError = true;
      });
      if (hadError)
        return WalkResult::interrupt();

      // Check definitions.
      ssize_t index = 0;
      itf.walkDefinitions([&](ParamDeclAttr decl, const ParamDefValue &value) {
        FailureOr<ParamDefinition *> def = recordDef(*this, decl, op);
        if (failed(def)) {
          hadError = true;
          return;
        }
        isDefOrDecl = true;
        (*def)->index = index++;
        bool unusedHasConstExpr = false;
        size_t unusedRequiredSignatureDepth = 0;
        for (Attribute expr : value.exprs) {
          c.collectUsesFromAttr(expr, (*def)->uses, unusedHasConstExpr,
                                unusedRequiredSignatureDepth);
        }
        // If the definition of this parameter depends on a region, defer
        // processing of the nested region uses.
        if (!value.regions.empty())
          regionValues.emplace_back(decl, value.regions);
      });
      if (hadError)
        return WalkResult::interrupt();
    }

    // Collect parameter uses from this operation.
    collectUses(*this, c, op, isDefOrDecl);
    return result;
  };

  // Process the scope's parent op - don't recurse because the parent op might
  // have multiple regions.
  if (processOp(scope->getParentOp()).wasInterrupted())
    return failure();

  // Now walk the scope and not sibling regions!
  WalkResult result = scope->walk<mlir::WalkOrder::PreOrder>(processOp);
  if (result.wasInterrupted())
    return failure();

  // Check the validity of all parameter references.
  for (auto &[op, uses] : opUses) {
    for (ParamDeclRefAttr use : uses) {
      auto it = decls.find(use.getName());
      // Handle parameters without a declaration.
      if (it == decls.end()) {
        // If we're not verifying, assume this is a capture.
        if (!evaluationContext) {
          usesFromAbove.insert(use);
          continue;
        }
        // Ensure that the use refers to a parameter that was declared.
        return op->emitOpError("invalid use of parameter with no declaration ")
               << use.getName();
      }

      // Check that the type of the parameter references matches the type of its
      // declaration.
      if (evaluationContext && !isEqualCanon(it->second.type, use.getType())) {
        return (op->emitOpError("reference to parameter ")
                << use.getName() << " with incorrect type " << use.getType())
                   .attachNote(it->second.declOp->getLoc())
               << "parameter defined with type " << it->second.type;
      }

      // If the declaration of the parameter is outside the current scope,
      // indicate this as a parameter use from above.
      if (!scope->isAncestor(it->second.scope))
        usesFromAbove.insert(use);
    }
  }

  // Make sure every parameter declared in this scope has a definition.
  for (auto &[param, decl] : decls) {
    if (!scope->isAncestor(decl.scope))
      continue;
    auto it = defs.find(param);
    if (it == defs.end())
      return emitError(decls.find(param)->second.declOp->getLoc(), "parameter ")
             << param << " has no definition";
  }

  // If an error was encountered while collecting parameters, bail out here.
  if (c.hadError)
    return failure();

  // Process all nested scopes.
  for (Region *nestedScope : nestedDecls) {
    ParameterUseDefGraph nested(*nestedScope);
    // Propagate the current declarations into the nested scope.
    nested.decls = decls;
    if (failed(nested.calculateOrVerify(evaluationContext, cache)))
      return failure();

    // If there were no uses from above, notify the nested declaration that it
    // is isolated. Do not do this during verification.
    if (nested.usesFromAbove.empty()) {
      auto decl = cast<DeclInterface>(nestedScope->getParentOp());
      decl.notifyKnownIsolatedFromAbove(nestedScope->getRegionNumber());
    }

    // Bubble up the nested scopes and all nested uses from above.
    for (auto &[nestedParameterScope, nestedParameterGraph] :
         nested.nestedScopes) {
      nestedScopes.try_emplace(nestedParameterScope,
                               std::move(nestedParameterGraph));
    }
    nested.nestedScopes.clear();
    for (ParamDeclRefAttr use : nested.usesFromAbove) {
      auto it = decls.find(use.getName());
      if (it == decls.end() || !scope->isAncestor(it->second.scope))
        usesFromAbove.insert(use);
    }

    nestedScopes.try_emplace(nestedScope, std::move(nested));
  }

  // The parameter uses that a region parameter declaration depend on are
  // computed after the walk, since the walk is performed pre-order. Now that
  // we have the uses in the nested scopes, compute their dependent parameters.
  for (auto &[decl, regions] : regionValues) {
    ParamDefinition &def = defs.find(decl.getName())->second;
    for (Region *region : regions) {
      auto it = nestedScopes.find(region);
      assert(it != nestedScopes.end() && "didn't visit nested body?");
      llvm::append_range(def.uses, it->second.usesFromAbove);
    }
  }

  // Check that there is a definite partial ordering between parameters and emit
  // errors for any encountered cycles. Compute the new order.
  SmallVector<StringAttr> paramSolveOrder;
  for (auto sccIt = llvm::scc_begin(this); !sccIt.isAtEnd(); ++sccIt) {
    if (sccIt.hasCycle()) {
      emitCycleError(*this, *sccIt);
      return failure();
    }

    assert(sccIt->size() == 1 && "non-cyclic regions should have one node");
    StringAttr param = sccIt->front().param;
    if (param) {
      Region *paramScope = decls.find(param)->second.scope;
      if (paramScope && scope->isAncestor(paramScope))
        paramSolveOrder.push_back(param);
    }
  }
  params = std::move(paramSolveOrder);

  return success(!result.wasInterrupted());
}

void ParameterUseDefGraph::calculate(ParameterCollector::Analysis &cache) {
  [[maybe_unused]] LogicalResult result = calculateOrVerify(nullptr, cache);
  assert(succeeded(result) && "IR should be legal here!");
}

LogicalResult
ParameterUseDefGraph::verify(SymTabEvaluationContext *evaluationContext,
                             ParameterCollector::Analysis &cache) {
  return calculateOrVerify(evaluationContext, cache);
}

ParameterUseDefGraph ParameterUseDefGraph::copy(const IRMapping &map) const {
  VerboseCompilerTimeTraceScope traceScope("ParameterUseDefGraph::copy");

  // Note that we use map.lookupOrDefault here because only a subgraph might
  // have been copied, so we don't necessarily have the op/block in the
  // IRMapping.

  auto remapRegion = [&](Region *region) {
    // Look up the first remapped block in the region, and return that region.
    assert(!region->empty());
    Block *remappedBlock = map.lookupOrDefault(&region->front());
    return remappedBlock->getParent();
  };

  ParameterUseDefGraph out(*remapRegion(scope));

  // Copy over decls and defs.
  for (auto [name, decl] : decls)
    out.decls[name] = ParamDeclaration{
        decl.type, map.lookupOrDefault(decl.declOp), remapRegion(decl.scope)};
  for (auto [name, def] : defs)
    out.defs[name] = ParamDefinition{def.value, def.index,
                                     map.lookupOrDefault(def.defOp), def.uses};

  // Copy over param ops.
  for (auto paramOp : paramOps)
    out.paramOps.push_back(map.lookupOrDefault(paramOp));

  // These are trivial to copy over.
  out.params = params;
  out.usesFromAbove = usesFromAbove;

  // Copy over the op uses.
  for (auto [op, useVector] : opUses)
    out.opUses[map.lookupOrDefault(op)] = useVector;

  // Copy the remapped nested decls.
  for (auto nestedDecl : nestedDecls)
    out.nestedDecls.push_back(remapRegion(nestedDecl));

  // And finally, for each nested scope, we'll have to do the same thing.
  for (auto &[decl, graph] : nestedScopes)
    out.nestedScopes.try_emplace(remapRegion(decl), graph.copy(map));

  return out;
}

void ParameterUseDefGraph::dump() const {
  // Nested graphs are actually contained on the top-level map.
  const ParameterUseDefGraph &top = *this;

  // Dump the graphs in-order.
  std::deque<std::pair<const ParameterUseDefGraph *, unsigned>> bfsDump;
  bfsDump.emplace_back(this, 0);

  while (!bfsDump.empty()) {
    mlir::raw_indented_ostream os(llvm::errs());
    auto [cur, depth] = bfsDump.front();
    bfsDump.pop_front();
    os.indent(depth * 2);

    os << "Decls: ";
    llvm::interleaveComma(cur->decls, os, [&](const auto &nameAndDecl) {
      os << nameAndDecl.first;
    });
    os << "\n";

    os << "Defs: ";
    llvm::interleaveComma(
        cur->defs, os, [&](const auto &nameAndDef) { os << nameAndDef.first; });
    os << "\n";

    os << "Uses: ";
    llvm::interleaveComma(cur->opUses, os, [&](const auto &opAndUses) {
      llvm::interleaveComma(opAndUses.second, os,
                            [&](const auto &use) { os << use.getName(); });
    });
    os << "\n";

    for (Region *nested : cur->nestedDecls)
      bfsDump.emplace_back(&top.nestedScopes.at(nested), depth + 1);
  }
}

static void collectAllParamOpsHelper(ParameterUseDefGraph &graph,
                                     DenseSet<Operation *> &set,
                                     bool &hasParams) {
  set.insert(graph.paramOps.begin(), graph.paramOps.end());
  hasParams |= !graph.params.empty();
  for (auto &scope : graph.nestedScopes)
    collectAllParamOpsHelper(scope.second, set, hasParams);
}

FunctionParameterUseDefGraph::FunctionParameterUseDefGraph(Region &scope)
    : ParameterUseDefGraph(scope) {}

FunctionParameterUseDefGraph::FunctionParameterUseDefGraph(Region *scope)
    : ParameterUseDefGraph(scope) {}

void FunctionParameterUseDefGraph::calculate(
    ParameterCollector::Analysis &cache) {
  ParameterUseDefGraph::calculate(cache);
  hasParams = false;
  collectAllParamOpsHelper(*this, paramOpsSet, hasParams);
}
