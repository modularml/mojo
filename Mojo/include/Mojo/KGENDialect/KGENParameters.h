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
// This file contains helper functions for working with KGEN parameter
// expressions and declarations.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPARAMETERS_H
#define KGEN_KGENPARAMETERS_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/ParameterReplacer.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// IndexRefRemapper
//===----------------------------------------------------------------------===//

/// Utility class for remapping named parameter references to index references,
/// see DCRTODS.
class IndexRefRemapper : public IndexParameterReplacer<IndexRefRemapper> {

public:
  IndexRefRemapper() = default;

  /// Populate the remapper with named input and result parameters.
  IndexRefRemapper(ArrayRef<ParamDeclAttr> inputParams, size_t offset = 0);
  IndexRefRemapper(ArrayRef<ParamDeclRefAttr> inputParams, size_t offset = 0);

  /// Append a parameter declaration to the remapper.
  void appendParamDecl(ParamDeclAttr paramDecl);

private:
  using Base = IndexParameterReplacer<IndexRefRemapper>;

  // CRTP methods.
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<IndexRefRemapper>;

  /// Mapping from parameter reference to an index.
  DenseMap<StringAttr, size_t> mapping;
  /// The index offset of references to root input parameters.
  size_t offset;
};

//===----------------------------------------------------------------------===//
// ParamRefRemapper
//===----------------------------------------------------------------------===//

// ParamRefRemapper converts parameter references by index (e.g., *(0,1))
// to references by name (e.g., "b"): The reverse of IndexRefRemapper.
//
// Problem:
// In Mojo parametric functions, parameter types can refer to earlier parameters
// using indices. For example, in "fn f[T: Baz, b: T]", the type of 'b' refers
// to 'T' via an index. When creating function types outside this context, we
// need named references instead.
//
// Example:
// Input
// Parameter types: [!lit.trait<@Baz>, !kgen.param<*(0,0)>]
// Parameter names: ["T", "b"]
//
// Output (canonical types):
// {"T": !lit.trait<@Baz>, "b": !kgen.param<"T">}
//
// Solution:
// We recursively traverse attributes, replacing ParamIndexRefAttr with
// ParamDeclRefAttr using a map from indices to parameter declarations.
// The recursion terminates because MLIR attributes are directed acyclic graphs.
struct ParamRefRemapper : public IndexParameterReplacer<ParamRefRemapper> {
  using Base = IndexParameterReplacer<ParamRefRemapper>;
  ParamRefRemapper() = default;
  ParamRefRemapper(ArrayRef<StringAttr> declNames) {
    for (auto n : declNames)
      parameters.push_back(n);
  }
  ParamRefRemapper(ArrayRef<ParamDeclAttr> declarations) {
    for (auto p : declarations)
      parameters.push_back(p.getName());
  }

  void appendParamDecl(ParamDeclAttr newDecl) {
    parameters.push_back(newDecl.getName());
  }

  Attribute tryReplace(Attribute attr, size_t depth) {
    auto indexRef = dyn_cast<ParamIndexRefAttr>(attr);
    if (!indexRef || indexRef.getDepth() != depth)
      return nullptr;
    if (indexRef.getIndex() >= parameters.size())
      return nullptr;

    StringAttr paramName = parameters[indexRef.getIndex()];
    Type mappedType = Base::replace(indexRef.getType());
    return ParamDeclRefAttr::get(paramName.strref(), mappedType);
  }
  Type tryReplace(Type t, size_t) { return {}; }
  SmallVector<StringAttr> parameters;
};

//===----------------------------------------------------------------------===//
// IndexDepthAdjuster
//===----------------------------------------------------------------------===//

/// This class is used exclusively to adjust the depths of index references that
/// reference signatures outside the current scope. See STCHDDDOS for more and
/// for why this is needed.
class IndexDepthAdjuster : public IndexParameterReplacer<IndexDepthAdjuster> {
public:
  explicit IndexDepthAdjuster(int64_t adjustDepth,
                              bool onlyAdjustIndexRef = false)
      : adjustDepth(adjustDepth), onlyAdjustIndexRef(onlyAdjustIndexRef) {}

private:
  // CRTP methods.
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<IndexDepthAdjuster>;

  /// Adjust the depth of index references when remapping.
  int64_t adjustDepth;

  /// Only adjust the index.ref attr, not a general index ref interface.
  bool onlyAdjustIndexRef;
};

//===----------------------------------------------------------------------===//
// EscapingReferenceFinder
//===----------------------------------------------------------------------===//

/// Detects whether an attribute or type contains a `ParamIndexRefAttr` that
/// refers to a parameter-decl *outside* the walked value -- an "escaping"
/// reference. The walk is depth-aware (see PSTIAIRAID), so a ref counts as
/// escaping when `getDepth() >=` the current signature depth. Such a reference
/// means the value is still abstract in the current scope, so it must not be
/// treated as fully concrete.
struct EscapingReferenceFinder
    : public IndexParameterReplacer<EscapingReferenceFinder> {
  Attribute tryReplace(Attribute attr, size_t depth) {
    if (auto ref = dyn_cast<ParamIndexRefAttr>(attr);
        ref && ref.getDepth() >= depth) {
      escapingReference = true;
      return attr;
    }
    return nullptr;
  }
  Type tryReplace(Type, size_t) { return {}; }

  /// Returns whether `value` contains an escaping parameter reference.
  template <typename T>
  static bool check(T value) {
    EscapingReferenceFinder finder;
    finder.replace(value);
    return finder.escapingReference;
  }

  bool escapingReference = false;
};

//===----------------------------------------------------------------------===//
// ParameterCollector
//===----------------------------------------------------------------------===//

class ParameterCollector {
public:
  /// The parameter collector contains a cache of parameter-less attributes and
  /// types that is valid throughout the lifetime of an MLIR context. This
  /// analysis allows the cache to be preserved across passes.
  struct Analysis {
    Analysis(Operation *op = nullptr) {}

    /// This analysis can never be invalid.
    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
      return false;
    }

    /// Cached facts for parameterless attributes and types.
    struct ParameterlessInfo {
      bool hasCtxEvalExpr = false;
      /// Intrinsic requirement of this sub-expression: minimum number of
      /// surrounding signature scopes needed for it to be valid.
      size_t requiredSignatureDepth = 0;
    };

    /// Types and attributes contained in this map are known to have no
    /// parameter uses as sub-elements. They are mapped to whether there is an
    /// unresolved parameter operator and the intrinsic required signature depth
    /// for the sub-elements to be valid.
    DenseMap<const void *, ParameterlessInfo> parameterLess;
  };

  /// Create a parameter collector with a collection cache.
  ParameterCollector(Analysis &cache) : cache(cache) {}

  virtual ~ParameterCollector() = default;

  /// Scan the specified attribute and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses into `uses`.
  /// `hasConstExpr` is set if unresolved param expressions are seen.
  /// `requiredSignatureDepth` is set to the intrinsic minimum number of
  /// surrounding signature scopes needed for this expression to be valid.
  void collectUsesFromAttr(Attribute attr,
                           SmallVectorImpl<ParamDeclRefAttr> &uses,
                           bool &hasConstExpr, size_t &requiredSignatureDepth);

  /// Scan the specified type and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses into `uses`.
  /// `hasConstExpr` is set if unresolved param expressions are seen.
  /// `requiredSignatureDepth` is set to the intrinsic minimum number of
  /// surrounding signature scopes needed for this type to be valid.
  void collectUsesFromType(Type type, SmallVectorImpl<ParamDeclRefAttr> &uses,
                           bool &hasConstExpr, size_t &requiredSignatureDepth);

private:
  void collectUsesFromAttrImpl(Attribute attr,
                               SmallVectorImpl<ParamDeclRefAttr> &uses,
                               bool &hasCtxEvalExpr,
                               size_t &requiredSignatureDepth);
  void collectUsesFromTypesImpl(Type type,
                                SmallVectorImpl<ParamDeclRefAttr> &uses,
                                bool &hasCtxEvalExpr,
                                size_t &requiredSignatureDepth);

  /// The first time we encounter an attribute with a reference to an
  /// out-of-line declaration, verify it.
  virtual void verifyRefAttr(DeclRefAttrInterface refAttr) {}

  /// When we encounter a StructType, check that its parameter bindings match
  /// the parameter declarations on the type declaration.
  virtual void verifyRefType(StructTypeInterface refType) {}

  /// Optionally perform verification and emit an error.
  virtual void
  maybeVerify(function_ref<LogicalResult(function_ref<InFlightDiagnostic()>)>
                  verifyFn) {}

  /// Attributes and types are memoized and exist in tree structures with reuse:
  /// naively scanning them can lead to exponential compile time behavior.  As
  /// such, we memoize the attributes and types we've already checked that we
  /// know have no parameters in them and whether the paramless attributes are
  /// constant parameter expressions. The memoized required signature depth is
  /// intrinsic to each sub-expression and context-independent.
  Analysis &cache;

  /// An internal stack of scoped parameter types representing the input param
  /// types of the current nested signatures.
  SmallVector<
      SmartVariant<ParameterScopeAttrInterface, ParameterScopeTypeInterface>>
      signatures;
};

//===----------------------------------------------------------------------===//
// ParamIndexRefAttrFinder
//===----------------------------------------------------------------------===//

// Class to determine if there are any parameter references in the attribute
// value.
class ParamIndexRefAttrFinder {
public:
  bool hasReferences(TypedAttr value) {
    return findOneReference(value).has_value();
  }
  bool hasReferences(Type type) { return findOneReference(type).has_value(); }

  /// This scans the specified value or type for parameter references.  It runs
  /// nullopt if there are none, or the index of the first that it encounters.
  std::optional<size_t> findOneReference(TypedAttr value);
  std::optional<size_t> findOneReference(Type type);

private:
  // Depth aware cache to avoid visiting the same attr twice.
  DenseMap<std::pair<size_t, const void *>, ssize_t> cached;
};

//===----------------------------------------------------------------------===//
// ParameterUseDefGraph
//===----------------------------------------------------------------------===//

/// The definition of a parameter. The parameter definition contains its value
/// and the operation which contains the value attribute. Not all declared
/// parameters have definitions. Input parameters to a function, for example,
/// have no definition within the function, and are treated as leaves.
struct ParamDefinition {
  /// If the expression that defines the parameter can be narrowed to a simple
  /// attribute, this field will contain that expression.
  Attribute value;
  /// The index of the parameter into the operation's result parameters. This is
  /// -1 for a parameter that is not a result parameter.
  ssize_t index = -1;
  /// The defining operation.
  Operation *defOp = nullptr;
  /// The dependent parameters of the definition.
  SmallVector<ParamDeclRefAttr> uses;
};

/// The declaration of a parameter. The parameter declaration contains the type
/// of the parameter and the operation that declares it. A parameter can be
/// declared and defined by different operations: a return parameter, for
/// example, is declared by the surrounding function but defined by its return
/// operation.
struct ParamDeclaration {
  /// The type of the parameter as it was declared.
  Type type;
  /// The operation that declares the parameter.
  Operation *declOp;
  /// The parent declaration scope.
  Region *scope;
};

/// This class defines the use-def graph for parameters. There are two types of
/// parameter uses: operations and parameter definitions. The use-def graph of
/// parameter declarations and definitions is of most interest: there can be
/// no cycles in this graph.
///
/// The elaborator must first resolve this graph by providing values for the
/// leaf nodes (input parameters) and computing all the parameter definition
/// expressions to a simple constant value. Then, all operations that use
/// parameters (in an attribute, type, or location) can be concretized in any
/// order.
struct ParameterUseDefGraph {
  ParameterUseDefGraph(Region &scope) : scope(&scope) {}
  ParameterUseDefGraph(Region *scope) : scope(scope) {}
  virtual ~ParameterUseDefGraph() = default;

  /// Map of parameter name to its declaration.
  DenseMap<StringAttr, ParamDeclaration> decls;
  /// Map of parameter name to its definition.
  DenseMap<StringAttr, ParamDefinition> defs;

  /// The scope at which this graph is computed.
  Region *scope;

  /// A list of parametric operations. These are the operations that must be
  /// concretized by the elaborator once all parameters in the scope have been
  /// computed to simple constant values.
  std::vector<Operation *> paramOps;

  /// A list of all parameters defined within the scope.
  SmallVector<StringAttr> params;

  /// These are the parameter uses in the current scope that were captured from
  /// a higher scope.
  llvm::SetVector<ParamDeclRefAttr> usesFromAbove;

  /// Track the operations that reference parameters. Use this information to
  /// diagnose references to parameters without declarations.
  llvm::MapVector<Operation *, SmallVector<ParamDeclRefAttr>> opUses;

  /// A list of nested parameter scopes.
  SmallVector<Region *> nestedDecls;

  /// A map of nested scopes to their use-def graph. Note that when calculating
  /// the use-def graph, the top-level use-def graph contains the mappings for
  /// ALL the nested scopes. The graphs of nested scopes must be looked up on
  /// the top-level graph.
  DenseMap<Region *, ParameterUseDefGraph> nestedScopes;

  /// Compute the parameter declarations, definitions, and uses within the
  /// provided parameter declaration scope. If the the root scope is not
  /// isolated from above, the use-def graph expects to be primed with the
  /// parent scope's declarations before this function is called.
  virtual void calculate(ParameterCollector::Analysis &cache);

  /// Verify the validity of the parameter declarations, uses, and definitions
  /// within the current scope.
  LogicalResult verify(SymTabEvaluationContext *evaluationContext,
                       ParameterCollector::Analysis &cache);

  /// Copy this graph into a new instance, remapping all the operations using
  /// `map`.
  ParameterUseDefGraph copy(const IRMapping &map) const;

  /// Print the graph to llvm::errs().
  void dump() const;

  /// Disable implicit copying.
  ParameterUseDefGraph(const ParameterUseDefGraph &) = delete;
  ParameterUseDefGraph &operator=(const ParameterUseDefGraph &) = delete;
  ParameterUseDefGraph(ParameterUseDefGraph &&) = default;
  ParameterUseDefGraph &operator=(ParameterUseDefGraph &&) = default;

private:
  /// Calculate the parameter use-def graph and perform verification if a symbol
  /// table is provided.
  LogicalResult calculateOrVerify(SymTabEvaluationContext *evaluationContext,
                                  ParameterCollector::Analysis &cache);
};

struct FunctionParameterUseDefGraph : ParameterUseDefGraph {
  FunctionParameterUseDefGraph(Region &scope);
  FunctionParameterUseDefGraph(Region *scope);
  virtual ~FunctionParameterUseDefGraph() {}

  /// A flattened set of parametric operations includes ops in all nested
  /// regions. This is used to help speed up interpreting parametric functions.
  DenseSet<Operation *> paramOpsSet;

  /// Quick flag indicating whether a region has parameters or not.
  /// This is used to help speed up interpreting parametric functions.
  bool hasParams = true;

  /// Compute the parameter declarations, definitions, and uses within the
  /// provided parameter declaration scope. If the the root scope is not
  /// isolated from above, the use-def graph expects to be primed with the
  /// parent scope's declarations before this function is called.
  void calculate(ParameterCollector::Analysis &cache) override;
};

} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
