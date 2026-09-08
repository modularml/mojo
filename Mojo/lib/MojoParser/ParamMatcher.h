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
// This file provides the ParamMatcher for Mojo parsers to "co-iterate" two
// types/parameters in order to extract parameter value to be inferred.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARAMMATCHER_H
#define KGEN_MOJOPARSER_PARAMMATCHER_H

#include "InferenceState.h"

#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Support/ADT/SmartVariant.h"

namespace M::KGEN::LIT {

class ExprNode;
class SharedState;

//===----------------------------------------------------------------------===//
// MatchFailure
//===----------------------------------------------------------------------===//

/// These are the different failure modes that we know happen.
struct MatchFailure {
  /// This failure happens when a parameter is found of the wrong type.
  struct TypeConflict {
    size_t paramIdx; // TODO: Render this name.
    ASTType paramType, argParamType;
    /// From the upcast check that rejected this match, absent when no upcast
    /// was tried or it had nothing to report.
    std::optional<ConstraintResult> upCastFailure;
  };

  /// This failure happens when a parameter is inferred to two different values.
  struct ValueConflict {
    size_t paramIdx;
    TypedAttr v1, v2;
  };

  /// This failure happens when a parameter is explicitly unbound by the user,
  /// but then inferable from another parameter.
  struct UnboundButInferrable {
    size_t unboundParamIdx;
  };

  /// This failure happens when merge* is called, but the expected type/value
  /// still has an unresolved dependent type which can't be inferred.
  struct DependsOnUnresolved {
    size_t paramIdx;
  };

  /// This failure hasn't been categorized yet.
  /// FIXME: Remove this.
  struct Unclassified {};

  template <typename Failure>
  MatchFailure(Failure info) : info(info) {}

  /// Append a note explaining this failure.
  void addExplanation(MojoInflightDiag &diag) const;

  /// If this failure is due to an unresolved parameter, return the index of the
  /// parameter.
  std::optional<size_t> getIfDependentOnUnresolved() const {
    if (isa<DependsOnUnresolved>(info)) {
      return cast<DependsOnUnresolved>(info).paramIdx;
    }
    return std::nullopt;
  }

  bool isUnboundButInferrable() const {
    return isa<UnboundButInferrable>(info);
  }

private:
  SmartVariant<TypeConflict, ValueConflict, DependsOnUnresolved, Unclassified,
               UnboundButInferrable>
      info;
};

/// This class implements logic for match parameters between an actual value
/// present at a call site, and expected value in the callee.  The former value
/// is always concrete, but the latter may contain symbolic parameters from the
/// callees signature that we're trying to infer.  It is also possible this
/// candidate is completely invalid!  The result of match is one of several
/// cases:
///
/// - Match: the parameters match.
/// - Error: the parameters do not match.  The error code is set to indicate
///   the first reason.
/// - Retry: the parameters matched and led to a parameter getting inferred! The
///   parameter # is set to indicate which one.
///
/// You might wonder why we need to retry matching from a root when inferring a
/// parameter.  It turns out that some values can only be matched after
/// simplification.  Consider a situation like:
///
///    struct S[a: Int, b: Int]:
///    def take[v: Int](s: S[v, v+1]):
///
/// In this case, we *must* stop after inferring the value of `v`, backtrack
/// up call call stack, and then substitute the value of `v` into the expected
/// type.  If we don't do this, we won't be able to match calls that pass,
/// A[1, 2] because the "v+1=2" knowledge can only be had by substituting which
/// allows the Int addition to fold.
class ParamMatcher {
public:
  ParamMatcher(const ExprNode *expr, InferenceState &state,
               bool allowImplicitConversion);
  ~ParamMatcher() {}

  /// This is set when an error is encountered.
  std::optional<MatchFailure> failureReason;

  static TypedAttr preAlignParam(TypedAttr attr);

  LogicalResult matchTypes(Type actualType, Type expectedType);
  LogicalResult matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);
  LogicalResult matchFunctionTypes(FnTypeGeneratorType actual,
                                   FnTypeGeneratorType expected);
  LogicalResult matchSingleEltStruct(TypedAttr actual, TypedAttr expected);

  /// When matching parameters, we sometimes try different options.  This object
  /// gives us a scoped way to revert() a series of matches that didn't work
  /// out - making sure to remove any bindings that were tentatively inferred.
  struct FailableScope {
    FailableScope(ParamMatcher &matcher);
    FailableScope(const FailableScope &other) = default;

    /// This reverts the error code and any matches that were formed between the
    /// creation of the FailableScope and its call.
    void revert();

    /// Save the current state of the matcher in an error state so we can
    /// reapply it later.
    struct SavedStateTy {
      MatchFailure failureReason;
      SmallVector<TypedAttr, 8> values;
      size_t numImplicitConversions;
    };
    SavedStateTy saveState() {
      assert(matcher.failureReason && "not in error state");
      return {
          *matcher.failureReason,
          SmallVector<TypedAttr, 8>(matcher.state.evaluator.getIndexBindings()),
          matcher.state.numImplicitConversions};
    }
    static void restore(SavedStateTy savedState, ParamMatcher &matcher) {
      matcher.failureReason = savedState.failureReason;
      for (auto [idx, binding] : llvm::enumerate(savedState.values))
        matcher.state.evaluator.overwriteIndexBinding(idx, binding);
      matcher.state.numImplicitConversions = savedState.numImplicitConversions;
    }

  private:
    ParamMatcher &matcher;
    llvm::BitVector inferredIdx;
    size_t numImplicitConversions;
  };

private:
  LogicalResult error(MatchFailure &&reason) {
    failureReason = std::move(reason);
    return failure();
  }

  /// This is how many signature types deep inference is inside parameter
  /// expressions and determines which index references we match against.
  ///
  /// As we search for param-refs, recursively, we'll be recursing past
  /// `FnTypeGeneratorType`s (and other `ParameterScopeTimeInterface`s),
  /// which changes what param-ref depths we're watching for; the param-refs'
  /// depths would be greater (have to reach further outward so to speak, past
  /// more generator types) to reference param-decls in the
  /// ParameterInferenceState's original scope. paramIndexRefDepth tracks that
  /// number.
  ///
  /// In other words, these paramIndexRefDepth adjustments are for
  /// depth-aware searching, see PSTIAIRAID.
  size_t paramIndexRefDepth = 0;

  /// This is the expression we're inferring within.
  const ExprNode *const expr;
  InferenceState &state;
  SharedState &shared;

  // Whether we allow implicit conversion when matching, e.g., function type
  // conversion.
  bool allowImplicitConversions;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMMATCHER_H
