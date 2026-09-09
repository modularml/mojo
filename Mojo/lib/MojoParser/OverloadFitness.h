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
// This file declares the components for overload fitness evaluation.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_OVERLOADFITNESS_H
#define KGEN_MOJOPARSER_OVERLOADFITNESS_H

#include "InferenceState.h"
#include "Mojo/MojoParser/CallOperands.h"

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "Mojo/MojoParser/MojoDiags.h"

namespace M::KGEN {
class PogListAttr;
}

namespace M::KGEN::LIT {
class CallOperands;
class FnTypeGeneratorType;

/// This struct indicates whether a signature can be successfully applied to a
/// parameter binding and argument list. If so, it keeps track of several
/// metrics that allow comparing different candidates, and if not, it indicates
/// the reason for the mismatch.
class OverloadFitness {
public:
  OverloadFitness(OverloadFitness &&other) = default;

  ~OverloadFitness() {
    if (diag)
      takeDiag().abandon();
  }

  /// Return the parameter bindings if the candidate is valid.
  const VerifiedParamBindings &getParamBindings() const {
    assert(getValidity() >= Validity::kFunctionConstraintInconclusive);
    return paramBindings;
  }

  /// Return the number of implicit conversions if the candidate is valid.
  size_t getNumImplicitConversions() const {
    assert(getValidity() >= Validity::kFunctionConstraintInconclusive);
    return payload.numImplicitConversions;
  }

  /// Returns whether this fitness is strictly better than another one.
  bool isBetter(const OverloadFitness &other) const;

  /// Consume the diagnostic if the candidate is invalid.
  MojoInflightDiag takeDiag() {
    assert(getValidity() == Validity::kInvalid);
    return std::move(*diag);
  }

  /// Consume the diagnostic if the candidate is invalid.
  const MojoInflightDiag &getDiag() const {
    assert(getValidity() == Validity::kInvalid);
    return *diag;
  }

  /// Return the unprovable constraints if the candidate is inconclusive.
  ArrayRef<ConstraintAttr> getUnprovableConstraints() const {
    assert(isInconclusive());
    return unprovableConstraints;
  }

  // Ordered from worst to best validity.
  enum class Validity {
    kInvalid,                        // This is definitely invalid.
    kFunctionConstraintInconclusive, // Unprovable function constraints.
    kValid,                          // This is definitely valid.
  };

  /// Return whether the candidate was valid.
  Validity getValidity() const {
    if (!unprovableConstraints.empty())
      return Validity::kFunctionConstraintInconclusive;
    return diag ? Validity::kInvalid : Validity::kValid;
  }

  /// Return whether the candidate is any kind of inconclusive.
  bool isInconclusive() const {
    return getValidity() == Validity::kFunctionConstraintInconclusive;
  }

  /// Determine whether the specified signature can be invoked with the
  /// parameter bindings specified in `callable` and the arguments specified in
  /// `callOperands`.
  ///
  /// The 'funcIfDirect' member is set if this is a direct call, or null if
  /// indirect.  It can be used to tune diagnostics.
  static OverloadFitness evaluate(FnTypeGeneratorType signature,
                                  ASTDecl *funcIfDirect,
                                  const OverloadSet &callable,
                                  const CallOperands &callOperands,
                                  bool allowImplicitConversions);

  /// Determine whether the specified signature can be invoked with the
  /// parameter bindings specified in `callable`.
  static OverloadFitness
  evaluate(ASTDecl *candidate, const OverloadSet &callable, PValue selfPValue);

  /// Build a fitness for a candidate already known to be valid. For callers
  /// that decide validity themselves.
  static OverloadFitness valid(VerifiedParamBindings paramBindings) {
    return OverloadFitness(std::move(paramBindings), {});
  }

  /// Build a fitness for a candidate that is neither valid nor definitively
  /// rejected: `unprovableConstraints` could be neither proven nor disproven.
  static OverloadFitness
  constraintInconclusive(VerifiedParamBindings paramBindings,
                         ArrayRef<ConstraintAttr> unprovableConstraints) {
    OverloadFitness fitness(std::move(paramBindings), {});
    fitness.unprovableConstraints.assign(unprovableConstraints.begin(),
                                         unprovableConstraints.end());
    return fitness;
  }

  /// Return the set of args that need to be emitted to MValues to select this
  /// candidate.
  const OperandsNeedingOriginsList &getOperandsNeedingOrigins() const {
    return operandsNeedingOrigins;
  }

private:
  /// For valid candidates, this defines the parameter bindings to use.
  VerifiedParamBindings paramBindings;
  /// The diagnostic for invalid candidates, or null for valid ones.
  std::optional<MojoInflightDiag> diag = std::nullopt;
  /// Any unprovable constraints.
  SmallVector<ConstraintAttr> unprovableConstraints;

  /// If this candidate requires any arguments to be emitted (from a PValue or
  /// SValue to an MValue) so a origin can be inferred, the argument is tracked
  /// in this list.
  const OperandsNeedingOriginsList operandsNeedingOrigins;

  /// Describes the metrics that can be used to compare candidates.
  struct Payload {
    /// The number of implicit conversions required.  Normal implicit
    /// conversions count as 2 each, nonmaterializable value conversions count
    /// as 1.
    size_t numImplicitConversions = 0;

    /// For each mismatch in "preferred" argument convention, penalize the
    /// overload. This is to resolve ambiguities that can arise from synthesized
    /// thunks for converting calling conventions.
    size_t numMismatchedConventions = 0;
    /// Whether the candidate has a (non-empty) variadic argument.
    bool passesVarArgArgument = false;
  } payload;

  /// Constructor for invalid candidates (kInvalid).
  OverloadFitness(MojoInflightDiag &&diag) : diag(std::move(diag)) {}
  /// Constructor for valid candidates (kValid).
  /// To create a kFunctionConstraintInconclusive fitness, just insert the
  /// constraints into `unprovableConstraints`.
  OverloadFitness(VerifiedParamBindings paramBindings,
                  OperandsNeedingOriginsList &&operandsNeedingOrigins)
      : paramBindings(std::move(paramBindings)),
        operandsNeedingOrigins(std::move(operandsNeedingOrigins)) {}
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_OVERLOADFITNESS_H
