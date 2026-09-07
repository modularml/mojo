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

#include "Mojo/MojoParser/Constraints.h"
#include "MojoUtils.h"
#include "ParamBindings.h"
#include "Traits.h"

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/DeclSignaturePrinter.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

ConstraintResult
LIT::makeConstraintResult(TriBool verdict,
                          SmallVector<ConstraintAttr, 2> constraints) {
  if (verdict.isTrue())
    return ConstraintResult::yes();
  if (verdict.isFalse())
    return ConstraintResult::no(std::move(constraints));
  return ConstraintResult::unknown(std::move(constraints));
}

void LIT::attachConstraintNotes(MojoInflightDiag &diag,
                                ArrayRef<ConstraintAttr> constraints,
                                StringRef kind) {
  for (ConstraintAttr constraint : constraints) {
    diag.attachNote(constraint.getLoc()) << kind << " constraint";
    if (StringAttr message = constraint.getMessage())
      diag << ": " << message.getValue();
  }
}

void LIT::attachConstraintNotes(MojoInflightDiag &diag,
                                const ConstraintResult &result) {
  if (result.isNo())
    attachConstraintNotes(diag, result.getNo(), "failed");
  else if (result.isUnknown())
    attachConstraintNotes(diag, result.getUnknown(), "unproven");
}

/// Emit a note explaining why a constraint is inconclusive. The incoming
/// constraint is expected to be the folded form with all input parameters
/// already substituted.
void LIT::emitConstraintInconclusive(DeclResolver &resolver,
                                     MojoInflightDiag &diag,
                                     ConstraintAttr constraint) {
  TypedAttr prop = constraint.getProposition();

  // First point to the constraint declaration and explain what it folded into.
  diag.attachNote(constraint.getLoc())
      << "constraint declared here needs evidence for " << prop;
  if (StringAttr message = constraint.getMessage())
    diag << ": " << message.getValue();

  // Walk the proposition to look for signs of inconclusiveness.
  TypedAttr canonProp = getCanonicalAttr(prop);
  canonProp.walk([&](ParamOperatorAttr op) {
    // If the constraint involves a function call, it must be inconclusive
    // because it calls a function that is not always_inline("builtin").
    if (op.getOpcode() == POC::Apply) {
      auto callee = op.getOperand(0);
      if (auto sym = dyn_cast<SymbolConstantAttr>(callee)) {
        ASTDecl *calleeDecl = resolver.getDeclForFuncSymbol(sym.getSymbol());
        diag.attachNote(*calleeDecl)
            << "cannot evaluate call to non-builtin function declared here";
      }
    }
    return WalkResult::skip();
  });
}

TriBool LIT::canDischargeConstraintsInScope(
    ASTDecl &declScope, PogListAttr paramListAttr,
    ArrayRef<ConstraintAttr> constraints,
    ArrayRef<ConstraintAttr> origConstraints,
    llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag,
    SmallVectorImpl<ConstraintAttr> *unprovableConstraints,
    ParameterEvaluator *evaluator,
    ArrayRef<ConstraintAttr> additionalAssumptions,
    llvm::BitVector *provenConstraints) {
  if (constraints.empty())
    return TriBool::yes();

  SmallVector<ConstraintAttr> assumptions;
  declScope.getKnownAssumptionsIncludingParents(assumptions);
  // Add any additional assumptions passed in (e.g., conformance constraint).
  assumptions.append(additionalAssumptions.begin(),
                     additionalAssumptions.end());

  SmallVector<TypedAttr> overallAssumptionOperands;
  for (ConstraintAttr assumption : assumptions)
    overallAssumptionOperands.push_back(
        getCanonicalAttr(assumption.getProposition()));

  if (provenConstraints)
    provenConstraints->resize(constraints.size());

  SmallVector<std::pair<size_t, ConstraintAttr>> failedConstraints;
  SmallVector<ConstraintAttr> localUnprovableConstraints;
  for (auto [idx, constraint] : llvm::enumerate(constraints)) {
    TypedAttr origProp = constraint.getProposition();
    if (evaluator)
      origProp = evaluator->getReboundAttribute(origProp);
    TypedAttr canonProp = getCanonicalAttr(origProp);

    // If assumptions imply the constraint, skip it; if they contradict it,
    // treat it as violated.
    TriBool result = isPropositionImplied(canonProp, overallAssumptionOperands);
    if (result.isTrue()) {
      if (provenConstraints)
        provenConstraints->set(idx);
      continue;
    }
    if (result.isFalse()) {
      failedConstraints.emplace_back(idx, constraint);
      continue;
    }

    // Unprovable constraint.
    localUnprovableConstraints.push_back(ConstraintAttr::get(
        origProp, constraint.getLoc(), constraint.getMessage()));
  }

  if (!failedConstraints.empty()) {
    MojoInflightDiag &diag = getDiag({});
    diag << "violated constraint" << plural(failedConstraints.size());
    // Use constraints from the original signature since the ones in
    // `signature` have already been substituted with param bindings and
    // will have already been folded into `False`.
    IndexToDeclRefRemapper remapper(paramListAttr);
    for (auto [idx, constraint] : failedConstraints) {
      // Read the proposition and the user message from the same constraint:
      // prefer the original signature's constraint (the substituted one has
      // already been folded into `False`), falling back to `constraint` when
      // no originals were provided.
      ConstraintAttr src =
          !origConstraints.empty() ? origConstraints[idx] : constraint;
      TypedAttr prop = src.getProposition();
      diag.attachNote(constraint.getLoc(), prop)
          << "constraint declared here evaluated to False, expected "
          << remapper.replace(prop);
      if (StringAttr message = src.getMessage())
        diag << ": " << message.getValue();
    }

    return TriBool::no();
  }

  if (!localUnprovableConstraints.empty()) {
    // Populate the out-parameter if provided.
    if (unprovableConstraints)
      unprovableConstraints->append(localUnprovableConstraints.begin(),
                                    localUnprovableConstraints.end());
    return TriBool::unknown();
  }

  return TriBool::yes();
}

/// Rewrite cond(a, b, a) patterns to and(a, b) for constraint propositions.
/// This handles patterns from "and"/"or" operators to make constraints
/// decomposable:
///   cond(a, b, a) -> and(a, b)  // from "a and b"
///   cond(a, a, b) -> or(a, b)   // from "a or b"
/// Also distributes struct field extraction through cond operations:
///   struct.extract(cond(a, b, c), field) ->
///     and/or(struct.extract(b, field), struct.extract(c, field))
/// Recursively processes nested cond operations to handle "a and b and c".
TypedAttr LIT::deShortCircuitCond(TypedAttr value) {
  assert(isScalarOf<KGENDType::kBool>(value.getType()) &&
         "expected scalar<bool>");

  // Check if this is a struct.extract on top of a cond.
  LIT::StructExtractAttr extractAttr;
  TypedAttr innerValue = getCanonicalAttr(value);
  if (auto extract = dyn_cast<LIT::StructExtractAttr>(innerValue)) {
    extractAttr = extract;
    innerValue = extract.getStructValue();
  }

  // Check if the inner value is a cond operation.
  auto paramOp = dyn_cast<ParamOperatorAttr>(innerValue);
  if (!paramOp || paramOp.getOpcode() != POC::Cond)
    return value; // Not a cond, return unchanged.

  ArrayRef<TypedAttr> operands = paramOp.getOperands();
  assert(operands.size() == 3 && "cond should have 3 operands");

  TypedAttr condOrig = operands[0];
  TypedAttr trueVal = operands[1];
  TypedAttr falseVal = operands[2];

  // If we had a field extraction, apply it to each operand.
  if (extractAttr) {
    trueVal = LIT::StructExtractAttr::get(trueVal, extractAttr.getField(),
                                          extractAttr.getType());
    falseVal = LIT::StructExtractAttr::get(falseVal, extractAttr.getField(),
                                           extractAttr.getType());
  }

  POC opcode;
  if (condOrig == falseVal) {
    opcode = POC::And;
  } else if (condOrig == trueVal) {
    opcode = POC::Or;
  } else {
    // Not a pattern we recognize. Return unchanged.
    return value;
  }

  // Recursively rewrite nested conds in all operands.
  trueVal = deShortCircuitCond(trueVal);
  falseVal = deShortCircuitCond(falseVal);

  TypedAttr logicalOp = ParamOperatorAttr::get(opcode, {trueVal, falseVal});
  // Use Preserved sugar to indicate this is an internal transformation for
  // preserving nested sugar.
  TypedAttr sugarOp = SugarAttr::getPreserved(value, logicalOp);
  return sugarOp;
}

ConstraintAttr LIT::buildBranchAssumption(TypedAttr cond, bool invertCondition,
                                          Location loc) {
  // Convert `x and y` to `x & y` so we get better canonicalization.
  TypedAttr branchCondition = deShortCircuitCond(cond);
  if (invertCondition)
    branchCondition = ParamOperatorAttr::getNot(branchCondition);
  return ConstraintAttr::get(branchCondition, loc, /*message=*/StringAttr());
}

bool LIT::attrConformsToTraitUnderAssumptions(
    TypedAttr actualAttr, TraitType expectedTrait, SharedState &shared,
    ArrayRef<ConstraintAttr> assumptions) {
  ASTType actualType(actualAttr.getType());
  if (actualType.doesConformTo(expectedTrait, shared, {}).isTrue()) {
    return true;
  }

  // Keep original identity when possible: for type parameters this may be a
  // ParamDeclRefAttr, while canonicalized forms can lose that.
  TypedAttr typeExpr = extractParamDeclRef(actualAttr);
  if (!typeExpr)
    typeExpr = getCanonicalAttr(actualAttr);

  TraitType refinedBound =
      getTraitBoundFromAssumptions(typeExpr, shared, assumptions);
  if (!refinedBound)
    return false;

  return ASTType(refinedBound)
      .doesConformTo(expectedTrait, shared, {})
      .isTrue();
}
