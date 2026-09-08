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
// This file implements support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#include "OverloadSet.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadFitness.h"
#include "ParamInf.h"
#include "StabilityMarkers.h"
#include "Traits.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/DeclSignaturePrinter.h"
#include "Mojo/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

#include <limits>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

// Sizes come from stdlib and kernels precompile overload-resolution samples.
static constexpr unsigned kOverloadEvaluationsInlineSize = 17;
static constexpr unsigned kBestOverloadCandidatesInlineSize = 1;

//===----------------------------------------------------------------------===//
// OverloadSet Implementation
//===----------------------------------------------------------------------===//

OverloadSet::OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                         ParamBindings &&paramBindings, CallSyntax syntax,
                         bool erroneous)
    : baseName(baseName), fnDecls(fnDecls.begin(), fnDecls.end()),
      paramBindings(std::move(paramBindings)), syntax(syntax),
      erroneous(erroneous) {
  assert(llvm::all_of(fnDecls,
                      [](ASTDecl *decl) {
                        return decl->resolvedness >=
                               DeclResolvedness::signature;
                      }) &&
         "Overload set must contain fully resolved declarations");
}

SMLoc OverloadSet::getExprLoc() const { return getExpr()->getLoc(); }

/// For method and static method calls, extract the location of the method
/// name identifier from the expression. This is used for fixit suggestions
/// so that `obj.old_method()` -> `obj.new_method()` or
/// `Type.old_static()` -> `Type.new_static()` replaces only the method name.
/// Returns an invalid SMLoc for non-attribute-reference expressions.
static SMLoc getAttributeNameLoc(const ExprNode *expr) {
  // Unwrap CallNode to get to the underlying callee expression.
  if (auto *callNode = dyn_cast<CallNode>(expr))
    expr = callNode->callee;
  // Extract the identifier location from attribute reference.
  if (auto *attrRef = dyn_cast<AttributeRefNode>(expr))
    return attrRef->getAttributeNameRange().getStart();
  return {};
}

/// Resolve the callee into a single PValue callee.
static PValue getCallee(ASTDecl *fnDecl, const ParamBindings &paramBindings,
                        CallSyntax syntax,
                        const VerifiedParamBindings &verifiedBindings = {}) {
  // Check deprecation and stability warnings for the resolved function.
  // For method calls (instance or static via attribute access), compute the
  // fixit location as the method identifier, not the full expression.
  SMLoc fixitLoc;
  if (syntax == CallSyntax::kMethodCall || syntax == CallSyntax::kDirectCall)
    fixitLoc = getAttributeNameLoc(paramBindings.getExpr());
  checkDeclUsageWarnings(*fnDecl, paramBindings.getExprLoc(),
                         paramBindings.declScope, paramBindings.shared,
                         paramBindings.getExpr()->getRange(), syntax, fixitLoc);

  // Take the fast path to avoid verify binding again if a verified binding has
  // been provided.
  if (verifiedBindings) {
    return getBoundConstAttrForFn(*fnDecl, paramBindings.shared,
                                  verifiedBindings, paramBindings.declScope,
                                  paramBindings.getExprLoc());
  }
  return getBoundConstAttrForFn(*fnDecl, paramBindings);
}

/// Return if the given fitness is valid or function constraint inconclusive,
/// and drop the diagnostics otherwise.
static bool isValidOrFnInconclusive(OverloadFitness &eval) {
  OverloadFitness::Validity validity = eval.getValidity();
  if (validity >= OverloadFitness::Validity::kFunctionConstraintInconclusive)
    return true;
  if (validity == OverloadFitness::Validity::kInvalid)
    eval.takeDiag().abandon();
  return false;
}

/// Assuming we have at least one non-invalid candidate, filter the candidate
/// list in-place to those with the best fitness. If there is more than one
/// candidate with maximal fitness, we filter for non-static methods.
///
/// The input vector is modified in-place to contain only the best candidates.
/// All diagnostics from erroneous candidates are dropped.
///
/// Returns true if any of the best candidates have
/// kFunctionConstraintInconclusive validity.
static bool filterForBestCandidates(
    SmallVectorImpl<std::pair<ASTDecl *, OverloadFitness>> &allCandidates) {
  SmallVector<std::pair<ASTDecl *, OverloadFitness>,
              kBestOverloadCandidatesInlineSize>
      bestCandidates;
  bool areTheBestCandidatesStatic = true;
  bool areTheBestCandidatesImplicit = true;
  bool hasInconclusiveCandidate = false;

  // Find the first valid candidate.
  auto firstValid = llvm::find_if(allCandidates, [&](auto &candidate) {
    return isValidOrFnInconclusive(candidate.second);
  });
  MutableArrayRef<std::pair<ASTDecl *, OverloadFitness>> remainingCandidates(
      firstValid, allCandidates.end());
  assert(!remainingCandidates.empty() && "no valid candidates");

  // Track the best fitness seen so far.
  OverloadFitness *bestFitness = &remainingCandidates.front().second;

  for (auto &[candidate, eval] : remainingCandidates) {
    // Ignore all subsequent failures and candidates that are definitely worse.
    if (!isValidOrFnInconclusive(eval) || bestFitness->isBetter(eval))
      continue;

    // Ignore any functions explicitly marked as disabled.
    if (candidate->isDisabled())
      continue;

    // If we found a strictly better candidate, clear the list and update.
    if (eval.isBetter(*bestFitness)) {
      bestCandidates.clear();
      areTheBestCandidatesStatic = true;
      areTheBestCandidatesImplicit = true;
      hasInconclusiveCandidate = false;
    }

    // If the current best candidates are not static, we ignore new static
    // candidates.
    bool isStatic = candidate->isStaticMethodDecl();
    if (!areTheBestCandidatesStatic && isStatic)
      continue;

    // Explicit ctors takes precedence over implicit. This is to enable a trait
    // for explicit construction, but still allow some types to be implicitly
    // converted. Otherwise the implicit ctor + explicit trait ctor will be
    // ambiguous when initializing with an implicitly convertible type e.g.
    // Bool is Intable and ImplicitlyIntable, so `Int(True)` would be ambiguous.
    bool isImplicit = candidate->getDeclImplicitConversionKind() !=
                      ImplicitConversionKind::None;
    if (!areTheBestCandidatesImplicit && isImplicit)
      continue;

    // If the current best candidates are static, and we just found a non-static
    // one, we clear the list.
    if (areTheBestCandidatesStatic && !isStatic) {
      bestCandidates.clear();
      areTheBestCandidatesStatic = false;
      hasInconclusiveCandidate = false;
    }

    // If the current best candidates are implicit, and we just found a
    // non-implicit one, we clear the list.
    if (areTheBestCandidatesImplicit && !isImplicit) {
      bestCandidates.clear();
      areTheBestCandidatesImplicit = false;
      hasInconclusiveCandidate = false;
    }

    // Track if this candidate is inconclusive.
    hasInconclusiveCandidate |=
        eval.getValidity() ==
        OverloadFitness::Validity::kFunctionConstraintInconclusive;

    bestCandidates.emplace_back(candidate, std::move(eval));
    bestFitness = &bestCandidates.back().second;
  }

  allCandidates.clear();
  allCandidates.append(std::make_move_iterator(bestCandidates.begin()),
                       std::make_move_iterator(bestCandidates.end()));
  return hasInconclusiveCandidate;
}

static const char *getCalleeKind(CallSyntax syntax) {
  switch (syntax) {
  case CallSyntax::kDirectCall:       //< f()
  case CallSyntax::kTypeCall:         //< T()
  case CallSyntax::kImplicitConvert:  //< Conversion in an argument context
  case CallSyntax::kImplicitCopyCtor: //< Implicit copy ctor call.
  case CallSyntax::kImplicitMoveCtor: //< Implicit move ctor call.
    return "function";
  case CallSyntax::kParamBindings: //< symbol[x, val=y]
  case CallSyntax::kIndirectCall:  //< expr()
    return "value";
  case CallSyntax::kMethodCall:       //< x.f()
  case CallSyntax::kOperator:         //< -x and x + y
  case CallSyntax::kReversedOperator: //< y + x
  case CallSyntax::kSubscript:        // v[1, 2]
  case CallSyntax::kAttribute:        // v.x
  case CallSyntax::kDestructor:       //< Destructor due to a value definition.
  case CallSyntax::kTupleGetItem:     //< Call to getitem in a tuple assignment.
  case CallSyntax::kMethodCallSynthetic:
    return "method";
  }
  llvm_unreachable("invalid call syntax");
}

/// Explain inconclusive candidates into `diag`, which the caller owns: the
/// summary is appended to it and the per-candidate reasons are attached as
/// notes. This function will mutate the evaluations to drop any diags from
/// invalid candidates.
/// If `baseName` is empty, it is considered an indirect reference.
/// If `isCall` is true, the message is phrased as a "call", otherwise as a
/// "reference".
/// `deferralAttempted` indicates the caller had installed a deferred
/// body-constraint deferral context, but deferral was rejected (e.g. because
/// more than one candidate is body-constraint-inconclusive). When true, an
/// extra note is attached to explain why deferral did not apply.
static void explainInconclusiveCandidates(
    SharedState &shared, const ExprNode *expr, StringRef baseName, bool isCall,
    MutableArrayRef<std::pair<ASTDecl *, OverloadFitness>> candidates,
    MojoInflightDiag &diag, bool deferralAttempted = false) {
  // Figure out how many possibly valid candidates there are to make the error
  // message more precise.
  size_t numRemainingCandidates = 0;
  for (auto &[candidate, eval] : candidates) {
    if (eval.getValidity() != OverloadFitness::Validity::kInvalid)
      ++numRemainingCandidates;
  }

  // Determine the type of error.
  if (numRemainingCandidates == 1)
    diag << "invalid ";
  else
    diag << "ambiguous ";

  // If it is a named call, include the base name.
  if (!baseName.empty())
    diag << (isCall ? "call to '" : "reference to '") << baseName << "'";
  else
    diag << (isCall ? "indirect call" : "indirect reference");

  // Build the main error message.
  diag << ": lacking evidence to ";
  if (numRemainingCandidates == 1)
    diag << "prove correctness";
  else
    diag << "select candidate";
  diag << expr->getRange();

  // Add fitness information for function constraints (only once, from first
  // function constraint candidate). Function constraint inconclusive
  // candidates have valid parameter bindings, so we can access fitness
  // metrics.
  size_t minConversions =
      candidates.front().second.getNumImplicitConversions() / 2;
  if (minConversions) {
    diag << ", each candidate requires " << minConversions
         << " implicit conversion" << plural(minConversions)
         << ", disambiguate with an explicit cast";
  }

  // Collect constraints, add candidate-specific information, and gather fitness
  // info in a single pass.
  SmallVector<ConstraintAttr> allConstraints;
  for (auto &[candidate, eval] : candidates) {
    OverloadFitness::Validity validity = eval.getValidity();
    // Must abandon any diagnostics from invalid candidates as they'll be
    // dropped.
    if (validity == OverloadFitness::Validity::kInvalid) {
      eval.takeDiag().abandon();
      continue;
    }
    // This is an indirect reference with a constraint failure.
    if (!candidate) {
      diag << "cannot prove constraint";
      continue;
    }

    if (validity == OverloadFitness::Validity::kValid) {
      // This is a valid candidate, but we can't use it since we need to
      // disprove the other candidates first.
      diag.attachNote(*candidate)
          << "candidate is valid but cannot be selected until other "
             "candidates are disproved";
      continue;
    }
    ArrayRef<ConstraintAttr> constraints = eval.getUnprovableConstraints();
    diag.attachNote(*candidate) << "cannot prove";
    if (constraints.size() > 1)
      diag << " or disprove";
    diag << " constraint" << plural(constraints.size()) << " for candidate";
    for (auto constraint : constraints)
      LIT::emitConstraintInconclusive(shared.getDeclResolver(), diag,
                                      constraint);
  }

  // Add action item.
  diag.attachNote(expr->getLoc()) << "provide evidence for ";
  if (numRemainingCandidates > 1)
    diag << "or against ";
  diag << "the constraint" << plural(candidates.size())
       << " here to aid in candidate selection";

  // If deferral was attempted but rejected, attach a note explaining why.
  // Deferral only applies for body-constraint inconclusiveness with exactly
  // one inconclusive candidate; multi-candidate inconclusiveness must still
  // be diagnosed because we cannot pick a candidate to defer for.
  if (deferralAttempted && numRemainingCandidates > 1) {
    diag.attachNote(expr->getLoc())
        << "body constraints cannot be deferred because more than one "
           "candidate is inconclusive";
  }
}

/// Emit a standalone error for ambiguous candidates due to unprovable
/// constraints.
static void emitInconclusiveCandidatesError(
    SharedState &shared, const ExprNode *expr, StringRef baseName, bool isCall,
    MutableArrayRef<std::pair<ASTDecl *, OverloadFitness>> candidates,
    bool deferralAttempted = false) {
  auto diag = shared.emitError(expr->getLoc());
  explainInconclusiveCandidates(shared, expr, baseName, isCall, candidates,
                                diag, deferralAttempted);
}

/// Evaluate the fnDecls candidates and see if there is an unambiguous
/// candidate that works with the specified parameter bindings on the overload
/// set. If so, return the single entry that works.  If not, generate a
/// diagnostic and return null.
PValue OverloadSet::filterOverloadSetForParamBindings(
    DeferredTypingContext *deferredTypingContext) const {
  SmallVector<std::pair<ASTDecl *, OverloadFitness>,
              kOverloadEvaluationsInlineSize>
      evaluations;
  bool allInvalid = true;
  for (ASTDecl *candidate : fnDecls) {
    evaluations.emplace_back(
        candidate, OverloadFitness::evaluate(candidate, *this,
                                             baseValue.ir.getIfPValue()));
    OverloadFitness::Validity validity =
        evaluations.back().second.getValidity();
    allInvalid &= validity == OverloadFitness::Validity::kInvalid;
  }

  // If all candidates are invalid, emit an error.
  if (allInvalid) {
    if (isErroneous())
      return {};
    auto diag = getShared().emitError(
                    getExprLoc(),
                    "cannot form a reference to overloaded declaration of '")
                << baseName << "'" << getExpr()->getRange();
    for (auto &[candidate, eval] : evaluations) {
      diag.attachNote(*candidate)
          << "candidate not viable: " << eval.takeDiag();
    }
    return {};
  }

  // Ok, we have at least one valid candidate, so filter for the best matches.
  bool hasInconclusiveCandidates = filterForBestCandidates(evaluations);
  // If any of the top candidates are inconclusive (function constraints),
  // report the result specially. Allow body-constraint deferral when there
  // is exactly one inconclusive candidate and a deferred typing context is
  // installed.
  if (hasInconclusiveCandidates) {
    bool canDefer =
        deferredTypingContext && evaluations.size() == 1 &&
        evaluations[0].second.getValidity() ==
            OverloadFitness::Validity::kFunctionConstraintInconclusive;
    if (canDefer) {
      hasInconclusiveCandidates = false;
    } else {
      emitInconclusiveCandidatesError(
          getShared(), getExpr(), baseName,
          /*isCall=*/true, evaluations,
          /*deferralAttempted=*/deferredTypingContext != nullptr);
      return {};
    }
  }

  OverloadFitness &bestFitness = evaluations[0].second;
  if (evaluations.size() == 1) {
    ASTDecl *selectedDecl = evaluations[0].first;
    // We don't have arguments, so can't need re-emission.
    assert(bestFitness.getOperandsNeedingOrigins().empty() &&
           "No arguments to require re-emission");

    // If body-constraint deferral applies to this candidate, record the
    // unprovable constraints onto the context now that we are committed.
    if (deferredTypingContext &&
        bestFitness.getValidity() ==
            OverloadFitness::Validity::kFunctionConstraintInconclusive) {
      for (ConstraintAttr c : bestFitness.getUnprovableConstraints())
        deferredTypingContext->deferredConstraints.push_back({c, getExprLoc()});
    }

    // On success, wrap things up into one callee, we know that the best fitness
    // has a verified bindings coming out of parameter inference.
    return getCallee(selectedDecl, paramBindings, syntax,
                     bestFitness.getParamBindings());
  }
  if (isErroneous())
    return {};

  // Otherwise, we couldn't use the parameters to resolve the overload set.
  // We probably forgot to call it, which would provide arguments to resolve it.
  auto diag = getShared().emitError(getExprLoc());
  diag << "cannot form a reference to overloaded declaration of '" << baseName
       << "'" << getExpr()->getRange();

  if (size_t minConversions = bestFitness.getNumImplicitConversions() / 2)
    diag << ", each candidate requires " << minConversions
         << " implicit conversion" << plural(minConversions)
         << ", disambiguate with an explicit cast" << getExpr()->getRange();
  else
    diag.attachNote(getExprLoc()) << "add '()' to call the function";
  for (auto &[candidate, eval] : evaluations)
    diag.attachNote(*candidate) << "candidate declared here";
  return {};
}

/// This method is called when we have selected an overload candidate that
/// matches the operand list, but some argument requires an origin and the
/// argument doesn't have it. This could be because the argument is a computed
/// lvalue or SRValue, but either way we need to spill it to memory to expose
/// the origin.
///
/// This operands in question are indicated by the
/// info.getOperandsNeedingOrigins() bitset, and this method emits them and
/// mutates the operand list so that overload resolution can be iterated and
/// will succeed.
static LogicalResult emitOperandsNeedingOriginsToMemory(
    const OperandsNeedingOriginsList &operandsNeedingOrigins,
    CallOperands &operands, IREmitter &emitter) {
  assert(!operandsNeedingOrigins.empty() && "should emit something");

  SmallPtrSet<size_t, 4> operandsAlreadySpilled;

  // Emit each of the arguments that needs a origin to an MValue.
  for (const OperandNeedingOrigin &info : operandsNeedingOrigins) {
    auto operandIdx = info.operandIdx;
    auto expectedArgType = info.expectedArgType;
    // If the operand is a positional argument it will be in the normal
    // operand list, otherwise it will be in the kwargs list.
    assert((operandIdx < operands.size() ||
            operandIdx == OperandNeedingOrigin::kExprDestOperandIdx) &&
           "argument index incorrect");

    if (!operandsAlreadySpilled.insert(operandIdx).second)
      continue; // Ignore duplicates.

    // Handle the result slot if we have to materialize it into memory.
    if (operandIdx == OperandNeedingOrigin::kExprDestOperandIdx) {
      MLValue addr = operands.dest.getMLValueForResult(
          operands.getExprLoc(), expectedArgType, emitter);
      if (!addr)
        return failure();
      operands.dest = ExprDest(addr, operands.dest.getContext());
      continue;
    }

    // Derive the convention from the signature this argument belongs to, which
    // is the implicit constructor's signature when the operand is spilled to
    // feed a conversion rather than the call itself.  If the argument is a
    // variadic, the elements have their own convention.
    ArgConvention argConvention = info.getArgConvention();

    // If the argument is a DLValue, then we might have a getter/setter pair,
    // but we are only going to use the getter.  In the case when the getter
    // returns a reference, we can directly use that. Reference arguments cannot
    // support writeback anyway.
    //
    // TODO: Explicit handling of DLValues shouldn't be needed here -
    // emitMBValue (et al) already do this. The problem is that the code below
    // isn't handling all the arg conventions correctly, notably ref and mut can
    // bind to a mutable ref returned by getitem.
    // We should generalize "needsRValue" to a full ArgConvention.
    if (auto lv = operands[operandIdx].ir.getIfDLValue()) {
      ExprDest loadDest(EC_RefBinding);
      auto newVal = lv->emitLoad(loadDest, emitter);
      if (!newVal)
        return failure(); // Failed to emit the PValue/SValue to an MRValue.
      operands.values[operandIdx].ir = newVal;

      // If the getter returned a reference, then use that value, otherwise drop
      // an RValue into a temporary and bind an immutable ref.
      if (operands[operandIdx].ir.isMValue())
        continue;
    }

    // Ref convention isn't supported, so if we still have something in the
    // wrong address space, we need to emit a copy.
    if (operands[operandIdx].ir.isMValue() &&
        !operands[operandIdx].ir.getMValueType().isDefaultAddrSpace()) {
      auto eltType = operands[operandIdx].ir.getMValueType().getElementType();
      // Non-trivially copyable types cannot be copied from a non-default
      // address space, because copyinit doesn't allow 'ref'.
      if (!ASTType(eltType).isProvablyImplicitlyTriviallyCopyable(
              operands[operandIdx].expr->getLoc(), emitter.shared,
              emitter.declScope)) {
        emitter.emitError(
            operands[operandIdx].expr->getLoc(),
            "non-implicitly trivially copyable value cannot be copied from a "
            "non-default address space")
            << operands[operandIdx].expr->getRange();
        return failure();
      }

      // Because this is a trivially copyable value, then we can do a copy by
      // doing a load.
      operands.values[operandIdx].ir =
          emitter.emitSRValue(operands[operandIdx], EC_CallArgValue);
      if (!operands[operandIdx].ir)
        return failure();
    }

    AnyValue newVal;
    if (emitter.builder) {
      // We emit this as an MBValue instead of an MRValue specifically so
      // 'ref' arguments do not infer mutability from the temporary.
      if (argConvention != ArgConvention::OwnedMem) {
        newVal = emitter.emitMBValue({operands[operandIdx]},
                                     ExprContext::EC_CallRefArgValue,
                                     expectedArgType);
      } else {
        // "var" variadic list arguments need ot turn into RValues though.
        newVal = emitter.emitMRValue({operands[operandIdx]},
                                     ExprContext::EC_CallRefArgValue,
                                     expectedArgType);
      }
      if (!newVal)
        return failure(); // Failed to emit the PValue/SValue to an MBValue.

    } else {
      // In a comptime context, convert to expected type then drop in
      // memory.
      auto convertedArg =
          emitter.emitPValue(operands[operandIdx],
                             ExprContext::EC_CallRefArgValue, expectedArgType);
      if (!convertedArg)
        return failure(); // Failed to emit the PValue/SValue to an MRValue.
      if (argConvention != ArgConvention::OwnedMem)
        newVal = PMBValue::getFromPValue(convertedArg);
      else
        newVal = PMRValue::getFromPValue(convertedArg);
    }
    operands.values[operandIdx].ir = newVal;
  }
  return success();
}

bool LIT::isNeverCallableSynthesizedCandidate(ASTDecl *candidate) {
  auto func = cast_or_null<FnOp>(candidate->getIfOperation());
  if (!func || !func.isSynthetic())
    return false;
  for (ConstraintAttr constraint :
       func.getFullSignature().getParamListAttrs().getBodyConstraints())
    if (isTriviallyFalseConstraint(constraint))
      return true;
  return false;
}

/// Evaluate the fnDecls candidates and see if there is an unambiguous
/// candidate that works with the specified parameter bindings and provided
/// arguments.  If so, return the single entry that works.
///
/// If not, generate a diagnostic (when `emitDiagnosticOnFailure` is true) and
/// return null.
///
/// NOTE: Unless 'disableMaterialization' is true, this will mutate the operand
/// list, e.g. when calling a static method that doesn't need a self value, and
/// by pre-emitting PValues when not in an parameter context. The actual
/// emission needs to use the updated argument list.
CalleeResult OverloadSet::filterOverloadSet(
    CallOperands &operands, bool emitDiagnosticOnFailure, IREmitter &emitter,
    bool disableMaterialization,
    OperandsNeedingOriginsList *forwardedNeedingOrigins) const {
  // We allow implicit conversion of the operands to this call unless it is
  // itself an implicit conversion.  We don't want to allow A->B->C conversions.
  bool allowImplicitConversions = syntax != CallSyntax::kImplicitConvert;

  std::optional<OperandValue> savedSelfOperand;

  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<std::pair<ASTDecl *, OverloadFitness>,
              kOverloadEvaluationsInlineSize>
      evaluations;
  bool allInvalid = true;

  // We do allow:
  //
  // trait A:
  //     def foo(self):
  //         ...
  //
  // trait B(A):
  //     def foo(self):
  //         ...
  //
  // This set keeps track of the overload witness with the exact same type, and
  // skip one if needed.
  DenseSet<Type> seenWitness;
  ASTType selfTrait;
  for (ASTDecl *candidate : fnDecls) {
    if (candidate->isDisabled())
      continue;

    // If we are dealing with a static method, we check if the operands include
    // a self operand and remove it, otherwise the signature might not match.
    if (operands.hasSelfOperand && candidate->isStaticMethodDecl()) {
      savedSelfOperand = operands[0];
      operands.values.erase(operands.values.begin());
      operands.hasSelfOperand = false;
    }

    FnTypeGeneratorType desiredSignature = candidate->getDeclFullSignature();
    if (candidate->getIfWitness()) {
      auto sig = cast<FnTypeGeneratorType>(getCanonicalType(desiredSignature));
      ArrayRef<Type> inputParamTypes = sig.getInputParamTypes();
      if (inputParamTypes.empty() || !isa<TraitType>(inputParamTypes[0]))
        continue;
      // Does not really matter which we pick, we just want to test equality
      // without considering `_Self` parameter.
      if (!selfTrait)
        selfTrait = inputParamTypes[0];

      TraitSelfBinder selfBinder(
          TypeParamAttr::get(selfTrait, selfTrait.extractMetaType()));
      if (!seenWitness.insert(selfBinder.bindTraitFnSignature(sig)).second)
        continue; // skip if saw the same witness.
    }

    evaluations.emplace_back(
        candidate,
        OverloadFitness::evaluate(desiredSignature, candidate, *this, operands,
                                  allowImplicitConversions));
    OverloadFitness::Validity validity =
        evaluations.back().second.getValidity();
    allInvalid &= validity == OverloadFitness::Validity::kInvalid;

    // Restore 'self' if we moved it out of the way.
    if (savedSelfOperand.has_value()) {
      operands.values.insert(operands.values.begin(), savedSelfOperand.value());
      operands.hasSelfOperand = true;
      savedSelfOperand = {};
    }
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (allInvalid) {
    if (!emitDiagnosticOnFailure || isErroneous())
      return CalleeResult::no();

    // Candidates that are compiler-synthesized functions with a never-
    // satisfiable where clause (e.g. the move-init synthesized for a
    // `Movable where False` conformance) can never actually be called, so
    // they aren't real candidates for diagnostic purposes -- see
    // isNeverCallableSynthesizedCandidate.
    SmallVector<std::pair<ASTDecl *, OverloadFitness *>,
                kOverloadEvaluationsInlineSize>
        realCandidates;
    for (auto &[candidate, eval] : evaluations)
      if (!isNeverCallableSynthesizedCandidate(candidate))
        realCandidates.emplace_back(candidate, &eval);

    // Diagnose the case when there are no candidates found by lookup.
    if (fnDecls.empty() || realCandidates.empty()) {
      auto diag = getShared().emitError(getExprLoc()) << getExpr()->getRange();
      diag << "invalid call to '" << baseName << "': no candidates found";
      return CalleeResult::no();
    }

    // Otherwise, there is one or more candidate, and they all failed.  Emit the
    // primary error on the operand that failed if it is consistent, otherwise
    // fallback to the location of the call. We prefer to issue it on the
    // operand so we can know which of the 42 arguments failed.
    auto diagLoc = realCandidates[0].second->getDiag().getPrimaryLoc();
    for (auto &[candidate, eval] : realCandidates) {
      if (eval->getDiag().getPrimaryLoc() != diagLoc) {
        diagLoc = getShared().translateLocation(getExprLoc());
        break;
      }
    }
    auto diag = getShared().emitError(diagLoc) << getExpr()->getRange();

    // If we have one operand being passed to an __init__, get it to help tailor
    // type conversion errors.
    ASTType initResType, singleOperandType;
    if (baseName == "__init__" && !fnDecls.empty() && operands.size() == 1 &&
        !operands[0].keyword &&
        (syntax == CallSyntax::kTypeCall ||
         syntax == CallSyntax::kImplicitConvert)) {
      // Get the Self type returned by the first __init__.
      initResType = selfResultType;
      assert(selfResultType &&
             "Constructor syntax used without a self result type?");

      // FIXME: Why is this duplicating this logic from the normal overload
      // candidate resolution?
      if (auto cValue = operands[0].ir.getIfCValue())
        singleOperandType = cValue.getRValueType();
    }

    // Reject Int(x) where x is already an Int with an error + fixit.
    if (syntax == CallSyntax::kTypeCall && singleOperandType && initResType &&
        singleOperandType.isEqualCanon(initResType) &&
        isa<CallNode>(getExpr())) {
      const CallNode &callNode = *cast<CallNode>(getExpr());
      // This removes the constructor call, but does not remove the parens
      // because we don't want to introduce precedence problems.
      diag << "cannot construct " << initResType
           << " with itself, you can remove the constructor call"
           << operands[0].expr->getRange()
           << FixIt::remove(callNode.callee->getRange());
      return CalleeResult::no();
    }

    // Diagnose implicit conversions with a custom message.
    if (syntax == CallSyntax::kImplicitConvert && initResType &&
        singleOperandType) {
      // This is true if passing Int type to Int instead of Int() to Int.
      bool isConvertingTypeValue =
          initResType.extractMetaType() == singleOperandType;
      if (isConvertingTypeValue) {
        diag << "cannot implicitly convert " << initResType
             << " type as a value to an instance of " << initResType
             << "; did you mean to instantiate " << initResType << "?"
             << getExpr()->getRange();
      } else {
        diag << "cannot implicitly convert " << singleOperandType
             << " value to " << initResType << getExpr()->getRange();
      }

      return CalleeResult::no();
    }

    if (realCandidates.size() == 1)
      diag << "invalid ";
    else
      diag << "no matching " << getCalleeKind(syntax) << " in ";

    switch (syntax) {
    default:
      diag << "call to '" << baseName << "'";
      break;
    case CallSyntax::kTypeCall:
      diag << "initialization";
      break;
    case CallSyntax::kImplicitConvert:
      diag << "implicit conversion";
      break;
    case CallSyntax::kImplicitCopyCtor:
      diag << "implicit copy";
      break;
    case CallSyntax::kImplicitMoveCtor:
      diag << "implicit move";
      break;
    }

    // If there is a single callee, emit a specific error about the call.
    if (realCandidates.size() == 1) {
      diag << ": " << realCandidates[0].second->takeDiag();
      diag.attachNote(*realCandidates[0].first) << "function declared here";
      return CalleeResult::no();
    }

    // Add a note for what is wrong with each candidate.
    for (auto &[candidate, eval] : realCandidates) {
      diag.attachNote(*candidate)
          << "candidate not viable: " << eval->takeDiag();
    }

    return CalleeResult::no();
  }

  // Ok, we have at least one valid candidate, so filter for the best matches.
  bool hasInconclusiveCandidates = filterForBestCandidates(evaluations);

  // Notify the listener of the updated decl references for the call now that
  // invalid candidates have been filtered out.
  if (!evaluations.empty()) {
    SmallVector<ASTDecl *> bestDecls(llvm::make_first_range(evaluations));
    getShared().notifyListenerOnRef(bestDecls, baseName, getExpr(), syntax);
  }

  // If any of the top candidates are inconclusive (function constraints),
  // report the result specially. However, if the caller installed a deferred
  // deferred typing context and there is exactly one inconclusive candidate, we
  // can treat the candidate as viable and emission proceeds through the
  // normal single-candidate success path below; the unprovable body
  // constraints will be appended to the context at the leaf return below (after
  // any re-emission recursion has settled, to avoid duplicate appends).
  // Multi-candidate inconclusiveness cannot be deferred because we cannot
  // pick which candidate to commit to.
  if (hasInconclusiveCandidates) {
    auto *context = emitter.deferredTypingContext;
    bool canDefer =
        context && evaluations.size() == 1 &&
        evaluations[0].second.getValidity() ==
            OverloadFitness::Validity::kFunctionConstraintInconclusive;
    if (canDefer) {
      hasInconclusiveCandidates = false;
    } else {
      if (emitDiagnosticOnFailure && !isErroneous()) {
        emitInconclusiveCandidatesError(getShared(), getExpr(), baseName,
                                        /*isCall=*/true, evaluations,
                                        /*deferralAttempted=*/context !=
                                            nullptr);
      }
      // The return below is not a rejection: these candidates could not
      // be ruled out, only left undecided.
      return CalleeResult::unknown();
    }
  }

  // If we found exactly one viable candidate then we succeed.
  if (evaluations.size() == 1) {
    ASTDecl *selectedDecl = evaluations[0].first;
    OverloadFitness &bestFitness = evaluations[0].second;
    // If the target is static and there is a self operand, remove it from the
    // operand list so it doesn't get passed.
    if (operands.hasSelfOperand && selectedDecl->isStaticMethodDecl()) {
      operands.values.erase(operands.values.begin());
      operands.hasSelfOperand = false;
    }

    // Finally, wrap things up into one callee, resolving it to a PValue with
    // the parameters bound and substituted into its signature.
    PValue boundFunction = getCallee(selectedDecl, paramBindings, syntax,
                                     bestFitness.getParamBindings());

    if (emitDiagnosticOnFailure && syntax == CallSyntax::kImplicitConvert &&
        selectedDecl->getDeclImplicitConversionKind() ==
            ImplicitConversionKind::Deprecated) {
      auto diag = emitter.emitWarning(getExprLoc(),
                                      "deprecated implicit conversion from ")
                  << operands[0].ir.getRValueTypeIfResolvable() << " to "
                  << selfResultType << getExpr()->getRange();
      std::string resTypeStr = selfResultType.getAsString({&emitter.shared});
      diag.attachNote(getExprLoc())
          << "call '" << resTypeStr + "(...)' explicitly"
          << FixIt::insertBeforeToken(getExpr()->getRangeStart(),
                                      resTypeStr + "(")
          << FixIt::insertAfterToken(getExpr()->getRangeEnd(), ")",
                                     emitter.shared.diags)
          << getExpr()->getRange();
      diag.attachNote(*selectedDecl)
          << "implicit constructor for " << selfResultType << " declared here";
    }

    // It is possible this candidate needs some arguments emitted as MValues
    // (from PValue or SValues) to be passed as 'ref' arguments.  If this
    // happens, emit them now and then re-infer the correct origins.  If not,
    // we're done.
    if (!bestFitness.getOperandsNeedingOrigins().empty()) {
      if (forwardedNeedingOrigins)
        *forwardedNeedingOrigins = bestFitness.getOperandsNeedingOrigins();
      if (!disableMaterialization) {
        // Emit one or more operands to memory.  We know this can't infinitely
        // loop because there is a forward progress guarantee here.
        if (failed(emitOperandsNeedingOriginsToMemory(
                bestFitness.getOperandsNeedingOrigins(), operands, emitter)))
          return CalleeResult::no();

        // Now that we mutated the operand list by introducing some new memory
        // types to provide origins, try again.  This will re-evaluate parameter
        // bindings and either succeed or fail based on the new information.
        // The recursive call will push deferred body constraints (if any) onto
        // the context itself, so do not push them here.
        return filterOverloadSet(operands, emitDiagnosticOnFailure, emitter);
      }
    }

    // If body-constraint deferral applies to this candidate, record the
    // unprovable constraints onto the context now that we are committed to the
    // candidate (and not about to recurse).
    if (auto *context = emitter.deferredTypingContext) {
      if (bestFitness.getValidity() ==
          OverloadFitness::Validity::kFunctionConstraintInconclusive) {
        for (ConstraintAttr c : bestFitness.getUnprovableConstraints())
          context->deferredConstraints.push_back({c, getExprLoc()});
      }
    }

    // Otherwise, we're done! A null callee means binding failed; an error was
    // already emitted.
    if (!boundFunction)
      return CalleeResult::no();
    return CalleeResult::yes(std::move(boundFunction));
  }

  // Otherwise, we have multiple viable candidates that are ambiguous because
  // they all require the same number of implicit conversions.
  if (emitDiagnosticOnFailure && !isErroneous()) {
    auto diag = getShared().emitError(getExprLoc(), "ambiguous call to '")
                << baseName << "'" << getExpr()->getRange();
    if (size_t minConversions =
            evaluations[0].second.getNumImplicitConversions() / 2) {
      diag << ", each candidate requires " << minConversions
           << " implicit conversion" << plural(minConversions)
           << ", disambiguate with an explicit cast";
    }
    for (auto &[candidate, eval] : evaluations)
      diag.attachNote(*candidate) << "candidate declared here";
  }
  return CalleeResult::no();
}

CalleeAndDeclResult OverloadSet::filterOverloadSetForValueType(
    ASTType functionType,
    function_ref<MojoInflightDiag &(SMLoc)> emitError) const {

  // If the target type is something weird then don't filter.  Let the error be
  // reported another way.
  if (!sugarIsa<FnTypeGeneratorType, FnLiteralTypeGeneratorType>(
          functionType)) {
    if (emitError && !sugarIsa<TypeCheckErrorType>(functionType)) {
      auto &diag = emitError(getExprLoc())
                   << "cannot convert function to non-function type "
                   << functionType;
      for (ASTDecl *candidate : fnDecls)
        diag.attachNote(*candidate) << "candidate declared here";
    }
    return CalleeAndDeclResult::no();
  }

  // We do parameter inference to support cases like:
  //
  //    def foo[Type: mlirtype]() -> Type
  //    var f : ()-> Int = foo
  //
  // TODO: We could also support generating a lambda for fancy implicit
  // conversions and subtyping some day.
  auto getBindingsAndBoundCandidateType = [&](GeneratorType candidateType)
      -> std::pair<VerifiedParamBindings, GeneratorType> {
    // Apply any bound parameters to the candidate's type since they will be
    // applied when a reference is made.  We only do this if there are some
    // bindings present, because (unlike normal function calls) the result type
    // may have unbound parameters that we are trying to match, e.g. when in a
    // parameter expression context.
    ParamInf inference(paramBindings, candidateType.getInputParamTypes(),
                       candidateType.getParamListAttrs(),
                       /*allowImplicitConversions=*/true,
                       /*declIfDirect=*/nullptr,
                       /*discardError=*/true,
                       /*deferredTypingContext=*/nullptr,
                       additionalAssumptions);
    VerifiedParamBindings newBindings = inference.inferForStruct();
    if (!newBindings)
      return {{}, nullptr}; // If there is an error, return the problem.

    // If anything was bound, apply it to the signature so the expected
    // argument types are updated.
    candidateType =
        cast<GeneratorType>(newBindings.specializeGeneratorType(candidateType));
    return {std::move(newBindings), candidateType};
  };
  // Converting to `functionType` means discharging the candidate's body
  // constraints, and that can come back undecided.
  enum class Candidacy { kInvalid, kInconclusive, kValid };
  SmallVector<ConstraintAttr> unprovableConstraints;
  auto classifyCandidate = [&](GeneratorType boundCandidateType) -> Candidacy {
    unprovableConstraints.clear();
    // Pass our assumptions along: dropping the candidate's generator body
    // constraints to reach `functionType` may depend on them.
    auto conversion = IREmitter::classifyImplicitConversionWithDetails(
        {UnboundAttr::get(boundCandidateType), getExpr()}, functionType,
        getDeclScope(), additionalAssumptions, /*deferralCtx=*/nullptr);
    if (conversion.isYes())
      return Candidacy::kValid;
    if (conversion.isNo())
      return Candidacy::kInvalid;

    // Undecided: Record the constraints it could not discharge.
    llvm::append_range(unprovableConstraints,
                       std::move(conversion).getUnknown());
    return Candidacy::kInconclusive;
  };

  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<ASTDecl *> validCandidates;
  SmallVector<VerifiedParamBindings> candidateBindings;
  SmallVector<std::pair<ASTDecl *, OverloadFitness>,
              kOverloadEvaluationsInlineSize>
      inconclusiveEvaluations;
  for (ASTDecl *candidate : fnDecls) {
    // Skip functions explicitly marked as 'disabled'.
    if (candidate->isDisabled())
      continue;

    // A trait member reached through a composition has no `FnOp` to form a
    // function literal from, so it cannot be referenced as a function value.
    auto candidateFn = dyn_cast_or_null<FnOp>(candidate->getIfOperation());
    if (!candidateFn)
      continue;
    Type candidateType =
        candidateFn.getFuncLiteralGenerator(getShared().getEvaluationContext())
            .getType();
    auto [newBindings, boundCandidateType] = getBindingsAndBoundCandidateType(
        sugarCast<GeneratorType>(candidateType));
    if (!boundCandidateType)
      continue;

    switch (classifyCandidate(boundCandidateType)) {
    case Candidacy::kInvalid:
      break;
    case Candidacy::kValid:
      validCandidates.push_back(candidate);
      candidateBindings.push_back(std::move(newBindings));
      break;
    case Candidacy::kInconclusive:
      inconclusiveEvaluations.emplace_back(
          candidate, OverloadFitness::constraintInconclusive(
                         std::move(newBindings), unprovableConstraints));
      break;
    }
  }

  // An inconclusive candidate cannot be selected, but it also cannot be ruled
  // out, so nothing else may be selected over it: committing to another
  // candidate would silently discard a potential match.
  if (!inconclusiveEvaluations.empty()) {
    if (!emitError || isErroneous())
      return CalleeAndDeclResult::unknown();

    // A valid candidate is still worth reporting: it is the one the user
    // probably meant, and the note says why it cannot be selected yet.
    for (auto [candidate, bindings] :
         llvm::zip(validCandidates, candidateBindings))
      inconclusiveEvaluations.emplace_back(
          candidate, OverloadFitness::valid(std::move(bindings)));

    // Explain into the caller's diagnostic.
    MojoInflightDiag &diag = emitError(getExprLoc());
    explainInconclusiveCandidates(getShared(), getExpr(), baseName,
                                  /*isCall=*/false, inconclusiveEvaluations,
                                  diag);
    return CalleeAndDeclResult::unknown();
  }

  // Notify the listener of the updated decl references for the call now that
  // invalid candidates have been filtered out.
  if (!validCandidates.empty())
    getShared().notifyListenerOnRef(validCandidates, baseName, getExpr(),
                                    syntax);

  // If we have exactly one viable candidate, then we succeed.
  if (validCandidates.size() == 1) {
    ASTDecl *selectedMethod = validCandidates.front();

    // Use an emitter with invalid context, since errors aren't expected.
    IREmitter emitter(getDeclScope(), EC_OverloadResolution);
    PValue callee = getCallee(selectedMethod, paramBindings, syntax,
                              candidateBindings.front());
    PValue result = emitter.emitPValue({callee, getExpr()},
                                       EC_OverloadResolution, functionType);
    assert(result && "Conversion should always succeed");
    return CalleeAndDeclResult::yes({result, selectedMethod});
  }

  // If we aren't to emit a diagnostic, just return the failure.
  if (!emitError)
    return CalleeAndDeclResult::no();

  MojoInflightDiag &diag = emitError(getExprLoc());

  // Candidates that are compiler-synthesized functions with a never-
  // satisfiable where clause can never actually be called, so they aren't
  // real candidates for diagnostic purposes.
  SmallVector<ASTDecl *> realFnDecls;
  for (ASTDecl *candidate : fnDecls)
    if (!isNeverCallableSynthesizedCandidate(candidate))
      realFnDecls.push_back(candidate);

  ArrayRef<ASTDecl *> declsToReport;
  if (validCandidates.empty()) {
    diag << "no '" << baseName << "' candidates have type " << functionType
         << getExpr()->getRange();
    declsToReport = realFnDecls.empty() ? fnDecls : ArrayRef(realFnDecls);
  } else {
    diag << "ambiguous use of '" << baseName << "' as type " << functionType
         << getExpr()->getRange();
    declsToReport = validCandidates;
  }

  for (ASTDecl *candidate : declsToReport) {
    diag.attachNote(*candidate) << "candidate declared here with type ";
    FnTypeGeneratorType candidateType = candidate->getDeclFullSignature();
    // If there are bindings, specialize the candidate type and print the
    // specialized type.
    bool hadCandidate = false;
    if (!paramBindings.empty()) {
      auto [newBindings, boundCandidateType] =
          getBindingsAndBoundCandidateType(candidateType);
      if (boundCandidateType) {
        diag << ASTType(boundCandidateType) << " (specialized from "
             << ASTType(candidateType) << ")";
        hadCandidate = true;
      }
    }
    // If there are no bindings (or bindings were illegal), print the candidate
    // type as is.
    if (!hadCandidate)
      diag << ASTType(candidateType);
  }
  return CalleeAndDeclResult::no();
}

/// Perform substitutions of the specified bindings into the symbol, returning
/// the resultant LITSymbolConstant attr or producing an error message and
/// returning null. This allows producing a reference to a parameterized
/// function without the parameters specified.  They can be bound later.
TypedAttr OverloadSet::getBoundConstantAttr() const {
  if (fnDecls.size() == 1)
    return getCallee(fnDecls[0], paramBindings, syntax);

  // If we have multiple candidates, emit an ambiguity error.
  assert(!fnDecls.empty() && "DirectCallable malformed");
  auto diag = getShared().emitError(
                  getExprLoc(),
                  "cannot form a reference to overloaded declaration of '")
              << baseName << "'" << getExpr()->getRange();
  diag.attachNote(getExprLoc()) << "add '()' to call the function";

  for (ASTDecl *candidate : fnDecls)
    diag.attachNote(*candidate) << "candidate declared here";

  return {};
}

/// Get a OverloadSet for a lookup of a named method on the specified type.
/// If successful, this provides a non-null OverloadSet.
///
/// On failure, this returns a null OverloadSet and invokes errorHandler if
/// the problem hasn't already been diagnosed. This does not emit an error on
/// failure.
OverloadSet OverloadSet::lookup(ASTDecl &declScope, ASTType type,
                                StringRef methodName, const ExprNode *expr,
                                CallSyntax syntax,
                                function_ref<void()> errorHandler) {
  SharedState &shared = declScope.getShared();

  OverloadSet result(declScope, expr, syntax, /*isErroneous=*/false);
  result.baseName = methodName;

  // If this is a previously-reported error, ignore and don't report an
  // additional error.
  if (type.isTypeCheckErrorType()) {
    result.erroneous = true;
    return result;
  }

  SMLoc callLoc = expr->getLoc();
  if (auto genAttr = sugarDynCast<GeneratorAttr>(PValue(type));
      genAttr && LIT::isTypeExpr(genAttr.getBody()))
    // If this is a type generator, peel it off to expose the partially bound
    // type it encodes.
    type = ASTType(genAttr.getBody());

  // For struct types, we need to look in both the struct and its extensions.
  if (sugarIsa<LIT::StructType>(type)) {
    SmallVector<ASTDecl *, 4> structAndExtensions =
        declScope.collectTypeAndExtensions(type, callLoc);
    for (ASTDecl *containerDecl : structAndExtensions) {
      LookupResult lookup =
          shared.lookupAndResolveDecl(methodName, callLoc, *containerDecl,
                                      /*searchParentScopes=*/false);
      if (lookup.isErroneous()) {
        result.erroneous = true;
        return result;
      }
      if (lookup.isSuccess()) {
        ArrayRef<ASTDecl *> foundDecls = lookup.getIfSuccess();
        for (ASTDecl *decl : foundDecls) {
          if (!decl->isCallableDecl())
            continue;
          result.fnDecls.push_back(decl);
        }
      }
    }
  } else {
    // For non-struct types, we can just look up in the target's ASTDecl.
    LookupResult lookupResult =
        shared.lookupAndResolveDecl(methodName, callLoc, type,
                                    /*searchParentScopes=*/false);
    // If an error was already reported, propagate it.
    if (lookupResult.isErroneous()) {
      result.erroneous = true;
      return result;
    }

    // If we have candidates directly on the receiver, add them.
    if (lookupResult.isSuccess()) {
      ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
      assert(!resultDecls.empty() && "We know this succeeded");
      assert(result.fnDecls.empty() && "Already have entries");

      // Filter out disabled functions to avoid multiple definition conflicts
      for (ASTDecl *decl : resultDecls) {
        if (decl->isDisabled())
          continue;
        // If we find a vardecl or any other thing, then fail to find anything
        // because it cannot be called.
        if (!decl->isCallableDecl()) {
          // FIXME: This seems wrong. why aren't we emitting an error??
          return result;
        }
        result.fnDecls.push_back(decl);
      }
    }
  }

  // If the struct has a nonmaterializable target (e.g. "IntLiteral" will have
  // "Int" as a nonmaterializable target), then it is implicitly convertible to
  // that type.  Check to see if that type has the method: if so we can add them
  // into the overload set.
  //
  // We don't do this for initializers; if you use T() syntax, we only will give
  // you a T instance, even if it is nonmaterializable.
  if (ASTType nmTarget = type.getNonmaterializableTarget(shared)) {
    if (syntax != CallSyntax::kTypeCall &&
        syntax != CallSyntax::kImplicitConvert) {
      LookupResult lookupResult =
          shared.lookupAndResolveDecl(methodName, callLoc, nmTarget,
                                      /*searchParentScopes=*/false);
      if (lookupResult.isSuccess()) {
        ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
        assert(!resultDecls.empty() && "We know this succeeded");

        // Filter out disabled functions to avoid multiple definition conflicts
        for (ASTDecl *decl : resultDecls) {
          if (decl->isDisabled())
            continue;
          // If we find a vardecl or any other thing, then fail to find anything
          // because it cannot be called.
          if (!decl->isCallableDecl()) {
            // FIXME: This seems wrong. why aren't we emitting an error??
            return result;
          }
          result.fnDecls.push_back(decl);
        }
      }
    }
  }

  // If we get this far and there are no candidates in the set, then we can't
  // find anything.  Emit the error.
  if (result.fnDecls.empty() && errorHandler)
    errorHandler();

  return result;
}

/// Lookup of a named named method on the specified type, filtered to match a
/// concrete operand set. If successful, this provides a non-null PValue for a
/// single callee.
///
/// NOTE: This can mutate the operand list, e.g. when calling a static method
/// that doesn't need a self value, and by emitting PValues when not in an
/// parameter context. The actual emission needs to use the updated argument
/// list.
PValue OverloadSet::lookupAndResolve(
    ASTType type, StringRef methodName, CallOperands &callOperands,
    function_ref<void()> lookupFailureErrorHandler,
    bool shouldPrintOverloadErrors, IREmitter &emitter) {
  auto ovSet = OverloadSet::lookup(emitter.getDeclScope(), type, methodName,
                                   callOperands.callExpr, callOperands.syntax,
                                   lookupFailureErrorHandler);

  // If the core lookup failed, don't filter.
  if (ovSet.isNull())
    return {};

  // Filter the overload set with the actual operands list.  If this
  // fails, report an error (if we have an error handler) and reset to a
  // null state so the client can check this.
  auto calleeResult = ovSet.filterOverloadSet(
      callOperands,
      /*emitDiagnosticOnFailure=*/shouldPrintOverloadErrors, emitter);
  if (!calleeResult.isYes())
    return {};
  return calleeResult.getYes();
}

/// Try to resolve the overload set to a single function candidate, using the
/// expected type if provided or using current bindings if an emitter is
/// provided.  This emits errors if 'emitter' is non-null, but does not if it
/// is null.
PValue OverloadSet::getDirectSymbol(
    ASTType expectedType, DeferredTypingContext *deferredTypingContext) const {
  // Handle the case of a single candidate.
  if (fnDecls.size() == 1) {
    // Bind the parameters if there is any.
    return getBoundConstantAttr();
  }

  // With an emitter and an expected type, the overload set can definitely be
  // resolved to a single candidate or not.
  if (expectedType) {
    std::optional<MojoInflightDiag> diag;
    auto emitError = [&](SMLoc loc) -> MojoInflightDiag & {
      return diag.emplace(getShared().emitError(loc));
    };
    auto result = filterOverloadSetForValueType(expectedType, emitError);
    if (!result.isYes())
      return {};
    return result.getYes().callee;
  }

  // If the overload set has parameter bindings, try to resolve the candidates
  // using them.
  if (!paramBindings.empty())
    return filterOverloadSetForParamBindings(deferredTypingContext);

  // Otherwise, emit the "cannot form a reference to overloaded decl" error.
  return getBoundConstantAttr();
}

PValue OverloadSet::getIfPValue() const {
  // Overload sets with base values cannot be emitted as PValues since they
  // depend on a dynamic value.
  // TODO: A conversion can be emitted if the base value is a PValue.
  if (baseValue || fnDecls.size() != 1)
    return {};

  return getBoundConstAttrForFn(*fnDecls.front(), paramBindings);
}

/// Emit this as a RValue if it can be resolved, otherwise emit an ambiguity
/// error and return null.
CValue OverloadSet::emitAsCValue(IREmitter &emitter, ExprDest &dest) {
  // If we have an overload set with multiple possibilities, we'll fail to emit
  // this as a RValue.  Try to resolve it based on the destination's type.
  ASTType expectedType;
  if (fnDecls.size() > 1) {
    expectedType = dest.resolveImpliedType(getExprLoc(),
                                           /*no implied type*/ Type(), emitter);
  }

  // We allow unbound symbols here which can be emitted as an PValue.  In the
  // case where we are partially applying, that will force the unbound symbol
  // into a SRValue which will catch symbols that are not fully bound.
  PValue directSymbolAttr =
      getDirectSymbol(expectedType, emitter.deferredTypingContext);
  if (!directSymbolAttr)
    return {};

  // If we have no base value, then we are just a symbol, return it.
  if (!baseValue)
    return emitter.emitCResult(directSymbolAttr, getExpr(), dest);

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  [[maybe_unused]] auto calleeSignature =
      FnOrFnLiteralTypeGeneratorType::get(directSymbolAttr.getType().mlirType);

  if (!calleeSignature.getArguments().empty())
    assert(!calleeSignature.isAnyVarArg(0) &&
           "Error: self shouldn't be varargs");

  // TODO: Need to emit a closure instance that partially applies the 'self'
  // argument here.
  auto loc = getExprLoc();
  auto diag =
      emitter.emitError(
          loc, "member method closures are not supported; add '()' to call '")
      << baseName << "'";
  dest.resetForError(emitter);
  return {};
}

//===----------------------------------------------------------------------===//
// Call Emission Implementation
//===----------------------------------------------------------------------===//

/// Emit an indirect call to a resolved value in a try block, invoking a
/// callback to generate logic in the 'catch' block that is wrapped around the
/// call. This ensures that the ExprDest is updated and live after the try
/// block, which only works if the "catch" logic doesn't fall through.
///
/// This emits an error and returns null on failure.
CValue IREmitter::emitIndirectCallInTryBlock(
    CValue callee, CallOperands &&operands,
    std::function<void(VarDeclOp errDecl)> emitCatchLogic) {
  ExprDest finalDest(operands.dest.getContext());

  auto calleeSig = FnOrFnLiteralTypeGeneratorType::get(callee.getRValueType());
  auto callExpr = operands.callExpr;
  auto loc = translateLocation(callExpr->getLoc());

  // If the ExprDest is a lazy materialized vardecl, we need to materialize
  // it outside the try block. Note that we still don't know the result type
  // that we're binding to - the signature of the callee might have implicit
  // origins or other things substituted through it that can only be determined
  // as the call is emitted.
  VarDeclOp tmpResult;
  if (calleeSig.isRefResult()) {
    // A ref result will infer the origin of the ref from the arguments.
    // We will do an indirect dance here since the type will be inferred
    // from the result.
    auto varDecl =
        emitVarDecl("__ref_result_tmp__", UnresolvedType::get(getContext()),
                    loc, VarDeclKind::Bind);
    finalDest = std::move(operands.dest);
    operands.dest = ExprDest(varDecl, finalDest.getContext());
  } else if (operands.dest.hasExistingMemoryDest()) {
    // Emit the call result directly into the existing destination.
  } else {
    tmpResult = emitVarDecl("anonymous*", UnresolvedType::get(getContext()),
                            loc, VarDeclKind::Synthesized);
    finalDest = std::move(operands.dest);
    operands.dest = ExprDest(tmpResult, finalDest.getContext());
  }

  VarDeclOp errDecl =
      emitVarDecl("__call_error_tmp__", calleeSig.getUserThrownType(), loc,
                  VarDeclKind::Synthesized);

  // We're going to move the builder around, but restore it to the same
  // insertion point when we're done.
  auto savedBuilder = builder;
  auto tryOp = TryOp::create(*builder, loc, errDecl,
                             /*suppressWarnings=*/true);
  // Stub out the else and finally regions of this try.
  builder->createBlock(&tryOp.getElseRegion());
  TryYieldOp::create(*builder, tryOp.getLoc());
  builder->createBlock(&tryOp.getFinallyRegion());
  TryYieldOp::create(*builder, tryOp.getLoc());
  // Emit this call into the try region.
  builder->createBlock(&tryOp.getTryRegion());

  CValue result = emitIndirectCall(callee, std::move(operands));
  if (!result)
    finalDest.resetForError(*this);
  TryYieldOp::create(*builder, tryOp.getLoc());

  // Emit the except block now that we're good to go.
  builder->createBlock(&tryOp.getExceptRegion());
  emitCatchLogic(errDecl);

  // If we had a ref result, we would have emitted a call into our
  // __ref_result_tmp__ temporary above, and then call emission would have
  // emitted a lit.load.consume to get the value.  The problem is that it
  // drops it INTO the try block and we need it live afterwards.  Move it
  // now.
  if (calleeSig.isRefResult()) {
    Value resultVal = result.getIfMBValue();
    assert(resultVal && "ref result must be a MBValue");

    // We might have a rebind to adjust type sugar.
    if (auto rebindOp = resultVal.getDefiningOp<RebindOp>()) {
      rebindOp->moveAfter(tryOp);
      resultVal = rebindOp.getOperand();
    }

    auto loadConsume = resultVal.getDefiningOp<LIT::LoadConsumeOp>();
    assert(loadConsume && "expected ref result to be a lit.load.consume");
    loadConsume->moveAfter(tryOp);

    // Rebind the result of the ref call.  Assigning through the VarDecl
    // will turn this into an MBValue, stripping (parametric) mutability.
    // Restore this.
    result = CValue::getMValueForRef(resultVal);
  } else if (tmpResult) {
    // If we made a temporary, we can definitely move from it.
    result = MRValue(tmpResult);
  }

  // Emit the result into the "dest" ExprDest outside of the try block.
  builder = savedBuilder;
  return emitCResult(result, callExpr, finalDest);
}

/// Emit a function call to the specified callee with the specified operand
/// values.  This emits an error and returns null on failure.
CValue OverloadSet::emitCall(CallOperands &&operands, IREmitter &emitter) {
  // If we have a bound self, add it to the operand list to simplify the logic
  // below.
  if (baseValue)
    operands.addSelf(baseValue);

  // Check the direct callees to see if they can be unambiguously resolved
  // with the bindings list and specified arguments.
  auto calleeResult =
      filterOverloadSet(operands, /*emitDiagnosticOnFailure=*/true, emitter);
  if (!calleeResult.isYes()) {
    operands.dest.resetForError(emitter);
    return {};
  }

  return emitter.emitCallUnchecked(calleeResult.getYes(), std::move(operands));
}

CValue IREmitter::emitIndirectCall(CValue callee, CallOperands &&operands) {
  if (auto calleeSig =
          sugarDynCast<FuncLiteralTypeGeneratorType>(callee.getRValueType())) {
    // An indirect call to a function literal typed candidate becomes a direct
    // call to the literal itself.
    auto target = calleeSig.getSymbolConstantAttr();
    return emitIndirectCall(target, std::move(operands));
  }

  auto callExpr = operands.callExpr;
  auto calleeSig = sugarDynCast<FuncTypeGeneratorType>(callee.getRValueType());
  if (!calleeSig) {
    // If we are invoking something other than a FuncTypeGeneratorType, try to
    // invoke its `__call__` method.
    operands.addSelf({callee, callExpr});
    return emitNamedMethodCall("__call__", std::move(operands));
  }

  // If we have a function pointer, resolve it to an RValue.
  RValue calleeRV = emitRValue({callee, callExpr}, EC_CallCalleeValue);
  if (!calleeRV) {
    operands.dest.resetForError(*this);
    return {};
  }

  // Check to see if we can apply these operands to the callee signature.
  OverloadSet bindings{"callee", /*fnDecls=*/{},
                       ParamBindings(getDeclScope(), callExpr),
                       operands.syntax};
  auto fitness = OverloadFitness::evaluate(calleeSig, /*indirect*/ nullptr,
                                           bindings, operands,
                                           /*allowImplicitConversions=*/true);
  if (fitness.getValidity() != OverloadFitness::Validity::kValid) {
    // If not, diagnose it with an error.
    if (fitness.isInconclusive()) {
      SmallVector<std::pair<ASTDecl *, OverloadFitness>,
                  kBestOverloadCandidatesInlineSize>
          bestCandidates;
      bestCandidates.emplace_back(nullptr, std::move(fitness));
      emitInconclusiveCandidatesError(shared, callExpr, {}, /*isCall=*/true,
                                      bestCandidates);
      operands.dest.resetForError(*this);
      return {};
    }
    emitError(callExpr->getLoc(), "invalid indirect call: ")
        << fitness.takeDiag();
    operands.dest.resetForError(*this);
    return {};
  }

  // If we have inferred parameters, bind them here.
  auto boundCalleeRV = calleeRV;
  if (!fitness.getParamBindings().empty()) {
    if (auto calleePVal = calleeRV.getIfPValue()) {
      boundCalleeRV =
          PValue(fitness.getParamBindings().specializeGenerator(calleePVal));
    } else {
      // Calling through a function-pointer value.
      // BindParamsOp below requires an operand to be a function generator. A
      // callee whose type is named by a `comptime` alias (`FnT`, `Self.T`)
      // arrives wrapped in `#kgen.sugar`. `calleeSig` is the desugared
      // signature, so requesting it as the result type rebinds that sugar away.
      SRValue calleeVal =
          emitSRValue({calleeRV, callExpr}, EC_CallCalleeValue, calleeSig);
      if (!calleeVal) {
        operands.dest.resetForError(*this);
        return {};
      }
      const VerifiedParamBindings &paramBindings = fitness.getParamBindings();
      auto discharged = KGEN::getDenseBoolArrayAttr(
          getContext(), paramBindings.getDischargedBodyConstraints());
      auto paramValues =
          ParameterExprArrayAttr::get(getContext(), paramBindings.getValues());
      Location loc = translateLocation(callExpr->getLoc());
      auto bindOp = BindParamsOp::create(*builder, loc, calleeVal, paramValues,
                                         discharged);
      boundCalleeRV = SRValue(bindOp.getResult());
    }
  }

  // If the selected candidate needs some register operands emitted to memory,
  // do so and try again.
  if (!fitness.getOperandsNeedingOrigins().empty()) {
    // Emit one or more operands to memory.  We know this can't infinitely
    // loop because there is a forward progress guarantee here.
    if (failed(emitOperandsNeedingOriginsToMemory(
            fitness.getOperandsNeedingOrigins(), operands, *this))) {
      operands.dest.resetForError(*this);
      return {};
    }
    // Now that we mutated the operand list by introducing some new memory
    // types to provide origins, try again.  This will re-evaluate parameter
    // bindings and either succeed or fail based on the new information.
    return emitIndirectCall(calleeRV, std::move(operands));
  }

  // Otherwise, we resolved the callee correctly, emit the call.
  return emitCallUnchecked(boundCalleeRV, std::move(operands));
}

CValue IREmitter::emitNamedMethodCall(StringRef methodName,
                                      CallOperands &&operands) {
  assert(!operands.values.empty() &&
         "Cannot emit a method call without a receiver!");

  // Emit the first/self operand to a CValue so we can figure out which type to
  // lookup on.
  CValue selfVal = operands[0].ir.getIfCValue();
  if (!selfVal) {
    selfVal = emitCValue(operands[0], EC_CallArgValue);
    if (!selfVal) {
      operands.dest.resetForError(*this);
      return {};
    }
    operands[0].ir = selfVal;
  }

  ASTType type = selfVal.getRValueType();

  auto emitNoMethodError = [&]() {
    auto diag = emitError(operands.getExprLoc(), "")
                << type << " does not implement the '" << methodName
                << "' method";
    switch (operands.syntax) {
    case CallSyntax::kMethodCallSynthetic:
    case CallSyntax::kMethodCall:
      [[fallthrough]];
    case CallSyntax::kOperator:
      diag << operands[0].expr->getRange();
      break;
    case CallSyntax::kReversedOperator:
      diag << operands[1].expr->getRange();
      break;
    default:
      break;
    }
  };

  // If the type doesn't have the specified method, emit an error.
  PValue callee = OverloadSet::lookupAndResolve(type, methodName, operands,
                                                emitNoMethodError, true, *this);
  if (!callee) {
    operands.dest.resetForError(*this);
    return {};
  }

  return emitIndirectCall(callee, std::move(operands));
}

static ASTType getConstructorLookupType(ASTType type, ASTDecl &declScope) {
  Type refinedMlirType =
      maybeRefineTypeWithAssumptions(type.mlirType, declScope);
  if (refinedMlirType == type.mlirType)
    return type;
  return ASTType(refinedMlirType);
}

/// Emit a call to __init__, returning an instance of the specified type.  If
/// `allowImplicitConversion` is true, the provided args are allowed to
/// implicitly convert to the expectations of the constructor signatures.
CValue IREmitter::emitConstructorCall(ASTType type,
                                      CallOperands &&callOperands) {
  // If the dest type is invalid, then an error has already been reported.
  if (type.isTypeCheckErrorType()) {
    callOperands.dest.resetForError(*this);
    return {};
  }

  // Fast path for Bool(mlir_value: <scalar_bool>): build the struct attribute
  // directly, avoiding emitConstructorCall → OverloadSet::lookup("__init__",
  // Bool) → lookupAndResolveDecl → resolveBody(Bool).  resolveBody(Bool) can
  // trigger Bool.__init__(value: Scalar[DType.bool]), which needs
  // Scalar[DType.bool] → Scalar, causing a circular dependency when Scalar is
  // already in declsCurrentlyProcessing (e.g. while evaluating SIMD's
  // DevicePassable 'where' clause that uses dtype != DType.int).
  //
  // Detect: target type is Bool, single keyword arg `mlir_value`, value is a
  // scalar<bool> attribute.
  {
    auto boolType = shared.lookupBuiltinType("Bool", declScope,
                                             callOperands.getExpr()->getLoc());
    if (type.isEqualCanon(boolType) && callOperands.size() == 1) {
      auto &operand = callOperands[0];
      auto mlirValueName = StringAttr::get(getContext(), "mlir_value");
      if (operand.keyword == mlirValueName) {
        if (PValue pval = operand.ir.getIfPValue()) {
          if (auto simdAttr = dyn_cast<SIMDAttr>(pval.get())) {
            if (isScalarOf<KGENDType::kBool>(simdAttr.getType())) {
              if (auto boolStructType =
                      sugarDynCast<LIT::StructType>(boolType.mlirType)) {
                auto fieldName = StringAttr::get(getContext(), "_mlir_value");
                std::tuple<StringAttr, TypedAttr> field{fieldName,
                                                        (TypedAttr)simdAttr};
                if (TypedAttr boolStructAttr =
                        LITStructAttr::get({field}, boolStructType))
                  return emitRValue({AnyValue(PValue(boolStructAttr)),
                                     callOperands.getExpr()},
                                    callOperands.dest);
              }
            }
          }
        }
      }
    }
  }

  // Check to see if we can invoke an __init__ method to convert it.
  const ExprNode *expr = callOperands.getExpr();
  type = getConstructorLookupType(type, getDeclScope());
  OverloadSet callee = OverloadSet::lookup(getDeclScope(), type, "__init__",
                                           expr, callOperands.syntax);

  shared.notifyListenerOnCall(callee.fnDecls, expr->getRangeEnd(),
                              callOperands.syntax, callOperands);
  if (callee.isErroneous()) {
    callOperands.dest.resetForError(*this);
    return {};
  }

  // If there are no candidates at all, diagnose specific errors.
  bool hasNoRealCandidates =
      !callee.fnDecls.empty() &&
      llvm::all_of(callee.fnDecls, isNeverCallableSynthesizedCandidate);
  if (!callee || hasNoRealCandidates) {
    if (!type.getDecl(shared) &&
        callOperands.syntax != CallSyntax::kImplicitConvert) {
      emitError(expr->getLoc())
          << "MLIR type " << type
          << " must be created with an MLIR operation, not constructor "
             "syntax";
      callOperands.dest.resetForError(*this);
      return {};
    }

    // Diagnose implicit conversions with a custom message
    if (callOperands.syntax == CallSyntax::kImplicitConvert) {
      ASTType singleOperandType;
      assert(callOperands.size() == 1 &&
             "implicit conversions have one operand");
      if (auto cValue = callOperands[0].ir.getIfCValue())
        singleOperandType = cValue.getRValueType();

      auto diag = emitError(expr->getLoc());
      if (sugarIsa<StructType>(type)) {
        diag << "invalid implicit conversion to " << type
             << ": no constructors found";
        callOperands.dest.resetForError(*this);
        return {};
      }

      // This is true if passing Int type to Int instead of Int() to Int.
      bool isConvertingTypeValue = type.extractMetaType() == singleOperandType;
      bool isImplConvert =
          callOperands.dest.getContext() != EC_CallParamValue &&
          callOperands.dest.getContext() != EC_CallArgValue;
      diag << "cannot " << (isImplConvert ? "implicitly convert " : "pass ");

      if (isConvertingTypeValue)
        diag << type << " type as a ";
      else if (singleOperandType)
        diag << singleOperandType << " ";
      diag << "value" << (isImplConvert ? " to " : ", expected ");
      diag << (isConvertingTypeValue ? "an instance of " : "") << type
           << getContextMessage(callOperands.dest.getContext());

      if (isConvertingTypeValue)
        diag << "; did you mean to instantiate " << type << "?";
      diag << expr->getRange();
      callOperands.dest.resetForError(*this);
      return {};
    }
  }

  // Set the parameter bindings for the type we're creating - they can't be
  // inferred from the result type.
  callee.paramBindings =
      ParamBindings::getForDeclaredType(getDeclScope(), type, expr);
  callee.selfResultType = type;
  return callee.emitCall(std::move(callOperands), *this);
}

//===----------------------------------------------------------------------===//
// Type conversion helpers.

/// If the specified type can be constructed with the specified operands
/// return the initializer that would be invoked. If not, return null PValue.
/// If there were erroneous declarations when processing return failure so we
/// don't indicate downstream errors.
///
/// If there were erroneous declarations, an error has been raised about a
/// constructor that likely would have applied, which should be considered in
/// any error reporting. This does not generate any IR.
FailureOr<CalleeResult> OverloadSet::canConstructType(
    ASTType requiredType, CallOperands &operands, ASTDecl &declScope,
    OperandsNeedingOriginsList *forwardedNeedingOrigins) {
  // Check to see if we can do an implicit conversion by invoking a `__init__`
  // method on the expected type.
  requiredType = getConstructorLookupType(requiredType, declScope);
  OverloadSet callee = OverloadSet::lookup(
      declScope, requiredType, "__init__", operands.getExpr(), operands.syntax,
      /*no error emission on failure */ {});

  // If there are no viable candidates for the construction, we fail.
  if (!callee)
    return callee.isErroneous() ? FailureOr<CalleeResult>(failure())
                                : CalleeResult::no();

  // Install the Self type parameters on the callee directly, since they cannot
  // be inferred from the result.
  callee.paramBindings = ParamBindings::getForDeclaredType(
      declScope, requiredType, operands.getExpr());
  callee.selfResultType = requiredType;

  // Determine if we can emit this using an IREmitter in the parameter domain.
  // This ensures we don't emit any code converting parameters to MValues etc.
  IREmitter paramEmitter(declScope, ExprContext::EC_CallCalleeValue);

  // If we have at least one candidate, we check to see if any of them can
  // work. This needs to call filterOverloadSet manually because we might not
  CalleeResult result = callee.filterOverloadSet(
      operands,
      /*emitDiagnosticOnFailure=*/false, paramEmitter,
      /*disableMaterialization=*/true, forwardedNeedingOrigins);

  if (callee.isErroneous())
    return FailureOr<CalleeResult>(failure());
  if (!result.isYes())
    return result;

  // If we found an unambiguous initializer to build this value, make sure that
  // it returns the right thing we were expecting.  It is possible that
  // conditional conformances constrain the result type more than we were
  // expecting.
  auto resultTy = FnOrFnLiteralTypeGeneratorType::get(result.getYes().getType())
                      .getUserResultType();
  auto &shared = paramEmitter.shared;
  if (!requiredType.isEqualCanon(resultTy)) {
    // It is ok if the self type has different parameters than the
    // declaration, this is a form of conditional conformance.
    // TODO(requires / cond conformance): replace this with a better mechanism.
    if (!ASTType(requiredType).isEqualAllowingUnbound(resultTy, shared))
      return FailureOr<CalleeResult>(failure());
  }

  return result;
}

void OverloadSet::dump() const {
  auto &os = llvm::errs();
  os << "OverloadSet{ ";
  os << baseName << " base name, ";
  os << " functions:\n";
  for (auto f : fnDecls) {
    os << "\t";
    f->dump();
    os << "\n";
  }
  if (paramBindings.empty()) {
    os << "no bound params, ";
  } else {
    os << "param bindings: ";
    paramBindings.dump();
  }
  os << syntax << " call syntax";
  if (erroneous)
    os << ", <ERRONEOUS>";
  os << "\n}\n";
}
