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
// This file contains utilities for working with constraints in the Mojo
// parser, including checking constraints and manipulating constraint
// expressions.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CONSTRAINTS_H
#define KGEN_MOJOPARSER_CONSTRAINTS_H

#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/Support/TriBool.h"
#include "Mojo/Support/TriResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN {
class ParameterEvaluator;
class PogListAttr;

namespace LIT {

using llvm::ArrayRef;
using llvm::SmallVectorImpl;
using llvm::SMLoc;

class ASTDecl;
class DeclResolver;
class MojoInflightDiag;
class SharedState;

/// A TriResult that carries the problematic constraints with it: the ones that
/// were refuted on a `no`, the ones that could not be proven on an `unknown`. A
/// `yes` carries nothing, having no problem to explain.
using ConstraintResult = TriResult<void, SmallVector<ConstraintAttr, 2>,
                                   SmallVector<ConstraintAttr, 2>>;

/// Pair a verdict with the constraints behind it. A `yes` has nothing to
/// explain, so `constraints` is dropped there.
ConstraintResult
makeConstraintResult(TriBool verdict,
                     SmallVector<ConstraintAttr, 2> constraints);

/// Attach one note per constraint, each labeled `kind` ("failed"/"unproven")
/// and carrying the constraint's user message when it has one.
void attachConstraintNotes(MojoInflightDiag &diag,
                           ArrayRef<ConstraintAttr> constraints,
                           StringRef kind);

/// Attach a note per problematic constraint, labeled from the verdict. No-op
/// on `yes`.
void attachConstraintNotes(MojoInflightDiag &diag,
                           const ConstraintResult &result);

/// Emit a note explaining why a constraint is inconclusive. The incoming
/// constraint is expected to be the folded form with all input parameters
/// already substituted.
void emitConstraintInconclusive(DeclResolver &resolver, MojoInflightDiag &diag,
                                ConstraintAttr constraint);

/// Check that the given constraints are satisfied under the given scope.
/// Returns a three-valued verdict: `yes` if all constraints are satisfied, `no`
/// if some constraint is violated, and `unknown` if no constraint is violated
/// but some cannot be proven. An optional callback can be provided to emit
/// failures for constraint violations. If provided, unprovableConstraints will
/// be populated with any unprovable constraints encountered. An optional
/// ParameterEvaluator can be provided to substitute parameters into the
/// constraints. Additional assumptions can be passed to consider alongside the
/// scope's known assumptions (e.g., a conformance constraint during trait
/// checking). If provided, `provenConstraints` is sized to `constraints` and
/// has a bit set per constraint the assumptions proved, letting a caller that
/// needs per-constraint verdicts read them off this one pass instead of
/// re-checking each constraint on its own.
TriBool canDischargeConstraintsInScope(
    ASTDecl &declScope, PogListAttr paramListAttr,
    ArrayRef<ConstraintAttr> constraints,
    ArrayRef<ConstraintAttr> origConstraints,
    llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag,
    SmallVectorImpl<ConstraintAttr> *unprovableConstraints,
    ParameterEvaluator *evaluator,
    ArrayRef<ConstraintAttr> additionalAssumptions = {},
    llvm::BitVector *provenConstraints = nullptr);

/// Rewrite cond(a, b, a) patterns to and(a, b) for constraint propositions.
/// This breaks the "short-circuit" pattern of `and`/`or` operators, so is only
/// legal during constraint checking.
TypedAttr deShortCircuitCond(TypedAttr value);

/// Build a branch assumption from a comptime boolean condition `cond`,
/// canonicalizing short-circuit `and`/`or` into `&`/`|` (via
/// `deShortCircuitCond`) and optionally inverting it for the `else`/negated
/// branch. Shared by the `comptime if` statement path and the comptime ternary
/// `exp1 if cond else exp2` path so that a `conforms_to(T, Trait)` guard
/// refines `T` inside the corresponding branch. `loc` is attached to the
/// constraint.
ConstraintAttr buildBranchAssumption(TypedAttr cond, bool invertCondition,
                                     Location loc);

//===----------------------------------------------------------------------===//
// Type Refinement for Where Clauses
//===----------------------------------------------------------------------===//

/// Check if `actualAttr` satisfies `expectedTrait` either by declared
/// conformance or by scope-level comptime assumptions.
bool attrConformsToTraitUnderAssumptions(TypedAttr actualAttr,
                                         TraitType expectedTrait,
                                         SharedState &shared,
                                         ArrayRef<ConstraintAttr> assumptions);

} // namespace LIT
} // namespace M::KGEN

#endif // KGEN_MOJOPARSER_CONSTRAINTS_H
