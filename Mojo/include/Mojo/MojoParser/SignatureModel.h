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
// Shared in-memory shape for Mojo decl signatures. Captures everything needed
// to render a parameter or argument as text.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_SIGNATUREMODEL_H
#define KGEN_MOJOPARSER_SIGNATUREMODEL_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/IRValues.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <utility>

namespace M {
namespace KGEN {
namespace LIT {
class SharedState;
} // namespace LIT

/// User-facing convention with which a function argument is passed. Mirrors
/// the keyword that gets emitted before the argument's name.
enum class ArgumentConvention {
  kImm, // implicit `imm`, the default - no keyword emitted
  kDeinit,
  kInOut,
  kOwned,
  kRef,
  kOut,
};

/// Everything needed to render one parameter of a Mojo signature.
struct ParameterInfo {
  StringRef name;
  std::string type;
  PassingKind passingKind;
  VariadicKind variadicKind;
  mlir::TypedAttr defaultValue;
  std::string constraints;
};

/// Everything needed to render one argument of a Mojo signature.
struct ArgumentInfo {
  StringRef name;
  std::string prefix; // e.g. `[origin]` for ref args
  std::string type;
  PassingKind passingKind;
  VariadicKind variadicKind;
  mlir::TypedAttr defaultValue;
  ArgumentConvention convention;
  bool isSelf;
};

//===----------------------------------------------------------------------===//
// Population
//===----------------------------------------------------------------------===//

/// Populate `params` from a parameter list extracted from an MLIR
/// signature/generator. Returns the `ParameterEvaluator` that holds the
/// index -> name bindings so callers can also rebind argument-side attributes.
/// `selfType` is honored only by the function path - struct/alias parameters
/// don't get "Self"-substitution.
ParameterEvaluator populateParameterInfos(
    LIT::SharedState &shared, ArrayRef<mlir::Type> paramTypes,
    PogListAttr paramListAttr, SmallVectorImpl<ParameterInfo> &params,
    std::optional<LIT::ASTType> selfType = std::nullopt);

/// Populate `args` from a function signature. `selfResultFn`, when non-null
/// and returning true, signals that the function's by-ref result represents
/// `self` (the `__init__`/`__copyinit__`/`__moveinit__` family); the relevant
/// out-arg is marked `isSelf`. Otherwise the first arg is treated as `self`.
void populateArgumentInfos(LIT::SharedState &shared,
                           LIT::FnTypeGeneratorType signature,
                           ArrayRef<mlir::Type> userArgTypes,
                           std::optional<LIT::ASTType> selfType,
                           ParameterEvaluator &evaluator,
                           function_ref<bool()> hasSelfResultFn,
                           SmallVectorImpl<ArgumentInfo> &args);

//===----------------------------------------------------------------------===//
// Constraint merging
//===----------------------------------------------------------------------===//

/// Render a constraint proposition as Mojo source-level syntax, e.g.
/// `conforms_to(T, Writable)`. `evaluator`, when non-null, rebinds the
/// proposition first, substituting any positional parameter reference for the
/// named one registered by `populateParameterInfos`. Propositions emitted with
/// the parameters already in scope carry named references throughout, so the
/// rebind is a no-op for them; pass the evaluator anyway rather than rely on
/// that holding for a given caller.
///
/// Propositions are stored de-short-circuited, which leaves an `and`/`or`
/// spelled as `b if a else a`; the source operator is recovered before
/// printing. A conjunction of `conforms_to` predicates over one parameter is
/// then composed into a single `conforms_to(T, A & B)`.
std::string renderConstraintProposition(TypedAttr proposition,
                                        ParameterEvaluator *evaluator,
                                        LIT::SharedState &shared);

/// Walk `constraints` and merge any single-param `conforms_to` ones into the
/// trait bounds of the matching parameter's type string. Constraints that
/// can't be merged are returned as a " where ..." suffix. `evaluator`, when
/// non-null, is used to rebind the constraint proposition's attributes first.
std::string mergeConformsToConstraints(ArrayRef<ConstraintAttr> constraints,
                                       ParameterEvaluator *evaluator,
                                       LIT::SharedState &shared,
                                       SmallVectorImpl<ParameterInfo> &params);

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

/// Return the user-facing keyword for the given argument convention
/// ("imm", "deinit", "mut", "var", "ref", or "out").
StringRef getConventionString(ArgumentConvention conv);

/// Render one parameter into `os` (name[: type][ & Trait...][ = default]).
void renderParameterInfo(const ParameterInfo &p, LIT::SharedState &shared,
                         raw_ostream &os);

/// Render one argument into `os` (convention prefix name: type [= default]).
void renderArgumentInfo(const ArgumentInfo &a, LIT::SharedState &shared,
                        raw_ostream &os);

/// Render a parameter list ([p0, p1, ...]). `offsets`, when non-null, is
/// populated with one [start, end) byte range per parameter, suitable for
/// downstream highlighting/navigation.
void printParameterList(
    ArrayRef<ParameterInfo> params, LIT::SharedState &shared,
    llvm::raw_string_ostream &os,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets = nullptr);

/// Render an argument list (a0, a1, ...). `suppressSlashAfterSelf` skips
/// the positional-only slash that would otherwise appear immediately after
/// the `self` argument of a method, since that's redundant with method-call
/// syntax.
void printArgumentList(
    ArrayRef<ArgumentInfo> args, LIT::SharedState &shared,
    llvm::raw_string_ostream &os, bool suppressSlashAfterSelf,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets = nullptr);

//===----------------------------------------------------------------------===//
// Signatures
//===----------------------------------------------------------------------===//

struct SignatureOffsets;

/// Render the full text of a Mojo function signature from already-populated
/// inputs:
///
///     def name[params](args) -> returnType <fnConstraints>
///
/// Handles the `__init__`-style permutation of the trailing out-arg to the
/// front of the argument list (when `isInit` is set and `args.back().convention
/// == kOut`), mutating `args` in place - callers don't need to do this.
///
/// `returnType` is emitted verbatim after " -> " when non-empty; pass an
/// empty `StringRef` to suppress the return clause entirely (e.g. for
/// initializer methods whose out-arg has been hoisted). `fnConstraints`, when
/// non-empty, is appended verbatim (typically of the form " where ...").
void printFunctionSignatureFromInfos(StringRef name,
                                     SmallVectorImpl<ArgumentInfo> &args,
                                     ArrayRef<ParameterInfo> params,
                                     StringRef returnType,
                                     StringRef fnConstraints, bool isInit,
                                     bool isMethod, LIT::SharedState &shared,
                                     llvm::raw_string_ostream &os,
                                     const SignatureOffsets &offsets);

/// Render the full text of a Mojo struct signature from already-populated
/// inputs:
///
///     struct name[params] <bodyConstraints>
void printStructSignatureFromInfos(StringRef name,
                                   ArrayRef<ParameterInfo> params,
                                   StringRef bodyConstraints,
                                   LIT::SharedState &shared,
                                   llvm::raw_string_ostream &os,
                                   const SignatureOffsets &offsets);

void printAliasSignatureFromInfos(StringRef name, StringRef type,
                                  ArrayRef<ParameterInfo> params,
                                  StringRef bodyConstraints,
                                  LIT::SharedState &shared,
                                  llvm::raw_string_ostream &os,
                                  const SignatureOffsets &offsets);

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// True when the parameter should be hidden from rendered signatures -
/// internal compiler params, name-mangled autoparams, and compiler-synthesized
/// inferred params.
///
/// Mojo parameters fall into several categories:
///   1. Implicit parameters - internal compiler parameters, always hidden.
///   2. Compiler-synthesized inferred parameters - generated by the compiler,
///      hidden.
///   3. Explicitly declared inferred parameters - written in source code,
///      shown.
///   4. Regular parameters - always shown.
bool shouldExcludeParameterFromDocs(PassingKind passingKind,
                                    StringRef paramName);

/// Try to parse a printed constraint string as "conforms_to(ParamName, Traits)"
/// where ParamName is a simple identifier (no dots). If successful, returns
/// true and fills in paramName and traitStr.
bool parseConformsToString(StringRef printed, StringRef &paramName,
                           StringRef &traitStr);

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

/// Generate a user-readable representation of the given pvalue.
std::string generatePValueString(LIT::SharedState &shared, LIT::PValue value);

/// Render the origin part of a `ref` argument or result for display. Suppress
/// `[]` clauses that would just repeat the current argument (implicit-passed
/// param-index origins).
std::string getSignatureOrigin(LIT::SharedState &shared, mlir::TypedAttr origin,
                               LIT::FnTypeGeneratorType signature,
                               bool isRefResult);

/// Render the bracketed prefix for a `ref` argument or result - i.e. the
/// `[origin, addrspace]` clause (including the brackets and trailing space),
/// or an empty string if no prefix is needed.
std::string getRefPrefixAsString(LIT::SharedState &shared, LIT::RefType refType,
                                 LIT::FnTypeGeneratorType signature,
                                 bool isRefResult);

/// Render a Mojo type to a string, accounting for variadic kind, optional
/// passing convention, and an optional `Self`-type substitution. If `selfType`
/// is provided and the rendered type's canonical form equals it, emit `Self`
/// rather than the resolved type.
std::string
generateTypeString(LIT::SharedState &shared, LIT::ASTType type,
                   VariadicKind varKind,
                   std::optional<LIT::ASTType> selfType = std::nullopt,
                   std::optional<ArgConvention> convention = std::nullopt);

/// Prepend the appropriate variadic markers (`*args`, `**kwargs`) to an
/// identifier. Returns a `Twine` over the inputs - consume immediately.
Twine prependVariadicIdentifiers(const Twine &identifier, VariadicKind varKind);

/// Emit `name(*|**)? [: type]?` to `os`. When `elideType` is set, the type
/// suffix is suppressed (used for `self`-typed arguments where the explicit
/// `Self` is redundant).
void dumpIdentifierWithType(raw_ostream &os, StringRef identifier,
                            StringRef type,
                            VariadicKind varKind = VariadicKind::None,
                            bool elideType = false);

/// Render a parameter expression that's serving as a default value, suitable
/// for placing after a `=` in a signature.
std::string getDefaultValueString(mlir::TypedAttr defaultValue,
                                  LIT::SharedState &shared);

//===----------------------------------------------------------------------===//
// String Utils
//===----------------------------------------------------------------------===//

/// Strip implicit auto-parameters of the form `argName(.member)+` from a
/// Mojo-rendered type string. These parameters are fully determined by the
/// argument and noisy for human readers - e.g. given an argument named `c`,
/// the type `Container[c.T]` is reduced to `Container`.
///
/// Examples:
///   stripImplicitArgParams("Container[c.T]", "c")              == "Container"
///   stripImplicitArgParams("Foo[Bar[arg.x], arg.y]", "arg")    == "Foo[Bar]"
///   stripImplicitArgParams("Foo[outputFoo]", "output")         ==
///   "Foo[outputFoo]" stripImplicitArgParams("MyStruct[arg.x]", "arg") ==
///   "MyStruct"
///
/// Assumptions about the input string:
///   - identifiers match `[A-Za-z_][A-Za-z0-9_]*`
///   - brackets are balanced
///   - identifiers do not contain `[`, `]`, or `,`.
///
/// Scope: this is only intended for argument types. There is no obvious
/// "owning name" for a return type, so callers should not invoke this for
/// return positions.
std::string stripImplicitArgParams(llvm::StringRef typeStr,
                                   llvm::StringRef argName);

} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOPARSER_SIGNATUREMODEL_H
