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
// This file declares utility functions primarily for parsing, printing and
// verifying LIT related operations and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITUTILS_H
#define KGEN_LITDIALECT_LITUTILS_H

#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/Support/TriBool.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/SMLoc.h"

#define UNI_CLOSURE_TRAIT_NAME "##__mojo_closure__##"

namespace mlir {
class SymbolOpInterface;
class SymbolTable;
} // namespace mlir

namespace M {
class StringArrayAttr;
template <typename T>
class ErrorOr;

namespace KGEN {
class FnEffects;
class ParamDeclAttr;
class ParamDeclRefAttr;
class ParamIndexRefAttr;
class SugarAttr;
class ParamDeclArrayAttr;
class ParameterEvaluator;
class ParameterExprArrayAttr;
enum class ArgConvention : uint32_t;
enum class PassingKind : uint32_t;
enum class VariadicKind : uint32_t;

namespace LIT {
class ImplicitOriginRefAttr;

/// This specifies how the argument or parameter is passed (e.g. `**x`).  This
/// is not the same as PassingKind or ArgConventions.
enum class ArgUnpackStyle {
  kPositional, ///< Positional operand like foo(x)
  kKeyword,    ///< Keyword operand: foo(arg=x)
  kStar,       ///< Splat list of positional values like: foo(*x)
  kStarStar,   ///< Splat list of keyword values like: foo(**x)
};

/// Returns whether the given type is a LIT meta type.
bool isMetaType(Type type);
bool isVariadicOfMetaType(Type type);

/// Returns whether the given attribute is a (variadic of) LIT type expression.
bool isTypeExpr(TypedAttr attr);
bool isVariadicOfTypeExpr(TypedAttr attr);

/// Returns whether the given attribute is a LIT level-1 type expression (e.g.,
/// a type expression that describe a struct type, such as struct meta type and
/// trait type).
bool isFirstLevelTypeExpr(TypedAttr attr);

/// Returns the value of a struct type with no fields, which is the sole
/// inhabitant of that type. Returns null if `type` is not a lit struct type.
/// The caller is responsible for the type having no fields.
TypedAttr getEmptyStructValue(Type type);

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

/// Print/Parse a (potentially) parametric mutability specifier and then a
/// value.  The three forms are: "imm expr", "mut expr", "mut=<expr>, expr"
/// without quotes. This prints "kgen style" not user style.
ParseResult parseOriginParamValue(AsmParser &p, TypedAttr &result);
void printOriginParamValue(AsmPrinter &p, TypedAttr value);
inline void printOriginParamValue(AsmPrinter &p, Operation *, TypedAttr value) {
  printOriginParamValue(p, value);
}

/// Printer for origin parameter expressions (for example inside
/// `origin_of(x)`), suitable for use in user-facing diagnostics. Subclasses can
/// customize how declaration references and other context-sensitive pieces are
/// printed.
class OriginPrinter {
public:
  virtual ~OriginPrinter() = default;

  /// When true, use diagnostic pretty-printing (for example elide mutcast
  /// wrappers and use `origin_of` syntax).
  virtual bool isPrettyPrint() const { return true; }

  /// Print a non-origin parameter (interior origin names, fallback cases).
  virtual void printParam(raw_ostream &os, TypedAttr param) const;

  /// Print a ParamDeclRefAttr as an origin reference.
  virtual void printDeclRef(raw_ostream &os, ParamDeclRefAttr declRef) const;

  /// Resolve a ParamIndexRefAttr to a ParamDeclRefAttr when possible.
  virtual ParamDeclRefAttr resolveIndexRef(raw_ostream &os,
                                           ParamIndexRefAttr idxRef) const;

  /// Resolve an ImplicitOriginRefAttr to an argument name when possible.
  virtual std::optional<llvm::StringRef>
  resolveImplicitOriginRef(raw_ostream &os,
                           ImplicitOriginRefAttr originRef) const;

  /// Choose the SugarAttr operand to print.
  virtual TypedAttr prepareSugarParam(raw_ostream &os, SugarAttr sugar) const;

  /// Print an origin parameter expression. When \p elideOriginOf is true, omit
  /// the outer `origin_of(` / `)` wrapper.
  void print(raw_ostream &os, TypedAttr param, bool elideOriginOf) const;
};

// This magic name is used to convert a subtree origin (a~) into an interior
// origin within CheckLifetimes.  OriginPrinter prints it as a subtree origin.
static constexpr StringRef subtreeInteriorOriginAttrName =
    ".mojo.subtree.interior.origin";

/// Given a subtree origin (a~), return an interior origin (a["magicname"]) that
/// we can use for analysis of an interior origin that might be contained within
/// it.  This is used by CheckLifetimes to model subtree origins.
InteriorOriginAttr getInteriorForSubtreeOrigin(OriginSubtreeAttr subtree);

/// Pretty print a nested symbol reference to a name.
void printNestedSymbolReference(raw_ostream &os, SymbolRefAttr symbol);

/// Parse an optional default value of the given type, with reference-type
/// unwrapping. `defaultVal` is not modified if a default value was not
/// present. If `hasAddress` is set, the default value is parsed as if `type`
/// is an address type: either a pointer or reference. The method is tolerant
/// if `type` is not actually one.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type, bool hasAddress);
void printOptionalDefaultValue(AsmPrinter &p, TypedAttr defaultVal, Type type,
                               bool hasAddress);

/// Parse and print a origin set.
ParseResult parseOriginSet(AsmParser &p, SmallVectorImpl<TypedAttr> &lifetimes);
OptionalParseResult
parseOptionalOriginSet(AsmParser &p, SmallVectorImpl<TypedAttr> &lifetimes);
void printOriginSet(AsmPrinter &p, ArrayRef<TypedAttr> lifetimes);

/// Return true if the origin set parameter is an empty set.
bool isEmptyOriginSet(TypedAttr attr);

void printFnType(AsmPrinter &p, FuncType signature);

//===----------------------------------------------------------------------===//
// MangledSymbol
//===----------------------------------------------------------------------===//

/// This class provides a wrapper around a mojo FuncOp that mangles its name (in
/// `mangled`) but also provides all the components of the mangled name. If the
/// func is already mangled, this will pull everything apart.
struct MangledSymbol {
  /// Mangle the symbol for this op by walking upwards and adding struct/module
  /// names.
  static MangledSymbol mangle(mlir::SymbolOpInterface op);
  /// Demangle this mangled name by parsing it into its component parts.
  static FailureOr<MangledSymbol> demangle(StringAttr mangled,
                                           bool parseSignature = true);

  /// The format for a mangled name is roughly:
  ///  $<module name>::<struct name>[::<struct name>]
  ///    ::<function name>[<comma separated params>]
  ///      (<comma-separated args>)<comma-separated results>

  /// The fully mangled name.
  StringAttr mangled;
  /// The various strings that make up the mangled name.
  SmallVector<StringAttr, 1> moduleNames;
  /// We support nested structs, so there may be more than one struct name.
  SmallVector<StringAttr, 1> structNames;
  /// The bare name of the symbol, which may include parameters.
  StringAttr symName;
  /// The bare name of the symbol without parameters.
  StringAttr identifier;
  /// If the symbol has a signature mangled into the name, then it will be here.
  FunctionType signature;
};

/// Print a mangled symbol.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MangledSymbol &ms);

//===----------------------------------------------------------------------===//
// Verifier helpers
//===----------------------------------------------------------------------===//

/// Verify the the order of passing kinds, and that the number of defaults
/// doesn't exceed the number of corresponding passing kinds.
LogicalResult verifyPassingKinds(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<PogMetadataAttr> pogs,
                                 StringRef argOrParam);

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

// Trait symbols sorting and canonicalization.
void sortAndDeduplicateTraitSymbols(SmallVectorImpl<TraitSymbolAttr> &symbols);
void canonicalizeTraitCompositionSymbols(
    SmallVectorImpl<TraitSymbolAttr> &symbols,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver);

/// Simplify a conforms_to attr by checking if the type value's trait bounds
/// already prove conformance. Extracts the TraitType from the type value
/// automatically, handling both parser and post-lower-lit representations.
FailureOr<TypedAttr> simplifyConformsToAgainstTypeValue(
    TypeConformsToTraitAttr conformsTo,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver);

/// Fold a DowncastAttr when its input is a concrete LIT struct type value.
/// Returns the folded TypeParamAttr, or null if the downcast can't be folded.
TypedAttr foldDowncastToStructType(DowncastAttr downcast);

/// LIT dialect evaluation context. Resolves LIT struct declarations for struct
/// reflection operations. Inherits common dispatch from base class.
class LITSymTabEvaluationContext : public SymTabEvaluationContext {
public:
  using SymTabEvaluationContext::SymTabEvaluationContext;

protected:
  /// Resolve struct info for LIT dialect structs with fallback to KGEN.
  FailureOr<ResolvedStructHandle> resolveStructOp(TypedAttr typeValue,
                                                  bool acceptAsync) override;

  /// Resolve a function symbol via the LIT/KGEN symbol table. Returns the
  /// `lit.fn` op if present, otherwise falls back to the kgen-generator
  /// lookup inherited from `SymTabEvaluationContext`.
  FuncInterface resolveFunctionDecl(SymbolRefAttr symbol) override;

  /// Handle LIT-specific attributes (Downcast, TypeConformsToTrait).
  FailureOr<TypedAttr>
  evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr) override;
};

//===----------------------------------------------------------------------===//
// IndexToDeclRefRemapper
//===----------------------------------------------------------------------===//

/// Utility class for remapping index references to parameter declaration
/// references using metadata from a PogListAttr.
class IndexToDeclRefRemapper
    : public IndexParameterReplacer<IndexToDeclRefRemapper> {
public:
  IndexToDeclRefRemapper(PogListAttr paramListAttr)
      : paramListAttr(paramListAttr) {}

private:
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<IndexToDeclRefRemapper>;

  PogListAttr paramListAttr;
};

//===----------------------------------------------------------------------===//
// TraitSelfBinder
//===----------------------------------------------------------------------===//

/// The signature for a trait requirement will have a Self parameter first whose
/// type is a TraitType for the trait it was found in.  We want to force
/// substitute a new parameter for the Self references even though it has a
/// different metatype.  This doesn't remove the parameter, that will be done
/// later.
class TraitSelfBinder : public IndexParameterReplacer<TraitSelfBinder> {
public:
  TraitSelfBinder(TypedAttr selfValue) : selfValue(selfValue) {}

  FnTypeGeneratorType bindTraitFnSignature(FnTypeGeneratorType sig);

private:
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<TraitSelfBinder>;

  TypedAttr selfValue;
};

//===----------------------------------------------------------------------===//
// ImplicitOriginRefAttrReplacer
//===----------------------------------------------------------------------===//

/// Utility class for replacing implicit origin references that point all the
/// way up to the root scope of the walked value (see PSTIAIRAID) with
/// references to explicitly *named* parameter-decls, creating one decl per
/// distinct origin.
class ImplicitOriginToNameRefAttrReplacer
    : public IndexParameterReplacer<ImplicitOriginToNameRefAttrReplacer> {
public:
  /// Both containers stay owned by the caller: newly created decls are
  /// appended to `newOriginParamDecls`, and `implicitOriginToNewParamRef`
  /// records the origins that were already given a name.
  ImplicitOriginToNameRefAttrReplacer(MLIRContext *ctx,
                                      StringRef namePostfix = StringRef())
      : ctx(ctx), namePostfix(namePostfix) {}

  std::vector<ParamDeclAttr> &getNewOriginParamDecls() { return originDecls; }

private:
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<ImplicitOriginToNameRefAttrReplacer>;

  MLIRContext *ctx;
  StringRef namePostfix;

  std::vector<ParamDeclAttr> originDecls;
  llvm::MapVector<ImplicitOriginRefAttr, ParamDeclRefAttr>
      implicitOriginToNewParamRef;
};

//===----------------------------------------------------------------------===//
// OriginDeclRemapper
//===----------------------------------------------------------------------===//

/// The inverse of `ImplicitOriginRefAttrReplacer`: replaces references to the
/// *named* implicit origin decls it was constructed with by index-based
/// `ImplicitOriginRefAttr` references.
class NameToImplicitOriginRefRemapper
    : public IndexParameterReplacer<NameToImplicitOriginRefRemapper> {
public:
  NameToImplicitOriginRefRemapper(ArrayRef<ParamDeclAttr> originDecls,
                                  size_t depthOffset);

private:
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<NameToImplicitOriginRefRemapper>;

  /// Subtracted from the depth of the created references, because we may be
  /// replacing the signature directly. Theories on what that means:
  /// https://github.com/modularml/modular/pull/62096#discussion_r2114820289
  size_t depthOffset;
  llvm::StringMap<size_t> mapping;
};

//===----------------------------------------------------------------------===//
// Constraint Checking
//===----------------------------------------------------------------------===//

/// Determine whether an assumption proves or disproves a proposition.
///
/// An internally inconsistent assumption proves the proposition vacuously.
/// Top-level `eq` facts in the assumption are closed under symmetry and
/// transitivity via union-find, so e.g. `(A == C) ∧ (B == C)` proves `A == B`.
TriBool isPropositionImplied(TypedAttr proposition, TypedAttr assumption);

/// Returns true if `assumption` implies `proposition`.
inline bool isImplicationProven(TypedAttr proposition, TypedAttr assumption) {
  return isPropositionImplied(proposition, assumption).isTrue();
}

/// Determine how a list of assumptions relate to a goal proposition.
///
/// The assumptions are interpreted as a conjunction; an empty list therefore
/// represents `True`. Returns `yes` if the conjunction proves the goal, `no` if
/// it disproves the goal, and `unknown` when neither relation is provable.
TriBool isPropositionImplied(TypedAttr proposition,
                             ArrayRef<TypedAttr> assumptions);

/// Similar to isPropositionImplied but rebinds propositions using the
/// evaluator prior to checking the constraint.
TriBool isPropositionImplied(ConstraintAttr proposition,
                             ArrayRef<ConstraintAttr> assumptions,
                             ParameterEvaluator &evaluator);

/// Given a type expression and a set of assumptions, compute the effective
/// trait bound implied by any `conforms_to(type, Trait)` constraints. Returns a
/// null TraitType if no refinements apply.
TraitType getTraitBoundFromAssumptions(
    TypedAttr typeAttr, ArrayRef<ConstraintAttr> assumptions,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver);

/// TODO: `ClosureEmitter.cpp` has a nearly identical helper
/// (`getUnderlyingParamRef`). Unify these implementations to avoid drift.
///
/// Extract the underlying ParamDeclRefAttr from a type expression by peeling
/// UpcastAttr and TypeParamAttr/ParamType wrappers.
/// Returns a null attr if no ParamDeclRefAttr can be extracted.
ParamDeclRefAttr extractParamDeclRef(TypedAttr attr);

std::optional<ParameterEvaluator>
populateTraitBindingEvaluator(TraitSymbolAttr traitSymbol,
                              TraitDeclOp traitDecl);

} // namespace LIT

//===----------------------------------------------------------------------===//
// Mojo identifier strings consumed by the Mojo parser.
//
// These are the canonical home for Mojo-side identifier names that both KGEN
// and the GraphCompiler need to agree on. GraphCompiler's `MojoIdentifiers.h`
// re-exports them from here so downstream callers keep using a single
// namespace; do not duplicate definitions on the GraphCompiler side.
//===----------------------------------------------------------------------===//

/// Mojo source-level decorator and method names.
constexpr StringLiteral kFnRegisterInternal = "register_internal";
constexpr StringLiteral kFnRegister = "register";
constexpr StringLiteral kFnRegisterShapeFunction = "register_shape_function";
constexpr StringLiteral kMoggExecuteFuncName = "execute";

/// Mojo package and type leaf names used to recognise extensibility kernel
/// types in MLIR symbol references.
constexpr StringLiteral kPackageStd = "std";
constexpr StringLiteral kPackageExtensibility = "extensibility";
constexpr StringLiteral kLeafManagedTensorSlice = "ManagedTensorSlice";
constexpr StringLiteral kLeafDeviceContext = "DeviceContext";

} // namespace KGEN
} // namespace M

#endif // KGEN_LITDIALECT_LITUTILS_H
