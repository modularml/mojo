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
// This file provides the implementation of the ASTType class.
//
//===----------------------------------------------------------------------===//

#include "IREmitter.h"
#include "ParserEvaluationContext.h"

#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/ExprNode.h"
#include "Support/Compiler/OperationUtils.h"

#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// Mojo Diagnostics
//===----------------------------------------------------------------------===//

MojoInflightDiag MojoDiags::emitError(llvm::SMLoc loc, const Twine &message) {
  return MojoInflightDiag(Diags::emitError(loc, message), {});
}
MojoInflightDiag MojoDiags::emitWarning(llvm::SMLoc loc, const Twine &message) {
  return MojoInflightDiag(Diags::emitWarning(loc, message), {});
}

MojoInflightDiag MojoInflightDiag::attachNote(llvm::SMLoc loc) && {
  if (!getDiags())
    return std::move(*this);
  return std::move(*this).attachNote(getDiags()->translateLocation(loc));
}
MojoInflightDiag &MojoInflightDiag::attachNote(llvm::SMLoc loc) & {
  InflightDiag::attachNote(loc);
  return *this;
}

void MojoInflightDiag::addEmittedParam(TypedAttr param,
                                       std::optional<Location> loc,
                                       ASTDecl *ctxDecl) {
  // Remember that we emitted this type.
  emittedParams.push_back({loc.value_or(getLastLoc()), param, ctxDecl});
}

void M::addToDiagnostic(TypedAttr paramValue, InflightDiag &diag) {
  SharedState *shared =
      static_cast<MojoInflightDiag &>(diag).getSharedIfActive();
  if (!shared)
    return; // Ignore discarded diagnostics.

  // Format it of course.
  diag << '\'' << ASTType::getParamAsString(paramValue, /*forDiag=*/shared)
       << '\'';

  // Remember the context decl for when this type was emitted - it could change
  // before the diagnostic is emitted.  This happens (e.g.) in overload
  // resolution where lots of diagnostics are producted (with different callees)
  // and then are emitted in a deferred way.
  ASTDecl *ctxDecl = shared->declResolver->getDiagnosticDeclContext();

  // Remember we emitted this parameter so we can post-process the diagnostic.
  auto &mdiag = static_cast<MojoInflightDiag &>(diag);
  mdiag.addEmittedParam(paramValue, {}, ctxDecl);
}

void M::addToDiagnostic(ASTType type, InflightDiag &diag) {
  if (!diag.getDiags())
    return; // Ignore discarded diagnostics.
  if (!type) {
    diag << "<<NULL TYPE>>";
    return;
  }
  addToDiagnostic(PValue(type), diag);
}

void M::addToDiagnostic(MojoInflightDiag &&otherDiag, InflightDiag &diag) {
  auto &mdiag = static_cast<MojoInflightDiag &>(diag);

  for (auto [loc, param, ctxDecl] : otherDiag.getEmittedParams())
    mdiag.addEmittedParam(param, loc, ctxDecl);

  diag.addDiag(std::move(otherDiag));
}

namespace {
/// Two `ImplicitOriginRefAttr`s with the same `[depth, index]` coordinate are
/// the positionally-corresponding implicit origins of their respective
/// signatures (see `ImplicitOriginRefAttr` in LITAttrs.td).
/// When diffing two
/// independent signatures — e.g. a closure trait's synthesized `__call__`
/// requirement against a struct's `__call__` — each carries its own `self`
/// origin at coordinate `[0,0]`. That is not a user-meaningful difference: it
/// renders as the uninformative `'*[0,0]' but ... '*[0,0]'` and, worse, hides
/// the real difference (e.g. the return type) that lies later in the signature.
/// Treat such refs as equal so the differ looks past them.
static bool areCorrespondingImplicitOrigins(TypedAttr lhs, TypedAttr rhs) {
  auto lo = sugarDynCast<ImplicitOriginRefAttr>(lhs);
  auto ro = sugarDynCast<ImplicitOriginRefAttr>(rhs);
  return lo && ro && lo.getDepth() == ro.getDepth() &&
         lo.getIndex() == ro.getIndex();
}

/// Canonical type equality for diffing, but treating positionally-corresponding
/// implicit origins (see `areCorrespondingImplicitOrigins`) as equal. Recurses
/// through references so a `ref` differing only in such an origin is considered
/// equal.
static bool isEqualForDiff(ASTType lhs, ASTType rhs) {
  if (lhs.isEqualCanon(rhs))
    return true;
  auto lref = dyn_cast<RefType>(lhs);
  auto rref = dyn_cast<RefType>(rhs);
  if (!lref || !rref)
    return false;
  if (!isEqualCanon(lref.getAddressSpace(), rref.getAddressSpace()))
    return false;
  if (!isEqualCanon(lref.getOrigin(), rref.getOrigin()) &&
      !areCorrespondingImplicitOrigins(lref.getOrigin(), rref.getOrigin()))
    return false;
  return isEqualForDiff(lref.getElementType(), rref.getElementType());
}

/// This struct implements textual type+parameter "diffing" to help dig into
/// long type names and identify what parts of them differ.
///
/// Context: A common complaint about mojo is that advanced metaprogramming can
/// produce very long types, particularly when using LayoutTensor. Given this,
/// it can be very difficult to understand what is going on when the compiler
/// barfs out some extremely type type name that isn't compatible with something
/// else.
///
/// We address this through maintenance and selective unwrapping of type sugar,
/// but still some types will have over a dozen parameters.  This digs into the
/// type tree to print something like:
///  .field.size of left value is 'SomeType.size_int' but the right value is '4'
struct ParamDiffer {
  SharedState &shared;
  std::string accessPath;
  TypedAttr leftNested, rightNested;

  /// The diff functions analyze the two specified attrs/types and either decide
  /// they are either 1) atomically incompatible, or 2) there is some
  /// subcomponent that is different.  In the first case, this should set
  /// leftNested/rightNested with the current values.  In the second case, this
  /// adds path information to accessPath and recurses on the subcomponents that
  /// disagree.
  void diff(TypedAttr lhs, TypedAttr rhs) {
    assert(!isEqualCanon(lhs, rhs) && "Cannot diff equal attrs!");

    // Look through type<->attr conversions.
    if (auto lhsTypeParam = dyn_cast<TypeParamAttr>(lhs)) {
      if (auto rhsTypeParam = dyn_cast<TypeParamAttr>(rhs)) {
        // Normally, the type values are inequal, so diff them.
        if (!isEqualCanon(lhsTypeParam.getTypeValue(),
                          rhsTypeParam.getTypeValue()))
          return diff(lhsTypeParam.getTypeValue(), rhsTypeParam.getTypeValue());
        if (!isEqualCanon(lhs.getType(), rhs.getType())) {
          accessPath += ".metatype";
          return diff(lhs.getType(), rhs.getType());
        }
      }
    }

    // Look through sugar to find problems.
    if (auto sugarAttr = dyn_cast<SugarAttr>(lhs)) {
      // Sugar representation of memberalias is weird.
      if (sugarAttr.getKind() != SugarKind::MemberAlias)
        diff(sugarAttr.getSugared(), rhs);
      else
        diff(sugarAttr.getExpanded(), rhs);
      if (leftNested == sugarAttr.getSugared())
        leftNested = sugarAttr; // Preserve the sugar.
      return;
    }
    if (auto sugarAttr = dyn_cast<SugarAttr>(rhs)) {
      // Sugar representation of memberalias is weird.
      if (sugarAttr.getKind() != SugarKind::MemberAlias)
        diff(lhs, sugarAttr.getSugared());
      else
        diff(lhs, sugarAttr.getExpanded());
      if (rightNested == sugarAttr.getSugared())
        rightNested = sugarAttr; // Preserve the sugar.
      return;
    }

    leftNested = lhs;
    rightNested = rhs;
  }

  void diff(ASTType lhs, ASTType rhs) {
    assert(!lhs.isEqualCanon(rhs) && "Cannot diff equal types!");

    // Look through type<->attr conversions.
    if (auto lhsParam = dyn_cast<ParamType>(lhs)) {
      if (auto rhsParam = dyn_cast<ParamType>(rhs))
        return diff(lhsParam.getParam(), rhsParam.getParam());
    }
    if (auto lhsTypeValue = dyn_cast<TypeValueType>(lhs)) {
      if (auto rhsTypeValue = dyn_cast<TypeValueType>(rhs))
        return diff(lhsTypeValue.getTypeValue(), rhsTypeValue.getTypeValue());
    }

    // If these are two metatypes, just transparently look through them.
    if (auto lhsMeta = dyn_cast<StructMetaType>(lhs)) {
      if (auto rhsMeta = dyn_cast<StructMetaType>(rhs))
        return diff(lhsMeta.getType(), rhsMeta.getType());
    }

    // Look through two generator types with the same parameter lists.
    if (auto lhsGen = dyn_cast<GeneratorType>(lhs)) {
      if (auto rhsGen = dyn_cast<GeneratorType>(rhs))
        // NOTE: We could complain about different pogs, but unclear how to
        // identify the right problem.
        if (lhsGen.getParamListAttrs() == rhsGen.getParamListAttrs()) {
          // We know the parameter lists have the same names and kinds, check
          // that their types line up.
          for (auto [idx, lhsParam, rhsParam] : llvm::enumerate(
                   lhsGen.getInputParamTypes(), rhsGen.getInputParamTypes())) {
            if (isEqualCanon(lhsParam, rhsParam))
              continue;

            accessPath += "." + lhsGen.getParamListAttrs().getName(idx).str();
            return diff(lhsParam, rhsParam);
          }
          // If the parameter lists match, check that the bodies match.
          return diff(lhsGen.getBody(), rhsGen.getBody());
        }
    }

    // If we have a func_literal and a func type, ignore the symbol and compare
    // the func types.
    if (auto lhsFn = dyn_cast<FuncLiteralType>(lhs))
      if (auto rhsFn = dyn_cast<FuncType>(rhs))
        if (!isEqualCanon(lhsFn.getFuncLiteral().getType(), rhsFn))
          return diff(lhsFn.getFuncLiteral().getType(), rhsFn);
    if (auto lhsFn = dyn_cast<FuncType>(lhs))
      if (auto rhsFn = dyn_cast<FuncLiteralType>(rhs))
        if (!isEqualCanon(lhsFn, rhsFn.getFuncLiteral().getType()))
          return diff(lhsFn, rhsFn.getFuncLiteral().getType());

    // If we have two function types, try to diff them.
    if (auto lhsFn = dyn_cast<FuncType>(lhs))
      if (auto rhsFn = dyn_cast<FuncType>(rhs)) {
        // If the argument list pogs are the same, diff the arg types.
        auto lhsPogs = lhsFn.getArgListAttrs();
        auto rhsPogs = rhsFn.getArgListAttrs();
        if (lhsPogs.size() == rhsPogs.size()) {
          for (auto [idx, lhsArg, rhsArg] :
               llvm::enumerate(lhsFn.getArguments(), rhsFn.getArguments())) {
            // Don't diff if the argument disagrees on arg convention, this
            // can cause us to compare return types to arguments etc.  We don't
            // require name matches though.
            auto conv = lhsFn.getArgConvention(idx);
            if (isEqualForDiff(lhsArg, rhsArg) ||
                conv != rhsFn.getArgConvention(idx) ||
                lhsPogs.getVariadicKind(idx) != rhsPogs.getVariadicKind(idx))
              continue;

            if (conv == ArgConvention::ByRefResult)
              accessPath += "result type";
            else if (conv == ArgConvention::ByRefError)
              accessPath += "error type";
            else if (!lhsPogs.getName(idx).empty())
              accessPath += "." + lhsPogs.getName(idx).str();
            else if (!rhsPogs.getName(idx).empty())
              accessPath += "." + rhsPogs.getName(idx).str();
            else
              accessPath += ".arg" + std::to_string(idx);
            return diff(lhsArg, rhsArg);
          }
        }
        // TODO: Handle other kinds of errors.
      }

    // Handle references, which commonly come up in function args.
    if (auto lhsRef = dyn_cast<RefType>(lhs))
      if (auto rhsRef = dyn_cast<RefType>(rhs)) {
        if (!isEqualCanon(lhsRef.getAddressSpace(), rhsRef.getAddressSpace())) {
          accessPath += ".address_space";
          return diff(lhsRef.getAddressSpace(), rhsRef.getAddressSpace());
        }
        if (!isEqualCanon(lhsRef.getOrigin(), rhsRef.getOrigin()) &&
            !areCorrespondingImplicitOrigins(lhsRef.getOrigin(),
                                             rhsRef.getOrigin())) {
          accessPath += ".origin";
          return diff(lhsRef.getOrigin(), rhsRef.getOrigin());
        }
        // The origin may have been skipped as a corresponding implicit origin;
        // only recurse into the element type if it is the actual difference.
        // Otherwise the refs differ solely by that origin — nothing meaningful
        // to report — so fall through to the atomic case.
        if (!isEqualCanon(lhsRef.getElementType(), rhsRef.getElementType()))
          return diff(lhsRef.getElementType(), rhsRef.getElementType());
      }

    // Check to see if these are two structs or struct meta types with differing
    // parameters values.  If so, diagnose that difference.
    auto lhsDecl = lhs.getDecl(shared);
    if (lhsDecl && lhsDecl == rhs.getDecl(shared)) {
      assert(lhs.getParamBindings().size() == rhs.getParamBindings().size() &&
             "Type with the same decl should have consistent number of params");

      for (auto [idx, lhsParam, rhsParam] :
           llvm::enumerate(lhs.getParamBindings(), rhs.getParamBindings())) {
        if (isEqualCanon(lhsParam, rhsParam))
          continue;

        // Ok, we found a difference, recursively diff the two parameters.
        auto structDecl = cast<LIT::StructDeclOp>(lhsDecl->getIfOperation());

        // Don't dig into the raw _mlir_origin inside the Origin type.
        if (isa<OriginType>(lhsParam.getType()))
          continue;

        accessPath += "." + structDecl.getParams()[idx].getName().str();
        return diff(lhsParam, rhsParam);
      }
      // It's possible that the parameters exactly match but one is a StructType
      // and the other is StructMetaType.  Just diagnose as different types.
    }

    // Must have the same declarations to compare, we just say that Int vs
    // String are different, we don't "diff" them.

    leftNested = PValue(lhs);
    rightNested = PValue(rhs);
  }
};
} // end anonymous namespace

/// On destruction, emit notes about any sugared values in the types we emitted.
/// There may be more than one type, in which case we're complaining about a X
/// != Y sort of event. We should only unwrap any given identical alias once.
MojoInflightDiag::~MojoInflightDiag() {
  SharedState *shared = getSharedIfActive();
  // If abandoned, don't do anything.
  if (!shared || emittedParams.empty())
    return;

  // Copy the attribute list so we don't get more entries as we emit notes.
  auto emitted = emittedParams;

  // If we have multiple types emitted, then we're comparing the types.  It is
  // possible we have two small types like Scalar[f32] and Scalar[f64], but it
  // is also possible we have ridiculously huge types like happens in kernel
  // programming.  In this case, we should dig into the type to understand what
  // is going on and explain it in a way that doesn't require too much squinting
  // at long type names.
  for (size_t i = 0; i + 1 < emittedParams.size(); ++i) {
    auto lhs = emittedParams[i];
    auto rhs = emittedParams[i + 1];
    if (isEqualCanon(lhs.value, rhs.value))
      continue; // Ignore exact dupes, nothing to diff.

    ParamDiffer differ{*shared, "", {}, {}};
    differ.diff(lhs.value, rhs.value);
    if (differ.accessPath.empty())
      continue; // Didn't look similar.

    assert(differ.leftNested && differ.rightNested && "differ broken");

    // Only do this for very long type names. Don't clutter things up for
    // SIMD types that disagree obviously.
    auto first = ASTType::getParamAsString(lhs.value, shared);
    if (first.size() > 30) {
      std::string leftStr, rightStr;
      {
        DeclResolver::DiagnosticDeclContextChanger x(lhs.ctxDecl);
        leftStr = ASTType::getParamAsString(differ.leftNested, shared);
      }
      {
        DeclResolver::DiagnosticDeclContextChanger x(rhs.ctxDecl);
        rightStr = ASTType::getParamAsString(differ.rightNested, shared);
      }
      // Only surface the diff if the two sub-values actually render
      // differently. When they stringify identically (e.g. two distinct
      // origins that both print as '*[0,0]'), the note would read
      // "... is 'X' but ... is 'X'" — confusing noise that hides, rather than
      // explains, the real difference. This is a general backstop; the common
      // corresponding-origin case is already handled structurally by
      // `isEqualForDiff`/`areCorrespondingImplicitOrigins` above.
      if (leftStr != rightStr) {
        const char *kind =
            LIT::isTypeExpr(differ.leftNested) ? "type" : "value";
        attachNote(rhs.loc) << differ.accessPath << " of the first " << kind
                            << " is '" << leftStr << "' but the second " << kind
                            << " is '" << rightStr << "'";
      }
    }

    // Keep track of these as printed so we can unpack sugar if needed.
    emitted.push_back({lhs.loc, differ.leftNested, lhs.ctxDecl});
    emitted.push_back({rhs.loc, differ.rightNested, rhs.ctxDecl});

    // If the nested values differ as the result of a parameter operator, emit
    // a note suggesting a rebind.  It is plausible we cannot prove equality.
    if (sugarIsa<ParamOperatorAttr>(differ.leftNested) ||
        sugarIsa<ParamOperatorAttr>(differ.rightNested)) {
      attachNote(getPrimaryLoc()) << "types parameters include unfolded "
                                     "expression at parser time; try "
                                     "rebinding to a consistent type?";
    }
  }

  // Don't unpack a single attribute more than once, even if printed multiple
  // times.
  SmallPtrSet<Attribute, 4> unpackedAttr;

  // Finally, take a look at any of the parameters we've printed to see if they
  // have top-level sugar.  If so, unpack them so the user has a better chance
  // of understanding what is going on.
  for (auto [loc, attrValue, ctxDecl] : emitted) {
    // See if anything has alias sugar on it, and if so, unpack it so the user
    // has a better chance of understanding what is going on.  We don't want to
    // look into opaque sugar kinds (AlwaysInlineBuiltin, Preserved) though!
    TypedAttr desugared = SugarAttr::strip(attrValue, /*keepOpaque=*/true);
    if (desugared == attrValue || !unpackedAttr.insert(attrValue).second)
      continue;

    // Make sure to unpack this in the right context so any parameter references
    // are referring to the right declaration.
    DeclResolver::DiagnosticDeclContextChanger x(ctxDecl);

    // Ensure the strings are textually different.
    auto attrString = ASTType::getParamAsString(attrValue, /*forDiag=*/shared);
    auto sugString = ASTType::getParamAsString(desugared, /*forDiag=*/shared);
    if (attrString != sugString)
      attachNote(loc) << "'" << attrString << "' is aka '" << sugString << "'";
  }
}

//===----------------------------------------------------------------------===//
// ASTType
//===----------------------------------------------------------------------===//

// Initialize an ASTType from a parameter expression of metatype type.
ASTType::ASTType(TypedAttr typeParamExpr) {
  if (!typeParamExpr) // Null attribute.
    return;

  // ParamType is the canonical way to turn a parameter expression into a type.
  // It handles stripping of metatype information, looks through upcasts etc.
  assert(LIT::isTypeExpr(typeParamExpr) &&
         "parameter expr must be a type expression");
  mlirType = ParamType::get(typeParamExpr);
}

/// Extract the metatype of this type, always return non-null is the ASTType
/// itself is non-null.
Type ASTType::extractMetaType() const {
  if (!mlirType)
    return {};

  auto type = SugarAttr::strip(mlirType);
  if (auto pti = dyn_cast<ParameterTypeInterface>(type))
    if (auto metaType = pti.getMetaType())
      return metaType;

  if (auto fnGen = dyn_cast<FnLiteralTypeGeneratorType>(mlirType))
    return FnLiteralTypeGeneratorMetaType::get(fnGen);

  // Otherwise, it is a generic MLIR type.
  return NonStructTypeType::get(mlirType.getContext());
}

static bool isMetaTypeForUserDefinedType(Type type) {
  return !sugarIsa<FnLiteralTypeGeneratorMetaType, NonStructTypeType, TypeType>(
      type);
}

/// Unwrap a type to the **first level type** that is defined by user
/// (either a trait type or a struct type or a module type).
static ASTType getDeclDefineType(ASTType t) {
  if (!t)
    return {};

  // Peels off the generator to see the body type, the metatype of the generator
  // type might not be accurate E.g., `comptime T : AnyType = MyStruct`, we want
  // the `MyStruct` declaration instead of the `AnyType`.
  if (auto genAttr = sugarDynCast<GeneratorAttr>(PValue(t));
      genAttr && LIT::isTypeExpr(genAttr.getBody()))
    return getDeclDefineType(ASTType(genAttr.getBody()));

  // We get the declaration from the metatype of the type.  For example, if we
  // have a parametric type like "T" where "T: AnyType", we can know that T has
  // AnyType bound.
  // Canonicalize the type first to strip sugar rebind which could hide things
  // like `upcast` which would otherwise be looked through.
  Type type = ASTType(getCanonicalType(t)).extractMetaType();
  if (!isMetaTypeForUserDefinedType(type))
    return {};

  // If our metatype is itself parametric, for example, we have something like:
  //     !kgen.param<:!lit.anytrait<<@Movable>> elt_trait>
  // Then this type conforms to some parametric trait that is bound by at least
  // Movable.  Use Movable as the declaration we're working with.
  if (auto paramRef = dyn_cast<ParamType>(type)) {
    // AnyTrait is the only metatype of a metatype.
    type = sugarCast<AnyTraitType>(paramRef.getParam().getType());
  }

  if (auto anyStruct = dyn_cast<StructMetaType>(type))
    return anyStruct.getType();

  if (auto anyMeta = dyn_cast<StructMetaMetaType>(type))
    return anyMeta.getType().getType();

  if (auto anyTrait = dyn_cast<AnyTraitType>(type))
    type = anyTrait.getTraitType();

  return type;
}

/// If this is a user declared type, return the declaration that this came
/// from.  If this is a raw MLIR type or a metatype, return null.
ASTDecl *ASTType::getDecl(SharedState &shared) const {
  // We get the declaration from the metatype of the type.  For example, if we
  // have a parametric type like "T" where "T: AnyType", we can know that T has
  // AnyType bound.
  // Canonicalize the type first to strip sugar rebind which could hide things
  // like `upcast` which would otherwise be looked through.
  ASTType strippedType = getDeclDefineType(*this);
  if (!strippedType)
    return {};

  if (auto anyStruct = dyn_cast<StructType>(strippedType))
    return shared.declResolver->getDeclForTypeSymbolIfExists(
        anyStruct.getSymbol());

  if (auto traitType = dyn_cast<TraitType>(strippedType))
    return shared.declResolver->getTraitDecl(traitType);

  if (auto module = dyn_cast<ModuleType>(strippedType))
    return shared.declResolver->getDeclForTypeSymbolIfExists(
        module.getSymbol());

  return nullptr;
}

ArrayRef<TypedAttr> ASTType::getParamBindings() const {
  Type metatype = SugarAttr::strip(extractMetaType());
  if (auto metaType = dyn_cast_or_null<StructMetaType>(metatype))
    return metaType.getParamValues();
  if (auto mmType = dyn_cast_or_null<StructMetaMetaType>(metatype))
    return mmType.getParamValues();
  if (auto genAttr = sugarDynCastIfPresent<GeneratorAttr>(PValue(*this));
      genAttr && LIT::isTypeExpr(genAttr.getBody()))
    return ASTType(genAttr.getBody()).getParamBindings();

  return {};
}

TypeSignatureType ASTType::getSignature() const {
  Type metatype = SugarAttr::strip(extractMetaType());
  if (auto metaType = dyn_cast_or_null<StructMetaType>(metatype))
    return metaType.getSignature();
  if (auto mmType = dyn_cast_or_null<StructMetaMetaType>(metatype))
    return mmType.getSignature();
  if (auto genAttr = sugarDynCastIfPresent<GeneratorAttr>(PValue(*this));
      genAttr && LIT::isTypeExpr(genAttr.getBody()))
    return ASTType(genAttr.getBody()).getSignature();

  return {};
}

/// Return this type with any parameter bindings removed.
ASTType ASTType::getWithoutParameters(SharedState &shared) const {
  if (!mlirType)
    return {};

  ASTType firstLevelType = getDeclDefineType(*this);
  if (!firstLevelType)
    return {};

  if (auto declRef = dyn_cast<StructType>(firstLevelType)) {
    auto unboundType =
        cast<StructDeclOp>(getDecl(shared)->getIfOperation()).bindReference();
    // Wrap it back to the original meta type.
    if (isa<StructMetaType>(*this))
      return StructMetaType::get(unboundType);
    if (isa<StructMetaMetaType>(*this))
      return StructMetaMetaType::get(StructMetaType::get(unboundType));
    return unboundType;
  }

  // Not parameterized.
  return *this;
}

bool ASTType::hasUnboundParameters() const {
  return llvm::any_of(getParamBindings(),
                      [](TypedAttr param) { return isa<UnboundAttr>(param); });
}

bool ASTType::isEqualCanon(ASTType other) const {
  // We have no type sugar yet so we can just do pointer equality tests.
  if (mlirType == other.mlirType)
    return true;
  // Struct types with the same metatype are always equal. This is used to
  // detect when two type aliases refer to the same underlying type.
  if (auto meta = dyn_cast_or_null<StructMetaType>(extractMetaType()))
    if (meta == other.extractMetaType())
      return true;

  return getCanonicalType(*this) == getCanonicalType(other);
}

/// Return true if this is the same as another ASTType are the same, or if they
/// match when unbound parameters in the 'this' type are treated as
/// the same as the corresponding parameter in the second type.
///    Foo[1] != Foo[2]   but  Bar[?, 1] == Bar[7, 1]
bool ASTType::isEqualAllowingUnbound(ASTType other, SharedState &shared) const {
  if (isEqualCanon(other))
    return true;

  // Must have the same struct declarations.
  if (getDecl(shared) != other.getDecl(shared))
    return false;

  ArrayRef<TypedAttr> lhsParams = getParamBindings();
  ArrayRef<TypedAttr> rhsParams = other.getParamBindings();
  assert(lhsParams.size() == rhsParams.size() &&
         "Type with the same decl should have consistent number of params");
  for (auto [lhsParam, rhsParam] : llvm::zip(lhsParams, rhsParams)) {
    if (!isa<UnboundAttr>(lhsParam) &&
        getCanonicalAttr(lhsParam) != getCanonicalAttr(rhsParam))
      return false;
  }
  return true;
}

/// Return true if this is a None type.
bool ASTType::isNoneType() const { return sugarIsa<KGEN::NoneType>(mlirType); }

/// Return true if this is a TypeCheckError type.
bool ASTType::isTypeCheckErrorType() const {
  return sugarIsa<TypeCheckErrorType>(mlirType);
}

/// If this type is a standard library Origin struct, return the !lit.origin
/// parameter, otherwise return null.
TypedAttr ASTType::isOriginStruct() const {
  auto structType = sugarDynCast<LIT::StructType>(mlirType);
  if (structType &&
      structType.getSymbol().getLeafReference().strref() == "Origin" &&
      structType.getParamValues().size() == 2) {
    auto result = structType.getParamValues()[1];
    if (sugarIsa<TypeCheckErrorType>(result.getType()))
      return {};
    assert(sugarIsa<OriginType>(result.getType()) &&
           "Origin struct should have a !lit.origin parameter");
    return result;
  }

  // Not the origin struct.
  return {};
}

/// Given a parameter that is a !lit.origin or an Origin, return the
/// underlying !lit.origin.  This returns null on failure.
TypedAttr ASTType::extractOriginOf(TypedAttr value) {
  // If this is the Origin[mut, litorigin] type, take the origin from the 2nd
  // parameter.
  if (auto originParam = ASTType(value.getType()).isOriginStruct())
    return originParam;

  // A raw !lit.origin always works.
  if (isa<OriginType>(value.getType()))
    return value;
  if (auto type = dyn_cast<OriginType>(getCanonicalType(value.getType())))
    return ParamOperatorAttr::getRebind(value, type);
  return {};
}

/// Return the @__nonmaterializable decorator target for the type, or null if
/// there is none.
ASTType ASTType::getNonmaterializableTarget(SharedState &shared) const {
  if (auto structDecl = getDecl(shared)) {
    // If the type is a MetaType itself, don't return the nonmaterializable
    // target, we could theoretically return a `meta<!target>` here too, but we
    // have to decide what implicit conversion between meta types means first.
    if (isa<StructMetaMetaType>(extractMetaType()))
      return {};

    if (auto structOp =
            dyn_cast_or_null<StructDeclOp>(structDecl->getIfOperation()))
      if (TypeAttr targetMlirType = structOp.getNonmaterializableTargetAttr())
        return ASTType(targetMlirType.getValue());
  } else if (auto f = sugarDynCast<FnLiteralTypeGeneratorType>(mlirType)) {
    // A function literal is non-materializable, the nonmaterializable target is
    // the function pointer type.
    return f.getSymbolConstantAttr().getType();
  }

  return {};
}

/// Return whether the specified type is known to be RegisterPassable; if
/// generic, this returns the 'genericsDefault' value.
static TypeConvention getRegisterPassability(ASTType type, llvm::SMLoc loc,
                                             SharedState &shared,
                                             TypeConvention genericDefault) {
  assert(type.mlirType && "getRegisterPassability called with null mlirType");

  auto checkPR = [&](ASTType ty, ASTDecl *decl, StringRef traitName) {
    auto trait = shared.lookupBuiltinTraitType(traitName, loc);
    // If builtin is not enabled (let's kill this!), we just assume __mlir_type
    // to be register-passable (by no means this is correct, but just to pass
    // some existing tests).
    if (!trait)
      return !decl;

    FailureOr<TriBool> upCast = IREmitter::canMetaTypeUpCastTo(
        shared, loc, ty.extractMetaType(), trait, decl);
    return succeeded(upCast) && upCast->isTrue();
  };

  // A type refinement (e.g. from
  // `comptime assert conforms_to(T, TrivialRegisterPassable)`) wraps the
  // parameter reference in a DowncastAttr whose bound records the refined
  // trait set. That refined bound can prove register passability even when the
  // declared bound cannot, so consult it directly.
  if (auto paramRefTy = sugarDynCast<ParamType>(type.mlirType))
    if (auto downcast = dyn_cast<DowncastAttr>(paramRefTy.getParam())) {
      ASTDecl *refinedDecl = type.getDecl(shared);
      TypeConvention refinedConvention = TypeConvention::MemoryOnly;
      if (checkPR(type, refinedDecl, "TrivialRegisterPassable"))
        refinedConvention = TypeConvention::RegisterPassableTrivial;
      else if (checkPR(type, refinedDecl, "RegisterPassable"))
        refinedConvention = TypeConvention::RegisterPassable;

      // Explicit trait downcasts can add a trait without merging the input
      // type's full bound into the result. A downcast preserves the input value
      // representation, so use the stronger convention proved by either side.
      TypeConvention inputConvention = getRegisterPassability(
          ASTType(DowncastAttr::strip(downcast)), loc, shared, genericDefault);
      if (refinedConvention == TypeConvention::RegisterPassableTrivial ||
          inputConvention == TypeConvention::RegisterPassableTrivial)
        return TypeConvention::RegisterPassableTrivial;
      if (refinedConvention == TypeConvention::RegisterPassable ||
          inputConvention == TypeConvention::RegisterPassable)
        return TypeConvention::RegisterPassable;
      return inputConvention;
    }

  ASTDecl *decl = type.getDecl(shared);
  if (sugarIsa<StructMetaType>(type.mlirType)) {
    // If this is a generic type, use the default specification.
    if (auto paramRefTy = sugarDynCast<ParamType>(type.mlirType))
      if (sugarIsa<ParamType, AnyTraitType>(paramRefTy.getParam().getType()))
        return genericDefault;
  }

  // We don't yet have a runtime representation for packages or modules, but
  // when we do, it will not be register-passable. A module reference is an
  // ImportOp; the raw FileModuleOp/PackageOp cases are kept for the underlying
  // decls.
  if (decl && isa_and_nonnull<FileModuleOp, PackageOp, ImportOp>(
                  decl->getIfOperation()))
    return TypeConvention::MemoryOnly;

  if (checkPR(type, decl, "TrivialRegisterPassable"))
    return TypeConvention::RegisterPassableTrivial;

  if (checkPR(type, decl, "RegisterPassable"))
    return TypeConvention::RegisterPassable;

  if (decl && isa<TraitType>(type.extractMetaType())) {
    // We can not prove non-register passability for a type value bound by
    // trait, return the generic default.
    return genericDefault;
  }

  // A struct with a conditional RegisterPassable conformance is
  // pessimistically MemoryOnly (the constraint hasn't been evaluated yet).
  // However, it *might* be register-passable once the constraint is resolved,
  // so return the caller's generic default to keep that possibility open.
  if (decl) {
    if (auto structOp =
            dyn_cast_or_null<StructDeclOp>(decl->getIfOperation())) {
      if (structOp.getRegisterPassableConstraintAttr())
        return genericDefault;
    }
  }

  return TypeConvention::MemoryOnly;
}

/// Return the StructDeclOp::RegisterPassable enum for this type.
TypeConvention ASTType::getRegisterPassability(llvm::SMLoc loc,
                                               SharedState &shared) const {
  // If this is a generic type, we treat it as memory only. If the metatype
  // is a parameter reference, then pessimistically assume it is memory-only.
  return ::getRegisterPassability(*this, loc, shared,
                                  TypeConvention::MemoryOnly);
}

/// Return true if this type is a 'trivial' type, that is one that can be
/// passed around by copying the bits, and whose destructor is a noop.
bool ASTType::isTrivial(llvm::SMLoc loc, SharedState &shared) const {
  return getRegisterPassability(loc, shared) ==
         TypeConvention::RegisterPassableTrivial;
}

TriBool ASTType::isSpecialFunctionTrivial(llvm::SMLoc loc,
                                          SpecialFunctionKind kind,
                                          SharedState &shared) const {
  // MLIR types and types conforming to AnyTrivialRegType are assumed to be
  // trivial for all purposes
  if (isTrivialRegisterType(loc, shared))
    return TriBool::yes();

  StringRef traitName;
  StringRef isTrivialHook;
  switch (kind) {
  default:
    llvm_unreachable("Invalid special function kind");
  case SpecialFunctionKind::kDeinit:
    traitName = "Deinitable";
    isTrivialHook = "__del__is_trivial";
    break;
  case SpecialFunctionKind::kCopyCtor:
    traitName = "Copyable";
    isTrivialHook = "__copy_ctor_is_trivial";
    break;
  case SpecialFunctionKind::kMoveCtor:
    traitName = "Movable";
    isTrivialHook = "__move_ctor_is_trivial";
    break;
  }

  ASTDecl *typeDecl = getDecl(shared);
  assert(typeDecl && "MLIR types shouldn't reach here");

  // If it doesn't conform to the corresponding trait. Only return `no`
  // (provably non-trivial) when we can definitively prove non-conformance.
  auto [conformanceResult, traitDecl] =
      conformsToBuiltinTrait(traitName, loc, shared, {});
  if (conformanceResult.isFalse())
    return TriBool::no();

  if (!typeDecl->getParentDecl())
    return TriBool::unknown();

  auto witnessName = StringAttr::get(shared.getContext(), isTrivialHook);
  auto traitSymbol = TraitSymbolAttr::get(traitDecl->getSymbolRef());

  ASTType boolType =
      shared.lookupBuiltinType("Bool", *typeDecl->getParentDecl(), loc);
  TypedAttr fieldIsTrivial =
      shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
          PValue(*this), traitSymbol, witnessName, boolType);

  auto structAttr = dyn_cast_if_present<LITStructAttr>(fieldIsTrivial);
  if (!structAttr)
    return TriBool::unknown();

  assert(structAttr.getType() == boolType);
  auto structVals = structAttr.getValues();
  if (structVals.size() != 1)
    return TriBool::unknown();

  if (auto &[name, boolVal] = structVals.front(); name == "_mlir_value") {
    if (auto boolAttr = dyn_cast<SIMDAttr>(boolVal))
      return boolAttr.getAsBool() ? TriBool::yes() : TriBool::no();
  }

  return TriBool::unknown();
}

bool ASTType::isProvablyImplicitlyTriviallyCopyable(llvm::SMLoc loc,
                                                    SharedState &shared,
                                                    ASTDecl &scope) const {
  return isImplicitlyCopyable(loc, shared, scope) &&
         isSpecialFunctionTrivial(loc, SpecialFunctionKind::kCopyCtor,
                                  shared) == TriBool::yes();
}

bool ASTType::isProvablyTriviallyMoveable(llvm::SMLoc loc,
                                          SharedState &shared) const {
  return isSpecialFunctionTrivial(loc, SpecialFunctionKind::kMoveCtor,
                                  shared) == TriBool::yes();
}

bool ASTType::isProvablyTriviallyDeletable(llvm::SMLoc loc,
                                           SharedState &shared) const {
  return isSpecialFunctionTrivial(loc, SpecialFunctionKind::kDeinit, shared) ==
         TriBool::yes();
}

/// Return true if this type is a register-passable type that can be passed
/// around and copied in SSA values instead of having to live in memory.
///
/// The location specifies the location of the reference in case the use is
/// invalid in this location.
bool ASTType::isRegisterPassable(llvm::SMLoc loc, SharedState &shared) const {
  TypeConvention convention = getRegisterPassability(loc, shared);
  return convention == TypeConvention::RegisterPassable ||
         convention == TypeConvention::RegisterPassableTrivial;
}

/// Return true if this type is RegisterPassable or if it is a generic type
/// that could bind to a concrete RegisterPassable type.
bool ASTType::mightBeRegisterPassable(llvm::SMLoc loc,
                                      SharedState &shared) const {
  // If this is a generic type, we treat it as register passable conservatively.
  return ::getRegisterPassability(*this, loc, shared,
                                  TypeConvention::RegisterPassable) !=
         TypeConvention::MemoryOnly;
}

bool ASTType::isCopyable(llvm::SMLoc loc, SharedState &shared, bool isImplicit,
                         ASTDecl &scope) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR Types are copyable.

  // If the type is trivial, then it is copyable.
  if (isTrivial(loc, shared))
    return true;

  StringRef traitName = isImplicit ? "ImplicitlyCopyable" : "Copyable";

  // Conservative: only claim copyable when provably so. If conformance
  // depends on unresolved constraints (`unknown`), return false and let
  // the constraint system prove it in the appropriate context.
  return provenConformsToBuiltinTrait(traitName, typeDecl->getLoc(), shared,
                                      ASTDecl::getAssumptionsFromScope(&scope));
}

/// Return true if this type is implicitly copyable, either because it is
/// trivial or conforms to ImplicitlyCopyable trait. Note: this resolves the
/// body of a struct type.
bool ASTType::isImplicitlyCopyable(llvm::SMLoc loc, SharedState &shared,
                                   ASTDecl &scope) const {
  return isCopyable(loc, shared, /*isImplicit=*/true, scope);
}

/// Return true if this type is explicitly copyable, either because it is
/// trivial or conforms to the Copyable trait. Note: this resolves the
/// body of a struct type.
bool ASTType::isExplicitlyCopyable(llvm::SMLoc loc, SharedState &shared,
                                   ASTDecl &scope) const {
  return isCopyable(loc, shared, /*isImplicit=*/false, scope);
}

/// Return true if this type is movable from its own type, either because it
/// is trivial or has a move constructor from self. Note: this resolves the
/// body of a struct type.
bool ASTType::isMovable(llvm::SMLoc loc, SharedState &shared,
                        ASTDecl &scope) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR types are movable.

  // If the type is register-passable, it is trivially movable.
  if (isRegisterPassable(loc, shared))
    return true;

  // Check whether the type conforms to `Movable` trait.  Use
  // conformsToBuiltinTrait (not doesNominalTypeConformTo directly) so that
  // concrete parameter bindings are available to evaluate conditional
  // conformance constraints — matching isCopyable's behavior.
  return provenConformsToBuiltinTrait("Movable", typeDecl->getLoc(), shared,
                                      ASTDecl::getAssumptionsFromScope(&scope));
}

TriBool
ASTType::doesConformTo(TraitType trait, SharedState &shared,
                       ArrayRef<ConstraintAttr> callerAssumptions) const {
  // FIXME: this seems pretty wrong, `getDecl` is type depth insensitive,
  // meaning that it is true for `meta<meta<!struct>> conforms_to AnyType`...
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return TriBool::no();
  return typeDecl->doesNominalTypeConformTo(trait, *this, callerAssumptions);
}

ConstraintResult ASTType::doesConformToWithDetails(
    TraitType trait, SharedState &shared,
    ArrayRef<ConstraintAttr> callerAssumptions) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return ConstraintResult::no({});
  return typeDecl->doesNominalTypeConformToWithDetails(trait, *this,
                                                       callerAssumptions);
}

/// Given a standard trait like Copyable, look up the conformance.  On
/// success, the ASTDecl of the trait itself is returned, it is otherwise
/// null.
std::pair<TriBool, ASTDecl *> ASTType::conformsToBuiltinTrait(
    StringRef traitName, llvm::SMLoc loc, SharedState &shared,
    ArrayRef<ConstraintAttr> callerAssumptions) const {
  ASTDecl *traitDecl = shared.lookupBuiltinTrait(traitName, loc);
  if (!traitDecl)
    return {TriBool::no(), {}};

  auto trait = dyn_cast_or_null<TraitDeclOp>(traitDecl->getIfOperation());
  if (!trait)
    return {TriBool::no(), {}};

  // Micro optimization to avoid creating a canonical trait type for for
  // checking conformance, we only care the root symbol.
  return {doesConformTo(TraitType::get(getFullyResolvedSymbolRef(trait)),
                        shared, callerAssumptions),
          traitDecl};
}

/// This returns true if the current type unconditionally conforms to the
/// specified builtin trait, e.g. "Movable".
bool ASTType::provenConformsToBuiltinTrait(
    StringRef traitName, llvm::SMLoc loc, SharedState &shared,
    ArrayRef<ConstraintAttr> callerAssumptions) const {
  auto [conformanceResult, traitDecl] =
      conformsToBuiltinTrait(traitName, loc, shared, callerAssumptions);
  return conformanceResult.isTrue();
}

bool ASTType::isRegisterType(llvm::SMLoc loc, SharedState &shared) const {
  TypeConvention convention = getRegisterPassability(loc, shared);
  return convention == TypeConvention::RegisterPassable ||
         convention == TypeConvention::RegisterPassableTrivial;
}

bool ASTType::isTrivialRegisterType(llvm::SMLoc loc,
                                    SharedState &shared) const {
  return getRegisterPassability(loc, shared) ==
         TypeConvention::RegisterPassableTrivial;
}

/// Given a reference, return the element as an ASTType.  This aborts
/// if the current type isn't a reference.
///
ASTType ASTType::getReferenceElementType() const {
  return ASTType(sugarCast<RefType>(mlirType).getElementType());
}

RefType ASTType::VariadicListInfo::getElementRefType() const {
  return RefType::get(elementType, origin);
}

/// Given a type of ParameterList, return the element type and the values
/// list that are bound to it.
ASTType::ParameterListInfo ASTType::getParameterListInfo() const {
  auto structType = sugarDynCast<LIT::StructType>(*this);
  if (!structType)
    return {{}, {}};

  if (structType.getSymbol().getLeafReference().strref() != "ParameterList" &&
      structType.getSymbol().getLeafReference().strref() != "TypeList")
    return {{}, {}};

  auto bindings = getParamBindings();
  assert(bindings.size() == 2 &&
         (sugarIsa<TraitType, AnyTraitType>(
             bindings[0].getType())) &&                    // elementType
         sugarIsa<ParamListType>(bindings[1].getType()) && // values
         "Not a VariadicList struct?");

  return {ASTType(bindings[0]), bindings[1]};
}

/// Given a VariadicList, return the element type from it.
ASTType::VariadicListInfo ASTType::getVariadicListInfo() const {
  assert(!isa<RefType>(mlirType) && "looking at a RefType not a VariadicList");
  auto bindings = getParamBindings();
  assert(bindings.size() == 5 &&
         sugarIsa<LIT::StructType>(bindings[0].getType()) && // Bool
         sugarIsa<OriginType>(bindings[1].getType()) &&
         sugarIsa<LIT::StructType>(bindings[2].getType()) && // Origin
         sugarIsa<LIT::TraitType>(bindings[3].getType()) &&  // AnyType
         sugarIsa<LIT::StructType>(bindings[4].getType()) && // Bool
         "Not a VariadicList struct?");

  // The "owned" bit is guaranteed to be a constant boolean.
  auto isOwned = cast<SIMDAttr>(
      std::get<1>(sugarCast<LITStructAttr>(bindings[4]).getValues()[0]));
  return {ASTType(bindings[3]), bindings[1], isOwned.getAsBool()};
}

/// Return the RefPackType that corresponds to the VariadicPack instance.
RefPackType ASTType::getVariadicPackInfo(SharedState &shared) const {
  assert(!isa<RefType>(mlirType) && "looking at a RefType not a VariadicPack");
  auto bindings = getParamBindings();
  assert(bindings.size() == 7 &&
         sugarIsa<LIT::StructType>(bindings[0].getType()) && // elt_is_mut
         sugarIsa<OriginType>(bindings[1].getType()) &&      // mlir_origin
         sugarIsa<LIT::StructType>(bindings[2].getType()) && // Origin
         sugarIsa<AnyTraitType>(bindings[3].getType()) &&    // element_trait
         sugarIsa<ParamListType>(bindings[4].getType()) &&   // elt_types.value
         sugarIsa<LIT::StructType>(bindings[5].getType()) && // is_owned
         sugarIsa<LIT::StructType>(bindings[6].getType()) && // element_types
         "Not a VariadicPack struct?");
  return RefPackType::get(
      /*variadicList*/ bindings[4], /*mlirOrigin*/ bindings[1],
      /*addrSpace*/ IntegerAttr::get(IndexType::get(shared.getContext()), 0));
}

/// Decode the parameters list of VariadicPack.
ASTType::VariadicPackInfo ASTType::getVariadicPackInfo() const {
  assert(!isa<RefType>(mlirType) && "looking at a RefType not a VariadicPack");
  auto bindings = getParamBindings();
  assert(bindings.size() == 7 &&
         sugarIsa<LIT::StructType>(bindings[0].getType()) && // elt_is_mut
         sugarIsa<OriginType>(bindings[1].getType()) &&      // mlir_origin
         sugarIsa<LIT::StructType>(bindings[2].getType()) && // Origin
         sugarIsa<AnyTraitType>(bindings[3].getType()) &&    // element_trait
         sugarIsa<ParamListType>(bindings[4].getType()) &&   // elt_types.value
         sugarIsa<LIT::StructType>(bindings[5].getType()) && // is_owned
         sugarIsa<LIT::StructType>(bindings[6].getType()) && // element_types
         "Not a VariadicPack struct?");
  VariadicPackInfo result;
  result.typeList = bindings[4];
  result.typeListStruct = bindings[6];
  result.isOwned = bindings[5];
  return result;
}

ASTType ASTType::getKwargsDictValueType() const {
  return ASTType(getParamBindings()[0]);
}

ASTType ASTType::getKwargsDictRefValueType() const {
  return getReferenceElementType().getKwargsDictValueType();
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
ASTType ASTType::getSignatureUserResultType() const {
  auto sigGenType = FnOrFnLiteralTypeGeneratorType::get(mlirType);
  return LIT::getSignatureUserResultType(sigGenType, sigGenType.getArguments(),
                                         sigGenType.getResults().front());
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const {
  llvm::errs() << getAsString(ASTTypePrinterContext{}) << '\n';
}

RefType ASTType::getRefForArgument(const Twine &argName, bool isMut) {
  auto ctx = mlirType.getContext();
  auto selfOrigin = ParamDeclRefAttr::get(StringAttr::get(ctx, argName + "`"),
                                          OriginType::get(ctx, isMut));
  return RefType::get(mlirType, selfOrigin, /*addressSpace=*/0);
}

/// If this type is parameterized, and if any of the parameters refer to a
/// ParamIndexRefAttr, replace it with an UnboundAttr so parameter inference
/// will infer it.
///
/// This makes parameter inference sensitive to what to propagate vs infer. For
/// example, if expectedType is known to be 'SIMD[uint8, 1]', then we can infer
/// which constructor to use when the input is an IntLiteral.
///
/// On the other hand, if expectedType is something like 'SIMD[?, 1]' and the
/// argument is an Int8, then we need the implicit conversion to infer the
/// base element.  Our solution to this is to rip and replace parameters that
/// contain unbound parameters, replacing them with UnboundAttr so inference
/// can find them.
ASTType ASTType::getWithUnknownParametersReplaced(SharedState &shared) const {
  // If this is a struct type, try unbinding just the parameters that have
  // parameter references in it.
  if (auto drt = sugarDynCast<StructType>(mlirType)) {
    ParamIndexRefAttrFinder finder;

    // Otherwise, check each bound parameter to see if it is unknown.  If so,
    // replace it.
    SmallVector<TypedAttr> newParams;
    bool anyBound = false;
    for (auto curValue : getParamBindings()) {
      if (!finder.hasReferences(curValue)) {
        // Keep this value if it has no references.
        anyBound = true;
      } else {
        // Keep the argument type if it has no references.
        Type paramType = curValue.getType();
        if (finder.hasReferences(paramType))
          paramType =
              ASTType(UnboundAttr::get(TypeType::get(paramType.getContext())));
        curValue = UnboundAttr::get(paramType);
      }
      newParams.push_back(curValue);
    }

    if (anyBound)
      return cast<StructDeclOp>(getDecl(shared)->getIfOperation())
          .bindReference(newParams);
  }

  // Otherwise return it with all parameters replaced.
  if (Type nonParam = getWithoutParameters(shared))
    return nonParam;
  return *this;
}

/// Return true if this type contains any origins that are unmaterializable
/// from comptime to runtime. Consider some code like this:
///
///   alias ptr = String("foo"+"bar").unsafe_ptr()
///   alias elt1 = ptr[0] # Yields "f", which works fine.
///   # This can't work.
///   var runtime_ptr = ptr
///
bool ASTType::containsUnmaterializableOrigins(SharedState &shared) const {
  for (auto o :
       shared.cachedOriginFinder.findOriginsIn(getCanonicalType(mlirType))) {

    o = OriginType::stripMutCastAndRebind(o);

    // Ignore field sensitivity.
    while (auto field = dyn_cast<OriginFieldAttr>(o))
      o = field.getBase();

    // Actually global memory /can/ be materialized, so that it totally fine. We
    // allow AnyOriginAttr because it is the general "disable checking" origin.
    // Banning it prevents many important patterns from working, e.g. default
    // arguments of null UnsafePointer.
    if (sugarIsa<StaticOriginAttr, AnyOriginAttr>(o))
      continue;

    // We can materialize parametric origins from a caller.
    if (sugarIsa<ParamDeclRefAttr, ImplicitOriginRefAttr>(o))
      continue;

    // Ignore interior origins.
    // FIXME: figure out their semantics.
    if (sugarIsa<InteriorOriginAttr>(o))
      continue;

    // Otherwise, it is something we can't track.
    return true;
  }

  return false;
}

/// Return true if this type is a singleton type. This is a type that has one
/// value, e.g. a !lit.origin or a struct whose fields are all singleton types.
bool ASTType::isSingleton(SharedState &shared) const {
  // Raw lit.origin's and origin sets are always singleton types.
  if (sugarIsa<OriginType, OriginSetType>(mlirType))
    return true;

  if (auto structType = sugarDynCast<LIT::StructType>(mlirType)) {
    ASTDecl *decl = getDecl(shared);
    assert(decl && "struct decl not found");
    if (failed(shared.declResolver->resolveBody(*decl, decl->getLoc())))
      return false;
    auto structDecl = cast<StructDeclOp>(decl->getIfOperation());
    return structDecl.field_begin() == structDecl.field_end();
  }

  return false;
}
