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

#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// LITDialect
//===----------------------------------------------------------------------===//

void LITDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Mojo/LITDialect/LITAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// FnMetaOriginDataAttr
//===----------------------------------------------------------------------===//

static ParseResult parseCaptureOrigins(AsmParser &p, TypedAttr &origins) {
  auto type = OriginSetType::get(p.getContext());
  if (failed(p.parseOptionalComma())) {
    origins = OriginSetAttr::get({}, type);
    return success();
  }
  return parseParamValue(p, origins, type);
}

static void printCaptureOrigins(AsmPrinter &p, TypedAttr origins) {
  if (isEmptyOriginSet(origins))
    return;
  p << ", ";
  printParamValue(p, origins);
}

FnMetaOriginDataAttr FnMetaOriginDataAttr::get(MLIRContext *context) {
  return FnMetaOriginDataAttr::get(
      context, /*numImplicitOriginDecls=*/0,
      /*captureOrigins=*/OriginSetAttr::get(context, {}),
      /*isNestedOriginsReadOnly=*/false,
      /*definesInteriorOrigins=*/false);
}

FnMetaOriginDataAttr FnMetaOriginDataAttr::get(MLIRContext *ctx,
                                               size_t numImplicitOriginDecls,
                                               TypedAttr captureOrigins,
                                               bool isNestedOriginsReadOnly,
                                               bool definesInteriorOrigins) {
  if (!captureOrigins)
    captureOrigins = OriginSetAttr::get(ctx, {});
  return Base::get(ctx, numImplicitOriginDecls, captureOrigins,
                   isNestedOriginsReadOnly, definesInteriorOrigins);
}

SmallVector<VariadicKind>
LIT::getContextualVariadicParams(ArrayRef<Operation *> ops) {
  SmallVector<VariadicKind> variadics;
  for (Operation *op : ops) {
    // If we are dealing with a struct, trait, or extension, we concatenate
    // their variadic masks. For extensions, use the target struct's variadic
    // information.
    PogListAttr paramListAttr;
    if (auto structDecl = ::dyn_cast<StructDeclOp>(op))
      paramListAttr = structDecl.getSignature().getParamListAttrs();
    else if (auto traitDecl = ::dyn_cast<TraitDeclOp>(op))
      paramListAttr = traitDecl.getSignature().getParamListAttrs();
    else if (auto extensionDecl = ::dyn_cast<ExtensionDeclOp>(op)) {
      // Extensions inherit parameters from their target struct.
      // The MojoParser already mirrors these parameters during resolution.
      paramListAttr = extensionDecl.getSignature().getParamListAttrs();
    } else
      continue;

    for (PogMetadataAttr pogAttr : paramListAttr.getPogs())
      variadics.emplace_back(pogAttr.getVariadic());
  }
  return variadics;
}

LogicalResult FnMetaOriginDataAttr::verifyFuncType(
    function_ref<InFlightDiagnostic()> emitError, FunctionType values,
    ArrayRef<ArgConvention> argConventions, FnEffects effects) const {
  // Verify input conventions.
  size_t numInputs = values.getNumInputs();
  if (size_t numArgConv = argConventions.size(); numInputs != numArgConv) {
    return emitError()
           << "number of arguments does not match number of input conventions: "
           << numInputs << " != " << numArgConv;
  }

  for (auto [i, argType, conv] :
       llvm::enumerate(values.getInputs(), argConventions)) {
    if (conv == ArgConvention::ByRefResult && i != values.getNumInputs() - 1)
      return emitError() << "'byref_result' argument must be the last argument";

    // Verify argument conventions.
    if (hasAddress(conv) && !::isa<RefType>(argType)) {
      return emitError()
             << "argument #" << i << " with convention '" << stringifyEnum(conv)
             << "' in signature type should be a `!lit.ref` but got: "
             << argType;
    }
  }

  return success();
}

FnMetaOriginDataAttr
FnMetaOriginDataAttr::addCaptureOrigins(TypedAttr origins) {
  auto type = OriginType::get(getContext(), /*isMutable=*/true);
  SmallVector<TypedAttr> originUnion{
      OriginSetUnionAttr::get(getCaptureOrigins(), type),
      OriginSetUnionAttr::get(origins, type)};
  return get(getContext(), getNumImplicitOriginDecls(),
             OriginSetAttr::get(getContext(), originUnion),
             getIsNestedOriginsReadOnly(), getDefinesInteriorOrigins());
}

void FnMetaOriginDataAttr::printFuncTypeBody(AsmPrinter &p,
                                             FuncType sig) const {

  p << '<';
  printFnType(p, sig);
  p << '>';
}

void FnMetaOriginDataAttr::printFuncType(AsmPrinter &p, FuncType sig) const {
  p << "!lit.fn";
  printFuncTypeBody(p, sig);
}

void FnMetaOriginDataAttr::printFuncTypeInline(AsmPrinter &p,
                                               FuncType sig) const {
  printFnType(p, sig);
}

bool FnMetaOriginDataAttr::equals(FnMetadataAttrInterface otherMetadata) const {
  auto other = ::dyn_cast<FnMetaOriginDataAttr>(otherMetadata);
  if (!other)
    return false;

  return getNumImplicitOriginDecls() == other.getNumImplicitOriginDecls() &&
         getCaptureOrigins() == other.getCaptureOrigins() &&
         getIsNestedOriginsReadOnly() == other.getIsNestedOriginsReadOnly() &&
         getDefinesInteriorOrigins() == other.getDefinesInteriorOrigins();
}

//===----------------------------------------------------------------------===//
// UnboundMLIROperationAttr
//===----------------------------------------------------------------------===//

Type UnboundMLIROperationAttr::getType() const {
  return mlir::NoneType::get(getContext());
}

//===----------------------------------------------------------------------===//
// EllipsisAttr
//===----------------------------------------------------------------------===//

Type EllipsisAttr::getType() const { return EllipsisType::get(getContext()); }

//===----------------------------------------------------------------------===//
// UnpackedAttr
//===----------------------------------------------------------------------===//

static ParseResult parseUnpackKind(AsmParser &p, bool &kwOnly) {
  if (succeeded(p.parseOptionalKeyword("kw"))) {
    kwOnly = true;
    return success();
  }
  kwOnly = false;
  return p.parseKeyword("pos");
}

static void printUnpackKind(AsmPrinter &p, bool kwOnly) {
  p << (kwOnly ? "kw" : "pos");
}

//===----------------------------------------------------------------------===//
// StaticOriginAttr
//===----------------------------------------------------------------------===//

// Origins are treated as simple constants, allowing folding of function calls.
bool StaticOriginAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ComptimeOriginAttr
//===----------------------------------------------------------------------===//

// Origins are treated as simple constants, allowing folding of function calls.
bool ComptimeOriginAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// OriginEqAttr
//===----------------------------------------------------------------------===//

TypedAttr OriginEqAttr::get(TypedAttr lhs, TypedAttr rhs) {
  assert(isa<OriginType>(lhs.getType()) && isa<OriginType>(rhs.getType()) &&
         "only works with origins");
  if (lhs == rhs)
    return BoolAttr::get(rhs.getContext(), true);

  // If we can tell they are different origins, return false.
  if (ParameterAttr::isSimpleConstant(getCanonicalAttr(lhs)) &&
      ParameterAttr::isSimpleConstant(getCanonicalAttr(rhs)))
    return BoolAttr::get(rhs.getContext(), false);

  // Otherwise keep symbolic.
  return OriginEqAttr::Base::get(lhs.getContext(), lhs, rhs);
}

Type OriginEqAttr::getType() const { return IntegerType::get(getContext(), 1); }

//===----------------------------------------------------------------------===//
// OriginUnionAttr
//===----------------------------------------------------------------------===//

OriginUnionAttr OriginUnionAttr::getFromBytecode(ArrayRef<TypedAttr> operands,
                                                 OriginType type) {
  return Base::get(type.getContext(), operands, type);
}

static bool unionArgCompare(TypedAttr lhs, TypedAttr rhs) {
  // Ignore OriginMutCastAttr's for comparison.
  return ParameterAttr::compare(OriginMutCastAttr::strip(lhs),
                                OriginMutCastAttr::strip(rhs));
}

static void removeDuplicates(SmallVectorImpl<TypedAttr> &operands) {
  if (operands.size() > 1) {
    for (size_t i = 0, e = operands.size() - 1; i != e; ++i) {
      if (operands[i] != operands[i + 1])
        continue;

      operands.erase(operands.begin() + i + 1);
      --e, --i;
    }
  }
}

// Canonicalize the operands, sorting by name/index and eliminating the union
// entirely if possible.
TypedAttr OriginUnionAttr::get(ArrayRef<TypedAttr> operandsIn,
                               OriginType type) {
  // If obviously a single element, just return it.
  if (operandsIn.size() == 1)
    return operandsIn[0];

  llvm::SetVector<TypedAttr> operandSet;
  llvm::SmallPtrSet<Attribute, 2> subtreeOrigins;

  // Preprocess operands.
  for (TypedAttr operand : operandsIn) {
    assert(operand.getType() == type &&
           "all members of a origin union must have matching type");
    // Union{a, b, #lit.any.origin} => #lit.any.origin since #lit.origin
    // represents "possibly anything".
    if (sugarIsa<AnyOriginAttr>(operand))
      return operand;

    // Flatten any of the same operation into the operand list:
    // `(union x, (union y, z))` => `(union x, y, z)`.
    if (auto subexpr = sugarDynCast<OriginUnionAttr>(operand)) {
      operandSet.insert(subexpr.getOperands().begin(),
                        subexpr.getOperands().end());
    } else {
      operandSet.insert(operand);
    }

    // Look through mutcasts: mixed-mutability unions wrap members in
    // OriginMutCastAttr (e.g. {imm x, mut y~}), and absorption must still
    // see the underlying subtree.
    if (auto subtree = dyn_cast<OriginSubtreeAttr>(
            OriginType::stripMutCastAndRebind(operand)))
      subtreeOrigins.insert(subtree.getBase());
  }

  // If we have subtree origins, absorb operands that are already covered by a
  // wider subtree view and re-canonicalize.
  if (!subtreeOrigins.empty()) {
    for (TypedAttr operand : operandSet) {
      // Walk up the list of fields and indirect origins, checking to see if we
      // have a subtree that contains the whole thing, e.g. we want to know that
      // x.y.z is contained within x~ and x.y~.  'subtree' origins will contain
      // the origins for which we have a subtree in the set (it contains their
      // bases).
      auto operandAsSubtree = dyn_cast<OriginSubtreeAttr>(
          OriginType::stripMutCastAndRebind(operand));
      bool isAbsorbed = false;
      operand.walk([&](TypedAttr subOperand) {
        // If this suboperand isn't contained within a subtree, ignore it.
        if (!subtreeOrigins.contains(subOperand))
          return WalkResult::advance();

        // If "operand" it itself a subtree x.y.z~ then don't stop just because
        // we found x.y.z.  However, do stop if we see a x.y~ also in the set.
        if (operandAsSubtree && operandAsSubtree.getBase() == subOperand)
          return WalkResult::advance();

        isAbsorbed = true;
        return WalkResult::interrupt();
      });
      if (isAbsorbed) {
        operandSet.remove(operand);
        return OriginUnionAttr::get(operandSet.takeVector(), type);
      }
    }
  }

  SmallVector<TypedAttr> operands = operandSet.takeVector();

  // If one result, use it.
  if (operands.size() == 1)
    return operands[0];

  // Impose an ordering on the operands, sorting by name where possible - but
  // predictably ordered w.r.t. each other.
  llvm::stable_sort(operands, unionArgCompare);

  // Otherwise, for multiple elements actually form a union.
  return OriginUnionAttr::Base::get(type.getContext(), operands, type);
}

TypedAttr OriginUnionAttr::get(MLIRContext *ctx, ArrayRef<TypedAttr> origins) {
  // In the empty case, the origin is immutable.
  if (origins.empty())
    return OriginUnionAttr::get(OriginType::get(ctx, /*mutable=*/false));
  if (origins.size() == 1)
    return origins.front();

  // Determine the overall mutability of the union, as the intersection of the
  // mutabilities of the members.
  bool needMutCast = false;
  auto getMut = [&](TypedAttr origin) {
    // If the type of a member is sugared, force cast it to a consistent origin
    // type so printing and parsing will work correctly.
    if (!isa<OriginType>(origin.getType()))
      needMutCast = true;

    auto isMut = sugarCast<OriginType>(origin.getType()).getIsMutable();
    // Remove any sugar from the mutability to get a pure scalar<bool>.
    if (!isScalarOf<KGENDType::kBool>(isMut.getType()))
      isMut = ParamOperatorAttr::getRebind(
          isMut, SIMDType::get(isMut.getContext(), 1, KGENDType::kBool));
    return isMut;
  };

  // The resultant mutability is the worst case of the input mutabilities.
  TypedAttr mutability = getMut(origins.front());
  for (TypedAttr other : origins.drop_front()) {
    TypedAttr otherMut = getMut(other);
    if (otherMut == mutability)
      continue;
    mutability = ParamOperatorAttr::get(POC::And, mutability, otherMut);
    needMutCast = true;
  }

  auto resultType = OriginType::get(mutability);
  // Add any OriginMutCastAttr wrappers as needed.
  SmallVector<TypedAttr> newOrigins;
  if (needMutCast) {
    for (TypedAttr origin : origins)
      newOrigins.push_back(OriginMutCastAttr::get(origin, resultType));
    origins = newOrigins;
  }

  return OriginUnionAttr::get(origins, resultType);
}

// Origins unions are simple constants if all their elements are.
bool OriginUnionAttr::isConstant() const {
  for (auto op : getOperands())
    if (!ParameterAttr::isSimpleConstant(op))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// OriginMutCastAttr
//===----------------------------------------------------------------------===//

bool OriginMutCastAttr::isLessThan(Attribute rhs) const {
  // Compare the underlying references.
  auto cast = ::cast<OriginMutCastAttr>(rhs);
  return ParameterAttr::compare(getOperand(), cast.getOperand());
}

OriginMutCastAttr OriginMutCastAttr::getFromBytecode(TypedAttr operand,
                                                     OriginType type) {
  return Base::get(type.getContext(), operand, type);
}

TypedAttr OriginMutCastAttr::get(TypedAttr operand, TypedAttr isMutable) {
  auto curTy = sugarCast<OriginType>(operand.getType());
  if (curTy.isMutable() == isMutable)
    return operand;

  // Fold some common cases to canonicalize.
  // mutcast(mutcast(x)) -> mutcast(x), often canceling out.
  if (auto mutCast = sugarDynCast<OriginMutCastAttr>(operand))
    return get(mutCast.getOperand(), isMutable);

  // Singletons don't need a cast, just form one with the new mutability.
  if (sugarIsa<AnyOriginAttr>(operand))
    return AnyOriginAttr::get(isMutable);
  if (sugarIsa<ComptimeOriginAttr>(operand))
    return ComptimeOriginAttr::get(isMutable);

  // Push into union so it cancels out.
  if (auto unionAttr = sugarDynCast<OriginUnionAttr>(operand)) {
    SmallVector<TypedAttr> elts;
    for (auto elt : unionAttr.getOperands())
      elts.push_back(OriginMutCastAttr::get(elt, isMutable));
    return OriginUnionAttr::get(elts, OriginType::get(isMutable));
  }

  auto context = operand.getContext();
  return OriginMutCastAttr::Base::get(context, operand,
                                      OriginType::get(isMutable));
}

TypedAttr OriginMutCastAttr::get(TypedAttr operand, Type type) {
  if (operand.getType() == type)
    return operand;

  auto attr = get(operand, sugarCast<OriginType>(type).isMutable());
  // Ensure sugar lines up correctly.
  return ParamOperatorAttr::getRebind(attr, type);
}

TypedAttr OriginMutCastAttr::get(TypedAttr operand, bool isMutable) {
  auto curTy = sugarCast<OriginType>(operand.getType());
  if (curTy.isMutableKnown(isMutable))
    return operand;
  return get(operand, SIMDAttr::getScalarBool(operand.getContext(), isMutable));
}

// Casts are simple constants if their base is.
bool OriginMutCastAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getOperand());
}

//===----------------------------------------------------------------------===//
// OriginFieldAttr
//===----------------------------------------------------------------------===//

TypedAttr OriginFieldAttr::get(TypedAttr structOrigin, StringAttr field) {
  // Check to see if there are any permutations we can fold.

  // If we have the global mutable origin, treat it conservatively by
  // returning the global mutable origin.  It isn't wise to try to derive
  // information from something where origins have been casted away.
  if (sugarIsa<AnyOriginAttr>(structOrigin))
    return structOrigin;

  // Cannot form field(subtree(x)) since it erases fields.
  assert(!sugarIsa<OriginSubtreeAttr>(structOrigin) &&
         "field origin cannot be applied to a subtree");

  // We push any mutability casts outside of ourselves.
  //     mutcast(x).myfield => mutcast(x.myfield)
  if (auto mutCast = sugarDynCast<OriginMutCastAttr>(structOrigin)) {
    auto inner = OriginFieldAttr::get(mutCast.getOperand(), field);
    return OriginMutCastAttr::get(inner, mutCast.getType());
  }

  // We push this inside a origin.union as well, so we get the union on the
  // outside.
  if (auto unionAttr = sugarDynCast<OriginUnionAttr>(structOrigin)) {
    SmallVector<TypedAttr> elts;
    for (auto elt : unionAttr.getOperands())
      elts.push_back(OriginFieldAttr::get(elt, field));
    // Field accesses don't affect mutability, so we use the same type.
    return OriginUnionAttr::get(elts, unionAttr.getType());
  }

  // The structOriginRef must have a OriginType, which we propagate.
  auto structType = sugarCast<OriginType>(structOrigin.getType());
  return OriginFieldAttr::Base::get(structOrigin.getContext(), structOrigin,
                                    field, structType);
}

OriginFieldAttr OriginFieldAttr::getFromBytecode(TypedAttr structOrigin,
                                                 StringAttr field,
                                                 OriginType type) {
  return Base::get(type.getContext(), structOrigin, field, type);
}

// Fields are simple constants if their base is.
bool OriginFieldAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getBase());
}

bool OriginFieldAttr::isLessThan(Attribute rhs) const {
  auto rhsField = ::cast<OriginFieldAttr>(rhs);
  if (getBase() == rhsField.getBase())
    return getField().getValue() < rhsField.getField().getValue();
  return ParameterAttr::compare(getBase(), rhsField.getBase());
}

//===----------------------------------------------------------------------===//
// InteriorOriginAttr
//===----------------------------------------------------------------------===//

TypedAttr InteriorOriginAttr::get(TypedAttr baseOrigin, StringRef userName) {
  auto *ctx = baseOrigin.getContext();
  TypedAttr userNameAttr = StringAttr::get(userName, StringType::get(ctx));
  return get(baseOrigin, userNameAttr);
}

TypedAttr InteriorOriginAttr::get(TypedAttr baseOrigin, TypedAttr userName) {
  // Check to see if there are any permutations we can fold.

  // If we have the global mutable origin, treat it conservatively by
  // returning the global mutable origin.  It isn't wise to try to derive
  // information from something where origins have been casted away.
  if (sugarIsa<AnyOriginAttr>(baseOrigin))
    return baseOrigin;

  // Cannot form interior(subtree(x)) since it erases interiors.
  assert(!sugarIsa<OriginSubtreeAttr>(baseOrigin) &&
         "interior origin cannot be applied to asubtree");

  // We push any mutability casts outside of ourselves.
  //     mutcast(x)[] => mutcast(x[])
  if (auto mutCast = sugarDynCast<OriginMutCastAttr>(baseOrigin)) {
    auto inner = InteriorOriginAttr::get(mutCast.getOperand(), userName);
    return OriginMutCastAttr::get(inner, mutCast.getType());
  }

  // We push this inside a origin.union as well, so we get the union on the
  // outside.
  if (auto unionAttr = sugarDynCast<OriginUnionAttr>(baseOrigin)) {
    SmallVector<TypedAttr> elts;
    for (auto elt : unionAttr.getOperands())
      elts.push_back(InteriorOriginAttr::get(elt, userName));
    // Field accesses don't affect mutability, so we use the same type.
    return OriginUnionAttr::get(elts, unionAttr.getType());
  }

  // The result type is the same as baseOrigin's.
  auto baseType = ::cast<OriginType>(baseOrigin.getType());
  return InteriorOriginAttr::Base::get(baseOrigin.getContext(), baseOrigin,
                                       userName, baseType);
}

InteriorOriginAttr InteriorOriginAttr::getFromBytecode(TypedAttr base,
                                                       TypedAttr userName,
                                                       OriginType type) {
  return Base::get(type.getContext(), base, userName, type);
}

LogicalResult
InteriorOriginAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           TypedAttr base, TypedAttr userName,
                           OriginType type) {
  if (!isa<StringType>(userName.getType()))
    return emitError() << "userName must be of `!kgen.string` type";
  return success();
}

// Fields are simple constants if their base is.
bool InteriorOriginAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getBase());
}

bool InteriorOriginAttr::isLessThan(Attribute rhs) const {
  auto rhsInterior = ::cast<InteriorOriginAttr>(rhs);
  if (getBase() == rhsInterior.getBase())
    return ParameterAttr::compare(getUserName(), rhsInterior.getUserName());
  return ParameterAttr::compare(getBase(), rhsInterior.getBase());
}

//===----------------------------------------------------------------------===//
// OriginSubtreeAttr
//===----------------------------------------------------------------------===//

TypedAttr OriginSubtreeAttr::get(TypedAttr origin) {
  // If we have the global mutable origin, treat it conservatively by
  // returning the global mutable origin.  It isn't wise to try to derive
  // information from something where origins have been casted away.
  if (sugarIsa<AnyOriginAttr>(origin))
    return origin;

  // We push any mutability casts outside of ourselves.
  //     mutcast(x) as input => mutcast(subtree(x))
  if (auto mutCast = sugarDynCast<OriginMutCastAttr>(origin)) {
    auto inner = OriginSubtreeAttr::get(mutCast.getOperand());
    return OriginMutCastAttr::get(inner, mutCast.getType());
  }

  // Push this inside an origin.union as well, so we get the union on the
  // outside.
  if (auto unionAttr = sugarDynCast<OriginUnionAttr>(origin)) {
    SmallVector<TypedAttr> elts;
    for (auto elt : unionAttr.getOperands())
      elts.push_back(OriginSubtreeAttr::get(elt));
    return OriginUnionAttr::get(elts, unionAttr.getType());
  }

  // subtree(subtree(x)) => subtree(x)
  if (auto subtree = sugarDynCast<OriginSubtreeAttr>(origin))
    return subtree;

  auto originType = sugarCast<OriginType>(origin.getType());
  return OriginSubtreeAttr::Base::get(origin.getContext(), origin, originType);
}

OriginSubtreeAttr OriginSubtreeAttr::getFromBytecode(TypedAttr origin,
                                                     OriginType type) {
  return Base::get(type.getContext(), origin, type);
}

bool OriginSubtreeAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getBase());
}

bool OriginSubtreeAttr::isLessThan(Attribute rhs) const {
  auto rhsSubtree = ::cast<OriginSubtreeAttr>(rhs);
  return ParameterAttr::compare(getBase(), rhsSubtree.getBase());
}

//===----------------------------------------------------------------------===//
// ImplicitOriginRefAttr
//===----------------------------------------------------------------------===//

// Origins are treated as simple constants, allowing folding of function calls.
bool ImplicitOriginRefAttr::isConstant() const { return true; }

// Make sure that these are implicitly sorted w.r.t. each other when KGEN
// canonicalizes them.
bool ImplicitOriginRefAttr::isLessThan(Attribute rhs) const {
  auto ref = ::cast<ImplicitOriginRefAttr>(rhs);
  return std::make_tuple(getDepth(), getIndex()) <
         std::make_tuple(ref.getDepth(), ref.getIndex());
}

IndexRefAttrInterface
ImplicitOriginRefAttr::replace(size_t depth, size_t index,
                               ArrayRef<Attribute> attrs,
                               ArrayRef<Type> types) const {
  assert(attrs.empty() && types.size() == 1);
  return ImplicitOriginRefAttr::get(depth, index, types.front());
}

//===----------------------------------------------------------------------===//
// OriginSetAttr
//===----------------------------------------------------------------------===//

OriginSetAttr OriginSetAttr::getFromBytecode(ArrayRef<TypedAttr> operands,
                                             OriginSetType type) {
  return Base::get(type.getContext(), operands, type);
}

TypedAttr OriginSetAttr::get(MLIRContext *ctx, ArrayRef<TypedAttr> operands,
                             OriginSetType type) {
  return get(operands, type);
}

TypedAttr OriginSetAttr::get(MLIRContext *ctx, ArrayRef<TypedAttr> operands) {
  return get(operands, OriginSetType::get(ctx));
}

TypedAttr OriginSetAttr::get(ArrayRef<TypedAttr> operands, OriginSetType type) {
  SmallVector<TypedAttr> newOperands;
  for (TypedAttr operand : operands) {
    // If we have the global mutable origin, treat it conservatively by
    // returning the global mutable origin.  It isn't wise to try to derive
    // information from something where origins have been casted away.
    if (sugarIsa<AnyOriginAttr>(operand)) {
      newOperands.push_back(operand);
      continue;
    }
    // Break up unions into their constituents without mutcasts.
    if (auto unionAttr = sugarDynCast<OriginUnionAttr>(operand)) {
      for (TypedAttr origin : unionAttr.getOperands())
        newOperands.push_back(OriginMutCastAttr::strip(origin));
      continue;
    }
    newOperands.push_back(OriginMutCastAttr::strip(operand));
  }

  // Now sort the operands by mutability and value.
  llvm::stable_sort(newOperands, [&](TypedAttr lhs, TypedAttr rhs) {
    TypedAttr lhsMut = ::cast<OriginType>(lhs.getType()).isMutable();
    TypedAttr rhsMut = ::cast<OriginType>(rhs.getType()).isMutable();
    if (ParameterAttr::compare(lhsMut, rhsMut))
      return true;
    if (ParameterAttr::compare(rhsMut, lhsMut))
      return false;
    return ParameterAttr::compare(lhs, rhs);
  });
  removeDuplicates(newOperands);

  // If we find a set union and there is only one operand, collapse the union.
  if (newOperands.size() == 1)
    if (auto setUnion = sugarDynCast<OriginSetUnionAttr>(newOperands.front()))
      return setUnion.getValue();

  return Base::get(type.getContext(), newOperands, type);
}

//===----------------------------------------------------------------------===//
// OriginSetUnionAttr
//===----------------------------------------------------------------------===//

OriginSetUnionAttr OriginSetUnionAttr::getFromBytecode(TypedAttr value,
                                                       OriginType type) {
  return Base::get(type.getContext(), value, type);
}

TypedAttr OriginSetUnionAttr::get(TypedAttr value, OriginType type) {
  // Fold `set.union(set) -> union`.
  if (auto set = sugarDynCast<OriginSetAttr>(value)) {
    return OriginMutCastAttr::get(
        OriginUnionAttr::get(type.getContext(), set.getOperands()), type);
  }
  return Base::get(type.getContext(), value, type);
}

//===----------------------------------------------------------------------===//
// LITStructAttr
//===----------------------------------------------------------------------===//

TypedAttr LITStructAttr::get(MLIRContext *ctx,
                             ArrayRef<std::tuple<StringAttr, TypedAttr>> values,
                             StructType type) {
  // If we are forming a struct from an single extract from the same type,
  // canonicalize it away so we get type equality for important types like
  // Origin and AddressSpace etc.
  if (values.size() == 1) {
    auto [fieldName, value] = values[0];
    if (auto extract = sugarDynCast<LIT::StructExtractAttr>(value))
      if (extract.getField() == fieldName &&
          extract.getStructValue().getType() == type)
        return extract.getStructValue();
  }

  return LITStructAttr::Base::get(ctx, values, type);
}

static ParseResult
parseStructElements(AsmParser &p,
                    SmallVector<std::tuple<StringAttr, TypedAttr>> &values) {
  std::string name;
  Type type;
  TypedAttr value;
  auto parseElt = [&]() -> ParseResult {
    if (p.parseKeywordOrString(&name) || parseColonTypeOrIndex(p, type) ||
        p.parseEqual() || parseParamValue(p, value, type))
      return failure();
    values.emplace_back(StringAttr::get(p.getContext(), name), value);
    return success();
  };
  return p.parseCommaSeparatedList(AsmParser::Delimiter::Braces, parseElt);
}

static void
printStructElements(AsmPrinter &p,
                    ArrayRef<std::tuple<StringAttr, TypedAttr>> values) {
  p << '{';
  llvm::interleaveComma(values, p, [&](const auto &value) {
    p.printKeywordOrString(std::get<0>(value));
    printColonTypeOrIndex(p, std::get<1>(value).getType());
    p << " = ";
    printParamValue(p, std::get<1>(value));
  });
  p << '}';
}

LogicalResult
LITStructAttr::verifySymbolUses(SymTabEvaluationContext &evaluationContext,
                                Location loc) const {
  Operation *module = evaluationContext.module;
  mlir::LockedSymbolTableCollection &symtab = evaluationContext.symtab;

  SymbolRefAttr symbolRef = getType().getSymbol();
  auto structDecl = symtab.lookupSymbolIn<StructDeclOp>(module, symbolRef);
  if (!structDecl) {
    return emitError(loc) << "struct attribute type " << symbolRef
                          << " does not refer to a struct declaration";
  }

  ParameterEvaluator evaluator(structDecl.getInputParams(),
                               getType().getParamValues());
  evaluator.setEvaluationContext(&evaluationContext);

  auto fields = structDecl.getFieldDecls();
  unsigned numFields = std::distance(fields.begin(), fields.end());
  if (numFields != getValues().size()) {
    return (emitError(loc) << "struct declaration expected " << numFields
                           << " fields but struct attribute has "
                           << getValues().size())
               .attachNote(structDecl.getLoc())
           << "see struct declaration here";
  }

  for (auto [fieldDecl, value, i] :
       llvm::zip(fields, getValues(), llvm::seq<unsigned>(0, numFields))) {
    StringAttr nameInDecl = fieldDecl.getNameAttr();
    if (nameInDecl != std::get<0>(value)) {
      return (emitError(loc)
              << "struct attribute field name " << std::get<0>(value)
              << " at position #" << i << " does not match the name "
              << nameInDecl << " in the struct declaration")
                 .attachNote(structDecl.getLoc())
             << "see struct declaration here";
    }

    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (!isEqualCanon(reboundType, std::get<1>(value).getType())) {
      return (emitError(loc)
              << "struct attribute field #" << i << " has type "
              << std::get<1>(value).getType()
              << " but corresponding struct field " << fieldDecl.getNameAttr()
              << " expected " << reboundType)
                 .attachNote(structDecl.getLoc())
             << "see struct declaration here";
    }
  }

  return success();
}

bool LITStructAttr::isConstant() const {
  return llvm::all_of(getValues(), [&](const auto &value) {
    return ParameterAttr::isSimpleConstant(std::get<1>(value));
  });
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

bool LIT::StructExtractAttr::isConstant() const { return false; }

bool LIT::StructExtractAttr::isLessThan(Attribute rhs) const {
  // Compare the underlying references if the fields are the same.
  auto rhsExtract = ::cast<LIT::StructExtractAttr>(rhs);
  if (getField() == rhsExtract.getField())
    return ParameterAttr::compare(getStructValue(),
                                  rhsExtract.getStructValue());
  return getField().getValue() < rhsExtract.getField().getValue();
}

LIT::StructExtractAttr
LIT::StructExtractAttr::getFromBytecode(TypedAttr structValue, StringAttr field,
                                        Type type) {
  return Base::get(type.getContext(), structValue, field, type);
}

TypedAttr LIT::StructExtractAttr::get(TypedAttr structValue,
                                      StructFieldOp fieldOp) {
  auto structType = sugarCast<StructType>(structValue.getType());
  ParameterEvaluator evaluator(fieldOp.getParentOp().getInputParams(),
                               structType.getParamValues());
  auto resultType = evaluator.getReboundType(fieldOp.getType());
  return get(structValue, fieldOp.getNameAttr(), resultType);
}

TypedAttr LIT::StructExtractAttr::get(TypedAttr structValue, StringAttr field,
                                      Type resultType) {
  return get(structValue.getContext(), structValue, field, resultType);
}

TypedAttr LIT::StructExtractAttr::get(MLIRContext *context,
                                      TypedAttr structValue, StringAttr field,
                                      Type resultType) {

  if (auto value = sugarDynCastIfPresent<LITStructAttr>(structValue)) {
    auto it = llvm::find_if(value.getValues(), [&](const auto &p) {
      return std::get<0>(p) == field;
    });
    // Return it if we found it.
    if (it != value.getValues().end()) {
      // Make sure type sugar on the field value doesn't break invariants.
      TypedAttr result =
          ParamOperatorAttr::getRebind(std::get<1>(*it), resultType);
      // Maintain sugar by applying struct-extract on the sugared form too.
      // Create it as Preserved sugar to indicate this is an internal
      // transformation not meant to be printed as an "aka" in diagnostics.
      if (auto sugarWrapper = dyn_cast<SugarAttr>(structValue)) {
        result = SugarAttr::getPreserved(
            Base::get(context, sugarWrapper.getSugared(), field, resultType),
            result);
      }
      return result;
    }
  }

  // Fold valueless struct operands for convenience.
  if (::isa<UnknownAttr>(structValue))
    return UnknownAttr::get(resultType);

  return Base::get(context, structValue, field, resultType);
}

//===----------------------------------------------------------------------===//
// RefPackAttr
//===----------------------------------------------------------------------===//

static ParseResult parsePackElements(AsmParser &p,
                                     SmallVector<TypedAttr> &values,
                                     RefPackType packType) {
  auto variadic = packType.getVariadicIfResolved();
  if (!variadic)
    return p.emitError(p.getCurrentLocation())
           << "lit.ref.pack attribute expected a variadic constant, but got "
           << packType.getVariadic();

  // Parse one element for each type in the list.
  return failableInterleave(
      variadic.getValues(),
      [&](TypedAttr eltType) {
        return parseParamValue(
            p, values.emplace_back(),
            packType.getElementRefTypeFor(ParamType::get(eltType)));
      },
      [&] { return p.parseComma(); });
}

static void printPackElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                              RefPackType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

OptionalParseResult RefPackType::parseValue(AsmParser &p,
                                            TypedAttr &value) const {
  if (failed(p.parseOptionalLess()))
    return std::nullopt;
  SmallVector<TypedAttr> values;
  if (failed(parsePackElements(p, values, *this)))
    return failure();

  value = RefPackAttr::get(values, *this);
  return p.parseGreater();
}

LogicalResult RefPackType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto packAttr = ::dyn_cast<RefPackAttr>(value);
  if (!packAttr)
    return failure();

  p << "<";
  printPackElements(p, packAttr.getValues(), *this);
  p << ">";
  return success();
}

LogicalResult
ImportPathAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       unsigned relativeLevel,
                       ArrayRef<StringAttr> components) {
  if (relativeLevel == 0 && components.empty())
    return emitError()
           << "import path must be relative or have at least one component";
  return success();
}

LogicalResult RefPackAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<TypedAttr> values,
                                  RefPackType packType) {
  auto variadic = packType.getVariadicIfResolved();
  if (!variadic)
    return emitError()
           << "pack attribute expected a variadic constant, but got "
           << packType.getVariadic();

  ArrayRef<TypedAttr> expected = variadic.getValues();
  if (values.size() != expected.size())
    return emitError() << "pack attribute type requires " << expected.size()
                       << " elements, but got " << values.size();

  // Check that the element constants have the right types.
  for (auto [i, value, type] : llvm::enumerate(values, expected)) {
    auto eltType = packType.getElementRefTypeFor(ParamType::get(type));
    if (value.getType() != eltType)
      return emitError() << "pack attribute element #" << i << " has type "
                         << value.getType() << " but expected " << type;
  }
  return success();
}

bool RefPackAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Mojo/LITDialect/LITEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Mojo/LITDialect/LITAttrs.cpp.inc"
