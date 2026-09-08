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

#include "Mojo/KGENDialect/FoldUtils.h"
#include "Mojo/Interpreter/InterpreterState.h"
#include "Mojo/Interpreter/ParametricInterpreterState.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"

namespace M::KGEN {

FailureOr<TypedAttr> foldAttrWithTarget(ParameterEvaluationContext &context,
                                        ArrayRef<Attribute> operands,
                                        TargetAwareFoldFn fold) {
  auto target = context.getTargetInfo();
  if (!target)
    return failure();
  if (auto result = fold(FoldValues(operands), target)) {
    assert(result.getAttr() && "attribute fold should produce an attribute");
    return result.getAttr();
  }
  return failure();
}

ErrorTreeOrSuccess interpretOpWithFold(Location loc, StringRef opName,
                                       ArrayRef<Attribute> operands,
                                       InterpreterState &state,
                                       TargetAwareFoldFn fold) {
  if (auto result = fold(FoldValues(operands), state.getTarget())) {
    if (auto attr = result.getAttr()) {
      return state.mapResults(attr);
    }
  }
  return ErrorTree(loc, "failed to interpret " + opName);
}

ErrorTreeOrSuccess interpretOpWithFold(Location loc, StringRef opName,
                                       ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state,
                                       TargetAwareFoldFn fold) {
  if (auto result = fold(FoldValues(operands), state.getTarget())) {
    if (auto attr = result.getAttr()) {
      return state.mapResults(attr);
    }
  }
  return ErrorTree(loc, "failed to interpret " + opName);
}

template <typename T>
static bool compareConstants(CmpPredicate pred, T lhs, T rhs) {
  switch (pred) {
  case CmpPredicate::EQ:
    return lhs == rhs;
  case CmpPredicate::NE:
    return lhs != rhs;
  case CmpPredicate::LT:
    return lhs < rhs;
  case CmpPredicate::GT:
    return lhs > rhs;
  case CmpPredicate::LE:
    return lhs <= rhs;
  case CmpPredicate::GE:
    return lhs >= rhs;
  }
  llvm_unreachable("invalid CmpPredicate");
}

FoldValue foldSIMDCmp(CmpPredicate cc, FoldValues operands,
                      TargetInfoAttr target) {
  std::optional<int64_t> indexBitWidth;
  if (target)
    indexBitWidth = target.resolveIndexBitWidth();
  assert(operands.size() == 2 && "expected binary compare operands");

  std::optional<int64_t> size =
      cast<SIMDType>(operands[0].getType()).getResolvedSize();

  std::optional<KGENDType> inDType =
      cast<SIMDType>(operands[0].getType()).getResolvedDType();
  if (!inDType)
    inDType = cast<SIMDType>(operands[1].getType()).getResolvedDType();

  // Fold cmp(x, x) for int-like types (NaN prevents this for floats).
  if (operands[0] == operands[1] && size && inDType && inDType->isIntLike()) {
    bool isTrue = llvm::is_contained(
        {CmpPredicate::EQ, CmpPredicate::LE, CmpPredicate::GE}, cc);
    SmallVector<DTypeValue> vals(*size, {isTrue, KGENDType::kBool});
    MLIRContext *ctx = operands[0].getType().getContext();
    return FoldValue(
        SIMDAttr::get(vals, SIMDType::get(ctx, *size, KGENDType::kBool)));
  }

  // Constant fold when both operands are SIMDAttr constants.
  if (auto fold = foldSIMDOpResult<kOtherResult>(
          operands.getAttrs(), KGENDType::kBool, indexBitWidth,
          [&](APSInt l, APSInt r) { return compareConstants(cc, l, r); },
          [&](APFloat l, APFloat r) { return compareConstants(cc, l, r); },
          [&](bool l, bool r) { return compareConstants(cc, l, r); }))
    return FoldValue(fold);

  auto lhsAttr = operands.getAttr<SIMDAttr>(0);
  auto rhsAttr = operands.getAttr<SIMDAttr>(1);

  // Fold `eq(true, x) -> x` and `ne(false, x) -> x`.
  if (inDType && *inDType == KGENDType::kBool &&
      llvm::is_contained({CmpPredicate::EQ, CmpPredicate::NE}, cc)) {
    SIMDAttr constAttr = lhsAttr ? lhsAttr : rhsAttr;
    FoldValue otherValue = lhsAttr ? operands[1] : operands[0];
    if (constAttr && otherValue && llvm::all_equal(constAttr.getValues()) &&
        (cc == CmpPredicate::EQ) == constAttr.getValues().front().getBoolVal())
      return otherValue;
  }

  // Fold unsigned comparisons with zero:
  //   gt(0, x) -> false, le(0, x) -> true
  //   ge(x, 0) -> true,  lt(x, 0) -> false
  if (inDType && size && inDType->isUInt()) {
    auto tryFoldWithZero = [&](SIMDAttr zeroCandidate, SIMDAttr otherCandidate,
                               CmpPredicate foldTrue,
                               CmpPredicate foldFalse) -> TypedAttr {
      if (!llvm::is_contained({foldTrue, foldFalse}, cc))
        return {};
      if (zeroCandidate && otherCandidate)
        return {};
      if (!zeroCandidate)
        return {};
      if (llvm::all_equal(zeroCandidate.getValues()) &&
          zeroCandidate.getValues()[0].getData().isZero()) {
        SmallVector<DTypeValue> values(
            *size, DTypeValue(cc == foldTrue, KGENDType::kBool));
        MLIRContext *ctx = operands[0].getType().getContext();
        return SIMDAttr::get(values,
                             SIMDType::get(ctx, *size, KGENDType::kBool));
      }
      return {};
    };
    if (auto res = tryFoldWithZero(lhsAttr, rhsAttr, CmpPredicate::LE,
                                   CmpPredicate::GT))
      return res;
    if (auto res = tryFoldWithZero(rhsAttr, lhsAttr, CmpPredicate::GE,
                                   CmpPredicate::LT))
      return res;
  }

  return {};
}

/// Whether this dtype's value depends on the target's index bit width.
static bool isTargetDependentDType(KGENDType dtype) {
  return dtype.isIndex() || dtype.isUIndex() || dtype.isAddress();
}

static bool hasTargetDependentDType(SIMDAttr attr) {
  std::optional<KGENDType> dtype = attr.getType().getResolvedDType();
  return dtype && isTargetDependentDType(*dtype);
}

/// What a value tree holds that stops its representation from being canonical
/// for the value it denotes.
struct NonCanonicalLeaves {
  /// A value nobody knows. Nothing decides a comparison against it.
  bool unknown = false;
  /// Index-like data, so the target's index bit width decides the comparison.
  bool targetDependent = false;
};

/// Find non-canonical leaves anywhere in an attribute tree.
static NonCanonicalLeaves findNonCanonicalLeaves(Attribute attr) {
  NonCanonicalLeaves found;
  attr.walk([&](Attribute sub) {
    if (isa<UnknownAttr>(sub)) {
      // An unknown decides the whole verdict, so nothing further matters.
      found.unknown = true;
      return mlir::WalkResult::interrupt();
    }
    if (auto simd = dyn_cast<SIMDAttr>(sub);
        simd && hasTargetDependentDType(simd)) {
      found.targetDependent = true;
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

bool containsUnknownValue(Attribute attr) {
  return findNonCanonicalLeaves(attr).unknown;
}

/// Re-express `attr`'s index-like leaves as the values they denote at
/// `indexBitWidth`, so that two trees denoting the same value at that width
/// become the same uniqued attribute.
static Attribute normalizeIndexData(Attribute attr, int64_t indexBitWidth) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](SIMDAttr simd) -> std::pair<Attribute, mlir::WalkResult> {
        if (!hasTargetDependentDType(simd))
          return {simd, mlir::WalkResult::advance()};

        KGENDType dtype = *simd.getType().getResolvedDType();
        // Index is signed; the other index-like dtypes are unsigned.
        bool isUnsigned = !dtype.isIndex();
        assert(!isUnsigned || dtype.isUIndex() || dtype.isAddress());
        SmallVector<DTypeValue> values;
        for (const DTypeValue &value : simd.getValues()) {
          // Store at exactly `indexBitWidth`, so both sides of a comparison end
          // up stored the same way.
          const APInt &data = value.getData();
          values.emplace_back(isUnsigned ? data.zextOrTrunc(indexBitWidth)
                                         : data.sextOrTrunc(indexBitWidth),
                              dtype);
        }
        // Nothing below a SIMD constant needs normalizing.
        return {SIMDAttr::get(values, simd.getType()),
                mlir::WalkResult::skip()};
      });
  return replacer.replace(attr);
}

void PreparedConstant::scanLeaves() {
  if (scanned)
    return;
  NonCanonicalLeaves leaves = findNonCanonicalLeaves(attr);
  unknown = leaves.unknown;
  targetDependent = leaves.targetDependent;
  scanned = true;
}

bool PreparedConstant::hasUnknown() {
  scanLeaves();
  return unknown;
}

bool PreparedConstant::isTargetDependent() {
  scanLeaves();
  assert(!unknown && "the leaf scan stops at an unknown, so what it found "
                     "about index-like data is only part of the answer");
  return targetDependent;
}

Attribute PreparedConstant::getKeyImpl(Attribute &cache,
                                       int64_t untargetedBitWidth) {
  if (!cache) {
    // With nothing index-like in the tree, the representation is already
    // canonical for the value at every width.
    if (!isTargetDependent()) {
      cache = attr;
    } else {
      int64_t bitWidth =
          target ? target.resolveIndexBitWidth() : untargetedBitWidth;
      cache = normalizeIndexData(attr, bitWidth);
    }
  }
  return cache;
}

Attribute PreparedConstant::getKey() { return getKeyImpl(key, 64); }

Attribute PreparedConstant::get32BitKey() {
  assert(!target && "a target pins one width, so this is `getKey()` again");
  return getKeyImpl(key32Bit, 32);
}

std::optional<bool> areSimpleConstantsEqual(PreparedConstant &lhs,
                                            PreparedConstant &rhs) {
  assert(lhs.getTarget() == rhs.getTarget() &&
         "keys prepared at different index bit widths do not compare");

  // An unknown stands in for a value nobody knows, so no comparison of the
  // representations says anything about the values, not even the same
  // representation twice.
  if (lhs.hasUnknown() || rhs.hasUnknown())
    return std::nullopt;

  // Uniquing already answers this, whatever the target.
  if (lhs.getAttr() == rhs.getAttr())
    return true;

  // With no index-like data anywhere in either tree, the representation is
  // canonical for the value and inequality is the answer.
  if (!lhs.isTargetDependent() && !rhs.isTargetDependent())
    return false;

  // With a target this key is the one width in force, so equality is identity.
  // Without one it is the 64-bit candidate, and equal values there truncate to
  // equal values at 32, so agreement settles both candidates at once.
  if (lhs.getKey() == rhs.getKey())
    return true;

  // A mismatch under a target is final.
  if (lhs.getTarget())
    return false;
  // Without a target, it takes a mismatch at both candidate widths to mean "a
  // different value".
  if (lhs.get32BitKey() == rhs.get32BitKey())
    return std::nullopt;
  return false;
}

} // namespace M::KGEN
