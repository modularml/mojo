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
// Shared helpers for expressing folds over SSA values and typed attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_FOLDUTILS_H
#define KGEN_KGENDIALECT_FOLDUTILS_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>

namespace M::KGEN {

class ParameterEvaluationContext;

/// A foldable value that can carry an SSA value, a typed attribute, or both.
/// When both are present, they represent the same logical operand/result.
class FoldValue {
public:
  /// A default-constructed FoldValue is falsey. It can only exist as a return
  /// value from a fold operation, and means that the fold operation did not
  /// produce a value.
  FoldValue() = default;

  /// Create a FoldValue from an SSA value and an optional typed attribute. For
  /// use during op-based folding.
  FoldValue(Value value, TypedAttr attr = {}) : value(value), attr(attr) {}

  /// Create a FoldValue from a typed attribute. For use during attribute-based
  /// folding.
  FoldValue(TypedAttr attr) : attr(attr) {}

  explicit operator bool() const { return value || attr; }

  Value getValue() const { return value; }

  template <typename AttrT = TypedAttr>
  AttrT getAttr() const {
    return dyn_cast_or_null<AttrT>(attr);
  }

  Type getType() const {
    assert(*this && "expected fold value");
    if (value)
      return value.getType();
    return attr.getType();
  }

  OpFoldResult asOpFoldResult() const {
    assert(*this && "expected fold value");
    if (attr)
      return attr;
    return value;
  }

  bool operator==(const FoldValue &other) const {
    if (value && other.value)
      return value == other.value;
    if (attr && other.attr)
      return attr == other.attr;
    return false;
  }

private:
  Value value;
  TypedAttr attr;
};

/// A lightweight view over fold operands carried as attributes plus optional
/// parallel SSA values.
class FoldValues {
public:
  FoldValues(ArrayRef<Attribute> attrs, ValueRange values = {})
      : attrs(attrs), values(values) {
    assert((values.empty() || values.size() == attrs.size()) &&
           "expected one value per attribute");
  }

  size_t size() const { return attrs.size(); }

  ArrayRef<Attribute> getAttrs() const { return attrs; }

  template <typename AttrT = TypedAttr>
  AttrT getAttr(size_t index) const {
    assert(index < size() && "operand index out of bounds");
    return dyn_cast_or_null<AttrT>(attrs[index]);
  }

  Value getValue(size_t index) const {
    assert(index < size() && "operand index out of bounds");
    if (values.empty())
      return {};
    return values[index];
  }

  FoldValue operator[](size_t index) const {
    return FoldValue(getValue(index), getAttr(index));
  }

private:
  ArrayRef<Attribute> attrs;
  ValueRange values;
};

//===----------------------------------------------------------------------===//
// Fold helpers
//
// These reduce the boilerplate for connecting ops and attributes to a shared
// fold function.  Each fold function has the canonical signature:
//
//   FoldValue foldFoo(FoldValues operands, TargetInfoAttr target);
//
// The helpers below adapt that signature for the four standard hooks:
//   - Attr::get            (fold-on-construction, no target)
//   - evaluateWithContext  (contextual eval, target from context)
//   - Op::fold             (op folding, target from module)
//   - Op::interpret        (interpreter, target from state)
//===----------------------------------------------------------------------===//

/// The canonical fold function signature accepted by all fold helpers.
using TargetAwareFoldFn = function_ref<FoldValue(FoldValues, TargetInfoAttr)>;

/// Try to fold an attribute during construction (no target info available).
/// A null TargetInfoAttr is passed, relying on the fold function to treat it
/// as "unknown target" and only fold when safe.
inline TypedAttr tryFoldAttr(ArrayRef<Attribute> operands,
                             TargetAwareFoldFn fold) {
  if (auto result = fold(FoldValues(operands), {})) {
    assert(result.getAttr() && "attribute fold should produce an attribute");
    return result.getAttr();
  }
  return {};
}

/// Evaluate an attribute with context using a target-aware fold function.
/// Returns the folded attribute if target info is available and the fold
/// succeeds, or failure() otherwise.
FailureOr<TypedAttr> foldAttrWithTarget(ParameterEvaluationContext &context,
                                        ArrayRef<Attribute> operands,
                                        TargetAwareFoldFn fold);

/// Whether `attr` contains an unknown value at any depth, including inside
/// nested types. An unknown carries no value, so no comparison of
/// representations containing one decides anything, not even comparing a
/// representation with itself.
bool containsUnknownValue(Attribute attr);

/// One operand of a constant comparison, memoizing what the comparison learns
/// about it so that comparing it against many others costs one scan and one
/// normalization rather than one per pair.
///
/// Nothing is computed until a rule asks: the rules short-circuit, so most
/// comparisons settle without ever walking a tree.
class PreparedConstant {
public:
  /// Prepare `attr` for comparison against others prepared with the same
  /// `target`, which may be null.
  PreparedConstant(TypedAttr attr, TargetInfoAttr target)
      : attr(attr), target(target) {}

  TypedAttr getAttr() const { return attr; }
  TargetInfoAttr getTarget() const { return target; }

  /// Whether an unknown value sits anywhere in the tree.
  bool hasUnknown();

  /// Whether index-like data sits anywhere in the tree, so that the target's
  /// index bit width decides what the representation denotes. Only meaningful
  /// once `hasUnknown()` is known false: the scan stops at the first unknown,
  /// since nothing past it can change the verdict.
  bool isTargetDependent();

  /// The tree with its index-like leaves re-expressed at the target's index bit
  /// width, or at 64 bits where there is no target, so that two trees denoting
  /// the same value at that width are the same uniqued attribute.
  Attribute getKey();

  /// The same at 32 bits, the other width an untargeted comparison has to
  /// consider.
  Attribute get32BitKey();

private:
  void scanLeaves();
  Attribute getKeyImpl(Attribute &cache, int64_t untargetedBitWidth);

  TypedAttr attr;
  TargetInfoAttr target;

  /// Whether `unknown` and `targetDependent` below have been computed.
  bool scanned = false;
  bool unknown = false;
  bool targetDependent = false;

  /// Null until asked for.
  Attribute key;
  Attribute key32Bit;
};

/// Decide whether two fully-evaluated parameter values denote the same value.
///
/// The caller must have established that both operands are simple constants
/// (`ParameterAttr::isSimpleConstant`), which means equality is pointer
/// equality, except for these two cases:
///
///  - target-dependent data, like index-typed values whose number depends on
///    the index bit width.
///  - unknown values, which carry no value for a representation to be canonical
///    for. No target settles these.
///
/// Returns nullopt for both, i.e. whenever the comparison cannot be made on the
/// information given. Both operands must have been prepared with the same
/// target, or their keys would stand for values at different widths.
std::optional<bool> areSimpleConstantsEqual(PreparedConstant &lhs,
                                            PreparedConstant &rhs);

/// Fold an op using a target-aware fold function.
inline OpFoldResult foldOpWithTarget(FoldValues operands, TargetInfoAttr target,
                                     TargetAwareFoldFn fold) {
  if (auto result = fold(operands, target))
    return result.asOpFoldResult();
  return {};
}

/// Interpret an op using a target-aware fold function.
ErrorTreeOrSuccess interpretOpWithFold(Location loc, StringRef opName,
                                       ArrayRef<Attribute> operands,
                                       InterpreterState &state,
                                       TargetAwareFoldFn fold);
ErrorTreeOrSuccess interpretOpWithFold(Location loc, StringRef opName,
                                       ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state,
                                       TargetAwareFoldFn fold);

//===----------------------------------------------------------------------===//
// SIMD Folder Helpers
//===----------------------------------------------------------------------===//

/// This enum indicates how index folding should be done.
enum IndexFold : uint8_t {
  kNoIndex,     // no index folding allowed
  kIndexResult, // index operation creates an index
  kOtherResult, // index operation does not create an index
};

namespace Detail {
/// Detector for whether `T` possesses a `has_value` method.
template <typename T>
using IsOptionalType = decltype(std::declval<T>().has_value());

/// `std::optional<T>` -> `T`
template <typename T>
struct remove_optional {
  using type = T;
};
template <typename T>
struct remove_optional<std::optional<T>> {
  using type = T;
};

/// Perform folding of an n-ary SIMD vector operation of a given dtype by
/// applying the operation `op` to each vector element. `getValue` transforms a
/// `DTypeValue` to the value used to represent the dtype: `APSInt` for
/// integers, `APFloat` for floats, and `bool` for bools.
template <size_t... I, typename OpFn, typename GetValueFn>
static SIMDAttr foldSIMDOpImpl(std::index_sequence<I...>,
                               ArrayRef<Attribute> operands, OpFn op,
                               KGENDType dtype, GetValueFn getValue) {
  SmallVector<DTypeValue> results;
  auto firstArg = cast<SIMDAttr>(operands.front());
  for (unsigned i = 0, e = firstArg.getValues().size(); i != e; ++i) {
    auto result =
        std::apply(op, std::make_tuple(getValue(
                           cast<SIMDAttr>(operands[I]).getValues()[i])...));
    // Allow folders to return failure. This indicates undefined behaviour,
    // which we do not fold.
    if constexpr (llvm::is_detected<IsOptionalType, decltype(result)>::value) {
      if (!result)
        return {};
      results.emplace_back(*result, dtype);
    } else {
      results.emplace_back(result, dtype);
    }
  }
  auto type = cast<SIMDType>(cast<TypedAttr>(operands.front()).getType());
  return SIMDAttr::get(results, SIMDType::get(type.getContext(),
                                              *type.getResolvedSize(), dtype));
}

/// Perform the folding of a SIMD vector reduction of a given dtype by
/// accumulatively applying the binary operation `op` to each vector
/// element, in order, starting with the first. `getValue` transforms a
/// `DTypeValue` to the value used to represent the dtype: `APSInt` for
/// integers, `APFloat` for floats, and `bool` for bools.
template <typename OpFn, typename GetValueFn>
static SIMDAttr foldSIMDReduceOpImpl(Attribute operand, OpFn op,
                                     KGENDType dtype, GetValueFn getValue) {
  auto firstArg = cast<SIMDAttr>(operand);
  auto values = firstArg.getValues();
  auto accumResult = getValue(values[0]);
  for (unsigned i = 1, e = values.size(); i < e; ++i) {
    auto res = std::apply(op, std::make_pair(accumResult, getValue(values[i])));
    // Allow folders to return failure. This indicates undefined behaviour,
    // which we do not fold.
    if constexpr (llvm::is_detected<IsOptionalType, decltype(res)>::value) {
      if (!res)
        return {};
      accumResult = *res;
    } else {
      accumResult = res;
    }
  }
  return SIMDAttr::get(DTypeValue(accumResult, dtype),
                       SIMDType::get(operand.getContext(), 1, dtype));
}

/// Return true if the function type `OpFn` is a function whose first argument
/// type is `TestType`, which can be an integer, float, index, or bool type.
template <typename OpFn, typename TestType>
static constexpr bool testOpFnType() {
  return std::is_same_v<
      TestType,
      std::decay_t<typename llvm::function_traits<OpFn>::template arg_t<0>>>;
}

/// Base case for getting an op function of a given type. This one returns none.
template <typename TestType>
static constexpr auto getOpFnOfType() {
  return std::nullopt;
}

/// This function selects a function which can be applied to `TestType` from
/// `OpFns`. If the head op function is of the given type, return it. Otherwise,
/// check the rest of the functions.
template <typename TestType, typename OpFn, typename... OpFns>
static constexpr auto getOpFnOfType([[maybe_unused]] OpFn op, OpFns &&...fns) {
  if constexpr (testOpFnType<OpFn, TestType>())
    return op;
  else
    return getOpFnOfType<TestType>(std::forward<OpFns>(fns)...);
}

/// Try to fold the operation using one of the provided fold functions for a
/// given dtype. If a fold function for that dtype is not provided, if such a
/// dtype is encountered by the folder, it will assert; a folder must be
/// provided for each dtype for which the operation is valid.
template <typename TestType, typename GetValueFn, typename... OpFns>
static SIMDAttr foldSIMDOpDType([[maybe_unused]] GetValueFn getValue,
                                [[maybe_unused]] ArrayRef<Attribute> operands,
                                [[maybe_unused]] KGENDType dtype,
                                OpFns &&...ops) {
  auto op = getOpFnOfType<TestType>(std::forward<OpFns>(ops)...);
  if constexpr (std::is_same_v<decltype(op), std::nullopt_t>) {
    llvm_unreachable("unhandled dtype");
  } else {
    return foldSIMDOpImpl(std::make_index_sequence<
                              llvm::function_traits<decltype(op)>::num_args>(),
                          operands, op, dtype, getValue);
  }
}

/// Try to fold the operation using one of the provided fold functions for a
/// given dtype. If a fold function for that dtype is not provided, if such a
/// dtype is encountered by the folder, it will assert; a folder must be
/// provided for each dtype for which the operation is valid.
template <typename TestType, typename GetValueFn, typename... OpFns>
static SIMDAttr foldSIMDReduceOpDType([[maybe_unused]] GetValueFn getValue,
                                      [[maybe_unused]] Attribute operand,
                                      [[maybe_unused]] KGENDType dtype,
                                      OpFns &&...ops) {
  auto op = getOpFnOfType<TestType>(std::forward<OpFns>(ops)...);
  if constexpr (std::is_same_v<decltype(op), std::nullopt_t>) {
    llvm_unreachable("unhandled dtype");
  } else {
    return foldSIMDReduceOpImpl(operand, op, dtype, getValue);
  }
}

/// Try to fold an operation with index dtype using one of the provided fold
/// functions. Index folds are performed using the same function as integer
/// dtype folds. An index fold is performed by computing the result in 64-bit
/// and 32-bit arithmetic. If the results match, then the operation can fold.
/// See the MLIR `index` dialect for more details.
template <IndexFold foldType, typename... OpFns>
static SIMDAttr foldSIMDOpIndex(ArrayRef<Attribute> operands, KGENDType dtype,
                                OpFns &&...ops) {
  auto op = getOpFnOfType<APSInt>(std::forward<OpFns>(ops)...);
  if constexpr (std::is_same_v<decltype(op), std::nullopt_t>) {
    llvm_unreachable("unhandled dtype");
  } else {
    // Define the index fold function using the integer fold function. Detect a
    // bool function. Return a bool instead of an index value in that case.
    using OpResultT = typename llvm::function_traits<decltype(op)>::result_t;
    // Check if the fold function is failable. If the fold function can fail,
    // make sure to propagate the failure in both 64-bit and 32-bit arithmetic.
    constexpr bool isOptional =
        llvm::is_detected<IsOptionalType, OpResultT>::value;
    using ResultT = typename remove_optional<OpResultT>::type;
    constexpr bool isIndexResult = foldType == kIndexResult;
    auto indexOp = [&op](auto... args)
        -> std::optional<std::conditional_t<isIndexResult, int64_t, ResultT>> {
      auto unwrap = [](OpResultT value) {
        if constexpr (isOptional)
          return *value;
        else
          return value;
      };

      OpResultT result64 = op(args...);
      if constexpr (isOptional)
        if (!result64.has_value())
          return {};

      OpResultT result32 = op(args.trunc(32)...);
      if constexpr (isOptional)
        if (!result32.has_value())
          return {};
      // Compare the results. Return the index value if the fold results match.
      // If the result type isn't an index represented as an APSInt, just
      // compare the results directly.
      if constexpr (isIndexResult) {
        if (unwrap(result64).trunc(32) == unwrap(result32))
          return unwrap(result64).getSExtValue();
        return {};
      } else {
        if (unwrap(result64) == unwrap(result32))
          return unwrap(result64);
        return {};
      }
    };
    return foldSIMDOpImpl(std::make_index_sequence<
                              llvm::function_traits<decltype(op)>::num_args>(),
                          operands, indexOp, dtype, [](DTypeValue val) {
                            return APSInt(
                                APInt(64, val.getIndexVal()),
                                /*isUnsigned=*/!val.getDType().isIndex());
                          });
  }
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <IndexFold indexFoldType, typename... OpFns>
SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands, KGENDType inputDType,
                    KGENDType resultDType, std::optional<int64_t> indexBitWidth,
                    OpFns &&...ops) {
  if (inputDType.isInt())
    return Detail::foldSIMDOpDType<APSInt>(
        [](const DTypeValue &val) { return val.getIntVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  // FIXME: Should we even do floating point folds? Results don't match hardware
  // and not all float semantics are supported.
  if (inputDType.isFloat())
    return Detail::foldSIMDOpDType<APFloat>(
        [](const DTypeValue &val) { return val.getFloatVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  if (inputDType.isBool())
    return Detail::foldSIMDOpDType<bool>(
        [](const DTypeValue &val) { return val.getBoolVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  if (inputDType.isIndex() || inputDType.isUIndex() || inputDType.isAddress()) {
    // If we know the index type's bit width, treat it as if it were an integer
    // type of that same bit width. This avoids the complexities of dealing with
    // index types.
    if (indexBitWidth) {
      int64_t bitWidth = *indexBitWidth;
      bool isUnsigned = !inputDType.isIndex();
      return Detail::foldSIMDOpDType<APSInt>(
          [bitWidth, isUnsigned](const DTypeValue &val) {
            auto indexAPInt = APInt(64, val.getIndexVal());
            return APSInt(indexAPInt, isUnsigned).extOrTrunc(bitWidth);
          },
          operands, resultDType, std::forward<OpFns>(ops)...);
    }
    return Detail::foldSIMDOpIndex<indexFoldType>(operands, resultDType,
                                                  std::forward<OpFns>(ops)...);
  }
  llvm_unreachable("unhandled dtype");
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <typename... OpFns>
SIMDAttr foldBitwiseSIMDReduceOp(Attribute operand, KGENDType inputDType,
                                 KGENDType resultDType, OpFns &&...ops) {
  if (inputDType.isBool())
    return Detail::foldSIMDReduceOpDType<bool>(
        [](const DTypeValue &val) { return val.getBoolVal(); }, operand,
        resultDType, std::forward<OpFns>(ops)...);
  // For bitwise reductions we can treat index-like types as ints. The result
  // would be the same no matter the eventual index bitwidth, whether it was
  // extended/truncated before or after the fold.
  if (inputDType.isIntLike())
    return Detail::foldSIMDReduceOpDType<APSInt>(
        [](const DTypeValue &val) { return val.getIntVal(); }, operand,
        resultDType, std::forward<OpFns>(ops)...);
  llvm_unreachable("unhandled dtype");
}
} // namespace Detail

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <IndexFold indexFoldType, typename... OpFns>
SIMDAttr foldSIMDOpResult(ArrayRef<Attribute> operands, KGENDType resultDType,
                          std::optional<int64_t> indexBitWidth,
                          OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  return Detail::foldSIMDOp<indexFoldType>(
      operands, *cast<SIMDAttr>(operands.front()).getType().getResolvedDType(),
      resultDType, indexBitWidth, std::forward<OpFns>(ops)...);
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <IndexFold indexFoldType, typename... OpFns>
SIMDAttr foldSIMDOpResult(ArrayRef<Attribute> operands, KGENDType resultDType,
                          OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  return Detail::foldSIMDOp<indexFoldType>(
      operands, *cast<SIMDAttr>(operands.front()).getType().getResolvedDType(),
      resultDType, /*indexBitWidth=*/std::nullopt, std::forward<OpFns>(ops)...);
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype, assuming the result dtype is the
/// same as the operands' dtypes.
template <typename... OpFns>
SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands,
                    std::optional<int64_t> indexBitWidth, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  KGENDType dtype =
      *cast<SIMDAttr>(operands.front()).getType().getResolvedDType();
  return Detail::foldSIMDOp<kIndexResult>(operands, dtype, dtype, indexBitWidth,
                                          std::forward<OpFns>(ops)...);
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype, assuming the result dtype is the
/// same as the operands' dtypes.
template <typename... OpFns>
SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  KGENDType dtype =
      *cast<SIMDAttr>(operands.front()).getType().getResolvedDType();
  return Detail::foldSIMDOp<kIndexResult>(operands, dtype, dtype,
                                          /*indexBitWidth=*/std::nullopt,
                                          std::forward<OpFns>(ops)...);
}

/// Try to fold a SIMD vector reduction operation using one of the provided
/// functions for each possible operand dtype, assuming the result dtype is the
/// same as the operands' dtypes.
template <typename... OpFns>
SIMDAttr foldBitwiseSIMDReduceOp(Attribute operand, OpFns &&...ops) {
  if (!isa_and_nonnull<SIMDAttr>(operand))
    return {};
  KGENDType dtype = *cast<SIMDAttr>(operand).getType().getResolvedDType();
  return Detail::foldBitwiseSIMDReduceOp(operand, dtype, dtype,
                                         std::forward<OpFns>(ops)...);
}

/// Convert a POC to the full CmpPredicate.
inline CmpPredicate toCmpPredicate(POC cc) {
  switch (cc) {
  case POC::EQ:
    return CmpPredicate::EQ;
  case POC::LT:
    return CmpPredicate::LT;
  case POC::LE:
    return CmpPredicate::LE;
  default:
    llvm_unreachable("invalid POC");
  }
}

/// Fold a SIMD comparison operation. Handles constant folding, bool identity
/// folds (eq(true, x) -> x), and unsigned comparisons with zero. Returns null
/// if no fold applies.
FoldValue foldSIMDCmp(CmpPredicate cc, FoldValues operands,
                      TargetInfoAttr target = {});

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_FOLDUTILS_H
