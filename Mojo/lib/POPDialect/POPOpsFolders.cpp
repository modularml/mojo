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

#include "Mojo/Interpreter/ParametricInterpreterState.h"
#include "Mojo/Interpreter/Utils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPEnums.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/POPDialect/POPUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/IR/Diagnostics.h>
#include <unistd.h>

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

Operation *POPDialect::materializeConstant(OpBuilder &b, Attribute value,
                                           Type type, Location loc) {
  return ParamConstantOp::create(b, loc, type, cast<TypedAttr>(value));
}

//===----------------------------------------------------------------------===//
// SIMD Construction Checks

static ErrorTreeOrSuccess validateSIMDConstruction(int64_t size, DType dtype,
                                                   const Location &loc) {
  if (dtype.isInvalid())
    return ErrorTree(loc, "simd type cannot be DType.invalid");
  if (!llvm::isPowerOf2_64(size))
    return ErrorTree(loc, "simd width must be a power of 2");
  // MOCO-1388: Until LLVM's issue #122571 is fixed, LLVM's SelectionDAG has
  // a limit of 2^15 for the number of operands of the instruction.
  // NOTE: Even after the limit increases in LLVM, compile time might be 3x
  // slower than with GCC, therefore until we have a real use case for large
  // SIMD, we better to keep limit at 2^15.
  // NOTE: Might need to revisit the limit for targets that use GlobalISel
  // as it does have smaller limit now.
  if (size > (1LL << 15))
    return ErrorTree(loc, "simd size must be less than 2^15");
  return success();
}

static ErrorTreeOrSuccess validateSIMDConstruction(SIMDType simdType,
                                                   const Location &loc) {
  if (!simdType.getResolvedSize())
    return ErrorTree(loc, "simd size must be known");
  if (!simdType.getResolvedDType())
    return ErrorTree(loc, "simd DType must be known");
  return validateSIMDConstruction(*simdType.getResolvedSize(),
                                  *simdType.getResolvedDType(), loc);
}

//===----------------------------------------------------------------------===//
// Arithmetic Operation Folders
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Unary Operations

OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt val) { return -val; },
      [](APFloat val) { return llvm::neg(val); });
}

OpFoldResult FloorOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt val) { return val; },
      [](APFloat val) {
        val.roundToIntegral(APFloat::rmTowardNegative);
        return val;
      },
      [](bool val) { return val; });
}

OpFoldResult CeilOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt val) { return val; },
      [](APFloat val) {
        val.roundToIntegral(APFloat::rmTowardPositive);
        return val;
      },
      [](bool val) { return val; });
}

OpFoldResult TruncOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt val) { return val; },
      [](APFloat val) {
        val.roundToIntegral(APFloat::rmTowardZero);
        return val;
      },
      [](bool val) { return val; });
}

FoldValue AbsOp::unifiedFold(FoldValues operands, TargetInfoAttr target) {
  return foldSIMDAbs(operands, target);
}

OpFoldResult RoundOp::fold(FoldAdaptor adaptor) {
  if (auto fold =
          foldSIMDRound(adaptor.getOperand(), lookupTargetInfo(*this))) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;
  }
  return {};
}

ErrorTreeOrSuccess RoundOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  if (auto fold = foldSIMDRound(operands[0], state.getTarget())) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold))) {
      return state.mapResults(ret);
    }
  }
  return ErrorTree(getLoc(), "failed to interpret POP::RoundOp");
}

ErrorTreeOrSuccess
RoundOp::parametric_interpret(ArrayRef<Attribute> operands,
                              ParametricInterpreterState &state) {
  if (auto fold = foldSIMDRound(operands[0], state.getTarget())) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold))) {
      return state.mapResults(ret);
    }
  }
  return ErrorTree(getLoc(), "failed to interpret POP::RoundOp");
}

//===----------------------------------------------------------------------===//
// Binary Operations

// Check if the input is an integer constant and return it.
// In case of a SIMD input, check that all values are equal.
static std::optional<APSInt> getIntVal(Value val) {
  SIMDAttr constAttr;
  if (!mlir::matchPattern(val, mlir::m_Constant(&constAttr)))
    return std::nullopt;
  const APSInt &constVal = constAttr.getValues().front().getIntVal();
  if (!llvm::all_of(constAttr.getValues(), [&](const DTypeValue &val) {
        return val.getIntVal() == constVal;
      }))
    return std::nullopt;
  return constVal;
}

template <typename OpT>
static bool hasIntLikeType(OpT op) {
  std::optional<KGENDType> dtype = op->getType().getResolvedDType();
  return dtype && dtype->isIntLike();
}

static bool isIntZero(Value val) {
  std::optional<APSInt> maybeConst = getIntVal(val);
  return maybeConst && maybeConst->isZero();
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  if (SIMDAttr const &res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) { return lhs + rhs; },
          [](APFloat lhs, APFloat rhs) { return lhs + rhs; })) {
    return res;
  }
  // integer X+0 or 0+X -> X
  //
  // for floating-point types that have negative zero, this optimization is
  // not valid because -0 + 0 = 0
  // TODO: this optimization can be done for fp types
  // if we add a 'fast fp math' or 'ignore negative 0' config parameter.
  if (hasIntLikeType(this)) {
    if (isIntZero(getLhs()))
      return getRhs();
    if (isIntZero(getRhs()))
      return getLhs();
  }
  return {};
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  if (SIMDAttr const &res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) { return lhs - rhs; },
          [](APFloat lhs, APFloat rhs) { return lhs - rhs; })) {
    return res;
  }
  // X-0 -> X
  // Note that unlike the 'add' case above, this optimization
  // is valid for floating-point types as well, because -0 - 0 = -0
  // TODO: generalize to support floating-point types.
  if (hasIntLikeType(this)) {
    if (isIntZero(getRhs()))
      return getLhs();
  }
  return {};
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  if (auto res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) { return lhs * rhs; },
          [](APFloat lhs, APFloat rhs) { return lhs * rhs; }))
    return res;

  if (!hasIntLikeType(this))
    return {};

  // Pattern-match trivial cases, such as 0*x or 1*x. Support both scalar and
  // SIMD types.
  auto foldTrivialMultiplication = [&](Value lhs, Value rhs) -> OpFoldResult {
    if (auto maybeVal = getIntVal(lhs)) {
      auto constVal = maybeVal.value();
      if (constVal.isZero())
        return lhs;
      if (constVal.isOne())
        return rhs;
    }
    return {};
  };

  // Try to fold trivial multiplication expecting a constant operand in lhs.
  // For example, 0*x = 0
  if (auto res = foldTrivialMultiplication(getLhs(), getRhs()))
    return res;

  // Otherwise, swap operands and try again. This will help to fold trivial
  // multiplication such as x*0 = 0
  if (auto res = foldTrivialMultiplication(getRhs(), getLhs()))
    return res;

  return {};
}

LogicalResult DivOp::canonicalize(DivOp op, PatternRewriter &b) {
  std::optional<KGEN::KGENDType> dtype = op.getType().getResolvedDType();
  if (!dtype)
    return b.notifyMatchFailure(op, "result type isn't resolved");

  if (!dtype->isIntLike())
    return b.notifyMatchFailure(op, "result type isn't int-like");

  std::optional<size_t> size = op.getType().getResolvedSize();
  if (!size)
    return b.notifyMatchFailure(op, "result type size isn't resolved");

  // Canonicalize "x / 2^n" into "x >> n"

  if (!dtype->isUInt()) {
    // Note we could perform this optimization if we knew that the LHS values
    // were all non-negative or a negated power of two. We don't have that
    // analysis right now, though. If the LHS were all constants we could
    // obviously infer this, but then we'll just end up constant folding it
    // elsewhere anyway.
    return b.notifyMatchFailure(
        op, "lhs values are signed and not known to be safe");
  }

  SIMDAttr rhsAttr;
  if (!mlir::matchPattern(op.getRhs(), mlir::m_Constant(&rhsAttr)))
    return b.notifyMatchFailure(op, "rhs is not a constant");

  if (!llvm::all_of(rhsAttr.getValues(), [&](const DTypeValue &val) {
        APInt intVal = val.getIntVal();
        return intVal.isStrictlyPositive() && intVal.isPowerOf2();
      })) {
    return b.notifyMatchFailure(op, "rhs values are not positive power of 2");
  }

  ssize_t intWidth = dtype->getWidthInBits();
  if (dtype->isIndex() || dtype->isUIndex()) {
    TargetInfoAttr target = lookupTargetInfo(op);
    if (!target)
      return b.notifyMatchFailure(op, "target isn't resolved");
    intWidth = target.resolveIndexBitWidth();
  }
  assert(intWidth > 0 && "Could not determine size of an integer");

  SmallVector<DTypeValue> values;
  values.reserve(*size);
  for (size_t i = 0, e = *size; i < e; ++i) {
    APInt intVal = rhsAttr.getValues()[i].getIntVal();
    values.push_back(DTypeValue(
        APInt(intWidth, intVal.logBase2(), dtype->isSInt()), *dtype));
  }

  b.replaceOpWithNewOp<ShrOp>(
      op, op.getType(), op.getLhs(),
      ParamConstantOp::create(b, op.getLoc(),
                              SIMDAttr::get(values, op.getType())));

  return success();
}

FoldValue DivOp::unifiedFold(FoldValues operands, TargetInfoAttr target) {
  return foldSIMDDiv(operands, target);
}

/// Helper function to fold rem operations. The target is used for index types.
static OpFoldResult foldRemOp(ArrayRef<Attribute> operands,
                              TargetInfoAttr target) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  std::optional<KGENDType> dtype =
      cast<SIMDAttr>(operands.front()).getType().getResolvedDType();
  if (!dtype)
    return {};

  std::optional<int64_t> indexBitWidth;
  if (target)
    indexBitWidth = target.resolveIndexBitWidth();

  return foldSIMDOp(
      operands, indexBitWidth,
      [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
        if (rhs.isZero())
          return std::nullopt;
        return lhs % rhs;
      },
      [](APFloat lhs, APFloat rhs) -> std::optional<APFloat> {
        if (rhs.isZero())
          return std::nullopt;
        (void)lhs.mod(rhs);
        return lhs;
      });
}

OpFoldResult RemOp::fold(FoldAdaptor adaptor) {
  return foldRemOp(adaptor.getOperands(), lookupTargetInfo(*this));
}

ErrorTreeOrSuccess RemOp::interpret(ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  if (OpFoldResult result = foldRemOp(operands, state.getTarget())) {
    if (auto attr = dyn_cast<Attribute>(result)) {
      return state.mapResults(attr);
    }
  }
  return ErrorTree(getLoc(), "failed to interpret POP::RemOp");
}

ErrorTreeOrSuccess
RemOp::parametric_interpret(ArrayRef<Attribute> operands,
                            ParametricInterpreterState &state) {
  if (OpFoldResult result = foldRemOp(operands, state.getTarget())) {
    if (auto attr = dyn_cast<Attribute>(result)) {
      return state.mapResults(attr);
    }
  }
  return ErrorTree(getLoc(), "failed to interpret POP::RemOp");
}

FoldValue FloorDivOp::unifiedFold(FoldValues operands, TargetInfoAttr target) {
  return foldSIMDFloorDiv(operands, target);
}

template <typename OpT>
static bool hasEqualOperands(OpT op) {
  return op->getLhs() == op->getRhs();
}

OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  if (SIMDAttr const &res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) -> APSInt {
            return lhs > rhs ? lhs : rhs;
          },
          [](APFloat lhs, APFloat rhs) -> APFloat {
            return llvm::maxnum(lhs, rhs);
          },
          [](bool lhs, bool rhs) -> bool { return lhs | rhs; })) {
    return res;
  }
  if (hasEqualOperands(this)) {
    return getLhs();
  }
  return {};
}

OpFoldResult MinOp::fold(FoldAdaptor adaptor) {
  if (SIMDAttr const &res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) -> APSInt {
            return lhs < rhs ? lhs : rhs;
          },
          [](APFloat lhs, APFloat rhs) -> APFloat {
            return llvm::minnum(lhs, rhs);
          },
          [](bool lhs, bool rhs) -> bool { return lhs & rhs; })) {
    return res;
  }
  if (hasEqualOperands(this)) {
    return getLhs();
  }
  return {};
}

FoldValue ShlOp::unifiedFold(FoldValues operands, TargetInfoAttr target) {
  return foldSIMDShl(operands, target);
}

FoldValue ShrOp::unifiedFold(FoldValues operands, TargetInfoAttr target) {
  return foldSIMDShr(operands, target);
}

//===----------------------------------------------------------------------===//
// Ternary Operations

OpFoldResult FMAOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(),
      [](APSInt a, APSInt b, APSInt c) { return a * b + c; },
      [](APFloat a, APFloat b, APFloat c) {
        (void)a.fusedMultiplyAdd(b, c, APFloat::rmNearestTiesToEven);
        return a;
      });
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

/// We can fold loads of `pop.global_constant` ops.
OpFoldResult LoadOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getOrdering() != AtomicOrdering::NOT_ATOMIC ||
      mightBeVolatile()) {
    // Don't fold volatile or atomic loads.
    return {};
  }

  Operation *parent = getPtr().getDefiningOp();
  if (!parent)
    return {};

  // `load(global_constant())` is a load of the whole value.
  if (auto cst = dyn_cast<GlobalConstantOp>(parent))
    return cst.getValue();

  auto findValueAt = [&](GlobalConstantOp cst, uint64_t idx) -> OpFoldResult {
    auto attr = dyn_cast<POP::ArrayAttr>(cst.getValue());
    if (!attr || idx >= attr.getValues().size() ||
        attr.getType().getElementType() != getType())
      return {};
    return attr.getValues()[idx];
  };

  auto findOffsetValueAt = [&](GlobalConstantOp cst,
                               Value offset) -> OpFoldResult {
    APInt idx;
    if (!mlir::matchPattern(offset, mlir::m_ConstantInt(&idx)) ||
        idx.isNegative())
      return {};
    return findValueAt(cst, idx.getLimitedValue());
  };

  // `load(gep(global_constant()))` is a load of a specific element, if the gep
  // index is a constant.
  if (auto gep = dyn_cast<ArrayGEPOp>(parent)) {
    if (auto cst = gep.getArray().getDefiningOp<GlobalConstantOp>())
      return findOffsetValueAt(cst, gep.getIndex());
    return {};
  }

  // `load(offset(bitcast(global_constant())))` where the offset index is known.
  if (auto offset = dyn_cast<OffsetOp>(parent)) {
    if (auto bitcast = offset.getPtr().getDefiningOp<PointerBitcastOp>())
      if (auto cst = bitcast.getInput().getDefiningOp<GlobalConstantOp>())
        return findOffsetValueAt(cst, offset.getIndex());
    return {};
  }

  // `load(bitcast(global_constant())` where the element types are equal is a
  // load of the first element.
  if (auto bitcast = dyn_cast<PointerBitcastOp>(parent)) {
    if (auto cst = bitcast.getInput().getDefiningOp<GlobalConstantOp>())
      return findValueAt(cst, 0);
    return {};
  }

  return {};
}

LogicalResult LoadOp::canonicalize(LoadOp op, PatternRewriter &b) {
  if (op.getOrdering() != AtomicOrdering::NOT_ATOMIC || op.mightBeVolatile()) {
    // Don't canonicalize atomic or volatile loads.
    return failure();
  }

  // Canonicalize "store x -> ptr; tmp = load ptr" into "store; tmp = x".
  if (auto store = dyn_cast_if_present<StoreOp>(op->getPrevNode())) {
    if ((store.getPtr() == op.getPtr()) &&
        (store.getOrdering() == AtomicOrdering::NOT_ATOMIC)) {
      b.replaceOp(op, store.getArg());
      return success();
    }
  }

  // load(bitcast(ptr<struct> -> ptr<T>)), T the struct's leading (offset-zero,
  // nested) element -> load(struct.gep chain): the gep addresses the same
  // element, so the load is unchanged, but the opaque cast is gone so SROA can
  // decompose. See `bitcast_to_leading_element`.
  if (auto cast = op.getPtr().getDefiningOp<PointerBitcastOp>()) {
    auto srcPtr = dyn_cast<PointerType>(cast.getInput().getType());
    auto dstPtr = dyn_cast<PointerType>(cast.getType());
    if (srcPtr && dstPtr && isa<StructType>(srcPtr.getElementType()) &&
        srcPtr.getAddressSpace() == dstPtr.getAddressSpace()) {
      // Descend index-0 elements from the struct to the loaded type.
      Type targetElt = dstPtr.getElementType();
      Type cur = srcPtr.getElementType();
      unsigned depth = 0;
      while (cur != targetElt) {
        auto structTy = dyn_cast<StructType>(cur);
        if (!structTy)
          break;
        std::optional<SmallVector<Type>> elts = structTy.getElementTypes();
        if (!elts || elts->empty())
          break;
        cur = elts->front();
        ++depth;
      }

      if (cur == targetElt && depth > 0) {
        b.setInsertionPoint(op);
        Value ptr = cast.getInput();
        for (unsigned i = 0; i < depth; ++i)
          ptr = StructGEPOp::create(b, op.getLoc(), ptr, /*index=*/0);
        b.modifyOpInPlace(op, [&] { op.setOperand(ptr); });
        return success();
      }
    }
  }

  // Canonicalize `load(bitcast(ptr)) -> bitcast(load(ptr))` if the element type
  // is also a pointer.
  if (!isa<PointerType>(op.getType()))
    return b.notifyMatchFailure(op.getLoc(), "element type is not a pointer");
  auto bitcast = op.getPtr().getDefiningOp<PointerBitcastOp>();
  if (!bitcast || !bitcast->hasOneUse())
    return b.notifyMatchFailure(op.getLoc(), "pointer is not a bitcast");
  Value ptr = bitcast.getInput();
  auto ptrType = dyn_cast<PointerType>(ptr.getType());
  if (!ptrType || !isa<PointerType>(ptrType.getElementType()))
    return b.notifyMatchFailure(op.getLoc(), "bitcast input is not a pointer");

  // Rewrite the load in-place.
  b.setInsertionPointAfter(op);
  auto newBitcast = PointerBitcastOp::create(b, op.getLoc(), op.getType(), op);
  b.modifyOpInPlace(op, [&] {
    op.setOperand(ptr);
    Value(op).setType(ptrType.getElementType());
  });
  b.replaceAllUsesExcept(op, newBitcast, newBitcast);
  return success();
}

ErrorTreeOrSuccess LoadOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state) {
  ErrorOr<Attribute> result =
      state.readAttributeFromPointer(operands[0], getType());
  if (result.isError())
    return ErrorTree(getLoc(), result.takeError());
  return state.mapResults(result.takeValue());
}

ErrorTreeOrSuccess
LoadOp::parametric_interpret(ArrayRef<Attribute> operands,
                             ParametricInterpreterState &state) {
  ErrorOr<Attribute> result = state.readAttributeFromPointer(
      operands[0], state.getReboundType(getType()));
  if (result.isError())
    return ErrorTree(getLoc(), result.takeError());
  return state.mapResults(result.takeValue());
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

FoldValue CmpOp::unifiedFold(FoldValues operands, TargetInfoAttr target) {
  return foldSIMDCmp(getPred(), operands, target);
}

//===----------------------------------------------------------------------===//
// Bool Operation Folders
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<BoolAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<BoolAttr>(adaptor.getRhs());
  if (lhs && rhs)
    return BoolAttr::get(getContext(), lhs.getValue() && rhs.getValue());

  // Commutative operation, constant operands are pushed to the end.
  if (rhs) {
    // lhs && true == lhs
    if (rhs.getValue())
      return getLhs();

    // lhs && false == false
    return BoolAttr::get(getContext(), false);
  }
  return {};
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<BoolAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<BoolAttr>(adaptor.getRhs());
  if (lhs && rhs)
    return BoolAttr::get(getContext(), lhs.getValue() || rhs.getValue());

  // Commutative operation, constant operands are pushed to the end.
  if (rhs) {
    // lhs || false == lhs
    if (!rhs.getValue())
      return getLhs();

    // lhs || true == true
    return BoolAttr::get(getContext(), true);
  }
  return {};
}

OpFoldResult XOrOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<BoolAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<BoolAttr>(adaptor.getRhs());

  if (lhs && rhs)
    return BoolAttr::get(getContext(), lhs.getValue() ^ rhs.getValue());

  if (rhs) {
    // `xor(x, 0)` -> `x`.
    if (!rhs.getValue())
      return getLhs();

    // `xor(xor(x, 1), 1) -> x`.
    auto xorOp = getLhs().getDefiningOp<XOrOp>();
    if (xorOp && mlir::matchPattern(xorOp.getRhs(), mlir::m_One()))
      return xorOp.getLhs();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Bitwise Operation Folders
//===----------------------------------------------------------------------===//

OpFoldResult SIMDAndOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt lhs, APSInt rhs) { return lhs & rhs; },
      [](bool lhs, bool rhs) { return lhs && rhs; });
}

OpFoldResult SIMDOrOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt lhs, APSInt rhs) { return lhs | rhs; },
      [](bool lhs, bool rhs) { return lhs || rhs; });
}

OpFoldResult SIMDXOrOp::fold(FoldAdaptor adaptor) {
  auto ret = foldSIMDOp(
      adaptor.getOperands(), [](APSInt lhs, APSInt rhs) { return lhs ^ rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs ^ rhs; });

  // Prefer to fold to a constant if possible.
  if (isa_and_nonnull<Attribute>(ret))
    return ret;

  // If we couldn't fold to a constant, try some specific folds which may return
  // a value.
  SIMDAttr rhsAttr;
  if (mlir::matchPattern(getRhs(), mlir::m_Constant(&rhsAttr))) {
    // `xor(x, 0)` -> `x`.
    if (llvm::all_of(rhsAttr.getValues(), [](const DTypeValue &value) {
          return value.getData().isZero();
        })) {
      if (adaptor.getLhs())
        return adaptor.getLhs();
      return getLhs();
    }

    // `xor(xor(x, 1), 1) -> x`.
    auto pred =
        getType().getResolvedDType() == DType::kBool
            ? [](const DTypeValue &value) { return value.getBoolVal(); }
            : [](const DTypeValue &value) {
                return value.getData().isMask(value.getData().getBitWidth());
              };

    if (llvm::all_of(rhsAttr.getValues(), pred)) {
      auto xorOp = getLhs().getDefiningOp<SIMDXOrOp>();
      if (xorOp && xorOp.getRhs() == getRhs())
        return xorOp.getLhs();
    }
  }

  // Fall back to whatever the generic fold returned.
  return ret;
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

// Unlike other places, invoking foldSIMDOpResult with kOtherResult will
// require that 32 and 64 bit representation are the same, which is not needed
// to bitcast index to some other type.
// The helper function simply uses appropriate IndexFold type depending on a
// index's size within AS or calls foldSIMDOpResult if input type is not an
// index.
template <typename... OpsFns>
static OpFoldResult bitcastSIMDIndex(ArrayRef<Attribute> operands,
                                     KGENDType inputDType,
                                     KGENDType outputDType,
                                     TargetInfoAttr target, OpsFns &&...ops) {
  if (inputDType.isIndex() || inputDType.isUIndex() || inputDType.isAddress()) {
    if (!target)
      return {};
    std::optional<int64_t> indexBitWidth = target.resolveIndexBitWidth();
    return foldSIMDOpResult<kOtherResult>(operands, outputDType, indexBitWidth,
                                          std::forward<OpsFns>(ops)...);
  }
  return foldSIMDOpResult<kNoIndex>(operands, outputDType,
                                    std::forward<OpsFns>(ops)...);
}

static OpFoldResult reshape(SIMDAttr operand, KGENDType inputDType,
                            size_t inSize, KGENDType outputDType,
                            size_t outSize, TargetInfoAttr target) {
  if (!operand)
    return {};

  // The reshape is invalid.  Bool has 1 bit of data despite 8-bit storage.
  ssize_t outWidth = outputDType.isBool() ? 1 : outputDType.getWidthInBits();
  ssize_t inWidth = inputDType.isBool() ? 1 : inputDType.getWidthInBits();
  if (inSize * inWidth != outSize * outWidth)
    return {};

  SmallVector<DTypeValue> typeValues;
  auto addValue = [&outputDType, &typeValues](APInt value) -> void {
    if (outputDType.isBool()) {
      typeValues.push_back(DTypeValue(!value.isZero(), outputDType));
    } else if (outputDType.isFloat()) {
      const llvm::fltSemantics *sem = outputDType.getFloatSemantics();
      unsigned floatBits = APFloat::semanticsSizeInBits(*sem);
      APInt extractedBits = value.extractBits(floatBits, 0);
      typeValues.push_back(DTypeValue(extractedBits, outputDType));
    } else {
      typeValues.push_back(DTypeValue(value, outputDType));
    }
  };

  bool isLittleEndian =
      target ? target.getDataLayout().getIsLittleEndian() : true;
  if (inWidth < outWidth) {
    unsigned elementsPerOutput = outWidth / inWidth;
    for (unsigned outIdx = 0; outIdx < outSize; ++outIdx) {
      APInt combined(outWidth, 0);

      for (unsigned elemIdx = 0; elemIdx < elementsPerOutput; ++elemIdx) {
        unsigned inIdx = outIdx * elementsPerOutput + elemIdx;
        APInt inputValue = operand.getValues()[inIdx].getData();
        if (inputDType.isBool())
          inputValue = inputValue.trunc(1);

        unsigned shiftAmount =
            isLittleEndian ? (elemIdx * inWidth)
                           : ((elementsPerOutput - 1 - elemIdx) * inWidth);

        combined |= inputValue.zext(outWidth) << shiftAmount;
      }
      addValue(combined);
    }
  } else {
    unsigned chunkSizeBits = outWidth;
    unsigned numChunks = inWidth / outWidth;

    for (unsigned i = 0; i < inSize; ++i) {
      APInt value = operand.getValues()[i].getData();
      for (unsigned chunk = 0; chunk < numChunks; ++chunk) {
        unsigned bitPos = isLittleEndian
                              ? (chunk * chunkSizeBits)
                              : ((numChunks - 1 - chunk) * chunkSizeBits);

        APInt extractedBits = value.extractBits(chunkSizeBits, bitPos);
        addValue(extractedBits);
      }
    }
  }

  return SIMDAttr::get(
      typeValues, SIMDType::get(operand.getContext(), outSize, outputDType));
}

static OpFoldResult evaluateBitcastOp(SIMDType resultType, SIMDType inputType,
                                      TargetInfoAttr target,
                                      Attribute operand) {
  // Don't fold if the size changes. This requires knowing the endianness of the
  // target.
  std::optional<KGENDType> dtype = resultType.getResolvedDType();
  std::optional<KGENDType> inputDType = inputType.getResolvedDType();
  std::optional<unsigned> inputSize = inputType.getResolvedSize();
  std::optional<unsigned> outputSize = resultType.getResolvedSize();
  if (!dtype || !inputDType || !inputSize || !outputSize)
    return {};

  if (isa_and_nonnull<SIMDAttr>(operand) &&
      (inputSize != outputSize || inputDType->isBool() || dtype->isBool())) {
    return reshape(cast<SIMDAttr>(operand), *inputDType, *inputSize, *dtype,
                   *outputSize, target);
  }

  // Bool can only be folded through reshape() above.
  if (inputDType->isBool() || dtype->isBool())
    return {};

  if (dtype->isInt()) {
    return bitcastSIMDIndex(
        operand, *inputDType, *dtype, target,
        [&](const APSInt &in) { return APSInt(in, dtype->isUInt()); },
        [&](const APFloat &in) {
          return APSInt(in.bitcastToAPInt(), dtype->isUInt());
        });
  }
  if (dtype->isIndex() || dtype->isUIndex()) {
    return foldSIMDOpResult<kOtherResult>(
        operand, *dtype,
        [&](const APSInt &in) -> APSInt {
          // Must zero extend to 64bit, otherwise there will be segfault during
          // SIMDAttr construction as it expects 64bit index by default.
          return APSInt(in, /*isUnsigned=*/true).extend(64);
        },
        [&](const APFloat &in) -> APSInt {
          // Must zero extend to 64bit, otherwise there will be segfault during
          // SIMDAttr construction as it expects 64bit index by default.
          return APSInt(in.bitcastToAPInt(), /*isUnsigned=*/true).extend(64);
        });
  }
  assert(dtype->isFloat());
  // Check to make sure we have a supported float dtype.
  const llvm::fltSemantics *sem = dtype->getFloatSemantics();
  if (!sem)
    return {};
  return bitcastSIMDIndex(
      operand, *inputDType, *dtype, target,
      [&](const APSInt &in) { return APFloat(*sem, in); },
      [&](const APFloat &in) { return APFloat(*sem, in.bitcastToAPInt()); });
}

ErrorTreeOrSuccess BitcastOp::interpret(ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  OpFoldResult result = evaluateBitcastOp(getType(), getInput().getType(),
                                          state.getTarget(), operands.front());

  if (result && isa<Attribute>(result)) {
    return state.mapResults(cast<Attribute>(result));
  }
  return ErrorTree(getLoc(), "failed to interpret bitcast");
}

ErrorTreeOrSuccess
BitcastOp::parametric_interpret(ArrayRef<Attribute> operands,
                                ParametricInterpreterState &state) {
  auto resultType = cast<SIMDType>(state.getReboundType(getType()));
  auto inputType = cast<SIMDType>(cast<TypedAttr>(operands[0]).getType());

  OpFoldResult result = evaluateBitcastOp(resultType, inputType,
                                          state.getTarget(), operands.front());
  if (result && isa<Attribute>(result)) {
    return state.mapResults(cast<Attribute>(result));
  }
  return ErrorTree(getLoc(), "failed to interpret bitcast");
}

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor) {
  SIMDType resultType = getType();
  SIMDType inputType = getInput().getType();
  return evaluateBitcastOp(resultType, inputType, lookupTargetInfo(*this),
                           adaptor.getOperands().front());
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerBitcastOp::fold(FoldAdaptor adaptor) {
  if (auto ptr = dyn_cast_or_null<PointerAttr>(adaptor.getInput()))
    return PointerAttr::get(ptr.getAddr(), getType());

  auto cast = getInput().getDefiningOp<PointerBitcastOp>();
  if (cast && cast.getInput().getType() == getType())
    return cast.getInput();
  return {};
}

LogicalResult PointerBitcastOp::canonicalize(PointerBitcastOp op,
                                             PatternRewriter &b) {
  // `bitcast(bitcast(x)) -> bitcast(x)`, only if the intermediate bitcast has
  // one use.
  if (auto cast = op.getInput().getDefiningOp<PointerBitcastOp>()) {
    if (cast->hasOneUse()) {
      b.replaceOpWithNewOp<PointerBitcastOp>(op, op.getType(), cast.getInput());
      // Erase the intermediate cast -- its only use has been removed.
      b.eraseOp(cast);
      return success();
    }
  }

  // Bitcast of an array pointer to its element pointer is `&array[0]`; rewrite
  // to `pop.array.gep %p[0]` so SROA sees element-wise access, not an opaque
  // cast.
  if (auto srcPtr = dyn_cast<PointerType>(op.getInput().getType())) {
    if (auto arrTy = dyn_cast<ArrayType>(srcPtr.getElementType())) {
      // Skip if the bitcast also changes address space. The only allowed
      // exception is if it changes it to zero, which is acceptable now, because
      // !pop.array is always within address space 0.
      if (op.getType() == PointerType::get(arrTy.getElementType())) {
        auto zero = ParamConstantOp::create(b, op.getLoc(), b.getIndexAttr(0));
        b.replaceOpWithNewOp<ArrayGEPOp>(op, op.getType(), op.getInput(), zero);
        return success();
      }
    }
  }

  return b.notifyMatchFailure(op.getLoc(),
                              "no applicable bitcast canonicalization");
}

ErrorTreeOrSuccess PointerBitcastOp::interpret(ArrayRef<Attribute> operands,
                                               InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
PointerBitcastOp::parametric_interpret(ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state) {
  Type type = state.getReboundType(getType());

  if (auto ptr = dyn_cast_or_null<PointerAttr>(operands.front())) {
    return state.mapResults(PointerAttr::get(ptr.getAddr(), type));
  }

  return state.mapResults(operands);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

static OpFoldResult castOpfoldHelper(CastOp op, TypedAttr operand,
                                     SIMDType resultType, SIMDType inputType,
                                     SIMDType outputType,
                                     std::optional<int64_t> indexBitWidth) {
  auto in = dyn_cast_if_present<SIMDAttr>(operand);
  std::optional<KGENDType> dtype = resultType.getResolvedDType();
  if (!in || !dtype) {
    if (inputType == outputType)
      return op.getInput();
    return {};
  }

  return POP::foldCast(operand, resultType, inputType, outputType,
                       indexBitWidth);
}

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  return castOpfoldHelper(*this, cast_if_present<TypedAttr>(adaptor.getInput()),
                          getType(), getInput().getType(),
                          getOutput().getType(),
                          /*indexBitWidth=*/std::nullopt);
}

/// Canonicalize integer type `cast(cast(x : T1 to T2) : T3) -> cast(T1 to T3)`,
/// when second cast discards the result of the first cast.
LogicalResult CastOp::canonicalize(CastOp op, PatternRewriter &b) {
  auto cast = op.getInput().getDefiningOp<CastOp>();
  if (!cast)
    return b.notifyMatchFailure(op.getLoc(), "not a cast of a cast");
  if (!cast->hasOneUse())
    return b.notifyMatchFailure(op.getLoc(),
                                "intermediate cast has multiple uses");

  // The chain is `cast(cast(x : T1 to T2) : T3)`.
  std::optional<KGENDType> inType =
      cast.getInput().getType().getResolvedDType();
  std::optional<KGENDType> intermediateType = cast.getType().getResolvedDType();
  std::optional<KGENDType> outType = op.getType().getResolvedDType();

  auto isUnsupportedType = [](auto t) {
    return !t || (!t->isIntLike() && (t->isComplex() || !t->isFloat()));
  };

  Location loc = op.getLoc();
  // Both cast should convert to/from integer-like or floating point types.
  if (isUnsupportedType(inType) || isUnsupportedType(intermediateType) ||
      isUnsupportedType(outType) ||
      intermediateType->isIntLike() != inType->isIntLike() ||
      intermediateType->isIntLike() != outType->isIntLike())
    return b.notifyMatchFailure(loc, "not all types are known or supported");

  auto getWidthInBits = [&](KGENDType type) -> ssize_t {
    if (ssize_t width = type.getWidthInBits(); width != -1)
      return width;
    if (!type.isIndex() && !type.isUIndex())
      return -1;
    TargetInfoAttr targetInfo = lookupTargetInfo(op);
    if (!targetInfo)
      return -1;
    return targetInfo.resolveIndexBitWidth();
  };

  ssize_t inWidth = getWidthInBits(*inType);
  ssize_t intermediateWidth = getWidthInBits(*intermediateType);
  ssize_t outWidth = getWidthInBits(*outType);

  if (inWidth == -1 || intermediateWidth == -1 || outWidth == -1)
    return b.notifyMatchFailure(loc, "bitwidths of types are unknown");

  if (outWidth < intermediateWidth) {
    // T2 is redundant unless it already changed the value: an int T2 dropped
    // bits T3 keeps, or a float T2 rounded (`fast` accepts double rounding).
    if ((inType->isIntLike() && outWidth > inWidth) ||
        (!inType->isIntLike() && intermediateWidth <= inWidth &&
         !FastmathFlagsInterface(op).isFast())) {
      return b.notifyMatchFailure(loc,
                                  "intermediate truncation affects result");
    }
  } else if (outWidth > intermediateWidth) {
    // T3 is wider than T2. Possible to optimize:
    //  - zext(zext)
    //  - sext(sext)
    //  - fpext(fpext)
    if (intermediateType->isIntLike() &&
        intermediateType->isSInt() != outType->isSInt()) {
      return b.notifyMatchFailure(loc, "intermediate extension affects result");
    }

    // Because `pop.cast` allows `si32 -> ui32`, need to be extra careful with
    // sign in a chain of casts.
    if (intermediateWidth <= inWidth ||
        (inType->isSInt() && !intermediateType->isSInt())) {
      return b.notifyMatchFailure(loc, "intermediate extension affects result");
    }
  } else {
    // T3 reinterprets T2, so only a truncating T2 or a sign change matters.
    if (intermediateWidth < inWidth) {
      return b.notifyMatchFailure(loc, "intermediate extension affects result");
    }

    // Final cast converts integer to/from integer of a different sign.
    if (intermediateType->isInt() && !outType->isIndex() &&
        !outType->isUIndex() &&
        intermediateType->isSInt() != outType->isSInt()) {
      return b.notifyMatchFailure(loc, "intermediate extension affects result");
    }
  }

  b.replaceOpWithNewOp<CastOp>(op, op.getType(), cast.getInput());
  // Erase the intermediate cast -- its only use has been removed.
  b.eraseOp(cast);
  return success();
}

ErrorTreeOrSuccess CastOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state) {
  TypedAttr operand = cast<TypedAttr>(operands[0]);
  // First try to fold the cast. If that fails, fallback to special cases.
  auto resultType = cast<SIMDType>(getType());
  auto inputType = cast<SIMDType>(operand.getType());
  auto outputType = cast<SIMDType>(getOutput().getType());

  if (auto result =
          castOpfoldHelper(*this, operand, resultType, inputType, outputType,
                           state.getTarget().resolveIndexBitWidth())) {
    return state.mapResults(cast<Attribute>(result));
  }

  if (auto ret = validateSIMDConstruction(resultType, getLoc()))
    return ret;

  return ErrorTree(getLoc(), "failed to interpret POP::CastOp");
}

ErrorTreeOrSuccess
CastOp::parametric_interpret(ArrayRef<Attribute> operands,
                             ParametricInterpreterState &state) {
  TypedAttr operand = cast<TypedAttr>(operands[0]);
  // First try to fold the cast. If that fails, fallback to special cases.
  auto resultType = cast<SIMDType>(state.getReboundType(getType()));

  auto inputType = cast<SIMDType>(state.getReboundType(operand.getType()));
  auto outputType = cast<SIMDType>(state.getReboundType(getOutput().getType()));

  if (auto result =
          castOpfoldHelper(*this, operand, resultType, inputType, outputType,
                           state.getTarget().resolveIndexBitWidth())) {
    return state.mapResults(cast<Attribute>(result));
  }

  if (auto ret = validateSIMDConstruction(resultType, getLoc()))
    return ret;

  return ErrorTree(getLoc(), "failed to interpret POP::CastOp");
}

//===----------------------------------------------------------------------===//
// SIMDExtractElementOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDExtractElementOp::fold(FoldAdaptor adaptor) {
  // Extracting from a scalar is always going to return the scalar.
  if (getVector().getType().isScalar()) {
    if (Attribute attr = adaptor.getVector())
      return attr;
    return getVector();
  }

  auto vec = dyn_cast_if_present<SIMDAttr>(adaptor.getVector());
  auto idx = dyn_cast_if_present<IntegerAttr>(adaptor.getPosition());
  if (!vec || !idx)
    return {};
  return SIMDAttr::get(vec.getValues()[idx.getInt()], getType());
}

ErrorTreeOrSuccess SIMDExtractElementOp::interpret(ArrayRef<Attribute> operands,
                                                   InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
SIMDExtractElementOp::parametric_interpret(ArrayRef<Attribute> operands,
                                           ParametricInterpreterState &state) {
  // cast<SIMDType>(state.getReboundType(getVector().getType()));
  auto vectorType = cast<SIMDType>(
      state.getReboundType(cast<TypedAttr>(operands[0]).getType()));

  if (vectorType.isScalar()) {
    return state.mapResults(operands[0]);
  }

  auto vec = dyn_cast_if_present<SIMDAttr>(operands[0]);
  auto idx = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!vec || !idx)
    return ErrorTree(getLoc(), "non-constant inputs");

  return state.mapResults(
      SIMDAttr::get(vec.getValues()[idx.getInt()],
                    cast<SIMDType>(state.getReboundType(getType()))));
}

//===----------------------------------------------------------------------===//
// SIMDInsertElementOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDInsertElementOp::fold(FoldAdaptor adaptor) {
  auto vec = dyn_cast_if_present<SIMDAttr>(adaptor.getVector());

  // Treat insert into undef as being an insert into zero.
  if (!vec) {
    if (auto vecCst = adaptor.getVector())
      if (isa<UninitMemAttr>(vecCst))
        vec = SIMDAttr::getZeroValue(getType());
  }

  auto val = dyn_cast_if_present<SIMDAttr>(adaptor.getValue());
  auto idx = dyn_cast_if_present<IntegerAttr>(adaptor.getPosition());
  if (!vec || !val || !idx)
    return {};
  SmallVector<DTypeValue> values(vec.getValues());
  values[idx.getInt()] = val.getValues().front();
  return SIMDAttr::get(values, getType());
}

ErrorTreeOrSuccess SIMDInsertElementOp::interpret(ArrayRef<Attribute> operands,
                                                  InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
SIMDInsertElementOp::parametric_interpret(ArrayRef<Attribute> operands,
                                          ParametricInterpreterState &state) {
  auto vec = dyn_cast_if_present<SIMDAttr>(operands[0]);
  auto type = cast<SIMDType>(state.getReboundType(getType()));

  // Treat insert into undef as being an insert into zero.
  if (!vec) {
    if (auto vecCst = operands[0])
      if (isa<UninitMemAttr>(vecCst))
        vec = SIMDAttr::getZeroValue(type);
  }

  auto val = dyn_cast_if_present<SIMDAttr>(operands[1]);
  auto idx = dyn_cast_if_present<IntegerAttr>(operands[2]);
  if (!vec || !val || !idx)
    return ErrorTree(getLoc(), "non-constant inputs");

  SmallVector<DTypeValue> values(vec.getValues());
  values[idx.getInt()] = val.getValues().front();
  return state.mapResults(SIMDAttr::get(values, type));
}

//===----------------------------------------------------------------------===//
// SIMDSelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDSelectOp::fold(FoldAdaptor adaptor) {
  auto condVals = dyn_cast_or_null<SIMDAttr>(adaptor.getCondition());
  auto trueVals = dyn_cast_or_null<SIMDAttr>(adaptor.getTrueValue());
  auto falseVals = dyn_cast_or_null<SIMDAttr>(adaptor.getFalseValue());
  if (condVals && trueVals && falseVals) {
    SmallVector<DTypeValue> results;
    for (auto [cond, trueVal, falseVal] : llvm::zip(
             condVals.getValues(), trueVals.getValues(), falseVals.getValues()))
      results.push_back(cond.getBoolVal() ? trueVal : falseVal);
    return SIMDAttr::get(results, getType());
  }

  // Fold `select(x, y, y) -> y`.
  if (getTrueValue() == getFalseValue())
    return getTrueValue();

  // Check if all the values are true or false then fold to either of the
  // operands in that case.
  if (condVals) {
    bool allTrue = true, allFalse = true;
    for (auto cond : condVals.getValues()) {
      if (cond.getBoolVal())
        allFalse = false;
      else
        allTrue = false;
    }

    // Fold `select(true, x, y) -> x`
    if (allTrue)
      return getTrueValue();

    // Fold `select(false, x, y) -> y`
    if (allFalse)
      return getFalseValue();
  }

  // Fold `select(x, true, false) -> x`.
  if (getType().getResolvedDType() == KGENDType::kBool && trueVals &&
      falseVals) {
    if (llvm::all_of(
            trueVals.getValues(),
            [](const DTypeValue &value) { return value.getBoolVal(); }) &&
        llvm::all_of(falseVals.getValues(), [](const DTypeValue &value) {
          return !value.getBoolVal();
        }))
      return getCondition();
  }

  return {};
}

/// Canonicalize `select(x, false, true) -> not(x)`.
LogicalResult SIMDSelectOp::canonicalize(SIMDSelectOp op, PatternRewriter &b) {
  if (op.getType().getResolvedDType() != KGENDType::kBool)
    return b.notifyMatchFailure(op.getLoc(), "not bool dtype");

  SIMDAttr trueVals, falseVals;
  if (!mlir::matchPattern(op.getTrueValue(), mlir::m_Constant(&trueVals)) ||
      !mlir::matchPattern(op.getFalseValue(), mlir::m_Constant(&falseVals)))
    return b.notifyMatchFailure(op.getLoc(), "values are not constants");

  if (!llvm::all_of(
          trueVals.getValues(),
          [](const DTypeValue &value) { return !value.getBoolVal(); }) ||
      !llvm::all_of(falseVals.getValues(),
                    [](const DTypeValue &value) { return value.getBoolVal(); }))
    return b.notifyMatchFailure(
        op.getLoc(), "values are not 'false' and 'true' respectively");

  // The pattern has matched. Re-use the 'true' constant.
  b.replaceOpWithNewOp<SIMDXOrOp>(op, op.getCondition(), op.getFalseValue());
  return success();
}

ErrorTreeOrSuccess SIMDSelectOp::interpret(ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
SIMDSelectOp::parametric_interpret(ArrayRef<Attribute> operands,
                                   ParametricInterpreterState &state) {
  auto condVals = dyn_cast_or_null<SIMDAttr>(operands[0]);
  auto trueVals = dyn_cast_or_null<SIMDAttr>(operands[1]);
  auto falseVals = dyn_cast_or_null<SIMDAttr>(operands[2]);
  auto type = cast<SIMDType>(state.getReboundType(getType()));

  if (condVals && trueVals && falseVals) {
    SmallVector<DTypeValue> results;
    for (auto [cond, trueVal, falseVal] : llvm::zip(
             condVals.getValues(), trueVals.getValues(), falseVals.getValues()))
      results.push_back(cond.getBoolVal() ? trueVal : falseVal);
    return state.mapResults(SIMDAttr::get(results, type));
  }

  // Fold `select(x, y, y) -> y`.
  if (getTrueValue() == getFalseValue()) {
    return state.mapResults(operands[1]);
  }

  // Check if all the values are true or false then fold to either of the
  // operands in that case.
  if (condVals) {
    bool allTrue = true, allFalse = true;
    for (auto cond : condVals.getValues()) {
      if (cond.getBoolVal())
        allFalse = false;
      else
        allTrue = false;
    }

    // Fold `select(true, x, y) -> x`
    if (allTrue) {
      return state.mapResults(operands[1]);
    }

    // Fold `select(false, x, y) -> y`
    if (allFalse) {
      return state.mapResults(operands[2]);
    }
  }

  // Fold `select(x, true, false) -> x`.
  if (type.getResolvedDType() == KGENDType::kBool && trueVals && falseVals) {
    if (llvm::all_of(
            trueVals.getValues(),
            [](const DTypeValue &value) { return value.getBoolVal(); }) &&
        llvm::all_of(falseVals.getValues(), [](const DTypeValue &value) {
          return !value.getBoolVal();
        })) {

      return state.mapResults(operands[0]);
    }
  }

  return ErrorTree(getLoc(), "cannot interpret SIMDSelect");
}

//===----------------------------------------------------------------------===//
// SIMDShuffleOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDShuffleOp::fold(FoldAdaptor adaptor) {
  std::optional<int64_t> size = getType().getResolvedSize();
  auto lhs = dyn_cast_if_present<SIMDAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_if_present<SIMDAttr>(adaptor.getRhs());
  auto mask = dyn_cast_if_present<ArrayAttr>(getMaskAttr());
  if (!size || !lhs || !rhs || !mask)
    return {};

  // Is the mask a known constant.
  if (llvm::any_of(mask.getValues(), [](Attribute operand) {
        return !isa_and_nonnull<IntegerAttr>(operand);
      }))
    return {};

  // Concatenate the input simd vectors.
  SmallVector<DTypeValue> args(lhs.getValues());
  llvm::append_range(args, rhs.getValues());

  // Perform the permutation based on the mask.
  SmallVector<DTypeValue> result;
  result.reserve(mask.getValues().size());
  for (TypedAttr maskVal : mask.getValues())
    result.emplace_back(args[cast<IntegerAttr>(maskVal).getInt()]);

  return SIMDAttr::get(result, getType());
}

ErrorTreeOrSuccess SIMDShuffleOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
SIMDShuffleOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  auto type = cast<SIMDType>(state.getReboundType(getType()));
  std::optional<int64_t> size = type.getResolvedSize();
  auto lhs = dyn_cast_if_present<SIMDAttr>(operands[0]);
  auto rhs = dyn_cast_if_present<SIMDAttr>(operands[1]);
  auto mask =
      dyn_cast_if_present<ArrayAttr>(state.getReboundAttribute(getMaskAttr()));
  if (!size || !lhs || !rhs || !mask)
    return ErrorTree(getLoc(), "non-constant inputs");

  // Is the mask a known constant.
  if (llvm::any_of(mask.getValues(), [](Attribute operand) {
        return !isa_and_nonnull<IntegerAttr>(operand);
      }))
    return ErrorTree(getLoc(), "mask is unknown constant");

  // Concatenate the input simd vectors.
  SmallVector<DTypeValue> args(lhs.getValues());
  llvm::append_range(args, rhs.getValues());

  // Perform the permutation based on the mask.
  SmallVector<DTypeValue> result;
  result.reserve(mask.getValues().size());
  for (TypedAttr maskVal : mask.getValues())
    result.emplace_back(args[cast<IntegerAttr>(maskVal).getInt()]);

  return state.mapResults(SIMDAttr::get(result, type));
}

//===----------------------------------------------------------------------===//
// SIMDSplatOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDSplatOp::fold(FoldAdaptor adaptor) {
  return KGEN::foldSIMDSplat(getScalar(), adaptor.getScalar(), getType());
}

ErrorTreeOrSuccess SIMDSplatOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  SIMDType resultType = cast<SIMDType>(getType());
  if (auto ret = validateSIMDConstruction(resultType, getLoc()))
    return ret;
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
SIMDSplatOp::parametric_interpret(ArrayRef<Attribute> operands,
                                  ParametricInterpreterState &state) {
  SIMDType resultType = cast<SIMDType>(state.getReboundType(getType()));
  std::optional<int64_t> size = resultType.getResolvedSize();

  if (size == 1) {
    return state.mapResults(operands.front());
  }

  if (auto ret = validateSIMDConstruction(resultType, getLoc()))
    return ret;

  auto scalar = dyn_cast_if_present<SIMDAttr>(operands.front());
  if (!size || !scalar)
    return ErrorTree(getLoc(), "cannot find size or is not a scalar");

  SmallVector<DTypeValue> values(*size, scalar.getValues().front());
  return state.mapResults(SIMDAttr::get(values, resultType));
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::canonicalize(StoreOp op, PatternRewriter &b) {
  if (op.getOrdering() != AtomicOrdering::NOT_ATOMIC || op.mightBeVolatile()) {
    // Don't canonicalize atomic or volatile stores.
    return failure();
  }

  // Storing an unknown to a pointer is a nop as it is legal to assume the
  // memory is already the same value.
  if (auto cst = dyn_cast_or_null<KGEN::ParamConstantOp>(
          op.getArg().getDefiningOp())) {
    if (isa<UninitMemAttr>(cst.getValue())) {
      b.eraseOp(op);
      return success();
    }
  }

  // Canonicalize `store x, bitcast(ptr) -> store bitcast(x), ptr` if the
  // element type is a pointer type.
  if (!isa<PointerType>(op.getArg().getType()))
    return b.notifyMatchFailure(op.getLoc(), "arg is not a pointer");
  auto bitcast = op.getPtr().getDefiningOp<PointerBitcastOp>();
  if (!bitcast)
    return b.notifyMatchFailure(op.getLoc(), "ptr is not a bitcast");
  auto ptrType = dyn_cast<PointerType>(bitcast.getInput().getType());
  if (!ptrType || !isa<PointerType>(ptrType.getElementType()))
    return b.notifyMatchFailure(op.getLoc(), "bitcast input is not a pointer");

  // Rewrite the store in-place.
  auto newBitcast = PointerBitcastOp::create(
      b, op.getLoc(), ptrType.getElementType(), op.getArg());
  b.modifyOpInPlace(op, [&] {
    op.getPtrMutable().set(bitcast.getInput());
    op.getArgMutable().set(newBitcast);
  });
  return success();
}

ErrorTreeOrSuccess StoreOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  auto value = cast_or_null<TypedAttr>(operands[0]);
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[1]);
  if (!value || !ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  ErrorOrSuccess result = state.writeAttributeToMemory(ptr.getAddr(), value);
  if (result.isError())
    return ErrorTree(getLoc(), result.takeError());
  return success();
}

ErrorTreeOrSuccess
StoreOp::parametric_interpret(ArrayRef<Attribute> operands,
                              ParametricInterpreterState &state) {
  auto value = cast_or_null<TypedAttr>(operands[0]);
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[1]);
  if (!value || !ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  ErrorOrSuccess result = state.writeAttributeToMemory(ptr.getAddr(), value);
  if (result.isError())
    return ErrorTree(getLoc(), result.takeError());
  return success();
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess OffsetOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  if (!state.getTarget())
    return ErrorTree(getLoc(), "operation requires a target model");

  auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]);
  auto offset = dyn_cast_or_null<IntegerAttr>(operands[1]);
  if (!ptr || !offset)
    return ErrorTree(getLoc(), "non-constant inputs");
  // Stride by the alloc size, matching parametric_interpret and the LLVM GEP
  // lowering: for types whose store size is not a multiple of their alignment.
  std::optional<int64_t> elSize = DataLayoutInterface::getTypeAllocSize(
      state.getTarget(), cast<PointerType>(ptr.getType()).getElementType());
  if (!elSize)
    return ErrorTree(getLoc(), "could not query pointer element size");
  return state.mapResults(PointerAttr::get(
      ptr.getAddr() + *elSize * offset.getInt(), ptr.getType()));
}

ErrorTreeOrSuccess
OffsetOp::parametric_interpret(ArrayRef<Attribute> operands,
                               ParametricInterpreterState &state) {
  if (!state.getTarget())
    return ErrorTree(getLoc(), "operation requires a target model");

  auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]);
  auto offset = dyn_cast_or_null<IntegerAttr>(operands[1]);
  if (!ptr || !offset)
    return ErrorTree(getLoc(), "non-constant inputs");
  auto ptrType = state.getReboundType(ptr.getType());
  auto elemType = cast<PointerType>(ptrType).getElementType();

  std::optional<int64_t> elSize =
      DataLayoutInterface::getTypeAllocSize(state.getTarget(), elemType);
  if (!elSize)
    return ErrorTree(getLoc(), "could not query pointer element size");
  return state.mapResults(
      PointerAttr::get(ptr.getAddr() + *elSize * offset.getInt(), ptrType));
}

OpFoldResult OffsetOp::fold(FoldAdaptor adaptor) {
  auto offset = dyn_cast_or_null<IntegerAttr>(adaptor.getIndex());
  if (!offset)
    return {};

  if (offset.getInt() != 0)
    return {};

  return getPtr();
}

LogicalResult OffsetOp::canonicalize(OffsetOp op, PatternRewriter &b) {
  // Canonicalize (offset (array.gep a, 0), j) -> (array.gep a, i)
  // TODO: could generalize (offset (array.gep a, i), j) -> (array.gep a, i + j)
  // but we don't have a way to add two index values here.
  if (auto gep = op.getPtr().getDefiningOp<ArrayGEPOp>()) {
    if (!mlir::matchPattern(gep.getIndex(), mlir::m_Zero()))
      return b.notifyMatchFailure(op, "not a constant offset");

    b.replaceOpWithNewOp<ArrayGEPOp>(op, op.getType(), gep.getArray(),
                                     op.getIndex());
    return success();
  }

  // Canonicalize `%ptr[%c0][%c1] -> %ptr[%c0 + %c1]`, where the indices are
  // constants.
  APInt c1;
  if (!mlir::matchPattern(op.getIndex(), mlir::m_ConstantInt(&c1)))
    return b.notifyMatchFailure(op, "not a constant offset");
  auto parent = op.getPtr().getDefiningOp<OffsetOp>();
  if (!parent)
    return b.notifyMatchFailure(op, "parent is not an offset");
  APInt c0;
  if (!mlir::matchPattern(parent.getIndex(), mlir::m_ConstantInt(&c0)))
    return b.notifyMatchFailure(op, "parent is not a constant offset");

  // However unlikely, don't canonicalize if the arithmetic overflows. Note that
  // it is always valid to fold addition of index values, regardless of width.
  bool ov = false;
  APInt newOffset = c0.sadd_ov(c1, ov);
  if (ov)
    return b.notifyMatchFailure(op, "offset addition overflows");

  b.replaceOpWithNewOp<OffsetOp>(
      op, parent.getPtr(),
      ParamConstantOp::create(b, op.getLoc(),
                              b.getIndexAttr(newOffset.getSExtValue())));
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  // Narrow to one of the conditional values.
  if (auto cond = dyn_cast_if_present<SIMDAttr>(adaptor.getCondition())) {
    if (cond.getAsBool()) {
      if (Attribute attr = adaptor.getTrueValue())
        return attr;
      return getTrueValue();
    }
    if (Attribute attr = adaptor.getFalseValue())
      return attr;
    return getFalseValue();
  }

  // Fold `select x, true, false -> x`.
  if (getCondition().getType() == getType()) {
    auto trueAttr = dyn_cast_if_present<SIMDAttr>(adaptor.getTrueValue());
    auto falseAttr = dyn_cast_if_present<SIMDAttr>(adaptor.getFalseValue());
    if (trueAttr && falseAttr && trueAttr.getAsBool() == true &&
        falseAttr.getAsBool() == false)
      return getCondition();
  }

  // Fold `select x, undef, y -> y` and `select x, y, undef -> y`.
  if (isa_and_nonnull<UninitMemAttr>(adaptor.getTrueValue()))
    return getFalseValue();
  if (isa_and_nonnull<UninitMemAttr>(adaptor.getFalseValue()))
    return getTrueValue();

  // `x ? y : y -> y`.
  if (getTrueValue() == getFalseValue())
    return getTrueValue();

  return {};
}

namespace {

/// Canonicalize `select x, (select x, a, b), c` into `select x, a, c` or
/// `select x, a, (select x, b, c)` into `select x, a, c`.
struct SelectOfSelect : OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &b) const override {
    bool knownCondition;
    SelectOp branchSelect;
    if ((branchSelect = op.getTrueValue().getDefiningOp<SelectOp>())) {
      knownCondition = true;
    } else if ((branchSelect = op.getFalseValue().getDefiningOp<SelectOp>())) {
      knownCondition = false;
    } else {
      return b.notifyMatchFailure(
          op.getLoc(), "true or false value not defined by a select");
    }
    if (branchSelect.getCondition() != op.getCondition()) {
      return b.notifyMatchFailure(op.getLoc(),
                                  "branch select condition is not the same");
    }
    Value foldedValue = knownCondition ? branchSelect.getTrueValue()
                                       : branchSelect.getFalseValue();
    b.modifyOpInPlace(
        op, [&] { op->setOperand(1 + (knownCondition ? 0 : 1), foldedValue); });
    return success();
  }
};
} // namespace

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<SelectOfSelect>(context);
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess StackAllocationOp::compile(Payload &payload,
                                          TargetInfoAttr target) {
  auto countAttr = dyn_cast<IntegerAttr>(getCount());
  if (!countAttr)
    return Error("array size is not a constant");
  int64_t count = countAttr.getInt();

  if (!target) {
    if (count != 1)
      return Error("array allocation requires a target model");
    return success();
  }

  // Determine the allocation size.
  Type type = cast<PointerType>(getType()).getElementType();
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeAllocSize(target, type);
  if (!size)
    return Error("could not query type size");

  // Determine the alignment. If the alignment is unspecified or zero, query
  // the natural alignment of the type.
  int64_t align = 0;
  if (TypedAttr alignAttr = getAlignmentAttr())
    align = cast<IntegerAttr>(alignAttr).getInt();
  if (align < 0)
    return Error("invalid alignment value: " + Twine(align));
  if (align == 0) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeABIAlign(target, type);
    if (!typeAlign)
      return Error("could not query type alignment");
    align = *typeAlign;
  }

  payload.size = count * *size;
  payload.align = align;
  return success();
}

ErrorOrSuccess
StackAllocationOp::parametric_compile(Payload &payload, TargetInfoAttr target,
                                      ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  auto countAttr = dyn_cast<IntegerAttr>(state.getReboundAttribute(getCount()));

  if (!countAttr)
    return Error("array size is not a constant");
  int64_t count = countAttr.getInt();

  if (!target) {
    if (count != 1)
      return Error("array allocation requires a target model");
    return success();
  }

  Type reboundType = state.getReboundType(getType());

  // Determine the allocation size.
  Type type = cast<PointerType>(reboundType).getElementType();
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeAllocSize(target, type);
  if (!size)
    return Error("could not query type size");

  // Determine the alignment. If the alignment is unspecified or zero, query
  // the natural alignment of the type.
  int64_t align = 0;
  if (getAlignmentAttr()) {
    TypedAttr alignAttr = state.getReboundAttribute(getAlignmentAttr());
    align = cast<IntegerAttr>(alignAttr).getInt();
  }

  if (align < 0)
    return Error("invalid alignment value: " + Twine(align));
  if (align == 0) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeABIAlign(target, type);
    if (!typeAlign)
      return Error("could not query type alignment");
    align = *typeAlign;
  }

  payload.size = count * *size;
  payload.align = align;
  return success();
}

ErrorTreeOrSuccess StackAllocationOp::interpret(ArrayRef<Attribute> operands,
                                                const Payload &payload,
                                                InterpreterState &state) {
  // If there is no target model, we know it is a count 1 alloc.
  if (!state.getTarget())
    return ErrorTree(getLoc(), "stack allocation requires a target model");

  ErrorOr<int64_t> addr =
      state.allocateStackMemory(payload.size, payload.align);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());
  return state.mapResults(PointerAttr::get(addr.takeValue(), getType()));
}

ErrorTreeOrSuccess
StackAllocationOp::parametric_interpret(ArrayRef<Attribute> operands,
                                        const Payload &payload,
                                        ParametricInterpreterState &state) {
  // If there is no target model, we know it is a count 1 alloc.
  if (!state.getTarget())
    return ErrorTree(getLoc(), "stack allocation requires a target model");

  ErrorOr<int64_t> addr =
      state.allocateStackMemory(payload.size, payload.align);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());
  return state.mapResults(
      PointerAttr::get(addr.takeValue(), state.getReboundType(getType())));
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeStartOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
StackAllocLifetimeStartOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state) {
  return success();
}

ErrorTreeOrSuccess StackAllocLifetimeStartOp::parametric_interpret(
    ArrayRef<Attribute> operands, ParametricInterpreterState &state) {
  return success();
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeEndOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
StackAllocLifetimeEndOp::interpret(ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  return success();
}

ErrorTreeOrSuccess StackAllocLifetimeEndOp::parametric_interpret(
    ArrayRef<Attribute> operands, ParametricInterpreterState &state) {
  return success();
}

//===----------------------------------------------------------------------===//
// AlignedFreeOp
//===----------------------------------------------------------------------===//

LogicalResult AlignedFreeOp::canonicalize(AlignedFreeOp op,
                                          PatternRewriter &b) {
  auto bitcast = op.getPtr().getDefiningOp<PointerBitcastOp>();
  if (!bitcast)
    return failure();
  b.modifyOpInPlace(op, [&] { op.getPtrMutable().set(bitcast.getInput()); });
  return success();
}

//===----------------------------------------------------------------------===//
// AlignedAllocOp
//===----------------------------------------------------------------------===//

/// Interpret an aligned allocation.
static ErrorTreeOrSuccess interpretAllocation(int64_t size, int64_t align,
                                              Location loc, Type type,
                                              InterpreterState &state) {
  // The default "system" alignment technically has no guarantees and varies
  // depending on the underlying allocator implementation. Just use 64 for
  // consistency.
  if (align <= 0)
    align = 64;

  ErrorOr<int64_t> addr = state.allocateHeapMemory(size, align);
  if (addr.isError())
    return ErrorTree(loc, addr.takeError());
  return state.mapResults(PointerAttr::get(addr.takeValue(), type));
}

ErrorTreeOrSuccess AlignedAllocOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  ErrorOr<int64_t> alignOr =
      getScalarIndexValue(dyn_cast_or_null<TypedAttr>(operands.front()));
  ErrorOr<int64_t> sizeOr =
      getScalarIndexValue(dyn_cast_or_null<TypedAttr>(operands.back()));
  if (alignOr.isError() || sizeOr.isError())
    return ErrorTree(getLoc(), "non-concrete inputs");

  if (sizeOr.get() < 0) {
    return ErrorTree(getLoc(), "alloc has negative size");
  }
  return interpretAllocation(sizeOr.get(), alignOr.get(), getLoc(), getType(),
                             state);
}

ErrorTreeOrSuccess
AlignedAllocOp::parametric_interpret(ArrayRef<Attribute> operands,
                                     ParametricInterpreterState &state) {
  ErrorOr<int64_t> alignOr =
      getScalarIndexValue(dyn_cast_or_null<TypedAttr>(operands.front()));
  ErrorOr<int64_t> sizeOr =
      getScalarIndexValue(dyn_cast_or_null<TypedAttr>(operands.back()));
  if (alignOr.isError() || sizeOr.isError())
    return ErrorTree(getLoc(), "non-concrete inputs");
  return interpretAllocation(sizeOr.get(), alignOr.get(), getLoc(),
                             state.getReboundType(getType()), state);
}

//===----------------------------------------------------------------------===//
// AlignedFreeOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess AlignedFreeOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  auto ptr = cast<PointerAttr>(operands.front());
  if (ErrorOrSuccess err = state.freeHeapMemory(ptr.getAddr()); err.isError())
    return ErrorTree(getLoc(), err.takeError());
  return success();
}

ErrorTreeOrSuccess
AlignedFreeOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  auto ptr = cast<PointerAttr>(operands.front());
  if (ErrorOrSuccess err = state.freeHeapMemory(ptr.getAddr()); err.isError())
    return ErrorTree(getLoc(), err.takeError());
  return success();
}

//===----------------------------------------------------------------------===//
// MemcpyOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess MemcpyOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  return interpretMemcpy(operands[0], operands[1], operands[2], getLoc(),
                         state);
}

ErrorTreeOrSuccess
MemcpyOp::parametric_interpret(ArrayRef<Attribute> operands,
                               ParametricInterpreterState &state) {
  return interpretMemcpy(state.getReboundAttribute(operands[0]),
                         state.getReboundAttribute(operands[1]),
                         state.getReboundAttribute(operands[2]), getLoc(),
                         state);
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayCreateOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> operands = adaptor.getOperands();
  SmallVector<TypedAttr> values;
  values.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return POP::ArrayAttr::get(values, getType());
}

ErrorTreeOrSuccess ArrayCreateOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
ArrayCreateOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  SmallVector<TypedAttr> values;
  values.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return state.mapResults(POP::ArrayAttr::get(
      values, cast<ArrayType>(state.getReboundType(getType()))));
}

//===----------------------------------------------------------------------===//
// ArrayRepeatOp
//===----------------------------------------------------------------------===//

static Attribute foldInterpretArrayRepeatOp(ArrayRef<Attribute> operands,
                                            POP::ArrayType type) {
  std::optional<int64_t> size = type.getResolvedSize();
  if (!size)
    return {};
  assert(size >= 0 && "size is non-negative");
  SmallVector<TypedAttr> args;
  args.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    args.push_back(value);
  }
  SmallVector<TypedAttr> values;
  values.reserve(*size);
  while (static_cast<int64_t>(values.size()) < *size)
    values.append(args);
  return POP::ArrayAttr::get(ArrayRef(values).take_front(*size), type);
}

OpFoldResult ArrayRepeatOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> operands = adaptor.getOperands();
  return foldInterpretArrayRepeatOp(operands, getType());
}

ErrorTreeOrSuccess ArrayRepeatOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
ArrayRepeatOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  auto boundResultType = cast<POP::ArrayType>(state.getReboundType(getType()));
  if (Attribute attr = foldInterpretArrayRepeatOp(operands, boundResultType)) {
    return state.mapResults(attr);
  }

  return ErrorTree(getLoc(), "cannot interpret pop.array.repeat");
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayGetOp::fold(FoldAdaptor adaptor) {

  // If the array comes from an undef constant then the result is also undef,
  // irrespective of the index.
  if (auto cst =
          dyn_cast_or_null<KGEN::ParamConstantOp>(getArray().getDefiningOp())) {
    if (isa<UninitMemAttr>(cst.getValue()))
      return UninitMemAttr::get(getType());
    if (isa<UnknownAttr>(cst.getValue()))
      return UnknownAttr::get(getType());
  }

  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (!index)
    return {};

  std::optional<int64_t> size = getArray().getType().getResolvedSize();
  if (!size)
    return {};

  // Bounds check the array access.
  int64_t idx = index.getInt();
  if (idx < 0 || idx >= *size)
    return {};

  // Try fold if the array is a constant.
  auto array = dyn_cast_if_present<POP::ArrayAttr>(adaptor.getArray());
  if (array)
    return array.getValues()[idx];

  // If we directly come from an `ArrayCreate` we can just fold to the operand
  // of that.
  if (auto arrayCreate = getArray().getDefiningOp<POP::ArrayCreateOp>())
    return arrayCreate.getOperand(idx);

  // If we come from a repeat we can work out which operand we are.
  if (auto repeat = getArray().getDefiningOp<POP::ArrayRepeatOp>())
    return repeat.getOperand(idx % repeat.getNumOperands());

  return {};
}

ErrorTreeOrSuccess ArrayGetOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
ArrayGetOp::parametric_interpret(ArrayRef<Attribute> operands,
                                 ParametricInterpreterState &state) {
  auto index = dyn_cast<IntegerAttr>(state.getReboundAttribute(getIndex()));
  std::optional<int64_t> size =
      cast<ArrayType>(cast<TypedAttr>(operands[0]).getType()).getResolvedSize();
  if (!index || !size)
    return ErrorTree(getLoc(), "non-constant inputs");

  // Bounds check the array access.
  int64_t idx = index.getInt();
  if (idx < 0 || idx >= *size)
    return ErrorTree(getLoc(), "out-of-bound array get");

  // Try fold if the array is a constant.
  auto array = dyn_cast_if_present<POP::ArrayAttr>(operands[0]);
  if (!array)
    return ErrorTree(getLoc(), "non-constant inputs");

  return state.mapResults(array.getValues()[idx]);
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayReplaceOp::fold(FoldAdaptor adaptor) {
  auto value = llvm::cast_if_present<TypedAttr>(adaptor.getValue());
  auto array = dyn_cast_if_present<POP::ArrayAttr>(adaptor.getArray());
  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (!value || !array || !index)
    return {};
  SmallVector<TypedAttr> values(array.getValues());
  values[index.getInt()] = value;
  return POP::ArrayAttr::get(values, getType());
}

LogicalResult ArrayReplaceOp::canonicalize(ArrayReplaceOp op,
                                           PatternRewriter &rewriter) {
  auto indexAttr = dyn_cast<IntegerAttr>(op.getIndex());
  if (!indexAttr)
    return rewriter.notifyMatchFailure(op, "dynamic index not supported");
  unsigned index = indexAttr.getInt();

  if (auto arrayCreateOp = op.getArray().getDefiningOp<ArrayCreateOp>()) {
    SmallVector<Value> newOperands = arrayCreateOp.getOperands();
    newOperands[index] = op.getValue();
    rewriter.replaceOpWithNewOp<ArrayCreateOp>(op, newOperands);
    return success();
  }

  if (auto paramConstantOp =
          op.getArray().getDefiningOp<KGEN::ParamConstantOp>()) {
    auto constArr = cast<POP::ArrayAttr>(paramConstantOp.getValue());
    SmallVector<Value> newOperands;
    newOperands.reserve(constArr.getValues().size());
    for (unsigned i = 0, e = constArr.getValues().size(); i < e; ++i) {
      if (i == index)
        newOperands.push_back(op.getValue());
      else
        newOperands.push_back(KGEN::ParamConstantOp::create(
            rewriter, paramConstantOp.getLoc(), constArr.getValues()[i]));
    }
    rewriter.replaceOpWithNewOp<ArrayCreateOp>(op, newOperands);
    return success();
  }

  return rewriter.notifyMatchFailure(
      op, "array must be a constant or an ArrayCreateOp");
}

ErrorTreeOrSuccess ArrayReplaceOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
ArrayReplaceOp::parametric_interpret(ArrayRef<Attribute> operands,
                                     ParametricInterpreterState &state) {
  auto value = llvm::cast_if_present<TypedAttr>(operands[0]);
  auto array = dyn_cast_if_present<POP::ArrayAttr>(operands[1]);
  auto index = dyn_cast<IntegerAttr>(state.getReboundAttribute(getIndex()));
  if (!value || !array || !index)
    return ErrorTree(getLoc(), "non-constant inputs");

  SmallVector<TypedAttr> values(array.getValues());
  values[index.getInt()] = value;

  return state.mapResults(POP::ArrayAttr::get(
      values, cast<ArrayType>(state.getReboundType(getType()))));
}

//===----------------------------------------------------------------------===//
// ArrayGEPOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ArrayGEPOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  if (!state.getTarget())
    return ErrorTree(getLoc(), "operation requires a target model");

  auto ptr = dyn_cast_if_present<PointerAttr>(operands[0]);
  auto index = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!ptr || !index)
    return ErrorTree(getLoc(), "non-constant inputs");

  auto arrayType = getArray().getType().getElementAs<POP::ArrayType>();
  Type elementType = arrayType.getElementType();
  std::optional<int64_t> stride =
      DataLayoutInterface::getTypeAllocSize(state.getTarget(), elementType);
  if (!stride)
    return ErrorTree(getLoc(), "failed to get array element stride");
  int64_t addr = ptr.getAddr() + index.getInt() * *stride;
  return state.mapResults(
      PointerAttr::get(addr, PointerType::get(elementType)));
}

ErrorTreeOrSuccess
ArrayGEPOp::parametric_interpret(ArrayRef<Attribute> operands,
                                 ParametricInterpreterState &state) {

  if (!state.getTarget())
    return ErrorTree(getLoc(), "operation requires a target model");

  auto ptr = dyn_cast_if_present<PointerAttr>(operands[0]);
  auto index = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!ptr || !index)
    return ErrorTree(getLoc(), "non-constant inputs");

  auto arrayType = cast<PointerType>(cast<TypedAttr>(operands[0]).getType())
                       .getElementAs<POP::ArrayType>();
  Type elementType = arrayType.getElementType();
  std::optional<int64_t> stride =
      DataLayoutInterface::getTypeAllocSize(state.getTarget(), elementType);
  if (!stride)
    return ErrorTree(getLoc(), "failed to get array element stride");
  int64_t addr = ptr.getAddr() + index.getInt() * *stride;
  return state.mapResults(
      PointerAttr::get(addr, PointerType::get(elementType)));
}

LogicalResult ArrayGEPOp::canonicalize(ArrayGEPOp op,
                                       PatternRewriter &rewriter) {
  PointerType ptrToArray = op.getArray().getType();
  std::optional<int64_t> size =
      ptrToArray.getElementAs<ArrayType>().getResolvedSize();

  // We are only going to canonicalize scalars.
  if (!size || *size != 1)
    return rewriter.notifyMatchFailure(op, "Size is not known to be scalar");

  // Don't repeatedly canonicalize already constant values.
  if (auto indexOp = op.getIndex().getDefiningOp();
      indexOp && indexOp->hasTrait<OpTrait::ConstantLike>())
    return rewriter.notifyMatchFailure(op,
                                       "ArrayGEP index is already constant.");

  // Otherwise we have gep into a array of one element with a dynamic value. It
  // is undefined behaviour for that to be anything but `0` so we can replace it
  // with the constant `0`. This frees the use to be DCE'd and unblocks other
  // optimizations.
  auto zero =
      ParamConstantOp::create(rewriter, op.getLoc(), rewriter.getIndexAttr(0));
  rewriter.replaceOpWithNewOp<ArrayGEPOp>(op, op.getType(), op.getArray(),
                                          zero);
  return success();
}

//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerToIndexOp::fold(FoldAdaptor adaptor) {
  // Check for a pointer input. The result must be a scalar index.
  if (auto ptr = dyn_cast_if_present<PointerAttr>(adaptor.getValue()))
    return Builder(getContext()).getIndexAttr(ptr.getAddr());
  return {};
}

LogicalResult PointerToIndexOp::canonicalize(PointerToIndexOp op,
                                             PatternRewriter &b) {
  auto bitcast = op.getValue().getDefiningOp<PointerBitcastOp>();
  if (!bitcast || !isa<PointerType>(bitcast.getInput().getType()))
    return failure();
  b.modifyOpInPlace(op, [&] { op.getValueMutable().set(bitcast.getInput()); });
  return success();
}

ErrorTreeOrSuccess PointerToIndexOp::interpret(ArrayRef<Attribute> operands,
                                               InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
PointerToIndexOp::parametric_interpret(ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state) {
  if (auto ptr = dyn_cast_if_present<PointerAttr>(operands.front())) {
    return state.mapResults(Builder(getContext()).getIndexAttr(ptr.getAddr()));
  }

  return ErrorTree(getLoc(), "input pointer is not an index");
}

//===----------------------------------------------------------------------===//
// CompilerGlobalLoadOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess CompilerGlobalLoadOp::interpret(ArrayRef<Attribute> operands,
                                                   InterpreterState &state) {
  Attribute value = state.getNamedGlobal(getNameAttr());
  if (!value)
    return ErrorTree(
        getLoc(),
        "cannot evaluate standalone capturing closure at compile time");
  // The global map is keyed by name, and one parametric generator's
  // instantiations share a name while their capture types can differ.
  if (!isAttributeTypeCompatible(value, getResult().getType())) {
    return ErrorTree(getLoc(),
                     "internal error: compiler global '" + getName().str() +
                         "' holds a value whose type does not match the type "
                         "it is loaded as");
  }
  return state.mapResults(value);
}

ErrorTreeOrSuccess
CompilerGlobalLoadOp::parametric_interpret(ArrayRef<Attribute> operands,
                                           ParametricInterpreterState &state) {
  Attribute value = state.getNamedGlobal(getNameAttr());
  if (!value)
    return ErrorTree(
        getLoc(),
        "cannot evaluate standalone capturing closure at compile time");
  // The global map is keyed by name, and one parametric generator's
  // instantiations share a name while their capture types can differ.
  if (!isAttributeTypeCompatible(value, getResult().getType())) {
    return ErrorTree(getLoc(),
                     "internal error: compiler global '" + getName().str() +
                         "' holds a value whose type does not match the type "
                         "it is loaded as");
  }
  return state.mapResults(value);
}

//===----------------------------------------------------------------------===//
// CompilerGlobalStoreOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
CompilerGlobalStoreOp::interpret(ArrayRef<Attribute> operands,
                                 InterpreterState &state) {
  state.setNamedGlobal(getNameAttr(), operands.front());
  return success();
}

ErrorTreeOrSuccess
CompilerGlobalStoreOp::parametric_interpret(ArrayRef<Attribute> operands,
                                            ParametricInterpreterState &state) {
  state.setNamedGlobal(getNameAttr(), operands.front());
  return success();
}

//===----------------------------------------------------------------------===//
// CastToBuiltinOp
//===----------------------------------------------------------------------===//

/// Convert a SIMD attribute to a vector-typed attribute.
template <typename AttrT, typename TransformFn>
static ArrayElementsAttr convertSIMDToVectorAttr(SIMDAttr simd, VectorType type,
                                                 TransformFn fn) {
  SmallVector<decltype(fn(std::declval<DTypeValue>()))> values;
  for (const DTypeValue &value : simd.getValues())
    values.push_back(fn(value));
  return AttrT::get(type, values);
}

OpFoldResult CastToBuiltinOp::fold(FoldAdaptor adaptor) {
  auto simd = dyn_cast_if_present<SIMDAttr>(adaptor.getInput());
  if (!simd) {
    // Fold A->B->A cast.
    if (auto parent = getInput().getDefiningOp<CastFromBuiltinOp>();
        parent && parent.getInput().getType() == getType())
      return parent.getInput();
    return {};
  }

  return KGEN::foldCastToBuiltin(simd, getType());
}

//===----------------------------------------------------------------------===//
// CastFromBuiltinOp
//===----------------------------------------------------------------------===//

OpFoldResult CastFromBuiltinOp::fold(FoldAdaptor adaptor) {
  auto val = llvm::cast_if_present<TypedAttr>(adaptor.getInput());
  if (!val) {
    // Fold A->B->A cast.
    if (auto parent = getInput().getDefiningOp<CastToBuiltinOp>();
        parent && parent.getInput().getType() == getType())
      return parent.getInput();
    return {};
  }

  return KGEN::foldCastFromBuiltin(val, getType());
}

ErrorTreeOrSuccess CastFromBuiltinOp::interpret(ArrayRef<Attribute> operands,
                                                InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
CastFromBuiltinOp::parametric_interpret(ArrayRef<Attribute> operands,
                                        ParametricInterpreterState &state) {
  auto val = llvm::cast_if_present<TypedAttr>(operands.front());
  SIMDType type = cast<SIMDType>(state.getReboundType(getType()));

  if (!val) {
    return state.mapResults(operands);
  }

  // Ensure the incoming value is an expected constant kind.
  if (!isa<IntArrayElementsAttr, FloatArrayElementsAttr, IndexArrayElementsAttr,
           IntegerAttr, FloatAttr>(val)) {
    return state.mapResults(operands);
  }

  // Conversion from vector constant.
  std::optional<KGENDType> dtype = type.getResolvedDType();

  if (!dtype)
    return {};
  if (auto vector = dyn_cast<VectorType>(val.getType())) {
    SmallVector<DTypeValue> values;
    if (dtype->isBool())
      for (APInt value : cast<IntArrayElementsAttr>(val).getValues())
        values.emplace_back(!value.isZero(), *dtype);
    else if (dtype->isIndex())
      for (int64_t value : cast<IndexArrayElementsAttr>(val))
        values.emplace_back(value, *dtype);
    else if (dtype->isInt())
      for (APInt value : cast<IntArrayElementsAttr>(val).getValues())
        values.emplace_back(value, *dtype);
    else
      for (APFloat value : cast<FloatArrayElementsAttr>(val).getValues())
        values.emplace_back(value, *dtype);
    return state.mapResults(SIMDAttr::get(values, getType()));
  }

  // Handle scalar constants.
  if (dtype->isBool()) {
    return state.mapResults(
        SIMDAttr::get({cast<BoolAttr>(val).getValue(), *dtype}, type));
  }

  if (dtype->isIndex()) {
    return state.mapResults(
        SIMDAttr::get({cast<IntegerAttr>(val).getInt(), *dtype}, type));
  }
  if (dtype->isInt()) {
    return state.mapResults(
        SIMDAttr::get({cast<IntegerAttr>(val).getValue(), *dtype}, type));
  }

  assert(dtype->isFloat() && "unexpected dtype");
  return state.mapResults(
      SIMDAttr::get({cast<FloatAttr>(val).getValue(), *dtype}, type));
}

//===----------------------------------------------------------------------===//
// GlobalConstantOp
//===----------------------------------------------------------------------===//
static ErrorTreeOrSuccess
interpretGlobalConstantOpHelper(TypedAttr input, Location loc, MLIRContext *ctx,
                                Type resultType, InterpreterState &state) {
  // Determine the allocation size.
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeAllocSize(state.getTarget(), input.getType());
  if (!size)
    return ErrorTree(loc, "cannot get pop.global_constant value size");

  // Determine the allocation alignment.
  std::optional<int64_t> align =
      DataLayoutInterface::getTypeABIAlign(state.getTarget(), input.getType());
  if (!align)
    return ErrorTree(loc, "cannot get pop.global_constant value alignment");

  ErrorOr<int64_t> addr =
      state.writeAttributeToConstantGlobalMemory(input, *size, *align);

  if (addr.isError())
    return ErrorTree(loc, addr.takeError());

  (void)state.mapResults(PointerAttr::get(ctx, *addr, resultType));

  return success();
}

ErrorTreeOrSuccess GlobalConstantOp::interpret(ArrayRef<Attribute> operands,
                                               InterpreterState &state) {
  return interpretGlobalConstantOpHelper(getValue(), getLoc(), getContext(),
                                         getType(), state);
}

ErrorTreeOrSuccess
GlobalConstantOp::parametric_interpret(ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state) {
  return interpretGlobalConstantOpHelper(
      state.getReboundAttribute(getValue()), getLoc(), getContext(),
      state.getReboundType(getType()), state);
}

//===----------------------------------------------------------------------===//
// StringAddressOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess StringAddressOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  // Ensure the string is null-terminated. This is safe because `StringAttr`
  // always stores a null terminator.
  auto value = dyn_cast<StringAttr>(operands.front());
  if (!value)
    return ErrorTree(getLoc(), Error("argument is not a concrete string"));
  StringRef str(value.data(), value.size() + 1);
  if (value.getValue().empty())
    str = "\0";

  MemoryHandleAttr hdl = MemoryHandleAttr::get(getContext(), str);
  ErrorOr<int64_t> addr = state.mapConstGlobalMemory(hdl);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());
  return state.mapResults(
      PointerAttr::get(getContext(), addr.takeValue(), getType()));
}

ErrorTreeOrSuccess
StringAddressOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  // Ensure the string is null-terminated. This is safe because `StringAttr`
  // always stores a null terminator.
  auto value = dyn_cast<StringAttr>(operands.front());
  if (!value)
    return ErrorTree(getLoc(), Error("argument is not a concrete string"));
  StringRef str(value.data(), value.size() + 1);
  if (value.getValue().empty())
    str = "\0";

  MemoryHandleAttr hdl = MemoryHandleAttr::get(getContext(), str);
  ErrorOr<int64_t> addr = state.mapConstGlobalMemory(hdl);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());
  return state.mapResults(PointerAttr::get(getContext(), addr.takeValue(),
                                           state.getReboundType(getType())));
}

//===----------------------------------------------------------------------===//
// StringSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult StringSizeOp::fold(FoldAdaptor adaptor) {
  if (auto str = dyn_cast_or_null<TypedAttr>(adaptor.getStr()))
    return StringSizeAttr::get(getContext(), str);
  return {};
}

//===----------------------------------------------------------------------===//
// DTypeToUI8
//===----------------------------------------------------------------------===//

OpFoldResult DTypeToUI8::fold(FoldAdaptor adaptor) {
  auto ui8Type = IntegerType::get(getContext(), 8,
                                  IntegerType::SignednessSemantics::Unsigned);
  if (auto dtype =
          dyn_cast_if_present<KGEN::DTypeConstantAttr>(adaptor.getDType()))
    return IntegerAttr::get(ui8Type, dtype.getDType().getValue());

  return {};
}

//===----------------------------------------------------------------------===//
// DTypeFromUI8
//===----------------------------------------------------------------------===//

OpFoldResult DTypeFromUI8::fold(FoldAdaptor adaptor) {
  if (auto val = dyn_cast_if_present<IntegerAttr>(adaptor.getValue()))
    return KGEN::DTypeConstantAttr::get(getContext(), KGENDType(val.getUInt()));

  return {};
}

//===----------------------------------------------------------------------===//
// VariantBitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantBitcastOp::fold(FoldAdaptor adaptor) {
  if (auto ptr = dyn_cast_or_null<PointerAttr>(adaptor.getVariant()))
    return PointerAttr::get(ptr.getAddr(), getType());
  return {};
}

ErrorTreeOrSuccess VariantBitcastOp::interpret(ArrayRef<Attribute> operands,
                                               InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
VariantBitcastOp::parametric_interpret(ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state) {
  if (auto ptr = dyn_cast_or_null<PointerAttr>(operands[0])) {
    return state.mapResults(PointerAttr::get(
        ptr.getAddr(), cast<PointerType>(state.getReboundType(getType()))));
  }
  return ErrorTree(getLoc(), "non-const input");
}

//===----------------------------------------------------------------------===//
// VariantDiscrGEPOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess VariantDiscrGEPOp::compile(Payload &payload,
                                          TargetInfoAttr target) {
  if (!target)
    return Error("requires a target model");

  auto variantType = getVariant().getType().getElementAs<VariantType>();
  std::optional<int64_t> size = variantType.getContentSize(target);
  if (!size)
    return Error("failed to compute size");
  payload.offset = *size;
  return success();
}

ErrorOrSuccess
VariantDiscrGEPOp::parametric_compile(Payload &payload, TargetInfoAttr target,
                                      ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  if (!target)
    return Error("requires a target model");

  auto variantType =
      cast<PointerType>(state.getReboundTypeAlways(getVariant().getType()))
          .getElementAs<VariantType>();
  std::optional<int64_t> size = variantType.getContentSize(target);
  if (!size)
    return Error("failed to compute size");
  payload.offset = *size;
  return success();
}

ErrorTreeOrSuccess VariantDiscrGEPOp::interpret(ArrayRef<Attribute> operands,
                                                const Payload &payload,
                                                InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  if (!ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  return state.mapResults(
      PointerAttr::get(ptr.getAddr() + payload.offset, getType()));
}

ErrorTreeOrSuccess
VariantDiscrGEPOp::parametric_interpret(ArrayRef<Attribute> operands,
                                        const Payload &payload,
                                        ParametricInterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  if (!ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  return state.mapResults(PointerAttr::get(ptr.getAddr() + payload.offset,
                                           state.getReboundType(getType())));
}

//===----------------------------------------------------------------------===//
// GlobalAllocOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess GlobalAllocOp::compile(Payload &payload, TargetInfoAttr target) {
  if (!target)
    return Error("global alloc requires a target");

  auto countAttr = dyn_cast<IntegerAttr>(getCount());
  if (!countAttr)
    return Error("count is not concrete");

  Type type = getType().getElementType();
  payload.size =
      countAttr.getInt() * *DataLayoutInterface::getTypeAllocSize(target, type);

  if (auto alignAttr = dyn_cast_or_null<IntegerAttr>(getAlignmentAttr()))
    payload.align = alignAttr.getInt();
  else
    payload.align = *DataLayoutInterface::getTypeABIAlign(target, type);

  payload.addressSpace = getType().getAddrSpaceOrZero();
  return success();
}

ErrorOrSuccess
GlobalAllocOp::parametric_compile(Payload &payload, TargetInfoAttr target,
                                  ArrayRef<Attribute> operands,
                                  ParametricInterpreterState &state) {
  if (!target)
    return Error("global alloc requires a target");

  auto countAttr = dyn_cast<IntegerAttr>(state.getReboundAttribute(getCount()));
  if (!countAttr)
    return Error("count is not concrete");

  auto resultType = cast<PointerType>(state.getReboundType(getType()));

  Type type = resultType.getElementType();
  payload.size =
      countAttr.getInt() * *DataLayoutInterface::getTypeAllocSize(target, type);

  if (auto alignAttr = dyn_cast_or_null<IntegerAttr>(
          state.getReboundAttribute(getAlignmentAttr())))
    payload.align = alignAttr.getInt();
  else
    payload.align = *DataLayoutInterface::getTypeABIAlign(target, type);

  payload.addressSpace = resultType.getAddrSpaceOrZero();
  return success();
}

ErrorTreeOrSuccess GlobalAllocOp::interpret(ArrayRef<Attribute> operands,
                                            const Payload &payload,
                                            InterpreterState &state) {
  ErrorOr<int64_t> addr = state.allocatePersistentMemory(
      payload.size, payload.align, payload.addressSpace);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());

  if (auto init = getInitializer()) {
    ErrorOrSuccess result = state.writeAttributeToMemory(*addr, *init);
    if (result.isError())
      return ErrorTree(getLoc(), result.takeError());
  }

  return state.mapResults(PointerAttr::get(addr.takeValue(), getType()));
}

ErrorTreeOrSuccess
GlobalAllocOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    const Payload &payload,
                                    ParametricInterpreterState &state) {
  ErrorOr<int64_t> addr = state.allocatePersistentMemory(
      payload.size, payload.align, payload.addressSpace);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());

  if (auto init = getInitializer()) {
    ErrorOrSuccess result =
        state.writeAttributeToMemory(*addr, state.getReboundAttribute(*init));
    if (result.isError())
      return ErrorTree(getLoc(), result.takeError());
  }

  return state.mapResults(
      PointerAttr::get(addr.takeValue(), state.getReboundType(getType())));
}

//===----------------------------------------------------------------------===//
// ExternalCallOp
//===----------------------------------------------------------------------===//

static ErrorTreeOrSuccess interpretMalloc(ExternalCallOp op,
                                          ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  if (operands.size() != 1 || op->getNumResults() != 1) {
    return ErrorTree(op.getLoc(), "unable to interpret call to 'malloc', "
                                  "expected 1 operand and 1 result");
  }
  ErrorOr<int64_t> sizeOr =
      POP::getScalarIndexValue(dyn_cast_or_null<TypedAttr>(operands.front()));
  if (sizeOr.isError()) {
    return ErrorTree(op->getLoc(), "Unable to interpret call to 'malloc', "
                                   "could not interpret size.");
  }
  return interpretAllocation(sizeOr.get(), /*align=*/0, op.getLoc(),
                             op->getResultTypes()[0], state);
}

static ErrorTreeOrSuccess interpretFree(ExternalCallOp op,
                                        ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  if (operands.size() != 1 || op.getNumResults() != 0) {
    return ErrorTree(
        op.getLoc(),
        "unable to interpret call to 'free', expected 1 operand and 0 results");
  }

  auto ptr = cast<PointerAttr>(operands.front());
  if (ErrorOrSuccess err = state.freeHeapMemory(ptr.getAddr()); err.isError())
    return ErrorTree(op.getLoc(), err.takeError());
  return success();
}

static ErrorTreeOrSuccess interpreterWrite(ExternalCallOp op,
                                           ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  if (!(operands.size() == 3 && op.getNumResults() == 1))
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected 3 operands and 1 results");
  Type resultType = op.getResultTypes().front();
  if (!resultType.isIntOrIndex() && !isa<SIMDType>(resultType))
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected integer result type");
  ErrorOr<int64_t> fdOr =
      POP::getScalarIndexValue(dyn_cast_or_null<TypedAttr>(operands[0]));
  if (fdOr.isError())
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected integer typed first operand");

  PointerAttr buffer = cast<PointerAttr>(operands[1]);
  if (!buffer)
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected pointer typed second operand");

  ErrorOr<int64_t> nbytesOr =
      POP::getScalarIndexValue(dyn_cast_or_null<TypedAttr>(operands[2]));
  if (nbytesOr.isError())
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected integer typed third operand");
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  ErrorOr<const void *> mem =
      state.getReadableMemory(buffer.getAddr(), ptrSize);
  if (mem)
    return ErrorTree(op.getLoc(), mem.takeError());
  int size = static_cast<int>(nbytesOr.get());
  int numWritten = write(fdOr.get(), (const void *)*mem, size);
  if (auto simdType = dyn_cast<SIMDType>(resultType)) {
    auto simdAttr = SIMDAttr::get(numWritten, simdType);
    return state.mapResults(simdAttr);
  } else {
    return state.mapResults(IntegerAttr::get(resultType, numWritten));
  }
  return success();
}

static ErrorTreeOrSuccess interpretGetStackTrace(ExternalCallOp op,
                                                 ArrayRef<Attribute> operands,
                                                 InterpreterState &state) {
  // TODO: Support printing stack trace in interpreter
  return state.mapResults(IntegerAttr::get(op.getResultTypes().front(), 0));
}

/// FIXME(#26342): We shouldn't implement interpreter support for external_call,
/// this bakes assumptions about the functions. This is a temporary workaround
/// because of the fact that the gpu path does not use the dedicated pop memory
/// operations.
ErrorTreeOrSuccess ExternalCallOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  // external_call can take things through a !kgen.struct.  Expand that out
  // before we try to interpret it.
  SmallVector<Attribute> expandedOperands;
  expandedOperands.reserve(operands.size());
  for (auto attr : operands) {
    auto pack = dyn_cast<StructAttr>(attr);
    if (pack && pack.getType().getIsParamPack()) {
      expandedOperands.append(pack.getValues().begin(), pack.getValues().end());
    } else {
      expandedOperands.push_back(attr);
    }
  }

  StringRef callee = getCallee();
  if (callee == "malloc")
    return interpretMalloc(*this, expandedOperands, state);
  if (callee == "free")
    return interpretFree(*this, expandedOperands, state);
  if (callee == "write")
    return interpreterWrite(*this, expandedOperands, state);
  if (callee == "KGEN_CompilerRT_GetStackTrace")
    return interpretGetStackTrace(*this, expandedOperands, state);

  return ErrorTree(
      getLoc(),
      Twine("unable to interpret call to unknown external function: " + callee)
          .str());
}

ErrorTreeOrSuccess
ExternalCallOp::parametric_interpret(ArrayRef<Attribute> operands,
                                     ParametricInterpreterState &state) {
  // external_call can take things through a !kgen.struct.  Expand that out
  // before we try to interpret it.
  SmallVector<Attribute> expandedOperands;
  expandedOperands.reserve(operands.size());
  for (auto attr : operands) {
    auto pack = dyn_cast<StructAttr>(attr);
    if (pack && pack.getType().getIsParamPack()) {
      expandedOperands.append(pack.getValues().begin(), pack.getValues().end());
    } else {
      expandedOperands.push_back(attr);
    }
  }

  StringRef callee = cast<StringAttr>(state.getReboundAttribute(getFunc()));

  if (callee == "malloc")
    return interpretMalloc(*this, expandedOperands, state);
  if (callee == "free")
    return interpretFree(*this, expandedOperands, state);
  if (callee == "write")
    return interpreterWrite(*this, expandedOperands, state);
  if (callee == "KGEN_CompilerRT_GetStackTrace")
    return interpretGetStackTrace(*this, expandedOperands, state);

  return ErrorTree(
      getLoc(),
      Twine("unable to interpret call to unknown external function: " + callee)
          .str());
}

//===----------------------------------------------------------------------===//
// UnionBitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult UnionBitcastOp::fold(FoldAdaptor adaptor) {
  auto ptr = dyn_cast_or_null<PointerAttr>(adaptor.getValue());
  if (!ptr)
    return {};
  return PointerAttr::get(ptr.getAddr(), getType());
}

ErrorTreeOrSuccess UnionBitcastOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
UnionBitcastOp::parametric_interpret(ArrayRef<Attribute> operands,
                                     ParametricInterpreterState &state) {
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]);
  if (!ptr)
    return ErrorTree(getLoc(), "non-const input");

  return state.mapResults(
      PointerAttr::get(ptr.getAddr(), state.getReboundType(getType())));
}

//===----------------------------------------------------------------------===//
// UnionWrapOp
//===----------------------------------------------------------------------===//

OpFoldResult UnionWrapOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<TypedAttr>(adaptor.getValue()))
    return UnionAttr::get(attr, getType());

  // Fold `wrap(unwrap(x)) -> x` if the types are the same.
  if (auto unwrap = getValue().getDefiningOp<UnionUnwrapOp>();
      unwrap && unwrap.getValue().getType() == getType())
    return unwrap.getValue();

  return {};
}

ErrorTreeOrSuccess UnionWrapOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
UnionWrapOp::parametric_interpret(ArrayRef<Attribute> operands,
                                  ParametricInterpreterState &state) {
  if (auto attr = dyn_cast_or_null<TypedAttr>(operands[0])) {
    return state.mapResults(UnionAttr::get(attr, getType()));
  }
  return ErrorTree(getLoc(), "non-const input");
}

//===----------------------------------------------------------------------===//
// UnionUnwrapOp
//===----------------------------------------------------------------------===//

OpFoldResult UnionUnwrapOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<UnionAttr>(adaptor.getValue()))
    if (attr.getValue().getType() == getType())
      return attr.getValue();

  // Fold `unwrap(wrap(x)) -> x`.
  if (auto wrap = getValue().getDefiningOp<UnionWrapOp>();
      wrap && wrap.getValue().getType() == getType())
    return wrap.getValue();

  return {};
}

ErrorTreeOrSuccess UnionUnwrapOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
UnionUnwrapOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  if (auto attr = dyn_cast_or_null<UnionAttr>(operands[0])) {
    if (attr.getValue().getType() == state.getReboundType(getType())) {
      return state.mapResults(attr.getValue());
    }
  }
  return ErrorTree(getLoc(), "non-const input");
}

//===----------------------------------------------------------------------===//
// SIMDReduceOrOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDReduceOrOp::fold(FoldAdaptor adaptor) {
  return POP::foldSIMDReduceOr(getVector(), adaptor.getVector(),
                               getVector().getType());
}

//===----------------------------------------------------------------------===//
// SIMDReduceAndOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDReduceAndOp::fold(FoldAdaptor adaptor) {
  return POP::foldSIMDReduceAnd(getVector(), adaptor.getVector(),
                                getVector().getType());
}
