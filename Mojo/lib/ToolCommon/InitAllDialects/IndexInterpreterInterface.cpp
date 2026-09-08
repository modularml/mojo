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

#include "Mojo/ToolCommon/InitAllDialects/IndexInterpreterInterface.h"

using namespace M;
using namespace KGEN;

/// Compare two integers according to the comparison predicate.
static bool compareIndices(const APInt &lhs, const APInt &rhs,
                           mlir::index::IndexCmpPredicate pred) {
  switch (pred) {
  case mlir::index::IndexCmpPredicate::EQ:
    return lhs.eq(rhs);
  case mlir::index::IndexCmpPredicate::NE:
    return lhs.ne(rhs);
  case mlir::index::IndexCmpPredicate::SGE:
    return lhs.sge(rhs);
  case mlir::index::IndexCmpPredicate::SGT:
    return lhs.sgt(rhs);
  case mlir::index::IndexCmpPredicate::SLE:
    return lhs.sle(rhs);
  case mlir::index::IndexCmpPredicate::SLT:
    return lhs.slt(rhs);
  case mlir::index::IndexCmpPredicate::UGE:
    return lhs.uge(rhs);
  case mlir::index::IndexCmpPredicate::UGT:
    return lhs.ugt(rhs);
  case mlir::index::IndexCmpPredicate::ULE:
    return lhs.ule(rhs);
  case mlir::index::IndexCmpPredicate::ULT:
    return lhs.ult(rhs);
  }
  llvm_unreachable("unhandled IndexCmpPredicate predicate");
}

// Interpret result of the binary operation
template <bool isSigned, bool withOverflowCheck, typename BinaryFn>
static ErrorTreeOrSuccess
interpretBinaryOp(Location loc, MLIRContext *ctx, ArrayRef<Attribute> operands,
                  BinaryFn binaryFn, InterpreterState &state,
                  StringRef opName) {
  IntegerAttr lhsInt = dyn_cast_if_present<mlir::IntegerAttr>(operands[0]);
  IntegerAttr rhsInt = dyn_cast_if_present<mlir::IntegerAttr>(operands[1]);
  uint64_t targetBitwidth = state.getTarget().resolveIndexBitWidth();
  APInt lhs;
  APInt rhs;
  auto checkIfValueCanFit = [&](IntegerAttr valueAttr) -> ErrorTreeOrSuccess {
    APInt value = valueAttr.getValue();
    unsigned bitWidth = value.getBitWidth();
    if (bitWidth <= targetBitwidth)
      return success();
    bool isSignedValue = isSigned || value.isNegative();
    APInt maxValue;
    APInt minValue;
    if (isSignedValue) {
      maxValue = APInt::getSignedMaxValue(targetBitwidth).sext(bitWidth);
      minValue = APInt::getSignedMinValue(targetBitwidth).sext(bitWidth);
    } else {
      maxValue = APInt::getMaxValue(targetBitwidth).zext(bitWidth);
      minValue = APInt::getMinValue(targetBitwidth).zext(bitWidth);
    }

    if ((!isSignedValue && value.ugt(maxValue)) ||
        (isSignedValue && (value.sgt(maxValue) || value.slt(minValue)))) {
      std::string str;
      llvm::raw_string_ostream ss(str);
      ss << "value '" << value << "' of the operation `index." << opName
         << "` is too large for " << targetBitwidth << "-bit index";
      return ErrorTree(loc, str);
    }
    return success();
  };

  if (ErrorTreeOrSuccess error = checkIfValueCanFit(lhsInt))
    return error;
  if (ErrorTreeOrSuccess error = checkIfValueCanFit(rhsInt))
    return error;

  lhs = lhsInt.getValue().trunc(targetBitwidth);
  rhs = rhsInt.getValue().trunc(targetBitwidth);

  APInt result;
  if constexpr (withOverflowCheck) {
    bool overflow = false;
    result = binaryFn(lhs, rhs, overflow);
    if (overflow)
      return ErrorTree(loc, "`index." + opName + "` failed due to overflow");
  } else {
    result = binaryFn(lhs, rhs);
  }

  if constexpr (isSigned)
    result = result.sextOrTrunc(IndexType::kInternalStorageBitWidth);
  else
    result = result.zextOrTrunc(IndexType::kInternalStorageBitWidth);

  return state.mapResults(IntegerAttr::get(IndexType::get(ctx), result));
}

template <>
ErrorTreeOrSuccess
CmpOpInterpretInterface::interpret(mlir::index::CmpOp cmpOp,
                                   ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  assert(operands.size() == 2 && "cmp expected two operands");
  IntegerAttr lhs = dyn_cast_if_present<mlir::IntegerAttr>(operands[0]);
  if (!lhs)
    return ErrorTree(cmpOp.getLoc(), "non-constant lhs input");
  IntegerAttr rhs = dyn_cast_if_present<mlir::IntegerAttr>(operands[1]);
  if (!rhs)
    return ErrorTree(cmpOp.getLoc(), "non-constant rhs input");

  uint64_t targetBitwidth = state.getTarget().resolveIndexBitWidth();
  auto result =
      BoolAttr::get(cmpOp.getContext(),
                    compareIndices(lhs.getValue().truncSSat(targetBitwidth),
                                   rhs.getValue().truncSSat(targetBitwidth),
                                   cmpOp.getPred()));
  return state.mapResults(result);
}

template <>
ErrorTreeOrSuccess
SubOpInterpretInterface::interpret(mlir::index::SubOp subOp,
                                   ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  assert(operands.size() == 2 && "sub expected two operands");
  return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/true>(
      subOp.getLoc(), subOp.getContext(), operands,
      [](APInt lhs, APInt rhs, bool &overflow) {
        return lhs.ssub_ov(rhs, overflow);
      },
      state, "sub");
}

template <>
ErrorTreeOrSuccess
ShlOpInterpretInterface::interpret(mlir::index::ShlOp shlOp,
                                   ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  assert(operands.size() == 2 && "shl expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      shlOp.getLoc(), shlOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs << rhs; }, state, "shl");
}

template <>
ErrorTreeOrSuccess
ShrSOpInterpretInterface::interpret(mlir::index::ShrSOp shrsOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "shrs expected two operands");
  return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/false>(
      shrsOp.getLoc(), shrsOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs.ashr(rhs); }, state, "shrs");
}

template <>
ErrorTreeOrSuccess
ShrUOpInterpretInterface::interpret(mlir::index::ShrUOp shruOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "shru expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      shruOp.getLoc(), shruOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs.lshr(rhs); }, state, "shru");
}

template <>
ErrorTreeOrSuccess
AndOpInterpretInterface::interpret(mlir::index::AndOp andOp,
                                   ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  assert(operands.size() == 2 && "and expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      andOp.getLoc(), andOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs &= rhs; }, state, "and");
}

template <>
ErrorTreeOrSuccess
CeilDivUOpInterpretInterface::interpret(mlir::index::CeilDivUOp ceilDivUOp,
                                        ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  assert(operands.size() == 2 && "ceildivu expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      ceilDivUOp.getLoc(), ceilDivUOp.getContext(), operands,
      [](APInt lhs, APInt rhs) {
        return lhs.udiv(rhs) + (lhs.urem(rhs) != 0 ? 1 : 0);
      },
      state, "ceildivu");
}

template <>
ErrorTreeOrSuccess
CeilDivSOpInterpretInterface::interpret(mlir::index::CeilDivSOp ceilDivSOp,
                                        ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  assert(operands.size() == 2 && "ceildivs expected two operands");
  return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/false>(
      ceilDivSOp.getLoc(), ceilDivSOp.getContext(), operands,
      [](APInt lhs, APInt rhs) {
        return lhs.sdiv(rhs) + (lhs.srem(rhs).sgt(0) ? 1 : 0);
      },
      state, "ceildivs");
}

template <>
ErrorTreeOrSuccess
DivUOpInterpretInterface::interpret(mlir::index::DivUOp divUOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "divu expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      divUOp.getLoc(), divUOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs.udiv(rhs); }, state, "divu");
}

template <>
ErrorTreeOrSuccess
DivSOpInterpretInterface::interpret(mlir::index::DivSOp divSOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "divs expected two operands");
  return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/false>(
      divSOp.getLoc(), divSOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs.sdiv(rhs); }, state, "divs");
}

template <>
ErrorTreeOrSuccess
MaxUOpInterpretInterface::interpret(mlir::index::MaxUOp maxUOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "maxu expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      maxUOp.getLoc(), maxUOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return llvm::APIntOps::umax(lhs, rhs); },
      state, "maxu");
}

template <>
ErrorTreeOrSuccess
MaxSOpInterpretInterface::interpret(mlir::index::MaxSOp maxSOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "maxs expected two operands");
  return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/false>(
      maxSOp.getLoc(), maxSOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return llvm::APIntOps::smax(lhs, rhs); },
      state, "maxs");
}

template <>
ErrorTreeOrSuccess
MinUOpInterpretInterface::interpret(mlir::index::MinUOp minUOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "minu expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      minUOp.getLoc(), minUOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return llvm::APIntOps::umin(lhs, rhs); },
      state, "minu");
}

template <>
ErrorTreeOrSuccess
MinSOpInterpretInterface::interpret(mlir::index::MinSOp minSOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "mins expected two operands");
  return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/false>(
      minSOp.getLoc(), minSOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return llvm::APIntOps::smin(lhs, rhs); },
      state, "mins");
}

template <>
ErrorTreeOrSuccess
MulOpInterpretInterface::interpret(mlir::index::MulOp mulOp,
                                   ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  assert(operands.size() == 2 && "mul expected two operands");
  // FIXME: Re-enable overflow check after fix of KERN-1704
  // return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/true>(
  //     mulOp.getLoc(), mulOp.getContext(), operands,
  //     [](APInt lhs, APInt rhs, bool &overflow) { return lhs.smul_ov(rhs,
  //     overflow); }, state, "mul");
  return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/false>(
      mulOp.getLoc(), mulOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs * rhs; }, state, "mul");
}

template <>
ErrorTreeOrSuccess
OrOpInterpretInterface::interpret(mlir::index::OrOp orOp,
                                  ArrayRef<Attribute> operands,
                                  InterpreterState &state) {
  assert(operands.size() == 2 && "or expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      orOp.getLoc(), orOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs |= rhs; }, state, "or");
}

template <>
ErrorTreeOrSuccess
RemUOpInterpretInterface::interpret(mlir::index::RemUOp remuOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "remu expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      remuOp.getLoc(), remuOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs.urem(rhs); }, state, "remu");
}

template <>
ErrorTreeOrSuccess
RemSOpInterpretInterface::interpret(mlir::index::RemSOp remsOp,
                                    ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  assert(operands.size() == 2 && "rems expected two operands");
  return interpretBinaryOp</*isSigned=*/true, /*withOverflowCheck=*/false>(
      remsOp.getLoc(), remsOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs.srem(rhs); }, state, "rems");
}

template <>
ErrorTreeOrSuccess
XOrOpInterpretInterface::interpret(mlir::index::XOrOp xorOp,
                                   ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  assert(operands.size() == 2 && "xor expected two operands");
  return interpretBinaryOp</*isSigned=*/false, /*withOverflowCheck=*/false>(
      xorOp.getLoc(), xorOp.getContext(), operands,
      [](APInt lhs, APInt rhs) { return lhs ^= rhs; }, state, "xor");
}
