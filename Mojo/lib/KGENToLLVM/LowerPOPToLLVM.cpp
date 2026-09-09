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

#include "Mojo/ToolCommon/KGENPasses.h"

#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "LowerPOPToLLVMExternalCalls.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/MDialect/MAttrs.h"
#include "Target/TargetLowering.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"
#include <mlir/IR/SymbolTable.h>
#include <utility>

using namespace M;
using namespace KGEN;
using namespace POP;
namespace LLVM = mlir::LLVM;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERGLOBALPOPTOLLVM
#define GEN_PASS_DEF_LOWERPOPTOLLVM
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

/// POP dialect fastmath flags match the LLVM ones.
static LLVM::FastmathFlagsAttr
convertFastmathFlags(FastmathFlags fmf, ConversionPatternRewriter &rewriter) {
  return rewriter.getAttr<LLVM::FastmathFlagsAttr>(
      static_cast<LLVM::FastmathFlags>(fmf));
}

//===----------------------------------------------------------------------===//
// ConvertPOPNeg
//===----------------------------------------------------------------------===//

/// Convert an integer pop.neg(x) -> 0 - x
/// and float pop.neg(x) -> llvm.fneg(x)
struct ConvertPOPNeg : public ConvertPOPToLLVMPattern<NegOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(NegOp op, NegOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (!dtype.isInt() && !dtype.isIndex() && !dtype.isUIndex()) {
      rewriter.replaceOpWithNewOp<LLVM::FNegOp>(op, adaptor.getOperand(),
                                                fastmathFlagsOrDefault(op));
      return success();
    }

    Type type = adaptor.getOperand().getType();
    Value zero;
    if (auto vec = dyn_cast<VectorType>(type)) {
      auto intType = dyn_cast<IntegerType>(vec.getElementType());
      if (!intType)
        return op.emitError("could not handle integer type");
      auto apZero = APInt::getZero(intType.getWidth());
      zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                      DenseIntElementsAttr::get(vec, apZero));
    } else {
      zero = LLVM::ConstantOp::create(rewriter, op.getLoc(), type, 0);
    }

    rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, zero, adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPFloor
//===----------------------------------------------------------------------===//

struct ConvertPOPFloor : public ConvertPOPToLLVMPattern<FloorOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(FloorOp op, FloorOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (!dtype.isFloat()) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    Type type = this->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, type, rewriter.getStringAttr("llvm.floor"),
        SmallVector<Value>{adaptor.getOperand()});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPFloorDiv
//===----------------------------------------------------------------------===//

struct ConvertPOPFloorDiv : public ConvertPOPToLLVMPattern<FloorDivOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(FloorDivOp op, FloorDivOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type type = this->convertType(op.getType());
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isFloat()) {
      auto fdiv = LLVM::FDivOp::create(rewriter, loc, type, adaptor.getLhs(),
                                       adaptor.getRhs());
      rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
          op, type, rewriter.getStringAttr("llvm.floor"),
          SmallVector<Value>{fdiv});
      return success();
    }

    if (dtype.isUInt()) {
      rewriter.replaceOpWithNewOp<LLVM::UDivOp>(
          op, type, SmallVector<Value>{adaptor.getLhs(), adaptor.getRhs()});
      return success();
    }

    // For signed integers, emit the LLVM IR corresponding to:
    // int floordiv(int a, int b) {
    //   int d = a / b;
    //   return d * b == a ? d : d - ((a < 0) ^ (b < 0));
    // }
    // ... which, when turned into optimized LLVM IR, is:
    //   %sdiv = sdiv i32 %a, %b
    //   %mul = mul nsw i32 %sdiv, %b
    //   %icmp = icmp eq i32 %mul, %a
    //   %xorOp = xor i32 %b, %a
    //   %ashr = ashr i32 %xorOp, 31
    //   %sel = select i1 %icmp, i32 0, i32 %ashr
    //   %ret = add i32 %sdiv, %sel
    auto sdiv = LLVM::SDivOp::create(rewriter, loc, type, adaptor.getLhs(),
                                     adaptor.getRhs());
    auto mul = LLVM::MulOp::create(rewriter, loc, type, sdiv, adaptor.getRhs());
    auto icmp = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                     mul, adaptor.getLhs());
    auto xorOp = LLVM::XOrOp::create(rewriter, loc, type, adaptor.getLhs(),
                                     adaptor.getRhs());

    Value signBitShift, zero;
    if (auto vecType = dyn_cast<VectorType>(type)) {
      uint64_t size = *op.getType().getResolvedSize();
      Type eltTy = vecType.getElementType();
      unsigned eltNumBits = eltTy.getIntOrFloatBitWidth();
      SmallVector<APInt> zeroValues{size, APInt::getZero(eltNumBits)};
      SmallVector<APInt> signBitShiftValues{size,
                                            APInt(eltNumBits, eltNumBits - 1)};
      auto zeroAttrs =
          cast<TypedAttr>(IntArrayElementsAttr::get(vecType, zeroValues));
      auto signBitShiftAttrs = cast<TypedAttr>(
          IntArrayElementsAttr::get(vecType, signBitShiftValues));
      zero = LLVM::ConstantOp::create(rewriter, loc, zeroAttrs);
      signBitShift = LLVM::ConstantOp::create(rewriter, loc, signBitShiftAttrs);
    } else {
      assert(type.isIntOrFloat() && "Unexpected type in floordiv");
      zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                      rewriter.getIntegerAttr(type, 0));
      signBitShift = LLVM::ConstantOp::create(
          rewriter, op.getLoc(),
          rewriter.getIntegerAttr(type, type.getIntOrFloatBitWidth() - 1));
    }

    auto ashr = LLVM::AShrOp::create(rewriter, loc, type, xorOp, signBitShift);

    auto sel = LLVM::SelectOp::create(rewriter, loc, type, icmp, zero, ashr);

    rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, type,
                                             SmallVector<Value>{sdiv, sel});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCeil
//===----------------------------------------------------------------------===//

struct ConvertPOPCeil : public ConvertPOPToLLVMPattern<CeilOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CeilOp op, CeilOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (!dtype.isFloat()) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    Type type = this->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, type, rewriter.getStringAttr("llvm.ceil"),
        SmallVector<Value>{adaptor.getOperand()});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPTrunc
//===----------------------------------------------------------------------===//

struct ConvertPOPTrunc : public ConvertPOPToLLVMPattern<TruncOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(TruncOp op, TruncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (!dtype.isFloat()) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    Type type = this->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, type, rewriter.getStringAttr("llvm.trunc"),
        SmallVector<Value>{adaptor.getOperand()});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPRound
//===----------------------------------------------------------------------===//

struct ConvertPOPRound : public ConvertPOPToLLVMPattern<RoundOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(RoundOp op, RoundOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (!dtype.isFloat()) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    Type type = this->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, type, rewriter.getStringAttr("llvm.roundeven"),
        SmallVector<Value>{adaptor.getOperand()});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAbs
//===----------------------------------------------------------------------===//

struct ConvertPOPAbs : public ConvertPOPToLLVMPattern<AbsOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(AbsOp op, AbsOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isUInt() || dtype.isBool()) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    Type type = this->convertType(op.getType());
    if (!dtype.isFloat()) {
      // As per LLVM, if is_int_min_poison == 0, then if the operand is INT_MIN
      // the result is also INT_MIN.
      auto isIntMinPoison = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), rewriter.getIntegerType(1), false);
      rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
          op, type, rewriter.getStringAttr("llvm.abs"),
          SmallVector<Value>{adaptor.getOperand(), isIntMinPoison});
      return success();
    }

    // Else, just use the regular LLVM intrinsic
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, type, rewriter.getStringAttr("llvm.fabs"),
        SmallVector<Value>{adaptor.getOperand()});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPShr
//===----------------------------------------------------------------------===//

/// Lower to `llvm.ashr` if the result dtype is signed and `llvm.lshr`
/// otherwise.
struct ConvertPOPShr : public ConvertPOPToLLVMPattern<ShrOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, ShrOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isSInt() || dtype.isIndex())
      rewriter.replaceOpWithNewOp<LLVM::AShrOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<LLVM::LShrOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

/// Convert integer pop.fma(x, y, z) -> x * y + z
/// and float pop.fma(x, y, a) -> llvm.intr.fma(x, y, z)
struct ConvertPOPFMA : public ConvertPOPToLLVMPattern<FMAOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(FMAOp op, FMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isInt() || dtype.isIndex() || dtype.isUIndex()) {
      auto lhs = LLVM::MulOp::create(rewriter, op.getLoc(), adaptor.getA(),
                                     adaptor.getB());
      rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, lhs, adaptor.getC());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::FMAOp>(
          op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
          convertFastmathFlags(op.getFastmathFlags(), rewriter));
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCmp
//===----------------------------------------------------------------------===//

class ConvertPOPCmp : public ConvertPOPToLLVMPattern<CmpOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, CmpOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getLhs().getType().getResolvedDType();
    if (dtype.isBool() || dtype.isInt() || dtype.isIndex() ||
        dtype.isUIndex() || dtype.isAddress()) {
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
          op, getICmpPredicate(op.getPred(), dtype.isSInt()), adaptor.getLhs(),
          adaptor.getRhs());
    } else {
      assert(dtype.isFloat());
      Type i1Type = rewriter.getI1Type();
      if (auto simd = dyn_cast<SIMDType>(op.getLhs().getType())) {
        auto size = *simd.getResolvedSize();
        // Vectors of size 1 should remain scalars
        if (size != 1)
          i1Type = VectorType::get(size, i1Type);
      }
      rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
          op, i1Type, getFCmpPredicate(op.getPred()), adaptor.getLhs(),
          adaptor.getRhs(), fastmathFlagsOrDefault(op));
    }
    return success();
  }

private:
  /// Convert the integer comparison predicate to the LLVM predicate based on
  /// the signedness.
  static LLVM::ICmpPredicate getICmpPredicate(CmpPredicate pred,
                                              bool isSigned) {
    switch (pred) {
    case CmpPredicate::EQ:
      return LLVM::ICmpPredicate::eq;
    case CmpPredicate::NE:
      return LLVM::ICmpPredicate::ne;
    case CmpPredicate::LT:
      return isSigned ? LLVM::ICmpPredicate::slt : LLVM::ICmpPredicate::ult;
    case CmpPredicate::GT:
      return isSigned ? LLVM::ICmpPredicate::sgt : LLVM::ICmpPredicate::ugt;
    case CmpPredicate::LE:
      return isSigned ? LLVM::ICmpPredicate::sle : LLVM::ICmpPredicate::ule;
    case CmpPredicate::GE:
      return isSigned ? LLVM::ICmpPredicate::sge : LLVM::ICmpPredicate::uge;
    }
    llvm_unreachable("unknown predicate");
  }

  /// Convert the float comparison predicate to the LLVM predicate based on the
  /// signedness.
  static LLVM::FCmpPredicate getFCmpPredicate(CmpPredicate pred) {
    switch (pred) {
    case CmpPredicate::EQ:
      return LLVM::FCmpPredicate::oeq;
    case CmpPredicate::NE:
      return LLVM::FCmpPredicate::one;
    case CmpPredicate::LT:
      return LLVM::FCmpPredicate::olt;
    case CmpPredicate::GT:
      return LLVM::FCmpPredicate::ogt;
    case CmpPredicate::LE:
      return LLVM::FCmpPredicate::ole;
    case CmpPredicate::GE:
      return LLVM::FCmpPredicate::oge;
    }
    llvm_unreachable("unknown predicate");
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCast
//===----------------------------------------------------------------------===//

struct ConvertPOPCast : public ConvertPOPToLLVMPattern<CastOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastOp op, CastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType inDType = *op.getInput().getType().getResolvedDType();
    KGENDType outDType = *op.getOutput().getType().getResolvedDType();

    if ((inDType.isFloat() && inDType.getWidthInBits() <= 8) ||
        (outDType.isFloat() && outDType.getWidthInBits() <= 8)) {
      return op.emitError() << "illegal conversion";
    }

    int64_t inByteCount = getDTypeSizeInBytes(inDType);
    int64_t outByteCount = getDTypeSizeInBytes(outDType);

    // Select the element-wise cast to perform. LLVM integer types are signless,
    // but the signedness semantics of the operation's input and output types
    // affect which casts are selected. `bool` is `i1`.
    StringRef opName;
    if (inDType.isBool() || inDType.isInt() || inDType.isIndex() ||
        inDType.isUIndex()) {
      if (outDType.isBool() || outDType.isInt() || outDType.isIndex() ||
          outDType.isUIndex()) {
        // A bool should still become a cast as the bool is only 1 bit but
        // appears as 1 byte here.
        if (outByteCount > inByteCount || inDType.isBool()) {
          // Sign or zero extend.
          opName = inDType.isSInt() ? LLVM::SExtOp::getOperationName()
                                    : LLVM::ZExtOp::getOperationName();
        } else if (outByteCount < inByteCount || outDType.isBool()) {
          // Truncate.
          opName = LLVM::TruncOp::getOperationName();
        }
      } else {
        // Cast from an integer to a float.
        opName = inDType.isSInt() ? LLVM::SIToFPOp::getOperationName()
                                  : LLVM::UIToFPOp::getOperationName();
      }
    } else if (outDType.isBool() || outDType.isInt() || outDType.isIndex() ||
               outDType.isUIndex()) {
      // Cast from a float to an integer.
      opName = outDType.isSInt() ? LLVM::FPToSIOp::getOperationName()
                                 : LLVM::FPToUIOp::getOperationName();
    } else if (outByteCount > inByteCount) {
      // Extend
      opName = LLVM::FPExtOp::getOperationName();
    } else if (outByteCount < inByteCount) {
      // Truncate.
      if (isFPTyLoweredAsInt(
              getEquivalentFloatType(op.getContext(), outDType))) {
        // Can't truncate a floating point number to a floating point number
        // that's represented as an int (e.g., fp8 types).
        return emitError(op.getLoc(),
                         Twine("casts between " + inDType.getAsString() +
                               " and " + outDType.getAsString() +
                               " unsupported"));
      }

      opName = LLVM::FPTruncOp::getOperationName();
    } else if (outDType != inDType) {
      // FIXME: Unclear how to cast between `bf16` and `f16`.
      return rewriter.notifyMatchFailure(
          op, "casts between 'bf16' and 'f16' unsupported");
    }

    // If no cast was selected, this is a no-op conversion between equivalent
    // types.
    if (opName.empty()) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Create the cast.
    OperationState state(op.getLoc(), opName);
    state.addOperands(adaptor.getInput());
    state.addTypes(convertType(op.getOutput().getType()));
    Operation *cast = rewriter.create(state);
    rewriter.replaceOp(op, cast->getResults());
    return success();
  }

private:
  int64_t getDTypeSizeInBytes(KGENDType dtype) const {
    if (dtype.isIndex() || dtype.isUIndex())
      return getTypeConverter()->getIndexTypeBitwidth() / CHAR_BIT;
    return dtype.getSizeInBytes();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDSelect
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDSelect : public ConvertPOPToLLVMPattern<SIMDSelectOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDSelectOp op, SIMDSelectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue(), fastmathFlagsOrDefault(op));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDSplat
//===----------------------------------------------------------------------===//

/// Convert a SIMD splat to an `insertelement` into an `undef` and then a
/// zero-initialized `shufflevector`.
struct ConvertPOPSIMDSplat : public ConvertPOPToLLVMPattern<SIMDSplatOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDSplatOp op, SIMDSplatOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the vector is size 1, skip the shuffle.
    if (op.getType().isScalar()) {
      rewriter.replaceOp(op, adaptor.getScalar());
      return success();
    }

    SIMDType simdType = op.getType();
    int64_t size = *simdType.getResolvedSize();
    Value undef =
        LLVM::UndefOp::create(rewriter, op.getLoc(), convertType(simdType));
    Value zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                          rewriter.getI32IntegerAttr(0));
    Value vector = LLVM::InsertElementOp::create(rewriter, op.getLoc(), undef,
                                                 adaptor.getScalar(), zero);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
        op, vector, undef, /*mask=*/SmallVector<int32_t>(size, 0));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDInsertElement
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDInsertElement
    : public ConvertPOPToLLVMPattern<SIMDInsertElementOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDInsertElementOp op, SIMDInsertElementOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVector().getType().isScalar()) {
      // If the vector is size 1, return the value as is - it's a scalar.
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
        op, convertType(op.getType()), adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDShuffle
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDShuffle : public ConvertPOPToLLVMPattern<SIMDShuffleOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDShuffleOp op, SIMDShuffleOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mask = cast<POP::ArrayAttr>(adaptor.getMask());
    SmallVector<int32_t> maskValues;
    for (TypedAttr maskElement : mask.getValues())
      maskValues.push_back(cast<IntegerAttr>(maskElement).getInt());

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto inputSize = *op.getLhs().getType().getResolvedSize();
    if (inputSize != 1) {
      // Both LHS and RHS are vectors - generate LLVM ShuffleVector
      rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
          op, lhs, rhs, rewriter.getDenseI32ArrayAttr(maskValues));

      return success();
    }
    // Special handling for inputs consisting of just 1 element - instead of
    // converting them to vectors and generating shufflevector for them, we will
    // instead generate a sequence of insertelement operations.  Since there are
    // just two elements to pick from, mask should only contain 0s and 1s. If it
    // contains a different value, the behavior is undefined - we will simply
    // treat such a case as value 1.
    KGENDType dtype = *op.getType().getResolvedDType();
    auto llvmVecType = VectorType::get(
        mask.getValues().size(),
        *getMLIRTypeForDType(op.getType().getContext(), dtype,
                             getTypeConverter()->getIndexTypeBitwidth()));
    Value result = LLVM::UndefOp::create(rewriter, op.getLoc(), llvmVecType);
    int idx = 0;
    for (int32_t maskElement : maskValues) {
      Value pos = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                           rewriter.getI32IntegerAttr(idx));
      result = LLVM ::InsertElementOp::create(
          rewriter, op.getLoc(), result, maskElement == 0 ? lhs : rhs, pos);
      ++idx;
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDExtractElement
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDExtractElement
    : public ConvertPOPToLLVMPattern<SIMDExtractElementOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDExtractElementOp op, SIMDExtractElementOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Special handling for scalars
    if (op.getVector().getType().isScalar()) {
      rewriter.replaceOp(op, adaptor.getVector());
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(
        op, convertType(op.getType()), adaptor.getVector(),
        adaptor.getPosition());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPOffset
//===----------------------------------------------------------------------===//

struct ConvertPOPOffset : public ConvertPOPToLLVMPattern<OffsetOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(OffsetOp op, OffsetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType =
        typeConverter->convertType(op.getPtr().getType().getElementType());

    // Set the address space if specified.
    unsigned addrSpace = op.getPtr().getType().getAddrSpaceOrZero();

    // Coerce the index to the same type as the pointer which is required by the
    // address space.
    Type intPtrType = getIntPtrType(addrSpace);
    size_t intPtrTypeSize = intPtrType.getIntOrFloatBitWidth();
    size_t indexTypeSize = adaptor.getIndex().getType().getIntOrFloatBitWidth();

    Value offset;
    if (intPtrTypeSize == indexTypeSize) {
      offset = adaptor.getIndex();
    } else if (intPtrTypeSize < indexTypeSize) {
      offset = rewriter.createOrFold<LLVM::TruncOp>(op.getLoc(), intPtrType,
                                                    adaptor.getIndex());
    } else {
      offset = rewriter.createOrFold<LLVM::SExtOp>(op.getLoc(), intPtrType,
                                                   adaptor.getIndex());
    }
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, /*resultType=*/adaptor.getPtr().getType(),
        /*elementType=*/elementType,
        /*basePtr=*/adaptor.getPtr(),
        /*indices=*/ValueRange{offset},
        /*noWrapFlags=*/LLVM::GEPNoWrapFlags::inbounds);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSelect
//===----------------------------------------------------------------------===//

struct ConvertPOPSelect : public ConvertPOPToLLVMPattern<SelectOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SelectOp op, SelectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue(), fastmathFlagsOrDefault(op));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStackAllocation
//===----------------------------------------------------------------------===//

/// A `pop.stack_allocation` is lowered by converting it to an `llvm.alloca`
/// with lifetime markers and hoisting it to the top of the enclosing
/// function.
class ConvertPOPStackAllocation
    : public ConvertPOPToLLVMPattern<StackAllocationOp> {
public:
  explicit ConvertPOPStackAllocation(mlir::LLVMTypeConverter &typeConverter,
                                     TargetInfoAttr target)
      : ConvertPOPToLLVMPattern(typeConverter), target(target) {}

  LogicalResult
  matchAndRewrite(StackAllocationOp op, StackAllocationOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// The target info.
  TargetInfoAttr target;
  mlir::DominanceInfo domInfo;

  static unsigned resolveAlignment(std::optional<TypedAttr> alignment) {
    if (!alignment)
      return 0;
    return cast<IntegerAttr>(*alignment).getInt();
  }
};

/// Generate the LLVM IR to materialize an alloca with the given LLVM type and
/// count. The alloca is created at the top of the given block, and lifetime
/// markers are inserted at the end of the given operation's block.
static Value materializeLLVMAlloca(OpBuilder &b, TargetInfoAttr target,
                                   Type elementType, int64_t count,
                                   Operation *op, int64_t typeAllocSize,
                                   int64_t align) {
  unsigned addressSpace = 0;
  auto alloca = dyn_cast<StackAllocationOp>(op);
  if (alloca)
    addressSpace = alloca.getType().getAddrSpaceOrZero();

  bool needAddrSpaceCast = false;
  if (addressSpace == 0) {
    addressSpace = target.getDataLayout().getAllocaAddrSpace();
    needAddrSpaceCast = addressSpace != 0;
  }

  Value countVal =
      LLVM::ConstantOp::create(b, op->getLoc(), b.getI64IntegerAttr(count));
  Value ptr = LLVM::AllocaOp::create(
      b, op->getLoc(), LLVM::LLVMPointerType::get(b.getContext(), addressSpace),
      elementType, countVal, align);

  if (alloca && alloca.getMarkedLifetimes()) {
    // If this alloca has marked lifetimes, it always begins as dead.
    LLVM::LifetimeEndOp::create(b, op->getLoc(), ptr);
  } else {
    // Insert lifetime markers starting from the op to the end of its block.
    b.setInsertionPoint(op);
    auto start = LLVM::LifetimeStartOp::create(b, op->getLoc(), ptr);
    b.setInsertionPoint(op->getBlock(), --op->getBlock()->end());
    LLVM::LifetimeEndOp::create(b, op->getLoc(), ptr);
    b.setInsertionPointAfter(start);
  }

  if (needAddrSpaceCast) {
    ptr = LLVM::AddrSpaceCastOp::create(
        b, op->getLoc(),
        LLVM::LLVMPointerType::get(b.getContext(), /*addressSpace=*/0), ptr);
  }

  return ptr;
}

LogicalResult ConvertPOPStackAllocation::matchAndRewrite(
    StackAllocationOp op, StackAllocationOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  PointerType ptrType = cast<PointerType>(op.getType());
  Type elementType = convertType(ptrType.getElementType());
  if (!elementType)
    return op.emitError("could not lower pointer element type");

  // Compute the bytecount of the allocated buffer.
  std::optional<int64_t> typeAllocSize =
      DataLayoutInterface::getTypeAllocSize(target, ptrType.getElementType());
  if (!typeAllocSize)
    return op.emitError("could not get size of variadic element");

  // Compute alignment from the KGEN element type. This correctly handles
  // structs containing aligned fields (e.g., a struct with an @align(64) field
  // will have 64-byte alignment even if the containing struct has no explicit
  // @align decorator). We use the max of this computed alignment and any
  // explicit alignment specified on the op.
  std::optional<int64_t> typeAlign =
      DataLayoutInterface::getTypeABIAlign(target, ptrType.getElementType());
  if (!typeAlign)
    return op.emitError("could not get alignment of element type");
  int64_t alignment =
      std::max(*typeAlign, (int64_t)resolveAlignment(op.getAlignment()));

  // Check to see if this stack allocation has a single pop.store to it and
  // some number of pop.loads.  If so, we know the store will dominate the loads
  // so we can just completely eliminate this.  This is a form of guaranteed
  // optimization, and it also matters for LLVM intrinsic propagation.
  StoreOp theStore;
  SmallVector<LoadOp> loads;
  bool allSimple = true;
  for (Operation *user : op->getUsers()) {
    if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user))
      continue;
    // If this is the first store to the stack allocation, remember it.
    if (auto storeOp = dyn_cast<StoreOp>(user))
      if (storeOp.getOperand(1) == op.getResult() && !theStore &&
          !storeOp.mightBeVolatile()) {
        theStore = storeOp;
        continue;
      }
    // Remember all the loads.
    if (auto loadOp = dyn_cast<LoadOp>(user)) {
      if (!loadOp.mightBeVolatile()) {
        loads.push_back(loadOp);
        continue;
      }
    }
    allSimple = false;
    break;
  }

  // If all the accesses are simple, we can just remove this entirely.
  if (allSimple && theStore) {
    bool dominates = true;
    for (auto loadOp : loads) {
      if (!domInfo.dominates(theStore, loadOp)) {
        dominates = false;
        break;
      }
    }
    if (dominates) {
      for (auto load : loads) {
        load.replaceAllUsesWith(theStore.getOperand(0));
        rewriter.eraseOp(load);
      }
    }
  }

  Value alloca = materializeLLVMAlloca(
      rewriter, target, elementType, cast<IntegerAttr>(op.getCount()).getInt(),
      op, *typeAllocSize, alignment);
  rewriter.replaceOp(op, alloca);
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertPOPStackAllocLifetimeStart
//===----------------------------------------------------------------------===//

static Value stripAddrspaceCast(Value value) {
  while (isa<LLVM::AddrSpaceCastOp>(value.getDefiningOp()))
    value = value.getDefiningOp()->getOperand(0);
  return value;
}

template <typename OpT>
static LogicalResult lowerLifetimeMarker(Operation *op, ValueRange values,
                                         TargetInfoAttr target,
                                         ConversionPatternRewriter &b) {
  for (unsigned i = 0, e = values.size(); i < e; ++i) {
    Value ptr = op->getOperand(i);
    Value value = values[i];
    [[maybe_unused]] auto alloc =
        ptr.template getDefiningOp<StackAllocationOp>();
    assert(alloc && "expected a parent stack allocation");
    SmallVector<Value> newValues;

    // LLVM verifies that pointers used by lifetime marks are defined by alloca
    // instruction directly.
    value = stripAddrspaceCast(value);
    if (!isa<LLVM::AllocaOp>(value.getDefiningOp()))
      return op->emitError("lifetime marker is only allowed for alloca");

    OpT::create(b, op->getLoc(), value);
  }
  b.eraseOp(op);
  return success();
}

class ConvertPOPStackAllocLifetimeStart
    : public ConvertPOPToLLVMPattern<StackAllocLifetimeStartOp> {
public:
  explicit ConvertPOPStackAllocLifetimeStart(mlir::LLVMTypeConverter &tc,
                                             TargetInfoAttr target)
      : ConvertPOPToLLVMPattern(tc), target(target) {}

  LogicalResult matchAndRewrite(StackAllocLifetimeStartOp op,
                                StackAllocLifetimeStartOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    return lowerLifetimeMarker<LLVM::LifetimeStartOp>(op, adaptor.getValues(),
                                                      target, b);
  }

private:
  TargetInfoAttr target;
};

//===----------------------------------------------------------------------===//
// ConvertPOPStackAllocLifetimeEnd
//===----------------------------------------------------------------------===//

class ConvertPOPStackAllocLifetimeEnd
    : public ConvertPOPToLLVMPattern<StackAllocLifetimeEndOp> {
public:
  explicit ConvertPOPStackAllocLifetimeEnd(mlir::LLVMTypeConverter &tc,
                                           TargetInfoAttr target)
      : ConvertPOPToLLVMPattern(tc), target(target) {}

  LogicalResult matchAndRewrite(StackAllocLifetimeEndOp op,
                                StackAllocLifetimeEndOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    return lowerLifetimeMarker<LLVM::LifetimeEndOp>(op, adaptor.getValues(),
                                                    target, b);
  }

private:
  TargetInfoAttr target;
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayCreate
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayCreate : public ConvertPOPToLLVMPattern<ArrayCreateOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, ArrayCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = convertType(op.getType());
    if (!type)
      return op.emitError("failed to convert array type");

    Value array = LLVM::UndefOp::create(rewriter, op.getLoc(), type);
    for (auto [idx, val] : llvm::enumerate(adaptor.getOperands()))
      array =
          LLVM::InsertValueOp::create(rewriter, op.getLoc(), array, val, idx);
    rewriter.replaceOp(op, array);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayRepeat
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayRepeat : public ConvertPOPToLLVMPattern<ArrayRepeatOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayRepeatOp op, ArrayRepeatOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = convertType(op.getType());
    if (!type)
      return op.emitError("failed to convert array type");

    Value array = LLVM::UndefOp::create(rewriter, op.getLoc(), type);
    // Fill the consecutive elements of the array by cycling through the
    // operands until the array is filled.
    for (unsigned i = 0, size = *op.getType().getResolvedSize(); i < size;) {
      for (auto it = adaptor.getOperands().begin(),
                e = adaptor.getOperands().end();
           it != e && i < size; ++it, ++i) {
        array =
            LLVM::InsertValueOp::create(rewriter, op.getLoc(), array, *it, i);
      }
    }
    rewriter.replaceOp(op, array);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayGet
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayGet : public ConvertPOPToLLVMPattern<ArrayGetOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayGetOp op, ArrayGetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getArray(), cast<IntegerAttr>(op.getIndex()).getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayReplace
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayReplace : public ConvertPOPToLLVMPattern<ArrayReplaceOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayReplaceOp op, ArrayReplaceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getArray(), adaptor.getValue(),
        cast<IntegerAttr>(op.getIndex()).getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayGEP
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayGEP : public ConvertPOPToLLVMPattern<ArrayGEPOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayGEPOp op, ArrayGEPOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptrType = convertType(op.getType());
    Type elementType = convertType(op.getArray().getType().getElementType());
    if (!ptrType)
      return op.emitError("failed to convert result type");
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, ptrType, elementType, adaptor.getArray(),
        ArrayRef<LLVM::GEPArg>{0, adaptor.getIndex()},
        /*noWrapFlags=*/LLVM::GEPNoWrapFlags::inbounds);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// getAtomicOrdering
//===----------------------------------------------------------------------===//

static LLVM::AtomicOrdering getAtomicOrdering(AtomicOrdering ordering) {
  switch (ordering) {
  case AtomicOrdering::NOT_ATOMIC:
    return LLVM::AtomicOrdering::not_atomic;
  case AtomicOrdering::UNORDERED:
    return LLVM::AtomicOrdering::unordered;
  case AtomicOrdering::MONOTONIC:
    return LLVM::AtomicOrdering::monotonic;
  case AtomicOrdering::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case AtomicOrdering::RELEASE:
    return LLVM::AtomicOrdering::release;
  case AtomicOrdering::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  case AtomicOrdering::SEQUENTIALLY_CONSISTENT:
    return LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("unknown atomic ordering");
}

//===----------------------------------------------------------------------===//
// ConvertPOPLoad
//===----------------------------------------------------------------------===//

struct ConvertPOPLoad : ConvertPOPToLLVMPattern<LoadOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, LoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = cast<PointerType>(op.getPtr().getType());
    Type elementType = typeConverter->convertType(ptrType.getElementType());
    unsigned alignment =
        getAlignment(getTypeConverter(), ptrType, adaptor.getAlignmentAttr());

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, elementType, adaptor.getPtr(), /*alignment=*/alignment,
        /*isVolatile=*/cast<BoolAttr>(adaptor.getIsVolatile()).getValue(),
        /*isNonTemporal=*/cast<BoolAttr>(adaptor.getIsNonTemporal()).getValue(),
        /*isInvariant=*/cast<BoolAttr>(adaptor.getIsInvariant()).getValue(),
        /*isInvariantGroup=*/false,
        /*ordering=*/getAtomicOrdering(adaptor.getOrdering()),
        /*syncscope=*/adaptor.getSyncscope()
            ? cast<StringAttr>(*adaptor.getSyncscope())
            : StringRef());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStore
//===----------------------------------------------------------------------===//

struct ConvertPOPStore : ConvertPOPToLLVMPattern<StoreOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, StoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto ptrType = cast<PointerType>(op.getPtr().getType());
    unsigned alignment =
        getAlignment(getTypeConverter(), ptrType, adaptor.getAlignmentAttr());

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getArg(), adaptor.getPtr(), /*alignment=*/alignment,
        /*isVolatile=*/cast<BoolAttr>(adaptor.getIsVolatile()).getValue(),
        /*isNonTemporal=*/cast<BoolAttr>(adaptor.getIsNonTemporal()).getValue(),
        /*isInvariantGroup=*/false,
        /*ordering=*/getAtomicOrdering(adaptor.getOrdering()),
        /*syncscope=*/adaptor.getSyncscope()
            ? cast<StringAttr>(*adaptor.getSyncscope())
            : StringRef());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPMemcpy
//===----------------------------------------------------------------------===//

struct ConvertPOPMemcpy : ConvertPOPToLLVMPattern<MemcpyOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemcpyOp op, MemcpyOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::MemcpyOp>(
        op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
        cast<BoolAttr>(adaptor.getIsVolatile()).getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCastToBuiltin
//===----------------------------------------------------------------------===//

struct ConvertPOPCastToBuiltin : ConvertPOPToLLVMPattern<CastToBuiltinOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastToBuiltinOp op, CastToBuiltinOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCastFromBuiltin
//===----------------------------------------------------------------------===//

struct ConvertPOPCastFromBuiltin : ConvertPOPToLLVMPattern<CastFromBuiltinOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastFromBuiltinOp op, CastFromBuiltinOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPInlineAsm
//===----------------------------------------------------------------------===//

/// Expand one level of structs so kgen.pack elements are passed as individual
/// values instead of as a kgen.struct.
///
/// Takes both the converted LLVM values and the original KGEN types to
/// correctly map logical field indices to LLVM field indices (which may
/// differ due to alignment padding fields).
static SmallVector<Value> expandOperands(ConversionPatternRewriter &rewriter,
                                         Location loc, ValueRange args,
                                         TypeRange origTypes,
                                         const POPToLLVMTypeConverter &tc) {
  SmallVector<Value> operands;
  operands.reserve(args.size());
  for (auto [value, origType] : llvm::zip(args, origTypes)) {
    // Squash pointless conversion casts that will get in the way of folds.
    value = squashPointlessCasts(value);

    // Check if the original type is a KGEN struct type.
    if (auto kgenStructTy = dyn_cast<StructType>(origType)) {
      // Get the element types - should be resolved at this point.
      if (auto elemTypes = kgenStructTy.getElementTypes()) {
        // Iterate over KGEN logical field indices and extract using remapped
        // LLVM indices to account for any padding fields.
        for (size_t i = 0, e = elemTypes->size(); i != e; ++i) {
          int64_t llvmIdx = tc.getRemappedFieldIndex(kgenStructTy, i);
          auto elt =
              rewriter.createOrFold<LLVM::ExtractValueOp>(loc, value, llvmIdx);
          operands.push_back(elt);
        }
      } else {
        // Parametric struct - just push the value as-is.
        operands.push_back(value);
      }
    } else {
      operands.push_back(value);
    }
  }
  return operands;
}

struct ConvertPOPInlineAsm : ConvertPOPToLLVMPattern<InlineAsmOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(InlineAsmOp op, InlineAsmOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 1> types;
    if (op.getNumResults()) {
      types.push_back(
          getTypeConverter()->packFunctionResults(op->getResultTypes()));
      if (!types.back())
        return failure();
    }

    auto asmOp = LLVM::InlineAsmOp::create(
        rewriter, op.getLoc(), types.empty() ? Type() : types.front(),
        expandOperands(rewriter, op.getLoc(), adaptor.getOperands(),
                       op.getOperands().getTypes(), *getTypeConverter()),
        cast<StringAttr>(adaptor.getAssembly()),
        cast<StringAttr>(adaptor.getConstraints()),
        cast<BoolAttr>(adaptor.getHasSideEffects()).getValue(),
        cast<BoolAttr>(adaptor.getIsStackAligned()).getValue(),
        LLVM::TailCallKind::None,
        LLVM::AsmDialectAttr::get(op->getContext(), LLVM::AsmDialect::AD_ATT),
        adaptor.getOperandAttrsAttr());

    if (op.getNumResults() <= 1) {
      rewriter.replaceOp(op, asmOp);
      return success();
    }
    // Unpack the results.
    SmallVector<Value> results;
    for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
      results.push_back(LLVM::ExtractValueOp::create(rewriter, op.getLoc(),
                                                     asmOp.getResult(0), i));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAtomicCmpXchg
//===----------------------------------------------------------------------===//

class ConvertPOPAtomicCmpXchg
    : public ConvertPOPToLLVMPattern<AtomicCmpXchgOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(AtomicCmpXchgOp op, AtomicCmpXchgOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::AtomicCmpXchgOp>(
        op, adaptor.getPtr(), adaptor.getCmp(), adaptor.getVal(),
        getAtomicOrdering(op.getSuccessOrdering()),
        getAtomicOrdering(op.getFailureOrdering()),
        adaptor.getSyncscope() ? cast<StringAttr>(*adaptor.getSyncscope())
                               : StringRef(),
        resolveAlignment(adaptor), adaptor.getWeak().has_value());
    return success();
  }

  static unsigned resolveAlignment(AtomicCmpXchgOpAdaptor adaptor) {
    if (auto alignment = adaptor.getAlignment())
      return cast<IntegerAttr>(*alignment).getInt();

    // If alignment is not set on the op, use the alignment of the pointer.
    Value ptr = adaptor.getPtr();
    if (!ptr.getDefiningOp()->hasAttr("alignment"))
      return 0;

    return cast<IntegerAttr>(ptr.getDefiningOp()->getAttr("alignment"))
        .getInt();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAtomicRMW
//===----------------------------------------------------------------------===//

class ConvertPOPAtomicRMW : public ConvertPOPToLLVMPattern<AtomicRMWOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(AtomicRMWOp op, AtomicRMWOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *cast<SIMDType>(op.getType()).getResolvedDType();
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
        op, getAtomicBinOp(dtype, adaptor.getBinOp()), adaptor.getPtr(),
        adaptor.getVal(), getAtomicOrdering(op.getOrdering()),
        adaptor.getSyncscope() ? cast<StringAttr>(*adaptor.getSyncscope())
                               : StringRef());
    return success();
  }

private:
  static LLVM::AtomicBinOp getAtomicBinOp(KGENDType dtype, AtomicBinOp binOp) {
    switch (binOp) {
    case AtomicBinOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    case AtomicBinOp::ADD:
      return dtype.isFloat() ? LLVM::AtomicBinOp::fadd : LLVM::AtomicBinOp::add;
    case AtomicBinOp::SUB:
      return dtype.isFloat() ? LLVM::AtomicBinOp::fsub : LLVM::AtomicBinOp::sub;
    case AtomicBinOp::AND:
      return LLVM::AtomicBinOp::_and;
    case AtomicBinOp::NAND:
      return LLVM::AtomicBinOp::nand;
    case AtomicBinOp::OR:
      return LLVM::AtomicBinOp::_or;
    case AtomicBinOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case AtomicBinOp::MAX:
      if (dtype.isSInt())
        return LLVM::AtomicBinOp::max;
      if (dtype.isUInt())
        return LLVM::AtomicBinOp::umax;
      if (dtype.isFloat())
        return LLVM::AtomicBinOp::fmax;
      break;
    case AtomicBinOp::MIN:
      if (dtype.isSInt())
        return LLVM::AtomicBinOp::min;
      if (dtype.isUInt())
        return LLVM::AtomicBinOp::umin;
      if (dtype.isFloat())
        return LLVM::AtomicBinOp::fmin;
      break;
    }
    llvm_unreachable("unknown atomic ordering");
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPFence
//===----------------------------------------------------------------------===//

class ConvertPOPFence : public ConvertPOPToLLVMPattern<FenceOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(FenceOp op, FenceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(
        op, getAtomicOrdering(adaptor.getOrdering()),
        adaptor.getSyncscopeAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStringAddress
//===----------------------------------------------------------------------===//

struct ConvertPOPStringAddress
    : public ConvertPOPToLLVMPattern<StringAddressOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StringAddressOp op, StringAddressOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // The first operand is a !kgen.string lowered to
    // !llvm.struct<(ptr<i8>, index)>, grab the first field: the address
    // of the string.
    Value extractedAddr =
        LLVM::ExtractValueOp::create(b, adaptor.getOperands().front(), 0);
    rewriter.replaceOp(op, extractedAddr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStringSize
//===----------------------------------------------------------------------===//

struct ConvertPOPStringSize : public ConvertPOPToLLVMPattern<StringSizeOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StringSizeOp op, StringSizeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // The first operand is a !kgen.string lowered to
    // !llvm.struct<(ptr<i8>, index)>, grab the second field: the size
    // of the string.
    Value extractedAddr =
        LLVM::ExtractValueOp::create(b, adaptor.getOperands().front(), 1);
    rewriter.replaceOp(op, extractedAddr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPDTypeToUI8
//===----------------------------------------------------------------------===//

struct ConvertPOPDTypeToUI8 : public ConvertPOPToLLVMPattern<DTypeToUI8> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(DTypeToUI8 op, DTypeToUI8Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, type, adaptor.getDType());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPDTypeFromUI8
//===----------------------------------------------------------------------===//

struct ConvertPOPDTypeFromUI8 : public ConvertPOPToLLVMPattern<DTypeFromUI8> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(DTypeFromUI8 op, DTypeFromUI8Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCallLLVMIntrinsic
//===----------------------------------------------------------------------===//

struct ConvertPOPCallLLVMIntrinsic
    : public ConvertPOPToLLVMPattern<CallLLVMIntrinsicOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CallLLVMIntrinsicOp op, CallLLVMIntrinsicOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return failure();

    StringRef intrinsicName = cast<StringAttr>(op.getIntrin());

    // Special handling for masked.load: convert alignment operand to attribute
    if (intrinsicName == "llvm.masked.load") {
      auto operands =
          expandOperands(rewriter, op.getLoc(), adaptor.getOperands(),
                         op.getOperands().getTypes(), *getTypeConverter());
      if (operands.size() != 4) {
        return op.emitError("llvm.masked.load expects 4 operands "
                            "(ptr, alignment, mask, passthrough), got ")
               << operands.size();
      }

      Value ptr = operands[0];
      Value alignmentOp = operands[1];
      Value mask = operands[2];
      Value passthrough = operands[3];

      // Extract alignment constant from the operand
      auto alignConstOp = alignmentOp.getDefiningOp<LLVM::ConstantOp>();
      if (!alignConstOp) {
        return op.emitError(
            "llvm.masked.load alignment must be a constant, got runtime value");
      }

      auto alignAttr = dyn_cast<IntegerAttr>(alignConstOp.getValue());
      if (!alignAttr) {
        return op.emitError("llvm.masked.load alignment must be an integer");
      }

      // Create MaskedLoadOp with alignment as attribute
      rewriter.replaceOpWithNewOp<LLVM::MaskedLoadOp>(op, types[0], ptr, mask,
                                                      passthrough, alignAttr,
                                                      /*nontemporal=*/nullptr);
      return success();
    }

    // Special handling for masked.store: convert alignment operand to attribute
    if (intrinsicName == "llvm.masked.store") {
      auto operands =
          expandOperands(rewriter, op.getLoc(), adaptor.getOperands(),
                         op.getOperands().getTypes(), *getTypeConverter());
      if (operands.size() != 4) {
        return op.emitError("llvm.masked.store expects 4 operands "
                            "(value, ptr, alignment, mask), got ")
               << operands.size();
      }

      Value value = operands[0];
      Value ptr = operands[1];
      Value alignmentOp = operands[2];
      Value mask = operands[3];

      // Extract alignment constant from the operand
      auto alignConstOp = alignmentOp.getDefiningOp<LLVM::ConstantOp>();
      if (!alignConstOp) {
        return op.emitError("llvm.masked.store alignment must be a constant, "
                            "got runtime value");
      }

      auto alignAttr = dyn_cast<IntegerAttr>(alignConstOp.getValue());
      if (!alignAttr) {
        return op.emitError("llvm.masked.store alignment must be an integer");
      }

      // Create MaskedStoreOp with alignment as attribute
      rewriter.replaceOpWithNewOp<LLVM::MaskedStoreOp>(op, value, ptr, mask,
                                                       alignAttr);
      return success();
    }

    // Special handling for masked.gather: convert alignment operand to
    // attribute
    if (intrinsicName == "llvm.masked.gather") {
      auto operands =
          expandOperands(rewriter, op.getLoc(), adaptor.getOperands(),
                         op.getOperands().getTypes(), *getTypeConverter());
      if (operands.size() != 4) {
        return op.emitError("llvm.masked.gather expects 4 operands "
                            "(ptr_vec, alignment, mask, passthrough), got ")
               << operands.size();
      }

      Value ptrs = operands[0];
      Value alignmentOp = operands[1];
      Value mask = operands[2];
      Value passthrough = operands[3];

      // Extract alignment constant from the operand
      auto alignConstOp = alignmentOp.getDefiningOp<LLVM::ConstantOp>();
      if (!alignConstOp) {
        return op.emitError("llvm.masked.gather alignment must be a constant, "
                            "got runtime value");
      }

      auto alignAttr = dyn_cast<IntegerAttr>(alignConstOp.getValue());
      if (!alignAttr) {
        return op.emitError("llvm.masked.gather alignment must be an integer");
      }

      // Create arg_attrs with alignment on the first argument (ptrs)
      auto alignNamedAttr = rewriter.getNamedAttr("align", alignAttr);
      auto ptrAttrs = rewriter.getDictionaryAttr({alignNamedAttr});
      auto emptyAttrs = rewriter.getDictionaryAttr({});
      auto argAttrs = rewriter.getArrayAttr({ptrAttrs, emptyAttrs, emptyAttrs});

      // Create CallIntrinsicOp with alignment as arg attribute on ptrs
      auto callOp = LLVM::CallIntrinsicOp::create(
          rewriter, op.getLoc(), types, cast<StringAttr>(op.getIntrin()),
          SmallVector<Value>{ptrs, mask, passthrough},
          convertFastmathFlags(op.getFastmathFlags(), rewriter),
          /*op_bundle_operands=*/ArrayRef<ValueRange>{},
          /*op_bundle_tags=*/mlir::ArrayAttr{},
          /*arg_attrs=*/argAttrs,
          /*res_attrs=*/mlir::ArrayAttr{});
      rewriter.replaceOp(op, callOp);
      return success();
    }

    // Special handling for masked.scatter: convert alignment operand to
    // attribute
    if (intrinsicName == "llvm.masked.scatter") {
      auto operands =
          expandOperands(rewriter, op.getLoc(), adaptor.getOperands(),
                         op.getOperands().getTypes(), *getTypeConverter());
      if (operands.size() != 4) {
        return op.emitError("llvm.masked.scatter expects 4 operands "
                            "(value, ptr_vec, alignment, mask), got ")
               << operands.size();
      }

      Value value = operands[0];
      Value ptrs = operands[1];
      Value alignmentOp = operands[2];
      Value mask = operands[3];

      // Extract alignment constant from the operand
      auto alignConstOp = alignmentOp.getDefiningOp<LLVM::ConstantOp>();
      if (!alignConstOp) {
        return op.emitError("llvm.masked.scatter alignment must be a constant, "
                            "got runtime value");
      }

      auto alignAttr = dyn_cast<IntegerAttr>(alignConstOp.getValue());
      if (!alignAttr) {
        return op.emitError("llvm.masked.scatter alignment must be an integer");
      }

      // Create arg_attrs with alignment on the second argument (ptrs)
      auto emptyAttrs = rewriter.getDictionaryAttr({});
      auto alignNamedAttr = rewriter.getNamedAttr("align", alignAttr);
      auto ptrAttrs = rewriter.getDictionaryAttr({alignNamedAttr});
      auto argAttrs = rewriter.getArrayAttr({emptyAttrs, ptrAttrs, emptyAttrs});

      // Create CallIntrinsicOp with alignment as arg attribute on ptrs
      auto callOp = LLVM::CallIntrinsicOp::create(
          rewriter, op.getLoc(), types, cast<StringAttr>(op.getIntrin()),
          SmallVector<Value>{value, ptrs, mask},
          convertFastmathFlags(op.getFastmathFlags(), rewriter),
          /*op_bundle_operands=*/ArrayRef<ValueRange>{},
          /*op_bundle_tags=*/mlir::ArrayAttr{},
          /*arg_attrs=*/argAttrs,
          /*res_attrs=*/mlir::ArrayAttr{});
      rewriter.replaceOp(op, callOp);
      return success();
    }

    // just emit regular LLVM intrinsic call.
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, types, cast<StringAttr>(op.getIntrin()),
        expandOperands(rewriter, op.getLoc(), adaptor.getOperands(),
                       op.getOperands().getTypes(), *getTypeConverter()),
        convertFastmathFlags(op.getFastmathFlags(), rewriter));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPPointerBitcast
//===----------------------------------------------------------------------===//

struct ConvertPOPPointerBitcast
    : public ConvertPOPToLLVMPattern<PointerBitcastOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(PointerBitcastOp op, PointerBitcastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    // The LLVMPointerType doesn't maintain an element type, just an address
    // space.  Insert an address space cast if needed.
    auto srcVal = adaptor.getOperands()[0];
    if (srcVal.getType() != resultTy)
      rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op, resultTy, srcVal);
    else
      rewriter.replaceOp(op, srcVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPUnionBitcast
//===----------------------------------------------------------------------===//

struct ConvertPOPUnionBitcast : public ConvertPOPToLLVMPattern<UnionBitcastOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(UnionBitcastOp op,
                                UnionBitcastOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    b.replaceOp(op, adaptor.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPUnionWrap
//===----------------------------------------------------------------------===//

static FailureOr<Value> materializeLLVMUnionAlloca(TargetInfoAttr target,
                                                   OpBuilder &b, Operation *op,
                                                   Type popUnionType,
                                                   Type llvmUnionType) {
  std::optional<int64_t> typeAllocSize =
      DataLayoutInterface::getTypeAllocSize(target, popUnionType);
  std::optional<int64_t> typeABIAlign =
      DataLayoutInterface::getTypeABIAlign(target, popUnionType);
  if (!typeAllocSize || !typeABIAlign)
    return op->emitError("failed to get union type size and alignment");

  return materializeLLVMAlloca(b, target, llvmUnionType, 1, op, *typeAllocSize,
                               *typeABIAlign);
}

struct ConvertPOPUnionWrap : public ConvertPOPToLLVMPattern<UnionWrapOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(UnionWrapOp op, UnionWrapOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {

    auto variantType =
        dyn_cast_or_null<LLVM::LLVMStructType>(convertType(op.getType()));
    if (!variantType)
      return failure();

    TargetInfoAttr target = getTypeConverter()->getTarget();
    FailureOr<Value> ptrOr =
        materializeLLVMUnionAlloca(target, b, op, op.getType(), variantType);

    if (failed(ptrOr))
      return failure();

    LLVM::StoreOp::create(b, op->getLoc(), adaptor.getValue(), *ptrOr);
    b.replaceOpWithNewOp<LLVM::LoadOp>(op, variantType, *ptrOr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPUnionUnwrap
//===----------------------------------------------------------------------===//

struct ConvertPOPUnionUnwrap : public ConvertPOPToLLVMPattern<UnionUnwrapOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(UnionUnwrapOp op, UnionUnwrapOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Type valueType = convertType(op.getType());
    if (!valueType)
      return failure();

    auto contentType = cast<LLVM::LLVMStructType>(adaptor.getValue().getType());
    if (contentType.getBody().empty()) {
      b.replaceOpWithNewOp<LLVM::UndefOp>(op, contentType);
      return success();
    }
    assert((contentType.getBody().size() == 1 ||
            contentType.getBody().size() == 2) &&
           "must have 1 or 2 fields for union struct.");

    TargetInfoAttr target = getTypeConverter()->getTarget();
    FailureOr<Value> ptrOr = materializeLLVMUnionAlloca(
        target, b, op, op.getValue().getType(), contentType);

    if (failed(ptrOr))
      return failure();

    LLVM::StoreOp::create(b, op->getLoc(), adaptor.getValue(), *ptrOr);
    b.replaceOpWithNewOp<LLVM::LoadOp>(op, valueType, *ptrOr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDReduceOr
//===----------------------------------------------------------------------===//

/// Convert a SIMD reduction into an llvm.vector.reduce.or intrinsic. If the
/// vector only has one element, extract that element and return it.
struct ConvertPOPSIMDReduceOr : public ConvertPOPToLLVMPattern<SIMDReduceOrOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDReduceOrOp op, SIMDReduceOrOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = convertType(op.getType());

    // If the original vector is size 1, skip the reduction and just extract the
    // first element (note the "vector" in this case is already a scalar type,
    // so we do a straightforward replace).
    if (op.getVector().getType().isScalar()) {
      rewriter.replaceOp(op, adaptor.getVector());
      return success();
    }

    auto reduction = LLVM::vector_reduce_or::create(rewriter, loc, resType,
                                                    adaptor.getVector());

    rewriter.replaceOp(op, reduction);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDReduceAnd
//===----------------------------------------------------------------------===//

/// Convert a SIMD reduction into an llvm.vector.reduce.and intrinsic. If the
/// vector only has one element, extract that element and return it.
struct ConvertPOPSIMDReduceAnd
    : public ConvertPOPToLLVMPattern<SIMDReduceAndOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDReduceAndOp op, SIMDReduceAndOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = convertType(op.getType());

    // If the original vector is size 1, skip the reduction and just extract the
    // first element (note the "vector" in this case is already a scalar type,
    // so we do a straightforward replace).
    if (op.getVector().getType().isScalar()) {
      rewriter.replaceOp(op, adaptor.getVector());
      return success();
    }

    auto reduction = LLVM::vector_reduce_and::create(rewriter, loc, resType,
                                                     adaptor.getVector());

    rewriter.replaceOp(op, reduction);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Trivial Conversions
//===----------------------------------------------------------------------===//

using ConvertPOPAnd = mlir::OneToOneConvertToLLVMPattern<AndOp, LLVM::AndOp>;
using ConvertPOPOr = mlir::OneToOneConvertToLLVMPattern<OrOp, LLVM::OrOp>;
using ConvertPOPXOr = mlir::OneToOneConvertToLLVMPattern<XOrOp, LLVM::XOrOp>;
using ConvertPOPSIMDAnd =
    mlir::OneToOneConvertToLLVMPattern<SIMDAndOp, LLVM::AndOp>;
using ConvertPOPSIMDOr =
    mlir::OneToOneConvertToLLVMPattern<SIMDOrOp, LLVM::OrOp>;
using ConvertPOPSIMDXOr =
    mlir::OneToOneConvertToLLVMPattern<SIMDXOrOp, LLVM::XOrOp>;
using ConvertPOPAdd =
    OneToOneFloatOrIntConversion<AddOp, LLVM::FAddOp, LLVM::AddOp>;
using ConvertPOPSub =
    OneToOneFloatOrIntConversion<SubOp, LLVM::FSubOp, LLVM::SubOp>;
using ConvertPOPMul =
    OneToOneFloatOrIntConversion<MulOp, LLVM::FMulOp, LLVM::MulOp>;
using ConvertPOPDiv = OneToOneFloatOrIntConversion<DivOp, LLVM::FDivOp,
                                                   LLVM::SDivOp, LLVM::UDivOp>;
using ConvertPOPRem = OneToOneFloatOrIntConversion<RemOp, LLVM::FRemOp,
                                                   LLVM::SRemOp, LLVM::URemOp>;
using ConvertPOPMax = OneToOneFloatOrIntConversion<MaxOp, LLVM::MaxNumOp,
                                                   LLVM::SMaxOp, LLVM::UMaxOp>;
using ConvertPOPMin = OneToOneFloatOrIntConversion<MinOp, LLVM::MinNumOp,
                                                   LLVM::SMinOp, LLVM::UMinOp>;
using ConvertPOPBitcast =
    mlir::OneToOneConvertToLLVMPattern<BitcastOp, LLVM::BitcastOp>;
using ConvertPOPShl = mlir::OneToOneConvertToLLVMPattern<ShlOp, LLVM::ShlOp>;
using ConvertPOPPointerToIndex =
    mlir::OneToOneConvertToLLVMPattern<PointerToIndexOp, LLVM::PtrToIntOp>;

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populatePOPToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns) {
  patterns.insert<
      // clang-format off
      ConvertPOPAbs,
      ConvertPOPAdd,
      ConvertPOPAnd,
      ConvertPOPArrayCreate,
      ConvertPOPArrayGEP,
      ConvertPOPArrayGet,
      ConvertPOPArrayRepeat,
      ConvertPOPArrayReplace,
      ConvertPOPAtomicCmpXchg,
      ConvertPOPAtomicRMW,
      ConvertPOPBitcast,
      ConvertPOPCallLLVMIntrinsic,
      ConvertPOPCast,
      ConvertPOPCastFromBuiltin,
      ConvertPOPCastToBuiltin,
      ConvertPOPCmp,
      ConvertPOPDiv,
      ConvertPOPDTypeFromUI8,
      ConvertPOPDTypeToUI8,
      ConvertPOPCeil,
      ConvertPOPFence,
      ConvertPOPFloor,
      ConvertPOPFloorDiv,
      ConvertPOPFMA,
      ConvertPOPInlineAsm,
      ConvertPOPLoad,
      ConvertPOPMemcpy,
      ConvertPOPMul,
      ConvertPOPNeg,
      ConvertPOPOffset,
      ConvertPOPOr,
      ConvertPOPPointerBitcast,
      ConvertPOPPointerToIndex,
      ConvertPOPRem,
      ConvertPOPRound,
      ConvertPOPSelect,
      ConvertPOPShl,
      ConvertPOPShr,
      ConvertPOPSIMDAnd,
      ConvertPOPSIMDExtractElement,
      ConvertPOPSIMDInsertElement,
      ConvertPOPSIMDOr,
      ConvertPOPSIMDSelect,
      ConvertPOPSIMDShuffle,
      ConvertPOPSIMDSplat,
      ConvertPOPSIMDXOr,
      ConvertPOPStore,
      ConvertPOPStringAddress,
      ConvertPOPStringSize,
      ConvertPOPSub,
      ConvertPOPTrunc,
      ConvertPOPUnionBitcast,
      ConvertPOPUnionUnwrap,
      ConvertPOPUnionWrap,
      ConvertPOPXOr,
      ConvertPOPSIMDReduceOr,
      ConvertPOPSIMDReduceAnd
      // clang-format on
      >(typeConverter);
}

//===----------------------------------------------------------------------===//
// LowerPOPToLLVMPass
//===----------------------------------------------------------------------===//

namespace {
struct LowerPOPToLLVMPass
    : public KGEN::impl::LowerPOPToLLVMBase<LowerPOPToLLVMPass> {
  using LowerPOPToLLVMBase::LowerPOPToLLVMBase;

  void runOnOperation() override;

  /// Verify that the operation is a function and has no nested CFGs.
  FailureOr<mlir::FunctionOpInterface> validateOperation();
};
} // namespace

FailureOr<mlir::FunctionOpInterface> LowerPOPToLLVMPass::validateOperation() {
  auto func = dyn_cast<mlir::FunctionOpInterface>(getOperation());
  if (!func)
    return getOperation()->emitError(
        "lower-pop-to-llvm must be nested on a FunctionOpInterface");

  // Stack allocations cannot be lowered in the presence of CFGs.
  Operation *cfgOp = nullptr;
  func->walk([&cfgOp](Operation *op) {
    if (llvm::none_of(op->getRegions(), [](Region &region) {
          return region.getBlocks().size() > 1;
        }))
      return WalkResult::advance();
    cfgOp = op;
    return WalkResult::interrupt();
  });
  if (!cfgOp)
    return func;

  InFlightDiagnostic diag = cfgOp->emitError(
      "lower-pop-to-llvm cannot run on operations with CFG regions");
  diag.attachNote() << "try running it before lower-control-flow";
  return diag;
}

void LowerPOPToLLVMPass::runOnOperation() {
  FailureOr<mlir::FunctionOpInterface> func = validateOperation();
  if (failed(func))
    return signalPassFailure();

  // If the function body is empty, return.
  if (func->getFunctionBody().empty())
    return;

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<POPDialect>();
  target.addIllegalDialect<mlir::index::IndexDialect>();
  target.addLegalDialect<DebugInfo::DebugInfoDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // These ops are handled by other passes.
  target.addLegalOp<GlobalAllocOp>();
  target.addLegalOp<GlobalConstantOp>();
  target.addLegalOp<ExternalCallOp>();
  target.addLegalOp<ExternPointerSymbolOp>();
  target.addLegalOp<AlignedAllocOp>();
  target.addLegalOp<AlignedFreeOp>();

  // Set LLVM lowering options.
  TargetInfoAttr targetInfo = lookupTargetInfo(*func);
  if (!targetInfo) {
    mlir::emitError(func->getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  ErrorOr<const TargetLowering *> loweringOr =
      TargetLoweringRegistry::get().lookup(targetInfo.getTriple());
  const TargetLowering *lowering = loweringOr.isError() ? nullptr : *loweringOr;

  // Ops the target lowers in the global pass are left legal (skipped) here.
  target.addDynamicallyLegalOp<POP::MaxOp, POP::MinOp, POP::AtomicRMWOp,
                               POP::AtomicCmpXchgOp, POP::CastOp, POP::FenceOp,
                               CallLLVMIntrinsicOp>([&](Operation *op) {
    return lowering && lowering->isLoweredInGlobalPOPPass(op);
  });

  POPToLLVMTypeConverter typeConverter(targetInfo);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populatePOPToLLVMPatterns(typeConverter, patterns);

  patterns.insert<ConvertPOPMax, ConvertPOPMin>(typeConverter);
  mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
  patterns.insert<ConvertPOPStackAllocation, ConvertPOPStackAllocLifetimeStart,
                  ConvertPOPStackAllocLifetimeEnd>(typeConverter, targetInfo);

  // Target-specific patterns contributed by the target lowering, inserted after
  // the generic patterns. Target patterns use a higher benefit to win for the
  // ops they handle.
  if (lowering) {
    lowering->populateLowerPOPToLLVMPatterns(patterns, typeConverter,
                                             targetInfo);
  }

  if (failed(mlir::applyPartialConversion(*func, target, std::move(patterns))))
    return signalPassFailure();
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertPOPAlignedAlloc
//===----------------------------------------------------------------------===//

static constexpr llvm::StringLiteral kAllocFamilyName =
    "kgen_aligned_allocator";

/// This pattern will generate the aligned alloc function with the appropriate
/// attributes to teach LLVM about the allocator. This would enable LLVM, for
/// example, to promote heap-to-stack among other optimizations. This enables
/// the aligned alloc function to receive similar treatment to `malloc`.
struct ConvertPOPAlignedAlloc : public ConvertSymbolOpToLLVM<AlignedAllocOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  static constexpr llvm::StringLiteral kAllocFnName =
      "KGEN_CompilerRT_AlignedAlloc";

  LogicalResult matchAndRewrite(AlignedAllocOp op,
                                AlignedAllocOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    // Try to find an existing function
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(kAllocFnName);
    if (!func) {
      // No function found. Create one with the appropriate attributes.
      const mlir::LLVMTypeConverter &tc = *getTypeConverter();
      OpBuilder::InsertionGuard guard(b);
      b.clearInsertionPoint();

      // The function signature is `ptr(index, index)`.
      auto allocFnSig =
          LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(getContext()),
                                      {tc.getIndexType(), tc.getIndexType()});

      SmallVector<Attribute> passthrough;
      func = LLVM::LLVMFuncOp::create(b, mlir::UnknownLoc::get(getContext()),
                                      kAllocFnName, allocFnSig);

      // `noalias` result.
      func.setResultAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(),
                         b.getUnitAttr());
      // `allocalign` on the first argument.
      func.setArgAttr(0, LLVM::LLVMDialect::getAllocAlignAttrName(),
                      b.getUnitAttr());

      // `allockind("alloc,aligned,uninitialized")` enum encoding.
      // FIXME: The encoding of integer attributes is a string?!
      passthrough.push_back(b.getArrayAttr(
          {b.getStringAttr("allockind"),
           b.getStringAttr(Twine(static_cast<int64_t>(
               llvm::AllocFnKind::Alloc | llvm::AllocFnKind::Aligned |
               llvm::AllocFnKind::Uninitialized)))}));

      // `allocsize(1)` with `-1` in lower 32 bits.
      // FIXME: The encoding of integer attributes is a string?!
      // FIXME: `packAllocSizeArgs` is not an exposed function.
      passthrough.push_back(b.getArrayAttr(
          {b.getStringAttr("allocsize"),
           b.getStringAttr(Twine(uint32_t(-1) | (uint64_t(1) << 32)))}));
      // `"alloc-family"="kgen_alloc"`.
      passthrough.push_back(
          b.getArrayAttr({b.getStringAttr("alloc-family"),
                          b.getStringAttr(kAllocFamilyName)}));

      func.setPassthroughAttr(attachTargetPassthroughAttrs(
          b, getTypeConverter()->getTarget(), b.getArrayAttr(passthrough)));
      symtab.insert(func);
    }

    LLVM::CallOp call =
        createLLVMCall(b, op.getLoc(), func, adaptor.getOperands());
    b.replaceOpWithNewOp<LLVM::BitcastOp>(op, convertType(op.getType()),
                                          call.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAlignedFree
//===----------------------------------------------------------------------===//

/// This pattern will generate the aligned free function with the appropriate
/// attributes to teach LLVM about the allocator. This would enable LLVM, for
/// example, to promote heap-to-stack among other optimizations. This enables
/// the aligned free function to receive similar treatment to `free`.
struct ConvertPOPAlignedFree : public ConvertSymbolOpToLLVM<AlignedFreeOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  static constexpr llvm::StringLiteral kFreeFnName =
      "KGEN_CompilerRT_AlignedFree";

  LogicalResult matchAndRewrite(AlignedFreeOp op, AlignedFreeOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    // Try to find an existing function
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(kFreeFnName);
    if (!func) {
      // No function found. Create one with the appropriate attributes.
      OpBuilder::InsertionGuard guard(b);
      b.clearInsertionPoint();

      // The function signature is `void(ptr)`.
      auto freeFnSig =
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()),
                                      LLVM::LLVMPointerType::get(getContext()));

      SmallVector<Attribute> passthrough;
      func = LLVM::LLVMFuncOp::create(b, mlir::UnknownLoc::get(getContext()),
                                      kFreeFnName, freeFnSig);

      // `allocptr` on first argument.
      func.setArgAttr(0, LLVM::LLVMDialect::getAllocatedPointerAttrName(),
                      b.getUnitAttr());

      // `allockind("alloc,aligned,uninitialized")` enum encoding.
      // FIXME: The encoding of integer attributes is a string?!
      passthrough.push_back(b.getArrayAttr(
          {b.getStringAttr("allockind"),
           b.getStringAttr(
               Twine(static_cast<uint64_t>(llvm::AllocFnKind::Free)))}));

      // `"alloc-family"="kgen_alloc"`.
      passthrough.push_back(
          b.getArrayAttr({b.getStringAttr("alloc-family"),
                          b.getStringAttr(kAllocFamilyName)}));

      func.setPassthroughAttr(attachTargetPassthroughAttrs(
          b, getTypeConverter()->getTarget(), b.getArrayAttr(passthrough)));
      symtab.insert(func);
    }

    Value ptr = LLVM::BitcastOp::create(
        b, op.getLoc(), LLVM::LLVMPointerType::get(getContext()),
        adaptor.getPtr());
    LLVM::CallOp call = createLLVMCall(b, op.getLoc(), func, ptr);
    b.replaceOp(op, call);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPGlobalAlloc
//===----------------------------------------------------------------------===//

struct ConvertPOPGlobalAlloc : public ConvertSymbolOpToLLVM<GlobalAllocOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  // Tracks the next available local index per parent function to give
  // deterministic names.
  // mark `mutable` here because it is mutated in a `const` method
  // matchAndRewrite. This pattern must remain an ModuleOp pattern
  // instead of FunctionOp pattern so that this pattern is not being running in
  // parallel while this mutable is not thread-safe.
  mutable llvm::DenseMap<mlir::Operation *, unsigned> gpuSharedAllocIdx;

  LogicalResult matchAndRewrite(GlobalAllocOp op, GlobalAllocOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    // Set the alignment if specified. Otherwise use the natural alignment.
    auto kgenPtrType = cast<PointerType>(op.getType());
    auto elementType = typeConverter->convertType(kgenPtrType.getElementType());
    unsigned alignment =
        getAlignment(getTypeConverter(), kgenPtrType, op.getAlignmentAttr());

    // Set the address space if specified.
    unsigned addrSpace = op.getType().getAddrSpaceOrZero();

    // (HACK) Add a postfix to the name here so that we can identify
    // this type of variables in the llvm module.
    // This is a workaround to LLVM MLIR lowering doesn't allow
    // GlobalValues to have passthrough metadata.

    std::string name = cast<StringAttr>(adaptor.getName()).str();
    if (op.getMemoryType() == POP::GlobalAllocAddressSpace::GPU_SHARED) {
      // Scope the name to the enclosing function so every GPU_SHARED global
      // gets a deterministic unique name per kernel.
      if (auto parentFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>()) {
        unsigned localIdx = gpuSharedAllocIdx[parentFuncOp.getOperation()]++;
        name = parentFuncOp.getSymName().str() + "." + name + "_" +
               std::to_string(localIdx);
      }
      name += "._gpu_shared_mem";
    }

    // Create the global.
    b.clearInsertionPoint();
    auto arrayType = LLVM::LLVMArrayType::get(
        elementType, cast<IntegerAttr>(op.getCount()).getInt());
    auto global = LLVM::GlobalOp::create(
        b, op.getLoc(), arrayType,
        /*isConstant=*/false, LLVM::Linkage::Internal, name,
        /*value=*/Attribute(), alignment, addrSpace);

    // If an initializer is present, emit an initializer region.
    // The verifier guarantees count == 1 when an initializer is set.
    if (auto init = op.getInitializer()) {
      global.getBodyRegion().push_back(new Block);
      ImplicitLocOpBuilder ib(op.getLoc(), op.getContext());
      ib.setInsertionPointToStart(global.getBody());
      ErrorOr<Value> value =
          convertParameterToLLVM(ib, *getTypeConverter(), /*imc=*/nullptr,
                                 /*scope=*/nullptr, *init);
      if (value.isError()) {
        ib.emitError(value.getError());
        return failure();
      }
      // Wrap the scalar value in the array type.
      Value array = LLVM::UndefOp::create(ib, arrayType);
      array = LLVM::InsertValueOp::create(ib, array, value.get(),
                                          ArrayRef<int64_t>{0});
      LLVM::ReturnOp::create(ib, array);
    }

    symtab.insert(global);

    // Replace the alloc op with an `addressof`.
    b.setInsertionPoint(op);
    auto opaquePtrType = LLVM::LLVMPointerType::get(getContext(), addrSpace);
    auto ptr = LLVM::AddressOfOp::create(b, op.getLoc(), global);
    b.replaceOpWithNewOp<LLVM::BitcastOp>(
        op,
        LLVM::LLVMPointerType::get(opaquePtrType.getContext(),
                                   opaquePtrType.getAddressSpace()),
        ptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPNoAliasPointerCast
//===----------------------------------------------------------------------===//

static LLVM::LLVMFuncOp getOrCreateNoAliasCastIntrinsic(SymbolTable &symtab,
                                                        mlir::RewriterBase &b) {
  constexpr llvm::StringLiteral fnName = "__kgen_noalias_cast";
  auto name = b.getStringAttr(fnName);
  if (auto func = symtab.lookup<LLVM::LLVMFuncOp>(name))
    return func;

  // Create the function. It has the form `noalias ptr (ptr noalias returned)`.
  // Since the returned pointer can have arbitrary effects on it, we can't
  // annotate the argument with any.
  OpBuilder::InsertionGuard guard(b);
  b.clearInsertionPoint();
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  auto func = LLVM::LLVMFuncOp::create(
      b, UnknownLoc::get(b.getContext()), name,
      LLVM::LLVMFunctionType::get(ptrType, ptrType), LLVM::Linkage::Internal);
  symtab.insert(func);

  // Set the `noalias` attributes.
  func.setArgAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(), b.getUnitAttr());
  func.setResultAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(),
                     b.getUnitAttr());

  constexpr llvm::StringLiteral funcAttrs[] = {
      "alwaysinline", "mustprogress", "nofree",    "norecurse",
      "nosync",       "nounwind",     "willreturn"};
  SmallVector<Attribute> attrs;
  for (StringRef attr : funcAttrs)
    attrs.push_back(b.getStringAttr(attr));
  // memory(none)
  attrs.push_back(
      b.getArrayAttr({b.getStringAttr("memory"), b.getStringAttr("0")}));
  func.setPassthroughAttr(b.getArrayAttr(attrs));

  // Populate the body.
  Block *body = b.createBlock(&func.getBody());
  LLVM::ReturnOp::create(b, func.getLoc(),
                         body->addArgument(ptrType, func.getLoc()));

  return func;
}

struct ConvertPOPNoAliasPointerCast
    : ConvertSymbolOpToLLVM<NoAliasPointerCastOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult matchAndRewrite(NoAliasPointerCastOp op,
                                NoAliasPointerCastOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    LLVM::LLVMFuncOp func = getOrCreateNoAliasCastIntrinsic(symtab, b);
    LLVM::CallOp call = createLLVMCall(b, op.getLoc(), func, adaptor.getIn());
    b.replaceOp(op, call);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPGlobalConstant
//===----------------------------------------------------------------------===//

/// Lower a global constant. Unique the constant value.
class ConvertPOPGlobalConstant
    : public ConvertPOPToLLVMPattern<GlobalConstantOp> {
public:
  ConvertPOPGlobalConstant(
      SymbolTable &symtab,
      DenseMap<std::pair<TypedAttr, TypedAttr>, LLVM::GlobalOp> &constants,
      mlir::LLVMTypeConverter &typeConverter, InterpreterMemoryConverter &imc)
      : ConvertPOPToLLVMPattern(typeConverter), symtab(symtab),
        constants(constants), imc(imc) {}

  LogicalResult
  matchAndRewrite(GlobalConstantOp op, GlobalConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kgenPtrType = cast<PointerType>(op.getType());
    auto opaquePtrType = LLVM::LLVMPointerType::get(getContext());
    Type elementType = convertType(kgenPtrType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op.getLoc(), "failed to convert constant result type");

    // Unique the constant.
    auto [it, inserted] = constants.try_emplace(
        std::make_pair(op.getValue(), op.getAlignmentAttr()), nullptr);
    if (inserted) {
      // If the constant doesn't exist, create it and insert it in the module.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.clearInsertionPoint();

      LLVM::GlobalOp global = LLVM::GlobalOp::create(
          rewriter, op.getLoc(), elementType, true, LLVM::Linkage::Internal,
          "global_constant", Attribute(),
          getAlignment(getTypeConverter(), kgenPtrType,
                       adaptor.getAlignmentAttr()));
      // Emit the constant using an initializer region.
      global.getBodyRegion().push_back(new Block);
      ImplicitLocOpBuilder b(op.getLoc(), op.getContext());
      b.setInsertionPointToStart(global.getBody());
      InterpreterMemoryConverter::MaterializationScope scope =
          imc.createScope();
      ErrorOr<Value> value = convertParameterToLLVM(
          b, *getTypeConverter(), /*imc=*/&imc, &scope, op.getValue());
      if (value.isError()) {
        b.emitError(value.getError());
        return failure();
      }
      LLVM::ReturnOp::create(b, value.get());

      // Insert the global into the module.
      symtab.insert(it->second = global);
    }

    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(
        op, opaquePtrType, FlatSymbolRefAttr::get(it->second.getSymNameAttr()));
    return success();
  }

private:
  /// The symbol table.
  SymbolTable &symtab;
  /// Uniqued constants.
  DenseMap<std::pair<TypedAttr, TypedAttr>, LLVM::GlobalOp> &constants;
  InterpreterMemoryConverter &imc;
};

//===----------------------------------------------------------------------===//
// ConvertExternPointerSymbol
//===----------------------------------------------------------------------===//

/// Lower external pointer symbol, this replaces the pointer with an external
/// global value.
struct ConvertExternPointerSymbol
    : public ConvertSymbolOpToLLVM<ExternPointerSymbolOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult matchAndRewrite(ExternPointerSymbolOp op,
                                ExternPointerSymbolOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    int64_t addressSpace = op.getResSymbol().getType().getAddrSpaceOrZero();
    Type resType = convertType(op.getResSymbol().getType().getElementType());
    unsigned align = getAlignment(
        getTypeConverter(), op.getResSymbol().getType(), op.getAlignmentAttr());

    b.clearInsertionPoint();
    auto global = LLVM::GlobalOp::create(
        b, op.getLoc(), resType, /*constant=*/false, LLVM::Linkage::External,
        cast<StringAttr>(op.getName()), /*value=*/nullptr, align, addressSpace,
        /*dso_local=*/true);
    symtab.insert(global);

    b.setInsertionPoint(op);
    b.replaceOpWithNewOp<LLVM::AddressOfOp>(
        op, LLVM::LLVMPointerType::get(getContext(), addressSpace),
        FlatSymbolRefAttr::get(global.getSymNameAttr()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerGlobalPOPToLLVMPass
//===----------------------------------------------------------------------===//

struct LowerGlobalPOPToLLVMPass
    : public KGEN::impl::LowerGlobalPOPToLLVMBase<LowerGlobalPOPToLLVMPass> {
  using LowerGlobalPOPToLLVMBase::LowerGlobalPOPToLLVMBase;

  void runOnOperation() override;
};

} // namespace

void LowerGlobalPOPToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<DebugInfo::DebugInfoDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set LLVM lowering options.
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  ErrorOr<const TargetLowering *> loweringOr =
      TargetLoweringRegistry::get().lookup(targetInfo.getTriple());
  const TargetLowering *lowering = loweringOr.isError() ? nullptr : *loweringOr;

  POPToLLVMTypeConverter typeConverter(targetInfo);
  InterpreterMemoryConverter imc(symtab, typeConverter);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  // Ops the target lowers in this pass are illegal here so they get converted.
  target.addDynamicallyLegalOp<POP::MaxOp, POP::MinOp, POP::AtomicRMWOp,
                               POP::AtomicCmpXchgOp, POP::CastOp, POP::FenceOp,
                               CallLLVMIntrinsicOp>([&](Operation *op) {
    return !(lowering && lowering->isLoweredInGlobalPOPPass(op));
  });

  // Convert external calls.
  target.addIllegalOp<GlobalAllocOp, ExternalCallOp, ExternPointerSymbolOp>();
  patterns.insert<ConvertPOPGlobalAlloc, ConvertExternPointerSymbol,
                  ConvertPOPAlignedAlloc, ConvertPOPAlignedFree,
                  ConvertPOPNoAliasPointerCast>(typeConverter, symtab);
  if (lowering)
    lowering->populateLowerGlobalPOPToLLVMPatterns(patterns, typeConverter,
                                                   symtab, targetInfo);
  populateLowerPOPExternalCallPatterns(patterns, typeConverter, symtab);

  // Convert global constants.
  DenseMap<std::pair<TypedAttr, TypedAttr>, LLVM::GlobalOp> constants;
  target.addIllegalOp<GlobalConstantOp>();
  patterns.insert<ConvertPOPGlobalConstant>(symtab, constants, typeConverter,
                                            imc);

  // pop.compiler.* are all illegal.
  target.addIllegalOp<CompilerGlobalLoadOp, CompilerGlobalStoreOp>();

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
