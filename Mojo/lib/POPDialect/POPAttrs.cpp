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

#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

void POPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Mojo/POPDialect/POPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// UnionAttr
//===----------------------------------------------------------------------===//

LogicalResult UnionAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                TypedAttr value, UnionType type) {
  // Skip verification for unresolved (parameterized) union types.
  // The types are not yet known, so we cannot verify membership.
  if (!type.isResolved())
    return success();
  auto it = llvm::find(type.getTypes(), value.getType());
  if (it != type.getTypes().end())
    return success();
  return emitError() << "value type " << value.getType()
                     << " is not a union element type of " << type;
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//
OptionalParseResult POP::ArrayType::parseValue(AsmParser &p,
                                               TypedAttr &value) const {
  if (failed(p.parseOptionalLSquare()))
    return std::nullopt;
  if (!getResolvedSize())
    return p.emitError(p.getCurrentLocation(),
                       "array attribute expected a concrete size");
  if (succeeded(p.parseOptionalRSquare())) {
    value = POP::ArrayAttr::get({}, *this);
    return mlir::success();
  }
  SmallVector<TypedAttr> values;
  if (failed(parseSequenceElements(p, values, *this)))
    return failure();
  value = POP::ArrayAttr::get(values, *this);
  return p.parseRSquare();
}

LogicalResult POP::ArrayType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto array = ::dyn_cast<POP::ArrayAttr>(value);
  if (!array)
    return failure();
  p << '[';
  llvm::interleaveComma(array.getValues(), p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << ']';
  return mlir::success();
}

/// The array attribute is a constant if all element values are constants.
bool POP::ArrayAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

LogicalResult
POP::ArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<TypedAttr> values, ArrayType type) {
  std::optional<int64_t> size = type.getResolvedSize();
  if (!size)
    return emitError() << "array attribute expected a concrete size";
  Type elementType = type.getElementType();
  if (*size != static_cast<int64_t>(values.size()))
    return emitError() << "array attribute type requires " << *size
                       << " elements but value has " << values.size();
  for (auto [idx, value] : llvm::enumerate(values))
    if (value.getType() != elementType)
      return emitError() << "array element #" << idx << " has type "
                         << value.getType() << " but expected " << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// IntLiteralAttr
//===----------------------------------------------------------------------===//

static ParseResult parseIntLiteral(AsmParser &p, IPInt &result) {
  APInt resultAP;
  OptionalParseResult parseResult = p.parseInteger(resultAP);
  if (!parseResult.has_value() || failed(*parseResult)) {
    result = {};
    return failure();
  }
  result = IPInt(resultAP);
  return success();
}

static void printIntLiteral(AsmPrinter &p, const IPInt &value) {
  p.getStream() << value;
}

//===----------------------------------------------------------------------===//
// IntLiteralAttr
//===----------------------------------------------------------------------===//

Type IntLiteralAttr::getType() const {
  return IntLiteralType::get(this->getContext());
}

bool IntLiteralAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// FloatLiteralAttr
//===----------------------------------------------------------------------===//

Attribute FloatLiteralAttr::parse(AsmParser &p, Type type) {
  if (p.parseLess())
    p.emitError(p.getCurrentLocation(), "expected '<' character");
  std::optional<IPRational> rational;
  FloatLiteralSpecialValuesAttr specialAttr;

  // Try to parse rational number first, then fall back to parsing special
  // value.
  APInt numerator;
  OptionalParseResult isRational = p.parseOptionalInteger(numerator);
  if (isRational.has_value() && !isRational.value()) {
    // MLIR's AsmParser doesn't have `parseSlash` or a more generic way to parse
    // literal strings/characters, so we will use the pipe "|" character
    // instead. https://github.com/modularml/modular/issues/23387
    APInt denominator;
    if (p.parseVerticalBar() || p.parseInteger(denominator)) {
      p.emitError(p.getCurrentLocation(),
                  "expected rational number with pipe in parens");
      return {};
    }
    if (denominator == 0) {
      p.emitError(p.getCurrentLocation(),
                  "expected rational number with non-zero denominator");
    }
    rational = IPRational(numerator, denominator);
    specialAttr = FloatLiteralSpecialValuesAttr::get(
        p.getContext(), FloatLiteralSpecialValues::Normal);
  } else if (p.parseCustomAttributeWithFallback(specialAttr)) {
    p.emitError(p.getCurrentLocation(),
                "expected FloatLiteralSpecialValueAttr");
    return {};
  }

  if (p.parseGreater()) {
    p.emitError(p.getCurrentLocation(), "expected '>' character");
  }

  return FloatLiteralAttr::get(p.getContext(), specialAttr, rational);
}

void FloatLiteralAttr::print(AsmPrinter &p) const {
  p.getStream() << "<";
  if (getSpecial().getValue() == FloatLiteralSpecialValues::Normal) {
    assert(getRational().has_value() &&
           "rational has value when special value is normal");
    p.getStream() << getRational().value();
  } else {
    p.getStream() << getSpecial().getValue();
  }
  p.getStream() << ">";
}

Type FloatLiteralAttr::getType() const {
  return FloatLiteralType::get(this->getContext());
}

bool FloatLiteralAttr::isConstant() const { return true; }

FloatLiteralAttr FloatLiteralAttr::get(MLIRContext *context,
                                       FloatLiteralSpecialValuesAttr input,
                                       std::optional<IPRational> value) {
  // Canonicalize special attributes to have no value.
  if (input.getValue() != FloatLiteralSpecialValues::Normal)
    value = {};
  return Base::get(context, input, value);
}

FloatLiteralAttr FloatLiteralAttr::get(MLIRContext *context, IPRational value) {
  return get(context,
             FloatLiteralSpecialValuesAttr::get(
                 context, FloatLiteralSpecialValues::Normal),
             value);
}

//===----------------------------------------------------------------------===//
// IntLiteralConvertAttr
//===----------------------------------------------------------------------===//

static ErrorOr<TypedAttr> foldIntLiteralConvert(TypedAttr input, Type outType) {
  auto literal = ::dyn_cast<IntLiteralAttr>(input);
  if (!literal)
    return Error("input must be IntLiteralAttr");

  const IPInt &inputIP = literal.getValue();
  const APInt &inputAP = inputIP.getAPInt();

  std::optional<KGEN::KGENDType> outDType;
  SIMDType outSIMDTy = dyn_cast<SIMDType>(outType);
  if (outSIMDTy)
    outDType = outSIMDTy.getResolvedDType();
  if (!outDType || outDType->isInvalid())
    return Error("Must have resolved valid DType");

  DTypeValue value = [&]() {
    if (outDType->isBool())
      return DTypeValue(!inputAP.isZero(), *outDType);

    // Floating-point conversions
    if (!outDType->isIntLike()) {
      APFloat outFPVal(*outDType->getFloatSemantics());
      outFPVal.convertFromAPInt(inputAP, /*IsSigned=*/true,
                                APFloat::rmNearestTiesToEven);
      return DTypeValue(outFPVal, *outDType);
    }

    // Note - we always sign-extend the input here to the right width to satisfy
    // DTypeValue. DTypeValue is the one that re-interprets the bits as unsigned
    // if the output DType type is unsigned.
    int outWidth = outDType->getWidthInBits();
    // Target-specific types return a width of -1. For now, assume the largest
    // target-specific type we support: 64. If we end up truncating down to 32
    // bits later, that's okay as we haven't lost any information.
    return DTypeValue(inputAP.sextOrTrunc(outWidth < 0 ? 64 : outWidth),
                      *outDType);
  }();

  auto scalarOut = SIMDAttr::get(value, SIMDType::get(1, outSIMDTy.getDType()));
  return SIMDSplatAttr::get(scalarOut, outSIMDTy);
}

TypedAttr IntLiteralConvertAttr::get(MLIRContext *ctx, Type type,
                                     TypedAttr input) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto result = foldIntLiteralConvert(input, type);
  if (!result.isError())
    return result.get();

  return Base::get(ctx, type, input);
}

bool IntLiteralConvertAttr::isConstant() const { return false; }

ErrorOrSuccess IntLiteralConvertAttr::validateForElaborator() const {
  auto result = foldIntLiteralConvert(getInput(), getType());
  assert(result.isError() && "Should be folded if present");
  return result.takeError();
}

//===----------------------------------------------------------------------===//
// IntLiteralBinAttr
//===----------------------------------------------------------------------===//

TypedAttr IntLiteralBinAttr::get(MLIRContext *ctx, TypedAttr lhs, TypedAttr rhs,
                                 IntLiteralBinKindAttr oper) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  IntLiteralAttr lAttr = sugarDynCastIfPresent<IntLiteralAttr>(lhs);
  IntLiteralAttr rAttr = sugarDynCastIfPresent<IntLiteralAttr>(rhs);
  if (!lAttr || !rAttr)
    return Base::get(ctx, lhs, rhs, oper);

  IPInt l = lAttr.getValue();
  IPInt r = rAttr.getValue();

  IPInt result;
  switch (oper.getValue()) {
  case IntLiteralBinKind::Add:
    result = l + r;
    break;
  case IntLiteralBinKind::Sub:
    result = l - r;
    break;
  case IntLiteralBinKind::Mul:
    result = l * r;
    break;
  case IntLiteralBinKind::FloorDiv: {
    IPInt zero(0);
    if (r == zero) { // x // 0 = 0.
      result = zero;
      break;
    }

    if ((l >= zero) == (r >= zero) || l % r == zero)
      result = l / r;
    else
      result = (l / r) - IPInt(1);
    break;
  }
  case IntLiteralBinKind::Mod: {
    // Python's mod:
    // The result sign matches the RHS sign.
    // If the signs match, the value is the same as: sign(abs(l) % abs(r)),
    // where sign is determined by the RHS sign. If the signs don't match, the
    // value is the same as: sign((abs(r) - (abs(l) % abs(r))) % abs(r)).
    IPInt zero(0);
    if (r == zero) { // x % 0 = 0.
      result = zero;
      break;
    }
    bool signMatch = (l >= zero) == (r >= zero);
    IPInt lAbs = l.abs();
    IPInt rAbs = r.abs();
    result = (lAbs % rAbs).abs();
    if (!signMatch && result != zero)
      result = rAbs - result;
    if (r < zero)
      result = zero - result;
    break;
  }
  case IntLiteralBinKind::Lshift:
    if (r < IPInt(0))
      result = IPInt(0);
    else
      result = l << r;
    break;
  case IntLiteralBinKind::Rshift:
    if (r < IPInt(0))
      result = IPInt(0);
    else
      result = l >> r;
    break;
  case IntLiteralBinKind::And:
    result = l & r;
    break;
  case IntLiteralBinKind::Or:
    result = l | r;
    break;
  case IntLiteralBinKind::Xor:
    result = l ^ r;
    break;
  case IntLiteralBinKind::Pow:
    // Exponentiating 0 by a negative amount is not defined for integers.
    if (l == IPInt(0) && r < IPInt(0))
      result = IPInt(0);
    else
      result = l.exponentiate(r);
    break;
  }

  return IntLiteralAttr::get(lAttr.getContext(), IPInt(result));
}

bool IntLiteralBinAttr::isConstant() const { return false; }

Type IntLiteralBinAttr::getType() const {
  return IntLiteralType::get(getContext());
}

//===----------------------------------------------------------------------===//
// IntLiteralCmpAttr
//===----------------------------------------------------------------------===//

TypedAttr IntLiteralCmpAttr::get(MLIRContext *ctx, IntLiteralCmpPredAttr pred,
                                 TypedAttr lhs, TypedAttr rhs) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  IntLiteralAttr lAttr = sugarDynCastIfPresent<IntLiteralAttr>(lhs);
  IntLiteralAttr rAttr = sugarDynCastIfPresent<IntLiteralAttr>(rhs);
  if (!lAttr || !rAttr)
    return Base::get(ctx, pred, lhs, rhs);

  IPInt l = lAttr.getValue();
  IPInt r = rAttr.getValue();
  switch (pred.getValue()) {
  case IntLiteralCmpPred::Eq:
    return BoolAttr::get(lAttr.getContext(), l == r);
  case IntLiteralCmpPred::Ne:
    return BoolAttr::get(lAttr.getContext(), l != r);
  case IntLiteralCmpPred::Lt:
    return BoolAttr::get(lAttr.getContext(), l < r);
  case IntLiteralCmpPred::Le:
    return BoolAttr::get(lAttr.getContext(), l <= r);
  case IntLiteralCmpPred::Gt:
    return BoolAttr::get(lAttr.getContext(), l > r);
  case IntLiteralCmpPred::Ge:
    return BoolAttr::get(lAttr.getContext(), l >= r);
  }
  llvm_unreachable("invalid cmp predicate");
}

bool IntLiteralCmpAttr::isConstant() const { return false; }

Type IntLiteralCmpAttr::getType() const {
  return IntegerType::get(getContext(), 1);
}

//===----------------------------------------------------------------------===//
// FloatLiteralConvertAttr
//===----------------------------------------------------------------------===//

/// Take an IPRational along with a specification for an output float type and
/// return the IEEE-style float bit string as an APInt.
static APInt
floatLiteralConvertGetBitstring(const IPRational &input,
                                const llvm::fltSemantics &fltSemantics) {
  unsigned totalLength = APFloat::getSizeInBits(fltSemantics);
  unsigned bias = 1 - APFloat::semanticsMinExponent(fltSemantics);
  unsigned exponentLength =
      llvm::Log2_64(APFloat::semanticsMaxExponent(fltSemantics) -
                    APFloat::semanticsMinExponent(fltSemantics) + 3);

  // Throughout this function I use “significand” to mean the float value
  // including the digit before the decimal, and “mantissa” to mean just the
  // part after the decimal, IE the bit pattern that is actually present in the
  // float value.  That's not technically correct, but it was helpful for me to
  // distinguish the two.
  unsigned mantissaLength = totalLength - exponentLength - 1;
  IPInt maxExponentZeroBias = (IPInt(1) << exponentLength) - 1;
  IPInt maxExponent = maxExponentZeroBias - bias;
  IPInt minExponent = IPInt(-1) * IPInt(bias - 1);

  // The maxSignificandIPIntLength is longer than the float mantissa bit width
  // to allow for:
  // * leading 0 in IPInt format
  // * most significant 1 bit that is removed in final encoding
  // * extra precision bits to ensure correct rounding
  unsigned maxSignificandIPIntRoundedLength = mantissaLength + 2;
  static const unsigned kSignificandRoundingLength = 3;
  unsigned maxSignificandIPIntLength =
      maxSignificandIPIntRoundedLength + kSignificandRoundingLength;

  // To support subnormal numbers (IE numbers with minimum exponent that have an
  // implicit leading 0 instead of implicit leading 1), we need to support lower
  // exponents during calculation.
  IPInt minCalculationExponent = minExponent - mantissaLength;

  if (input.getNumerator() == 0)
    return APInt(totalLength, 0);

  bool negativeSign = input.getNumerator() < 0;
  APInt signBits = APInt(totalLength, negativeSign ? 1 : 0);
  signBits = signBits << (totalLength - 1);

  IPInt initialNumerator = input.getNumerator().abs();
  const IPInt &denominator = input.getDenominator();
  IPInt significand = initialNumerator / denominator;
  IPInt remainder = initialNumerator % denominator;
  IPInt exponent = 0;
  bool exponentFinalized = false;
  if (significand > 0) {
    // The IPInt encoding of the number will have a leading 0 bit (because it is
    // positive), and the exponent when treating the most significant one bit is
    // one less than the number of bits representing the number with no leading
    // zeroes.
    exponent = significand.getAPInt().getBitWidth() - 2;
    exponentFinalized = true;
  }

  auto keepDoingLongDivision = [&]() -> bool {
    if (remainder == 0)
      return false;
    if (exponent < minCalculationExponent || exponent > maxExponent)
      return false;
    if (significand.getAPInt().getBitWidth() > maxSignificandIPIntLength)
      return false;
    return true;
  };

  // Do long division loop.
  while (keepDoingLongDivision()) {
    unsigned nBitsToShift = denominator.getAPInt().getBitWidth() -
                            remainder.getAPInt().getBitWidth();
    if (nBitsToShift == 0)
      nBitsToShift = 1;
    IPInt nCur = remainder << nBitsToShift;
    if (!exponentFinalized) {
      exponent = exponent - nBitsToShift;
    }
    IPInt quotient = nCur / denominator;
    remainder = nCur % denominator;
    if (quotient > 0)
      exponentFinalized = true;
    significand = (significand << nBitsToShift) + quotient;
  }

  // If we finished long division with “enough” rounding bits, but the remainder
  // is still not zero, it means that eventually there will be another 1 bit,
  // which would break a rounding tie.  Appending any further 1 bit will have
  // the same effect on rounding (no effect other than tie breaking), so we just
  // add the next one.
  if (remainder != 0)
    significand = (significand << 1) + 1;

  // Early return for obvious zero case because our later logic requires a
  // non-zero significand.
  if (significand == 0)
    return signBits;

  // Pad to mantissa length before performing rounding, etc.
  if (significand.getAPInt().getBitWidth() < maxSignificandIPIntLength) {
    significand = significand << (maxSignificandIPIntLength -
                                  significand.getAPInt().getBitWidth());
  }

  auto performRounding = [](IPInt &significand, IPInt &exponent,
                            unsigned maxSignificandIPIntRoundedLength) {
    APInt roundingBits = significand.getAPInt().extractBits(
        /*numBits=*/significand.getAPInt().getBitWidth() -
            maxSignificandIPIntRoundedLength,
        /*bitPosition=*/0);
    unsigned roundingBitsActualLength = roundingBits.getBitWidth();
    APInt roundingMidpoint = APInt(roundingBitsActualLength, 1)
                             << (roundingBitsActualLength - 1);
    // Truncate bits first.
    significand = significand >> roundingBitsActualLength;
    // Now that we've truncated, rounding either means doing nothing (for
    // round toward zero) or adding one to the significand representation
    // (for rounding away from zero). The default rounding mode for IEEE
    // floats is “round to nearest, ties to even”. It might be good to take
    // an option to do other rounding modes, but for now we just support the
    // default.
    if (roundingBits.ugt(roundingMidpoint))
      significand = significand + 1;
    else if (roundingBits == roundingMidpoint && significand % 2 == 1)
      significand = significand + 1;
    // If rounding up increased digit count, we need to convert that into a
    // larger exponent and re-truncate.
    if (significand.getAPInt().getBitWidth() >
        maxSignificandIPIntRoundedLength) {
      exponent = exponent + 1;
      significand = significand >> 1;
    }
  };

  // Do rounding now unless we are dealing with a subnormal number, which needs
  // some extra handling before rounding.
  if (exponent >= minExponent)
    performRounding(significand, exponent, maxSignificandIPIntRoundedLength);

  if (exponent > maxExponent) {
    // Return +/- infinity.
    APInt exponentOnes = APInt::getAllOnes(exponentLength);
    APInt exponentBits = APInt(totalLength, 0);
    exponentBits.insertBits(exponentOnes, mantissaLength);
    // Mantissa for infinity is zero.
    return signBits | exponentBits;
  }

  // Handle subnormal numbers, including zero values.  (I'm not sure whether
  // zero counts technically as a subnormal number, but it fits the subnormal
  // encoding.)
  if (exponent < minExponent) {
    // Below the minExponent we can still convert to subnormal numbers.
    // The subnormal range is tagged with minExponent - 1, but the exponent
    // value is effectively the same as minExponent. However, instead of an
    // implicit leading 1 before the decimal, there is a leading 0. So subnormal
    // numbers cover down to minExponent - mantissaWidth exponent, but
    // losing one bit of mantissa precision for each exponent lowering.
    if (exponent < minCalculationExponent) {
      // We could let this fall through and be handled by the shifting and bit
      // mangling, but at this point we know that every bit is zero except
      // (maybe) the sign.
      return signBits;
    }
    IPInt shiftBits = minExponent - exponent;
    IPInt shiftTag = IPInt(1) << (IPInt(significand.getAPInt().getBitWidth()) -
                                  IPInt(2) + shiftBits);
    // The significand is now
    // `01<correct-bit-pattern><at-least-one-extra-bit>`.
    significand = shiftTag + significand;
    exponent = minExponent - 1;
    // If rounding increases the exponent and carries to a new high bit, then we
    // end up at 1000... for the significand with minExponent, and thus the
    // right number.  Cool.
    performRounding(significand, exponent, maxSignificandIPIntRoundedLength);
  }

  // Whether or not the value was subnormal, the significand now has the bit
  // pattern `01<correct-bit-pattern><maybe-extra-bit-due-to-rounding>`.  So we
  // drop the leading 2 bits and the trailing extra bits to arrive at the final
  // bit pattern for the mantissa.

  unsigned extraSignificandBits =
      significand.getAPInt().getBitWidth() - (mantissaLength + 2);
  significand = significand >> extraSignificandBits;
  assert(significand.getAPInt().getBitWidth() == mantissaLength + 2 &&
         "proper mantissa bit length");
  APInt mantissaLowBits = significand.getAPInt().extractBits(
      /*numBits=*/mantissaLength,
      /*bitPosition=*/0);
  APInt mantissaBits = APInt(totalLength, 0);
  mantissaBits.insertBits(mantissaLowBits, /*bitPosition=*/0);

  // Floating point numbers encode the exponent as `bias + exponent`, so that
  // the result is always a natural number, where `bias + exponent = 0`
  // signifies subnormal (including zero) numbers, and all ones is the
  // exponent for infinity and the NAN values.
  exponent = exponent + bias;
  // Place the bits into an APInt at the appropriate place.
  APInt exponentBits = APInt(totalLength, 0);
  exponentBits.insertBits(exponent.getAPInt(), mantissaLength);

  // Combine pieces to get final bit string: <sign><exponent><mantissa>.
  return signBits | exponentBits | mantissaBits;
}

static ErrorOr<TypedAttr> foldFloatLiteralConvert(TypedAttr input,
                                                  Type outType) {
  auto inputLitAttr = sugarDynCastIfPresent<FloatLiteralAttr>(input);
  if (!inputLitAttr)
    return Error("input must be FloatLiteralAttr");

  const llvm::fltSemantics *fltSemantics = nullptr;

  // Handle !scalar<f32> aka !simd<f32, 1>
  auto simd = dyn_cast<SIMDType>(outType);
  if (auto dtype = simd.getResolvedDType())
    if (simd.getResolvedSize() && dtype->isFloat())
      fltSemantics = dtype->getFloatSemantics();

  if (!fltSemantics) {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << outType;
    return Error("float literal conversion: unsupported output type: " +
                 os.str());
  }

  APFloat resultValue(*fltSemantics, APFloat::uninitialized);
  switch (inputLitAttr.getSpecial().getValue()) {
  case FloatLiteralSpecialValues::Nan:
    // Set the payload to uint64_t::max to make the NaN fill all the low bits
    // to 1. This makes the NaN value aligned with the NaN values generated by
    // CUDA libraries.
    resultValue = APFloat::getNaN(
        *fltSemantics,
        /*Negative=*/false, /*payload=*/std::numeric_limits<uint64_t>::max());
    break;
  case FloatLiteralSpecialValues::Inf:
    resultValue = APFloat::getInf(*fltSemantics, /*negative=*/false);
    break;
  case FloatLiteralSpecialValues::NegInf:
    resultValue = APFloat::getInf(*fltSemantics, /*negative=*/true);
    break;
  case FloatLiteralSpecialValues::NegZero:
    resultValue = APFloat::getZero(*fltSemantics, /*negative=*/true);
    break;
  case FloatLiteralSpecialValues::Normal: {
    std::optional<IPRational> inRat = inputLitAttr.getRational();
    assert(inRat.has_value() && "normal FloatLiteral values have a rational");
    APInt floatBits =
        floatLiteralConvertGetBitstring(inRat.value(), *fltSemantics);
    resultValue = APFloat(*fltSemantics, floatBits);
    break;
  }
  }

  // Form a SIMDAttr for values of !simd type, splating the value out as needed.
  DTypeValue value(resultValue, *simd.getResolvedDType());
  SmallVector<DTypeValue> values(*simd.getResolvedSize(), value);
  return SIMDAttr::get(values, simd);
}

TypedAttr FloatLiteralConvertAttr::get(MLIRContext *ctx, Type type,
                                       TypedAttr input) {
  assert(!::isa<FloatLiteralType>(type) && !type.isF64() &&
         "should convert to SIMD type");

  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration simplifies things.
  auto errOrAttr = foldFloatLiteralConvert(input, type);
  if (errOrAttr.isError())
    return Base::get(ctx, type, input);
  return errOrAttr.get();
}

bool FloatLiteralConvertAttr::isConstant() const { return false; }

ErrorOrSuccess FloatLiteralConvertAttr::validateForElaborator() const {
  auto result = foldFloatLiteralConvert(getInput(), getType());
  assert(result.isError() && "Should be folded if present");
  return result.takeError();
}

//===----------------------------------------------------------------------===//
// IntToFloatLiteralAttr
//===----------------------------------------------------------------------===//

TypedAttr IntToFloatLiteralAttr::get(MLIRContext *ctx, TypedAttr input) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto inputAttr = sugarDynCastIfPresent<IntLiteralAttr>(input);
  if (!inputAttr)
    return Base::get(ctx, input);

  return FloatLiteralAttr::get(inputAttr.getContext(),
                               IPRational(inputAttr.getValue(), IPInt(1)));
}

bool IntToFloatLiteralAttr::isConstant() const { return false; }

Type IntToFloatLiteralAttr::getType() const {
  return FloatLiteralType::get(getContext());
}

//===----------------------------------------------------------------------===//
// FloatToIntLiteralAttr
//===----------------------------------------------------------------------===//

TypedAttr FloatToIntLiteralAttr::get(MLIRContext *ctx, TypedAttr input) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto inputAttr = sugarDynCastIfPresent<FloatLiteralAttr>(input);
  if (!inputAttr)
    return Base::get(ctx, input);

  IPInt result;
  switch (inputAttr.getSpecial().getValue()) {
  case FloatLiteralSpecialValues::Nan:
  case FloatLiteralSpecialValues::Inf:
  case FloatLiteralSpecialValues::NegInf:
  case FloatLiteralSpecialValues::NegZero:
    result = 0;
    break;
  case FloatLiteralSpecialValues::Normal:
    assert(inputAttr.getRational().has_value() &&
           "normal FloatLiterals have rational");
    result = inputAttr.getRational()->getNumerator() /
             inputAttr.getRational()->getDenominator();
    break;
  }
  return IntLiteralAttr::get(inputAttr.getContext(), result);
}

bool FloatToIntLiteralAttr::isConstant() const { return false; }

Type FloatToIntLiteralAttr::getType() const {
  return IntLiteralType::get(getContext());
}

//===----------------------------------------------------------------------===//
// FloatLiteralBinAttr
//===----------------------------------------------------------------------===//

static bool isNan(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Nan;
}
static bool isNegZero(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::NegZero;
}
static bool isInf(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Inf;
}
static bool isNegInf(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::NegInf;
}
static bool isNormal(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Normal;
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralAdd(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    if (isNegZero(rSpecial))
      return {FloatLiteralSpecialValues::Normal, 0};
    return {rSpecial, rhs};
  case FloatLiteralSpecialValues::Inf:
    if (isNegInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::Inf, 0};
  case FloatLiteralSpecialValues::NegInf:
    if (isInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::NegInf, 0};
  case FloatLiteralSpecialValues::Nan:
    return {FloatLiteralSpecialValues::Nan, 0};
  case FloatLiteralSpecialValues::Normal:
    if (isNormal(rSpecial))
      return {FloatLiteralSpecialValues::Normal, lhs + rhs};
    return floatLiteralAdd(rSpecial, lSpecial, rhs, lhs);
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralSub(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    // When adding zeroes, the signs are basically XORed, like with
    // multiplication.
    if (isNegZero(rSpecial))
      return {FloatLiteralSpecialValues::Normal, 0};
    if (isNormal(rSpecial) && rhs == 0)
      return {FloatLiteralSpecialValues::NegZero, 0};
    return floatLiteralSub(FloatLiteralSpecialValues::Normal, rSpecial, 0, rhs);
  case FloatLiteralSpecialValues::Inf:
    if (isInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::Inf, 0};
  case FloatLiteralSpecialValues::NegInf:
    if (isNegInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::NegInf, 0};
  case FloatLiteralSpecialValues::Nan:
    return {FloatLiteralSpecialValues::Nan, 0};
  case FloatLiteralSpecialValues::Normal:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::NegZero:
      return {lSpecial, lhs};
    case FloatLiteralSpecialValues::Inf:
      return {FloatLiteralSpecialValues::NegInf, 0};
    case FloatLiteralSpecialValues::NegInf:
      return {FloatLiteralSpecialValues::Inf, 0};
    case FloatLiteralSpecialValues::Nan:
      return {FloatLiteralSpecialValues::Nan, 0};
    case FloatLiteralSpecialValues::Normal:
      return {FloatLiteralSpecialValues::Normal, lhs - rhs};
    }
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

/// Helper for multiplication, to keep the special case matching table separate.
/// Assumes that at least one of lSpecial and rSpecial is non-normal.
static FloatLiteralSpecialValues
floatLiteralMulSpecialCases(const FloatLiteralSpecialValues &lSpecial,
                            const FloatLiteralSpecialValues &rSpecial,
                            const IPRational &lhs, const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::Inf:
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::Normal:
      if (rhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Inf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::NegInf;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Inf;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::NegInf;
      return FloatLiteralSpecialValues::Inf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::NegInf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Inf;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::NegInf;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Inf;
      return FloatLiteralSpecialValues::NegInf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Nan:
    return FloatLiteralSpecialValues::Nan;
  case FloatLiteralSpecialValues::Normal:
    // The case of both being normal is handled up front, so we don't worry
    // about it here.  Instead just recur with flipped operand order to handle
    // the case that LHS is normal.
    return floatLiteralMulSpecialCases(rSpecial, lSpecial, rhs, lhs);
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralMul(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  if (isNormal(lSpecial) && isNormal(rSpecial)) {
    IPRational ratResult = lhs * rhs;
    if (ratResult == 0 && ((lhs < 0) || (rhs < 0)))
      return {FloatLiteralSpecialValues::NegZero, {}};
    return {FloatLiteralSpecialValues::Normal, ratResult};
  }
  return {floatLiteralMulSpecialCases(lSpecial, rSpecial, lhs, rhs), 0};
}

/// Helper to separate the special case logic for division.  Assumes that at
/// least one of lSpecial and rSpecial is non-normal.
static FloatLiteralSpecialValues
floatLiteralDivSpecialCases(const FloatLiteralSpecialValues &lSpecial,
                            const FloatLiteralSpecialValues &rSpecial,
                            const IPRational &lhs, const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::NegZero;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Inf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
    case FloatLiteralSpecialValues::NegInf:
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::NegInf;
      return FloatLiteralSpecialValues::Inf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::NegInf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
    case FloatLiteralSpecialValues::NegInf:
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Inf;
      return FloatLiteralSpecialValues::NegInf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Nan:
    return FloatLiteralSpecialValues::Nan;
  case FloatLiteralSpecialValues::Normal:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Inf:
      if (lhs < 0)
        return FloatLiteralSpecialValues::NegZero;
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::NegInf:
      if (lhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      llvm_unreachable("double normal case handled above");
    }
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralTrueDiv(FloatLiteralSpecialValues lSpecial,
                    FloatLiteralSpecialValues rSpecial, IPRational lhs,
                    IPRational rhs) {
  if (isNormal(lSpecial) && isNormal(rSpecial)) {
    if (rhs == 0)
      return {FloatLiteralSpecialValues::Nan, 0};
    IPRational ratResult = lhs / rhs;
    if (lhs == 0 && rhs < 0)
      return {FloatLiteralSpecialValues::NegZero, 0};
    return {FloatLiteralSpecialValues::Normal, ratResult};
  };
  return {floatLiteralDivSpecialCases(lSpecial, rSpecial, lhs, rhs), 0};
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralFloorDiv(FloatLiteralSpecialValues lSpecial,
                     FloatLiteralSpecialValues rSpecial, IPRational lhs,
                     IPRational rhs) {
  auto truediv = floatLiteralTrueDiv(lSpecial, rSpecial, lhs, rhs);

  // Special values are propagated.
  if (!isNormal(truediv.first))
    return truediv;

  // Get the result as an integer value rounded towards zero.
  auto intval = truediv.second.getNumerator() / truediv.second.getDenominator();

  // Ensure this equality doesn't hit any implicit conversions.
  if (truediv.second >= 0 || truediv.second == intval)
    return {truediv.first, intval};
  return {truediv.first, intval - 1};
}

TypedAttr FloatLiteralBinAttr::get(MLIRContext *ctx, TypedAttr lhsA,
                                   TypedAttr rhsA,
                                   FloatLiteralBinKindAttr oper) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto lAttr = sugarDynCastIfPresent<FloatLiteralAttr>(lhsA);
  auto rAttr = sugarDynCastIfPresent<FloatLiteralAttr>(rhsA);
  if (!lAttr || !rAttr)
    return Base::get(ctx, lhsA, rhsA, oper);

  std::pair<FloatLiteralSpecialValues, IPRational> (*implFunc)(
      FloatLiteralSpecialValues, FloatLiteralSpecialValues, IPRational,
      IPRational) = nullptr;
  switch (oper.getValue()) {
  case FloatLiteralBinKind::Add:
    implFunc = floatLiteralAdd;
    break;
  case FloatLiteralBinKind::Sub:
    implFunc = floatLiteralSub;
    break;
  case FloatLiteralBinKind::Mul:
    implFunc = floatLiteralMul;
    break;
  case FloatLiteralBinKind::TrueDiv:
    implFunc = floatLiteralTrueDiv;
    break;
  case FloatLiteralBinKind::FloorDiv:
    implFunc = floatLiteralFloorDiv;
    break;
  }
  assert(implFunc && "unknown FloatLiteralBinop type");

  FloatLiteralSpecialValues lSpecial = lAttr.getSpecial().getValue();
  IPRational lhs;
  if (isNormal(lSpecial)) {
    assert(lAttr.getRational().has_value() &&
           "rational has value when special value is normal");
    lhs = lAttr.getRational().value();
  }
  FloatLiteralSpecialValues rSpecial = rAttr.getSpecial().getValue();
  IPRational rhs;
  if (isNormal(rSpecial)) {
    assert(rAttr.getRational().has_value() &&
           "rational has value when special value is normal");
    rhs = rAttr.getRational().value();
  }

  auto [result, rational] = implFunc(lSpecial, rSpecial, lhs, rhs);
  return FloatLiteralAttr::get(
      lAttr.getContext(),
      FloatLiteralSpecialValuesAttr::get(lAttr.getContext(), result), rational);
}

bool FloatLiteralBinAttr::isConstant() const { return false; }

Type FloatLiteralBinAttr::getType() const {
  return FloatLiteralType::get(getContext());
}

//===----------------------------------------------------------------------===//
// FloatLiteralCmpAttr
//===----------------------------------------------------------------------===//

/// Helper for float literal comparison.  The lhs/rhs values are only meaningful
/// when lSpecial/rSpecial are normal.
static bool floatLiteralCmpHelper(const FloatLiteralCmpPred &pred,
                                  const FloatLiteralSpecialValues &lSpecial,
                                  const FloatLiteralSpecialValues &rSpecial,
                                  const IPRational &lhs,
                                  const IPRational &rhs) {
  switch (pred) {
  case FloatLiteralCmpPred::Eq:
    if (lSpecial == rSpecial) {
      if (isNormal(lSpecial))
        return lhs == rhs;
      return !isNan(lSpecial);
    }
    // Python treats -0 and 0 as equal.
    if (isNegZero(lSpecial) && isNormal(rSpecial) && rhs == 0)
      return true;
    if (isNegZero(rSpecial) && isNormal(lSpecial) && lhs == 0)
      return true;
    return false;
  case FloatLiteralCmpPred::Ne:
    return !floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                  lhs, rhs);
  case FloatLiteralCmpPred::Lt:
    switch (lSpecial) {
    case FloatLiteralSpecialValues::Normal:
      switch (rSpecial) {
      case FloatLiteralSpecialValues::Normal:
        return lhs < rhs;
      case FloatLiteralSpecialValues::Inf:
        return true;
      case FloatLiteralSpecialValues::NegZero:
        return lhs < 0;
      default:
        return false;
      }
    case FloatLiteralSpecialValues::NegZero:
      switch (rSpecial) {
      case FloatLiteralSpecialValues::Normal:
        // This would be <=, but Python treats -0 as equal to 0, so the RHS
        // needs to be strictly greater than positive zero.
        return IPRational(0) < rhs;
      case FloatLiteralSpecialValues::Inf:
        return true;
      default:
        return false;
      }
    case FloatLiteralSpecialValues::Inf:
    case FloatLiteralSpecialValues::Nan:
      return false;
    case FloatLiteralSpecialValues::NegInf:
      return !isNan(rSpecial) && !isNegInf(rSpecial);
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralCmpPred::Le:
    return floatLiteralCmpHelper(FloatLiteralCmpPred::Lt, lSpecial, rSpecial,
                                 lhs, rhs) ||
           floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                 lhs, rhs);
  case FloatLiteralCmpPred::Gt:
    if (isNan(lSpecial) || isNan(rSpecial))
      return false;
    return !floatLiteralCmpHelper(FloatLiteralCmpPred::Le, lSpecial, rSpecial,
                                  lhs, rhs);
  case FloatLiteralCmpPred::Ge:
    return floatLiteralCmpHelper(FloatLiteralCmpPred::Gt, lSpecial, rSpecial,
                                 lhs, rhs) ||
           floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                 lhs, rhs);
  }
  llvm_unreachable("invalid cmp predicate");
}

TypedAttr FloatLiteralCmpAttr::get(MLIRContext *ctx,
                                   FloatLiteralCmpPredAttr pred, TypedAttr lhsA,
                                   TypedAttr rhsA) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto lAttr = sugarDynCastIfPresent<FloatLiteralAttr>(lhsA);
  auto rAttr = sugarDynCastIfPresent<FloatLiteralAttr>(rhsA);
  if (!lAttr || !rAttr)
    return Base::get(ctx, pred, lhsA, rhsA);

  FloatLiteralSpecialValues lSpecial = lAttr.getSpecial().getValue();
  FloatLiteralSpecialValues rSpecial = rAttr.getSpecial().getValue();
  IPRational lhs;
  IPRational rhs;
  if (isNormal(lSpecial)) {
    assert(lAttr.getRational().has_value() &&
           "rational does not have a value when special value is normal");
    lhs = lAttr.getRational().value();
  }
  if (isNormal(rSpecial)) {
    assert(rAttr.getRational().has_value() &&
           "rational does not have a value when special value is normal");
    rhs = rAttr.getRational().value();
  }
  return BoolAttr::get(
      lAttr.getContext(),
      floatLiteralCmpHelper(pred.getValue(), lSpecial, rSpecial, lhs, rhs));
}

bool FloatLiteralCmpAttr::isConstant() const { return false; }

Type FloatLiteralCmpAttr::getType() const {
  return IntegerType::get(getContext(), 1);
}

//===----------------------------------------------------------------------===//
// FloatLiteralIsaAttr
//===----------------------------------------------------------------------===//

TypedAttr FloatLiteralIsaAttr::get(MLIRContext *ctx,
                                   FloatLiteralSpecialValuesAttr kind,
                                   TypedAttr input) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration simplifies things.
  if (auto inputAttr = sugarDynCastIfPresent<FloatLiteralAttr>(input))
    return BoolAttr::get(ctx, inputAttr.getSpecial() == kind);

  return Base::get(ctx, kind, input);
}

bool FloatLiteralIsaAttr::isConstant() const { return false; }

Type FloatLiteralIsaAttr::getType() const {
  return IntegerType::get(getContext(), 1);
}

//===----------------------------------------------------------------------===//
// StringSizeAttr
//===----------------------------------------------------------------------===//

TypedAttr StringSizeAttr::get(MLIRContext *ctx, TypedAttr str) {
  // If input is a string literal, we can fold this
  if (auto strAttr = ::dyn_cast_or_null<StringAttr>(str))
    return IntegerAttr::get(IndexType::get(ctx), strAttr.getValue().size());

  return Base::get(ctx, str);
}

bool StringSizeAttr::isConstant() const { return false; }

Type StringSizeAttr::getType() const { return IndexType::get(getContext()); }

//===----------------------------------------------------------------------===//
// StringConcatAttr
//===----------------------------------------------------------------------===//

TypedAttr StringConcatAttr::get(MLIRContext *ctx, TypedAttr lhs,
                                TypedAttr rhs) {
  // If both inputs are string literals, we can fold this
  if (auto lhsStr = ::dyn_cast_or_null<StringAttr>(lhs))
    if (auto rhsStr = ::dyn_cast_or_null<StringAttr>(rhs)) {
      return StringAttr::get(lhsStr.getValue() + rhsStr.getValue(),
                             lhsStr.getType());
    }

  return Base::get(ctx, lhs, rhs);
}

bool StringConcatAttr::isConstant() const { return false; }

Type StringConcatAttr::getType() const { return StringType::get(getContext()); }

//===----------------------------------------------------------------------===//
// CastAttr
//===----------------------------------------------------------------------===//

TypedAttr CastAttr::get(MLIRContext *ctx, TypedAttr arg, Type out_type) {
  // Fold if possible
  if (auto fold = POP::foldCast(arg, cast<SIMDType>(out_type),
                                cast<SIMDType>(arg.getType()),
                                cast<SIMDType>(out_type))) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;
  }
  return Base::get(ctx, arg, out_type);
}

TypedAttr CastAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, TypedAttr value,
                               Type out_type) {
  if (failed(verify(emitError, value, out_type)))
    return {};
  return CastAttr::get(context, value, out_type);
}

bool CastAttr::isConstant() const { return false; }

LogicalResult CastAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               TypedAttr value, Type out_type) {
  if (!isa<SIMDType>(value.getType()) || !isa<SIMDType>(out_type))
    return emitError()
           << "Invalid cast operands: input and output must be SIMD types";
  return success();
}

//===----------------------------------------------------------------------===//
// DTypeToUI8Attr
//===----------------------------------------------------------------------===//

TypedAttr DTypeToUI8Attr::get(MLIRContext *ctx, TypedAttr value) {
  // Fold into an IntegerAttr if the DType value is known
  if (auto constDTy = dyn_cast<DTypeConstantAttr>(value))
    return IntegerAttr::get(IntegerType::get(ctx, 8, IntegerType::Unsigned),
                            constDTy.getDType().getValue());
  return Base::get(ctx, value);
}

TypedAttr
DTypeToUI8Attr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                           MLIRContext *context, TypedAttr value) {
  if (failed(verify(emitError, value)))
    return {};
  return DTypeToUI8Attr::get(context, value);
}

bool DTypeToUI8Attr::isConstant() const { return false; }

Type DTypeToUI8Attr::getType() const {
  return IntegerType::get(getContext(), 8, IntegerType::Unsigned);
}

LogicalResult
DTypeToUI8Attr::verify(function_ref<InFlightDiagnostic()> emitError,
                       TypedAttr value) {
  if (!isa<DTypeType>(value.getType()))
    return emitError() << "DType type " << value.getType() << " is invalid";
  return success();
}

//===----------------------------------------------------------------------===//
// DTypeFromUI8Attr
//===----------------------------------------------------------------------===//

TypedAttr DTypeFromUI8Attr::get(MLIRContext *ctx, TypedAttr value) {
  // Fold into an DTypeConstantAttr if the ui8 value is known
  if (auto constInt = dyn_cast<IntLiteralAttr>(value))
    return DTypeConstantAttr::get(
        ctx, (KGENDType)constInt.getValue().getAPInt().getSExtValue());
  if (auto constInt = dyn_cast<IntegerAttr>(value))
    return DTypeConstantAttr::get(
        ctx, (KGENDType)constInt.getValue().getSExtValue());
  return Base::get(ctx, value);
}

TypedAttr
DTypeFromUI8Attr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             MLIRContext *context, TypedAttr value) {
  if (failed(verify(emitError, value)))
    return {};
  return DTypeFromUI8Attr::get(context, value);
}

bool DTypeFromUI8Attr::isConstant() const { return false; }

Type DTypeFromUI8Attr::getType() const { return DTypeType::get(getContext()); }

LogicalResult
DTypeFromUI8Attr::verify(function_ref<InFlightDiagnostic()> emitError,
                         TypedAttr value) {
  auto intTy = dyn_cast<IntegerType>(value.getType());
  if (!intTy || !intTy.isUnsignedInteger(8))
    return emitError() << "Input type " << value.getType()
                       << " is invalid: must be ui8";
  return success();
}

//===----------------------------------------------------------------------===//
// SIMD Unary Operation Attrs
//===----------------------------------------------------------------------===//

TypedAttr SIMDNegAttr::get(MLIRContext *ctx, TypedAttr operand) {
  // Fold if possible
  if (auto fold = foldSIMDOp(
          {operand}, [](APSInt operand) { return -operand; },
          [](APFloat operand) { return -operand; },
          [](bool operand) { return !operand; })) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;
  }
  return Base::get(ctx, operand);
}

//===----------------------------------------------------------------------===//
// SIMDFloorAttr
//===----------------------------------------------------------------------===//

TypedAttr SIMDFloorAttr::get(MLIRContext *ctx, TypedAttr operand) {
  // Fold if possible.
  if (auto fold = foldSIMDOp(
          {operand}, [](APSInt operand) { return operand; },
          [](APFloat operand) {
            operand.roundToIntegral(APFloat::rmTowardNegative);
            return operand;
          },
          [](bool operand) { return operand; })) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;
  }
  return Base::get(ctx, operand);
}

//===----------------------------------------------------------------------===//
// SIMDCeilAttr
//===----------------------------------------------------------------------===//

TypedAttr SIMDCeilAttr::get(MLIRContext *ctx, TypedAttr operand) {
  // Fold if possible.
  if (auto fold = foldSIMDOp(
          {operand}, [](APSInt operand) { return operand; },
          [](APFloat operand) {
            operand.roundToIntegral(APFloat::rmTowardPositive);
            return operand;
          },
          [](bool operand) { return operand; })) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;
  }
  return Base::get(ctx, operand);
}

//===----------------------------------------------------------------------===//
// SIMDTruncAttr
//===----------------------------------------------------------------------===//

TypedAttr SIMDTruncAttr::get(MLIRContext *ctx, TypedAttr operand) {
  // Fold if possible.
  if (auto fold = foldSIMDOp(
          {operand}, [](APSInt operand) { return operand; },
          [](APFloat operand) {
            operand.roundToIntegral(APFloat::rmTowardZero);
            return operand;
          },
          [](bool operand) { return operand; })) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;
  }
  return Base::get(ctx, operand);
}

//===----------------------------------------------------------------------===//
// SIMD Binary Operation Attrs
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SIMDSubAttr
//===----------------------------------------------------------------------===//

TypedAttr SIMDSubAttr::get(MLIRContext *ctx, TypedAttr lhs, TypedAttr rhs) {
  auto simdType = cast<SIMDType>(lhs.getType());
  // ParamOperatorAttr does not have a sub poc, only safe to turn the sub into
  // lhs + (-rhs) for signed int dtype.
  //
  // TODO: the gap should be filled, but it should be enough for the purpose of
  // int/simd unification.
  if (simdType.getResolvedDType() && simdType.getResolvedDType()->isSInt())
    return ParamOperatorAttr::getSub(lhs, rhs);

  // Fold if possible
  if (auto fold = foldSIMDOp(
          {lhs, rhs}, [](APSInt lhs, APSInt rhs) { return lhs - rhs; },
          [](APFloat lhs, APFloat rhs) { return lhs - rhs; },
          [](bool lhs, bool rhs) { return (bool)(lhs ^ rhs); })) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;
  }
  return Base::get(ctx, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// SIMDAbsAttr
//===----------------------------------------------------------------------===//

TypedAttr SIMDAbsAttr::get(MLIRContext *ctx, TypedAttr operand) {
  Attribute operands[] = {operand};
  if (TypedAttr folded = tryFoldAttr(operands, foldSIMDAbs))
    return folded;
  return Base::get(ctx, operand);
}

//===----------------------------------------------------------------------===//
// SIMDRoundAttr
//===----------------------------------------------------------------------===//

TypedAttr SIMDRoundAttr::get(MLIRContext *ctx, TypedAttr operand) {
  // Fold if possible.
  if (auto fold = POP::foldSIMDRound(operand,
                                     /*targetInfo=*/{})) {
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;
  }
  return Base::get(ctx, operand);
}

//===----------------------------------------------------------------------===//
// SIMDReduceOrAttr
//===----------------------------------------------------------------------===//

OpFoldResult SIMDReduceOrAttr::fold(TypedAttr vector, SIMDType outType) {
  return foldSIMDReduceOr(/*vector=*/{}, vector, outType);
}

//===----------------------------------------------------------------------===//
// SIMDReduceAndAttr
//===----------------------------------------------------------------------===//

OpFoldResult SIMDReduceAndAttr::fold(TypedAttr vector, SIMDType outType) {
  return foldSIMDReduceAnd(/*vector=*/{}, vector, outType);
}

//===----------------------------------------------------------------------===//
// SIMDShlAttr
//===----------------------------------------------------------------------===//

TypedAttr SIMDShlAttr::get(MLIRContext *ctx, TypedAttr value, TypedAttr shft) {
  // TODO: support mismatched operand types in `ParamOperatorAttr`
  if (value.getType() == shft.getType())
    return ParamOperatorAttr::get(POC::Shl, {value, shft});

  Attribute operands[] = {value, shft};
  if (TypedAttr folded = tryFoldAttr(operands, foldSIMDShl))
    return folded;
  return Base::get(ctx, value, shft);
}

//===----------------------------------------------------------------------===//
// SIMDShrAttr
//===----------------------------------------------------------------------===//

TypedAttr SIMDShrAttr::get(MLIRContext *ctx, TypedAttr value, TypedAttr shft) {
  // TODO: support mismatched operand types in `ParamOperatorAttr`
  if (value.getType() == shft.getType())
    return ParamOperatorAttr::get(POC::Shr, {value, shft});

  Attribute operands[] = {value, shft};
  if (TypedAttr folded = tryFoldAttr(operands, foldSIMDShr))
    return folded;
  return Base::get(ctx, value, shft);
}

//===----------------------------------------------------------------------===//
// VariadicToArrayAttr
//===----------------------------------------------------------------------===//

bool VariadicToArrayAttr::isConstant() const { return false; }

Type VariadicToArrayAttr::getElementType() const {
  return cast<ParamListType>(getVariadic().getType()).getElementType();
}

POP::ArrayType VariadicToArrayAttr::getType() const {
  auto size = ParamListSizeAttr::get(getVariadic());
  return ArrayType::get(size, getElementType());
}

TypedAttr VariadicToArrayAttr::get(TypedAttr variadic) {
  auto vaAttr = sugarDynCast<ParamListAttr>(variadic);
  if (vaAttr) {
    ArrayRef<TypedAttr> values = vaAttr.getValues();
    auto arrayType =
        ArrayType::get(values.size(), vaAttr.getType().getElementType());
    return POP::ArrayAttr::get(values, arrayType);
  }
  return Base::get(variadic.getContext(), variadic);
}

LogicalResult
VariadicToArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            TypedAttr variadic) {
  auto variadicType = dyn_cast<ParamListType>(variadic.getType());
  if (!variadicType)
    return emitError() << "expected a 'variadic' type for the input, got: "
                       << variadic.getType();
  return success();
}

TypedAttr VariadicToArrayAttr::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, TypedAttr variadic) {
  if (failed(verify(emitError, variadic)))
    return {};
  return get(variadic);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Mojo/POPDialect/POPAttrs.cpp.inc"
#include "Mojo/POPDialect/POPEnums.cpp.inc"
