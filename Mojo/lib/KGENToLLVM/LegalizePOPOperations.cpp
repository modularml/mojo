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
// Legalizes operations on types that LLVM and POP->LLVM don't support.
//
// Some operations (e.g., FP8 arithmetic) have no direct LLVM codegen path and
// aren't emulated during POP → LLVM lowering. This pass handles them by:
//   1. Casting operands to a wider, supported type
//   2. Performing the operation in that type
//   3. Casting the result back to the original type
//
// Example 0: Negating an f8e5m2 value
//
//   Before (illegal):
//     %res = pop.neg %input : !kgen.scalar<f8e5m2>
//
//   After (legal):
//     %0 = pop.cast %input : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
//     %1 = pop.neg %0 : !kgen.scalar<f16>
//     %2 = pop.cast %1 : !kgen.scalar<f16> to !kgen.scalar<f32>
//     %res = pop.cast %2 : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
//
// In this example the target has no direct conversion from f16 to f8e5m2, but
// POP->LLVM implements the f32->f8e5m2 conversion using special target
// instructions and f16->f32 is natively supported by fpext, therefore multiple
// conversions are required to convert the result back to f8e5m2.
//
// Example 1: Converting f8e5m2 to bf16
//    Before (illegal):
//      %res = pop.cast %input : !kgen.scalar<f8e5m2> to !kgen.scalar<bf16>
//
//    After (legal):
//      %0 = pop.cast %input : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
//      %1 = pop.cast %0 : !kgen.scalar<f16> to !kgen.scalar<f32>
//      %res = pop.cast %1 : !kgen.scalar<f32> to !kgen.scalar<bf16>
//
// In this example, there's no direct conversion from f8e5m2 to bf16, but it can
// be done with a sequence of conversions f8e5m2->f16->f32->bf16
//
// Scope:
//   - Unary and binary ops on FP8 or narrower types
//   - CastOps with no direct target-supported conversion
//
// TODO:
//  - Query legal conversions from `ConvertPOPCast`
//  - Revisit algorithm to find shortest/most cost effective sequence of
//    conversions
//
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/KGENPasses.h"

#include "LLVMLoweringUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/MDialect/MAttrs.h"
#include "Target/TargetLowering.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_LEGALIZEPOPOPERATIONS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

/// Return KGENDType that corresponds to the scalar type of the given \p type.
static KGENDType getScalarKGENDType(Type type) {
  if (auto simd = dyn_cast<SIMDType>(type)) {
    std::optional<KGENDType> dtype = simd.getResolvedDType();
    assert(dtype && "dtype must be resolved at this point");
    return *dtype;
  }
  return KGENDType();
}

/// Return similar type to \p type, but with scalar type of \p dtype
static Type convertKGENDTypeToType(KGENDType dtype, Type type) {
  if (auto simd = dyn_cast<SIMDType>(type))
    return SIMDType::get(simd.getContext(), *simd.getResolvedSize(), dtype);
  // TODO: add support for other types
  llvm_unreachable("unsupported type");
}

namespace {

class LegalizePOPOperations
    : public KGEN::impl::LegalizePOPOperationsBase<LegalizePOPOperations> {
public:
  void runOnOperation() override;

private:
  // A map of types for which the current target has implemented conversions:
  //              {input-type -> {output-types...}}
  // For example: {f8e4m3fn -> {f16}}
  // TODO: Query this information from `ConvertPOPCast`
  DenseMap<KGENDType, llvm::SetVector<KGENDType>> legalConversions;

  TargetInfoAttr target;
  const TargetLowering *lowering = nullptr;

  /// Initialize map of known conversions for a target
  void initializeTargetLegalConversions();

  /// Return true if legalization of the operation succeeded.
  LogicalResult legalizeOperation(Operation *op);

  /// Return 'success' if legalization has to be done for the operation. Return
  /// false otherwise. An operation requires legalization if lowering the
  /// operation to LLVM with converted types will generate invalid IR. For now
  /// legalize all arithmetic operations that operate on unsupported by LLVM
  /// types, such as F8 types.
  bool operationRequiresLegalization(Operation *op) const;

  /// Return true if lowering of the conversion is supported
  bool conversionRequiresLegalization(CastOp castOp) const;

  /// Return 'success' if legalization has to be done for the conversion.
  LogicalResult legalizeConversion(CastOp castOp);

  /// Return true if the type requires legalization on the current target.
  bool typeRequiresLegalization(KGENDType dtype) const;
  bool typeRequiresLegalization(Type type) const;

  /// Helper function to find a sequence of available conversions from \p
  /// fromType to \p toType That is, this function returns a vector {type0,
  /// type1, ..., typeN} such that fromType -> type0 -> type1 -> ... -> typeN ->
  /// toType
  SmallVector<Type> findConversionSequence(Type fromType, Type toType) const;
};

//===----------------------------------------------------------------------===//
// initializeTargetLegalConversions
//===----------------------------------------------------------------------===//

void LegalizePOPOperations::initializeTargetLegalConversions() {
  auto addConversion = [&](KGENDType fromDType, KGENDType toDType) {
    legalConversions[fromDType].insert(toDType);
  };

  // Helper function to add all supported conversions that can be represented
  // via `fpext` and `fptrunc` in LLVM IR. These are common to every target.
  auto addLLVMNativeConversions = [&]() {
    // List of all supported conversions from f16
    addConversion(KGENDType::f16, KGENDType::f32);
    addConversion(KGENDType::f16, KGENDType::f64);

    // List of all supported conversions from bf16
    addConversion(KGENDType::bf16, KGENDType::f32);
    addConversion(KGENDType::bf16, KGENDType::f64);

    // List of all supported conversions from f32
    addConversion(KGENDType::f32, KGENDType::f16);
    addConversion(KGENDType::f32, KGENDType::bf16);
    addConversion(KGENDType::f32, KGENDType::f64);

    // List of all supported conversions from f64
    addConversion(KGENDType::f64, KGENDType::f16);
    addConversion(KGENDType::f64, KGENDType::bf16);
    addConversion(KGENDType::f64, KGENDType::f32);
  };

  // The native fpext/fptrunc conversions are supported on every target.
  addLLVMNativeConversions();

  // Let the current target contribute its own (e.g. fp8) conversions.
  if (lowering)
    lowering->populateLegalConversions(target, addConversion);

  // TODO: This list has to be complete for all supported types, but ideally it
  // should be taken from lowering of the POP::CastOp.
}

//===----------------------------------------------------------------------===//
// typeRequiresLegalization
//===----------------------------------------------------------------------===//

/// Return true if the type requires legalization on the current target.
bool LegalizePOPOperations::typeRequiresLegalization(KGENDType dtype) const {
  if (dtype.isInvalid())
    return false;
  // Assume that any floating point type below 8 bits are not supported by LLVM
  if (dtype.isFloat() && dtype.getWidthInBits() <= 8)
    return true;
  return lowering && lowering->typeNeedsExtraLegalization(dtype, target);
}

bool LegalizePOPOperations::typeRequiresLegalization(Type type) const {
  return typeRequiresLegalization(getScalarKGENDType(type));
}

//===----------------------------------------------------------------------===//
// operationRequiresLegalization
//===----------------------------------------------------------------------===//

bool LegalizePOPOperations::operationRequiresLegalization(Operation *op) const {
  // If operands have supported types, do not need to do anything extra with the
  // operation.
  for (auto operand : op->getOperands())
    if (!typeRequiresLegalization(operand.getType()))
      return false;

  return isa<NegOp, AddOp, SubOp, MulOp, DivOp, RemOp, MaxOp, MinOp, FMAOp>(op);
}

//===----------------------------------------------------------------------===//
// conversionRequiresLegalization
//===----------------------------------------------------------------------===//

bool LegalizePOPOperations::conversionRequiresLegalization(
    CastOp castOp) const {
  Type fromType = castOp.getInput().getType();
  Type toType = castOp.getOutput().getType();
  if (fromType == toType) {
    // should be folded instead.
    return false;
  }

  KGENDType fromDType = getScalarKGENDType(fromType);
  KGENDType toDType = getScalarKGENDType(toType);
  if (!typeRequiresLegalization(fromType) &&
      !typeRequiresLegalization(toType) &&
      (!fromDType.isFloat() || !toDType.isFloat() ||
       fromDType.getWidthInBits() != toDType.getWidthInBits())) {
    // No need to legalize conversion if it's natively supported by LLVM.
    // Also need to legalize it if size of to/from types is the same, because
    // `fpext` and `fptrunc` instructions do assert that size of `from type` is
    // different to size to `to type.
    return false;
  }

  // Some casts are handled by a later target-specific pass, so they don't need
  // legalization here.
  if (lowering && lowering->castHandledByLaterPass(fromDType, toDType, target))
    return false;

  // If there's no direct conversion available, then legalization is required.
  auto typeConversionsIt = legalConversions.find(fromDType);
  if (typeConversionsIt == legalConversions.end())
    return true;

  return !typeConversionsIt->second.contains(toDType);
}

//===----------------------------------------------------------------------===//
// findConversionSequence
//===----------------------------------------------------------------------===//

SmallVector<Type>
LegalizePOPOperations::findConversionSequence(Type fromType,
                                              Type toType) const {
  KGENDType fromDType = getScalarKGENDType(fromType);
  KGENDType toDType = getScalarKGENDType(toType);

  SmallVector<Type> types;
  DenseSet<KGENDType> visited;
  // Recursively find a sequence of available conversions that can be used to
  // convert fromType to toType
  // TODO: Revisit algorithm if shortest/most cost effective sequence is
  // required.
  std::function<bool(KGENDType, KGENDType)> walker =
      [this, &walker, &types, toType, &visited](KGENDType fromDType,
                                                KGENDType toDType) -> bool {
    // To avoid possible cycles, do not visit the same type twice
    if (!visited.insert(fromDType).second)
      return false;

    auto fromTypeIt = legalConversions.find(fromDType);
    if (fromTypeIt == legalConversions.end())
      return false;

    size_t fromTypeSize = fromDType.getWidthInBits(target);
    size_t toTypeSize = toDType.getWidthInBits(target);
    bool isUpconversion = fromTypeSize < toTypeSize;

    for (KGENDType commonDType : fromTypeIt->second) {
      // Do not try to select commonType if original conversion is:
      // - upconversion and common type is smaller or equal than the fromType
      // - downconversion and common type is smaller or equal than the toType
      // otherwise this will lead to precision losses.
      // TODO: Revisit "equal than" condition: it might be possible to use that
      // type if it won't introduce precision losses
      size_t commonTypeSize = commonDType.getWidthInBits(target);
      if ((commonTypeSize <= fromTypeSize && isUpconversion) ||
          (commonTypeSize <= toTypeSize && !isUpconversion))
        continue;

      auto commonTypeIt = legalConversions.find(commonDType);
      if (commonTypeIt == legalConversions.end())
        continue;

      // If direct conversion from commonType to toType exists, we can safely
      // use it, otherwise try to find a type to convert from commonType to
      // toType
      // TODO: Return the smallest available type
      if (commonTypeIt->second.contains(toDType) ||
          walker(commonDType, toDType)) {
        types.push_back(convertKGENDTypeToType(commonDType, toType));
        return true;
      }
    }
    return false;
  };
  (void)walker(fromDType, toDType);
  return SmallVector<Type>(llvm::reverse(types));
}

//===----------------------------------------------------------------------===//
// legalizeConversion
//===----------------------------------------------------------------------===//

LogicalResult LegalizePOPOperations::legalizeConversion(CastOp castOp) {
  assert(conversionRequiresLegalization(castOp) && "legalization not required");
  ImplicitLocOpBuilder b(castOp.getLoc(), castOp);

  Type type = castOp.getType();

  SmallVector<Type> commonTypes =
      findConversionSequence(castOp.getInput().getType(), type);
  if (commonTypes.empty()) {
    KGENDType fromDType = getScalarKGENDType(castOp.getInput().getType());
    KGENDType toDType = getScalarKGENDType(type);
    return castOp->emitError("conversion from '")
           << fromDType.getAsString() << "' to '" << toDType.getAsString()
           << "' is not implemented";
  }

  Value newResult = castOp.getInput();
  for (Type commonType : commonTypes)
    newResult = CastOp::create(b, commonType, newResult);

  newResult = CastOp::create(b, type, newResult);

  castOp.replaceAllUsesWith(newResult);
  return success();
}

//===----------------------------------------------------------------------===//
// legalizeOperation
//===----------------------------------------------------------------------===//
LogicalResult LegalizePOPOperations::legalizeOperation(Operation *op) {
  assert(operationRequiresLegalization(op) && "legalization not required");
  ImplicitLocOpBuilder b(op->getLoc(), op);

  if (!llvm::all_of(op->getOperands(), [&](Value operand) {
        return operand.getType() == op->getOperand(0).getType();
      })) {
    return op->emitError(
        "Cannot legalize operation with different operand types");
  }

  if (!llvm::all_of(op->getResultTypes(), [&](Type resultType) {
        return resultType == op->getResultTypes()[0];
      })) {
    return op->emitError(
        "Cannot legalize operation with different result types");
  }

  if (op->getOperand(0).getType() != op->getResultTypes()[0])
    return op->emitError("Cannot legalize non-homogeneous operation ");

  Type inputType = op->getOperand(0).getType();
  SmallVector<Type> types = findConversionSequence(inputType, inputType);
  if (types.empty()) {
    KGENDType inputDType = getScalarKGENDType(inputType);
    return op->emitError("operation with a type '")
           << inputDType.getAsString() << "' is not implemented";
  }
  SmallVector<Value> newOperands;

  // First type for which it's safe to perform an operation
  auto supportedTypeIt = llvm::find_if(
      types, [&](Type type) { return !typeRequiresLegalization(type); });
  if (supportedTypeIt == types.end()) {
    KGENDType inputDType = getScalarKGENDType(inputType);
    return op->emitError("no supported intermediate type found for '")
           << inputDType.getAsString() << "'";
  }

  for (Value operand : op->getOperands()) {
    Value newOperand = operand;
    // Do conversion of the operand until it has supported type
    for (auto typeIt = types.begin(); typeIt != supportedTypeIt; ++typeIt)
      newOperand = CastOp::create(b, *typeIt, newOperand);
    newOperand = CastOp::create(b, *supportedTypeIt, newOperand);
    newOperands.push_back(newOperand);
  }

  SmallVector<Type> newResultTypes(op->getNumResults(), *supportedTypeIt);

  OperationState state(op->getLoc(), op->getName(), newOperands, newResultTypes,
                       op->getAttrs());
  // Construct new operation that with a supported type
  Operation *newOp = b.create(state);

  // Finally perform remaining conversion of the result down to the original
  // type.
  for (auto [newResult, oldResult] :
       llvm::zip(newOp->getResults(), op->getResults())) {
    Value resultToUse = newResult;
    for (auto typeIt = std::next(supportedTypeIt, 1); typeIt != types.end();
         ++typeIt) {
      resultToUse = CastOp::create(b, *typeIt, resultToUse);
    }
    resultToUse = CastOp::create(b, oldResult.getType(), resultToUse);
    oldResult.replaceAllUsesWith(resultToUse);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// runOnOperation
//===----------------------------------------------------------------------===//

void LegalizePOPOperations::runOnOperation() {
  Operation *op = getOperation();
  target = lookupTargetInfo(op);
  ErrorOr<const TargetLowering *> loweringOr =
      TargetLoweringRegistry::get().lookup(target.getTriple());
  lowering = loweringOr.isError() ? nullptr : *loweringOr;

  initializeTargetLegalConversions();

  if (op->walk([&](Operation *op) {
          if (auto castOp = dyn_cast<CastOp>(op)) {
            if (conversionRequiresLegalization(castOp)) {
              if (failed(legalizeConversion(castOp)))
                return WalkResult::interrupt();
              castOp->erase();
            }
            return WalkResult::advance();
          }
          if (!operationRequiresLegalization(op))
            return WalkResult::advance();

          if (failed(legalizeOperation(op)))
            return WalkResult::interrupt();
          op->erase();
          return WalkResult::advance();
        }).wasInterrupted())
    return signalPassFailure();
}

} // namespace
