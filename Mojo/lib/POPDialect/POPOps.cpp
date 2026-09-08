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
// This file implements the POP dialect operations.
//
//===----------------------------------------------------------------------===//

#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPEnums.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/POPDialect/POPUtils.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MTypes.h"
#include "Target/TargetTraits.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

using namespace M;
using namespace KGEN;
using namespace POP;

/// This is used by the `ArrayElementType` and `ParamListElementType`
/// constraints to match a type range against a single type.
static bool typeRangeMatches(Type type, TypeRange range) {
  return llvm::all_of(range, [&](Type e) { return type == e; });
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

/// Return a SIMD type whose dtype is bool with the same size as the given type.
static Type getBoolOfSameParentType(Type type) {
  auto boolType = DTypeConstantAttr::get(type.getContext(), DType::kBool);
  if (auto simd = dyn_cast<SIMDType>(type))
    return SIMDType::get(simd.getSize(), boolType);
  return nullptr;
}

LogicalResult CmpOp::inferReturnTypes(MLIRContext *ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type> &types) {
  Type argType = adaptor.getLhs().getType();
  types.push_back(getBoolOfSameParentType(argType));
  if (types.back())
    return success();
  return mlir::emitError(loc.value_or(adaptor.getLhs().getLoc()),
                         "expected a scalar or simd operand type");
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getType();

  // First, check the input and output types must be of the same kind.
  // TODO: In theory we can support casting a scalar type to a vector type (e.g.
  // f64 to a 2xf32) or vice versa. We should support this when the use case
  // arises.
  std::optional<KGENDType> inputDType = inputType.getResolvedDType();
  std::optional<KGENDType> outputDType = outputType.getResolvedDType();

  // If neither dtype could be resolved, allow the cast.
  if (!inputDType || !outputDType)
    return success();

  TargetInfoAttr target = lookupTargetInfo(*this);
  ssize_t inputDTypeWidth = inputDType->getWidthInBits(target);
  ssize_t outputDTypeWidth = outputDType->getWidthInBits(target);

  // If we have a simd type, then the bitwidths must match.
  std::optional<int64_t> inputSize = 1, outputSize = 1;
  if (auto inputSimd = dyn_cast<SIMDType>(inputType)) {
    auto outputSimd = cast<SIMDType>(outputType);
    inputSize = inputSimd.getResolvedSize();
    outputSize = outputSimd.getResolvedSize();
    // If neither size could be resolved, allow the cast.
    if (!inputSize || !outputSize)
      return success();
  }

  if (inputDType->isBool())
    return success(*inputSize == outputDTypeWidth * *outputSize);
  if (outputDType->isBool())
    return success(*outputSize == inputDTypeWidth * *inputSize);
  // If the sizes do not match, then we cannot cast.
  if (inputDTypeWidth * *inputSize == outputDTypeWidth * *outputSize)
    return success();
  return emitOpError("input type ") << inputType << " and result type "
                                    << outputType << " are cast incompatible";
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

bool PointerBitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  return isa<ParamType, PointerType, FuncTypeGeneratorType>(inputs.front()) &&
         isa<ParamType, PointerType, FuncTypeGeneratorType>(outputs.front());
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 && outputs.size() == 1);
  auto input = dyn_cast<SIMDType>(inputs.front());
  auto output = dyn_cast<SIMDType>(outputs.front());
  if (!input || !output) // unbound
    return true;
  return input.getSize() == output.getSize();
}

//===----------------------------------------------------------------------===//
// SIMDShuffleOp
//===----------------------------------------------------------------------===//

static ParseResult parseShuffleMask(AsmParser &p, TypedAttr &mask,
                                    Type resultType) {

  return parseColonTypeParamValue(p, mask);
}

static void printShuffleMask(AsmPrinter &p, Operation *op, TypedAttr mask,
                             Type resultType) {
  printColonTypeParamValue(p, mask);
}

LogicalResult SIMDShuffleOp::verify() {
  std::optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return success();
  auto maskType = cast<ArrayType>(getMask().getType());
  if (!isa<IndexType>(maskType.getElementType()))
    return emitOpError("expected mask to be a list of indices");
  auto mask = dyn_cast_or_null<ArrayAttr>(getMask());
  if (!mask)
    return success();

  if (*size != static_cast<int64_t>(mask.getValues().size()))
    return emitOpError("expected result to be a vector of ")
           << mask.getValues().size() << " elements";

  auto lhsType = cast<SIMDType>(getLhs().getType());
  if (lhsType.getDType() != getType().getDType())
    return emitOpError("expected result dtype to match operand dtypes");

  if (std::optional<int64_t> size = lhsType.getResolvedSize()) {
    for (TypedAttr indexAttr : mask.getValues()) {
      auto index = dyn_cast<IntegerAttr>(indexAttr);
      if (!index)
        continue;
      if (index.getInt() >= *size * 2)
        return emitOpError("mask element ")
               << index.getInt() << " is out of bounds";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SIMDSplatOp
//===----------------------------------------------------------------------===//

LogicalResult SIMDSplatOp::verify() {
  std::optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return success();

  if (*size <= 0)
    return emitOpError("requires a non-negative size");

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   std::optional<unsigned> alignment, bool isVolatile,
                   bool isInvariant, bool isNonTemporal,
                   AtomicOrdering ordering,
                   std::optional<StringAttr> syncscope) {
  build(b, state, ptr, alignment ? b.getIndexAttr(*alignment) : TypedAttr(),
        b.getBoolAttr(isVolatile), b.getBoolAttr(isInvariant),
        b.getBoolAttr(isNonTemporal), ordering,
        syncscope ? *syncscope : TypedAttr());
}

LogicalResult LoadOp::verify() {
  AtomicOrdering ordering = getOrdering();
  if (ordering != AtomicOrdering::NOT_ATOMIC) {
    bool isKnownInvariat = false;
    if (auto boolAttr = ::dyn_cast<BoolAttr>(getIsInvariantAttr()))
      isKnownInvariat = boolAttr.getValue();
    if ((isVolatile().has_value() && *isVolatile()) || isKnownInvariat)
      return emitOpError(
          "invalid combination of volatile or invariant with atomic load");
  }

  if (ordering == AtomicOrdering::NOT_ATOMIC && getSyncscope())
    return emitOpError("cannot specify syncscope without an atomic load");

  if (llvm::is_contained(
          {AtomicOrdering::RELEASE, AtomicOrdering::ACQUIRE_RELEASE},
          ordering)) {
    return emitOpError("invalid atomic ordering '")
           << stringifyAtomicOrdering(ordering)
           << "' for load operation. Valid orderings are: unordered, "
              "monotonic, acquire, seq_cst";
  }

  return success();
}

void LoadOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getPtrMutable());
  if (mightBeVolatile()) {
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
  }
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(OpBuilder &b, OperationState &state, Value arg, Value ptr,
                    std::optional<unsigned> alignment, bool isVolatile,
                    bool isNonTemporal, AtomicOrdering ordering,
                    std::optional<StringAttr> syncscope) {
  build(b, state, arg, ptr,
        alignment ? b.getIndexAttr(*alignment) : TypedAttr(),
        b.getBoolAttr(isVolatile), b.getBoolAttr(isNonTemporal), ordering,
        syncscope ? *syncscope : TypedAttr());
}

LogicalResult StoreOp::verify() {
  AtomicOrdering ordering = getOrdering();
  if (ordering != AtomicOrdering::NOT_ATOMIC) {
    if (auto volatileVal = isVolatile(); volatileVal && *volatileVal)
      return emitOpError("volatile stores cannot be atomic");
  }

  if (ordering == AtomicOrdering::NOT_ATOMIC && getSyncscope())
    return emitOpError("cannot specify syncscope without an atomic store");

  if (llvm::is_contained(
          {AtomicOrdering::ACQUIRE, AtomicOrdering::ACQUIRE_RELEASE},
          ordering)) {
    return emitOpError("invalid atomic ordering '")
           << stringifyAtomicOrdering(ordering)
           << "' for store operation. Valid orderings are: unordered, "
              "monotonic, release, seq_cst";
  }

  return success();
}

void StoreOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getPtrMutable());
  if (mightBeVolatile()) {
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
  }
}

//===----------------------------------------------------------------------===//
// MemcpyOp
//===----------------------------------------------------------------------===//

void MemcpyOp::build(OpBuilder &b, OperationState &state, Value dst, Value src,
                     Value len, bool isVolatile) {
  build(b, state, dst, src, len, b.getBoolAttr(isVolatile));
}

LogicalResult MemcpyOp::verify() {
  auto dstType = cast<PointerType>(getDst().getType());
  auto srcType = cast<PointerType>(getSrc().getType());

  if (dstType.getElementType() != srcType.getElementType()) {
    return emitOpError(
               "source and destination must have same element type, got ")
           << srcType.getElementType() << " and " << dstType.getElementType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayCreateOp::verify() {
  int64_t size = *getType().getResolvedSize();
  if (size != getNumOperands())
    return emitOpError("expected ")
           << size << " operands to create array but got " << getNumOperands();
  return success();
}

void ArrayCreateOp::build(OpBuilder &b, OperationState &state,
                          ValueRange elements) {
  return build(b, state, ArrayType::get(elements), elements);
}

//===----------------------------------------------------------------------===//
// ArrayRepeatOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayRepeatOp::verify() {
  std::optional<int64_t> size = getType().getResolvedSize();
  // Can only verify with concrete size.
  if (!size)
    return success();
  if (*size < 0)
    return emitOpError("requires a non-negative size");
  if (*size != 0 && getNumOperands() == 0)
    return emitOpError("requires at least one operand to create an array whose "
                       "size is non-zero");
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

// If the array has a concrete size, do a bounds check.
static LogicalResult verifyArrayIndex(Operation *op, TypedAttr indexExpr,
                                      POP::ArrayType arrayType) {
  std::optional<int64_t> size = arrayType.getResolvedSize();
  auto indexAttr = dyn_cast<IntegerAttr>(indexExpr);
  if (!size || !indexAttr)
    return success();

  int64_t index = indexAttr.getInt();
  if (index < 0 || index >= *size)
    return op->emitOpError("array index out of bounds: ") << index;
  return success();
}

void ArrayGetOp::build(OpBuilder &b, OperationState &state, Value array,
                       int64_t index) {
  return build(b, state, cast<ArrayType>(array.getType()).getElementType(),
               array, b.getIndexAttr(index));
}

LogicalResult ArrayGetOp::verify() {
  return verifyArrayIndex(*this, getIndex(), getArray().getType());
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayReplaceOp::verify() {
  return verifyArrayIndex(*this, getIndex(), getArray().getType());
}

//===----------------------------------------------------------------------===//
// ArrayGEPOp
//===----------------------------------------------------------------------===//

static Type getPointerToArrayElementType(Type arrayPtr) {
  auto ptr = dyn_cast<PointerType>(arrayPtr);
  if (!ptr)
    return Type();
  auto array = dyn_cast<POP::ArrayType>(ptr.getElementType());
  return array ? PointerType::get(array.getElementType()) : Type();
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

/// Parse an address space parameter if present.
static void printOptionalAddressSpaceParamValue(AsmPrinter &p, Operation *op,
                                                TypedAttr addressSpace) {
  if (!addressSpace)
    return;

  // If the address space is an integer and zero, then we can skip since that's
  // the default address space.
  if (auto addressSpaceInt = dyn_cast<IntegerAttr>(addressSpace);
      addressSpaceInt && addressSpaceInt.getValue().isZero())
    return;

  p << " address_space ";
  printParamValue(p, addressSpace);
  p << " ";
}

/// Parse a parameter value that is known to be an address space type.
static ParseResult parseOptionalAddressSpaceParamValue(AsmParser &p,
                                                       TypedAttr &result) {
  if (p.parseOptionalKeyword("address_space")) {
    // The default address space is 0.
    result = p.getBuilder().getIndexAttr(0);
    return success();
  }

  return parseIndexParamValue(p, result);
}

/// Parse the element type of the allocated pointer type.
static ParseResult parsePointerOf(AsmParser &p, Type &result) {
  Type elementType;
  TypedAttr addressSpace;
  if (parseParamType(p, elementType) ||
      parseOptionalAddressSpaceParamValue(p, addressSpace))
    return failure();

  result = PointerType::get(elementType, addressSpace);
  return success();
}

/// Print the element type of the allocated pointer type.
static void printPointerOf(AsmPrinter &p, Operation *op, Type result) {
  auto ptrType = cast<PointerType>(result);
  printParamType(p, ptrType.getElementType());
  printOptionalAddressSpaceParamValue(p, op, ptrType.getAddressSpace());
}

void StackAllocationOp::build(OpBuilder &b, OperationState &state, Type result,
                              int64_t count, TypedAttr alignment,
                              bool markedLifetimes) {
  build(b, state, result, b.getIndexAttr(count), alignment, markedLifetimes);
}

void StackAllocationOp::build(OpBuilder &b, OperationState &state,
                              bool markedLifetimes, Type result) {
  build(b, state, result, /*count=*/1, /*alignment=*/{}, markedLifetimes);
}

LogicalResult StackAllocationOp::verify() {
  TargetInfoAttr target = lookupTargetInfo(*this);
  if (!target) {
    // Operation may not be in complete state yet. Assume it's good.
    return success();
  }

  std::optional<unsigned> addressSpace = getType().getAddrSpace();
  if (!addressSpace) {
    // The address space is not yet concrete. Assume it's good.
    return success();
  }

  // Some targets constrain a stack allocation's address space (e.g. AMDGPU
  // private, NVPTX local); the required value comes from the target's
  // TargetTraits. TODO: Re-enable this check (guarded by `if (0`) when stdlib
  // is updated.
  ErrorOr<const TargetTraits *> traitsOr =
      TargetTraitsRegistry::get().lookup(target.getTriple());
  const TargetTraits *traits = traitsOr.isError() ? nullptr : *traitsOr;
  std::optional<unsigned> expected =
      traits ? traits->requiredStackAllocationAddressSpace() : std::nullopt;
  if (0 && expected && *addressSpace != *expected) {
    return emitOpError("expected address space (")
           << *expected << "), but got address space (" << *addressSpace << ')';
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeStartOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyLifetimeMarker(Operation *op) {
  llvm::SmallDenseSet<Value> seen;
  for (auto [idx, value] : llvm::enumerate(op->getOperands())) {
    auto alloc = value.getDefiningOp<StackAllocationOp>();
    if (!alloc) {
      InFlightDiagnostic diag = op->emitOpError()
                                << "operand #" << idx
                                << " is not defined by a stack allocation op";
      diag.attachNote(value.getLoc()) << "value is defined here";
      return diag;
    }
    if (!alloc.getMarkedLifetimes()) {
      InFlightDiagnostic diag =
          op->emitOpError()
          << "operand #" << idx
          << " is not defined by a stack allocation with marked lifetimes";
      diag.attachNote(alloc.getLoc()) << "stack allocation defined here";
      return diag;
    }
    if (!seen.insert(value).second) {
      InFlightDiagnostic diag = op->emitOpError("operand #")
                                << idx << " is a duplicate";
      diag.attachNote(value.getLoc()) << "operand defined here";
      return diag;
    }
  }
  return success();
}

LogicalResult StackAllocLifetimeStartOp::verify() {
  return verifyLifetimeMarker(*this);
}

void StackAllocLifetimeStartOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  // This op allocates all its operands.
  for (OpOperand &value : getValuesMutable())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), &value);
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeEndOp
//===----------------------------------------------------------------------===//

LogicalResult StackAllocLifetimeEndOp::verify() {
  return verifyLifetimeMarker(*this);
}

void StackAllocLifetimeEndOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  // This op frees all its operands.
  for (OpOperand &value : getValuesMutable())
    effects.emplace_back(mlir::MemoryEffects::Free::get(), &value);
}

//===----------------------------------------------------------------------===//
// ExternalCallOp
//===---------------------------------------------------------------------===//

static ParseResult parseExternalCallee(AsmParser &p, TypedAttr &callee) {
  StringAttr concreteCallee;
  // Try `@foo`.
  if (succeeded(p.parseOptionalSymbolName(concreteCallee))) {
    callee = StringAttr::get(concreteCallee.getValue(),
                             StringType::get(p.getContext()));
    return success();
  }
  // Otherwise, parse a string expression inside square brackets.
  if (p.parseLSquare() ||
      parseParamValue(p, callee, StringType::get(p.getContext())) ||
      p.parseRSquare())
    return failure();
  return success();
}

static void printExternalCallee(AsmPrinter &p, Operation *op,
                                TypedAttr callee) {
  // Print a symbol name if the callee is concrete.
  if (auto concrete = dyn_cast<StringAttr>(callee)) {
    p.printSymbolName(concrete);
    return;
  }
  // Otherwise, print the string expression in square brackets to disambiguate
  // `callee(` as a parameter operator.
  p << '[';
  printParamValue(p, callee);
  p << ']';
}

void ExternalCallOp::build(OpBuilder &b, OperationState &state, StringRef func,
                           ValueRange operands) {
  build(b, state, {}, func, operands);
}

void ExternalCallOp::build(OpBuilder &b, OperationState &state, Type result,
                           StringRef func, ValueRange operands) {
  build(b, state, result,
        StringAttr::get(func, StringType::get(b.getContext())), operands,
        mlir::IntegerAttr(), mlir::ArrayAttr(), mlir::ArrayAttr(),
        mlir::DictionaryAttr(), Attribute());
}

void ExternalCallOp::build(OpBuilder &b, OperationState &state, Type result,
                           StringRef func, ValueRange operands,
                           int64_t numFixedArgs) {
  build(b, state, result,
        StringAttr::get(func, StringType::get(b.getContext())), operands,
        b.getIndexAttr(numFixedArgs), mlir::ArrayAttr(), mlir::ArrayAttr(),
        mlir::DictionaryAttr(), Attribute());
}

LogicalResult
ExternalCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return success();
}

LogicalResult ExternalCallOp::verify() {
  // `numFixedArgs` may still be an unresolved parameter expression here, so
  // only a materialized count can be checked. The operand count is not an
  // upper bound either: argument packs are expanded into individual operands
  // later, by LowerArgConventions, so the post-expansion bound is checked when
  // lowering to LLVM.
  auto fixedArgCount = dyn_cast_or_null<IntegerAttr>(getNumFixedArgsAttr());
  if (fixedArgCount && fixedArgCount.getInt() < 0)
    return emitOpError("'numFixedArgs' must be non-negative, found ")
           << fixedArgCount.getInt();

  if (mlir::ArrayAttr argAttrs = getArgAttrsAttr()) {
    size_t numArgs;
    if (fixedArgCount)
      numArgs = fixedArgCount.getInt();
    else
      numArgs = getNumOperands();
    if (argAttrs.size() != numArgs) {
      return mlir::emitError(getLoc(), "external callee has ")
             << numArgs << " arguments but " << argAttrs.size()
             << " argument attributes specified";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalAllocOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalAllocOp::verify() {
  if (getInitializer()) {
    auto countAttr = dyn_cast<IntegerAttr>(getCount());
    if (countAttr && countAttr.getInt() != 1)
      return emitOpError("with an initializer requires count to be 1, but got ")
             << countAttr.getInt();
  }
  return success();
}

static ParseResult parseOptionalGlobalAllocInitializer(OpAsmParser &p,
                                                       TypedAttr &value,
                                                       Type resultType) {
  if (failed(p.parseOptionalEqual())) {
    value = nullptr;
    return success();
  }
  Type elementType = cast<PointerType>(resultType).getElementType();
  if (p.parseLess() || parseParamValue(p, value, elementType) ||
      p.parseGreater())
    return failure();
  return success();
}

static void printOptionalGlobalAllocInitializer(OpAsmPrinter &p, Operation *,
                                                TypedAttr value, Type) {
  if (!value)
    return;
  p << "= <";
  printParamValue(p, value);
  p << ">";
}

//===----------------------------------------------------------------------===//
// GlobalConstantOp
//===----------------------------------------------------------------------===//

void GlobalConstantOp::build(OpBuilder &b, OperationState &state,
                             TypedAttr value) {
  build(b, state, value, TypedAttr());
}

static ParseResult parseGlobalConstantOpValue(OpAsmParser &p, TypedAttr &value,
                                              Type &resultType) {
  Type elementType;
  if (parseColonTypeOrIndex(p, elementType) || p.parseEqual() ||
      p.parseLess() || parseParamValue(p, value, elementType) ||
      p.parseGreater())
    return failure();
  resultType = PointerType::get(elementType);
  return success();
}

static void printGlobalConstantOpValue(OpAsmPrinter &p, Operation *,
                                       TypedAttr value, Type type) {
  printColonTypeOrIndex(p, cast<PointerType>(type).getElementType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

LogicalResult
GlobalConstantOp::inferReturnTypes(MLIRContext *ctx,
                                   std::optional<Location> loc, Adaptor adaptor,
                                   SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      PointerType::get(adaptor.getValueAttr().getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// CallLLVMIntrinsicOp
//===----------------------------------------------------------------------===//

void CallLLVMIntrinsicOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  auto hasSideEffects = dyn_cast<IntegerAttr>(getHasSideEffects());
  if (!hasSideEffects || hasSideEffects.getInt())
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

mlir::Speculation::Speculatability CallLLVMIntrinsicOp::getSpeculatability() {
  auto hasSideEffects = dyn_cast<IntegerAttr>(getHasSideEffects());
  if (hasSideEffects && !hasSideEffects.getInt())
    return mlir::Speculation::Speculatable;
  return mlir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

bool PointerToIndexOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 && outputs.size() == 1);
  return isa<PointerType>(inputs.front()) && isa<IndexType>(outputs.front());
}

//===----------------------------------------------------------------------===//
// CastToBuiltinOp
//===----------------------------------------------------------------------===//

LogicalResult CastToBuiltinOp::verify() {
  return verifyConversionCast(
      [this](StringRef msg) { return emitOpError(msg); }, getInput().getType(),
      getType(), /*fromSimd=*/true);
}

//===----------------------------------------------------------------------===//
// CastFromBuiltinOp
//===----------------------------------------------------------------------===//

LogicalResult CastFromBuiltinOp::verify() {
  return verifyConversionCast(
      [this](StringRef msg) { return emitOpError(msg); }, getType(),
      getInput().getType(), /*fromSimd=*/false);
}

//===----------------------------------------------------------------------===//
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//

/// Return an KGEN struct type with any integer or pointer followed by a
/// boolean.
static Type getCmpXChgResultType(Type type) {
  auto pointerType = dyn_cast<PointerType>(type);
  if (!pointerType)
    return nullptr;
  Type eltType = pointerType.getElementType();
  auto boolType =
      SIMDType::get(1, DTypeConstantAttr::get(type.getContext(), DType::kBool));
  return StructType::get(type.getContext(), {eltType, boolType});
}

//===----------------------------------------------------------------------===//
// FenceOp
//===----------------------------------------------------------------------===//

LogicalResult FenceOp::verify() {
  if (llvm::is_contained({AtomicOrdering::NOT_ATOMIC, AtomicOrdering::UNORDERED,
                          AtomicOrdering::MONOTONIC},
                         getOrdering()))
    return emitOpError("can be given only acquire, release, acq_rel, "
                       "and seq_cst orderings");
  return success();
}

//===----------------------------------------------------------------------===//
// VariantBitcastOp
//===----------------------------------------------------------------------===//

LogicalResult VariantBitcastOp::verify() {
  auto ptrType = getVariant().getType();
  if (ptrType.getAddressSpace() != getType().getAddressSpace()) {
    return emitOpError("result pointer should have the same address space as "
                       "the input pointer");
  }

  // Only verify the result type if the variant type is concrete.
  auto variant = cast<VariantType>(ptrType.getElementType());
  if (!isa<ParamListAttr>(variant.getVariadic()))
    return success();
  auto indexAttr = dyn_cast<IntegerAttr>(getIndex());
  if (!indexAttr)
    return success();
  unsigned index = indexAttr.getInt();

  if (index >= variant.getNumTypes()) {
    return emitOpError("variant index ")
           << index << " is out of bounds in range [0, "
           << variant.getNumTypes() << ")";
  }
  Type elementType = variant.getType(index);
  if (elementType == getType().getElementType())
    return success();
  return emitOpError("variant element at index ")
         << index << " expected type " << elementType << " but result has type "
         << getType().getElementType();
}

//===----------------------------------------------------------------------===//
// VariantDiscrGEPOp
//===----------------------------------------------------------------------===//

LogicalResult VariantDiscrGEPOp::verify() {
  auto ptrType = getVariant().getType();
  if (ptrType.getAddressSpace() != getType().getAddressSpace()) {
    return emitOpError("result pointer should have the same address space as "
                       "the input pointer");
  }

  // Only verify the result type if the variant type is concrete.
  auto variant = cast<VariantType>(ptrType.getElementType());
  if (!isa<ParamListAttr>(variant.getVariadic()))
    return success();

  auto discrType = cast<SIMDType>(getDiscr().getType().getElementType());
  std::optional<DType> dtype = discrType.getResolvedDType();
  if (!dtype)
    return success();

  // Both the element types list and discriminant type are concrete. Let's
  // verify them.
  size_t variantSize = variant.getDiscrSizeInBits();
  size_t discrSize = dtype->getIntegerWidthInBits();
  if (variantSize == discrSize)
    return success();
  return emitOpError("variant expected discriminant bitwidth to be ")
         << variantSize << " but result returns uint with width " << discrSize;
}

//===----------------------------------------------------------------------===//
// UnionBitcastOp
//===----------------------------------------------------------------------===//

static bool findUnionType(UnionType unionType, Type type) {
  auto it = llvm::find(unionType.getTypes(), type);
  return it != unionType.getTypes().end();
}

static LogicalResult verifyUnionType(Operation *op, UnionType unionType,
                                     Type type, StringRef desc) {
  // Skip verification for parameterized (unresolved) union types.
  // When the union is not resolved, the union types are not yet
  // specialized, so we cannot verify type membership at compile time.
  if (!unionType.isResolved())
    return success();
  if (findUnionType(unionType, type))
    return success();
  return op->emitOpError(desc)
         << " type " << type << " is not an element type of " << unionType;
}

LogicalResult UnionBitcastOp::verify() {
  return verifyUnionType(*this, getValue().getType().getElementAs<UnionType>(),
                         getType().getElementType(), "result pointer element");
}

//===----------------------------------------------------------------------===//
// UnionWrapOp
//===----------------------------------------------------------------------===//

LogicalResult UnionWrapOp::verify() {
  return verifyUnionType(*this, getResult().getType(), getValue().getType(),
                         "operand");
}

//===----------------------------------------------------------------------===//
// UnionUnwrapOp
//===----------------------------------------------------------------------===//

LogicalResult UnionUnwrapOp::verify() {
  return verifyUnionType(*this, getValue().getType(), getResult().getType(),
                         "result");
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "Mojo/POPDialect/POP.cpp.inc"
