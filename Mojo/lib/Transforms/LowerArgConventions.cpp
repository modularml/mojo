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
// This pass performs the lowering of argument input conventions of concrete
// functions. This pass must run before inlining, but after elaboration. This
// pass will:
//
// 1. Move register passable types passed as `{owned,borrowed}_in_mem` to be
//    passed in register.
// 2. Promote register passable `byref_result` arguments to function results.
//    - This also handles functions that throw.
// 3. Unpacks kgen.pack typed arguments.
//
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "Target/TargetLowering.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERARGCONVENTIONS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

static StructType getIfParamPack(Type type) {
  if (auto packStruct = ::dyn_cast<StructType>(type))
    if (packStruct.getIsParamPack())
      return packStruct;
  return {};
}

/// If the specified value has uses, replace them with a dummy value.
static void replaceUsesWithDummy(Value v, ImplicitLocOpBuilder &b,
                                 TypedAttr value = {}) {
  if (v.use_empty())
    return;
  // A dropped value's stand-in is genuinely uninitialized memory: it is never
  // meant to be read, and it has to survive to codegen as an undef.
  if (!value)
    value = UninitMemAttr::get(v.getType());
  auto dummy = ParamConstantOp::create(b, value);
  v.replaceAllUsesWith(dummy);
}

namespace {
struct LowerArgConventionsPass
    : KGEN::impl::LowerArgConventionsBase<LowerArgConventionsPass> {
  using LowerArgConventionsBase::LowerArgConventionsBase;
  void runOnOperation() override;
};
} // namespace

/// Return the lowered type for an in-memory passed argument. If lowering is not
/// needed, return null.
///
/// `isCABI` delegates lowering of all structs to the platform ABI classifier,
/// so the Mojo-specific checks below are skipped.
static Type lowerPointerType(Type type, TargetInfoAttr target,
                             unsigned maxInlineSize, bool isCABI) {
  // Only pointer types should be lowered.
  auto argPtr = dyn_cast<PointerType>(type);
  if (!argPtr)
    return {};

  // We don't lower memory-only structs.
  Type elType = argPtr.getElementType();
  if (!isCABI)
    if (auto structType = dyn_cast<StructType>(elType))
      if (structType.isDefinitelyMemoryOnly())
        return {};

  // Don't promote a pointer whose element this target keeps in memory form.
  if (target) {
    ErrorOr<const TargetLowering *> loweringOr =
        TargetLoweringRegistry::get().lookup(target.getTriple());
    const TargetLowering *lowering =
        loweringOr.isError() ? nullptr : *loweringOr;
    if (lowering && lowering->lowerKernelArgToMemory(elType))
      return {};
  }

  if (target) {
    // Parameter packs represent a list of arguments. Unconditionally promote
    // them. Promotion will be considered for each individual member of the
    // pack.
    if (auto structType = dyn_cast<StructType>(elType);
        structType && structType.getIsParamPack())
      return elType;

    std::optional<int64_t> size =
        DataLayoutInterface::getTypeStoreSize(target, elType);
    // If layout can't provide a stable size, prefer keeping the promotion path
    // so lowering can continue on the unpacked/value form.
    if (!size || *size < 0)
      return elType;
    if (!isCABI && *size > static_cast<int64_t>(maxInlineSize))
      return {};
  }

  // We must be dealing with something register passable (e.g. index).
  return elType;
}

namespace {

// Whether to convert the error slot or the result slot to a register, and/or
// whether to remove the error slot entirely.
enum ABI {
  DontPromote = 0b000,
  PromoteResult = 0b001,
  PromoteError = 0b010,
  PromoteBoth = 0b011,
  RemoveError = 0b100,
  PromoteResultRemoveError = 0b101,
};
struct TransformResult {
  SmallVector<Type> newResultTypes;
  SmallVector<ArgConvention> newArgConventions;
  unsigned abiLowering = ABI::DontPromote;
};
class Transform {
public:
  Transform(TargetInfoAttr target, DebugInfo::DISubprogramAttr spAttr,
            unsigned maxInlineSize)
      : target(target), spAttr(spAttr), maxInlineSize(maxInlineSize) {}
  virtual ~Transform() = default;
  virtual Type typeOfValueAt(unsigned operandIndex) = 0;
  virtual void performResultTransform(unsigned operandIndex, Type loweredType,
                                      ArgConvention convention) = 0;
  virtual void performThrowNeverElimination(unsigned operandIndex) = 0;

  /// respond to a transform from PTR<X> to X
  virtual void applyPointerTransform(unsigned operandIndex, Type elType) = 0;
  /// respond to a transform from PACK<X,Y> to X,Y
  virtual void applyPackTransform(unsigned operandIndex, ArrayRef<Type> types,
                                  StructType packStruct) = 0;
  /// respond to a transform from X to PTR<X>
  virtual void applyValueTransform(unsigned operandIndex, Type ptrType) = 0;
  Location addDI(Location loc);
  TargetInfoAttr target;
  DebugInfo::DISubprogramAttr spAttr;
  unsigned maxInlineSize;
};
class CallsiteTransform : public Transform {
public:
  CallsiteTransform(ImplicitLocOpBuilder &b, Operation *callOp,
                    TargetInfoAttr target, DebugInfo::DISubprogramAttr spAttr,
                    unsigned maxInlineSize)
      : Transform(target, spAttr, maxInlineSize), b(b), callOp(callOp) {}
  Type typeOfValueAt(unsigned operandIndex) override;
  void performResultTransform(unsigned operandIndex, Type loweredType,
                              ArgConvention convention) override;
  void performThrowNeverElimination(unsigned operandIndex) override;

  void applyPointerTransform(unsigned operandIndex, Type elType) override;
  void applyPackTransform(unsigned operandIndex, ArrayRef<Type> types,
                          StructType packStruct) override;
  void applyValueTransform(unsigned operandIndex, Type ptrType) override;
  ImplicitLocOpBuilder &b;
  Operation *callOp;
  Value errorOperand;
  Value resultOperand;
};

class SignatureTransform : public Transform {
public:
  SignatureTransform(FuncType oldSig, TargetInfoAttr target,
                     DebugInfo::DISubprogramAttr spAttr, unsigned maxInlineSize)
      : Transform(target, spAttr, maxInlineSize), oldSig(oldSig) {
    llvm::append_range(newInputs, oldSig.getValues().getInputs());
  }
  Type typeOfValueAt(unsigned operandIndex) override;
  void performResultTransform(unsigned operandIndex, Type loweredType,
                              ArgConvention convention) override;
  void performThrowNeverElimination(unsigned operandIndex) override;
  void applyPointerTransform(unsigned operandIndex, Type elType) override;
  void applyPackTransform(unsigned operandIndex, ArrayRef<Type> types,
                          StructType packStruct) override;
  void applyValueTransform(unsigned operandIndex, Type ptrType) override;
  FuncType oldSig;
  SmallVector<Type> newInputs;
};

class FuncTransform : public Transform {
public:
  FuncTransform(ImplicitLocOpBuilder &b, FuncOp funcOp, TargetInfoAttr target,
                unsigned maxInlineSize);
  void performResultTransform(unsigned operandIndex, Type loweredType,
                              ArgConvention convention) override;
  void performThrowNeverElimination(unsigned operandIndex) override;
  Type typeOfValueAt(unsigned operandIndex) override;
  void applyPointerTransform(unsigned operandIndex, Type elType) override;
  void applyPackTransform(unsigned operandIndex, ArrayRef<Type> types,
                          StructType packStruct) override;
  void applyValueTransform(unsigned operandIndex, Type ptrType) override;
  ImplicitLocOpBuilder &b;

  Block &block;
  Value newResPtr;
  Value newErrPtr;
  SmallVector<Attribute> LLVMArgMetadata;
  bool hasError = false;

private:
  void applyOneToOneTransform(unsigned operandIndex, Type newType,
                              llvm::function_ref<Value(Location, Value)> apply);
};
} // namespace

static void
insertAndUpdateConventions(SmallVectorImpl<ArgConvention> &conventions,
                           unsigned argConventionIndex, ArrayRef<Type> types,
                           int depth) {
  if (types.size() == 0) {
    conventions[argConventionIndex] = ArgConvention::ImmReg;
    return;
  }
  unsigned packSize = types.size();
  conventions.resize(conventions.size() + packSize - 1);
  for (unsigned i = conventions.size() - 1; i >= argConventionIndex + packSize;
       i--)
    conventions[i] = conventions[i - (packSize - 1)];
  for (unsigned i = 0; i < packSize; ++i) {
    ArgConvention newConvention = ArgConvention::ImmReg;
    if (auto ptr = dyn_cast<PointerType>(types[i])) {
      // if the depth is 0 then this is a top level kgen.pack and we do not know
      // if it holds an address that is potentially written to.
      if (depth == 0)
        newConvention = ArgConvention::Mut;
      else if (!isa<KGEN::NoneType>(ptr.getElementType()))
        newConvention = ArgConvention::ImmMem;
    }
    conventions[argConventionIndex + i] = newConvention;
  }
}

static void transformNonResultValue(Transform *transform, unsigned operandIndex,
                                    SmallVector<ArgConvention> &conventions,
                                    unsigned argConventionIndex, bool isCABI,
                                    int depth = 0) {

  Type type = transform->typeOfValueAt(operandIndex);
  ArgConvention convention = conventions[argConventionIndex];
  const TargetLowering *lowering = nullptr;
  if (transform->target) {
    ErrorOr<const TargetLowering *> loweringOr =
        TargetLoweringRegistry::get().lookup(transform->target.getTriple());
    lowering = loweringOr.isError() ? nullptr : *loweringOr;
  }

  // Apply any extra indirection this target requires for a register argument.
  if (lowering) {
    if (Type indirected =
            lowering->getKernelArgIndirectionType(type, convention)) {
      transform->applyValueTransform(operandIndex, indirected);
      conventions[argConventionIndex] = ArgConvention::ImmMem;
    }
  }

  /// LOWER PTR
  if (isa<PointerType>(type) && !(convention == ArgConvention::ImmMem ||
                                  convention == ArgConvention::OwnedMem ||
                                  convention == ArgConvention::DeinitMem))
    return;

  if (auto elType = lowerPointerType(type, transform->target,
                                     transform->maxInlineSize, isCABI)) {
    transform->applyPointerTransform(operandIndex, elType);
    conventions[argConventionIndex] =
        (conventions[argConventionIndex] == ArgConvention::OwnedMem ||
         conventions[argConventionIndex] == ArgConvention::DeinitMem)
            ? ArgConvention::OwnedReg
            : ArgConvention::ImmReg;
    transformNonResultValue(transform, operandIndex, conventions,
                            argConventionIndex, isCABI, ++depth);
  }

  /// LOWER PACK. Look for a kgen.struct with the "isParamPack" attribute.
  if (auto packStruct = getIfParamPack(type)) {
    if (convention != ArgConvention::OwnedReg &&
        convention != ArgConvention::ImmReg)
      return;

    auto variadic =
        cast_or_null<ParamListAttr>(packStruct.getElementTypesVariadic());
    assert(variadic && "expected variadic pack type");
    SmallVector<Type> types;
    for (auto member : variadic.getValues()) {
      Type memberType = member.getType();
      if (auto typeValue = dyn_cast<KGEN::TypeParamAttr>(member))
        memberType = typeValue.getMlirType();
      types.push_back(memberType);
    }
    transform->applyPackTransform(operandIndex, types, packStruct);
    insertAndUpdateConventions(conventions, argConventionIndex, types, depth);
    transformNonResultValue(transform, operandIndex, conventions,
                            argConventionIndex, isCABI, ++depth);
  }

  /// LIFT REG. Lower an argument this target cannot pass in registers back to
  /// its memory form.
  if (!lowering)
    return;
  if (Type newArgTy = lowering->lowerKernelArgToMemory(type)) {
    transform->applyValueTransform(operandIndex, newArgTy);
    conventions[argConventionIndex] = ArgConvention::ImmMem;
  }
}

FuncTransform::FuncTransform(ImplicitLocOpBuilder &b, FuncOp funcOp,
                             TargetInfoAttr target, unsigned maxInlineSize)
    : Transform(target, funcOp.getSubprogramScope(), maxInlineSize), b(b),
      block(funcOp.getBodyRegion().front()),
      LLVMArgMetadata(funcOp.getLLVMArgMetadata().getValue()) {}

Location Transform::addDI(Location loc) {
  if (!spAttr)
    return loc;
  return FusedLoc::get(loc.getContext(), loc, spAttr);
}

Type FuncTransform::typeOfValueAt(unsigned operandIndex) {
  return block.getArgument(operandIndex).getType();
}

void FuncTransform::performResultTransform(unsigned operandIndex,
                                           Type loweredType,
                                           ArgConvention convention) {
  Value argVal = block.getArgument(operandIndex);
  auto alloc = POP::StackAllocationOp::create(b, addDI(argVal.getLoc()),
                                              argVal.getType());
  argVal.replaceAllUsesWith(alloc);
  block.eraseArgument(operandIndex);
  if (convention == ArgConvention::ByRefError)
    newErrPtr = alloc;
  else if (convention == ArgConvention::ByRefResult)
    newResPtr = alloc;
}

void FuncTransform::performThrowNeverElimination(unsigned operandIndex) {
  replaceUsesWithDummy(block.getArgument(operandIndex), b);
  block.eraseArgument(operandIndex);
}

void FuncTransform::applyOneToOneTransform(
    unsigned operandIndex, Type newType,
    llvm::function_ref<Value(Location, Value)> apply) {
  auto point = b.saveInsertionPoint();
  auto resetState = llvm::scope_exit([&] { b.restoreInsertionPoint(point); });
  b.setInsertionPointToStart(&block);
  Location originalLocation = block.getArgument(operandIndex).getLoc();
  BlockArgument arg =
      block.insertArgument(operandIndex + 1, newType, originalLocation);
  Location location = addDI(block.getArgument(operandIndex).getLoc());
  auto image = apply(location, arg);
  block.getArgument(operandIndex).replaceAllUsesWith(image);
  block.eraseArgument(operandIndex);
}

void FuncTransform::applyPointerTransform(unsigned operandIndex, Type elType) {
  auto application = [&](Location location, Value newArg) -> Value {
    auto ptr = POP::StackAllocationOp::create(
        b, location, PointerType::get(newArg.getType()));
    POP::StoreOp::create(b, location, newArg, ptr);
    return ptr;
  };
  applyOneToOneTransform(operandIndex, elType, application);
}

void FuncTransform::applyValueTransform(unsigned operandIndex, Type ptrType) {
  auto application = [&](Location location, Value newArg) -> Value {
    return POP::LoadOp::create(b, location, newArg).getResult();
  };
  applyOneToOneTransform(operandIndex, ptrType, application);
}

void FuncTransform::applyPackTransform(unsigned operandIndex,
                                       ArrayRef<Type> types, StructType type) {
  auto point = b.saveInsertionPoint();
  auto resetState = llvm::scope_exit([&] { b.restoreInsertionPoint(point); });
  Location originalLocation = block.getArgument(operandIndex).getLoc();
  b.setInsertionPointToStart(&block);
  SmallVector<Value> newArgs;
  unsigned curr = operandIndex;
  for (auto member : types)
    newArgs.push_back(block.insertArgument(++curr, member, originalLocation));
  auto pack =
      KGEN::StructCreateOp::create(b, addDI(originalLocation), type, newArgs);
  block.getArgument(operandIndex).replaceAllUsesWith(pack);
  if (newArgs.empty())
    block.insertArgument(++curr, KGEN::NoneType::get(type.getContext()),
                         originalLocation);
  block.eraseArgument(operandIndex);

  // Update the per-argument LLVM metadata to remain aligned with the updated
  // argument list.
  if (LLVMArgMetadata.empty())
    return;

  auto dict = cast<DictionaryAttr>(LLVMArgMetadata[operandIndex]);
  if (!dict.empty()) {
    block.getParentOp()->emitError()
        << "cannot unpack argument " << operandIndex
        << " that has LLVMArgMetadata";
    hasError = true;
    return;
  }

  if (types.size() > 1) {
    // Insert (types.size() - 1) empty entries after operandIndex, preserving
    // the existing empty entry at operandIndex to correspond to the first
    // unpacked arg since it's already known to be an empty DictionaryAttr.
    LLVMArgMetadata.insert(LLVMArgMetadata.begin() + operandIndex + 1,
                           types.size() - 1,
                           DictionaryAttr::get(type.getContext()));
  }
}

void CallsiteTransform::performResultTransform(unsigned operandIndex,
                                               Type loweredType,
                                               ArgConvention convention) {
  if (convention == ArgConvention::ByRefError)
    errorOperand = callOp->getOperand(operandIndex);
  else if (convention == ArgConvention::ByRefResult)
    resultOperand = callOp->getOperand(operandIndex);
  callOp->eraseOperand(operandIndex);
}

void CallsiteTransform::performThrowNeverElimination(unsigned operandIndex) {
  callOp->eraseOperand(operandIndex);
}

void CallsiteTransform::applyPointerTransform(unsigned operandIndex,
                                              Type elType) {
  b.setInsertionPoint(callOp);
  Value arg = callOp->getOperands()[operandIndex];
  Value newArg = POP::LoadOp::create(b, arg);
  callOp->setOperand(operandIndex, newArg);
}

void CallsiteTransform::applyValueTransform(unsigned operandIndex,
                                            Type ptrType) {
  b.setInsertionPoint(callOp);
  Value arg = callOp->getOperands()[operandIndex];
  Value newArg = POP::StackAllocationOp::create(b, ptrType);
  POP::StoreOp::create(b, arg, newArg);
  callOp->setOperand(operandIndex, newArg);
}

/// Emit StructExtractOp for each element of a pack-typed value.
static SmallVector<Value> emitPackExtracts(OpBuilder &builder, Location loc,
                                           Value pack, ArrayRef<Type> types) {
  SmallVector<Value> results;
  for (auto [i, type] : llvm::enumerate(types))
    results.push_back(StructExtractOp::create(
        builder, loc, type, pack, IntegerAttr::get(builder.getIndexType(), i)));
  return results;
}

void CallsiteTransform::applyPackTransform(unsigned operandIndex,
                                           ArrayRef<Type> types,
                                           StructType packStruct) {
  b.setInsertionPoint(callOp);
  Value operand = callOp->getOperands()[operandIndex];
  SmallVector<Value> newArgs =
      emitPackExtracts(b, operand.getLoc(), operand, types);
  SmallVector<Value> newOperands;
  for (unsigned i = 0; i < operandIndex; ++i)
    newOperands.push_back(callOp->getOperand(i));

  if (types.empty())
    newOperands.push_back(ParamConstantOp::create(b, b.getAttr<NoneAttr>()));
  else
    llvm::append_range(newOperands, newArgs);

  for (unsigned i = operandIndex + 1; i < callOp->getNumOperands(); ++i)
    newOperands.push_back(callOp->getOperand(i));
  callOp->setOperands(newOperands);
}

Type CallsiteTransform::typeOfValueAt(unsigned operandIndex) {
  return callOp->getOperandTypes()[operandIndex];
}

void SignatureTransform::performResultTransform(unsigned operandIndex,
                                                Type loweredType,
                                                ArgConvention convention) {
  newInputs.erase(newInputs.begin() + operandIndex);
}

void SignatureTransform::performThrowNeverElimination(unsigned operandIndex) {
  newInputs.erase(newInputs.begin() + operandIndex);
}

void SignatureTransform::applyPointerTransform(unsigned operandIndex,
                                               Type elType) {
  newInputs[operandIndex] = elType;
}

void SignatureTransform::applyValueTransform(unsigned operandIndex,
                                             Type ptrType) {
  newInputs[operandIndex] = ptrType;
}

void SignatureTransform::applyPackTransform(unsigned operandIndex,
                                            ArrayRef<Type> types,
                                            StructType packStruct) {
  if (types.empty()) {
    newInputs[operandIndex] = KGEN::NoneType::get(packStruct.getContext());
    return;
  }
  auto eraseIt = newInputs.begin() + operandIndex;
  newInputs.erase(eraseIt);
  auto insertIt =
      newInputs.begin() + operandIndex; // Position where the erased element was
  newInputs.insert(insertIt, types.begin(), types.end());
}

Type SignatureTransform::typeOfValueAt(unsigned operandIndex) {
  return newInputs[operandIndex];
}

static TransformResult lowerSignature(FuncType oldSig, size_t operandOffset,
                                      Transform *transform) {
  TransformResult result;
  Type loweredErrorType, loweredResultType;

  llvm::append_range(result.newArgConventions, oldSig.getArgConventions());
  SmallVector<ArgConvention> &argConventions = result.newArgConventions;
  for (size_t idx = 0; idx < argConventions.size(); ++idx) {
    auto convention = argConventions[idx];
    if (!isResultSlot(convention)) {
      transformNonResultValue(transform, idx + operandOffset, argConventions,
                              idx, oldSig.isCABI());
      continue;
    }
    if (oldSig.isAsync()) // Async is broken.
      continue;

    Type valueType = transform->typeOfValueAt(idx + operandOffset);

    // Preserve throw-never elimination independent of size-capped register
    // passing. `!kgen.pointer<!kgen.never>` may not produce a concrete size,
    // but we should still remove the error slot entirely.
    if (convention == ArgConvention::ByRefError) {
      if (auto ptrTy = dyn_cast<PointerType>(valueType);
          ptrTy && isa<NeverType>(ptrTy.getElementType())) {
        argConventions.erase(argConventions.begin() + idx);
        result.abiLowering |= ABI::RemoveError;
        transform->performThrowNeverElimination(idx + operandOffset);
        --idx;
        continue;
      }
    }

    // This is either a byref_result or byref_error argument. See if it can be
    // lowered to being returned directly in a register.
    assert(!(oldSig.isCABI() && convention == ArgConvention::ByRefError) &&
           "abi(\"C\") function must not have an error slot");

    // Promoting a C result slot lets the platform ABI return it in a register
    // or through an sret pointer.
    Type loweredType =
        lowerPointerType(valueType, transform->target, transform->maxInlineSize,
                         oldSig.isCABI());
    if (!loweredType)
      continue;

    argConventions.erase(argConventions.begin() + idx);

    // Otherwise note that we're returning an error slot or a result slot.
    if (convention == ArgConvention::ByRefError) {
      loweredErrorType = loweredType;
      result.abiLowering |= ABI::PromoteError;
    } else if (convention == ArgConvention::ByRefResult) {
      loweredResultType = loweredType;
      result.abiLowering |= ABI::PromoteResult;
    }
    transform->performResultTransform(idx + operandOffset, loweredType,
                                      convention);
    --idx;
  }

  switch (result.abiLowering) {
  case ABI::DontPromote:
    llvm::append_range(result.newResultTypes, oldSig.getResults());
    break;
  case ABI::PromoteError:
    // For a throwing function, this retains the i1 result by default unless
    // we're able to lower the normal result or the error result.  If we are
    // able to lower one of them, we end up with "i1, othertype" as two results
    // from the KGEN function, if both can be lowered to registers then we
    // replace the i1 with "variant<normal, error>".
    result.newResultTypes.push_back(
        SIMDType::getScalarBoolType(oldSig.getContext()));
    result.newResultTypes.push_back(loweredErrorType);
    break;
  case ABI::PromoteBoth:
    result.newResultTypes.push_back(
        VariantType::get({loweredErrorType, loweredResultType}));
    break;
  case ABI::PromoteResult:
    if (oldSig.isThrows())
      result.newResultTypes.push_back(
          SIMDType::getScalarBoolType(oldSig.getContext()));
    [[fallthrough]];
  case ABI::PromoteResultRemoveError:
    result.newResultTypes.push_back(loweredResultType);
    break;
  case ABI::RemoveError: // No result.
    break;
  }

  return result;
}

/// Lowers the given signature if needed
static FuncType lowerSignature(FuncType sig, TargetInfoAttr target,
                               DebugInfo::DISubprogramAttr spAttr,
                               unsigned maxInlineSize) {
  SignatureTransform transform(sig, target, spAttr, maxInlineSize);
  TransformResult result = lowerSignature(sig, 0, &transform);
  FuncType newSig =
      FuncType::get(FunctionType::get(sig.getContext(), transform.newInputs,
                                      result.newResultTypes),
                    result.newArgConventions, sig.getFnEffects(),
                    sig.getMetadata(), sig.getArgListAttrs());
  return newSig;
}

/// Helper to perform the bulk of the lowering for `kgen.call` and
/// `kgen.call_indirect` ops.
static void lowerCallOpImpl(Operation *op, FuncType oldSig,
                            DebugInfo::DISubprogramAttr spAttr,
                            unsigned maxInlineSize) {

  ImplicitLocOpBuilder b(op->getLoc(), op);
  unsigned operandIndex = isa<CallIndirectOp>(op) ? 1 : 0;
  CallsiteTransform transform(b, op, lookupTargetInfo(op), spAttr,
                              maxInlineSize);
  TransformResult result = lowerSignature(oldSig, operandIndex, &transform);
  unsigned abiLowering = result.abiLowering;

  b.setInsertionPointAfter(op);
  OpResult res;
  if (op->getNumResults())
    res = op->getResult(0);

  // Now update the result, if needed.
  switch (abiLowering) {
  case DontPromote:
    break;
  case RemoveError: {
    // Never thrown.
    replaceUsesWithDummy(res, b,
                         SIMDAttr::getScalarBool(op->getContext(), false));
    // We can't just drop the result of an MLIR op, removing the i1 result
    // requires replacing the call.
    OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                         /*resultTypes=*/{});
    state.attributes = op->getAttrDictionary();
    auto newOp = b.create(state);
    op->erase();
    op = newOp;
    break;
  }
  case PromoteBoth: {
    // If the callee throws and both error and result were rewritten into a
    // variant, then we have to extract the relevant values from the variant.

    // Replace the i1 with a variant check.
    res.setType(result.newResultTypes[0]);
    auto isError = VariantIsOp::create(b, res, 0);
    res.replaceAllUsesExcept(isError, isError);

    auto ifOp = HLCF::IfOp::create(b, isError);
    b.createBlock(&ifOp.getThenRegion());
    POP::StoreOp::create(b, VariantGetOp::create(b, res, 0),
                         transform.errorOperand);
    HLCF::YieldOp::create(b);

    b.createBlock(&ifOp.getElseRegion());
    POP::StoreOp::create(b, VariantGetOp::create(b, res, 1),
                         transform.resultOperand);
    HLCF::YieldOp::create(b);
    break;
  }
  case PromoteResult:
  case PromoteResultRemoveError:
    // If we are ending up with a single function result, do it.
    if (!oldSig.isThrows() || abiLowering == PromoteResultRemoveError) {
      // If the callee doesn't throw, then we can directly return the result.
      replaceUsesWithDummy(res, b,
                           SIMDAttr::getScalarBool(op->getContext(), false));

      // Then just store the new callee result into the old memory result.
      res.setType(result.newResultTypes[0]);
      POP::StoreOp::create(b, res, transform.resultOperand);
      break;
    }
    [[fallthrough]];

  case PromoteError:
    // We are either promoting the normal result without promoting the error, or
    // we are promoting the error without the normal result.  This means we need
    // an i1 discriminator.
    //
    // As such, we need to reallocate the operation with a different call
    // operation because we're returning the error result and an i1.  This is
    // generic to handle CallOp and CallIndirectOp.
    OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                         result.newResultTypes);
    state.attributes = op->getAttrDictionary();
    Operation *newOp = b.create(state);
    res.replaceAllUsesWith(newOp->getResult(0));

    // Store the relevant result in the branch where it was valid.
    auto ifOp = HLCF::IfOp::create(b, newOp->getResult(0));
    Block *thenBlock = b.createBlock(&ifOp.getThenRegion());
    HLCF::YieldOp::create(b);
    bool errorOnly = abiLowering == PromoteError;
    Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
    HLCF::YieldOp::create(b);
    b.setInsertionPointToStart(errorOnly ? thenBlock : elseBlock);
    POP::StoreOp::create(b, newOp->getResult(1),
                         errorOnly ? transform.errorOperand
                                   : transform.resultOperand);
    op->erase();
    op = newOp;
    break;
  }

  if (auto callOp = dyn_cast<CallOp>(op)) {
    FuncType newSig =
        FuncType::get(FunctionType::get(op->getContext(), op->getOperandTypes(),
                                        result.newResultTypes),
                      result.newArgConventions, oldSig.getFnEffects(),
                      oldSig.getMetadata(), oldSig.getArgListAttrs());
    callOp.setCalleeAttr(SymbolConstantAttr::get(
        callOp.getCallee().getSymbol(), GeneratorType::get({}, newSig)));
  }
}

/// Emit IR for repacking the returned variant in the body of a throwing
/// function that we are currently lowering. This returns the new variant result
/// of the give type `newVariantTy`.
static Value repackFuncVariantResult(ReturnOp returnOp,
                                     VariantType newVariantTy, Value newResPtr,
                                     Value newErrPtr) {
  Value oldRetVal = returnOp.getOperand(0);
  ImplicitLocOpBuilder b(returnOp.getLoc(), returnOp);

  // We check the result is coming from. If we can guarantee that it's either an
  // error or not, we can just repack the error or the valid result.
  SIMDAttr isError;
  if (mlir::matchPattern(oldRetVal, mlir::m_Constant(&isError))) {
    if (!isError.getAsBool()) {
      // This is guaranteed to be a normal return.
      return VariantCreateOp::create(b, newVariantTy,
                                     POP::LoadOp::create(b, newResPtr), 1);
    }
    // This is guaranteed to be an error return.
    return VariantCreateOp::create(b, newVariantTy,
                                   POP::LoadOp::create(b, newErrPtr), 0);
  }

  // We can't guarantee what the result is, so we emit conditional variant
  // repacking. We create an HCLF::IfOp, with a condition checking if there is
  // no error (i.e. the then branch will handle normal return). The result of
  // this IfOp is what we will return.
  auto ifOp = HLCF::IfOp::create(b, newVariantTy, oldRetVal);

  // Populate the then branch (normal return).
  Block *thenBlock = b.createBlock(&ifOp.getThenRegion());
  b.setInsertionPointToStart(thenBlock);
  Value thenRes = VariantCreateOp::create(b, newVariantTy,
                                          POP::LoadOp::create(b, newErrPtr), 0);
  HLCF::YieldOp::create(b, thenRes);

  // Populate the else branch (error return).
  Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
  b.setInsertionPointToStart(elseBlock);
  Value elseRes = VariantCreateOp::create(b, newVariantTy,
                                          POP::LoadOp::create(b, newResPtr), 1);
  HLCF::YieldOp::create(b, elseRes);

  return ifOp.getResult(0);
}

static LogicalResult lowerFuncOp(FuncOp funcOp, unsigned maxInlineSize) {
  FuncType sig = funcOp.getFuncTypeGenerator().getBody();
  ImplicitLocOpBuilder b(funcOp.getLoc(), funcOp);
  b.setInsertionPoint(&funcOp.getBodyRegion().front().front());
  FuncTransform transform(b, funcOp, lookupTargetInfo(funcOp), maxInlineSize);
  TransformResult result = lowerSignature(sig, 0, &transform);
  FuncType newSig = FuncType::get(
      FunctionType::get(funcOp->getContext(),
                        funcOp.getBodyRegion().front().getArgumentTypes(),
                        result.newResultTypes),
      result.newArgConventions, sig.getFnEffects(), sig.getMetadata(),
      sig.getArgListAttrs());
  funcOp.setFuncTypeGenerator(
      GeneratorType::get(/*inputParamTypes=*/{}, newSig));
  funcOp.setLLVMArgMetadataAttr(
      ArrayAttr::get(funcOp.getContext(), transform.LLVMArgMetadata));
  if (result.abiLowering != DontPromote) {
    Block &body = funcOp.getBodyRegion().front();
    // Find all return sites in the function and rewrite them.
    body.walk([&](ReturnOp returnOp) {
      b.setInsertionPoint(returnOp);

      // Remove the error result entirely if asked for.
      if (result.abiLowering == RemoveError) {
        returnOp->eraseOperand(0);
        return;
      }
      // Promoting the result means we load the result and then return it.
      if (!newSig.isThrows() ||
          result.abiLowering == PromoteResultRemoveError) {
        auto newRes =
            POP::LoadOp::create(b, returnOp.getLoc(), transform.newResPtr);
        returnOp.setOperand(0, newRes);
        return;
      }

      // If the function throws and we rewrote both the error and the
      // byref_result, we need to potentially unpack and repack the
      // result/error variant.
      if (result.abiLowering == PromoteBoth) {
        auto newVariantTy = cast<VariantType>(newSig.getResults()[0]);
        Value newRetVal = repackFuncVariantResult(
            returnOp, newVariantTy, transform.newResPtr, transform.newErrPtr);
        returnOp.setOperand(0, newRetVal);
        return;
      }

      // Otherwise, we load either the error or the result, depending on which
      // got rewritten.
      Value toLoad =
          transform.newErrPtr ? transform.newErrPtr : transform.newResPtr;
      assert(toLoad && "should have been rewritten");
      Value newRes = POP::LoadOp::create(b, returnOp.getLoc(), toLoad);
      returnOp->insertOperands(1, newRes);
    });
  }
  return success(!transform.hasError);
}

/// Expand pack operands on `pop.external_call` into individual elements.
///
/// Before this, an `external_call` taking a pack looks like:
///   pop.external_call @foo(%pack) : (!kgen.struct<[i32, f32] isParamPack>)->()
///
/// After expansion:
///   %0 = kgen.struct.extract %pack[0] : <[i32, f32]>
///   %1 = kgen.struct.extract %pack[1] : <[i32, f32]>
///   pop.external_call @foo(%0, %1) : (i32, f32) -> ()
///
/// Recursively flatten a pack-typed value into individual scalar operands.
/// Nested packs (e.g. from variadic C calls like `printf(pack(fmt,
/// pack(*args))`) are fully expanded so that every operand passed to the
/// external call is a non-pack value.
static void flattenPackOperand(OpBuilder &builder, Location loc, Value pack,
                               StructType packStruct,
                               SmallVectorImpl<Value> &results) {
  ParamListAttr variadic = packStruct.getVariadicIfResolved();
  assert(variadic && "unresolved variadic pack on pop.external_call");

  for (auto [i, typeExpr] : llvm::enumerate(variadic.getValues())) {
    Type elemTy = cast<TypeParamAttr>(typeExpr).getMlirType();
    Value extracted =
        StructExtractOp::create(builder, loc, elemTy, pack,
                                IntegerAttr::get(builder.getIndexType(), i));

    if (auto nestedPackStruct = getIfParamPack(elemTy))
      flattenPackOperand(builder, loc, extracted, nestedPackStruct, results);
    else
      results.push_back(extracted);
  }
}

static void lowerExternalCallOpImpl(POP::ExternalCallOp op) {
  SmallVector<Value> newOperands;
  bool changed = false;

  OpBuilder builder(op);

  for (Value operand : op.getOperands()) {
    if (auto packTy = getIfParamPack(operand.getType())) {
      changed = true;
      flattenPackOperand(builder, op.getLoc(), operand, packTy, newOperands);
    } else {
      newOperands.push_back(operand);
      continue;
    }
  }

  if (changed)
    op.getOperandsMutable().assign(newOperands);
}

void LowerArgConventionsPass::runOnOperation() {
  FuncOp func = getOperation();
  if (failed(lowerFuncOp(func, maxInlineSize)))
    return signalPassFailure();

  // Lower the ops in the function body.
  DebugInfo::DISubprogramAttr spAttr = func.getSubprogramScope();
  func.walk([&](Operation *op) {
    if (auto callOp = dyn_cast<CallOp>(op))
      return lowerCallOpImpl(callOp,
                             callOp.getCalleeSignature().getInstantiatedBody(),
                             spAttr, maxInlineSize);
    if (auto callOp = dyn_cast<CallIndirectOp>(op))
      return lowerCallOpImpl(callOp, callOp.getCallee().getType().getBody(),
                             spAttr, maxInlineSize);
    if (auto extCall = dyn_cast<POP::ExternalCallOp>(op))
      return lowerExternalCallOpImpl(extCall);
  });

  // We must do this in a second pass, otherwise ops like kgen.call_indirect
  // would be difficult to identify for lowering (since their argument types
  // would be lowered already).
  mlir::AttrTypeReplacer replacer;
  TargetInfoAttr target = lookupTargetInfo(func);
  replacer.addReplacement([&](FuncType sig) {
    return lowerSignature(sig, target, spAttr, maxInlineSize);
  });
  auto metatype = TypeType::get(&getContext());
  replacer.addReplacement([&](TypeParamAttr type) {
    // Canonicalize metatypes.
    return TypeParamAttr::get(type.getMlirType(), metatype);
  });
  func.walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
  });
}
