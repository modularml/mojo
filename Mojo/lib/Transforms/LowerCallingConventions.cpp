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

#include "Mojo/CODialect/COOps.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCALLINGCONVENTIONS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerCallingConventionsPass
    : KGEN::impl::LowerCallingConventionsBase<LowerCallingConventionsPass> {
  void runOnOperation() override;
};
} // namespace

/// Filters out none types from the given range, and returns a flag indicating
/// if there were any in the input.
static std::pair<bool, SmallVector<Type>>
removeNoneTypes(TypeRange types, mlir::AttrTypeReplacer *replacer = nullptr) {
  SmallVector<Type> newTypes;
  for (Type type : types) {
    Type newType = replacer ? replacer->replace(type) : type;
    newType = newType ? newType : type;
    if (!StructType::isNoneOrEmpty(newType))
      newTypes.push_back(newType);
  }
  return {newTypes.size() != types.size(), std::move(newTypes)};
}

/// Remove none types from the results of debuginfo subroutine types as well.
static DebugInfo::DISubroutineType
removeDINoneResults(DebugInfo::DISubroutineType type) {
  // None types in the subroutine type will be wrapped in an unresolved type.
  SmallVector<DebugInfo::DIType> newTypes;
  for (DebugInfo::DIType type : type.getResultTypes()) {
    auto unresolved = dyn_cast<DebugInfo::DIUnresolvedMLIRType>(type);
    if (!unresolved || !StructType::isNoneOrEmpty(unresolved.getType()))
      newTypes.push_back(type);
  }
  if (newTypes.size() == type.getResultTypes().size())
    return type;
  return DebugInfo::DISubroutineType::get(type.getContext(),
                                          type.getArgumentTypes(), newTypes);
}

/// Lower a concrete pack to a struct.
static StructType lowerPackStructType(StructType type) {
  // Leave non-pack structs alone.
  if (!type.getIsParamPack())
    return type;
  // Strip off the isParamPack bit.
  return StructType::get(type.getContext(), type.getElementTypesVariadic(),
                         type.getIsMemoryOnly(), type.getMinAlignment(), false);
}

/// Lower `!kgen.variant` to a pair of `!pop.union` and a `!kgen.scalar`.
static StructType lowerVariantType(VariantType type) {
  SmallVector<Type> types = llvm::to_vector(type.getTypes());
  auto unionType = POP::UnionType::get(type.getContext(), types);
  // Get the unsigned integer dtype with the right width.
  size_t discrWidth = type.getDiscrSizeInBits();
  FailureOr<DType> discrType = DType::getInt(discrWidth, /*isSigned=*/false);
  assert(succeeded(discrType) && "expected to get a discriminant dtype");

  return StructType::get(
      {unionType, SIMDType::get(type.getContext(), /*size=*/1, *discrType)});
}

/// Lower `#kgen.variant` to a struct attribute.
static StructAttr lowerVariantAttr(VariantAttr attr) {
  StructType structType = lowerVariantType(attr.getType());

  auto elementTypes = *structType.getElementTypes();
  auto unionType = cast<POP::UnionType>(elementTypes.front());
  auto unionAttr = POP::UnionAttr::get(attr.getValue(), unionType);

  auto scalarType = cast<SIMDType>(elementTypes.back());
  auto discrAttr = KGEN::SIMDAttr::get(attr.getIndex(), scalarType);

  return StructAttr::get({unionAttr, discrAttr}, structType);
}

static void lowerCreateRegStubOp(IRRewriter &b, CreateRegStubOp op) {
  FuncTypeGeneratorType resSigGen = op.getResult().getType();
  auto calleeSigGen = cast<FuncTypeGeneratorType>(op.getCallee().getType());
  SymbolConstantAttr callee = cast<SymbolConstantAttr>(op.getCallee());

  if (calleeSigGen == resSigGen) {
    // Signatures are equal, no need to transform the arguments.
    // Directly lower to CreateClosureOp.
    b.replaceOpWithNewOp<CreateClosureOp>(op, callee);
    return;
  }

  // The created closure is a synthesized operation. No debug scopes should be
  // carried into it from the originating op.
  LocationAttr loc = DebugInfo::stripDebugScopesRecursively(op->getLoc());
  auto closureWrapper = StageClosureOp::create(b, loc, resSigGen);
  closureWrapper.setCallLocAttr(op->getLoc());
  Block *body = b.createBlock(&closureWrapper.getBodyRegion());

  // Bitcast the function arguments and load the inputs.
  FuncType resSig = resSigGen.getBody();
  FuncType calleeSig = calleeSigGen.getBody();
  SmallVector<Value> insValues;
  SmallVector<Value> outsPointers;
  bool promotedOutputs =
      resSig.hasMemoryOnlyResult() && !calleeSig.hasMemoryOnlyResult();
  for (auto [i, rawArgTy] : llvm::enumerate(resSig.getValues().getInputs())) {
    ArgConvention conv = resSig.getArgConvention(i);
    Value arg = body->addArgument(rawArgTy, loc);
    Type argTy = op.getOriginalArgType(i);
    // Bitcast to the original type if needed.
    if (rawArgTy != argTy)
      arg = POP::PointerBitcastOp::create(b, loc, argTy, arg);

    if (promotedOutputs && conv == ArgConvention::ByRefResult) {
      // Output was a memory argument but got promoted to a register output.
      // Store will be inserted after the function.
      outsPointers.push_back(arg);
    } else if (arg.getType() == op.getCalleeArgType(i)) {
      // Input type remained the same.
      // Just pass it directly to the function.
      insValues.push_back(arg);
    } else {
      // Input was a memory argument but got lowered to a register argument.
      // Load it before passing it to the function.
      Value loadArg = POP::LoadOp::create(b, loc, arg);
      insValues.push_back(loadArg);
    }
  }

  // Insert the call.
  CallOp callOp = CallOp::create(b, loc, callee, insValues);

  // Add stores for the call outputs.
  for (auto [resultVal, resultPtr] :
       llvm::zip(callOp->getResults(), outsPointers))
    POP::StoreOp::create(b, loc, resultVal, resultPtr);

  // Add the terminator (KGEN::ReturnOp).
  ReturnOp::create(b, loc);

  b.replaceOp(op, closureWrapper);
}

/// We need to roll our own walk function because we are converting types and
/// operations at the same time. We need a pre-order walk to convert argument
/// types before their users, but we are also erasing ops with regions.
static void recursiveRewrite(Operation *op, mlir::AttrTypeReplacer &replacer);

/// Rewrite a single operation, recursing if it has regions.
static void rewriteFn(Operation *op, mlir::AttrTypeReplacer &replacer) {
  // Recursively replace all signatures in the operation. This will handle the
  // signatures of `kgen.func`, `kgen.stage_closure`, and `co.execute`.
  replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/true,
                             /*replaceTypes=*/true);

  // Handle exiting terminators.
  if (isa<ReturnOp, HLCF::YieldOp, HLCF::BreakOp>(op)) {
    // Remove none results.
    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands())
      if (!StructType::isNoneOrEmpty(operand.getType()))
        newOperands.push_back(operand);
    op->setOperands(newOperands);
    return;
  }

  // Handle `hlcf.if`, `hlcf.loop`, `kgen.call`, `kgen.call_indirect`, and
  // `co.get_results`.
  if (isa<HLCF::IfOp, HLCF::LoopOp, CallOp, CallIndirectOp, CO::GetResultsOp,
          CO::AwaitOp, CO::InvokeOp, CO::HotInvokeOp>(op)) {
    auto [anyNone, newResults] = removeNoneTypes(op->getResultTypes());
    // Exit early if there are no none results.
    if (!anyNone)
      return recursiveRewrite(op, replacer);
    OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                         newResults);
    // Micro-optimization: set the DictionaryAttr directly to avoid a re-hash.
    state.attributes = op->getAttrDictionary();
    for (Region &region : op->getRegions())
      state.addRegion()->takeBody(region);
    auto builder = OpBuilder(op);
    Operation *newOp = builder.create(state);

    // Lazily construct a none constant only when needed.
    Value noneImpl;
    auto getNone = [&](Type type) {
      if (!noneImpl || noneImpl.getType() != type) {
        TypedAttr attr;
        if (isa<KGEN::NoneType>(type))
          attr = NoneAttr::get(op->getContext());
        else // empty register-passable struct.
          attr = StructAttr::get({}, cast<StructType>(type));
        noneImpl = ParamConstantOp::create(builder, op->getLoc(), attr);
      }
      return noneImpl;
    };

    unsigned newResultIdx = 0;
    for (Value result : op->getResults()) {
      if (!StructType::isNoneOrEmpty(result.getType()))
        result.replaceAllUsesWith(newOp->getResult(newResultIdx++));
      else if (!result.use_empty())
        result.replaceAllUsesWith(getNone(result.getType()));
    }
    assert(newResultIdx == newOp->getNumResults());
    op->erase();
    // Recurse on the new op.
    recursiveRewrite(newOp, replacer);
    return;
  }

  // Handle pack operations. Be mindful here of the ODS methods because the
  // operand and result types will have already been lowered to `!kgen.struct`.
  IRRewriter b{OpBuilder(op)};
  if (auto load = dyn_cast<StructLoadIndirectOp>(op)) {
    SmallVector<Value> elements;
    auto types =
        *cast<StructType>(load.getStructValue().getType()).getElementTypes();
    elements.reserve(types.size());
    for (auto [i, _] : llvm::enumerate(types)) {
      auto ptr = StructExtractOp::create(b, op->getLoc(), load.getStructValue(),
                                         b.getIndexAttr(i));
      elements.push_back(POP::LoadOp::create(b, op->getLoc(), ptr));
    }
    b.replaceOpWithNewOp<StructCreateOp>(load, load->getResultTypes(),
                                         elements);
    return;
  }

  // Handle variant operations.
  if (auto create = dyn_cast<VariantCreateOp>(op)) {
    auto structType = cast<StructType>(create->getResultTypes().front());
    auto elementTypes = *structType.getElementTypes();
    auto unionType = cast<POP::UnionType>(elementTypes.front());
    auto discrType = cast<SIMDType>(elementTypes.back());
    Value unionVal = POP::UnionWrapOp::create(b, op->getLoc(), unionType,
                                              create.getOperand());
    Value discrVal = ParamConstantOp::create(
        b, op->getLoc(), KGEN::SIMDAttr::get(create.getIndex(), discrType));
    b.replaceOpWithNewOp<StructCreateOp>(op, structType,
                                         ValueRange{unionVal, discrVal});
    return;
  }
  if (auto is = dyn_cast<VariantIsOp>(op)) {
    Value variantVal = is->getOperand(0);
    auto structType = cast<StructType>(variantVal.getType());
    auto discrType = cast<SIMDType>((*structType.getElementTypes()).back());
    Value discrVal =
        StructExtractOp::create(b, op->getLoc(), variantVal, b.getIndexAttr(1));
    Value discrCst = ParamConstantOp::create(
        b, op->getLoc(), KGEN::SIMDAttr::get(is.getIndex(), discrType));
    Value isEq = POP::CmpOp::create(b, op->getLoc(), KGEN::CmpPredicate::EQ,
                                    discrVal, discrCst);
    b.replaceOp(op, isEq);
    return;
  }
  if (auto get = dyn_cast<VariantGetOp>(op)) {
    Value variantVal = get->getOperand(0);
    Value unionVal =
        StructExtractOp::create(b, op->getLoc(), variantVal, b.getIndexAttr(0));
    b.replaceOpWithNewOp<POP::UnionUnwrapOp>(op, get.getType(), unionVal);
    return;
  }
  if (auto bitcast = dyn_cast<POP::VariantBitcastOp>(op)) {
    Value variantPtr = bitcast->getOperand(0);
    Value unionPtr =
        StructGEPOp::create(b, op->getLoc(), variantPtr, /*index=*/0);
    b.replaceOpWithNewOp<POP::UnionBitcastOp>(op, bitcast.getType(), unionPtr);
    return;
  }
  if (auto gep = dyn_cast<POP::VariantDiscrGEPOp>(op)) {
    Value variantPtr = gep->getOperand(0);
    b.replaceOpWithNewOp<StructGEPOp>(op, variantPtr, /*index=*/1);
    return;
  }

  // Handle thunking operations.
  if (auto createRegStubOp = dyn_cast<CreateRegStubOp>(op)) {
    lowerCreateRegStubOp(b, createRegStubOp);
    return;
  }

  // In the general case, try to recurse on the op.
  recursiveRewrite(op, replacer);
}

/// Out-of-line definition.
static void recursiveRewrite(Operation *op, mlir::AttrTypeReplacer &replacer) {
  for (Region &region : op->getRegions())
    if (!region.empty())
      for (Operation &op : llvm::make_early_inc_range(region.front()))
        rewriteFn(&op, replacer);
}

void LowerCallingConventionsPass::runOnOperation() {
  mlir::AttrTypeReplacer replacer;

  // Lower the signature results by replacing all the `!kgen.none` results in a
  // signature. This will also set the signature to non-throwing, and erase
  // `byref_result` argument conventions.
  auto lowerResult = [&replacer](FuncType signature) -> FuncType {
    auto [anyNone, newResults] =
        removeNoneTypes(signature.getResults(), &replacer);
    // Micro-optimization: don't hash a new type if it won't change.
    if (!anyNone)
      return signature;

    // At this point in lowering, we turn byref_error into read_reg, because the
    // normal "throws" convention is done.
    SmallVector<ArgConvention> newArgConventions;
    for (auto conv : signature.getArgConventions()) {
      if (conv == ArgConvention::ByRefError)
        conv = ArgConvention::ImmReg;
      newArgConventions.push_back(conv);
    }

    return FuncType::get(
        FunctionType::get(signature.getContext(), signature.getArguments(),
                          newResults),
        newArgConventions, signature.getFnEffects().setThrows(false));
  };
  replacer.addReplacement(lowerResult);
  replacer.addReplacement(removeDINoneResults);
  replacer.addReplacement(lowerPackStructType);
  replacer.addReplacement(lowerVariantType);
  replacer.addReplacement(lowerVariantAttr);
  // Do not recurse into LinkageNameAttr sub-elements: by this point the
  // rename pass has either removed it or left it with a resolved StringAttr
  // name.  Either way, the sub-elements (possibly parametric pval attrs
  // from the interpreter) must not be walked by the replacer.
  replacer.addReplacement(
      [](LinkageNameAttr lna)
          -> std::optional<std::pair<Attribute, WalkResult>> {
        return std::make_pair(Attribute(lna), WalkResult::skip());
      });

  rewriteFn(getOperation(), replacer);
}
