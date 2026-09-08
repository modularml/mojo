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

#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPUtils.h"

#include "Mojo/Interpreter/ParametricInterpreterState.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace M;
using namespace KGEN;
using namespace POP;

static llvm::Type *convertDTypeToLLVM(KGENDType dtype,
                                      llvm::LLVMContext &llvmCtx,
                                      TargetInfoAttr target) {
  if (dtype.isInt())
    return llvm::IntegerType::get(llvmCtx, dtype.getIntegerWidthInBits());

  switch (dtype.getValue()) {
  case DType::kBool:
    return llvm::IntegerType::get(llvmCtx, 1);
  case DType::f32:
    return llvm::Type::getFloatTy(llvmCtx);
  case DType::f64:
    return llvm::Type::getDoubleTy(llvmCtx);
  case KGENDType::index:
  case KGENDType::uindex:
    if (target)
      return llvm::IntegerType::get(llvmCtx, target.resolveIndexBitWidth());
    return {};
  default:
    return {};
  }
}

// This attempts to lower the specified MLIR type into an LLVM IR rep.
static llvm::Type *convertTypeToLLVM(Type type, llvm::LLVMContext &llvmCtx,
                                     TargetInfoAttr target) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return llvm::Type::getIntNTy(llvmCtx, intType.getWidth());

  // Use target info to lower 'index' to the right LLVM bitwidth if available.
  if (isa<IndexType>(type) && target)
    return llvm::Type::getIntNTy(llvmCtx, target.resolveIndexBitWidth());

  // Lower SIMD types to LLVM vectors.
  if (auto simd = dyn_cast<SIMDType>(type)) {
    auto optDType = simd.getResolvedDType();
    auto optSize = simd.getResolvedSize();
    if (optDType && optSize) {
      if (auto dtypeType = convertDTypeToLLVM(*optDType, llvmCtx, target)) {
        if (*optSize == 1) // simd<1> is a scalar.
          return dtypeType;
        return llvm::VectorType::get(dtypeType, *optSize, /*scalable*/ false);
      }
    }
  }

  return {};
}

// This attempts to lower the specified operand value into an LLVM IR
// representation that can be passed to a llvm::CallInst.
static llvm::Constant *convertAttrToLLVM(TypedAttr attr,
                                         llvm::LLVMContext &llvmCtx,
                                         TargetInfoAttr target) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    llvm::Type *type = convertTypeToLLVM(attr.getType(), llvmCtx, target);
    if (!type)
      return {};

    return llvm::ConstantInt::get(type, intAttr.getValue());
  }

  // Convert SIMD values.
  if (auto simdAttr = dyn_cast<SIMDAttr>(attr)) {
    SmallVector<llvm::Constant *> elts;
    for (auto elt : simdAttr.getValues()) {
      llvm::Constant *value = nullptr;
      if (elt.getDType().isBool())
        value = llvm::ConstantInt::getBool(llvmCtx, elt.getBoolVal());
      else if (elt.getDType().isInt())
        value = llvm::ConstantInt::get(llvmCtx, elt.getIntVal());
      else if (elt.getDType().isFloat())
        value = llvm::ConstantFP::get(llvmCtx, elt.getFloatVal());
      else if (elt.getDType().isUIndex() && target) {
        value = llvm::ConstantInt::get(
            llvmCtx,
            elt.getIntVal().zextOrTrunc(target.resolveIndexBitWidth()));
      } else if (elt.getDType().isIndex() && target) {
        value = llvm::ConstantInt::get(
            llvmCtx,
            elt.getIntVal().sextOrTrunc(target.resolveIndexBitWidth()));
      }

      if (!value)
        return {};
      elts.push_back(value);
    }

    // Handle SIMD<1> as a scalar.
    if (elts.size() == 1)
      return elts[0];
    return llvm::ConstantVector::get(elts);
  }

  // Otherwise we don't know what it is.
  return {};
}

/// Given the result of constant folding a function call to some LLVM value, see
/// if we can convert it back into a MLIR attribute with the specified MLIR
/// type.  If not, return null.
static TypedAttr convertLLVMToAttr(llvm::Constant *value, Type type) {
  // Scalar integer result type.
  if (isa<IntegerType, IndexType>(type))
    if (auto ci = dyn_cast<llvm::ConstantInt>(value))
      return IntegerAttr::get(type, ci->getValue());

  // Expecting a pop.simd result type, which could be a scalar or a vector.
  if (auto simdType = dyn_cast<SIMDType>(type)) {
    auto dtype = simdType.getResolvedDType();
    if (!dtype)
      return {};

    if (isa<llvm::ConstantAggregateZero>(value))
      return SIMDAttr::getZeroValue(simdType);

    SmallVector<DTypeValue> values;

    auto addValue = [&](llvm::Value *value) -> LogicalResult {
      if (auto ci = dyn_cast<llvm::ConstantInt>(value)) {
        if (dtype->isBool())
          values.push_back(DTypeValue(!ci->getValue().isZero(), *dtype));
        else
          values.push_back(DTypeValue(ci->getValue(), *dtype));
      } else if (auto cf = dyn_cast<llvm::ConstantFP>(value)) {
        values.push_back(DTypeValue(cf->getValue(), *dtype));
      } else {
        return failure();
      }
      return success();
    };

    // Scalar float/int result type.
    if (auto cf = dyn_cast<llvm::ConstantFP>(value)) {
      values.push_back(DTypeValue(cf->getValue(), *dtype));
    } else if (auto ci = dyn_cast<llvm::ConstantInt>(value)) {
      if (dtype->isBool())
        values.push_back(DTypeValue(!ci->getValue().isZero(), *dtype));
      else
        values.push_back(DTypeValue(ci->getValue(), *dtype));
    } else if (auto simdValue = dyn_cast<llvm::ConstantVector>(value)) {
      for (auto i = simdValue->op_begin(), e = simdValue->op_end(); i != e;
           ++i) {
        if (failed(addValue(*i)))
          return {};
      }
    } else if (auto data = dyn_cast<llvm::ConstantDataSequential>(value)) {
      for (size_t i = 0, e = data->getNumElements(); i != e; ++i) {
        if (failed(addValue(data->getElementAsConstant(i))))
          return {};
      }

    } else {
      return {};
    }

    return SIMDAttr::get(values, simdType);
  }

  return {};
}

/// Get the declaration of an overloaded llvm intrinsic. First we get the
/// overloaded argument types and/or result type from the CallIntrinsicOp, and
/// then use those to get the correct declaration of the overloaded intrinsic.
static llvm::Function *
getOverloadedDeclaration(ArrayRef<llvm::Type *> operandTypes,
                         llvm::Type *resType, llvm::Intrinsic::ID id,
                         llvm::Module *module) {
  // ATM we do not support variadic intrinsics.
  llvm::FunctionType *ft =
      llvm::FunctionType::get(resType, operandTypes, false);

  SmallVector<llvm::Type *, 8> overloadedArgTys;
  if (!llvm::Intrinsic::isSignatureValid(id, ft, overloadedArgTys)) {
    return {};
  }

  return llvm::Intrinsic::getOrInsertDeclaration(module, id, overloadedArgTys);
}

template <typename T>
static std::string stringize(const T &value) {
  SmallVector<char> data;
  llvm::raw_svector_ostream os(data);
  os << value;
  return os.str().str();
}

static SmallVector<Attribute> expandOperands(ArrayRef<Attribute> args) {
  SmallVector<Attribute> operands;
  operands.reserve(args.size());
  for (auto value : args) {
    auto packAttr = dyn_cast<KGEN::StructAttr>(value);
    if (packAttr && packAttr.getType().getIsParamPack()) {
      operands.append(packAttr.getValues().begin(), packAttr.getValues().end());
    } else {
      operands.push_back(value);
    }
  }
  return operands;
}

//===----------------------------------------------------------------------===//
// InterpMemcpyOp
//===----------------------------------------------------------------------===//
static ErrorTreeOrSuccess interpretMemcpy(Location loc,
                                          ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  SmallVector<Attribute> values = expandOperands(operands[0]);
  if (values.size() != 3) {
    return ErrorTree(
        loc,
        "interpreting llvm.memcpy takes pack of 3: dst addr, src addr, count");
  }

  return POP::interpretMemcpy(values[0], values[1], values[2], loc, state);
}

static ErrorTreeOrSuccess interpretPOPIntrinsics(StringAttr name, Location loc,
                                                 ArrayRef<Attribute> operands,
                                                 InterpreterState &state,
                                                 bool &interpreted) {
  if (name == "llvm.memcpy") {
    interpreted = true;
    return interpretMemcpy(loc, operands, state);
  }

  return success();
}

// Interpreting an LLVM Intrinsic is a bit awkward.  We need to create an LLVM
// call operation, and then ask llvm to fold it for us.
ErrorTreeOrSuccess CallLLVMIntrinsicOp::interpret(ArrayRef<Attribute> operands,
                                                  InterpreterState &state) {
  // Check to see if we can resolve which intrinsic is being called.  If not,
  // then we can't fold it.
  auto name = dyn_cast<StringAttr>(getIntrinAttr());
  if (!name)
    return ErrorTree(getLoc(), "unknown intrinsic opcode");

  // Handle some special case used as pop intrinsics first.
  bool interpreted = false;
  ErrorTreeOrSuccess popInterpResult =
      interpretPOPIntrinsics(name, getLoc(), operands, state, interpreted);
  if (interpreted)
    return popInterpResult;

  // See if LLVM knows what this is.
  llvm::Intrinsic::ID id = llvm::Intrinsic::lookupIntrinsicID(name.strref());
  if (!id)
    return ErrorTree(getLoc(),
                     "could not find LLVM intrinsic: '" + name.str() + "'");

  llvm::LLVMContext llvmContext;

  // Figure out the LLVM representation for all the operands.
  SmallVector<llvm::Value *> loweredOperands;
  SmallVector<llvm::Constant *> loweredOperandsCst;
  for (auto v : expandOperands(operands)) {
    // Try to understand what this value is.
    auto typedOp = ::dyn_cast<TypedAttr>(v);
    if (!typedOp)
      return ErrorTree(getLoc(), "LLVM intrinsic call has unknown operand: " +
                                     stringize(v));
    llvm::Constant *loweredValue =
        convertAttrToLLVM(typedOp, llvmContext, state.getTarget());
    if (!loweredValue)
      return ErrorTree(getLoc(), "LLVM intrinsic operand has unknown value: " +
                                     stringize(typedOp));

    loweredOperands.push_back(loweredValue);
    loweredOperandsCst.push_back(loweredValue);
  }

  // Compute the LLVM result type.
  if (getNumResults() == 0)
    return ErrorTree(getLoc(),
                     "cannot constant fold zero-result LLVM intrinsic: " +
                         name.str());
  if (getNumResults() != 1)
    return ErrorTree(getLoc(), "LLVM intrinsic operand has multiple results: " +
                                   name.str());
  llvm::Type *resultTy =
      convertTypeToLLVM(getResult(0).getType(), llvmContext, state.getTarget());
  if (!resultTy)
    return ErrorTree(getLoc(), "LLVM intrinsic has unknown result type: " +
                                   stringize(getResult(0).getType()));

  // Try using "ConstantFoldIntrinsic" first - if it works, it avoids us
  // having to create a bunch of IR.
  llvm::Constant *result = nullptr;
  if (loweredOperands.size() == 2) {
    llvm::DataLayout dataLayout(state.getTarget().getDataLayout().toString());
    result = ConstantFoldIntrinsic(
        id, {loweredOperandsCst[0], loweredOperandsCst[1]}, resultTy,
        dataLayout);
  }

  if (!result) {
    // Otherwise, we handle this by creating a module with a call to the
    // intrinsic.
    llvm::Module module("folding", llvmContext);

    // Resolve the overloaded (or not) callee for the intrinsic call.
    llvm::Function *fn = nullptr;
    if (!llvm::Intrinsic::isOverloaded(id)) {
      fn = llvm::Intrinsic::getOrInsertDeclaration(&module, id, {});
      assert(fn && "should always succeed");
    } else {
      SmallVector<llvm::Type *, 8> argTys;
      for (auto val : loweredOperands)
        argTys.push_back(val->getType());
      fn = getOverloadedDeclaration(argTys, resultTy, id, &module);
      if (!fn)
        return ErrorTree(
            getLoc(),
            "could not find overloaded declaration of LLVM intrinsic: " +
                name.str());
    }

    // Okay, we got the prototype for the intrinsic to call.  Generate a call
    // to it in another function.  We need a basic block to hold the call -
    // just abuse the intrinsic itself to own it.
    auto *block = llvm::BasicBlock::Create(llvmContext, Twine(), fn);

    auto *call =
        llvm::CallInst::Create(fn->getFunctionType(), fn, loweredOperands,
                               /*name*/ Twine(), block);

    // Now that we have a call, we can finally try to constant fold!
    // NOTE: we aren't passing in a TargetLibraryInfo, which prevents folding
    // random libc functions, but that seems ok.
    result = llvm::ConstantFoldCall(call, fn, loweredOperandsCst,
                                    /*TLI*/ nullptr);
  }

  if (!result)
    return ErrorTree(getLoc(),
                     "LLVM could not constant fold intrinsic: " + name.str());

  // If we got something back from LLVM, repackage it back up for MLIR to look
  // at.
  auto attr = convertLLVMToAttr(result, getResult(0).getType());
  if (!attr)
    return ErrorTree(getLoc(), "could not convert result of intrinsic: " +
                                   stringize(*result));

  if (attr.getType() != getResult(0).getType())
    return ErrorTree(getLoc(),
                     "result type mismatch: " + stringize(attr.getType()) +
                         " != " + stringize(getResult(0).getType()));

  return state.mapResults(attr);
}

ErrorTreeOrSuccess
CallLLVMIntrinsicOp::parametric_interpret(ArrayRef<Attribute> operands,
                                          ParametricInterpreterState &state) {

  // Check to see if we can resolve which intrinsic is being called.  If not,
  // then we can't fold it.
  auto name = dyn_cast<StringAttr>(state.getReboundAttribute(getIntrinAttr()));
  if (!name)
    return ErrorTree(getLoc(), "unknown intrinsic opcode");

  // Handle some special case used as pop intrinsics first.
  bool interpreted = false;
  ErrorTreeOrSuccess popInterpResult =
      interpretPOPIntrinsics(name, getLoc(), operands, state, interpreted);
  if (interpreted)
    return popInterpResult;

  // See if LLVM knows what this is.
  llvm::Intrinsic::ID id = llvm::Intrinsic::lookupIntrinsicID(name.strref());
  if (!id)
    return ErrorTree(getLoc(),
                     "could not find LLVM intrinsic: '" + name.str() + "'");

  llvm::LLVMContext llvmContext;

  // Figure out the LLVM representation for all the operands.
  SmallVector<llvm::Value *> loweredOperands;
  SmallVector<llvm::Constant *> loweredOperandsCst;
  for (auto v : expandOperands(operands)) {
    // Try to understand what this value is.
    auto typedOp = ::dyn_cast<TypedAttr>(v);
    if (!typedOp)
      return ErrorTree(getLoc(), "LLVM intrinsic call has unknown operand: " +
                                     stringize(v));
    llvm::Constant *loweredValue =
        convertAttrToLLVM(typedOp, llvmContext, state.getTarget());
    if (!loweredValue)
      return ErrorTree(getLoc(), "LLVM intrinsic operand has unknown value: " +
                                     stringize(typedOp));
    loweredOperands.push_back(loweredValue);
    loweredOperandsCst.push_back(loweredValue);
  }

  // Compute the LLVM result type.
  if (getNumResults() == 0)
    return ErrorTree(getLoc(),
                     "cannot constant fold zero-result LLVM intrinsic: " +
                         name.str());
  if (getNumResults() != 1)
    return ErrorTree(getLoc(), "LLVM intrinsic operand has multiple results: " +
                                   name.str());
  Type resultType = state.getReboundType(getResult(0).getType());
  llvm::Type *resultTy =
      convertTypeToLLVM(resultType, llvmContext, state.getTarget());
  if (!resultTy)
    return ErrorTree(getLoc(), "LLVM intrinsic has unknown result type: " +
                                   stringize(resultType));

  // Try using "ConstantFoldIntrinsic" first - if it works, it avoids us
  // having to create a bunch of IR.
  llvm::Constant *result = nullptr;
  if (loweredOperands.size() == 2) {
    llvm::DataLayout dataLayout(state.getTarget().getDataLayout().toString());
    result = ConstantFoldIntrinsic(
        id, {loweredOperandsCst[0], loweredOperandsCst[1]}, resultTy,
        dataLayout);
  }

  if (!result) {
    // Otherwise, we handle this by creating a module with a call to the
    // intrinsic.
    llvm::Module module("folding", llvmContext);

    // Resolve the overloaded (or not) callee for the intrinsic call.
    llvm::Function *fn = nullptr;
    if (!llvm::Intrinsic::isOverloaded(id)) {
      fn = llvm::Intrinsic::getOrInsertDeclaration(&module, id, {});
      assert(fn && "should always succeed");
    } else {
      SmallVector<llvm::Type *, 8> argTys;
      for (auto val : loweredOperands)
        argTys.push_back(val->getType());
      fn = getOverloadedDeclaration(argTys, resultTy, id, &module);
      if (!fn)
        return ErrorTree(
            getLoc(),
            "could not find overloaded declaration of LLVM intrinsic: " +
                name.str());
    }

    // Okay, we got the prototype for the intrinsic to call.  Generate a call
    // to it in another function.  We need a basic block to hold the call -
    // just abuse the intrinsic itself to own it.
    auto *block = llvm::BasicBlock::Create(llvmContext, Twine(), fn);

    auto *call =
        llvm::CallInst::Create(fn->getFunctionType(), fn, loweredOperands,
                               /*name*/ Twine(), block);

    // Now that we have a call, we can finally try to constant fold!
    // NOTE: we aren't passing in a TargetLibraryInfo, which prevents folding
    // random libc functions, but that seems ok.
    result = llvm::ConstantFoldCall(call, fn, loweredOperandsCst,
                                    /*TLI*/ nullptr);
  }

  if (!result)
    return ErrorTree(getLoc(),
                     "LLVM could not constant fold intrinsic: " + name.str());

  // If we got something back from LLVM, repackage it back up for MLIR to look
  // at.
  auto attr = convertLLVMToAttr(result, resultType);
  if (!attr)
    return ErrorTree(getLoc(), "could not convert result of intrinsic: " +
                                   stringize(*result));

  if (attr.getType() != resultType)
    return ErrorTree(getLoc(),
                     "result type mismatch: " + stringize(attr.getType()) +
                         " != " + stringize(resultType));

  (void)state.mapResults(attr);

  return success();
}
