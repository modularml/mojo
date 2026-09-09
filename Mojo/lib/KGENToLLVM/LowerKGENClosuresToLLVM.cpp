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

#include "CABICallHelpers.h"
#include "LLVMLoweringUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

#define DEBUG_TYPE "lower-runtime-closures"

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERRUNTIMECLOSURES
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerRuntimeClosuresPass
    : M::KGEN::impl::LowerRuntimeClosuresBase<LowerRuntimeClosuresPass> {
  using LowerRuntimeClosuresBase::LowerRuntimeClosuresBase;

  void runOnOperation() override;
};
} // namespace

struct CreateClosureTypes {
public:
  CreateClosureTypes() {}
  static LogicalResult
  createClosureTypes(CreateClosureTypes &types, CreateClosureOp op,
                     const mlir::LLVMTypeConverter &typeConverter);
  SmallVector<Type> boundArgTypes;
  Type opaquePtrType;
  /// Two element struct where first element is void* to function
  /// and second member is struct of captured elements.
  Type liftedFunctionCaptureType;

  /// The opaque closure struct type is a struct of two opaque pointers
  /// The first element points to a function interface method.
  /// The second points to captured state + lifted function.
  Type unpackingFunctionAndCapturesType;
};

LogicalResult CreateClosureTypes::createClosureTypes(
    CreateClosureTypes &types, CreateClosureOp op,
    const mlir::LLVMTypeConverter &typeConverter) {
  MLIRContext *context = op.getContext();
  types.opaquePtrType = LLVM::LLVMPointerType::get(context);
  types.boundArgTypes.reserve(op.getCaptures().size());
  for (Value arg : op.getCaptures()) {
    Type ty = typeConverter.convertType(arg.getType());
    if (!ty)
      return failure();
    types.boundArgTypes.push_back(ty);
  }
  types.liftedFunctionCaptureType =
      LLVM::LLVMStructType::getLiteral(context, types.boundArgTypes);
  types.unpackingFunctionAndCapturesType = LLVM::LLVMStructType::getLiteral(
      context, {types.opaquePtrType, types.opaquePtrType});
  return success();
}

struct CreateRuntimeClosureOpConversion
    : public ConvertPOPToLLVMPattern<CreateClosureOp> {

  CreateRuntimeClosureOpConversion(SymbolTable &symTable,
                                   POPToLLVMTypeConverter &typeConverter)
      : ConvertPOPToLLVMPattern<CreateClosureOp>(typeConverter),
        symtab(symTable) {}

private:
  SymbolTable &symtab;
  unsigned nameIndex = 0;

  LLVM::LLVMFuncOp
  generateWrapperFunction(CreateClosureOp op,
                          ConversionPatternRewriter &rewriter) const {
    FuncTypeGeneratorType calleeSignature = op.getCalleeType();
    FunctionType calleeType = calleeSignature.getBody().getValues();
    MLIRContext *context = getContext();

    // the signature if the wrapper is (opaquePointer, PM, ..., PN) -> R, where
    // M is the number of captures and N is the number of arguments of the
    // lifted function
    Type packedResTy = LLVM::LLVMVoidType::get(context);
    if (calleeType.getNumResults())
      packedResTy =
          getTypeConverter()->packFunctionResults(calleeType.getResults());

    // Create input types
    auto opaquePtrType = LLVM::LLVMPointerType::get(context);
    SmallVector<Type> wrapperFnArgTypes;
    wrapperFnArgTypes.push_back(opaquePtrType);
    size_t captureCount = op.getCaptures().size();
    ArrayRef<Type> noncapturedInputs = calleeType.getInputs().slice(
        captureCount, calleeType.getNumInputs() - captureCount);
    for (Type inpTy : noncapturedInputs) {
      Type ty = convertType(inpTy);
      if (!ty)
        return {};
      wrapperFnArgTypes.push_back(ty);
    }
    auto wrapperFnType = LLVM::LLVMFunctionType::get(
        context, packedResTy, wrapperFnArgTypes, /*varArg=*/false);

    // Create the body of the wrapper function
    Block *wrapperFnBody = new Block;
    rewriter.setInsertionPointToStart(wrapperFnBody);

    // Add wrapper function arguments to the block arguments
    for (Type argTy : wrapperFnArgTypes)
      wrapperFnBody->addArgument(argTy, op.getLoc());
    rewriter.clearInsertionPoint();

    LLVM::LLVMFuncOp wrapperFn = createLLVMFunc(
        rewriter, getTypeConverter()->getTarget(), op.getLoc(),
        rewriter.getStringAttr(
            "closure_wrapper_fn_" +
            Twine(const_cast<CreateRuntimeClosureOpConversion *>(this)
                      ->nameIndex++)),
        wrapperFnType, LLVM::Linkage::Internal);

    // If possible, we need to add a subprogram scope to the new function.
    auto scope = DebugInfo::extractScopeFrom<DebugInfo::DISubprogramAttr>(
        op.getLoc(), DebugInfo::LocWalkPolicy::CalleePriority);
    if (scope) {
      // Use unresolved types now for simplicity, these will get resolved during
      // compilation.
      auto mapUnresolvedType = [&](Type type) -> DebugInfo::DIType {
        return DebugInfo::DIUnresolvedMLIRType::get(type);
      };
      auto spType = DebugInfo::DISubroutineType::get(
          op->getContext(), map_to_vector(wrapperFnArgTypes, mapUnresolvedType),
          map_to_vector(wrapperFnType.getReturnTypes(), mapUnresolvedType));

      auto fileLoc = op.getLoc()->findInstanceOf<FileLineColLoc>();
      auto sourceName = DebugInfo::SourceNameAttr::get(
          "closure_wrapper_fn." + Twine(nameIndex - 1), scope.getSourceName());
      wrapperFn->setLoc(FusedLoc::get(
          op.getContext(), Location(fileLoc),
          scope.cloneWith(sourceName, wrapperFn.getSymNameAttr(), spType)));
    }

    wrapperFn.getBody().push_back(wrapperFnBody);
    return wrapperFn;
  }

  /// The body of the closure wrapper function should consist of casting the
  /// opaque pointer to the captured state and unpacking the members in order to
  /// call the original lifted function.
  LogicalResult populateBodyOfWrapperFunction(
      LLVM::LLVMFuncOp wrapperFn, CreateClosureOp op,
      ConversionPatternRewriter &rewriter, CreateClosureOpAdaptor adaptor,
      SymbolTable &symbolTable, CreateClosureTypes const &types) const {
    Block &wrapperFnBody = wrapperFn.getBody().front();
    rewriter.setInsertionPointToStart(&wrapperFnBody);

    Type envCalleeType = adaptor.getCallee().getType();
    if (auto sigType = dyn_cast<FuncTypeGeneratorType>(envCalleeType))
      envCalleeType = typeConverter->convertType(sigType.getBody().getValues());

    SmallVector<Value> liftedNestedFunctionCallArgs(
        op.getCalleeType().getBody().getValues().getNumInputs());
    auto flatSymbol = dyn_cast<FlatSymbolRefAttr>(
        cast<SymbolConstantAttr>(op.getCallee()).getSymbol());
    if (!flatSymbol)
      return emitError(op.getLoc(),
                       "cannot lower call to nested symbol to LLVM");
    auto func =
        symbolTable.lookup<LLVM::LLVMFuncOp>(flatSymbol.getRootReference());
    if (!func)
      return emitError(op.getLoc(), "Callee does not reference llvm function");

    for (size_t i = 0; i < op.getCaptures().size(); ++i) {
      Type capturedArgType = types.boundArgTypes[i];
      LLVM::GEPOp boundArgPtr = LLVM::GEPOp::create(
          rewriter, wrapperFn.getLoc(), types.opaquePtrType,
          types.opaquePtrType, wrapperFnBody.getArgument(0),
          ArrayRef<LLVM::GEPArg>({0, static_cast<int32_t>(i)}));
      boundArgPtr.setElemType(types.liftedFunctionCaptureType);
      Value boundArg = LLVM::LoadOp::create(rewriter, wrapperFn.getLoc(),
                                            capturedArgType, boundArgPtr);
      liftedNestedFunctionCallArgs[i] = boundArg;
    }
    size_t numCaptures = op.getCaptures().size();
    size_t numberDynamicArgs =
        op.getCalleeType().getBody().getValues().getNumInputs() - numCaptures;
    for (size_t i = 0; i < numberDynamicArgs; ++i)
      liftedNestedFunctionCallArgs[i + numCaptures] =
          wrapperFnBody.getArgument(i + 1);

    ValueRange valueRange(liftedNestedFunctionCallArgs);
    LLVM::CallOp callLiftedFunction =
        createLLVMCall(rewriter, wrapperFn.getLoc(), func, valueRange);
    LLVM::ReturnOp::create(rewriter, wrapperFn.getLoc(),
                           callLiftedFunction.getResults());
    return success();
  }

  /// Replace the CreateClosureOp with the construction of a closure struct.
  LogicalResult generateClosureStruct(ConversionPatternRewriter &rewriter,
                                      CreateClosureOp op,
                                      CreateClosureOpAdaptor adaptor,
                                      LLVM::LLVMFuncOp wrapperFn,
                                      CreateClosureTypes const &types) const {
    MLIRContext *context = getContext();
    Value closureStruct = LLVM::UndefOp::create(
        rewriter, op.getLoc(), types.unpackingFunctionAndCapturesType);
    Value addressOfWrapperFunction =
        LLVM::AddressOfOp::create(rewriter, op.getLoc(), wrapperFn);
    closureStruct = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), closureStruct, addressOfWrapperFunction,
        static_cast<int64_t>(0));
    Value one = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                         IntegerType::get(context, 8), 1);
    LLVM::AllocaOp envStruct =
        LLVM::AllocaOp::create(rewriter, op.getLoc(), types.opaquePtrType, one);
    envStruct.setElemType(types.liftedFunctionCaptureType);
    // TODO: When data layouts are propagated properly, extract the data
    //  layout from TargetInfoAttr
    LLVM::LifetimeStartOp::create(rewriter, op.getLoc(), envStruct);
    for (auto [argIdx, boundArgValue] :
         llvm::enumerate(adaptor.getCaptures())) {
      LLVM::GEPOp getBoundArgPtr = LLVM::GEPOp::create(
          rewriter, op.getLoc(), /*resultType=*/types.opaquePtrType,
          /*basePtrType=*/types.opaquePtrType, /*basePtr=*/envStruct,
          ArrayRef<LLVM::GEPArg>({0, static_cast<int32_t>(argIdx)}));
      getBoundArgPtr.setElemType(types.liftedFunctionCaptureType);
      LLVM::StoreOp::create(rewriter, op.getLoc(), boundArgValue,
                            getBoundArgPtr.getResult());
    }

    // Add the environment struct to the closure struct
    closureStruct = LLVM::InsertValueOp::create(rewriter, op.getLoc(),
                                                closureStruct, envStruct, 1);

    // Insert lifetime marker at the end of the struct
    auto oldInsertionBlock = rewriter.getInsertionBlock();
    auto oldInsertionPoint = rewriter.getInsertionPoint();
    rewriter.setInsertionPoint(op->getBlock(), --op->getBlock()->end());
    LLVM::LifetimeEndOp::create(rewriter, op.getLoc(), envStruct);
    rewriter.setInsertionPoint(oldInsertionBlock, oldInsertionPoint);
    rewriter.replaceOp(op, closureStruct);

    return success();
  }

public:
  LogicalResult
  matchAndRewrite(CreateClosureOp op, CreateClosureOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().getBody().isCapturing() && op.getCaptures().empty()) {
      rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(
          op, convertType(op.getType()),
          cast<FlatSymbolRefAttr>(
              cast<SymbolConstantAttr>(op.getCallee()).getSymbol()));
      return success();
    }

    // Generate the function wrapper and populate it with extract and invoke the
    // original callee
    Block *oldInsertionBlock = rewriter.getInsertionBlock();
    Block::iterator oldInsertionPoint = rewriter.getInsertionPoint();
    rewriter.clearInsertionPoint();
    LLVM::LLVMFuncOp wrapperFn = this->generateWrapperFunction(op, rewriter);
    symtab.insert(wrapperFn);
    CreateClosureTypes types;
    if (failed(CreateClosureTypes::createClosureTypes(types, op,
                                                      *getTypeConverter())))
      return emitError(op.getLoc(),
                       "failed to convert kgen types to llvm closure types");

    if (failed(this->populateBodyOfWrapperFunction(wrapperFn, op, rewriter,
                                                   adaptor, symtab, types)))
      return failure();

    // Create the struct representing the closure back at the CreateClosure site
    rewriter.setInsertionPoint(oldInsertionBlock, oldInsertionPoint);
    if (failed(generateClosureStruct(rewriter, op, adaptor, wrapperFn, types)))
      return failure();

    return success();
  }
};

struct CallIndirectOpConversion
    : public ConvertPOPToLLVMPattern<CallIndirectOp> {
  using ConvertPOPToLLVMPattern<CallIndirectOp>::ConvertPOPToLLVMPattern;
  LogicalResult
  matchAndRewrite(CallIndirectOp op, CallIndirectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If there are no result types, set it to void. Otherwise, set the result
    // type to the packed result types.
    Type resultType;
    if (!op.getNumResults())
      resultType = getVoidType();
    else
      resultType = getTypeConverter()->packFunctionResults(op.getResultTypes());
    if (!resultType)
      return emitError(op.getLoc(), "failed to convert call result type");

    Value callee = op.getCallee();
    LLVM::CallOp llvmCall;
    // isCapturing and isCABI are mutually exclusive: a capturing closure
    // carries an environment struct, whereas a C ABI function pointer is a
    // bare pointer with no captured state.
    auto isClosureType = [](Type type) {
      if (auto sigType = dyn_cast<FuncTypeGeneratorType>(type))
        return sigType.getBody().isCapturing();
      return false;
    };
    auto isCABIFnPtr = [](Type type) {
      if (auto sigType = dyn_cast<FuncTypeGeneratorType>(type))
        return sigType.getBody().isCABI();
      return false;
    };

    if (isCABIFnPtr(callee.getType())) {
      // C ABI function-pointer call: apply platform C ABI coercion.
      CABICallHelper cabi(getTypeConverter(), getContext(), op.getOperation());
      Location loc = op.getLoc();

      // Original LLVM return type (null for void functions).
      Type origRetTy = op.getNumResults() > 0 ? resultType : Type{};

      auto prep =
          cabi.prepareCall(adaptor.getArguments().getTypes(),
                           adaptor.getArguments(), origRetTy, loc, rewriter);
      // Insert the indirect call target.
      prep.callArgs.insert(prep.callArgs.begin(), adaptor.getCallee());

      llvmCall = createLLVMCall(rewriter, loc, prep.signature, prep.callArgs);
      CABICallHelper::applySRetAttrIfNeeded(llvmCall, origRetTy, prep.usesSRet,
                                            rewriter);
      cabi.applyByvalAttrsToCall(llvmCall, prep.argClass,
                                 adaptor.getArguments().getTypes(),
                                 prep.usesSRet, rewriter);
      // A tail call is only safe when the callee and the caller's return ABIs
      // match. If the callee changed to using an sret "out pointer" due to the
      // ABI("C") annotation, that may no longer match the caller's return ABI,
      // leading to an ABI mismatch. If that's the case, avoid marking this
      // call as `tail` optimizable.
      if (!prep.usesSRet)
        applyTailKind(llvmCall, op.getTailKind());

      // Reverse-coerce the return value and replace the op.
      if (op.getNumResults() == 0) {
        rewriter.eraseOp(op);
        return success();
      }
      Value rawResult = prep.usesSRet ? Value{} : llvmCall.getResult();
      Value coerced = cabi.extractReturn(prep.retClass, rawResult, prep.sretPtr,
                                         origRetTy, loc, rewriter);
      rewriter.replaceOp(op, coerced);
      return success();
    }

    if (isClosureType(callee.getType())) {
      // Unpack the struct representation of the closure.
      auto pointerType = LLVM::LLVMPointerType::get(getContext());
      Value wrapperFnPtr = LLVM::ExtractValueOp::create(rewriter, op.getLoc(),
                                                        adaptor.getCallee(), 0);
      Value envStruct = LLVM::ExtractValueOp::create(rewriter, op.getLoc(),
                                                     adaptor.getCallee(), 1);

      // Compute the type of the wrapper function -- wrapper function type is
      // (!llvm.ptr, unboundArgTy0, ... unboundArgTyn) -> resultTypes
      SmallVector<Type> wrapperFnArgTypes;
      wrapperFnArgTypes.push_back(pointerType);

      auto calleeFuncTy =
          cast<FuncTypeGeneratorType>(callee.getType()).getBody().getValues();
      for (Type argTy : calleeFuncTy.getInputs()) {
        Type ty = convertType(argTy);
        if (!ty)
          return emitError(op.getLoc())
                 << "could not convert argument type " << argTy;
        wrapperFnArgTypes.push_back(ty);
      }

      auto wrapperFnType = LLVM::LLVMFunctionType::get(getContext(), resultType,
                                                       wrapperFnArgTypes, 0);

      // Create the call to the wrapper function.
      SmallVector<Value> llvmCallArgs;
      llvmCallArgs.push_back(wrapperFnPtr);
      llvmCallArgs.push_back(envStruct);
      llvm::append_range(llvmCallArgs, adaptor.getArguments());

      llvmCall =
          createLLVMCall(rewriter, op.getLoc(), wrapperFnType, llvmCallArgs);
    } else {
      // Mojo ABI indirect call (default path).
      // Note: adaptor.getOperands() is a list of callee followed by inputs.
      SmallVector<Type> wrapperFnArgTypes;
      llvm::append_range(wrapperFnArgTypes, adaptor.getArguments().getTypes());
      auto wrapperFnType = LLVM::LLVMFunctionType::get(getContext(), resultType,
                                                       wrapperFnArgTypes, 0);
      llvmCall = createLLVMCall(rewriter, op.getLoc(), wrapperFnType,
                                adaptor.getOperands());
    }
    applyTailKind(llvmCall, op.getTailKind());

    if (op.getNumResults() <= 1) {
      rewriter.replaceOp(op, llvmCall.getResults());
      return success();
    }

    // Unpack the struct if necessary.
    SmallVector<Value> results;
    results.reserve(op.getNumResults());
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i)
      results.push_back(LLVM::ExtractValueOp::create(rewriter, op.getLoc(),
                                                     llvmCall.getResult(), i));

    // Replace the call operation.
    rewriter.replaceOp(op, results);
    return success();
  }
};

void LowerRuntimeClosuresPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(targetInfo);

  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<KGENDialect>();
  target.addLegalDialect<POP::POPDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  target.addIllegalOp<CreateClosureOp>();
  target.addIllegalOp<CallIndirectOp>();
  patterns.insert<CreateRuntimeClosureOpConversion>(symtab, typeConverter);
  patterns.insert<CallIndirectOpConversion>(typeConverter);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
