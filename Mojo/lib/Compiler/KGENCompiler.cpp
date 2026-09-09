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

#include "Mojo/Compiler/KGENCompiler.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/Compiler/SaveAsmOutput.h"
#include "Mojo/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/Support/BuildInfo.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/Support/Constants.h"
#include "Mojo/Support/NameMangling.h"
#include "Mojo/ToolCommon/Debug.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/ToolCommon/PipelineTiming.h"
#include "Mojo/TransformUtils/SlicingUtils.h"
#include "ObjectCompiler/KGENToLLVMPipeline.h"
#include "Pipeline/Pipeline.h"
#include "Support/ADT/DenseStringMap.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/Config.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Target/TargetTraits.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include "xxh3.h"
#include "xxhash.h"

#define DEBUG_TYPE "kgen-compiler"
#define KGEN_DEBUG_TYPE "kgen-compiler"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// compileElaboratorAsm
//===----------------------------------------------------------------------===//

/// Generate a stub function that calls into the sliced function with input
/// parameters, then rename it to match the expected symbol name and export it
/// This is how compilation is rooted at instantiations of parametric functions.
static void
generateInstantiateStub(GeneratorOp func, SymbolConstantAttr symbol,
                        StringAttr name, IRMapping &mapping,
                        SymbolTable *symtab = nullptr,
                        std::optional<uint64_t> kernelId = std::nullopt) {

  GeneratorOp sliced = cast<GeneratorOp>(mapping.lookup(func));
  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(sliced));
  StringAttr stubName = b.getStringAttr(name.getValue() + "_asm_stub");
  FuncTypeGeneratorType sigGen = symbol.getType();
  FuncType sigBase = sigGen.getBody();

  // Build debuginfo for the stub if requested.
  if (auto sp = func.getSubprogramScope()) {
    // The original DISubroutineType for the subprogram may contain parameter
    // references that are no longer in scope in the stub. Re-create a
    // DISubroutineType from the concretized signature of the stub (this is ok
    // since the stub is a compiler-synthesized function).
    auto stubSourceName =
        DebugInfo::SourceNameAttr::get("asm_stub", sp.getSourceName());
    FunctionType stubFuncType = sigBase.getValues();
    DebugInfo::DIUnresolvedMLIRType (*mapToDIUnresolvedType)(Type) =
        &DebugInfo::DIUnresolvedMLIRType::get;
    auto stubSp = DebugInfo::DISubprogramAttr::get(
        sp.getCompileUnit(), sp.getScope(), stubSourceName, stubName,
        sp.getFile(), sp.getLine(), sp.getScopeLine(), sp.getSubprogramFlags(),
        DebugInfo::DISubroutineType::get(
            func.getContext(),
            SmallVector<DebugInfo::DIType>(
                map_range(stubFuncType.getInputs(), mapToDIUnresolvedType)),
            SmallVector<DebugInfo::DIType>(
                map_range(stubFuncType.getResults(), mapToDIUnresolvedType))));
    DebugInfo::DIAttrTypeReplacer replacer;
    replacer.addReplacement(
        [stubSp](DebugInfo::DISubprogramAttr) { return stubSp; });
    b.setLoc(cast<LocationAttr>(replacer.replace(b.getLoc())));
  }

  auto linkageNameAttr = sliced.getLinkageNameAttr();

  sliced.setNotExported();
  sliced.setInlineLevel(InlineLevel::Always);
  if (symtab) {
    // Clone first so the original retains its linkage name for any subsequent
    // stubs generated from the same generator (e.g. two instantiations of the
    // same kernel with different type parameters).
    sliced = sliced.clone();
    symtab->insert(sliced);
  }
  sliced.setSymNameAttr(stubName);
  if (linkageNameAttr)
    sliced.removeLinkageNameAttr();

  auto wrapper = GeneratorOp::create(b, name, sigGen);
  wrapper.setExported();

  // Transfer the sliced function's linkage name onto the wrapper.
  if (linkageNameAttr)
    wrapper.setLinkageNameAttr(linkageNameAttr);

  SmallVector<Attribute> metadataArray =
      llvm::to_vector(sliced.getLLVMMetadataArrayAttr().getValue());
  if (kernelId) {
    metadataArray.push_back(
        StringAttr::get(sliced->getContext(), "kgen.offload.kernelid"));
    metadataArray.push_back(b.getIndexAttr(*kernelId));
  }
  wrapper.setLLVMMetadataArrayAttr(
      ArrayAttr::get(sliced.getContext(), metadataArray));
  wrapper.setLLVMArgMetadataArrayAttr(sliced.getLLVMArgMetadataArrayAttr());
  Block *entry =
      b.createBlock(&wrapper.getBodyRegion(), {}, sigBase.getArguments(),
                    llvm::map_to_vector(sliced.getArguments(),
                                        [](Value v) { return v.getLoc(); }));

  // Re-declare the captured parameter values.
  for (auto [decl, value] :
       llvm::zip(sliced.getInputParams(), symbol.getParamValues()))
    ParamDeclareOp::create(b, decl, value);

  auto call = CallOp::create(
      b, SymbolConstantAttr::get(stubName, sigGen, symbol.getParamValues()),
      entry->getArguments());
  ReturnOp::create(b, call.getResults());
}

/// Find a FuncOp in the module by its kgen.offload.kernelid LLVM metadata.
/// Returns nullptr if no function with the given kernel ID is found.
static FuncOp findFuncByKernelId(ModuleOp module, uint64_t kernelId) {
  for (auto func : module.getOps<FuncOp>()) {
    if (auto meta = func.getLLVMMetadataAttr()) {
      if (auto idAttr = meta.get("kgen.offload.kernelid")) {
        if (cast<IntegerAttr>(idAttr).getInt() ==
            static_cast<int64_t>(kernelId))
          return func;
      }
    }
  }
  return {};
}

/// HACK HACK HACK https://github.com/modularml/modular/issues/22959
/// HACK: Read out the magic attribute used to propagate captures across device
/// boundaries, generate the capture function, and write them into the buffer.
static std::tuple<OwningOpRef<FuncOp>, unsigned, mlir::DenseI64ArrayAttr>
writeCaptureArgs(ModuleOp module, FuncOp sliced, StringAttr preRenameSym) {
  // This is held together with duct tape, so check the invariant.
  assert(sliced && sliced.isExported() && "expected a sliced function");
  // For offload kernels renamed by @__name, the host stub was created using the
  // pre-rename auto-mangled sym (in evaluateCompileOffloadClosureAttr). Pass
  // preRenameSym to use that same name so the fill step in the host elaborator
  // can match the populate function to the host stub by name.
  StringAttr name = preRenameSym ? preRenameSym : sliced.getSymNameAttr();
  ArrayRef<StringAttr> captures = sliced.getCrossDeviceCaptures();

  // The location to use for generated code. Remove all debuginfo from it.
  Location loc = sliced.getLoc();
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](mlir::FusedLocWith<DebugInfo::DIAttr> loc) {
    return FusedLoc::get(loc.getContext(), loc.getLocations());
  });
  loc = cast<LocationAttr>(replacer.replace(loc));

  // Generate a function on the host side that opaquely populates a piece of
  // memory with the capture values.
  ImplicitLocOpBuilder b(loc, OpBuilder(name.getContext()));

  // The expected signature is `fn(Pointer[None]) capturing -> None`.
  auto noneType = b.getType<KGEN::NoneType>();
  auto nonePtr = PointerType::get(noneType);
  auto sig = FuncType::get(b.getFunctionType(nonePtr, noneType),
                           ArgConvention::ImmReg, FnEffects().setCapturing());
  OwningOpRef<FuncOp> func =
      FuncOp::create(b, b.getStringAttr(name.getValue() + "_populate_captures"),
                     sig, InlineLevel::Always);

  // Populate the body. Generate a local variable for each capture argument
  // and store the addresses to the pointer.
  // The function has to be `always_inline`, so that the stack allocated ptr
  // will not go out of scope before use.
  // FIXME: This does not account for copy constructors, obviously.
  Block *body = b.createBlock(&func->getBodyRegion());
  Value argPtrs = body->addArgument(sig.getArguments().front(), b.getLoc());
  TargetInfoAttr target = lookupTargetInfo(module);
  SmallVector<int64_t> typeSizes;
  for (auto [i, capture] : llvm::enumerate(captures)) {
    // ```
    // %value = pop.compiler.global_load "var" : T
    // %ptr = pop.stack_allocation 1 x T
    // pop.store %value, %ptr
    // %gep = pop.offset %argPtrs[%i]
    // %opaque = pop.pointer.bitcast %ptr : pointer<T> to pointer<none>
    // pop.store %opaque, %gep
    // ```
    Type type = capture.getType();
    Value value = POP::CompilerGlobalLoadOp::create(
        b,
        // Make sure to strip off the type of the StringAttr.
        type, b.getStringAttr(capture.getValue()));
    Value ptr = POP::StackAllocationOp::create(b, PointerType::get(type));
    POP::StoreOp::create(b, value, ptr);
    Value argPtrPtrs =
        POP::PointerBitcastOp::create(b, PointerType::get(nonePtr), argPtrs);
    Value gep = POP::OffsetOp::create(
        b, argPtrPtrs, ParamConstantOp::create(b, b.getIndexAttr(i)));
    Value opaque = POP::PointerBitcastOp::create(b, nonePtr, ptr);
    std::optional<int64_t> typeStoreSize =
        DataLayoutInterface::getTypeStoreSize(target, type);
    if (!typeStoreSize)
      llvm_unreachable("unable to get the size of the type");
    typeSizes.push_back(*typeStoreSize);
    POP::StoreOp::create(b, opaque, gep);
  }
  ReturnOp::create(
      b, ParamConstantOp::create(b, b.getAttr<NoneAttr>()).getResult());

  return {std::move(func), captures.size(), b.getDenseI64ArrayAttr(typeSizes)};
}

static StringAttr getXXH3Hash(StringAttr strAttr) {
  XXH128_hash_t hash = XXH3_128bits(strAttr.data(), strAttr.size());
  StringRef hashStr(llvm::bit_cast<char *>(&hash), sizeof(XXH128_hash_t));
  return StringAttr::get(llvm::toHex(hashStr, true),
                         StringType::get(strAttr.getContext()));
}

static ElaboratorCompileOffloadRetType compileOffloads(
    ModuleOp theModule,
    llvm::MapVector<TargetInfoAttr, OffloadInfo> &targetOffloadInfos,
    const SymbolTable &symtab, CompilationOptions compilationOptions,
    ElaborateGeneratorsOptions elabOptions);

/// Given the pre-elaboration function `func` belonging to a module with the
/// symbol table `symtab`, slice out a standalone module rooted at `func` and
/// elaborate it and compile to assembly for the provided `target.
static ErrorOr<CrossDeviceFunction> compileElaboratorAsm(
    GeneratorOp func, SymbolConstantAttr symbol, StringAttr name,
    const SymbolTable &symtab, TargetInfoAttr target, EmitAs emissionKind,
    EmissionOptions emissionOptions, CompilationOptions compilationOptions,
    ElaborateGeneratorsOptions elaboratorOptions) {
  // Configure the compilation options given the new target.
  compilationOptions.targetTriple = target.getTripleStr();
  compilationOptions.targetCpu = target.getArch();
  compilationOptions.targetFeatures = target.getFeatures();
  compilationOptions.targetAbi = target.getAbi();
  if (compilationOptions.targetAccelerator.empty()) {
#if MLRT_ACCELERATOR_SUPPORT
    compilationOptions.targetAccelerator =
        Driver::Device::getAcceleratorArchOrEmpty();
#endif
  }
  compilationOptions.relocModel = target.getRelocationModel();
  StringRef targetDataLayout = target.getDataLayout().toString();
  if (!targetDataLayout.empty())
    compilationOptions.targetDataLayout = targetDataLayout;

  // Pull `target-abi` out of the emission options and apply it directly to
  // `compilationOptions` before the target machine is created: `createTarget-
  // Machine` reads `targetABI` as a struct field, not by re-querying LLVM's
  // registered `-target-abi` cl option, so letting it flow through the
  // generic cl-parsed list below would only mutate global state unseen here.
  compilationOptions.targetABI.clear();
  applyTargetABIEmissionOptions(emissionOptions, compilationOptions.targetABI);
  SmallVector<StringRef> clEmissionOptions;
  llvm::copy_if(
      emissionOptions, std::back_inserter(clEmissionOptions),
      [](StringRef item) { return !isTargetABIEmissionOption(item); });

  // Initialize the object compiler.
  PassManagerConfigOptions pmOptions;
  pmOptions.applyPassManagerCLOptions = true; // enable print options
  ErrorOr<std::unique_ptr<ObjectCompiler>> compilerOr = ObjectCompiler::create(
      kMojoCacheBaseDirName, compilationOptions, /*isJIT=*/
      false, *target.getContext(), pmOptions);

  if (compilerOr.isError())
    return compilerOr.takeError();

  std::unique_ptr<ObjectCompiler> compiler = compilerOr.takeValue();

  // Initialize the target machine.
  auto tmOr = createTargetMachine(compilationOptions, /*isJIT=*/false);
  if (tmOr.isError())
    return tmOr.takeError();
  std::unique_ptr<llvm::TargetMachine> tm = tmOr.takeValue();

  // Slice out a pre-elaboration module for the new target to compile for.
  ExportMap exportedSymbols;
  exportedSymbols.insert({func.getSymNameAttr(), ExportKind::Exported});
  // Make sure to slice out anything referenced in the input parameters. When
  // generator references are instantiated in the standalone module, they are
  // instantiated with the new target.
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](SymbolConstantAttr ref) {
    exportedSymbols.insert(
        {ref.getSymbol().getRootReference(), ExportKind::NotExported});
  });
  walker.addWalk([&](FuncSymbolAttr ref) {
    exportedSymbols.insert(
        {ref.getSymbol().getRootReference(), ExportKind::NotExported});
  });
  for (TypedAttr attr : symbol.getParamValues())
    walker.walk(attr);

  IRMapping mapping;
  OwningOpRef<ModuleOp> module = produceStandaloneModule(
      symtab, exportedSymbols, mapping, overrideExported(compilationOptions));
  // Override the target.
  eraseTargetInfo(*module);
  setTargetInfo(*module, target);

  // Tag the entry generator with a kernel ID so we can find the resulting
  // FuncOp after elaboration (which may rename symbols for offload targets).
  static constexpr uint64_t kAsmEntryKernelId = 0;
  if (!symbol.getParamValues().empty()) {
    generateInstantiateStub(func, symbol, name, mapping, /*symtab=*/nullptr,
                            kAsmEntryKernelId);
  } else {
    GeneratorOp sliced = cast<GeneratorOp>(mapping.lookup(func));
    ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(sliced));
    SmallVector<Attribute> metadataArray =
        llvm::to_vector(sliced.getLLVMMetadataArrayAttr().getValue());
    metadataArray.push_back(
        StringAttr::get(sliced->getContext(), "kgen.offload.kernelid"));
    metadataArray.push_back(b.getIndexAttr(kAsmEntryKernelId));
    sliced.setLLVMMetadataArrayAttr(
        ArrayAttr::get(sliced.getContext(), metadataArray));
  }

  // Run elaboration through to the end of the optimization pipeline.
  mlir::PassManager pm(target.getContext());
  // Ignore potential error that could happen when `mojo` tool is called, which
  // does not register pass manager options.
  (void)pmOptions.configurePassManager(pm);

  pm.addPass(createElaborateGenerators(target, elaboratorOptions,
                                       compilationOptions, compileElaboratorAsm,
                                       compileOffloads));
  buildPostElaborationPipeline(pm, compilationOptions);

  if (failed(pm.run(*module)))
    return Error("failed to run the pass manager");
  // Find the entry function by kernel ID (robust against offload name
  // sanitization).
  FuncOp entryFunc = findFuncByKernelId(*module, kAsmEntryKernelId);
  if (!entryFunc)
    return Error("internal error: cannot find kernel by its ID");
  auto [capturesFunc, numCaptures, captureSizes] =
      writeCaptureArgs(*module, entryFunc, name);

  // Handle the emission options.
  ErrorOrSuccess parseResult = parseEmissionOptions(clEmissionOptions);
  if (parseResult.isError()) {
    return parseResult.takeError();
  }

  // Prepare a buffer to write string output to.
  SmallVector<char> buf;
  buf.reserve(256 * 128); // 32 KB
  llvm::raw_svector_ostream os(buf);

  // Emit the module in the requested form.
  switch (emissionKind) {
  case EmitAs::ASM:
    if (ErrorOrSuccess err = compiler->emitAssembly(std::move(module), os))
      return err.takeError();
    break;

  case EmitAs::LLVM:
  case EmitAs::LLVM_BITCODE: {
    LLVMModuleAndContext llvmModule;
    if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
          return compiler->lowerAllFuncsToLLVM(ctx, *module);
        }))
      return err.takeError();
    if (emissionKind == EmitAs::LLVM_BITCODE) {
      if (ErrorOrSuccess err = compiler->emitBitcode(*llvmModule, os))
        return err.takeError();
    } else {
      llvmModule->print(os, nullptr);
    }
    break;
  }

  case EmitAs::LLVM_OPT:
  case EmitAs::LLVM_OPT_BITCODE: {
    LLVMModuleAndContext llvmModule;
    if (ErrorOrSuccess err =
            compiler->lowerAllFuncsToLLVMAndOptimize(*module, llvmModule))
      return err.takeError();
    if (emissionKind == EmitAs::LLVM_OPT_BITCODE) {
      if (ErrorOrSuccess err = compiler->emitBitcode(*llvmModule, os))
        return err.takeError();
    } else {
      llvmModule->print(os, nullptr);
    }
  } break;

  case EmitAs::OBJECT:
    if (ErrorOrSuccess err = compiler->emitSharedObject(std::move(module), os))
      return err.takeError();
    break;
  }

  return CrossDeviceFunction{
      StringAttr::get(buf, StringType::get(func.getContext())), numCaptures,
      std::move(capturesFunc)};
}

//===----------------------------------------------------------------------===//
// compileOffloads
//===----------------------------------------------------------------------===//

static ElaboratorCompileOffloadRetType compileOffloads(
    ModuleOp theModule,
    llvm::MapVector<TargetInfoAttr, OffloadInfo> &targetOffloadInfos,
    const SymbolTable &symtab, CompilationOptions compilationOptions,
    ElaborateGeneratorsOptions elabOptions) {

  DenseMap<
      TargetInfoAttr,
      llvm::DenseMap<std::string, DenseMap<uint64_t, OffloadCompilationResult>>>
      result;

  // Extract the initial bitcode library data from the original module.
  SmallVector<std::pair<bool, Attribute>> currentBitcodeLibs;
  if (auto bitcodeLibArrayAttr =
          theModule->getAttrOfType<LLVMBitcodeLibArrayAttr>(
              LLVMBitcodeLibArrayAttr::getBitcodeLibsAttrName()))
    bitcodeLibArrayAttr.externalize(currentBitcodeLibs);

  // Tracks how many times each sanitized kernel name has been used per
  // extension, to avoid collisions across targets. Keyed by "<name><ext>".
  llvm::StringMap<int> kernelNameCounts;

  // Pending offload writes: {fileName, content} pairs collected during
  // compilation and flushed to disk by flushOffloadWrites() after
  // cachedTransform.
  SmallVector<NamedAttribute> offloadPendingWrites;

  // The host `-fp-mode`, re-applied per group before any `contract=fast|off`
  // item from that offload's `emission_option` overrides it.
  const FpMode hostFpMode = compilationOptions.fpMode;

  // Compiling offload for different targets.
  // This loop cannot be parallelized since different targets may need
  // different llvm options that are global states.
  for (auto [target, info] : targetOffloadInfos) {
    llvm::DenseMap<std::string, DenseMap<uint64_t, OffloadCompilationResult>>
        &targetEmissionResult = result[target];

    for (auto [groupKey, offloadInfo] : info.groups) {

      DenseMap<uint64_t, OffloadCompilationResult> &targetResult =
          targetEmissionResult[groupKey];

      IRMapping mapping;

      // Configure the compilation options given the new target.
      compilationOptions.targetTriple = target.getTripleStr();
      compilationOptions.targetCpu = target.getArch();
      compilationOptions.targetFeatures = target.getFeatures();
      compilationOptions.targetAbi = target.getAbi();
      if (compilationOptions.targetAccelerator.empty()) {
#if MLRT_ACCELERATOR_SUPPORT
        compilationOptions.targetAccelerator =
            Driver::Device::getAcceleratorArchOrEmpty();
#endif
      }
      compilationOptions.relocModel = target.getRelocationModel();
      StringRef targetDataLayout = target.getDataLayout().toString();
      if (!targetDataLayout.empty())
        compilationOptions.targetDataLayout = targetDataLayout;
      // Pull any `contract=fast|off` fp-mode item out of this offload's
      // emission options (they are not global llvm cl options) and apply it on
      // top of the host `-fp-mode` for this kernel only.
      compilationOptions.fpMode = hostFpMode;
      std::string filteredEmissionOptions;
      if (std::optional<std::string> badItem = splitFpModeEmissionOptions(
              offloadInfo.emissionOptions, compilationOptions.fpMode,
              filteredEmissionOptions))
        return Error(llvm::formatv("invalid fp-mode emission option '{0}', "
                                   "expected 'contract=fast' or 'contract=off'",
                                   *badItem));
      offloadInfo.emissionOptions = filteredEmissionOptions;

      // Likewise pull `target-abi` into the compilation options. It stays in
      // the emission-option string, which identifies the offload in debug
      // output and cache keys, and is skipped when the remaining options are
      // applied as cl options.
      compilationOptions.targetABI.clear();
      {
        SmallVector<StringRef> items;
        StringRef(offloadInfo.emissionOptions)
            .split(items, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        applyTargetABIEmissionOptions(items, compilationOptions.targetABI);
      }

      compilationOptions.emissionOptions = offloadInfo.emissionOptions;
      compilationOptions.emissionLinkOptions = offloadInfo.emissionLinkOptions;

      OwningOpRef<ModuleOp> module =
          produceStandaloneModule(symtab, offloadInfo.exportedSymbols, mapping,
                                  overrideExported(target.getTriple()));

      // Override the target.
      eraseTargetInfo(*module);
      setTargetInfo(*module, target);
      SymbolTable slicedSymtab(*module);

      // Collect SymbolConstantAttr names to rename.
      DenseMap<SymbolRefAttr, StringAttr> symToRename;

      for (auto [op, symbolInfo] : offloadInfo.symbols) {
        // If there are input parameters, we have to go generate a stub to root
        // instantiation of the generator. Go find the cloned generator.
        auto func = cast<GeneratorOp>(op);
        StringAttr newCalleeName = StringAttr::get(
            func->getContext(),
            FlatSymbolRefAttr::get(func).getAttr().str() + "_callee");

        std::optional<StringAttr> newName;
        for (auto [symbol, kernelInfo] : symbolInfo) {
          if (!symbol.getParamValues().empty()) {
            // Add "_callee" postfix to the generator that is both a callee of a
            // kernel and a kernel entry function itself. So that we don't end
            // up with two functions with the same name one for the kernel entry
            // wrapper for instantiated stub, and one for the callee in another
            // kernel (they have different function bodies).
            module->walk(
                [&newCalleeName, &newName, &func, &symToRename](CallOp call) {
                  if (call.getCalleeSymbol() == FlatSymbolRefAttr::get(func)) {
                    newName = newCalleeName;
                    symToRename.insert(
                        {call.getCallee().getSymbol(), newCalleeName});
                  }
                });

            generateInstantiateStub(func, symbol, kernelInfo.name, mapping,
                                    &slicedSymtab, kernelInfo.kernelId);
          } else {
            // Set kernelId
            GeneratorOp sliced = cast<GeneratorOp>(mapping.lookup(func));
            ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(sliced));
            SmallVector<Attribute> metadataArray =
                llvm::to_vector(sliced.getLLVMMetadataArrayAttr().getValue());
            metadataArray.push_back(
                StringAttr::get(sliced->getContext(), "kgen.offload.kernelid"));
            metadataArray.push_back(b.getIndexAttr(kernelInfo.kernelId));
            sliced.setLLVMMetadataArrayAttr(
                ArrayAttr::get(sliced.getContext(), metadataArray));
          }
        }
        if (newName) {
          // Rename the generator since it is used both as kernel entry function
          // and callee for another kernel.
          GeneratorOp sliced = cast<GeneratorOp>(mapping.lookup(func));
          sliced.setSymNameAttr(*newName);
        }
      }

      // Replace the SymbolConstantAttr names for the renamed generator
      // references.
      if (!symToRename.empty()) {
        mlir::AttrTypeReplacer replacer;
        auto replaceSymbol = [&symToRename](auto attr) {
          auto iter = symToRename.find(attr.getSymbol());
          if (iter != symToRename.end()) {
            return decltype(attr)::get(iter->second, attr.getType(),
                                       attr.getParamValues());
          }
          return attr;
        };
        replacer.addReplacement(
            [&](SymbolConstantAttr attr) { return replaceSymbol(attr); });
        replacer.addReplacement(
            [&](FuncSymbolAttr attr) { return replaceSymbol(attr); });

        replacer.recursivelyReplaceElementsIn(*module, /*replaceAttrs=*/true,
                                              /*replaceLocs=*/true,
                                              /*replaceTypes=*/true);
      }

      // Initialize the object compiler.
      PassManagerConfigOptions pmOptions;
      pmOptions.applyPassManagerCLOptions = true; // enable print options
      ErrorOr<std::unique_ptr<ObjectCompiler>> compilerOr =
          ObjectCompiler::create(kMojoCacheBaseDirName,
                                 compilationOptions, /*isJIT=*/
                                 false, *target.getContext(), pmOptions);

      if (compilerOr.isError())
        return compilerOr.takeError();

      std::unique_ptr<ObjectCompiler> compiler = compilerOr.takeValue();

      // Set the current bitcode libraries on the ObjectCompiler.
      if (!currentBitcodeLibs.empty())
        compiler->getBitcodeLibs() = currentBitcodeLibs;

      // Initialize the target machine.
      auto tmOr = createTargetMachine(compilationOptions, /*isJIT=*/false);
      if (tmOr.isError())
        return tmOr.takeError();
      std::unique_ptr<llvm::TargetMachine> tm = tmOr.takeValue();

      // Run elaboration through to the end of the optimization pipeline.
      // The scope comes before the pass manager in this declaration. Thus
      // the program destroys the pass manager first. Then the pass manager
      // reports into a scope that is still active.
      mlir::TimingScope offloadTiming =
          nestMLIROffloadScope(compilationOptions);

      mlir::PassManager pm(target.getContext());
      // Ignore potential error that could happen when `mojo` tool is called,
      // which does not register pass manager options.
      (void)pmOptions.configurePassManager(pm);
      // `configurePassManager` stops at that error. It stops before it reads
      // the timing scope. Thus this code starts the timing. A command
      // without the option gives an empty scope. An empty scope adds no times
      // to the pass manager.
      pm.enableTiming(offloadTiming);

      pm.addPass(
          createElaborateGenerators(target, elabOptions, compilationOptions,
                                    compileElaboratorAsm, compileOffloads));

      buildPostElaborationPipeline(pm, compilationOptions);

      if (failed(pm.run(*module)))
        return Error("failed to run the pass manager for offload functions");

      struct CaptureEntry {
        OwningOpRef<FuncOp> func;
        unsigned numCaptures;
        mlir::DenseI64ArrayAttr captureSizes;
        mlir::StringAttr nameForFile;
      };
      llvm::MapVector<uint64_t, CaptureEntry> captures;

      llvm::DenseMap<uint64_t, llvm::SmallSet<EmitAs, 4>> kernelEmissionKinds;

      for (auto [op, symbols] : offloadInfo.symbols) {
        for (auto [symbol, kernel] : symbols) {
          FuncOp kernelFunc = findFuncByKernelId(*module, kernel.kernelId);
          if (!kernelFunc)
            return Error("internal error: cannot find kernel by its ID");
          auto [capturesFunc, numCaptures, captureSizes] =
              writeCaptureArgs(*module, kernelFunc, kernel.name);

          // renameFunctions has already run, so sym_name is the exact name
          // that will be emitted. Use it directly as the output file basename
          // so the file name always matches the symbol inside.
          mlir::StringAttr nameForFile = kernelFunc.getSymNameAttr();
          assert(nameForFile && !nameForFile.getValue().empty() &&
                 "kernel sym_name must be non-empty after renameFunctions");
          captures.insert({kernel.kernelId,
                           CaptureEntry{std::move(capturesFunc), numCaptures,
                                        captureSizes, nameForFile}});
          kernelEmissionKinds.insert({kernel.kernelId, kernel.emissionKinds});
        }
      }

      // If saving offload kernel output files, request the appropriate emission
      // kind (ASM or LLVM IR) for all kernels.
      if (!compilationOptions.offloadOutputPrefix.empty()) {
        ErrorOr<const TargetTraits *> traitsOr =
            TargetTraitsRegistry::get().lookup(target.getTriple());
        const TargetTraits *traits = traitsOr.isError() ? nullptr : *traitsOr;
        // Targets that don't emit a standalone offload object skip OBJECT; the
        // emitted file is aliased under the OBJECT key below so
        // rewriteCompileOffloadOp still finds a value to embed.
        bool aliasFileAsObject = traits && !traits->emitsOffloadObjectFile();
        for (auto &[id, kinds] : kernelEmissionKinds) {
          kinds.insert(compilationOptions.offloadOutputKind);
          if (aliasFileAsObject)
            kinds.erase(EmitAs::OBJECT);
        }
      }

      // Handle the emission options.
      // These are the global llvm options needed to compile this target.
      // These options can't be kernel specific since they are global llvm
      // states, they are shared for parallel kernel compilation. However,
      // offloads for different targets won't have to share since we compile
      // them in order and we can reset these options for each targets.
      SmallVector<StringRef> emissionOptions;
      StringRef(offloadInfo.emissionOptions)
          .split(emissionOptions, /*Separator=*/",",
                 /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      // `target-abi` was already applied to `compilationOptions.targetABI`
      // above; keep it out of the parse/reset lists so it doesn't also mutate
      // LLVM's shared, registered `-target-abi` cl option as a side effect.
      llvm::erase_if(emissionOptions, isTargetABIEmissionOption);

      KGEN_DEBUG(0, {
        llvm::dbgs() << "Emit offloads with options: "
                     << offloadInfo.emissionOptions << "\n";
      });
      ErrorOrSuccess parseResult = parseEmissionOptions(emissionOptions);
      if (parseResult.isError()) {
        return parseResult.takeError();
      }

      ErrorOr<DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>>>
          compiledKernelsOr = compiler->emitOffloadKernels(std::move(module),
                                                           kernelEmissionKinds);

      if (compiledKernelsOr.isError())
        return compiledKernelsOr.takeError();

      // Extract the updated bitcode libraries from ObjectCompiler for the
      // next iteration.
      if (!compiler->getBitcodeLibs().empty())
        currentBitcodeLibs = compiler->getBitcodeLibs();

      OpBuilder b(theModule);
      for (auto idAndKernels : *compiledKernelsOr) {
        uint64_t kernelID = idAndKernels.first;
        DenseMap<EmitAs, BufferRef> &bufs = idAndKernels.second;

        auto iter = captures.find(kernelID);
        if (iter == captures.end())
          return Error("Can't find offload capture.");

        OwningOpRef<FuncOp> func = std::move(iter->second.func);
        unsigned numCaptures = iter->second.numCaptures;
        mlir::DenseI64ArrayAttr captureSizes = iter->second.captureSizes;

        auto populate = cast<FuncOp>(func.get());
        auto populateFnRef = SymbolConstantAttr::get(populate);
        DenseMap<EmitAs, StringAttr> contents;
        DenseMap<EmitAs, StringAttr> moduleNames;

        for (auto kindAndContent : bufs) {
          EmitAs kind = kindAndContent.first;
          // Defer this offload output write: encode it on the module so it
          // survives cachedTransform serialization and flushOffloadWrites()
          // can flush it to disk after cachedTransform returns, on both
          // cache-hit and cache-miss paths.
          if (!compilationOptions.offloadOutputPrefix.empty() &&
              kind == compilationOptions.offloadOutputKind) {
            mlir::StringAttr rawName = iter->second.nameForFile;
            llvm::Triple triple(target.getTripleStr());
            ErrorOr<const TargetTraits *> traitsOr =
                TargetTraitsRegistry::get().lookup(triple);
            if (traitsOr.isError())
              return Error(traitsOr.getError());
            const TargetTraits *traits = *traitsOr;
            llvm::StringRef ext =
                compilationOptions.offloadOutputKind == EmitAs::LLVM
                    ? traits->getLLVMExtension()
                    : traits->getAsmExtension();
            constexpr size_t kFileNameMaxChars = 64;
            std::string fileName =
                reserveOffloadOutputBaseName(
                    sanitizeSymbolToUnderscores(rawName, kFileNameMaxChars),
                    ext, kernelNameCounts) +
                ext.str();
            offloadPendingWrites.push_back(
                {mlir::StringAttr::get(theModule->getContext(), fileName),
                 mlir::StringAttr::get(theModule->getContext(),
                                       kindAndContent.second->getBuffer())});
          }
          StringAttr content =
              StringAttr::get(kindAndContent.second->getBuffer(),
                              StringType::get(theModule->getContext()));
          contents.insert({kind, content});
          moduleNames.insert({kind, getXXH3Hash(content)});
        }

        // Targets that don't emit a standalone offload object didn't compile
        // OBJECT; alias the emitted file (ASM or LLVM IR) under the OBJECT key
        // so rewriteCompileOffloadOp, which always looks up contents[OBJECT],
        // still finds a value to embed in the host module.
        ErrorOr<const TargetTraits *> traitsOr =
            TargetTraitsRegistry::get().lookup(target.getTriple());
        const TargetTraits *traits = traitsOr.isError() ? nullptr : *traitsOr;
        if (!compilationOptions.offloadOutputPrefix.empty() && traits &&
            !traits->emitsOffloadObjectFile()) {
          EmitAs fileKind = compilationOptions.offloadOutputKind;
          auto fileIt = contents.find(fileKind);
          if (fileIt != contents.end() && !contents.count(EmitAs::OBJECT)) {
            contents.insert({EmitAs::OBJECT, fileIt->second});
            moduleNames.insert({EmitAs::OBJECT, moduleNames[fileKind]});
          }
        }

        targetResult.insert(
            {kernelID, OffloadCompilationResult{{std::move(func)},
                                                b.getIndexAttr(numCaptures),
                                                captureSizes,
                                                populateFnRef,
                                                std::move(contents),
                                                std::move(moduleNames)}});
      }

      // Reset the global llvm options once compiling this target is done.
      ErrorOrSuccess resetResult = resetEmissionOptions(emissionOptions);
      if (resetResult.isError()) {
        return resetResult.takeError();
      }
    }
  }

  // Encode pending offload writes as a module attribute so cachedTransform
  // serializes them; flushOffloadWrites() drains them after.
  if (!offloadPendingWrites.empty()) {
    theModule->setAttr(
        kOffloadWritesAttrName,
        DictionaryAttr::get(theModule->getContext(), offloadPendingWrites));
  }

  // Set the final updated bitcode library data back on the original module.
  if (!currentBitcodeLibs.empty()) {
    SmallVector<LLVMBitcodeLibAttr> finalLibAttrs;
    for (const auto &[used, library] : currentBitcodeLibs)
      finalLibAttrs.push_back(LLVMBitcodeLibAttr::get(used, library));

    LLVMBitcodeLibArrayAttr finalArrayAttr =
        LLVMBitcodeLibArrayAttr::get(theModule->getContext(), finalLibAttrs);
    theModule->setAttr(LLVMBitcodeLibArrayAttr::getBitcodeLibsAttrName(),
                       finalArrayAttr);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Caching
//===----------------------------------------------------------------------===//

/// Returns Mojo transform backend, or an error if the backend could not be
/// created.
static ErrorOr<RCRef<Cache::BlobCacheBackend>>
getMojoCacheBackend(const std::string baseExtra) {
  std::filesystem::path path(kMojoCacheBaseDirName.str());
  if (!baseExtra.empty())
    path = path / baseExtra;
  path = path / "transform";
  return Cache::getLocalDefaultBackendChain(path, getVersionString());
}

//===----------------------------------------------------------------------===//
// createElaborateGeneratorsWithDefaultJIT
//===----------------------------------------------------------------------===//

/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass> KGEN::createElaborateGeneratorsWithDefaultJIT() {
  return createElaborateGenerators(TargetInfoAttr(), /*elabOpts=*/{},
                                   /*options=*/{}, compileElaboratorAsm,
                                   compileOffloads);
}

std::unique_ptr<Pass> KGEN::createElaborateGeneratorsWithDefaultJIT(
    const std::string &cacheBaseExtra) {
  CompilationOptions options;
  options.cacheBaseExtra = cacheBaseExtra;
  return createElaborateGenerators(TargetInfoAttr(), /*elabOpts=*/{}, options,
                                   compileElaboratorAsm, compileOffloads);
}

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

void KGEN::populateElaborateModulePasses(mlir::PassManager &pm,
                                         TargetInfoAttr target,
                                         const CompilationOptions &options) {
  buildElaborateModulePipeline(pm, target, options, compileElaboratorAsm,
                               compileOffloads);
  buildPostElaborationPipeline(pm, options);
}

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<ExecutionEngine>> KGEN::initializeExecutionEngine(
    MLIRContext &context, const CompilationOptions &compilationOptions,
    ExecutionEngineOptions executionEngineOptions, bool isJIT,
    PassManagerConfigOptions pmOptions) {

  // Now create the execution engine so we can JIT.
  auto tmOr = createTargetMachine(compilationOptions, isJIT);
  if (tmOr.isError())
    return tmOr.takeError();

  return ExecutionEngine::createWithStandardLayers(
      std::move(executionEngineOptions), **tmOr);
}

//===----------------------------------------------------------------------===//
// KGENCompiler
//===----------------------------------------------------------------------===//

KGENCompiler::KGENCompiler(MLIRContext &context, CompilationOptions options,
                           PassManagerConfigOptions pmConfigOptions)
    : options(std::move(options)), pmConfigOptions(std::move(pmConfigOptions)),
      context(context) {}

ErrorOrSuccess KGENCompiler::runKGENPipeline(ModuleOp theModule,
                                             TargetInfoAttr target) {
  auto cacheBackend = getMojoCacheBackend(options.cacheBaseExtra);
  if (cacheBackend.isError())
    return cacheBackend.takeError();

  // Run the passes as a cached transform.
  ContextRef ctx = loadContext(target.getContext());

  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(*cacheBackend));

  return runKGENPipeline(
      theModule, target, transformCache,
      ctx->get<AsyncRT::CPUDevice>()->getReadyChain().copy());
}

static mlir::PassManager
createPassManager(const std::optional<std::string> &operationName,
                  MLIRContext *context) {
  if (operationName)
    return {context, *operationName};
  return {context};
}

/// Name of the module attribute recording which offload output files were
/// requested using the `--emit` flag on mojo build. Only the kind is recorded,
/// not the output prefix.
static constexpr llvm::StringLiteral kOffloadRequestAttrName =
    "kgen.offload_output_request";

ErrorOrSuccess
KGENCompiler::runKGENPipeline(ModuleOp theModule, TargetInfoAttr target,
                              RCRef<Cache::TransformCache> transformCache,
                              AnyAsyncValueRef chain) {
  // Set the target now, so it's included in the cache key.
  if (!getTargetInfo(theModule))
    setTargetInfo(theModule, target);

  // Elaboration only records the offload files when the build asked for them.
  // Put the request in the cache key too, so an `--emit asm` build does not
  // reuse an entry from a build that skipped the files.
  if (!options.offloadOutputPrefix.empty())
    theModule->setAttr(
        kOffloadRequestAttrName,
        mlir::StringAttr::get(&context,
                              stringifyEmitAs(options.offloadOutputKind)));

  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  ErrorOrSuccess configPM = pmConfigOptions.configurePassManager(pm);
  if (configPM)
    return configPM.takeError();

  // Populate the passes.
  buildGenerateLibraryPipeline(pm, options);
  populateElaborateModulePasses(pm, target, options);

  // Run the passes as a cached transform.
  AsyncRT::AnyAsyncValueRef ready = Cache::cachedTransform(
      theModule, transformCache.copy(), std::move(chain), pm);

  // This await here is important since pm is local in this function.
  AsyncRT::await(ready);
  if (ready.isError())
    return ready.takeDiagnostic().getMessage().copy();

  // Flush pending offload writes.  Runs after cachedTransform on both hit and
  // miss paths, so files are always produced.
  if (!options.offloadOutputPrefix.empty()) {
    if (auto err = flushOffloadWrites(theModule, options.offloadOutputPrefix))
      return err;
    // This attribute changes the elaborator behavior only.
    // The back-end does not use it, so remove the attribute
    // here, to avoid polluting the back end cache.
    theModule->removeAttr(kOffloadRequestAttrName);
  }

  return success();
}

ErrorOrSuccess KGENCompiler::runGenerateLibraryPipeline(ModuleOp module) {
  auto cacheBackend = getMojoCacheBackend(options.cacheBaseExtra);
  if (cacheBackend.isError())
    return cacheBackend.takeError();
  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(*cacheBackend));

  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  ErrorOrSuccess configPM = pmConfigOptions.configurePassManager(pm);
  if (configPM) {
    return Error(
        std::string("configure PassManager in "
                    "KGENCompiler::runGenerateLibraryPipeline failed, ") +
        configPM.getError());
  }

  buildGenerateLibraryPipeline(pm, options);

  AsyncRT::CPUDevice &cpuDevice =
      *loadContext(module.getContext())->get<AsyncRT::CPUDevice>();
  AsyncRT::AnyAsyncValueRef ready = Cache::cachedTransform(
      module, transformCache.copy(),
      AsyncValueRef<Chain>::createReady(cpuDevice), pm,
      [](mlir::Operation *) {}, [](mlir::Operation *) {});

  // This await here is important since pm is local in this function.
  AsyncRT::await(ready);
  if (ready.isError())
    return ready.takeDiagnostic().getMessage().copy();

  return success();
}

LogicalResult KGENCompiler::runCheckLITPipeline(ModuleOp module) {
  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  ErrorOrSuccess configPM = pmConfigOptions.configurePassManager(pm);
  if (configPM) {
    return Error(std::string("configure PassManager in "
                             "KGENCompiler::runCheckLITPipeline failed, ") +
                 configPM.getError());
  }

  buildCheckLITPipeline(pm, options);
  return pm.run(module);
}

/// Run the compilation pipeline till the end of elaboration to produce a fully
/// concrete KGEN module. This allows the transform to be cached.
/// Note that this function also awaits the AsyncValue because it uses
/// a local PassManager.
/// Returns the same AnyAsyncValueRef for error handling in the caller
/// if needed.
ErrorOrSuccess KGENCompiler::runElaborationPipeline(
    ModuleOp module, TargetInfoAttr target, AsyncRT::CPUDevice &cpuDevice,
    std::optional<AnyAsyncValueRef> chain,
    std::function<void(Operation *)> moreOnMiss,
    std::function<void(Operation *)> moreOnHit) {

  // Set the target now, so it's included in the cache key.
  if (!getTargetInfo(module))
    setTargetInfo(module, target);

  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  ErrorOrSuccess configPM = pmConfigOptions.configurePassManager(pm);
  if (configPM)
    return configPM.takeError();

  populateElaborateModulePasses(pm, target, options);
  auto cacheBackend = getMojoCacheBackend(options.cacheBaseExtra);

  if (cacheBackend.isError() || !chain) {
    if (failed(pm.run(module)))
      return Error("KGENCompiler::runElaborationPipeline failed");
    return success();
  }

  AnyAsyncValueRef ready = Cache::cachedTransform(
      module, RCRef<Cache::TransformCache>::create(std::move(*cacheBackend)),
      std::move(*chain), pm, std::move(moreOnMiss), std::move(moreOnHit));

  // This await here is important since pm is local in this function.
  AsyncRT::await(ready);
  if (ready.isError())
    return ready.takeDiagnostic().getMessage().copy();
  return success();
}

static ErrorOrSuccess
setEmissionOptions(llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options,
                   StringRef emissionOpt, bool reset) {
  if (!emissionOpt.contains("=")) {
    return Error("emission option must be of the form `option=value`");
  }

  auto [key, value] = emissionOpt.split("=");
  llvm::cl::Option *opt = options.lookup(key);
  if (!opt)
    return Error("emission option \"" + Twine(key) + "\" is not found");
  if (reset) {
    opt->reset();
    return success();
  }
  // `addOccurrence` dispatches to the option's own value parser, so any
  // registered option type (bool, int, enum, string) is supported. Bool
  // spellings are lowercased since the bool parser only accepts lowercase.
  bool isBoolSpelling =
      value.equals_insensitive("true") || value.equals_insensitive("false");
  if (opt->addOccurrence(0, key, isBoolSpelling ? value.lower() : value.str()))
    return Error("invalid value \"" + Twine(value) +
                 "\" for emission option \"" + key + "\"");
  return success();
}

ErrorOrSuccess KGEN::parseEmissionOptions(EmissionOptions emissionOptions) {
  // Handle the emission options.
  // Parse the emission options from a comma separated list of values.
  llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options =
      llvm::cl::getRegisteredOptions();

  for (StringRef elem : emissionOptions) {
    ErrorOrSuccess setOr = setEmissionOptions(options, elem, false);
    if (setOr.isError())
      return setOr.takeError();
  }
  return success();
}

ErrorOrSuccess KGEN::resetEmissionOptions(EmissionOptions emissionOptions) {
  // Handle the emission options.
  // Parse the emission options from a comma separated list of values.
  llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options =
      llvm::cl::getRegisteredOptions();

  for (StringRef elem : emissionOptions) {
    ErrorOrSuccess setOr = setEmissionOptions(options, elem, true);
    if (setOr.isError())
      return setOr.takeError();
  }
  return success();
}
