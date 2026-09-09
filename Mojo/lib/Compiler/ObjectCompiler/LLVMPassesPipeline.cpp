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

#include "LLVMPassesPipeline.h"
#include "LLVMAccessorHelper.h"
#include "MCLinker.h"
#include "Mojo/Compiler/LLVMOptimizationPipeline.h"
#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "LLVM/Transforms/LLVMIRDowngradePass.h"
#include "LLVM/Transforms/SetFunctionAttributes.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/CGProfile.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/AlignmentFromAssumptions.h"
#include "llvm/Transforms/Scalar/AnnotationRemarks.h"
#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/Transforms/Scalar/CallSiteSplitting.h"
#include "llvm/Transforms/Scalar/ConstraintElimination.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/DivRemPairs.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopDistribute.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/LoopInstSimplify.h"
#include "llvm/Transforms/Scalar/LoopLoadElimination.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopSink.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/SpeculativeExecution.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/LibCallsShrinkWrap.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/RelLookupTableConverter.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "llvm/Transforms/Vectorize/VectorCombine.h"

using namespace llvm;
using namespace M::KGEN;

static SimplifyCFGOptions
adjustSimplifyCFGOptions(SimplifyCFGOptions simplifyCFGOptions,
                         const CompilationOptions &options) {
  M::ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(llvm::Triple(options.targetTriple));
  const TargetBackend *backend = backendOr.isError() ? nullptr : *backendOr;
  return backend ? backend->adjustSimplifyCFGOptions(simplifyCFGOptions)
                 : simplifyCFGOptions;
}

static OptimizationLevel getOptimizationLevel(unsigned level) {
  switch (level) {
  case 0:
    return OptimizationLevel::O0;
  default:
    return OptimizationLevel::O3;
  }
}

static void addSanitizers(ModulePassManager &modulePassManager,
                          const CompilationOptions &options) {
  M::ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(llvm::Triple(options.targetTriple));
  const TargetBackend *backend = backendOr.isError() ? nullptr : *backendOr;
  if (backend) {
    backend->addSanitizers(modulePassManager, options);
  } else {
    // No registered backend: apply the default host policy.
    addAddressSanitizerPass(modulePassManager, options);
    addThreadSanitizerPass(modulePassManager, options);
  }
}

static FunctionPassManager
buildFunctionSimplificationPipeline(PassBuilder passBuilder,
                                    const CompilationOptions &options) {
  OptimizationLevel level = getOptimizationLevel(options.optimizationLevel);
  FunctionPassManager fpm;
  // Form SSA out of local memory accesses after breaking apart aggregates into
  // scalars.
  fpm.addPass(SROAPass(SROAOptions::ModifyCFG));

  // Catch trivial redundancies
  fpm.addPass(EarlyCSEPass(true /* Enable mem-ssa. */));

  // Speculative execution if the target has divergent branches; otherwise nop.
  fpm.addPass(SpeculativeExecutionPass(/* OnlyIfDivergentTarget =*/true));

  // Optimize based on known information about branches, and cleanup afterward.
  fpm.addPass(JumpThreadingPass());
  fpm.addPass(CorrelatedValuePropagationPass());

  SimplifyCFGOptions simplifyCFGOptions = adjustSimplifyCFGOptions(
      SimplifyCFGOptions().convertSwitchRangeToICmp(true), options);

  fpm.addPass(SimplifyCFGPass(simplifyCFGOptions));
  fpm.addPass(InstCombinePass());
  fpm.addPass(AggressiveInstCombinePass());
  passBuilder.invokePeepholeEPCallbacks(fpm, level);

  fpm.addPass(ConstraintEliminationPass());

  fpm.addPass(LibCallsShrinkWrapPass());

  fpm.addPass(TailCallElimPass());
  fpm.addPass(SimplifyCFGPass(simplifyCFGOptions));

  // Form canonically associated expression trees, and simplify the trees using
  // basic mathematical properties. For example, this will form (nearly)
  // minimal multiplication trees.
  fpm.addPass(ReassociatePass());

  // Add the primary loop simplification pipeline.
  // FIXME: Currently this is split into two loop pass pipelines because we run
  // some function passes in between them. These can and should be removed
  // and/or replaced by scheduling the loop pass equivalents in the correct
  // positions. But those equivalent passes aren't powerful enough yet.
  // Specifically, `SimplifyCFGPass` and `InstCombinePass` are currently still
  // used. We have `LoopSimplifyCFGPass` which isn't yet powerful enough yet to
  // fully replace `SimplifyCFGPass`, and the closest to the other we have is
  // `LoopInstSimplify`.
  LoopPassManager lpm1, lpm2;

  // Simplify the loop body. We do this initially to clean up after other loop
  // passes run, either when iterating on a loop or on inner loops with
  // implications on the outer loop.
  lpm1.addPass(LoopInstSimplifyPass());
  lpm1.addPass(LoopSimplifyCFGPass());

  // Try to remove as much code from the loop header as possible,
  // to reduce amount of IR that will have to be duplicated. However,
  // do not perform speculative hoisting the first time as LICM
  // will destroy metadata that may not need to be destroyed if run
  // after loop rotation.
  // TODO: Investigate promotion cap for O1.
  lpm1.addPass(LICMPass(/*LicmMssaOptCap*/ 100,
                        /*LicmMssaNoAccForPromotionCap*/ 250,
                        /*AllowSpeculation=*/false));

  // Disable header duplication in loop rotation at -Oz.
  lpm1.addPass(LoopRotatePass(/*EnableHeaderDuplication*/ true, false));
  // TODO: Investigate promotion cap for O1.
  lpm1.addPass(LICMPass(/*LicmMssaOptCap*/ 100,
                        /*LicmMssaNoAccForPromotionCap*/ 250,
                        /*AllowSpeculation=*/true));
  lpm1.addPass(SimpleLoopUnswitchPass(/* NonTrivial */ true));

  lpm2.addPass(LoopIdiomRecognizePass());
  lpm2.addPass(IndVarSimplifyPass());

  lpm2.addPass(LoopDeletionPass());

  fpm.addPass(createFunctionToLoopPassAdaptor(std::move(lpm1),
                                              /*UseMemorySSA=*/true));

  fpm.addPass(SimplifyCFGPass(simplifyCFGOptions));
  fpm.addPass(InstCombinePass());
  // The loop passes in LPM2 (LoopIdiomRecognizePass, IndVarSimplifyPass,
  // LoopDeletionPass and LoopFullUnrollPass) do not preserve MemorySSA.
  // *All* loop passes must preserve it, in order to be able to use it.
  fpm.addPass(createFunctionToLoopPassAdaptor(std::move(lpm2),
                                              /*UseMemorySSA=*/false));

  // Delete small array after loop unroll.
  fpm.addPass(SROAPass(SROAOptions::ModifyCFG));

  // Try vectorization/scalarization transforms that are both improvements
  // themselves and can allow further folds with GVN and InstCombine.
  fpm.addPass(VectorCombinePass(/*TryEarlyFoldsOnly=*/true));

  // Eliminate redundancies.
  fpm.addPass(MergedLoadStoreMotionPass());
  fpm.addPass(GVNPass());

  // Sparse conditional constant propagation.
  // FIXME: It isn't clear why we do this *after* loop passes rather than
  // before...
  fpm.addPass(SCCPPass());

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  fpm.addPass(BDCEPass());

  // Run instcombine after redundancy and dead bit elimination to exploit
  // opportunities opened up by them.
  fpm.addPass(InstCombinePass());
  passBuilder.invokePeepholeEPCallbacks(fpm, level);

  fpm.addPass(JumpThreadingPass());
  fpm.addPass(CorrelatedValuePropagationPass());

  // Finally, do an expensive DCE pass to catch all the dead code exposed by
  // the simplifications and basic cleanup after all the simplifications.
  // TODO: Investigate if this is too expensive.
  fpm.addPass(ADCEPass());

  // Specially optimize memory movement as it doesn't look like dataflow in SSA.
  fpm.addPass(MemCpyOptPass());

  fpm.addPass(DSEPass());
  fpm.addPass(createFunctionToLoopPassAdaptor(
      LICMPass(/*LicmMssaOptCap*/ 100, /*LicmMssaNoAccForPromotionCap*/ 250,
               /*AllowSpeculation=*/true),
      /*UseMemorySSA=*/true));

  fpm.addPass(SimplifyCFGPass(
      adjustSimplifyCFGOptions(SimplifyCFGOptions()
                                   .convertSwitchRangeToICmp(true)
                                   .hoistCommonInsts(true)
                                   .sinkCommonInsts(true),
                               options)));
  fpm.addPass(InstCombinePass());
  passBuilder.invokePeepholeEPCallbacks(fpm, level);

  return fpm;
}

static void addInlinerPasses(PassBuilder passBuilder, ModulePassManager &MPM,
                             const CompilationOptions &options) {
  ModuleInlinerWrapperPass miwp(
      getInlineParamsFromOptLevel(/*OptLevel=*/3),
      /*PerformMandatoryInliningsFirst*/ true,
      InlineContext{ThinOrFullLTOPhase::None, InlinePass::CGSCCInliner},
      InliningAdvisorMode::Default,
      /*MaxDevirtIterations*/ 4);

  // Require the GlobalsAA analysis for the module so we can query it within
  // the CGSCC pipeline.
  miwp.addModulePass(RequireAnalysisPass<GlobalsAA, Module>());
  // Invalidate AAManager so it can be recreated and pick up the newly
  // available GlobalsAA.
  miwp.addModulePass(
      createModuleToFunctionPassAdaptor(InvalidateAnalysisPass<AAManager>()));

  // Require the ProfileSummaryAnalysis for the module so we can query it
  // within the inliner pass.
  miwp.addModulePass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());
  // Now begin the main postorder CGSCC pipeline.
  // FIXME: The current CGSCC pipeline has its origins in the legacy pass
  // manager and trying to emulate its precise behavior. Much of this doesn't
  // make a lot of sense and we should revisit the core CGSCC structure.
  CGSCCPassManager &mainCgPipeline = miwp.getPM();

  // Note: historically, the PruneEH pass was run first to deduce nounwind and
  // generally clean up exception handling overhead. It isn't clear this is
  // valuable as the inliner doesn't currently care whether it is inlining an
  // invoke or a call.

  // Only recursive functions can affect the simplification pipeline below. The
  // rest are covered by the unrestricted run after it, matching upstream's
  // split in https://reviews.llvm.org/D145210.
  mainCgPipeline.addPass(PostOrderFunctionAttrsPass(/*SkipNonRecursive=*/true));

  // Lastly, add the core function simplification pipeline nested inside the
  // CGSCC walk.
  mainCgPipeline.addPass(createCGSCCToFunctionPassAdaptor(
      buildFunctionSimplificationPipeline(passBuilder, options),
      /*EagerlyInvalidateAnalyses*/ true,
      /*EnableNoRerunSimplificationPipeline*/ true));

  // Range attributes like `initializes` are only derivable once simplification
  // has collapsed loop-based initialization into single memory operations.
  mainCgPipeline.addPass(PostOrderFunctionAttrsPass());

  MPM.addPass(std::move(miwp));
}

static void addVectorPasses(FunctionPassManager &FPM,
                            const CompilationOptions &options) {
  // Eliminate loads by forwarding stores from the previous iteration to loads
  // of the current iteration.
  FPM.addPass(LoopLoadEliminationPass());

  // Cleanup after the loop optimization passes.
  FPM.addPass(InstCombinePass());

  // Now that we've formed fast to execute loop structures, we do further
  // optimizations. These are run afterward as they might block doing complex
  // analyses and transforms such as what are needed for loop vectorization.

  // Cleanup after loop vectorization, etc. Simplification passes like CVP and
  // GVN, loop transforms, and others have already run, so it's now better to
  // convert to more optimized IR using more aggressive simplify CFG options.
  // The extra sinking transform can create larger basic blocks, so do this
  // before SLP vectorization.
  FPM.addPass(SimplifyCFGPass(
      adjustSimplifyCFGOptions(SimplifyCFGOptions()
                                   .forwardSwitchCondToPhi(true)
                                   .convertSwitchRangeToICmp(true)
                                   .convertSwitchToLookupTable(true)
                                   .needCanonicalLoops(false)
                                   .hoistCommonInsts(true)
                                   .sinkCommonInsts(true),
                               options)));

  FPM.addPass(SLPVectorizerPass());
  // Enhance/cleanup vector code.
  FPM.addPass(VectorCombinePass());

  FPM.addPass(InstCombinePass());
  // Now that we are done with loop unrolling, be it either by LoopVectorizer,
  // or LoopUnroll passes, some variable-offset GEP's into alloca's could have
  // become constant-offset, thus enabling SROA and alloca promotion. Do so.
  // NOTE: we are very late in the pipeline, and we don't have any LICM
  // or SimplifyCFG passes scheduled after us, that would cleanup
  // the CFG mess this may created if allowed to modify CFG, so forbid that.
  FPM.addPass(SROAPass(SROAOptions::PreserveCFG));
  FPM.addPass(InstCombinePass());
  FPM.addPass(
      RequireAnalysisPass<OptimizationRemarkEmitterAnalysis, Function>());
  FPM.addPass(createFunctionToLoopPassAdaptor(
      LICMPass(/*LicmMssaOptCap*/ 100, /*LicmMssaNoAccForPromotionCap*/ 250,
               /*AllowSpeculation=*/true),
      /*UseMemorySSA=*/true));

  // Now that we've vectorized and unrolled loops, we may have more refined
  // alignment information, try to re-derive it here.
  FPM.addPass(AlignmentFromAssumptionsPass());
}

static ModulePassManager buildO3Pipeline(PassBuilder &passBuilder,
                                         const CompilationOptions &options,
                                         const TargetBackend *backend) {
  OptimizationLevel level = OptimizationLevel::O3;
  ModulePassManager mpm;

  if (backend)
    backend->addPipelineStartPasses(mpm, options);

  // Do basic inference of function attributes from known properties of system
  // libraries and other oracles.
  mpm.addPass(InferFunctionAttrsPass());

  passBuilder.invokePipelineStartEPCallbacks(mpm, OptimizationLevel::O3);

  // Create an early function pass manager to cleanup the output of the
  // frontend.
  FunctionPassManager earlyFpm;
  // Lower llvm.expect to metadata before attempting transforms.
  // Compare/branch metadata may alter the behavior of passes like SimplifyCFG.
  earlyFpm.addPass(LowerExpectIntrinsicPass());
  earlyFpm.addPass(SimplifyCFGPass());
  earlyFpm.addPass(SROAPass(SROAOptions::ModifyCFG));
  earlyFpm.addPass(EarlyCSEPass());
  earlyFpm.addPass(CallSiteSplittingPass());

  mpm.addPass(
      createModuleToFunctionPassAdaptor(std::move(earlyFpm),
                                        /*EagerlyInvalidateAnalyses*/ true));

  passBuilder.invokePipelineEarlySimplificationEPCallbacks(
      mpm, OptimizationLevel::O3, ThinOrFullLTOPhase::None);

  // Promote any localized globals to SSA registers.
  // FIXME: Should this instead by a run of SROA?
  // FIXME: We should probably run instcombine and simplifycfg afterward to
  // delete control flows that are dead once globals have been folded to
  // constants.
  mpm.addPass(createModuleToFunctionPassAdaptor(PromotePass()));

  // Create a small function pass pipeline to cleanup after all the global
  // optimizations.
  FunctionPassManager globalCleanupPm;
  globalCleanupPm.addPass(InstCombinePass());
  passBuilder.invokePeepholeEPCallbacks(globalCleanupPm, OptimizationLevel::O3);

  SimplifyCFGOptions simplifyCFGOptions = adjustSimplifyCFGOptions(
      SimplifyCFGOptions().convertSwitchRangeToICmp(true), options);

  globalCleanupPm.addPass(SimplifyCFGPass(simplifyCFGOptions));
  mpm.addPass(
      createModuleToFunctionPassAdaptor(std::move(globalCleanupPm),
                                        /*EagerlyInvalidateAnalyses*/ true));

  addInlinerPasses(passBuilder, mpm, options);

  // Optimize globals now that the module is fully simplified.
  mpm.addPass(GlobalDCEPass());

  CGSCCPassManager cgpm;
  passBuilder.invokeCGSCCOptimizerLateEPCallbacks(cgpm, level);
  mpm.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(cgpm)));

  // Do RPO function attribute inference across the module to forward-propagate
  // attributes where applicable.
  // FIXME: Is this really an optimization rather than a canonicalization?
  mpm.addPass(ReversePostOrderFunctionAttrsPass());

  // Re-compute GlobalsAA here prior to function passes. This is particularly
  // useful as the above will have inlined, DCE'ed, and function-attr
  // propagated everything. We should at this point have a reasonably minimal
  // and richly annotated call graph. By computing aliasing and mod/ref
  // information for all local globals here, the late loop passes and notably
  // the vectorizer will be able to use them to help recognize vectorizable
  // memory operations.
  mpm.addPass(RecomputeGlobalsAAPass());

  FunctionPassManager optimizePm;
  optimizePm.addPass(Float2IntPass());
  optimizePm.addPass(LowerConstantIntrinsicsPass());

  // FIXME: We need to run some loop optimizations to re-rotate loops after
  // simplifycfg and others undo their rotation.

  // Optimize the loop execution. These passes operate on entire loop nests
  // rather than on each loop in an inside-out manner, and so they are actually
  // function passes.

  LoopPassManager lpm;
  // First rotate loops that may have been un-rotated by prior passes.
  // Disable header duplication at -Oz.
  lpm.addPass(
      LoopRotatePass(/*EnableHeaderDuplication*/ true, /*LTOPreLink*/ false));
  // Some loops may have become dead by now. Try to delete them.
  // FIXME: see discussion in https://reviews.llvm.org/D112851,
  //        this may need to be revisited once we run GVN before loop deletion
  //        in the simplification pipeline.
  lpm.addPass(LoopDeletionPass());
  optimizePm.addPass(
      createFunctionToLoopPassAdaptor(std::move(lpm), /*UseMemorySSA=*/false));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependencies
  // into separate loop that would otherwise inhibit vectorization.  This is
  // currently only performed for loops marked with the metadata
  // llvm.loop.distribute=true or when -enable-loop-distribute is specified.
  optimizePm.addPass(LoopDistributePass());

  // Populates the VFABI attribute with the scalar-to-vector mappings
  // from the TargetLibraryInfo.
  optimizePm.addPass(InjectTLIMappings());

  addVectorPasses(optimizePm, options);

  // LoopSink pass sinks instructions hoisted by LICM, which serves as a
  // canonicalization pass that enables other optimizations. As a result,
  // LoopSink pass needs to be a very late IR pass to avoid undoing LICM
  // result too early.
  optimizePm.addPass(LoopSinkPass());

  // And finally clean up LCSSA form before generating code.
  optimizePm.addPass(InstSimplifyPass());

  // This hoists/decomposes div/rem ops. It should run after other sink/hoist
  // passes to avoid re-sinking, but before SimplifyCFG because it can allow
  // flattening of blocks.
  optimizePm.addPass(DivRemPairsPass());

  // Try to annotate calls that were created during optimization.
  optimizePm.addPass(TailCallElimPass());

  // LoopSink (and other loop passes since the last simplifyCFG) might have
  // resulted in single-entry-single-exit or empty blocks. Clean up the CFG.

  optimizePm.addPass(SimplifyCFGPass(simplifyCFGOptions));

  // Add the core optimizing pipeline.
  mpm.addPass(
      createModuleToFunctionPassAdaptor(std::move(optimizePm),
                                        /*EagerlyInvalidateAnalyses*/ true));

  passBuilder.invokeOptimizerLastEPCallbacks(mpm, level,
                                             ThinOrFullLTOPhase::None);
  // FIXME: We don't officially have full-lto, therefore no passes from
  // invokeFullLinkTimeOptimizationEarlyEPCallbacks are added here.

  // Add any relevant sanitizers.
  addSanitizers(mpm, options);

  // Now we need to do some global optimization transforms.
  // FIXME: It would seem like these should come first in the optimization
  // pipeline and maybe be the bottom of the canonicalization pipeline? Weird
  // ordering here.
  mpm.addPass(GlobalDCEPass());
  mpm.addPass(ConstantMergePass());

  mpm.addPass(CGProfilePass(false));

  // TODO: Relative look table converter pass caused an issue when full lto is
  // enabled. See https://reviews.llvm.org/D94355 for more details.
  // Until the issue fixed, disable this pass during pre-linking phase.
  mpm.addPass(RelLookupTableConverterPass());

  // Emit annotation remarks.
  mpm.addPass(createModuleToFunctionPassAdaptor(AnnotationRemarksPass()));

  return mpm;
}

static ModulePassManager buildO0Pipeline(PassBuilder &passBuilder,
                                         const CompilationOptions &options,
                                         const TargetBackend *backend) {
  OptimizationLevel level = getOptimizationLevel(options.optimizationLevel);
  ModulePassManager mpm;

  if (backend)
    backend->addPipelineStartPasses(mpm, options);

  passBuilder.invokePipelineStartEPCallbacks(mpm, OptimizationLevel::O0);

  passBuilder.invokePipelineEarlySimplificationEPCallbacks(
      mpm, OptimizationLevel::O0, ThinOrFullLTOPhase::None);

  // Build a minimal pipeline based on the semantics required by LLVM,
  // which is just that always inlining occurs. Further, disable generating
  // lifetime intrinsics to avoid enabling further optimizations during
  // code generation.
  mpm.addPass(AlwaysInlinerPass(
      /*InsertLifetimeIntrinsics=*/false));
  CGSCCPassManager cgpm;
  passBuilder.invokeCGSCCOptimizerLateEPCallbacks(cgpm, level);
  mpm.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(cgpm)));

  passBuilder.invokeOptimizerLastEPCallbacks(mpm, level,
                                             ThinOrFullLTOPhase::None);

  // Add any relevant sanitizers.
  addSanitizers(mpm, options);

  mpm.addPass(createModuleToFunctionPassAdaptor(AnnotationRemarksPass()));

  return mpm;
}

ModulePassManager
M::KGEN::buildLLVMOptimizationPipeline(PassBuilder &passBuilder,
                                       const CompilationOptions &options) {
  M::ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(llvm::Triple(options.targetTriple));
  const TargetBackend *backend = backendOr.isError() ? nullptr : *backendOr;

  // A backend may fully own its pipeline (e.g. Metal's AIR legalization),
  // replacing the standard optimization pipeline below.
  if (backend) {
    ModulePassManager mpm;
    if (backend->buildLLVMPipeline(mpm, passBuilder, options))
      return mpm;
  }

  CodeGenOptLevel optLevel = options.getCodeGenOptLevel();
  if (optLevel == CodeGenOptLevel::None)
    return buildO0Pipeline(passBuilder, options, backend);
  // All non-zero optimization levels are currently equivalent.
  return buildO3Pipeline(passBuilder, options, backend);
}

void M::KGEN::addLLVMIRDowngradePass(llvm::ModulePassManager &mpm) {
  mpm.addPass(LLVMIRDowngradePass());
}

void M::KGEN::registerKGENLLVMPasses(PassBuilder &passBuilder) {
  // Target-agnostic passes.
  passBuilder.registerPipelineParsingCallback(
      [](StringRef name, ModulePassManager &mpm,
         ArrayRef<PassBuilder::PipelineElement>) -> bool {
        if (name == "kgen-llvmir-downgrade") {
          mpm.addPass(LLVMIRDowngradePass());
          return true;
        }
        return false;
      });

  // Backend-specific passes (e.g. Metal's kgen-metal-air).
  for (const std::unique_ptr<TargetBackend> &backend :
       TargetBackendRegistry::get().backends())
    backend->registerPipelinePasses(passBuilder);
}

bool M::KGEN::addPassesToEmitFile(CompilationOptions &options,
                                  TargetMachine &targetMachine,
                                  llvm::legacy::PassManagerBase &pm,
                                  raw_pwrite_stream &out,
                                  raw_pwrite_stream *dwoOut,
                                  CodeGenFileType fileType, bool disableVerify,
                                  MachineModuleInfoWrapperPass *mmiwp) {
  return targetMachine.addPassesToEmitFile(pm, out, dwoOut, fileType,
                                           disableVerify, mmiwp);
}

/// Module pass to reset the machine function numbering base so that we can
/// uniqueing llvm::MachineFunctions across all llvm::Module splits.
namespace {
class SetMachineFunctionBasePass : public llvm::ImmutablePass {
public:
  static char ID; // Pass identification, replacement for typeid
  SetMachineFunctionBasePass(llvm::MachineModuleInfo &mmi, unsigned base);

  // Initialization and Finalization
  bool doInitialization(llvm::Module &) override;
  bool doFinalization(llvm::Module &) override;

private:
  llvm::MachineModuleInfo &mmi;
  unsigned base;
};
} // namespace

char SetMachineFunctionBasePass::ID;

SetMachineFunctionBasePass::SetMachineFunctionBasePass(
    llvm::MachineModuleInfo &mmi, unsigned base)
    : llvm::ImmutablePass(ID), mmi(mmi), base(base) {}

// Initialization and Finalization
bool SetMachineFunctionBasePass::doInitialization(llvm::Module &) {
  setNextFnNum(mmi, base);
  return false;
}

bool SetMachineFunctionBasePass::doFinalization(llvm::Module &) {
  return false;
}

/// Build a pipeline that does machine specific codegen but stops before
/// AsmPrint. Returns true if failed.
bool M::KGEN::addPassesToEmitMC(CompilationOptions &options,
                                llvm::TargetMachine &targetMachine,
                                llvm::legacy::PassManagerBase &pm,
                                llvm::raw_pwrite_stream &out,
                                bool disableVerify,
                                llvm::MachineModuleInfoWrapperPass *mmiwp,
                                unsigned numFnBase) {
  // Targets may override createPassConfig to provide a target-specific
  // subclass.
  TargetPassConfig *passConfig = targetMachine.createPassConfig(pm);

  // Set PassConfig options provided by TargetMachine.
  passConfig->setDisableVerify(disableVerify);
  pm.add(passConfig);
  pm.add(mmiwp);

  auto smfbp = new SetMachineFunctionBasePass(mmiwp->getMMI(), numFnBase);
  pm.add(smfbp);

  if (passConfig->addISelPasses())
    return true;

  passConfig->addMachinePasses();
  passConfig->setInitialized();

  return false;
}

/// Function pass to populate external MCSymbols to other llvm module split's
/// MCContext so that they can be unique across all splits. This uniqueing
/// is required for ORCJIT (not for generating binary .o).
namespace {
class SyncX86SymbolTables : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit SyncX86SymbolTables(SmallVectorImpl<MCInfo *> &);

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  SmallVectorImpl<MCInfo *> &mcInfos;
  DenseSet<StringRef> externSymbols;

  // Populate MCSymbol to all the MCContexts.
  void populateSymbol(StringRef, const MCSymbolTableValue &, MCContext *srcCtx);
};
} // namespace

char SyncX86SymbolTables::ID;

SyncX86SymbolTables::SyncX86SymbolTables(SmallVectorImpl<MCInfo *> &mcInfos)
    : MachineFunctionPass(ID), mcInfos(mcInfos) {}

void SyncX86SymbolTables::populateSymbol(StringRef name,
                                         const llvm::MCSymbolTableValue &value,
                                         MCContext *srcCtx) {
  for (MCInfo *mcInfo : mcInfos) {
    MCContext &currCtx = *mcInfo->mcContext;
    if (&currCtx == srcCtx)
      continue;
    MCSymbolTableEntry &entry =
        M::KGEN::getMCContextSymbolTableEntry(name, currCtx);
    if (!entry.second.Symbol) {
      entry.second.Symbol = value.Symbol;
      entry.second.NextUniqueID = value.NextUniqueID;
      entry.second.Used = value.Used;
    }
  }
}

bool SyncX86SymbolTables::runOnMachineFunction(MachineFunction &MF) {
  MCContext &ctx = MF.getContext();
  for (auto &[name, symbolEntry] : ctx.getSymbols()) {
    if (!symbolEntry.Symbol || !symbolEntry.Symbol->isUndefined() ||
        externSymbols.contains(name))
      continue;
    externSymbols.insert(name);
    populateSymbol(name, symbolEntry, &ctx);
  }
  return false;
}

/// Build a pipeline that does AsmPrint only.
/// Returns true if failed.
bool M::KGEN::addPassesToAsmPrint(CompilationOptions &options,
                                  llvm::TargetMachine &targetMachine,
                                  llvm::legacy::PassManagerBase &pm,
                                  llvm::raw_pwrite_stream &out,
                                  llvm::CodeGenFileType fileType,
                                  bool disableVerify,
                                  llvm::MachineModuleInfoWrapperPass *mmiwp,
                                  llvm::SmallVectorImpl<MCInfo *> &mcInfos) {
  TargetPassConfig *passConfig = targetMachine.createPassConfig(pm);
  if (!passConfig)
    return true;
  // Set PassConfig options provided by TargetMachine.
  passConfig->setDisableVerify(disableVerify);
  pm.add(passConfig);
  pm.add(mmiwp);
  passConfig->setInitialized();

  bool result = targetMachine.addAsmPrinter(pm, out, nullptr, fileType,
                                            mmiwp->getMMI().getContext());

  if (options.targetTriple.find("x86") != std::string::npos)
    pm.add(new SyncX86SymbolTables(mcInfos));
  return result;
}
