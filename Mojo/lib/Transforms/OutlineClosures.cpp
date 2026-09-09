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

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "outline-closures"

namespace M::KGEN {
#define GEN_PASS_DEF_OUTLINECLOSURES
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct OutlineClosuresPass
    : M::KGEN::impl::OutlineClosuresBase<OutlineClosuresPass> {
  using OutlineClosuresBase::OutlineClosuresBase;

  void runOnOperation() override;
};
} // namespace

/// Reconstruct the signature using a list of named input parameters and indices
/// indicating which one of them are variadic. These parameters are prepended to
/// the current signature and references are remapped to index references.
static FuncTypeGeneratorType
prependParams(FuncTypeGeneratorType sigGen,
              ArrayRef<ParamDeclAttr> parentParams) {
  assert(!sigGen.getParamListAttrs() && "unlowered lit signature");

  IndexRefRemapper remapper(parentParams, parentParams.size());
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : parentParams)
    inputParamTypes.push_back(remapper.replace(param.getType()));
  for (Type type : sigGen.getInputParamTypes())
    inputParamTypes.push_back(remapper.replace(type));

  FuncType sig = sigGen.getBody();
  PogListAttr argListAttrs;
  if (PogListAttr origArgListAttrs = sig.getArgListAttrs())
    argListAttrs = cast<PogListAttr>(remapper.replace(origArgListAttrs));
  return FuncTypeGeneratorType::get(
      inputParamTypes, remapper.replace(sig.getValues()),
      sig.getArgConventions(), sig.getFnEffects(),
      /*fnMetadata=*/{}, /*genMetadata=*/{}, argListAttrs);
}

void OutlineClosuresPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  // Walk over all the param.declare.region ops and create structs with the SSA
  // captures, use BindParamsAttr to deal with parameter captures.
  unsigned counter = 0;
  DenseSet<GeneratorOp> outlinedGenerators;
  for (auto generator : theModule.getOps<GeneratorOp>()) {
    unsigned varCounter = 0;

    // Calculate the parameter decls and uses for the region decl's parent.
    ParameterUseDefGraph uses(generator.getBodyRegion());
    uses.calculate(paramCache);

    bool hadError = false;
    SmallVector<Operation *> toErase;
    generator.walk([&](ParamDeclareRegionOp regionDecl) {
      StringRef regionName = regionDecl.getParamDecl().getName();

      auto emitAndSetError = [&hadError](Value capture,
                                         ParamDeclareRegionOp regionDecl,
                                         StringLiteral rootMsg) {
        InFlightDiagnostic diag = mlir::emitError(regionDecl.getLoc())
                                  << rootMsg;

        auto maybeUser = llvm::find_if(capture.getUsers(), [&](Operation *op) {
          return regionDecl->isProperAncestor(op);
        });

        if (maybeUser != capture.getUsers().end())
          diag.attachNote((*maybeUser)->getLoc())
              << "use of captured value here";
        diag.attachNote(capture.getLoc()) << "captured value defined here";
        hadError = true;
      };

      // Value captures are easy (ish)
      llvm::SetVector<Value> captures;
      mlir::getUsedValuesDefinedAbove(regionDecl->getRegions(), captures);
      if (!captures.empty() &&
          !regionDecl.getFuncTypeGenerator().getBody().isCapturing()) {
        emitAndSetError(captures.front(), regionDecl,
                        "nested function is marked as @noncapturing, but it "
                        "captures values");
        return;
      }

      // We will use this builder to build the lifted generator.
      ImplicitLocOpBuilder b(regionDecl->getLoc(), regionDecl.getContext());

      // Collect any parameters used from above that we need to capture for the
      // lifted generator.
      llvm::SetVector<ParamDeclAttr> capturedParamDecls;
      SmallVector<ParamDeclRefAttr> capturedParamValues;
      Region &region = regionDecl.getBodyRegion();
      auto regionDeclUses = uses.nestedScopes.find(&region);
      assert(regionDeclUses != uses.nestedScopes.end());

      // Scan the captured values for captured parameters.
      ParameterCollector collector(paramCache);
      SmallVector<ParamDeclRefAttr, 16> capturedUses;
      for (Value capture : captures) {
        bool unusedHasConstExpr = false;
        size_t unusedRequiredSignatureDepth = 0;
        collector.collectUsesFromType(capture.getType(), capturedUses,
                                      unusedHasConstExpr,
                                      unusedRequiredSignatureDepth);
      }

      // Scan locations for captured parameters when in a debug build.
      if (debugBuild) {
        regionDecl.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
          // Since nested regions aren't being deleted, walk over them.
          if (op != regionDecl && isa<ParamDeclareRegionOp>(op))
            return WalkResult::skip();
          bool unusedHasConstExpr = false;
          size_t unusedRequiredSignatureDepth = 0;
          collector.collectUsesFromAttr(op->getLoc(), capturedUses,
                                        unusedHasConstExpr,
                                        unusedRequiredSignatureDepth);
          return WalkResult::advance();
        });
      }

      // Add additional uses to the captures set.
      for (ParamDeclRefAttr capturedUse : capturedUses) {
        auto declOpIter =
            regionDeclUses->second.decls.find(capturedUse.getName());
        // If a parameter was defined in a nested scope like kgen.param.for,
        // it is not at or above the current region scope.
        // Hence, parameter will not be in the map, and it is safe to ignore it.
        if (declOpIter == regionDeclUses->second.decls.end())
          continue;

        Operation *declOp = declOpIter->second.declOp;
        if (!regionDecl->isAncestor(declOp))
          regionDeclUses->second.usesFromAbove.insert(capturedUse);
      }

      for (ParamDeclRefAttr useFromAbove :
           regionDeclUses->second.usesFromAbove) {
        auto decl =
            ParamDeclAttr::get(useFromAbove.getName(), useFromAbove.getType());
        if (capturedParamDecls.insert(decl))
          capturedParamValues.push_back(useFromAbove);
      }

      // Create a wrapper that knows how to handle the global variable. It has
      // the same parameter signature as the lifted region, but it has the same
      // value signature as the original parameter region (no captures - those
      // come from global variables).
      SmallVector<ParamDeclAttr> inputParamDecls(
          capturedParamDecls.getArrayRef());
      llvm::append_range(inputParamDecls, regionDecl.getInputParams());

      FuncTypeGeneratorType wrapperSignature = prependParams(
          regionDecl.getFuncTypeGenerator(), capturedParamDecls.getArrayRef());

      b.setInsertionPoint(generator);
      auto uniqueName = b.getStringAttr(getUniqueSymbolName(
          (generator.getName() + "_" + regionName).str(), symtab, counter));

      auto liftedWrapper = GeneratorOp::create(
          b, uniqueName, regionDecl.getSourceNameAttr(), wrapperSignature,
          regionDecl.getFunctionType(), inputParamDecls,
          regionDecl.getInlineLevel(), /*inlinedForm=*/nullptr,
          regionDecl.getLinkageNameAttr(), regionDecl.getLLVMMetadataArray(),
          regionDecl.getLLVMArgMetadataArray());
      symtab.insert(liftedWrapper);
      outlinedGenerators.insert(liftedWrapper);
      auto wrapperSymbol = SymbolConstantAttr::get(liftedWrapper);

      // Take the body from the param region.
      Region &body = liftedWrapper.getBodyRegion();
      body.takeBody(region);

      // Add the original arguments to the call after the captures. Since the
      // captures are the last N arguments, we can simply drop them.
      b.setInsertionPointToStart(liftedWrapper.getBody());

      // Fill the body of the wrapper.
      for (auto [idx, capture] : llvm::enumerate(captures)) {
        if (isa<KGEN::NoneType>(capture.getType())) {
          emitAndSetError(capture, regionDecl,
                          "we do not expect the capturing of None type.");
          return;
        }

        auto load = POP::CompilerGlobalLoadOp::create(
            b, capture.getType(),
            b.getStringAttr(generator.getName() + "_context_var_" +
                            Twine(varCounter + idx)));
        // HACK: Because we don't track lifetimes of captured variables in
        // parameter closures correctly, we might get erroneous origin markers
        // of captured stack allocations. Just clear them out for now.
        for (OpOperand &use : llvm::make_early_inc_range(capture.getUses())) {
          Operation *user = use.getOwner();
          if (body.isAncestor(user->getParentRegion())) {
            if (isa<POP::StackAllocLifetimeStartOp,
                    POP::StackAllocLifetimeEndOp>(user))
              user->eraseOperand(use.getOperandNumber());
            else
              use.set(load);
          }
        }
      }

      // Since the lifted generator will have a new name, we need to update the
      // linkage name in the subprogram information.
      DebugInfo::updateSubprogram(liftedWrapper,
                                  liftedWrapper.getSymNameAttr());

      Attribute signature = wrapperSymbol;
      // If we have parameter captures, create a BindParamsAttr.
      if (!capturedParamValues.empty()) {
        // OK cool, now we need a partial binding. First we insert the lifted
        // symbol at the beginning of the vector.
        SmallVector<TypedAttr> partialBindings;
        llvm::append_range(partialBindings, capturedParamValues);

        // Ignore implicit lifetimes.
        for (Type paramType :
             regionDecl.getFuncTypeGenerator().getInputParamTypes())
          partialBindings.push_back(UnboundAttr::get(paramType));

        signature = BindParamsAttr::get(wrapperSymbol.getContext(),
                                        wrapperSymbol, partialBindings,
                                        /*evaluationContext=*/nullptr);
      }

      // The param region's location should be the fusion of the file location
      // and the program scope.
      mlir::FunctionOpInterface caller =
          regionDecl->getParentOfType<mlir::FunctionOpInterface>();
      if (DebugInfo::DISubprogramAttr scope = DebugInfo::extractScope(caller)) {
        Location fileOnlyLoc =
            DebugInfo::extractSourceLoc(regionDecl->getLoc());
        b.setLoc(FusedLoc::get(scope.getContext(), fileOnlyLoc, scope));
      }

      // Set the insertion point to the regionDecl.
      b.setInsertionPoint(regionDecl);

      // Create a container for the struct with all the various captures.
      for (auto [idx, capture] : llvm::enumerate(captures)) {

        POP::CompilerGlobalStoreOp::create(
            b,
            b.getStringAttr(generator.getName() + "_context_var_" +
                            Twine(varCounter + idx)),
            capture);
      }

      // Create the decl that replaces the regionDecl with its parameter being
      // this new partial binding.
      ParamDeclareOp::create(b, regionDecl.getParamDecl(),
                             cast<TypedAttr>(signature));

      // And we can drop the regionDecl now, we're done with it.
      toErase.push_back(regionDecl);
      varCounter += captures.size();
    });
    if (hadError)
      return signalPassFailure();

    for (Operation *op : toErase)
      op->erase();
  }
}
