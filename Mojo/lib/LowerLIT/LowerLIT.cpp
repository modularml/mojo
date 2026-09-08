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

#include "LowerLITTypes.h"
#include "SingletonTypeHelper.h"

#include "ConcreteBindings.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLIT
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// This processes a `lit.fn` and returns the param declarations for the
/// normal input parameters, ignoring the origin parameters.
static ArrayRef<ParamDeclAttr> extractImplicitOriginParams(FnOp func) {
  size_t numImplicitOrigins =
      func.getFuncTypeGenerator().getNumImplicitOriginDecls();
  return func.getInputParams().drop_back(numImplicitOrigins);
}

/// The param decl positions that have been dropped.
using ParamDeclDropMask = llvm::BitVector;

/// Check a list of parameter declarations to see if any of the parameters are
/// singletons like origin parameters.  If so, remove them from the list.
static ParamDeclDropMask
removeSingletonParamDecls(SingletonTypeHelper &singletonTypeHelper,
                          SmallVectorImpl<ParamDeclAttr> &paramDecls) {
  ParamDeclDropMask mask(paramDecls.size());
  size_t numRemoved = 0;
  for (auto [idx, paramDecl] : llvm::enumerate(paramDecls)) {
    // If this is a parameter we are supposed to remove, bind it.
    if (singletonTypeHelper.isSingletonType(paramDecl.getType())) {
      // We can just remove the parameter without inserting a placeholder
      // in the body. This is safe because we unconditionally replace
      // all attributes of origin type at the end of this pass with
      // #lit.any.origin, which will conveniently get all references to
      // this. That said, we need to remember the index so we can update
      // the signature.
      ++numRemoved;
      mask.set(idx);
      continue;
    }

    // If we removed any before it, copy this down.
    if (numRemoved)
      paramDecls[idx - numRemoved] = paramDecls[idx];
  }

  // Drop any removed parameters.
  paramDecls.resize(paramDecls.size() - numRemoved);
  return mask;
}

/// Lower `lit.bind_params` to a no-op by replacing it with its generator
/// operand. This is only valid when every input parameter on the generator is
/// a singleton type, since those parameters are removed during lowering and
/// the bound/unbound generator types become identical.
static LogicalResult
lowerBindParamsOp(BindParamsOp op, SingletonTypeHelper &singletonTypeHelper) {
  FnTypeGeneratorType genType = op.getGenerator().getType();
  for (auto [idx, paramType] : llvm::enumerate(genType.getInputParamTypes())) {
    if (singletonTypeHelper.isSingletonType(paramType))
      continue;
    return op.emitError()
           << "may only bind singleton compile-time parameters during "
              "lowering; parameter "
           << idx << " has non-singleton type " << paramType;
  }

  IRRewriter rewriter{OpBuilder(op)};
  rewriter.replaceOp(op, op.getGenerator());
  return success();
}

//===----------------------------------------------------------------------===//
// Op Lowering
//===----------------------------------------------------------------------===//

namespace {
struct LITLowerer {
  LITLowerer(mlir::SymbolTableAnalysis &symbolTables,
             DenseMap<StringAttr, StringAttr> &renamedSymbols,
             SingletonTypeHelper &singletonTypeHelper, StructDecls &structDecls)
      : symbolTables(symbolTables), renamedSymbols(renamedSymbols),
        singletonTypeHelper(singletonTypeHelper), structDecls(structDecls),
        typeType(TypeType::get(
            symbolTables.getTopLevelSymbolTable().getOp()->getContext())) {}

  /// Given a function, check to see if it is a top-level function.  If not,
  /// lower it to a ParamDeclareRegionOp.
  void lowerNestedFunction(FnOp func);
  /// Lower LIT dialect operations in a function body.
  void lowerLITOps(FnOp func, bool &hadErrors);

  /// Lower a function from LIT FnOp to KGEN GeneratorOp.
  /// Caller must handle removal of the original symbol and pass the
  /// pre-calculated mangled name as `newName` beforehand, since different
  /// contexts (top-level vs nested, struct methods) have different
  /// requirements. This function handles symbol table invalidation.
  LogicalResult lowerFunction(FnOp func,
                              ArrayRef<ParamDeclAttr> parentInputParams,
                              Block::iterator mainSymbolTablePosIter,
                              StringAttr newName);

  /// Lower lit.struct.decl and its nested structures.
  LogicalResult lowerStructDecl(StructDeclOp structDecl,
                                Block::iterator mainSymbolTablePosIter);
  /// Lower lit.extension.decl and its nested structures.
  LogicalResult lowerExtensionDecl(ExtensionDeclOp extensionDecl,
                                   Block::iterator mainSymbolTablePosIter);

  /// Lower lit.trait.decl and its nested structures.
  LogicalResult lowerTraitDecl(TraitDeclOp traitDecl,
                               Block::iterator mainSymbolTablePosIter);
  /// Lower the constructs within the body of a module decl.
  /// isTopLevel indicates whether operations in this module body are direct
  /// children of the top-level symbol table (true) or nested within other
  /// operations like FileModuleOp/PackageOp (false). This determines the
  /// removal strategy for operations.
  LogicalResult lowerModuleDecl(Block *moduleBody,
                                Block::iterator mainSymbolTablePosIter,
                                bool isTopLevel);

  /// Recursively process all structs in the module hierarchy.
  LogicalResult lowerAllStructs(Block *moduleBody,
                                Block::iterator mainSymbolTablePosIter,
                                bool isTopLevel);

  /// Recursively process all extensions in the module hierarchy.
  LogicalResult lowerAllExtensions(Block *moduleBody,
                                   Block::iterator mainSymbolTablePosIter,
                                   bool isTopLevel);

  SymbolTable &getTopLevelSymbolTable() {
    return symbolTables.getTopLevelSymbolTable();
  }

  mlir::SymbolTableCollection &getSymbolTableCollection() {
    return symbolTables.getSymbolTables();
  }

  mlir::SymbolTableAnalysis &symbolTables;
  DenseMap<StringAttr, StringAttr> &renamedSymbols;
  SingletonTypeHelper &singletonTypeHelper;
  StructDecls &structDecls;
  TypeType typeType;
  /// For each symbol name (post-rename), the param decls that were dropped.
  DenseMap<StringAttr, ParamDeclDropMask> symbolDroppedParamDecls;
  bool foundAnyPatterns = false;
};
} // namespace

void LITLowerer::lowerLITOps(FnOp func, bool &hadErrors) {
  func.getBodyRegion().walk([&](Operation *op) {
    // Lower any aliases within the function body to param declare.
    IRRewriter b{OpBuilder(op)};
    if (AliasDeclOp alias = dyn_cast<AliasDeclOp>(op)) {
      // Aliases are eagerly substituted for their value, so they are no longer
      // referenced anymore.
      op->erase();
    } else if (isa<OwnershipUseOp, OwnershipMarkInitializedOp,
                   OwnershipMarkDestroyedOp, OwnershipMarkConsumedOp, ImportOp,
                   UnresolvedImportOp, UnresolvedWildcardImportOp>(op)) {
      // lit.ownership.* are used internally by the
      // frontend and ownership lowering, but is not needed after that.
      op->erase();
    } else if (auto loadConsume = dyn_cast<LoadConsumeOp>(op)) {
      b.replaceOpWithNewOp<RefLoadOp>(loadConsume, loadConsume.getRef());
    } else if (auto call = dyn_cast<LIT::CallOp>(op)) {
      if (auto symbolCst = dyn_cast<SymbolConstantAttr>(call.getCallee())) {
        b.replaceOpWithNewOp<KGEN::CallOp>(call, call.getResultTypes(),
                                           symbolCst, call.getOperands(),
                                           call.getTailKindAttr());
      } else {
        b.replaceOpWithNewOp<KGEN::CallParamOp>(
            call, call.getResultTypes(), call.getCallee(), call.getOperands(),
            call.getTailKindAttr());
      }
    } else if (auto call = dyn_cast<LIT::CallIndirectOp>(op)) {
      b.replaceOpWithNewOp<KGEN::CallIndirectOp>(
          call, call.getResultTypes(), call.getCallee(), call.getArguments(),
          call.getTailKindAttr());
    } else if (auto bindParams = dyn_cast<BindParamsOp>(op)) {
      if (failed(lowerBindParamsOp(bindParams, singletonTypeHelper)))
        hadErrors = true;
    } else if (auto call = dyn_cast<LIT::AsyncCallOp>(op)) {
      b.replaceOpWithNewOp<CO::InvokeOp>(call, call.getCallee(),
                                         call.getOperands());
    } else if (auto returnOp = dyn_cast<ErrorReturnOp>(op)) {
      b.replaceOpWithNewOp<KGEN::ReturnOp>(returnOp, returnOp.getResult());
    } else if (auto elifOp = dyn_cast<HLCF::ElifOp>(op)) {
      HLCF::replaceElifWithIfOps(elifOp);
    } else if (auto funcOp = dyn_cast<FnOp>(op)) {
      lowerNestedFunction(funcOp);
    }
  });
}

/// Rename the given symbol operation to its flattened/mangled name, remove it
/// from its current position, and reinsert it at the specified location in the
/// symbol table. Returns the flattened symbol name.
template <typename T>
static StringAttr
flattenNameAndReinsertOp(T op, SymbolTable &symbolTable,
                         Block::iterator mainSymbolTablePosIter) {
  auto mangled = MangledSymbol::mangle(op);
  StringAttr name = mangled.mangled;
  // No mangling occurred.
  if (name == op.getNameAttr())
    return name;

  // Remove the operation in preparation for re-insertion. This gets handled
  // differently depending on if we are already tracking this op in the symbol
  // table.
  if (op->getParentOp() == symbolTable.getOp())
    symbolTable.remove(op);
  else
    op->remove();

  op.setSymbolName(mangled.mangled);
  symbolTable.insert(op, mainSymbolTablePosIter);
  return mangled.mangled;
}

LogicalResult
LITLowerer::lowerFunction(FnOp func, ArrayRef<ParamDeclAttr> parentInputParams,
                          Block::iterator mainSymbolTablePosIter,
                          StringAttr newName) {
  // Caller is responsible for removing `func` since removal patterns differ:
  // - Top-level functions: symbolTable.remove()
  //   (Top-level functions are tracked in the symbol table and must be removed
  //   via symbolTable.remove() to maintain symbol table integrity)
  // - Nested functions: op->remove()
  //   (Nested functions are not tracked in the symbol table and can be removed
  //   directly from their parent operation)
  // Invalidate the symbol table for this FnOp. All FnOps have the SymbolTable
  // trait, so the SymbolTableCollection needs to be notified before erasure.
  getSymbolTableCollection().invalidateSymbolTable(func);

  // If this function has a subprogram attached, update its information to
  // account for the new name.
  if (newName != func.getSymNameAttr()) {
    func.setSymbolName(newName);
    DebugInfo::updateSubprogram(func, newName);
  }

  bool hadErrors = false;
  lowerLITOps(func, hadErrors);
  if (hadErrors)
    return failure();

  FnTypeGeneratorType signature = func.getFuncTypeGenerator();

  // Build the parameter list of the new function, prepending the parameters
  // from the parent decl if present.
  SmallVector<ParamDeclAttr> inputParams;
  if (!parentInputParams.empty()) {
    // Concat the parent and generator input parameter decls.
    llvm::append_range(inputParams, parentInputParams);
    // Offset index references within the current signature to make room.
    // Remap parent input parameter references to indices.
    signature =
        FnTypeGeneratorType::prependParams(signature, parentInputParams);
  }
  llvm::append_range(inputParams, extractImplicitOriginParams(func));

  // Snapshot the source-declared parameter list (names + passing kinds +
  // variadic kinds) before `lowerAttributesAndTypes` strips metadata from
  // the live signature. This snapshot is the source of truth for reflection
  // queries; the live `funcTypeGenerator` and `inputParams` may be rewritten
  // by later transforms such as `RemoveUnusedParams`.
  //
  // Default values and constraints are dropped here because they can hold
  // `ParamIndexRefAttr`s that need a contextual signature; the op's attribute
  // dictionary doesn't establish one, so leaving them in would fail
  // `verify-parameters`. Reflection only needs the structural names today.
  PogListAttr sourceParamList;
  if (PogListAttr fullList = signature.getParamListAttrs()) {
    SmallVector<PogMetadataAttr> strippedPogs;
    strippedPogs.reserve(fullList.getPogs().size());
    for (PogMetadataAttr pog : fullList.getPogs()) {
      strippedPogs.push_back(PogMetadataAttr::get(
          pog.getName(), pog.getPassingKind(), pog.getVariadic()));
    }
    sourceParamList = PogListAttr::get(func->getContext(), strippedPogs,
                                       /*bodyConstraints=*/{},
                                       fullList.getOrigVariadicConvention());
  }

  // Now that we have the full parameter list, remove any singleton parameters.
  // This ensures that the elaborator doesn't instantiate the function based on
  // lifetimes.
  ParamDeclDropMask droppedParams =
      removeSingletonParamDecls(singletonTypeHelper, inputParams);
  if (droppedParams.any())
    symbolDroppedParamDecls[func.getSymNameAttr()] = droppedParams;

  OpBuilder b(func->getContext());
  auto inputParamsArr = ParamDeclArrayAttr::get(b.getContext(), inputParams);
  auto sigAttr = TypeAttr::get(signature);

  // Snapshot the full source signature into `sourceFuncTypeGenerator`, wrapped
  // in a `TypeParamAttr` so `LowerLITTypes` lowers it in the value domain (the
  // snapshot's argument/result types come out in the value domain, which
  // reflection clients need). `lowerAttributesAndTypes` strips its metadata
  // like the live signature's, but it is left untouched by later transforms
  // (e.g. `RemoveUnusedParams`) that rewrite the live one.
  TypedAttr sourceFuncTypeGen = TypeParamAttr::get(signature, typeType);

  // Directly lower since these operations are exactly identical right now.
  OperationState state(func.getLoc(), GeneratorOp::getOperationName());
  GeneratorOp::build(b, state, func.getSymNameAttr(),
                     /*sym_visibility=*/nullptr, func.getSourceNameAttr(),
                     sigAttr, func.getFunctionTypeAttr(), inputParamsArr,
                     func.getDecoratorsAttr(), func.getInlineLevelAttr(),
                     func.getExportKindAttr(), func.getExternalAttr(),
                     /*inlinedForm=*/nullptr, func.getLinkageNameAttr(),
                     func.getLLVMMetadataArray(),
                     func.getLLVMArgMetadataArray(), sourceParamList,
                     sourceFuncTypeGen);

  for (const NamedAttribute &attr : func->getDialectAttrs())
    state.attributes.push_back(attr);

  auto newFunc = cast<GeneratorOp>(b.create(state));

  // Move over the body.
  newFunc.getBodyRegion().takeBody(func.getBodyRegion());

  // Insert the lowered GeneratorOp and cleanup the original FnOp.
  // Caller should have already removed the LIT function from its parent.
  getTopLevelSymbolTable().insert(newFunc, mainSymbolTablePosIter);
  func.erase();
  return success();
}

void LITLowerer::lowerNestedFunction(FnOp func) {
  // Process a nested function by lowering it straight to a
  // `kgen.param.declare.region`. Nested functions are denoted with an
  // parameter declaration on the function declaration.
  ParamDeclAttr decl = func.getParamDeclAttr();
  assert(decl && "expected nested function to declare a parameter");

  ImplicitLocOpBuilder b(func.getLoc(), func);

  // The new param.declare.region will drop implicit lifetimes.
  SmallVector<ParamDeclAttr> inputParams;
  llvm::append_range(inputParams, extractImplicitOriginParams(func));
  removeSingletonParamDecls(singletonTypeHelper, inputParams);

  StringAttr sourceName = func.getSourceNameAttr();
  if (!sourceName)
    sourceName = decl.getName();
  auto region = ParamDeclareRegionOp::create(
      b, /*sym_name=*/nullptr, /*sym_visibility=*/nullptr, decl, sourceName,
      func.getFuncTypeGenerator(), func.getFunctionType(), inputParams,
      func.getInlineLevel(), func.getLinkageNameAttr(),
      func.getLLVMMetadataArray(), func.getLLVMArgMetadataArray());
  region.getBodyRegion().takeBody(func.getBodyRegion());
  func.erase();
}

LogicalResult
LITLowerer::lowerStructDecl(StructDeclOp structDecl,
                            Block::iterator mainSymbolTablePosIter) {
  // Update the name of this struct, incorporating any parents.
  StringAttr structName = flattenNameAndReinsertOp(
      structDecl, getTopLevelSymbolTable(), mainSymbolTablePosIter);

  // Build a StructGeneratorOp as its replacement.
  StructDecl info{};
  info.sourceName = structDecl.getSourceNameAttr();
  info.decls = structDecl.getParamsAttr();

  // Build the isMemoryOnly attribute. For unconditional RP, this is a simple
  // BoolAttr. For conditional RP, build a parametric expression that negates
  // the RP constraint proposition.
  auto *ctx = structDecl.getContext();
  if (structDecl.isRegisterPassable()) {
    info.isMemoryOnlyAttr = BoolAttr::get(ctx, false);
  } else if (auto rpConstraint =
                 structDecl.getRegisterPassableConstraintAttr()) {
    info.isMemoryOnlyAttr = ParamOperatorAttr::getNot(CastToBuiltinAttr::get(
        rpConstraint.getProposition(), IntegerType::get(ctx, 1)));
  } else {
    info.isMemoryOnlyAttr = BoolAttr::get(ctx, true);
  }

  // Provide default alignment of 1 if not explicitly specified.
  if (auto minAlign = structDecl.getMinAlignmentAttr())
    info.minAlignment = minAlign;
  else
    info.minAlignment =
        IntegerAttr::get(IndexType::get(structDecl.getContext()), 1);
  info.loc = structDecl.getLoc();

  // Collect the struct fields.
  SmallVector<StructDefFieldAttr> fieldDecls;
  for (auto [idx, field] : llvm::enumerate(structDecl.getFieldDecls())) {
    info.fields.emplace_back(field.getNameAttr(), field.getType());
    structDecls.fieldIndices.try_emplace({structName, field.getNameAttr()},
                                         idx);
    TypedAttr fieldTypeValue = TypeParamAttr::get(field.getType(), typeType);
    fieldDecls.push_back(
        StructDefFieldAttr::get(field.getNameAttr(), fieldTypeValue));
  }

  // Create struct-generator.
  SmallVector<StringAttr> paramNames;
  SmallVector<Type> paramTypes;
  SmallVector<TypedAttr> paramValues;
  for (ParamDeclAttr decl : info.decls) {
    paramNames.push_back(decl.getName());
    paramTypes.push_back(decl.getType());
    paramValues.push_back(ParamDeclRefAttr::get(decl));
  }

  auto structInstType = StructInstanceType::get(
      structName, paramNames, paramValues, fieldDecls, info.isMemoryOnlyAttr);

  OpBuilder b(structDecl->getContext());
  auto structGen = StructGeneratorOp::create(
      b, info.loc, structName, /*sym_visibility=*/nullptr, info.decls,
      structInstType, typeType);
  Block *structGenBody = b.createBlock(&structGen.getRegion());

  for (Operation &member : llvm::make_early_inc_range(
           structDecl.getFields().front().getOperations())) {
    if (isa<StructFieldOp>(member))
      continue; // Already lowered field.
    if (isa<AliasDeclOp>(member)) {
      member.erase();
      continue;
    }
    if (auto conformance = dyn_cast<ConformanceOp>(member)) {
      conformance->moveBefore(structGenBody, structGenBody->end());
      continue;
    }

    auto func = dyn_cast<FnOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");

    // Calculate new name, mangled if not top level. Must be before
    // removal since MangledSymbol::mangle crawls up the ancestors.
    StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
    // This is out here because removal is different for each
    // lowerFunction caller.
    func->remove();
    if (failed(lowerFunction(func, structDecl.getInputParams(),
                             mainSymbolTablePosIter, nameToUse)))
      return failure();
  }

  getTopLevelSymbolTable().remove(structDecl);
  info.symRef = SymbolRefAttr::get(
      getTopLevelSymbolTable().insert(structGen, mainSymbolTablePosIter));
  getSymbolTableCollection().invalidateSymbolTable(structDecl);
  structDecl.erase();
  structDecls.structDecls.try_emplace(structName, std::move(info));
  return success();
}

LogicalResult
LITLowerer::lowerExtensionDecl(ExtensionDeclOp extensionDecl,
                               Block::iterator mainSymbolTablePosIter) {
  SymbolRefAttr targetStructRef = extensionDecl.getTargetStruct().value();

  // Flatten the symbol reference to get the proper name for lookup
  StringAttr structName = flattenSymbolRefAttr(targetStructRef).getAttr();

  StructDecl &targetStructDeclInfo = structDecls.get(structName);

  Operation *kgenOp = getTopLevelSymbolTable().lookupSymbolIn(
      getTopLevelSymbolTable().getOp(), targetStructDeclInfo.symRef);
  if (!kgenOp) {
    return extensionDecl.emitError("cannot find extension target struct");
  }

  StructGeneratorOp kgenStructGenOp = dyn_cast<StructGeneratorOp>(kgenOp);
  if (!kgenStructGenOp) {
    return extensionDecl.emitError("extension target is not a struct");
  }

  for (Operation &member : llvm::make_early_inc_range(
           extensionDecl.getFields().front().getOperations())) {
    assert(!isa<StructFieldOp>(member) && "Extensions can't have fields");
    if (isa<AliasDeclOp>(member)) {
      member.erase();
      continue;
    }
    if (auto conformance = dyn_cast<ConformanceOp>(member)) {
      Block *structGenBody = &kgenStructGenOp.getRegion().front();
      // Extension conformances need to be moved to the target struct's
      // generator, because that's what the elaborator expects.
      conformance->moveBefore(structGenBody, structGenBody->end());
      continue;
    }

    auto func = dyn_cast<FnOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");

    ArrayRef<ParamDeclAttr> inputParams = targetStructDeclInfo.decls;
    StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
    func->remove();
    if (failed(lowerFunction(func, inputParams, mainSymbolTablePosIter,
                             nameToUse)))
      return failure();
  }
  // Invalidate symbol table before erasing to maintain consistency.
  getSymbolTableCollection().invalidateSymbolTable(extensionDecl);
  // Remove from symbol table if present, otherwise erase directly.
  // Note: Unlike StructDecl, we don't use flattenNameAndReinsertOp since we're
  // just erasing, not moving it first.
  // TODO(MOCO-522): Either move this first too, or change that about
  // structs/traits.
  if (getTopLevelSymbolTable().lookup(extensionDecl.getSymNameAttr())) {
    getTopLevelSymbolTable().erase(extensionDecl);
  } else {
    extensionDecl.erase();
  }
  return success();
}

LogicalResult
LITLowerer::lowerTraitDecl(TraitDeclOp traitDecl,
                           Block::iterator mainSymbolTablePosIter) {
  // Update the name of this trait, incorporating any parents.
  flattenNameAndReinsertOp(traitDecl, getTopLevelSymbolTable(),
                           mainSymbolTablePosIter);

  // Process operations within the trait body.
  for (Operation &member : llvm::make_early_inc_range(
           traitDecl.getFields().front().getOperations())) {
    if (auto func = dyn_cast<FnOp>(member)) {
      // Check if the function has a non-empty body (more than just
      // kgen.unreachable).
      Block *funcBody = func.getBody();
      bool hasEmptyBody = funcBody->getOperations().size() == 1 &&
                          isa<KGEN::UnreachableOp>(funcBody->front());

      if (!hasEmptyBody) {
        // Calculate new name, mangled if not top level. Must be before
        // removal since MangledSymbol::mangle crawls up the ancestors.
        StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
        // This is out here because removal is different for each
        // lowerFunction caller.
        func->remove();
        if (failed(lowerFunction(func, traitDecl.getInputParams(),
                                 mainSymbolTablePosIter, nameToUse)))
          return failure();
      }
    }
    // We don't care about other operations in the trait body for now.
  }

  getTopLevelSymbolTable().erase(traitDecl);
  // invalidateSymbolTable since we're removing from the top-level
  // symbol table.
  getSymbolTableCollection().invalidateSymbolTable(traitDecl);
  return success();
}

LogicalResult
LITLowerer::lowerAllStructs(Block *moduleBody,
                            Block::iterator mainSymbolTablePosIter,
                            bool isTopLevel) {

  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    if (auto structDecl = dyn_cast<StructDeclOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          mainSymbolTablePosIter == Block::iterator() ? op.getIterator()
                                                      : mainSymbolTablePosIter;
      if (failed(lowerStructDecl(structDecl, childMainSymbolTablePos)))
        return failure();
    } else if (auto fileModule = dyn_cast<LIT::FileModuleOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          mainSymbolTablePosIter == Block::iterator() ? op.getIterator()
                                                      : mainSymbolTablePosIter;
      if (failed(lowerAllStructs(fileModule.getBody(), childMainSymbolTablePos,
                                 /*isTopLevel=*/false)))
        return failure();
    } else if (auto package = dyn_cast<LIT::PackageOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          mainSymbolTablePosIter == Block::iterator() ? op.getIterator()
                                                      : mainSymbolTablePosIter;
      if (failed(lowerAllStructs(package.getBody(), childMainSymbolTablePos,
                                 /*isTopLevel=*/false)))
        return failure();
    }
  }
  return success();
}

LogicalResult
LITLowerer::lowerAllExtensions(Block *moduleBody,
                               Block::iterator mainSymbolTablePosIter,
                               bool isTopLevel) {
  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    if (auto extensionDecl = dyn_cast<ExtensionDeclOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      auto extensionPos =
          isTopLevel ? op.getIterator() : mainSymbolTablePosIter;
      if (failed(lowerExtensionDecl(extensionDecl, extensionPos)))
        return failure();
    } else if (auto fileModule = dyn_cast<LIT::FileModuleOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          isTopLevel ? op.getIterator() : mainSymbolTablePosIter;
      if (failed(lowerAllExtensions(fileModule.getBody(),
                                    childMainSymbolTablePos,
                                    /*isTopLevel=*/false)))
        return failure();
    } else if (auto package = dyn_cast<LIT::PackageOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          isTopLevel ? op.getIterator() : mainSymbolTablePosIter;
      if (failed(lowerAllExtensions(package.getBody(), childMainSymbolTablePos,
                                    /*isTopLevel=*/false)))
        return failure();
    }
  }
  return success();
}

LogicalResult
LITLowerer::lowerModuleDecl(Block *moduleBody,
                            Block::iterator mainSymbolTablePosIter,
                            bool isTopLevel) {
  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    // If we are already in the symbol table, use the the operations iterator.
    auto opSymTableIt = isTopLevel ? op.getIterator() : mainSymbolTablePosIter;

    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(&op)
            .Case([&](LIT::FnOp op) {
              // Calculate new name, mangled if not top level. Must be before
              // removal since MangledSymbol::mangle crawls up the ancestors.
              StringAttr nameToUse = !isTopLevel
                                         ? MangledSymbol::mangle(op).mangled
                                         : op.getSymNameAttr();
              // Function removal is handled at call site because top-level and
              // nested functions require different removal strategies.
              if (isTopLevel)
                getTopLevelSymbolTable().remove(op);
              else
                op->remove();
              return lowerFunction(op, {}, opSymTableIt, nameToUse);
            })
            .Case([&](StructDeclOp op) {
              // Structs should have been processed earlier by lowerAllStructs.
              assert(false && "Structs should have been lowered already");
              return failure();
            })
            .Case([&](ExtensionDeclOp op) {
              // Extensions should have been processed earlier by
              // lowerAllExtensions.
              assert(false && "Extensions should have been lowered already");
              return failure();
            })
            .Case([&](TraitDeclOp op) {
              return lowerTraitDecl(op, opSymTableIt);
            })
            .Case<LIT::FileModuleOp, LIT::PackageOp>([&](auto op) {
              // Make sure to remove the op from the symbol table if needed.
              if (op->getParentOp() == getTopLevelSymbolTable().getOp())
                getTopLevelSymbolTable().remove(op);

              // Lower the constructs within the body.
              Block *fileBody = op.getBody();
              if (failed(lowerModuleDecl(fileBody, opSymTableIt,
                                         /*isTopLevel=*/false)))
                return failure();

              // Inline the remaining body of the file into the parent.
              op->getBlock()->getOperations().splice(
                  op->getIterator(), fileBody->getOperations(),
                  fileBody->begin(), fileBody->end());

              // invalidateSymbolTable since we're removing from the top-level
              // symbol table.
              getSymbolTableCollection().invalidateSymbolTable(op);
              op->erase();
              return mlir::success();
            })
            .Case<AliasDeclOp, ImportOp, UnresolvedImportOp,
                  UnresolvedWildcardImportOp>([&](auto op) {
              op->erase();
              return mlir::success();
            })
            .Case([&](mlir::SymbolOpInterface symbol) {
              flattenNameAndReinsertOp(symbol, getTopLevelSymbolTable(),
                                       opSymTableIt);
              return mlir::success();
            })
            .Default(mlir::success());
    if (failed(result))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type lowering
//===----------------------------------------------------------------------===//

/// Check to see if any of the parameters of the specified signature are
/// singletons like origin parameters.  If so, bind them to a dummy value and
/// return the updated signature without them.
template <typename GenKind>
static std::conditional_t<std::is_base_of_v<Type, GenKind>, Type, Attribute>
removeSingletonParams(SingletonTypeHelper &singletonTypeHelper,
                      GenKind generator) {
  SmallVector<TypedAttr> paramsToBind;
  bool hasRemovals = false;

  for (Type paramType : generator.getInputParamTypes()) {
    if (singletonTypeHelper.isSingletonType(paramType)) {
      // Bind singleton parameters to their canonical value.
      paramsToBind.push_back(singletonTypeHelper.getSingletonValue(paramType));
      hasRemovals = true;
    } else {
      // Keep non-singleton parameters unbound.
      paramsToBind.push_back(UnboundAttr::get(paramType));
    }
  }

  // Update the generator if we dropped anything.
  if (hasRemovals) {
    generator = getSpecializedWithConcreteBindings(generator, paramsToBind);
    assert(generator && "didn't replace singletons correctly");
    if (generator.isFullyBound()) {
      // By back-compat, we never eliminate the empty generator type wrapper on
      // func types. This should eventually be made consistent with other types.
      // This follows the same pattern as BindParamsAttr.
      if constexpr (std::is_base_of_v<Type, GenKind>) {
        if (!isa<FuncType>(generator.getBody()))
          return generator.getBody();
      } else {
        if (!isa<FuncType>(generator.getBody().getType()))
          return generator.getBody();
      }
    }
  }
  return generator;
}

template <typename GenKind>
std::pair<std::conditional_t<std::is_base_of_v<Type, GenKind>, Type, Attribute>,
          WalkResult>
replaceGeneratorAttrType(SingletonTypeHelper &singletonTypeHelper,
                         mlir::AttrTypeReplacer &replacer, GenKind gen) {
  // Remove uses of any singleton attributes.
  SmallVector<Type> paramTypes;
  for (auto ty : gen.getInputParamTypes())
    paramTypes.push_back(replacer.replace(ty));

  // Remove metadata & remove singleton input param decls.
  auto newBody = cast<decltype(gen.getBody())>(replacer.replace(gen.getBody()));
  gen = GenKind::get(paramTypes, newBody);
  auto result = removeSingletonParams(singletonTypeHelper, gen);
  return std::make_pair(result, WalkResult::skip());
}

static LogicalResult lowerAttributesAndTypes(
    Operation *op, const DenseMap<StringAttr, StringAttr> &renamedSymbols,
    SingletonTypeHelper &singletonTypeHelper,
    DenseMap<StringAttr, ParamDeclDropMask> &symbolDroppedParamDecls,
    StructDecls &structDecls) {

  bool hadErrors = false;
  // This is the location of the current op that we're working on, updated as
  // we traverse the Module hierarchy.
  Location curOpLoc = op->getLoc();

  mlir::AttrTypeReplacer replacer;

  // OriginEqAttr may not survive to lowering.
  replacer.addReplacement(
      [&](OriginEqAttr attr)
          -> std::optional<std::pair<TypedAttr, WalkResult>> {
        mlir::emitError(curOpLoc)
            << "origin equality may only be tested in 'where' clauses";
        hadErrors = true;
        return std::make_pair(
            IntegerAttr::get(IntegerType::get(op->getContext(), 1), 0),
            WalkResult::skip());
      });

  // Member functions are reference with nested symbol references. After
  // lowering, the symbol tree will be flat. Concatenate all nested symbol
  // references in symbol constants. If something was renamed, perform the
  // renaming.
  replacer.addReplacement([&renamedSymbols](SymbolRefAttr ref) {
    auto flat = flattenSymbolRefAttr(ref);
    if (StringAttr renamed = renamedSymbols.lookup(flat.getAttr()))
      return SymbolRefAttr::get(renamed);
    return flat;
  });

  // Remove signature metadata.
  replacer.addReplacement([&](FuncType sig) {
    return std::make_pair(
        FuncType::get(cast<FunctionType>(replacer.replace(sig.getValues())),
                      sig.getArgConventions(), sig.getFnEffects()),
        WalkResult::skip());
  });

  replacer.addReplacement([&](GeneratorType gen) {
    // Remove uses of any singleton attributes.
    SmallVector<Type> paramTypes;
    for (auto ty : gen.getInputParamTypes())
      paramTypes.push_back(replacer.replace(ty));

    // Remove metadata & remove singleton input param decls.
    gen = GeneratorType::get(paramTypes, replacer.replace(gen.getBody()));
    auto result = removeSingletonParams(singletonTypeHelper, gen);
    return std::make_pair(result, WalkResult::skip());
  });

  replacer.addReplacement([&](GeneratorAttr gen) {
    return replaceGeneratorAttrType(singletonTypeHelper, replacer, gen);
  });

  replacer.addReplacement(
      [&](BindParamsAttr attr)
          -> std::optional<std::pair<TypedAttr, WalkResult>> {
        DenseBoolArrayAttr discharged = attr.getDischarged();
        if (!discharged || discharged.empty())
          return std::nullopt;

        // Remove discharge mask since it goes together with the generator
        // metadata's body constraints.
        TypedAttr generator =
            cast<TypedAttr>(replacer.replace(attr.getGenerator()));
        SmallVector<TypedAttr> paramValues;
        for (TypedAttr value : attr.getParamValues())
          paramValues.push_back(cast<TypedAttr>(replacer.replace(value)));
        return std::make_pair(
            BindParamsAttr::get(generator.getContext(), generator, paramValues,
                                /*evaluationContext=*/nullptr),
            WalkResult::skip());
      });

  replacer.addReplacement([&](StructInstanceType structInstType) {
    auto it = symbolDroppedParamDecls.find(structInstType.getName());
    if (it == symbolDroppedParamDecls.end() ||
        it->second.size() != structInstType.getParamValues().size())
      return std::make_pair(Type(structInstType), WalkResult::advance());

    SmallVector<StringAttr> remainingParamNames;
    SmallVector<TypedAttr> remainingParamValues;
    for (auto [idx, nameAndValue] :
         llvm::enumerate(llvm::zip(structInstType.getParamNames(),
                                   structInstType.getParamValues()))) {
      if (it->second[idx])
        continue;
      auto [name, value] = nameAndValue;
      remainingParamNames.push_back(name);
      remainingParamValues.push_back(cast<TypedAttr>(replacer.replace(value)));
    }

    SmallVector<StructDefFieldAttr> fields;
    fields.reserve(structInstType.getFields().size());
    for (StructDefFieldAttr field : structInstType.getFields()) {
      fields.push_back(StructDefFieldAttr::get(
          field.getName(),
          cast<TypedAttr>(replacer.replace(field.getTypeValue()))));
    }

    return std::make_pair(
        Type(StructInstanceType::get(structInstType.getName(),
                                     remainingParamNames, remainingParamValues,
                                     fields, structInstType.getIsMemoryOnly())),
        WalkResult::skip());
  });

  // Sugar attr is turned into canonical form.
  replacer.addReplacement(
      [&](SugarAttr sugar) { return replacer.replace(sugar.getCanonical()); });

  auto *debugInfoDialect =
      op->getContext()->getLoadedDialect<DebugInfo::DebugInfoDialect>();

  auto removeSingletonParams = [&](auto attr) -> decltype(attr) {
    SymbolRefAttr flatRef =
        cast<SymbolRefAttr>(replacer.replace(attr.getSymbol()));
    // Check the name & the number of params to ensure we don't operate on
    // SymbolConstantAttrs/FuncSymbolAttr that have already been processed.
    if (auto it = symbolDroppedParamDecls.find(flatRef.getLeafReference());
        it != symbolDroppedParamDecls.end() &&
        it->second.size() == attr.getParamValues().size()) {
      SmallVector<TypedAttr> remainingParams;
      for (auto [idx, value] : llvm::enumerate(attr.getParamValues()))
        if (!it->second[idx])
          remainingParams.push_back(cast<TypedAttr>(replacer.replace(value)));
      return decltype(attr)::get(
          flatRef,
          cast<decltype(attr.getType())>(replacer.replace(attr.getType())),
          remainingParams);
    }
    return nullptr;
  };

  replacer.addReplacement(
      [&](TypedAttr attr) -> std::optional<std::pair<TypedAttr, WalkResult>> {
        if (&attr.getDialect() == debugInfoDialect)
          return std::nullopt;

        // Canonicalize all values of singleton types.
        if (TypedAttr value =
                singletonTypeHelper.getSingletonValue(attr.getType()))
          return std::make_pair(value, WalkResult::advance());

        // Remove singleton parameter values from SymbolConstantAttr.
        if (auto symCst = dyn_cast<SymbolConstantAttr>(attr))
          if (auto newSymCst = removeSingletonParams(symCst))
            return std::make_pair(newSymCst, WalkResult::skip());

        // Remove singleton parameter values from FuncSymbolAttr.
        if (auto funcSym = dyn_cast<FuncSymbolAttr>(attr))
          if (auto newFuncSym = removeSingletonParams(funcSym))
            return std::make_pair(newFuncSym, WalkResult::skip());

        // Remove singleton parameter values from TypeGeneratorRefAttr.
        if (auto genRef = dyn_cast<TypeGeneratorRefAttr>(attr)) {
          SymbolRefAttr flatRef =
              cast<SymbolRefAttr>(replacer.replace(genRef.getSymbol()));
          if (auto it =
                  symbolDroppedParamDecls.find(flatRef.getLeafReference());
              it != symbolDroppedParamDecls.end() &&
              it->second.size() == genRef.getParamValues().size()) {
            SmallVector<TypedAttr> remainingParams;
            for (auto [idx, value] : llvm::enumerate(genRef.getParamValues()))
              if (!it->second[idx])
                remainingParams.push_back(
                    cast<TypedAttr>(replacer.replace(value)));
            return std::make_pair(
                TypeGeneratorRefAttr::get(attr.getContext(), flatRef,
                                          remainingParams, genRef.getType()),
                WalkResult::skip());
          }
        }

        // Remove singleton parameter values from BindParamsAttr.
        if (auto bindParams = dyn_cast<BindParamsAttr>(attr)) {
          SmallVector<TypedAttr> newOperands;
          for (auto [declType, param] : llvm::zip(
                   sugarCast<GeneratorType>(bindParams.getGenerator().getType())
                       .getInputParamTypes(),
                   bindParams.getParamValues())) {
            // Check for singleton type using the declared type on the
            // signature, instead of the concrete type of the param. This
            // prevents parametrically-singleton types from getting erased (only
            // always singleton params can be removed in general).
            if (!singletonTypeHelper.isSingletonType(
                    replacer.replace(declType)))
              newOperands.push_back(cast<TypedAttr>(replacer.replace(param)));
          }
          if (newOperands.size() != bindParams.getParamValues().size()) {
            TypedAttr generator =
                cast<TypedAttr>(replacer.replace(bindParams.getGenerator()));
            return std::make_pair(
                BindParamsAttr::get(generator.getContext(), generator,
                                    newOperands,
                                    /*evaluationContext=*/nullptr),
                WalkResult::skip());
          }
        }

        return std::nullopt;
      });

  // Walk the entire Module updating everything.
  op->walk([&](Operation *nestedOp) {
    curOpLoc = nestedOp->getLoc();
    replacer.replaceElementsIn(nestedOp, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
  });

  // Update saved types in struct decls.
  for (auto &decl : structDecls.structDecls) {
    decl.second.decls =
        cast<ParamDeclArrayAttr>(replacer.replace(decl.second.decls));
    for (auto &field : decl.second.fields) {
      field.second = replacer.replace(field.second);
    }
    decl.second.minAlignment =
        cast<TypedAttr>(replacer.replace(decl.second.minAlignment));
    decl.second.isMemoryOnlyAttr =
        cast<TypedAttr>(replacer.replace(decl.second.isMemoryOnlyAttr));
  }

  return failure(hadErrors);
}

//===----------------------------------------------------------------------===//
// Pass boilerplate.
//===----------------------------------------------------------------------===//

namespace {
struct LowerLITPass : public KGEN::impl::LowerLITBase<LowerLITPass> {
  using LowerLITBase::LowerLITBase;

  void runOnOperation() override {
    // TODO: This has to be a module pass because this mutates the body of
    // the module, but we could trivially parallelize this within the pass.
    ModuleOp module = getOperation();
    auto &symtab = getAnalysis<mlir::SymbolTableAnalysis>();
    StructDecls structDecls;

    {
      DenseMap<StringAttr, StringAttr> renamedSymbols;
      SingletonTypeHelper singletonTypeHelper(
          module, symtab.getTopLevelSymbolTable(), structDecls);
      LITLowerer lowerer(symtab, renamedSymbols, singletonTypeHelper,
                         structDecls);

      // Lower all structs first, so that extensions can find them when they
      // need to look up struct info.
      if (failed(lowerer.lowerAllStructs(module.getBody(), Block::iterator(),
                                         /*isTopLevel=*/true)))
        return signalPassFailure();

      // Lower all extensions now that the structs' info exists.
      if (failed(lowerer.lowerAllExtensions(module.getBody(), Block::iterator(),
                                            /*isTopLevel=*/true)))
        return signalPassFailure();

      // Now lower away everything else including modules etc.
      if (failed(lowerer.lowerModuleDecl(module.getBody(), Block::iterator(),
                                         /*isTopLevel=*/true)) ||
          failed(lowerAttributesAndTypes(
              module, renamedSymbols, singletonTypeHelper,
              lowerer.symbolDroppedParamDecls, structDecls)))
        return signalPassFailure();
    }

    // Keep lowering all the operations and types.
    mlir::LockedSymbolTableCollection lockedSymtab(symtab.getSymbolTables());
    if (failed(LIT::lowerLITTypes(module, structDecls, lockedSymtab)))
      signalPassFailure();
  }
};

} // namespace
