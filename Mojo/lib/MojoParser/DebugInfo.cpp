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

#include "DebugInfo.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"

using namespace M;
using namespace KGEN;
using namespace LIT;
using DebugInfo::SourceNameAttr;

void SourceNames::processDecorators(Operation *op,
                                    SmallVectorImpl<SourceNameAttr> &out) {
  auto kgenDecoratorToFunctionName =
      [&](TypedAttr opDecorator) -> SourceNameAttr {
    if (auto sym = dyn_cast_or_null<SymbolConstantAttr>(opDecorator)) {
      ASTDecl *decoratorDecl =
          shared.declResolver->getDeclForFuncSymbol(sym.getSymbol());
      if (decoratorDecl)
        if (auto decoratorFunc =
                dyn_cast_or_null<FnOp>(decoratorDecl->getIfOperation()))
          return getSourceName(decoratorFunc);
    }
    return {};
  };
  auto gatherDecorators = [&](ArrayRef<TypedAttr> opDecorators) {
    for (TypedAttr opDecorator : opDecorators) {
      if (SourceNameAttr name = kgenDecoratorToFunctionName(opDecorator))
        out.push_back(name);
    }
  };
  if (auto hasDecls = dyn_cast<StructDeclOp>(op))
    gatherDecorators(hasDecls.getDecorators());
  else if (auto hasDecls = dyn_cast<FnOp>(op))
    gatherDecorators(hasDecls.getDecorators());
}

SourceNameAttr SourceNames::getSourceName(mlir::SymbolOpInterface op) {
  // Try to find an already computed name.
  if (auto it = names.find(op); it != names.end())
    return it->second;

  StringAttr name;
  SmallVector<SourceNameAttr> paramTypes, argTypes;
  SourceNameAttr parent;

  DebugInfo::SourceNameKind kind = {};
  if (auto package = dyn_cast<PackageOp>(*op)) {
    // Query the source name. Fall back to the symbol name otherwise.
    name = package.getSymNameAttr();
    kind = DebugInfo::SourceNameKind::Package;
  } else if (auto fileModule = dyn_cast<FileModuleOp>(*op)) {
    // Query the source name. Fall back to the symbol name otherwise.
    name = fileModule.getSymNameAttr();
    kind = DebugInfo::SourceNameKind::Module;
  } else if (auto structOp = dyn_cast<StructDeclOp>(*op)) {
    // The symbol name is the source name.
    name = structOp.getSymNameAttr();
    // Bundle the source names of the parameter types.
    for (Type type : structOp.getSignature().getParamTypes())
      paramTypes.push_back(getSourceName(getCanonicalType(type)));
    kind = DebugInfo::SourceNameKind::Struct;
  } else if (auto func = dyn_cast<FnOp>(*op)) {
    // Query the source name. Fall back to the symbol name otherwise.
    name = func.getSourceNameAttr();
    if (!name)
      name = func.getSymNameAttr();
    // Bundle the source names of the argument and parameter types. Don't
    // include the memory-only result slot if it's there.
    FnTypeGeneratorType sig = func.getFuncTypeGenerator();
    for (Type type : sig.getInputParamTypes())
      paramTypes.push_back(getSourceName(getCanonicalType(type)));
    for (auto [i, t, conv] :
         llvm::enumerate(sig.getArguments(), sig.getArgConventions())) {
      if (isResultSlot(conv))
        continue;
      ASTType type = RefType::stripRefConvention(t, conv);

      // Unwrap variadics pointers if necessary.
      if (sig.isPosVarArg(i))
        type = type.getVariadicListInfo().elementType;
      argTypes.push_back(getSourceName(getCanonicalType(type)));
    }
    kind = DebugInfo::SourceNameKind::Fn;
    // The function will not have parameter values until elaboration.
  } else {
    // If we somehow end up here, just use the symbol name.
    name = op.getNameAttr();
  }

  if (auto parentOp = op->getParentOfType<mlir::SymbolOpInterface>())
    parent = getSourceName(parentOp);

  SmallVector<SourceNameAttr> decorators;
  processDecorators(op.getOperation(), decorators);

  auto sourceName =
      SourceNameAttr::get(name, paramTypes, argTypes,
                          /*paramValues=*/{}, parent, kind, decorators);
  names.try_emplace(op, sourceName);
  return sourceName;
}

SourceNameAttr SourceNames::getSourceName(Type type) {
  type = getCanonicalType(type);
  // If this is a reference to a source type, then we can use its full source
  // name.
  if (auto declRef = dyn_cast<StructType>(type)) {
    ASTDecl &decl =
        shared.declResolver->getDeclForTypeSymbol(declRef.getSymbol());
    StructDeclOp op = cast<StructDeclOp>(decl.getIfOperation());
    SourceNameAttr name = getSourceName(op);
    // Add the parameter values.
    SmallVector<StringAttr> paramValues;
    for (TypedAttr value : declRef.getParamValues())
      paramValues.push_back(getParamTypeAsString(getCanonicalAttr(value)));
    SmallVector<SourceNameAttr> decorators;
    processDecorators(op.getOperation(), decorators);
    return SourceNameAttr::get(
        name.getName(), name.getParamTypes(), name.getArgTypes(), paramValues,
        name.getParent(), DebugInfo::SourceNameKind::Struct, decorators);
  }
  // For anything else, use the full MLIR type.
  return SourceNameAttr::get(getTypeAsString(type));
}

void SharedState::setLocationDebugScope(FnOp funcOp) {
  if (!diBuilder) {
    // Ensure no debug scope for function.
    if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(
            funcOp->getLoc()))
      funcOp->setLoc(fusedLoc.getLocations().front());
    return;
  }

  FileLineColLoc fileLineCol =
      funcOp.getLoc()->findInstanceOf<FileLineColLoc>();

  // Synthesized functions do not correspond to any source code, so we do not
  // want to generate debug info and step into them. The same applies to
  // no-debug functions.
  if (funcOp.isSynthetic() ||
      funcOp.getInlineLevel() == InlineLevel::AlwaysNoDebug ||
      funcOp.getInlineLevel() == InlineLevel::AlwaysBuiltin) {
    funcOp->setLoc(fileLineCol);
    return;
  }

  // Compute the subprogram flags.
  /// If we have any optimizations, mark the subprogram as optimized.
  DebugInfo::SubprogramFlags spFlags =
      options.optimizationLevel ? DebugInfo::SubprogramFlags::Optimized
                                : DebugInfo::SubprogramFlags::None;
  /// If the function has a body, treat it as a definition.
  if (!funcOp.isExternal())
    spFlags = spFlags | DebugInfo::SubprogramFlags::Definition;

  // Use unresolved types now for simplicity, these will get resolved during
  // compilation.
  auto mapUnresolvedType = [](Type type) -> DebugInfo::DIType {
    return DebugInfo::DIUnresolvedMLIRType::get(getCanonicalType(type));
  };

  auto type = DebugInfo::DISubroutineType::get(
      funcOp.getContext(),
      llvm::map_to_vector(funcOp.getArgumentTypes(), mapUnresolvedType),
      llvm::map_to_vector(funcOp.getResultTypes(), mapUnresolvedType));

  // The linkage name is derived either from the symbol name or the param decl
  // name, if it's a nested function.
  StringAttr linkageName = funcOp.getSymNameAttr();
  if (!linkageName)
    linkageName = funcOp.getParamDeclAttr().getName();

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard = diBuilder->pushSubprogram(
      getSourceName(funcOp), linkageName, diBuilder->createFile(fileLineCol),
      fileLineCol.getLine(), fileLineCol.getLine(), spFlags, type);
  funcOp->setLoc(diBuilder->createScopedLoc(fileLineCol));
}
