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

#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include <cassert>

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DIBuilder
//===----------------------------------------------------------------------===//

DICompileUnitAttr DIBuilder::initializeCompileUnit(
    unsigned sourceLanguage, DIFileAttr file, StringRef producer,
    bool isOptimized, EmissionKind emissionKind, NameTableKind nameTableKind) {
  assert(!compileUnit && "compile unit already initialized");
  compileUnit = DICompileUnitAttr::get(
      sourceLanguage, file, producer, isOptimized, emissionKind, nameTableKind);
  return compileUnit;
}

//===----------------------------------------------------------------------===//
// Scopes

void DIBuilder::pushScope(DIScopeAttr scope) { scopes.push_back(scope); }

void DIBuilder::popScope() {
  assert(!scopes.empty() && "Cannot pop the compile unit scope!");
  scopes.pop_back();
}

Location DIBuilder::createScopedLoc(Location loc) {
  if (scopes.empty() || !scopes.back())
    return loc;

  // Check if this is already a scoped location with the expected scope.
  if (auto scopedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(loc))
    if (scopedLoc.getMetadata() == scopes.back())
      return loc;
  return FusedLoc::get(loc.getContext(), loc, scopes.back());
}

Location DIBuilder::createScopedLocIfMismatch(Location loc) {
  if (scopes.empty() || !scopes.back())
    return loc;

  DIScopeAttr currentScope = scopes.back();

  // Only perform strict scope validation when we have full debug information.
  // In non-full debug modes, operations may not have complete scope info.
  bool hasFullDebugInfo =
      compileUnit && compileUnit.getEmissionKind() == EmissionKind::Full;

  if (hasFullDebugInfo) {
    auto getRootSubprogram = [](DIScopeAttr scope) -> DISubprogramAttr {
      DISubprogramAttr subprogram = nullptr;
      while (scope) {
        if (auto sp = dyn_cast<DISubprogramAttr>(scope))
          subprogram = sp;
        scope =
            TypeSwitch<DIScopeAttr, DIScopeAttr>(scope)
                .Case([](DILexicalBlockAttr block) { return block.getScope(); })
                .Case([](DISubprogramAttr sp) { return sp.getScope(); })
                .Default(DIScopeAttr());
      }
      return subprogram;
    };

    DISubprogramAttr currentSubprogram = getRootSubprogram(currentScope);
    auto sharesSubprogramScope = [&](DIScopeAttr scope) -> bool {
      DISubprogramAttr scopeSubprogram = getRootSubprogram(scope);
      return scopeSubprogram == currentSubprogram;
    };

    // Walk through nested fused locations checking all scopes
    Location checkLoc = loc;
    if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(checkLoc)) {
      if (!sharesSubprogramScope(fusedLoc.getMetadata())) {
        return FusedLoc::get(loc.getContext(), loc, currentScope);
      }
      // the scope is a child of the current scope
      return loc;
    }
  }

  if (auto scopedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(loc))
    if (scopedLoc.getMetadata() == currentScope)
      return loc;
  return FusedLoc::get(loc.getContext(), loc, currentScope);
}

//===----------------------------------------------------------------------===//
// Creation

DILexicalBlockAttr DIBuilder::createNestedLexicalBlock(DIFileAttr file,
                                                       unsigned line,
                                                       unsigned column) {
  if (!scopes.back())
    return nullptr;
  auto scope = cast<DILocalScopeAttr>(scopes.back());
  return DILexicalBlockAttr::get(scope, file, line, column);
}

DISubprogramAttr DIBuilder::createSubprogram(SourceNameAttr sourceName,
                                             StringAttr linkageName,
                                             DIFileAttr file, unsigned int line,
                                             unsigned int scopeLine,
                                             SubprogramFlags subprogramFlags,
                                             DISubroutineType type) {
  // The only non-local scope we have is the file scope.
  // TODO(MOCO-834): Properly handle nested function.
  bool isDefinition =
      bitEnumContainsAny(subprogramFlags, SubprogramFlags::Definition);
  return DISubprogramAttr::get(isDefinition ? compileUnit : nullptr, file,
                               sourceName, linkageName, file, line, scopeLine,
                               subprogramFlags, type);
}

DIFileAttr DIBuilder::createFile(StringRef name, StringRef directory) {
  return DIFileAttr::get(context, name, directory);
}
DIFileAttr DIBuilder::createFile(FileLineColLoc loc) {
  return DIFileAttr::get(context, loc.getFilename(), "");
}

DILocalVariableAttr DIBuilder::createLocalVariable(StringRef name,
                                                   DIFileAttr file,
                                                   unsigned line, unsigned arg,
                                                   unsigned alignInBits,
                                                   DIType type, DIFlags flags) {
  auto scope = cast<DILocalScopeAttr>(scopes.back());
  return DILocalVariableAttr::get(scope, name, file, line, arg, alignInBits,
                                  type, flags);
}

LogicalResult DIBuilder::visitLexicalRegion(Region &region) {
  for (Operation &op : region.front()) {
    op.setLoc(createScopedLocIfMismatch(op.getLoc()));
    for (Region &region : op.getRegions()) {
      auto fileLoc = op.getLoc()->findInstanceOf<FileLineColLoc>();
      if (!fileLoc)
        return mlir::emitError(op.getLoc()) << "did not find a FileLineColLoc";
      DebugInfo::DIBuilder::ScopeGuard guard = pushNestedLexicalBlock(
          createFile(fileLoc), fileLoc.getLine(), fileLoc.getColumn());
      if (failed(visitLexicalRegion(region)))
        return failure();
    }
  }
  return success();
}
