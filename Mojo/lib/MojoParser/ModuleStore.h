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
// What a resolved module is once it has been loaded: the entity on disk, and
// the bindings that read out of it.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOPARSER_MODULESTORE_H
#define KGEN_LIB_MOJOPARSER_MODULESTORE_H

#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/ModuleSpec.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>
#include <optional>
#include <string>

namespace M::KGEN::LIT {

class ASTDecl;
struct ModuleState;

//===----------------------------------------------------------------------===//
// ModuleOrigin
//===----------------------------------------------------------------------===//

/// One importable filesystem entity: a source file, a package directory, or a
/// precompiled artifact whose single file holds many modules. Every binding
/// that reads out of it points at this one record.
struct ModuleOrigin {
  explicit ModuleOrigin(std::string canonicalPath)
      : canonicalPath(std::move(canonicalPath)) {}

  ~ModuleOrigin() {
    // Drop any remaining operations in the reader to avoid dangling
    // unmaterialized operations. If these were needed, they would have been
    // handled already as part of parsing.
    if (bytecodeReader)
      (void)bytecodeReader->finalize([](mlir::Operation *) { return false; });
  }

  /// The canonical path. Also the key this origin is stored under.
  std::string canonicalPath;

  /// The binding that fixes the symbol path this origin's contents are named
  /// by, and the one every later binding of the entity aliases to. First
  /// binding wins.
  ModuleState *canonicalBinding = nullptr;

  /// The dotted name that the canonical binding is mounted at. Set with it.
  /// Re-anchoring rewrites an artifact's references to exactly one path, which
  /// is this one.
  std::string canonicalMount;

  //===--------------------------------------------------------------------===//
  // Precompiled Artifact State
  //===--------------------------------------------------------------------===//

  /// Keeps the bytecode buffer alive for deferred lazy materialization.
  /// BytecodeReader holds bufferOwnerRef by reference, so this is declared
  /// first to outlive the reader below.
  std::shared_ptr<llvm::SourceMgr> sourceMgr;

  /// The reader every module bound out of this file materializes through. Null
  /// until the artifact is read, and for anything built from source.
  std::unique_ptr<mlir::BytecodeReader> bytecodeReader;

  /// A temporary module used to load the bytecode.
  mlir::ModuleOp tmpModule;

  /// Where the artifact was imported. Its source files are never parsed, so a
  /// decl materialized out of it later has no location of its own and gets
  /// reported at the import site instead.
  llvm::SMLoc bytecodeImportLoc;
};

//===----------------------------------------------------------------------===//
// ModuleState
//===----------------------------------------------------------------------===//

/// One binding of a module: what a single name in a single scope resolved to.
/// Several of these can share one `ModuleOrigin`.
struct ModuleState {
  ModuleState(ASTDecl *decl = nullptr) : decl(decl) {}
  ModuleState(ASTDecl *decl, const ModuleSpec &spec) : decl(decl), spec(spec) {}

  /// Insert a nested module state.
  ModuleState &insertNestedModule(mlir::StringAttr name,
                                  std::unique_ptr<ModuleState> module) {
    nestedModuleAllocations.emplace_back(std::move(module));
    nestedModules.insert({name, nestedModuleAllocations.back().get()});
    return *nestedModuleAllocations.back();
  }

  /// The decl associated with the module or package.
  ASTDecl *decl = nullptr;
  /// The module spec this state was created from. Absent for any state that
  /// resolution never produced a candidate for: the top-level state, error
  /// states, and the modules found inside an already-loaded artifact.
  std::optional<ModuleSpec> spec;

  /// The entity this state reads out of, shared with every other binding of
  /// the same one. Null for the top-level and erroneous states, and for a
  /// namespace, which spans several directories and so has no single one.
  ModuleOrigin *origin = nullptr;

  /// The optional source path of this module if it was loaded from source.
  std::optional<std::string> sourcePath() const {
    if (spec && !spec->isPrecompiled())
      return spec->path.string();
    return std::nullopt;
  }
  /// For a package, the location of the import statement that first pulled it
  /// in; used for diagnostics. Imported module states are shared across all
  /// compilation units so we can only meaningfully track one location, even if
  /// it's imported in multiple places.
  llvm::SMLoc importLoc;
  /// True for packages pulled in implicitly by the compiler (e.g., std/prelude)
  /// rather than by a user `import`. Such packages never get an `importLoc`, to
  /// avoid spurious "included from" locations.
  bool isImplicitImport = false;

  //===--------------------------------------------------------------------===//
  // Package Specific State
  //===--------------------------------------------------------------------===//

  /// The set of nested modules.
  llvm::SmallVector<std::unique_ptr<ModuleState>> nestedModuleAllocations;
  llvm::DenseMap<mlir::StringAttr, ModuleState *> nestedModules;

  /// Imports that failed to resolve through this scope, sharing one erroneous
  /// state per name. Lazily allocated.
  std::unique_ptr<
      llvm::DenseMap<mlir::StringAttr, std::unique_ptr<ModuleState>>>
      failedImports;

  /// For a failed-import state: the import locations already diagnosed.
  /// Resolver passes legitimately re-attempt the same statement and
  /// genuinely re-resolve, since failures aren't cached as modules; the
  /// re-attempt must not duplicate the report, while a distinct import site
  /// of the same missing name still gets its own.
  std::unique_ptr<llvm::SmallVector<llvm::SMLoc>> reportedFailureLocs;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump(unsigned indent = 0) const {
    for (auto &[name, state] : nestedModules) {
      llvm::dbgs() << llvm::indent(indent * 2) << name;
      if (state->nestedModules.empty()) {
        llvm::dbgs() << ",\n";
        continue;
      }
      llvm::dbgs() << " [\n";
      state->dump(indent + 1);
      llvm::dbgs() << llvm::indent(indent * 2) << "],\n";
    }
  }
#endif
};

//===----------------------------------------------------------------------===//
// ModuleStore
//===----------------------------------------------------------------------===//

/// Everything imported so far: the entities read off disk, and the bindings
/// naming them. Owned by `ModuleLoader`, which is the only thing that should
/// reach in here.
struct ModuleStore {
  /// A module state corresponding to the top-level decl. All imported packages
  /// or modules are nested within.
  std::unique_ptr<ModuleState> topLevelModuleState;

  /// A mapping between ASTDecl and the corresponding module state.
  llvm::MapVector<ASTDecl *, ModuleState *> moduleStates;

  /// A mapping between packages and their corresponding module state. A nullptr
  /// entry corresponds to the top level module state.
  /// FIXME(#17327): This only exists to work around the fact that we can't rely
  /// on an ASTDecl's parent reflecting the IR parent. When that issue gets
  /// fixed, this map should be removed in favor of just `moduleStates`.
  llvm::DenseMap<PackageOp, ModuleState *> packageStates;

  /// Every origin, owned here so it outlives the states pointing at it.
  llvm::SmallVector<std::unique_ptr<ModuleOrigin>> originAllocations;

  /// Origins by canonical path. One origin bound under two names is two
  /// ModuleStates, and so two of every type it declares, which is why a
  /// second differently-named binding is rejected rather than aliased.
  llvm::StringMap<ModuleOrigin *> originsByCanonicalPath;
};

} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJOPARSER_MODULESTORE_H
