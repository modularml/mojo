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
// Finding modules and packages on disk.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_MODULELOADER_H
#define KGEN_MOJOPARSER_MODULELOADER_H

#include "Mojo/MojoParser/ModuleSpec.h"
#include "Mojo/MojoParser/SharedState.h"

#include <memory>
#include <optional>

namespace M::KGEN::LIT {

/// The binding a module name is about to make: which entity it names, where
/// that lands, and whether some other name already binds it.
struct BindingSpec {
  /// The name being bound, in the scope binding it.
  StringAttr name;

  /// The entity read out of, shared by every binding of it. Null for a
  /// namespace, which spans several import roots and so names no single one.
  ModuleOrigin *origin = nullptr;

  /// Any existing binding this one was aliased to. Only set for a second
  /// binding of a given origin under a different name.
  ModuleState *aliasOf = nullptr;

  /// The dotted path `name` makes in that scope, empty when there is no origin.
  std::string mount;
};

/// Module & package handling logic.
///
/// Its primary purpose is resolution: given an import path, work out what it
/// names on disk and describe it as a `ModuleSpec`.
class ModuleLoader : public SharedStateUser {
public:
  ModuleLoader(SharedState &shared);
  ~ModuleLoader();

  /// Resolve the absolute path for a given module name. Returns nullopt if the
  /// module cannot be found.
  std::optional<ModuleSpec> resolveModulePath(StringRef moduleName,
                                              llvm::SMLoc includeLoc);

  /// Resolve the absolute path for a given module name within the provided
  /// directory. Returns nullopt if the module cannot be found.
  ///
  /// \p isInsideSourcePackage is true for submodule search inside a source
  /// package. In that mode, any `.mojoc` candidate is ignored.
  std::optional<ModuleSpec> resolveModulePath(StringRef moduleName,
                                              StringRef includeDir,
                                              bool ignorePrebuilt,
                                              bool isInsideSourcePackage);

  /// Collect the portion directories of the namespace described by
  /// `parentSpec`: for each import directory visible from
  /// `importBufferFileId`, descend the spec's namespace components as plain
  /// directories. Portions are returned in traversal order, deduplicated.
  ///
  /// A "portion" is one root's directory contributing to the namespace. For the
  /// example on `namespaceComponents`, the portions of `foo.bar` are
  ///
  ///   [ one/foo/bar, two/foo/bar ]
  ///
  /// The search path always contains the same directories for every
  /// import site, and portions are recomputed from it at each resolution
  /// step, so a root where `foo` is missing, or is claimed by a source
  /// package or module file (a closed candidate), simply contributes no
  /// portion.
  SmallVector<std::string>
  collectNamespacePortions(const ModuleSpec &parentSpec,
                           unsigned importBufferFileId);

  /// Resolve the name as a submodule of the namespace described by
  /// `parentSpec`, searching every portion. Returns every distinct thing the
  /// name could be: closed (non-directory) candidates win over directory
  /// candidates, which merge into a single nested-namespace spec rather than
  /// competing. More than one element therefore means the import is
  /// ambiguous; several portions provide a closed candidate.
  SmallVector<ModuleSpec>
  resolveNamespaceSubModule(StringRef moduleName, const ModuleSpec &parentSpec,
                            unsigned importBufferFileId);

  /// Traverse the directories available for importing modules and packages,
  /// calling the given callback for each directory found.
  void
  traverseImportDirectories(unsigned importBufferFileId,
                            function_ref<WalkResult(StringRef)> callback) const;

  /// Resolve what binding the name makes in the provided scope, creating the
  /// entity's origin if this is the first binding of it.
  ///
  /// An origin can only carry one symbol path; the first binding names the
  /// entity and any later one under a different name is an alias, which this
  /// installs and warns about at `importLoc` - the import that asked for it.
  /// An invalid `importLoc` means nothing asked, so nothing is reported.
  BindingSpec resolveModuleBinding(const ModuleSpec &spec, StringAttr name,
                                   ModuleState &parentState, SMLoc importLoc);

  /// Every origin created so far, in creation order.
  ArrayRef<std::unique_ptr<ModuleOrigin>> getOrigins() const;

  //===--------------------------------------------------------------------===//
  // Module states
  //===--------------------------------------------------------------------===//

  /// Create the state the whole import tree nests inside. Called once, when
  /// the top-level decl exists.
  void initializeTopLevel(ASTDecl &topLevelDecl);

  /// The state of the top-level decl, which every import is nested within.
  ModuleState &getTopLevelState() const;

  /// The state this decl was imported as, or null if it names no module.
  ModuleState *lookupState(ASTDecl *decl) const;

  /// Record `state` as what `decl` was imported as.
  ///
  /// Given the binding that created it, `state` also takes its origin, and
  /// claims the naming of that origin's contents when no binding has yet.
  void setState(ASTDecl &decl, ModuleState &state,
                const BindingSpec *binding = nullptr);

  /// Record `state` as what `packageOp` was imported as.
  void setPackageState(PackageOp packageOp, ModuleState &state);

  /// Forget the state of a decl whose import turned out to have failed.
  void eraseState(ASTDecl *decl);

  //===--------------------------------------------------------------------===//
  // Importing
  //===--------------------------------------------------------------------===//

  /// Import the specified module or package, returning the decl. Always returns
  /// a valid decl, even if a corresponding module or package could not be
  /// found.
  ASTDecl &importModule(const SharedState::ImportPath &path,
                        PackageOp currentPackage, llvm::SMLoc loc);

  /// Import the specified module or package, returning the module state.
  /// Always returns a valid module state, even if the module could not be
  /// found. `isImplicit` marks a package pulled in by the compiler rather than
  /// a user `import`.
  ModuleState &importModuleState(const SharedState::ImportPath &path,
                                 ASTDecl *context, llvm::SMLoc loc,
                                 bool isImplicit = false);

  /// Import the specified module or package nested within the given parent
  /// decl, returning the module state. Always returns a valid module state,
  /// even if the module could not be found.
  ModuleState &importSubModuleState(StringRef name, ASTDecl *parentDecl,
                                    llvm::SMLoc loc, llvm::SMLoc identifierLoc);

  /// Try to import a direct submodule with the given name, returning the
  /// submodule's decl, or null if no such submodule exists.
  ASTDecl *tryImportSubModule(ASTDecl &parent, StringRef name, llvm::SMLoc loc);

  /// Return true if the package has a nested module with the given name in the
  /// module-state cache. The package body must be resolved before calling this
  /// for the result to be accurate.
  bool hasNestedModule(PackageOp packageOp, StringRef name) const;

  /// Return the decls of every nested module/package currently materialized in
  /// the package's module-state cache. These are navigable but unlisted (not
  /// in the package's importable scope).
  SmallVector<ASTDecl *> getNestedModuleDecls(PackageOp packageOp) const;

  /// Scan a source package's directory and register a child decl for every
  /// sibling module/sub-package - a deferred `FileModuleOp` for each `.mojo`
  /// file and a (already-deferred) `PackageOp` for each sub-package. The
  /// children are unlisted (in the module-state cache + package IR, never the
  /// package's importable scope) and their bodies/files are opened lazily.
  void registerSourcePackageChildren(ASTDecl &packageDecl);

  //===--------------------------------------------------------------------===//
  // Loading
  //===--------------------------------------------------------------------===//

  /// Shared core of createModuleState and createDeferredModuleState: create the
  /// `FileModuleOp` + unlisted decl + nested module state. The caller supplies
  /// the parse cursor (valid for an already-open module, invalid for a deferred
  /// one) and is responsible for importing builtins (eagerly, or at
  /// materialization for a deferred module).
  ModuleState &createFileModuleState(ModuleState &parentState,
                                     FileLineColLoc loc, llvm::SMLoc declLoc,
                                     LexerCursor cursor, LexerCursor endCursor,
                                     const ModuleSpec &spec,
                                     const BindingSpec &binding);

  /// Create a new module state with the given name, location, and body.
  ModuleState &createModuleState(StringAttr declName,
                                 const llvm::MemoryBuffer *moduleBuffer,
                                 ModuleState &parentState, FileLineColLoc loc,
                                 const ModuleSpec &spec, SMLoc importLoc);

  /// Create a module state for a source module whose file has not been opened.
  ModuleState &createDeferredModuleState(ModuleSpec moduleSpec,
                                         ModuleState &parentState);

  /// Create a new module state for a package with the given spec, location,
  /// and body. The importLoc, if valid, is the location of the `import` that
  /// pulled the package in. The spec's kind must be a SourcePackage or
  /// SourceDir.
  ModuleState &createPackageState(ModuleSpec moduleSpec,
                                  ModuleState &parentState, SMLoc importLoc);

  /// Create a new module state for a binary package with the given spec.
  ModuleState &createBinaryPackageState(SMLoc loc, const ModuleSpec &spec,
                                        ModuleState &parentState);

  /// Create an error module state and emit the given error message. If
  /// `unlisted` is set, the erroneous decl is not registered in
  /// `errorContext`'s name table. A non-empty `note` is attached to the error.
  ModuleState &createErrorModuleState(SMLoc loc, StringAttr name,
                                      ASTDecl &errorContext,
                                      const Twine &errorMsg,
                                      bool unlisted = false,
                                      const Twine &note = {});

private:
  /// The state this package op was imported as, or null. Distinct from
  /// `lookupState` only because one op can be reached through several decls.
  ModuleState *lookupPackageState(PackageOp packageOp) const;

  /// Shared core of `importSubModuleState` and `tryImportSubModule`. With
  /// `emitErrors` clear, a name that is simply not a submodule returns null
  /// rather than a diagnostic, which is the probe the latter needs.
  ModuleState *importSubModuleStateImpl(StringRef name, ASTDecl *parentDecl,
                                        llvm::SMLoc loc,
                                        llvm::SMLoc identifierLoc,
                                        bool emitErrors);

  /// True when binding the spec to the given name would alias an entity some
  /// other name already binds.
  bool wouldAliasExistingBinding(const ModuleSpec &spec, StringRef name,
                                 ASTDecl &parentDecl) const;

  /// Look up a module by its name, in the specified parent scope, in the module
  /// cache. Returns nullptr on a miss. On a hit:
  ///   * Prevent module self-imports
  ///   * Memoizes the import loc on first resolution
  ModuleState *lookupModuleCache(StringRef name, ASTDecl *parentDecl,
                                 ModuleState *parentState, llvm::SMLoc loc,
                                 llvm::SMLoc identifierLoc, bool emitErrors);

  /// Import the specified module or package, which contains `.` indexing,
  /// returning the module state. Always returns a valid module state, even if
  /// the module could not be found.
  ModuleState &importRelativeModuleState(const SharedState::ImportPath &path,
                                         ASTDecl *parentDecl, llvm::SMLoc loc);

  /// The directories searched before the working directory and the source
  /// manager's include directories: configured search paths, or the defaults
  /// from the Mojo config when none were given.
  SmallVector<std::string> autoImportDirs;

  /// What has been imported so far.
  std::unique_ptr<ModuleStore> store;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_MODULELOADER_H
