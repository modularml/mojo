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

#include "Mojo/MojoParser/ModuleLoader.h"

#include "ClosureEmitter.h"
#include "ModuleStore.h"

#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/Lexer.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/Support/Configuration.h"
#include "Mojo/Support/MojoPrecompiledFile.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Filesystem/Paths.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

#define DEBUG_TYPE "mojo-module-loader"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ModuleSpec
//===----------------------------------------------------------------------===//

std::optional<ModuleSpec>
ModuleSpec::classify(const std::filesystem::path &path,
                     llvm::StringRef moduleName) {
  // For directory-based module filtering, we must have an exact match.
  if (auto name = path.filename().string();
      moduleName.empty() || name == moduleName) {
    if (Filesystem::isMojoSourcePackagePath(path))
      return ModuleSpec{name, path, ModuleSpec::Kind::SourcePackage};

    std::error_code ec;
    if (std::filesystem::is_directory(path, ec) && !ec)
      return ModuleSpec{name, path, ModuleSpec::Kind::SourceDir};
  }

  // For file-based module filtering, the name must match the filename's stem
  // (i.e., without the final extension).
  if (auto stem = path.filename().stem().string();
      moduleName.empty() || stem == moduleName) {
    if (Filesystem::isMojoBinaryPackagePath(path))
      return ModuleSpec{stem, path, ModuleSpec::Kind::Precompiled};

    if (Filesystem::isMojoSourceFile(path))
      return ModuleSpec{stem, path, ModuleSpec::Kind::SourceModule};
  }

  return std::nullopt;
}

std::string ModuleSpec::canonicalPath() const {
  std::error_code ec;
  std::filesystem::path canonical = std::filesystem::weakly_canonical(path, ec);
  return (ec ? path.lexically_normal() : canonical).string();
}

//===----------------------------------------------------------------------===//
// ModuleLoader
//===----------------------------------------------------------------------===//

/// Collect the import paths configured in the Mojo config, used when no search
/// paths were given explicitly.
static void collectDefaultImportPaths(SmallVector<std::string> &paths) {
  ErrorOr<MojoConfig> cfg = MojoConfig::open();
  if (failed(cfg)) {
    LLVM_DEBUG(llvm::dbgs()
               << "failed to open config: " << cfg.getError() << "\n");
    return;
  }

  // Add any paths specified in the config.
  SmallVector<StringRef> importPaths;
  cfg->getParserImportPaths(importPaths);
  LLVM_DEBUG(llvm::dbgs() << "Using import paths: "
                          << llvm::join(importPaths, ",") << "\n");

  for (StringRef path : importPaths)
    paths.push_back(path.str());
}

ModuleLoader::ModuleLoader(SharedState &shared)
    : SharedStateUser(shared), store(std::make_unique<ModuleStore>()) {
  const CompilationOptions &options = shared.options;
  if (!options.searchPaths.empty()) {
    SmallVector<StringRef> paths;
    StringRef(options.searchPaths)
        .split(paths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    llvm::append_range(autoImportDirs, paths);
  } else {
    collectDefaultImportPaths(autoImportDirs);
  }
  llvm::append_range(autoImportDirs, options.extraSearchPaths);
}

ModuleLoader::~ModuleLoader() = default;

std::optional<ModuleSpec>
ModuleLoader::resolveModulePath(StringRef moduleName, StringRef includeDir,
                                bool ignorePrebuilt,
                                bool isInsideSourcePackage) {
  // Find a path in `includeDir` that is an importable mojo construct matching
  // `moduleName`
  std::error_code ec;
  auto iter = std::filesystem::directory_iterator(includeDir.str(), ec);
  if (ec)
    return std::nullopt;

  // Gets the name of the file or directory in a case sensitive way. On non-case
  // sensitive systems we cannot just do `path / moduleName` since the
  // constructed path will not adhere to case sensitivity.
  std::optional<ModuleSpec> bestMatch;
  for (const auto &entry : iter) {
    if (auto moduleSpec = ModuleSpec::classify(entry.path(), moduleName)) {
      // A package can't legitimately nest a precompiled copy of itself or its
      // own submodules, so this ignores every `.mojoc` candidate when
      // resolving from within a package's own directory, unlike the
      // top-level `-I` search where `.mojoc`-before-`.mojo` precedence still
      // applies to genuine collisions.
      if (moduleSpec->isPrecompiled() &&
          (ignorePrebuilt || isInsideSourcePackage))
        continue;
      if (!bestMatch || moduleSpec->takesImportPrecedence(*bestMatch))
        bestMatch = moduleSpec;
    }
  }

  return bestMatch;
}

std::optional<ModuleSpec> ModuleLoader::resolveModulePath(StringRef moduleName,
                                                          SMLoc includeLoc) {
  unsigned includeBufferId =
      shared.getSourceMgr().FindBufferContainingLoc(includeLoc);

  // A closed (non-directory) candidate in the earliest directory wins
  // outright. Plain directories are namespace portions: they only name the
  // namespace when no closed candidate exists anywhere on the path, and the
  // returned spec records just the first portion; submodule resolution
  // re-derives the full portion set from the spec's namespace components.
  std::optional<ModuleSpec> result;
  std::optional<ModuleSpec> firstPortion;
  traverseImportDirectories(includeBufferId, [&](StringRef dir) {
    // Don't try to resolve modules that reside within a package.
    if (Filesystem::isMojoSourcePackagePath(dir.str())) {
      // TODO: It'd be nice to emit a list of potential modules that the
      // name might correspond with if it did resolve to one inside of this
      // package.
      return WalkResult::advance();
    }
    std::optional<ModuleSpec> candidate =
        resolveModulePath(moduleName, dir, shared.arePrebuiltPackagesDisabled(),
                          /*isInsideSourcePackage=*/false);
    if (!candidate)
      return WalkResult::advance();
    if (candidate->kind != ModuleSpec::Kind::SourceDir) {
      result = candidate;
      return WalkResult::interrupt();
    }
    if (!firstPortion) {
      firstPortion = std::move(candidate);
      firstPortion->namespaceComponents.push_back(moduleName.str());
    }
    return WalkResult::advance();
  });

  return result ? result : firstPortion;
}

SmallVector<std::string>
ModuleLoader::collectNamespacePortions(const ModuleSpec &parentSpec,
                                       unsigned importBufferFileId) {
  assert(parentSpec.isNamespace() && "expected a namespace spec");
  SmallVector<std::string> portions;
  llvm::StringSet<> seenPortions;
  traverseImportDirectories(importBufferFileId, [&](StringRef dir) {
    if (Filesystem::isMojoSourcePackagePath(dir.str()))
      return WalkResult::advance();
    std::filesystem::path portion(dir.str());
    for (const std::string &component : parentSpec.namespaceComponents) {
      std::error_code ec;
      auto iter = std::filesystem::directory_iterator(portion, ec);
      if (ec)
        return WalkResult::advance();
      // Only a plain directory contributes a portion; a source package (or
      // any other kind) owning this name is closed and resolves by itself.
      std::optional<std::filesystem::path> child;
      for (const auto &entry : iter) {
        if (auto spec = ModuleSpec::classify(entry.path(), component);
            spec && spec->kind == ModuleSpec::Kind::SourceDir) {
          child = entry.path();
          break;
        }
      }
      if (!child)
        return WalkResult::advance();
      portion = std::move(*child);
    }
    // Deduplicate: the buffer-derived working directory may coincide with an
    // include directory, and a duplicate portion must not fake an ambiguity.
    std::error_code ec;
    std::filesystem::path canonical =
        std::filesystem::weakly_canonical(portion, ec);
    std::string key = (ec ? portion.lexically_normal() : canonical).string();
    if (seenPortions.insert(key).second)
      portions.push_back(portion.string());
    return WalkResult::advance();
  });
  return portions;
}

SmallVector<ModuleSpec>
ModuleLoader::resolveNamespaceSubModule(StringRef moduleName,
                                        const ModuleSpec &parentSpec,
                                        unsigned importBufferFileId) {
  // Per portion, in-directory precedence picks one candidate. Across
  // portions, closed candidates win over directory candidates (plain
  // directories carry no marker, so a stray non-Mojo directory must not
  // shadow a real module), and directory candidates merge into a single
  // nested namespace. Every closed candidate is returned: more than one is
  // an ambiguity for the caller to report.
  SmallVector<ModuleSpec> closed;
  std::optional<ModuleSpec> firstDir;
  for (const std::string &portion :
       collectNamespacePortions(parentSpec, importBufferFileId)) {
    std::optional<ModuleSpec> candidate = resolveModulePath(
        moduleName, portion, shared.arePrebuiltPackagesDisabled(),
        /*isInsideSourcePackage=*/false);
    if (!candidate)
      continue;
    if (candidate->kind == ModuleSpec::Kind::SourceDir) {
      if (!firstDir)
        firstDir = std::move(candidate);
      continue;
    }
    closed.push_back(std::move(*candidate));
  }
  if (!closed.empty())
    return closed;

  if (!firstDir)
    return {};
  firstDir->namespaceComponents = parentSpec.namespaceComponents;
  firstDir->namespaceComponents.push_back(moduleName.str());
  return {std::move(*firstDir)};
}

/// Return the directory to use as the "working directory" for relative-ish
/// module lookup. This is the directory containing the given buffer's file,
/// walked up past any enclosing packages, falling back to the process's
/// working directory when the buffer identifier has no existing parent
/// directory. Returns an empty path if no absolute directory could be
/// derived.
static std::filesystem::path
deriveWorkingDirectory(const llvm::SourceMgr &sourceMgr,
                       unsigned importBufferFileId) {
  if (!importBufferFileId)
    return {};

  // The buffer identifier usually names a real file, but REPL and LSP
  // docstring code-block wrapper buffers have synthetic names formed by
  // suffixing the real path (e.g. "foo.mojo wrapper_at(42)"). The identifier
  // itself need not exist - only its parent directory does, which for wrapper
  // buffers is the real file's directory. Identifiers with no usable parent
  // (relative compile inputs, "<stdin>", REPL cells) fall back to the
  // process's working directory.
  std::optional<std::filesystem::path> path;
  if (auto *importBuffer = sourceMgr.getMemoryBuffer(importBufferFileId)) {
    std::filesystem::path identifier(importBuffer->getBufferIdentifier().str());
    if (identifier.has_parent_path() &&
        llvm::sys::fs::exists(identifier.parent_path().string()))
      path = std::move(identifier);
  }

  bool pathFromBuffer = path.has_value();

  // An empty relative path absolutizes to the process's working directory.
  SmallString<256> absolute(path.value_or("").string());
  if (llvm::sys::fs::make_absolute(absolute))
    return {};
  path = absolute.str().str();

  // The buffer's identifier names a file path - real, or a synthetic wrapper
  // name in an existing directory - so step up to its containing directory.
  // The process-CWD fallback is already the directory to search. Either way,
  // work back up to the top-most non-package directory.
  if (pathFromBuffer)
    path = path->parent_path();
  while (Filesystem::isMojoSourcePackagePath(*path))
    path = path->parent_path();
  return *path;
}

void ModuleLoader::traverseImportDirectories(
    unsigned importBufferFileId,
    function_ref<WalkResult(StringRef)> callback) const {
  // Python has lots of magic rules surrounding how modules get resolved. For
  // now, we search the auto-import directories, the working directory derived
  // from the importing buffer, and the source manager's include directories,
  // in that order.
  // Check the auto import directories.
  for (auto &rawPath : autoImportDirs) {
    if (callback(rawPath).wasInterrupted())
      return;

    // Cannot find the file, then check child directories of the auto import
    // directory.
    std::error_code ec;
    for (llvm::sys::fs::recursive_directory_iterator f(rawPath, ec), e; f != e;
         f.increment(ec)) {
      if (ec)
        continue;
      const std::string &path = f->path();
      // Skip non-directories and source packages, internal packages should be
      // imported using a relative import.
      if (!llvm::sys::fs::is_directory(path) ||
          Filesystem::isMojoSourcePackagePath(path))
        continue;
      if (callback(path).wasInterrupted())
        return;
    }
  }

  // Check the working directory: the entry point's directory, derived once
  // from the main buffer and visible to every import site alike. No import site
  // sees its own file's directory, meaning that resolution is stable and cannot
  // depend on which file triggered it. A null buffer id requests no working
  // directory at all.
  if (importBufferFileId) {
    llvm::SourceMgr &sourceMgr = shared.getSourceMgr();
    std::filesystem::path cwd =
        deriveWorkingDirectory(sourceMgr, sourceMgr.getMainFileID());
    if (!cwd.empty() && callback(cwd.string()).wasInterrupted())
      return;
  }

  // Check the include directories.
  for (StringRef includeDir : shared.getSourceMgr().getIncludeDirs())
    if (callback(includeDir).wasInterrupted())
      return;
}

//===----------------------------------------------------------------------===//
// Origins
//===----------------------------------------------------------------------===//

static std::string mountPathFor(StringRef boundName, ASTDecl &parentDecl) {
  std::string mount;
  if (SymbolRefAttr parent = parentDecl.getSymbolRef()) {
    mount = parent.getRootReference().str();
    for (FlatSymbolRefAttr nested : parent.getNestedReferences())
      mount += ("." + nested.getValue()).str();
    mount += ".";
  }
  mount += boundName;
  return mount;
}

/// Warn that the import bound as 'mount' at 'loc' names an entity already bound
/// under another name. Reported lazily from the lookup.
static void warnAliasedBinding(const ModuleLoader &loader, SMLoc loc,
                               StringRef mount, const ModuleOrigin &origin,
                               bool isModule) {
  StringRef noun = isModule ? "module" : "package";
  MojoInflightDiag diag = loader.emitWarning(
      loc, "'" + mount + "' and '" + origin.canonicalMount +
               "' name the same " + noun +
               "; remove the duplicate import root or file that "
               "reaches it twice");
  // Whichever name has already been bound is the canonical one and decides how
  // the contents are reported to users; point to that.
  SMLoc canonicalLoc = origin.canonicalBinding->importLoc;
  if (canonicalLoc.isValid()) {
    diag.attachNote(canonicalLoc)
        << "'" << origin.canonicalMount
        << "' is the name used in error messages and debug info";
  }
}

bool ModuleLoader::wouldAliasExistingBinding(const ModuleSpec &spec,
                                             StringRef name,
                                             ASTDecl &parentDecl) const {
  if (!spec.isSourceModule() && !spec.isSourcePackage() &&
      !spec.isPrecompiled())
    return false;

  auto it = store->originsByCanonicalPath.find(spec.canonicalPath());
  return it != store->originsByCanonicalPath.end() &&
         it->second->canonicalBinding &&
         it->second->canonicalMount != mountPathFor(name, parentDecl);
}

BindingSpec ModuleLoader::resolveModuleBinding(const ModuleSpec &spec,
                                               StringAttr name,
                                               ModuleState &parentState,
                                               SMLoc importLoc) {
  BindingSpec binding;
  binding.name = name;
  // A namespace is several directories under different import roots, so there
  // is no single entity for it to own, and so never aliases.
  if (!spec.isSourceModule() && !spec.isSourcePackage() &&
      !spec.isPrecompiled())
    return binding;

  binding.mount = mountPathFor(name.getValue(), *parentState.decl);

  std::string canonicalPath = spec.canonicalPath();
  auto it = store->originsByCanonicalPath.find(canonicalPath);
  if (it == store->originsByCanonicalPath.end()) {
    store->originAllocations.push_back(
        std::make_unique<ModuleOrigin>(std::move(canonicalPath)));
    binding.origin = store->originAllocations.back().get();
    store->originsByCanonicalPath[binding.origin->canonicalPath] =
        binding.origin;
    return binding;
  }

  binding.origin = it->second;
  // A second name for the one entity aliases the binding that already names it.
  if (binding.origin->canonicalMount != binding.mount &&
      binding.origin->canonicalBinding) {
    binding.aliasOf = binding.origin->canonicalBinding;
    parentState.nestedModules.insert({name, binding.aliasOf});
    // Only when actually resolving an import to this binding do we warn.
    if (importLoc.isValid()) {
      warnAliasedBinding(*this, importLoc, binding.mount, *binding.origin,
                         spec.isSourceModule());
    }
  }
  return binding;
}

ArrayRef<std::unique_ptr<ModuleOrigin>> ModuleLoader::getOrigins() const {
  return store->originAllocations;
}

//===----------------------------------------------------------------------===//
// Module states
//===----------------------------------------------------------------------===//

void ModuleLoader::initializeTopLevel(ASTDecl &topLevelDecl) {
  store->topLevelModuleState = std::make_unique<ModuleState>(&topLevelDecl);
  ModuleState &state = *store->topLevelModuleState;
  store->moduleStates[&topLevelDecl] = &state;
  store->packageStates[nullptr] = &state;
}

ModuleState &ModuleLoader::getTopLevelState() const {
  assert(store->topLevelModuleState && "loader has not been initialized");
  return *store->topLevelModuleState;
}

ModuleState *ModuleLoader::lookupState(ASTDecl *decl) const {
  return store->moduleStates.lookup(decl);
}

ModuleState *ModuleLoader::lookupPackageState(PackageOp packageOp) const {
  return store->packageStates.lookup(packageOp);
}

void ModuleLoader::setState(ASTDecl &decl, ModuleState &state,
                            const BindingSpec *binding) {
  store->moduleStates[&decl] = &state;
  if (!binding)
    return;

  state.origin = binding->origin;
  // First binding wins, so a state created for an origin something else
  // already names - a rebinding under the same name - shares it without
  // claiming it.
  if (binding->origin && !binding->origin->canonicalBinding) {
    binding->origin->canonicalBinding = &state;
    binding->origin->canonicalMount = binding->mount;
  }
}

void ModuleLoader::setPackageState(PackageOp packageOp, ModuleState &state) {
  store->packageStates[packageOp] = &state;
}

void ModuleLoader::eraseState(ASTDecl *decl) {
  store->moduleStates.erase(decl);
}

//===----------------------------------------------------------------------===//
// Importing
//===----------------------------------------------------------------------===//

ASTDecl &ModuleLoader::importModule(const SharedState::ImportPath &path,
                                    PackageOp currentPackage, llvm::SMLoc loc) {
  ModuleState *moduleState = lookupPackageState(currentPackage);
  assert(moduleState && "unexpected package without a module state");
  return *importModuleState(path, moduleState->decl, loc).decl;
}

ModuleState &
ModuleLoader::importModuleState(const SharedState::ImportPath &path,
                                ASTDecl *context, llvm::SMLoc loc,
                                bool isImplicit) {
  CompilerTimeTraceScope fullTimeScope("importModule",
                                       [&] { return path.toDottedString(); });

  // TODO: The terms "relative" and "submodule" are being stretched quite far
  // here. We're invoking "sub" on any trivial path ("std") and "relative" on
  // anything else.
  ModuleState &state =
      (path.components.size() > 1 || path.relativeLevel)
          ? importRelativeModuleState(path, context, loc)
          : importSubModuleState(path.components.front(),
                                 &shared.getTopLevelDecl(), loc, loc);

  // An implicit import gets no import location, so its diagnostics aren't
  // threaded through to the user file that triggered the implicit import. Clear
  // any location a nested resolution may already have set.
  if (isImplicit) {
    state.isImplicitImport = true;
    state.importLoc = SMLoc();
  }

  return state;
}

ModuleState &ModuleLoader::importSubModuleState(StringRef name,
                                                ASTDecl *parentDecl,
                                                llvm::SMLoc loc,
                                                llvm::SMLoc identifierLoc) {
  // emitErrors=true never returns null: on any failure it produces (and
  // returns) an error module state.
  return *importSubModuleStateImpl(name, parentDecl, loc, identifierLoc,
                                   /*emitErrors=*/true);
}

ModuleState *
ModuleLoader::lookupModuleCache(StringRef name, ASTDecl *parentDecl,
                                ModuleState *parentState, llvm::SMLoc loc,
                                llvm::SMLoc identifierLoc, bool emitErrors) {
  auto declNameAttr = StringAttr::get(getContext(), name);
  auto it = parentState->nestedModules.find(declNameAttr);
  if (it == parentState->nestedModules.end())
    return nullptr;

  ModuleState *state = it->second;

  // A standalone module importing its own bare name can only ever cache-hit
  // itself: the module is registered under its name at creation, so nothing
  // else can be found. Reject it conservatively so the choice of meaning stays
  // open.
  // A self-import inside a package names the enclosing package (a PackageOp)
  // and is unaffected, as are package-qualified imports of the module through
  // its own package.
  auto isSelfImport = [&](ModuleState *state) {
    if (parentDecl != &shared.getTopLevelDecl())
      return false;
    if (!isa_and_nonnull<FileModuleOp>(state->decl->getIfOperation()))
      return false;
    unsigned importerBufferId = getSourceMgr().FindBufferContainingLoc(loc);
    if (!importerBufferId)
      return false;
    // The importer's module is necessarily materialized, so a deferred target
    // (invalid decl loc) cannot be the importer itself.
    unsigned targetBufferId =
        getSourceMgr().FindBufferContainingLoc(state->decl->getLoc());
    if (targetBufferId && importerBufferId == targetBufferId)
      return true;
    // Imports written in a REPL/LSP docstring wrapper buffer are still
    // self-imports of the module they wrap.
    std::optional<StringRef> wrapped =
        shared.getWrappedSourcePath(importerBufferId);
    std::optional<std::string> sourcePath = state->sourcePath();
    return wrapped && sourcePath && *wrapped == *sourcePath;
  };

  // Reject self-imports with an unregistered erroneous state. The name *is*
  // resolvable so it must not be poisoned in the parent's name table; use an
  // unlisted decl for this.
  if (isSelfImport(state)) {
    if (!emitErrors)
      return state;
    return &createErrorModuleState(
        identifierLoc, declNameAttr, *parentState->decl,
        "module '" + name + "' cannot import itself", /*unlisted=*/true);
  }

  // Memoize the "imported from" location; the first resolution wins.
  if (!state->importLoc.isValid() && !state->isImplicitImport)
    state->importLoc = loc;

  return state;
}

/// Stdlib subpackages relocated to the `max` package. Entries are added or
/// removed as the stdlib restructuring for the compiler OSS release shapes up
/// in MSTDL-2788.
static constexpr StringLiteral kMovedStdlibSubpackages[] = {"runtime", "gpu",
                                                            "algorithm"};

/// Attached to import failures under those subpackages.
static constexpr StringLiteral kMovedStdlibNote =
    "many stdlib items recently moved to the `max` package, try `from "
    "max.<module>`";

ModuleState *ModuleLoader::importSubModuleStateImpl(StringRef name,
                                                    ASTDecl *parentDecl,
                                                    llvm::SMLoc loc,
                                                    llvm::SMLoc identifierLoc,
                                                    bool emitErrors) {
  // Grab the parent module state.
  ModuleState *parentState = lookupState(parentDecl);
  assert(parentState && "parent decl must have a module state");
  auto declNameAttr = StringAttr::get(getContext(), name);

  // Don't cascade diagnostics through an already-erroneous parent; propagate
  // its state silently.
  if (parentState->decl && parentState->decl->isErroneous())
    return parentState;

  // Check to see if we've already imported this module.
  if (ModuleState *state = lookupModuleCache(name, parentDecl, parentState, loc,
                                             identifierLoc, emitErrors)) {
    return state;
  }

  // On a genuine "no such submodule": null when probing (emitErrors=false), or
  // an error module state with the given message when importing (true).
  auto notFound = [&](const Twine &message) -> ModuleState * {
    if (!emitErrors)
      return nullptr;
    return &createErrorModuleState(identifierLoc, declNameAttr,
                                   *parentState->decl, message);
  };

  // As `notFound`, for a module missing under `std`: a subpackage that moved to
  // the `max` package also gets a migration note.
  auto notFoundModule = [&]() -> ModuleState * {
    if (!emitErrors)
      return nullptr;
    bool movedToMax =
        parentState->decl->getParentDecl() == &shared.getTopLevelDecl() &&
        parentState->spec && parentState->spec->name == "std" &&
        llvm::is_contained(kMovedStdlibSubpackages, name);
    return &createErrorModuleState(
        identifierLoc, declNameAttr, *parentState->decl,
        "unable to locate module '" + name + "'", /*unlisted=*/false,
        movedToMax ? Twine(kMovedStdlibNote) : Twine());
  };

  // Resolve the parent's body so that any lazily-materialized children (e.g.
  // from binary packages, or deferred source siblings) are registered in
  // nestedModules before we fall through to filesystem resolution.
  if (failed(getDeclResolver().resolveBody(*parentDecl, loc)))
    return notFound("failed to resolve parent package body");

  // Check the cache again after body resolution
  if (ModuleState *state = lookupModuleCache(name, parentDecl, parentState, loc,
                                             identifierLoc, emitErrors)) {
    return state;
  }

  // Resolve the path for this module.
  std::optional<ModuleSpec> modulePath;
  if (parentState->decl != &shared.getTopLevelDecl()) {
    if (parentState->spec && parentState->spec->isNamespace()) {
      // A plain directory from the import path is a namespace: one dotted
      // name may span several roots, so search every portion visible from
      // this import site rather than the single directory the name first
      // resolved through.
      SmallVector<ModuleSpec> candidates = resolveNamespaceSubModule(
          name, *parentState->spec,
          getSourceMgr().FindBufferContainingLoc(loc));
      if (candidates.size() > 1) {
        std::string paths;
        for (const ModuleSpec &candidate : candidates) {
          if (!paths.empty())
            paths += ", ";
          paths += "'" + candidate.path.string() + "'";
        }
        return notFound("ambiguous import '" + name + "': found " + paths);
      }
      if (!candidates.empty())
        modulePath = std::move(candidates.front());
    } else if (parentState->sourcePath()) {
      modulePath = resolveModulePath(name, *parentState->sourcePath(),
                                     shared.arePrebuiltPackagesDisabled(),
                                     parentState->spec->isSourcePackage());
    } else {
      return notFoundModule();
    }
  } else {
    // Otherwise, go through the normal import path.
    modulePath = resolveModulePath(name, loc);
  }

  if (!modulePath)
    return notFoundModule();

  // A name that previously failed to resolve through this scope now resolves
  // successfully: drop the stale failure record and disable its decl so neither
  // shadows the fresh binding.
  if (parentState->failedImports) {
    auto failedIt = parentState->failedImports->find(declNameAttr);
    if (failedIt != parentState->failedImports->end()) {
      failedIt->second->decl->markDisabled();
      eraseState(failedIt->second->decl);
      parentState->failedImports->erase(failedIt);
    }
  }

  // If the path was a source package, record the import location so the
  // package's __init__ is opened "included from" here.
  if (modulePath->isSourcePackageLike()) {
    return &createPackageState(*modulePath, *parentState, /*importLoc=*/loc);
  }

  const auto &pathRef = modulePath->path;

  // Check if the path is a precompiled file or binary package.
  if (modulePath->isPrecompiled())
    return &createBinaryPackageState(loc, *modulePath, *parentState);

  // Open + lex the module source file.
  assert(modulePath->isSourceModule() && "Unexpected import kind");
  SMLoc openLoc =
      parentState->importLoc.isValid() ? parentState->importLoc : loc;
  const llvm::MemoryBuffer *moduleBuffer =
      shared.openModuleFile(pathRef.string(), openLoc);
  if (!moduleBuffer)
    return notFound("unable to resolve imported module '" + pathRef.string() +
                    "'");
  auto fileLoc = shared.createLocation(moduleBuffer->getBufferIdentifier(),
                                       /*line=*/1, /*column=*/1);
  return &createModuleState(declNameAttr, moduleBuffer, *parentState, fileLoc,
                            *modulePath, /*importLoc=*/loc);
}

ModuleState &
ModuleLoader::importRelativeModuleState(const SharedState::ImportPath &path,
                                        ASTDecl *parentDecl, llvm::SMLoc loc) {
  ASTDecl &importContext = *parentDecl;
  llvm::SMLoc identifierLoc = loc.isValid() ? loc : parentDecl->getLoc();
  // These are structural path failures, not name-binding failures: the record
  // exists for per-site diagnostic dedup and to hand back an erroneous state.
  auto emitError = [&](const Twine &message = "") -> ModuleState & {
    return createErrorModuleState(
        identifierLoc, StringAttr::get(getContext(), path.toDottedString()),
        importContext, message, /*unlisted=*/true);
  };

  auto adjustIdentifierLoc = [&](unsigned offset) {
    if (!identifierLoc.isValid())
      return identifierLoc;
    return llvm::SMLoc::getFromPointer(identifierLoc.getPointer() + offset);
  };

  bool isRelative = path.relativeLevel > 0;
  if (!isRelative) {
    // We're resolving relative to a top-level package.
    assert(!path.components.empty() && "Importing empty path?");
    StringRef parentName = path.components.front();
    identifierLoc = adjustIdentifierLoc(parentName.size() + 1);
    parentDecl =
        importModuleState({parentName}, &shared.getTopLevelDecl(), loc).decl;
  } else {
    auto relativeLevel = path.relativeLevel;
    // Find the current package.
    identifierLoc = adjustIdentifierLoc(1);
    while (!isa_and_nonnull<PackageOp>(parentDecl->getIfOperation()) &&
           parentDecl->parentDecl)
      parentDecl = parentDecl->parentDecl;
    if (!isa_and_nonnull<PackageOp>(parentDecl->getIfOperation()))
      return emitError("cannot import relative to a top-level package");

    // Otherwise, this is a package relative to the current parent.
    while (--relativeLevel) {
      identifierLoc = adjustIdentifierLoc(1);
      if (!parentDecl->parentDecl ||
          !isa_and_nonnull<PackageOp>(
              parentDecl->parentDecl->getIfOperation())) {
        return emitError(
            "attempted relative import with no known parent package");
      }
      parentDecl = parentDecl->parentDecl;
    }

    // If the path itself is empty, we're grabbing the parent package.
    if (path.components.empty())
      return *lookupState(parentDecl);
  }

  // The rest of the path resolves a nested module or package from the current
  // parent. Use importSubModuleState for each segment, which checks
  // nestedModules first. The non-relative branch above has already consumed
  // the leading component as the top-level package.
  unsigned consumedComponents = isRelative ? 0 : 1;
  SmallVector<StringRef> remainingNames{
      path.components.begin() + consumedComponents, path.components.end()};
  StringRef leafModule = remainingNames.pop_back_val();
  for (auto [i, parentName] : enumerate(remainingNames)) {
    ModuleState &nextState =
        importSubModuleState(parentName, parentDecl, loc, identifierLoc);
    parentDecl = nextState.decl;

    // If we've recursed through a package, all is well; continue.
    if (isa_and_nonnull<PackageOp>(parentDecl->getIfOperation())) {
      identifierLoc = adjustIdentifierLoc(parentName.size() + 1);
      continue;
    }

    // Otherwise we've hit an error case.

    // We've found a *module* - not a package. We can't recurse any further. The
    // user has probably written one of the following:
    //   - import package.(module)+(.symbol)?
    //   - from package.(module)+(.symbol)? import other_symbol
    if (isa_and_nonnull<FileModuleOp>(parentDecl->getIfOperation())) {
      auto child =
          i + 1 < remainingNames.size() ? remainingNames[i + 1] : leafModule;
      return emitError(
          "'" + parentName +
          "' is a module, not a package; it has no nested module or package '" +
          child + "'");
    }

    // Otherwise the user has done something we can't recognise.
    return emitError("'" + parentName + "' does not refer to a nested package");
  }

  return importSubModuleState(leafModule, parentDecl, loc, identifierLoc);
}

bool ModuleLoader::hasNestedModule(PackageOp packageOp, StringRef name) const {
  ModuleState *packageState = lookupPackageState(packageOp);
  if (!packageState)
    return false;
  return packageState->nestedModules.count(
             StringAttr::get(getContext(), name)) > 0;
}

SmallVector<ASTDecl *>
ModuleLoader::getNestedModuleDecls(PackageOp packageOp) const {
  ModuleState *state = lookupPackageState(packageOp);
  if (!state)
    return {};
  SmallVector<std::pair<StringRef, ASTDecl *>> named;
  named.reserve(state->nestedModules.size());
  for (auto &[name, sub] : state->nestedModules)
    named.emplace_back(name.getValue(), sub->decl);
  llvm::sort(named,
             [](const auto &a, const auto &b) { return a.first < b.first; });
  SmallVector<ASTDecl *> result;
  result.reserve(named.size());
  for (auto &[name, decl] : named)
    result.push_back(decl);
  return result;
}

ASTDecl *ModuleLoader::tryImportSubModule(ASTDecl &parent, StringRef name,
                                          llvm::SMLoc loc) {
  // A submodule lives under a real package/module (never the synthetic
  // top-level scope) that has a module state.
  if (&parent == &shared.getTopLevelDecl() || !lookupState(&parent))
    return nullptr;
  // emitErrors=false: returns null (no diagnostic) when `name` is not a
  // submodule - that is the "this is a plain symbol, not a submodule" case the
  // caller falls back from. A genuine error (e.g. a parse failure in a
  // submodule that does exist) is still reported.
  ModuleState *state = importSubModuleStateImpl(name, &parent, loc, loc,
                                                /*emitErrors=*/false);
  return state ? state->decl : nullptr;
}

void ModuleLoader::registerSourcePackageChildren(ASTDecl &packageDecl) {
  ModuleState *parentState = lookupState(&packageDecl);
  if (!parentState || !parentState->sourcePath())
    return;
  // A namespace parent enumerates the union of all its portions; other
  // parents enumerate their single directory.
  SmallVector<std::string> directories;
  if (!parentState->spec->isNamespace()) {
    directories.push_back(*parentState->sourcePath());
  } else {
    unsigned bufferId =
        getSourceMgr().FindBufferContainingLoc(parentState->importLoc);
    directories = collectNamespacePortions(*parentState->spec, bufferId);
  }

  // Collect the directory entries and sort them, so children are registered in
  // a deterministic order across platforms. In-directory precedence applies
  // first, so cross-portion merging sees one candidate per name per portion:
  // portions merge, a closed candidate beats portions, and two closed
  // candidates from different portions are ambiguous; such a name is not
  // registered, and a direct import of it reports the ambiguity.
  std::map<std::string, ModuleSpec> packageChildren;
  llvm::StringSet<> ambiguousChildren;
  for (const std::string &directory : directories) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec) || ec)
      continue;
    std::map<std::string, ModuleSpec> portionChildren;
    for (const auto &entry :
         std::filesystem::directory_iterator(directory, ec)) {
      auto moduleSpec = ModuleSpec::classify(entry.path());
      if (!moduleSpec)
        continue;
      // Precompiled children aren't supported in source packages.
      if ((shared.arePrebuiltPackagesDisabled() ||
           parentState->spec->isSourcePackage()) &&
          moduleSpec->isPrecompiled())
        continue;
      if (auto it = portionChildren.find(moduleSpec->name);
          it == portionChildren.end() ||
          moduleSpec->takesImportPrecedence(it->second)) {
        portionChildren[moduleSpec->name] = *moduleSpec;
      }
    }
    for (auto &[childName, childSpec] : portionChildren) {
      auto it = packageChildren.find(childName);
      if (it == packageChildren.end()) {
        packageChildren[childName] = childSpec;
        continue;
      }
      bool haveDir = it->second.kind == ModuleSpec::Kind::SourceDir;
      bool newDir = childSpec.kind == ModuleSpec::Kind::SourceDir;
      // Portions merge; the first portion remains the (advisory) home.
      if (haveDir && newDir)
        continue;
      // A closed candidate beats portion directories in either order.
      if (haveDir != newDir) {
        if (haveDir)
          it->second = childSpec;
        continue;
      }
      ambiguousChildren.insert(childName);
    }
  }
  for (const auto &ambiguous : ambiguousChildren)
    packageChildren.erase(ambiguous.getKey().str());

  // A directory child of a namespace is itself a namespace: tag it with its
  // component chain so its own submodules resolve across portions.
  if (parentState->spec->isNamespace()) {
    for (auto &[childName, childSpec] : packageChildren) {
      if (childSpec.kind != ModuleSpec::Kind::SourceDir)
        continue;
      childSpec.namespaceComponents = parentState->spec->namespaceComponents;
      childSpec.namespaceComponents.push_back(childName);
    }
  }

  for (const auto &[name, value] : packageChildren) {
    // The package's own __init__ is resolved separately by resolveBody.
    if (name == "__init__")
      continue;
    // Skip names already registered (e.g. a sibling imported while resolving
    // __init__, or __init__ itself).
    auto declNameAttr = StringAttr::get(getContext(), value.name);
    if (parentState->nestedModules.count(declNameAttr))
      continue;
    // A child some other name already binds would be bound here as an alias. We
    // don't want to pre-emptively warn when we haven't yet explicitly been
    // asked to import this child. Skip it for now, we'll create the state if
    // and when we're required.
    if (wouldAliasExistingBinding(value, value.name, *parentState->decl))
      continue;
    if (value.isSourcePackageLike()) {
      // Registered by the directory scan so it gets no import location here.
      // We'll resolve that location if/when it's actually resolved.
      createPackageState(value, *parentState, /*importLoc=*/{});
    } else if (value.isPrecompiled()) {
      // NB: We don't call createBinaryPackageState here because it will eagerly
      // load the bytecode (and potentially even throw errors to our unknown
      // import location!). We instead skip registration and wait for the user
      // to actually import it, at which point we'll hit the file system and
      // load the bytecode module.
      // The tradeoff is that nothing will enumerate these precompiled children
      // (e.g., the LSP) until it's actually imported.
    } else {
      createDeferredModuleState(value, *parentState);
    }
  }
}

//===----------------------------------------------------------------------===//
// Loading
//===----------------------------------------------------------------------===//

ModuleState &ModuleLoader::createFileModuleState(
    ModuleState &parentState, FileLineColLoc loc, llvm::SMLoc declLoc,
    LexerCursor cursor, LexerCursor endCursor, const ModuleSpec &spec,
    const BindingSpec &binding) {
  assert(!binding.aliasOf && "an aliased binding creates no module");
  StringAttr declName = binding.name;

  auto moduleBuilder = parentState.decl->getDeclEndBuilder();
  Operation *fileOp = FileModuleOp::create(moduleBuilder, loc, declName);
  // Use createUnlistedDecl (not addDecl) so the module is NOT added to
  // parentState.decl->declsInScope. This prevents "leaky imports"; the module
  // stays navigable via ModuleState::nestedModules.
  ASTDecl &moduleDecl = shared.declResolver->createUnlistedDecl(
      fileOp, declLoc, parentState.decl, cursor, endCursor, /*indentation=*/-1);
  shared.declResolver->registerDeclSymbol(&moduleDecl);

  ModuleState &moduleState = parentState.insertNestedModule(
      declName, std::make_unique<ModuleState>(&moduleDecl, spec));
  setState(moduleDecl, moduleState, &binding);
  return moduleState;
}

ModuleState &
ModuleLoader::createModuleState(StringAttr declName,
                                const llvm::MemoryBuffer *moduleBuffer,
                                ModuleState &parentState, FileLineColLoc loc,
                                const ModuleSpec &spec, SMLoc importLoc) {
  // An eagerly-opened module: its cursor points at the freshly-lexed buffer.
  Lexer lexer(shared.diags, moduleBuffer);
  SMLoc declLoc = lexer.getToken().getLoc();
  BindingSpec binding =
      resolveModuleBinding(spec, declName, parentState, importLoc);
  if (binding.aliasOf)
    return *binding.aliasOf;

  ModuleState &moduleState =
      createFileModuleState(parentState, loc, declLoc, lexer.getCursor(),
                            LexerCursor::getEOF(moduleBuffer), spec, binding);
  // An erroneous state carries no module body, so nothing below applies to it.
  if (moduleState.decl->isErroneous())
    return moduleState;

  // Auto-import the core language modules.
  if (LLVM_LIKELY(shared.hasBuiltinModule()))
    shared.importBuiltinModules(*moduleState.decl);
  shared.notifyListenerOnModuleDecl(*moduleState.decl,
                                    moduleState.decl->getLoc());
  return moduleState;
}

ModuleState &ModuleLoader::createDeferredModuleState(ModuleSpec moduleSpec,
                                                     ModuleState &parentState) {
  // A deferred module: the FileModuleOp + decl exist but its file is NOT
  // opened. The decl carries an invalid cursor; it is opened + lexed on first
  // body resolution, at which point materializeDeferredModule sets its real
  // location.
  assert(moduleSpec.isSourceModule() && "Invalid module state");
  auto declNameAttr = StringAttr::get(shared.getContext(), moduleSpec.name);
  FileLineColLoc loc =
      shared.createLocation(moduleSpec.path.string(), /*line=*/1, /*column=*/1);
  // Scan-created: nothing imported this, so an alias would go unreported. The
  // scan skips children that would alias, so this never has one to make.
  BindingSpec binding = resolveModuleBinding(moduleSpec, declNameAttr,
                                             parentState, /*importLoc=*/{});
  if (binding.aliasOf)
    return *binding.aliasOf;

  return createFileModuleState(parentState, loc, /*declLoc=*/SMLoc(),
                               /*cursor=*/LexerCursor(),
                               /*endCursor=*/LexerCursor(), moduleSpec,
                               binding);
}

ModuleState &ModuleLoader::createPackageState(ModuleSpec moduleSpec,
                                              ModuleState &parentState,
                                              SMLoc importLoc) {
  StringAttr declName = StringAttr::get(shared.getContext(), moduleSpec.name);
  // Create a new decl for this module. We use createUnlistedDecl instead of
  // addDecl so the package is NOT added to parentState.decl->declsInScope.
  // This prevents "leaky imports" where importing a sub-module makes the
  // parent package globally accessible. The package is still navigable via
  // ModuleState::nestedModules (populated by insertNestedModule below).
  assert(moduleSpec.isSourcePackageLike() && "Invalid package kind");

  // The same directory bound under a second name is one package under two
  // names. A namespace gets no origin, so it is exempt without a kind check
  // here and never aliases.
  BindingSpec binding =
      resolveModuleBinding(moduleSpec, declName, parentState, importLoc);
  if (binding.aliasOf)
    return *binding.aliasOf;

  auto loc = shared.createLocation((moduleSpec.isSourcePackage()
                                        ? moduleSpec.path / "__init__.mojo"
                                        : moduleSpec.path)
                                       .string(),
                                   /*line=*/1, /*column=*/1);
  auto moduleBuilder = parentState.decl->getDeclEndBuilder();
  auto packageOp = PackageOp::create(moduleBuilder, loc, declName);
  // Note we intentionally don't set a valid 'loc' here. The real loc is set
  // if/when the module is actually opened on demand.
  ASTDecl &decl = shared.declResolver->createUnlistedDecl(
      static_cast<Operation *>(packageOp), /*loc=*/SMLoc(), parentState.decl,
      parentState.decl->getCursor(), parentState.decl->getCursor(),
      /*indentation=*/-1);
  // Register the symbol so ModuleType::getDecl() works.
  shared.declResolver->registerDeclSymbol(&decl);

  // Insert the newly created module state.
  ModuleState &moduleState = parentState.insertNestedModule(
      declName, std::make_unique<ModuleState>(&decl, moduleSpec));
  moduleState.importLoc = importLoc;
  // The binding carries no origin for a namespace, which owns no single entity.
  setState(decl, moduleState, &binding);
  setPackageState(packageOp, moduleState);

  return moduleState;
}

ModuleState &ModuleLoader::createBinaryPackageState(SMLoc loc,
                                                    const ModuleSpec &spec,
                                                    ModuleState &parentState) {
  std::string pathStr = spec.path.string();
  auto declNameAttr = StringAttr::get(shared.getContext(), spec.name);
  auto makeError = [&](const Twine &msg) -> ModuleState & {
    return createErrorModuleState(loc, declNameAttr, *parentState.decl, msg);
  };

  // Symbol references recorded in the artifact are rooted at its compiled
  // name, which resolves only for a top-level binding of that name; mounted
  // below the top level, every type escaping the package is unresolvable.
  // TODO(MOCO-4487): lift this once loading re-anchors recorded roots to the
  // mount point.
  if (parentState.decl != &shared.getTopLevelDecl()) {
    return makeError("precompiled package '" + pathStr +
                     "' must be imported directly from an import root, not "
                     "as '" +
                     mountPathFor(spec.name, *parentState.decl) + "'");
  }

  // One artifact bound under two names is one package under two names. Aliasing
  // before the load also means the second name costs no second read.
  BindingSpec binding =
      resolveModuleBinding(spec, declNameAttr, parentState, /*importLoc=*/loc);
  if (binding.aliasOf)
    return *binding.aliasOf;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> packageBuffer =
      llvm::MemoryBuffer::getFile(pathStr);
  if (!packageBuffer)
    return makeError("unable to open package file '" + pathStr + "'");

  // Read the cached package.
  OpBuilder builder = parentState.decl->getDeclEndBuilder();
  Block *block = builder.getBlock();
  // bytecodeReader refers to sourceMgr by reference,
  // so sourceMgr lifetime must be same or longer.
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  std::unique_ptr<mlir::BytecodeReader> bytecodeReader;
  {
    CompilerTimeTraceScope timeScope("readBytecodeFile");
    // Create a source manager to extend the lifetime of the package buffer.
    sourceMgr->AddNewSourceBuffer(std::move(*packageBuffer), SMLoc());
    const llvm::MemoryBuffer *memoryBuf =
        sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID());

    auto mlirBufOrErr = getMLIRBufferFromPrecompiledFile(
        *memoryBuf, shared.options.ignoreIncompatiblePrecompiledFileErrors);
    if (mlirBufOrErr.isError())
      return makeError(mlirBufOrErr.takeError().get());
    auto mlirResult = std::move(*mlirBufOrErr);
    // If the package was compressed, add the decompressed buffer to the source
    // manager to extend its lifetime beyond this scope.
    if (mlirResult.ownedData)
      sourceMgr->AddNewSourceBuffer(std::move(mlirResult.ownedData), SMLoc());

    // TODO(MOCO-522): Arcana docs on this lazy loading.
    bytecodeReader = std::make_unique<mlir::BytecodeReader>(
        mlirResult.buffer, shared.getBytecodeParserConfig(),
        /*lazyLoad=*/true, sourceMgr);

    // Read in the cached bytecode.
    if (failed(bytecodeReader->readTopLevel(block)))
      return makeError("unable to load package '" + pathStr + "'");

    // Add the package path to the set of included files.
    shared.addIncludedFile(pathStr);
  }

  // The bytecode module includes the package module and any function stubs.
  auto tmpModule = cast<ModuleOp>(block->back());
  if (failed(bytecodeReader->materialize(tmpModule)))
    return makeError("failed to materialize top-level module");

  // Move the package into the current decl.
  auto packageOp = cast<PackageOp>(tmpModule.getBody()->front());
  packageOp->remove();
  builder.insert(packageOp);

  // Process each of the stubs, deduplicating each of them into the shared
  // state. For any added thunks, we have to register a decl for them.
  auto theModule =
      cast_or_null<ModuleOp>(shared.getTopLevelDecl().getIfOperation());
  for (auto thunk : llvm::make_early_inc_range(tmpModule.getOps<FnOp>())) {
    Attribute key = thunk.getThunkKeyAttr();
    assert(key && "thunk is missing its key");
    if (!shared.tryRegisterConversionThunk(key, thunk))
      continue; // thunk already exists

    // Move the thunk into the top-level and add it as fully resolved.
    if (failed(bytecodeReader->materialize(thunk)))
      return makeError("failed to materialize function thunk");
    thunk->remove();
    theModule.push_back(thunk);
    ASTDecl &thunkDecl = shared.declResolver->addBytecodeDecl(
        &*thunk, thunk.getSourceNameAttr(), &shared.getTopLevelDecl(),
        DeclResolvedness::body);
    shared.declResolver->finalizeFuncSignature(thunk, thunkDecl);
  }
  for (auto trait :
       llvm::make_early_inc_range(tmpModule.getOps<TraitDeclOp>())) {
    if (!trait.getClosureSignature().has_value())
      continue;

    FnTypeGeneratorType key = *trait.getClosureSignature();
    auto creation = [&]() -> ASTDecl * {
      if (failed(bytecodeReader->materialize(trait)))
        return nullptr;
      // A closure trait with no methods is a stub from a package that
      // references but does not define the closure type. Skip it so the cache
      // slot stays empty and a later package with the full body can fill it.
      if (trait.getOps<FnOp>().empty())
        return nullptr;
      trait->remove();
      theModule.push_back(trait);
      ASTDecl &traitDecl = shared.declResolver->addBytecodeDecl(
          &*trait, trait.getSymNameAttr(), &shared.getTopLevelDecl(),
          DeclResolvedness::body);
      traitDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForTrait(trait));
      // Ensure that the trait's methods are registered, too.
      for (auto fn : trait.getOps<FnOp>()) {
        shared.declResolver->addBytecodeDecl(
            fn, fn.getSourceNameAttr(), &traitDecl, DeclResolvedness::body);
      }
      return &traitDecl;
    };
    shared.getClosureEmitter().getOrCreateClosureTrait(key, creation);
  }
  // Insert a new module decl. Use createUnlistedDecl instead of addBytecodeDecl
  // so the package is NOT added to parentState.decl->declsInScope.
  ASTDecl &decl = shared.declResolver->createUnlistedDecl(
      static_cast<Operation *>(packageOp),
      shared.diags.convertLocToSMLoc(packageOp->getLoc()), parentState.decl,
      LexerCursor(), LexerCursor(), /*indentation=*/-1);
  decl.loadedFromBytecode = true;
  decl.resolvedness = DeclResolvedness::signature;
  shared.declResolver->registerDeclSymbol(&decl);

  // Initialize the module state.
  ModuleState &moduleState = parentState.insertNestedModule(
      declNameAttr, std::make_unique<ModuleState>(&decl, spec));
  // Remember where this package was imported. The package's source files are
  // only opened at diagnostic time (they aren't parsed here), so when a decl
  // from this package is lazily materialized we use this to set its location
  // at the import site.
  moduleState.importLoc = loc;
  setState(decl, moduleState, &binding);
  setPackageState(cast_or_null<PackageOp>(decl.getIfOperation()), moduleState);

  // The reader and the buffers under it belong to the file, so every module
  // bound out of this artifact reaches them through the shared origin. The
  // module cache means an artifact is only ever read once.
  assert(moduleState.origin && "precompiled artifact without an origin");
  ModuleOrigin &origin = *moduleState.origin;
  assert(!origin.bytecodeReader && "artifact read twice");
  origin.bytecodeReader = std::move(bytecodeReader);
  // keep buffer alive for deferred materialize
  origin.sourceMgr = sourceMgr;
  origin.tmpModule = tmpModule;
  origin.bytecodeImportLoc = loc;

  return moduleState;
}

ModuleState &ModuleLoader::createErrorModuleState(SMLoc loc, StringAttr name,
                                                  ASTDecl &errorContext,
                                                  const Twine &errorMsg,
                                                  bool unlisted,
                                                  const Twine &note) {
  // Track the failure in the scope whose lookup failed.
  ModuleState *contextState = lookupState(&errorContext);
  if (!contextState)
    contextState = &getTopLevelState();

  if (!contextState->failedImports) {
    contextState->failedImports.reset(
        new DenseMap<StringAttr, std::unique_ptr<ModuleState>>());
  }
  std::unique_ptr<ModuleState> &state = (*contextState->failedImports)[name];
  if (!state) {
    ASTDecl *decl = &shared.declResolver->addErroneousDecl(
        name, loc, &errorContext, unlisted);
    state = std::make_unique<ModuleState>(decl);
    setState(*decl, *state);
  }

  // Report errors once per import site. This data is lazily allocated.
  if (!state->reportedFailureLocs)
    state->reportedFailureLocs.reset(new SmallVector<SMLoc>());
  if (!llvm::is_contained(*state->reportedFailureLocs, loc)) {
    state->reportedFailureLocs->push_back(loc);
    MojoInflightDiag diag = shared.emitError(loc, errorMsg);
    if (!note.isTriviallyEmpty())
      diag.attachNote(loc) << note;
  }
  return *state;
}
