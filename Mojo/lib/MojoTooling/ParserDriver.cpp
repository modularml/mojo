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
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/DocString.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoParser/Lexer.h"
#include "Mojo/MojoParser/ModuleLoader.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/CompilationOptions.h"

#include "ParserDriverImpl.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/Filesystem/Paths.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SourceMgr.h"

#include <filesystem>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// MojoParserContext::Impl
//===----------------------------------------------------------------------===//

MojoParserContext::Impl::Impl(llvm::SourceMgr &sourceMgr, ParserConfig &config)
    : sharedState(sourceMgr, config), replLocMapper(sourceMgr) {
  // Create the top-level outer decl, which will contain all things we parse.
  module = ModuleOp::create(UnknownLoc::get(sharedState.getContext()));
  topLevelDecl = &sharedState.declResolver->addDecl(
      *module, SMLoc(), StringAttr(), /*parentDecl=*/nullptr, LexerCursor(),
      LexerCursor(), /*indentation=*/-1);
  sharedState.initialize(*topLevelDecl);
}

//===----------------------------------------------------------------------===//
// MojoParserContext
//===----------------------------------------------------------------------===//

MojoParserContext::MojoParserContext(SourceMgr &sourceMgr, ParserConfig &config)
    : impl(std::make_unique<Impl>(sourceMgr, config)) {}

MojoParserContext::~MojoParserContext() {
  // Finalize any imported bytecode now that we've resolved everything. This
  // avoids dangling references to operations from the bytecode.
  (void)impl->sharedState.finalizeImportedBytecodeModules();
}

ModuleOp MojoParserContext::getModule() {
  return cast<ModuleOp>(impl->topLevelDecl->getIfOperation());
}

llvm::SourceMgr &MojoParserContext::getSourceMgr() {
  return impl->sharedState.getSourceMgr();
}

SharedState &MojoParserContext::getSharedState() { return impl->sharedState; }

std::vector<std::string>
MojoParserContext::getModuleSearchDirectories(unsigned fileId) {
  std::vector<std::string> searchDirs;
  impl->sharedState.getModuleLoader().traverseImportDirectories(
      fileId, [&](StringRef dir) {
        searchDirs.push_back(dir.str());
        return WalkResult::advance();
      });
  return searchDirs;
}

const KGEN::CompilationOptions &MojoParserContext::getCompilationOptions() {
  return impl->sharedState.options;
}

MojoASTDeclRef MojoParserContext::getDecl(MojoASTTypeRef type) {
  return type.getDecl(impl->sharedState);
}

MojoASTDeclRef
MojoParserContext::getTraitDecl(KGEN::TraitSymbolAttr traitSymbol) {
  return MojoASTDeclRef(&impl->sharedState.declResolver->getDeclForTypeSymbol(
      traitSymbol.getSymbol()));
}

MojoASTTypeRef MojoParserContext::concretizeType(MojoASTTypeRef base,
                                                 ArrayRef<TypedAttr> params,
                                                 MojoASTTypeRef type) {
  KGEN::ParameterEvaluator evaluator(
      cast<StructDeclOp>(base.getDecl(getSharedState()).decl->getIfOperation())
          .getInputParams(),
      params);

  return evaluator.replace(evaluator.getReboundType(type.getMLIRType()));
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Return the package name to use for the given Mojo source package directory.
/// A directory's whole name is the package name: any periods in it belong to
/// the name itself (importable via an escaped identifier), not an extension.
static std::string getNameForSourcePackage(const std::filesystem::path &path) {
  // FIXME: This is kind of a huge hack, but works around the fact that the
  // Mojo standard library was open sourced with a different package name than
  // the directory, and our tools aren't ready to support that yet. Until we can
  // properly support this case, special case the common project layout of
  // having a `src` directory (that contains the package), under a directory
  // with the package name.
  std::string name = path.filename().string();
  if (name == "src")
    return path.parent_path().filename().string();
  return name;
}

/// Import a module or package that is nested within a source package.
static ASTDecl *buildNestedModuleDecl(std::filesystem::path filepath,
                                      SharedState &sharedState) {
  // Collect all of the sub-package names and find the outer most package.
  // Only a file strips its extension to form its module name; a directory's
  // whole name is the name, periods included.
  SmallVector<std::string, 4> packageNames;
  std::error_code ec;
  while (Filesystem::isMojoSourcePackagePath(filepath.parent_path())) {
    bool isDir = std::filesystem::is_directory(filepath, ec) && !ec;
    packageNames.emplace_back(isDir ? filepath.filename().string()
                                    : filepath.stem().string());
    filepath = filepath.parent_path();
  }

  // Create the package using the outermost name.
  ASTDecl &packageDecl = sharedState.createPackage(
      filepath.string(), getNameForSourcePackage(filepath));

  // Import the file from within the package.
  std::reverse(packageNames.begin(), packageNames.end());
  SharedState::ImportPath path;
  path.relativeLevel = 1;
  for (const std::string &name : packageNames)
    path.components.push_back(name);
  return &sharedState.importModule(
      path, cast<PackageOp>(packageDecl.getIfOperation()), SMLoc());
}

/// Create an ASTDecl for the given file module.
static ASTDecl *buildModuleDecl(const std::filesystem::path &filepath,
                                const llvm::MemoryBuffer *sourceBuf,
                                SharedState &sharedState) {
  // If the file is within a package, we create a decl for the outermost package
  // and import this decl from there. This ensures we process relative imports
  // and other package-level constructs correctly.
  if (Filesystem::isMojoSourcePackagePath(filepath.parent_path()))
    return buildNestedModuleDecl(filepath, sharedState);

  // Otherwise, create a decl specifically for the module.
  auto fileLoc =
      FileLineColLoc::get(sharedState.getContext(), filepath.string(),
                          /*line=*/1, /*column=*/1);
  return &sharedState.createModule(filepath.stem().string(), sourceBuf,
                                   fileLoc);
}

/// Create an ASTDecl for the given package.
static ASTDecl *buildPackageDecl(const std::filesystem::path &filepath,
                                 SharedState &sharedState) {
  // If the file is a binary package, just import it.
  if (Filesystem::isMojoBinaryPackagePath(filepath)) {
    return &sharedState.createBinaryPackage(filepath.string(),
                                            filepath.stem().string());
  }
  // If this isn't a source package, bail out.
  if (!Filesystem::isMojoSourcePackagePath(filepath))
    return nullptr;

  // If the file is within a package, we create a decl for the outermost package
  // and import this decl from there. This ensures we process relative imports
  // and other package-level constructs correctly.
  if (Filesystem::isMojoSourcePackagePath(filepath.parent_path()))
    return buildNestedModuleDecl(filepath, sharedState);

  // Otherwise, create a new package.
  return &sharedState.createPackage(filepath.string(),
                                    getNameForSourcePackage(filepath));
}

/// Create an ASTDecl for the given module or package.
static ASTDecl *buildModuleOrPackageDecl(const std::filesystem::path &path,
                                         SharedState &sharedState) {
  // Handle the case of a file.
  if (path.extension() == ".mojo") {
    SourceMgr &sourceMgr = sharedState.getSourceMgr();
    std::string fullPath;
    int fileId = sourceMgr.AddIncludeFile(path, SMLoc(), fullPath);
    if (!fileId)
      return nullptr;
    return buildModuleDecl(path, sourceMgr.getMemoryBuffer(fileId),
                           sharedState);
  }
  return buildPackageDecl(path, sharedState);
}

MojoASTDeclRef MojoParserContext::parseFile(unsigned fileId,
                                            bool eraseUnparsedDecls) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();

  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(fileId);
  StringRef filepathStr = sourceBuf->getBufferIdentifier();
  std::filesystem::path filepath(filepathStr.str());

  ASTDecl *moduleDecl = buildModuleDecl(filepath, sourceBuf, impl->sharedState);
  impl->sharedState.declResolver->resolveAllReferencedFrom(*moduleDecl,
                                                           eraseUnparsedDecls);

  return MojoASTDeclRef(moduleDecl);
}

/// Resolves an unparsed decl enough for the language server to operate. We
/// fully parse everything descended from the root decl, and leave everything
/// else unparsed.
void M::resolveForLSP(DeclResolver &resolver, ASTDecl &decl) {
  CompilerTimeTraceScope traceScope("resolveForLSP", [&] {
    return decl.getUserNameIfOperation().value_or("").str();
  });

  // We need to resolve decls in a breadth-first style, hence why we use a queue
  // here instead of implementing this using recursion.
  std::deque<ASTDecl *> worklist({&decl});
  while (!worklist.empty()) {
    ASTDecl *declIt = worklist.back();
    worklist.pop_back();

    // Resolve the decl. We don't care if this fails - a diagnostic will be
    // reported as well, and that's all we want.
    (void)resolver.resolveBody(*declIt, declIt->getLoc());

    // When validating doc strings, we wish to only validate those defined on
    // decl in the main container. As this point the main container decl has
    // been fully resolved, so it's an opportune time to validate.
    validateDocString(*declIt);

    // Traverse the children of this decl.
    for (auto &[_, decls] : declIt->getDeclsInScope()) {
      for (ASTDecl *decl : decls)
        if (decl->getParentDecl() == declIt)
          worklist.push_front(decl);
    }
  }
}

MojoASTDeclRef MojoParserContext::parseFileForLSP(unsigned fileId) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();

  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(fileId);
  StringRef filepathStr = sourceBuf->getBufferIdentifier();
  std::filesystem::path filepath(filepathStr.str());

  ASTDecl *moduleDecl = buildModuleDecl(filepath, sourceBuf, impl->sharedState);
  resolveForLSP(*impl->sharedState.declResolver, *moduleDecl);
  resolveSignaturesForLSP(*impl->sharedState.declResolver);

  return MojoASTDeclRef(moduleDecl);
}

void M::resolveSignaturesForLSP(DeclResolver &resolver) {
  size_t i = 0;
  while (i != resolver.getParsedDeclList().size()) {
    ASTDecl *parsedDecl = resolver.getParsedDeclList()[i++];

    // Skip named imports (UnresolvedImportOp) that haven't been resolved yet,
    // regardless of whether they originate from source or bytecode. These are
    // resolved lazily on first use, so eagerly resolving them here would pull
    // in all their transitive imports — which can be very expensive (e.g.
    // _unicode_lookups.mojo with 17 large comptime tables). Once a named import
    // has been lazily resolved, its resolvedness is promoted to `body` by
    // aliasDeclsImpl, so it no longer matches this check and will be handled
    // normally.
    //
    // Wildcard imports cannot be skipped: they may contribute new names to the
    // scope's overload sets, and those names must be visible before any
    // subsequent signature resolution inspects the scope.
    if (parsedDecl->resolvedness == DeclResolvedness::unparsed &&
        isa_and_nonnull<LIT::UnresolvedImportOp>(parsedDecl->getIfOperation()))
      continue;

    if (parsedDecl->isLoadedFromBytecode()) {
      // For function ops, resolveSignature suffices for LSP. FnOp bodies are
      // already materialized in the IR (they are not IsolatedFromAbove, so they
      // are parsed eagerly when their parent container is materialized). We
      // skip resolveBody to avoid walking the function body and eagerly
      // resolving transitive symbol references (e.g. large unicode lookup
      // tables accessed through _unicode.mojo's helper functions).
      //
      // All other bytecode decls (container ops) still require resolveBody.
      // It isn't safe to leave them in a partially-resolved state: container
      // ops with a SymbolTable trait (e.g. FileModuleOp) must have at least
      // one Block in their body region — an LLVM invariant — but signature-
      // only resolution leaves the body empty. resolveBody materializes the
      // body from bytecode, satisfying the invariant and making the op's
      // members visible to LSP queries.
      if (isa_and_nonnull<LIT::FnOp>(parsedDecl->getIfOperation()))
        (void)resolver.resolveSignature(*parsedDecl, parsedDecl->getLoc());
      else
        (void)resolver.resolveBody(*parsedDecl, parsedDecl->getLoc());
    } else {
      (void)resolver.resolveSignature(*parsedDecl, parsedDecl->getLoc());
    }
  }
}

void MojoParserContext::ensureSignaturesResolved() {
  resolveSignaturesForLSP(*impl->sharedState.declResolver);
}

MojoASTDeclRef
MojoParserContext::parsePackage(const std::filesystem::path &path) {
  // Check that the path is actually a package.
  if (!(Filesystem::isMojoSourcePackagePath(path) ||
        Filesystem::isMojoBinaryPackagePath(path)))
    return nullptr;
  return parseFileOrPackage(path);
}

MojoASTDeclRef
MojoParserContext::parseFileOrPackage(const std::filesystem::path &path) {
  ASTDecl *moduleDecl = buildModuleOrPackageDecl(path, impl->sharedState);
  if (!moduleDecl)
    return nullptr;
  impl->sharedState.declResolver->resolveAllReferencedFrom(*moduleDecl);

  return MojoASTDeclRef(moduleDecl);
}

MojoASTDeclRef MojoParserContext::parseFileOrPackageNonRecursive(
    const std::filesystem::path &path) {
  ASTDecl *moduleDecl = buildModuleOrPackageDecl(path, impl->sharedState);
  if (!moduleDecl)
    return nullptr;

  // Resolve just the top-level decl.
  (void)impl->sharedState.declResolver->resolveBody(*moduleDecl, SMLoc());
  return MojoASTDeclRef(moduleDecl);
}

MojoASTDeclRef MojoParserContext::parseIsolatedFileOrPackage(
    const std::filesystem::path &path) {
  ASTDecl *moduleDecl = buildModuleOrPackageDecl(path, impl->sharedState);
  if (!moduleDecl) {
    return nullptr;
  }

  bool isPackage = isa<PackageOp>(moduleDecl->getIfOperation());

  std::deque<ASTDecl *> worklist({moduleDecl});
  while (!worklist.empty()) {
    auto decl = worklist.back();
    worklist.pop_back();

    if (isPackage) {
      if (auto parent = decl->getNearestDeclOfType<PackageOp>();
          parent != moduleDecl)
        continue;
    } else {
      if (auto parent = decl->getNearestDeclOfType<FileModuleOp>();
          parent != moduleDecl) {
        continue;
      }
    }

    (void)impl->sharedState.declResolver->resolveBody(*decl, SMLoc());
    for (auto &[_, decls] : decl->getDeclsInScope()) {
      for (ASTDecl *childDecl : decls) {
        if (childDecl->getParentDecl() == decl) {
          worklist.push_front(childDecl);
        }
      }
    }
  }

  return MojoASTDeclRef(moduleDecl);
}

bool MojoParserContext::wasErrorEmitted() const {
  return impl->sharedState.diags.isErrorEmitted();
}
