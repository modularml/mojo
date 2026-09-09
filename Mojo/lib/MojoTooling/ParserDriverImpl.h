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

#ifndef PARSERDRIVERIMPL_H
#define PARSERDRIVERIMPL_H

#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoTooling/ParserDriver.h"

namespace M {

/// Resolves an unparsed decl enough for the language server to operate. Fully
/// body-resolves all decls descended from the root, validates their doc
/// strings, and leaves everything else unparsed.
void resolveForLSP(KGEN::LIT::DeclResolver &resolver, KGEN::LIT::ASTDecl &decl);

/// Signature-resolves all parsed decls that are not yet resolved, skipping
/// lazy named imports and body-resolving non-FnOp bytecode decls. This is the
/// second stage of LSP resolution: after resolveForLSP covers the direct
/// children of a container, this step covers transitive dependencies that were
/// pulled in during resolution.
void resolveSignaturesForLSP(KGEN::LIT::DeclResolver &resolver);

/// This class represents the internal implementation of the parser driver.
struct MojoParserContext::Impl {
  Impl(llvm::SourceMgr &sourceMgr, KGEN::LIT::ParserConfig &config);

  //===--------------------------------------------------------------------===//
  // General State
  //===--------------------------------------------------------------------===//

  /// The shared state for the parser.
  KGEN::LIT::SharedState sharedState;

  /// The top level decl for everything being parsed.
  KGEN::LIT::ASTDecl *topLevelDecl = nullptr;

  /// The main module we are parsing into.
  mlir::OwningOpRef<ModuleOp> module;

  //===--------------------------------------------------------------------===//
  // REPL State
  //===--------------------------------------------------------------------===//

  /// The location mapper used for REPL expressions.
  MojoParserContext::REPLLocMapper replLocMapper;

  /// The decls of each REPL module that have been successfully parsed, mapped
  /// to the previous REPL module that they replaced.
  llvm::MapVector<KGEN::LIT::ASTDecl *, KGEN::LIT::ASTDecl *>
      prevReplModuleDecls;
  SmallVector<KGEN::LIT::ASTDecl *> replModuleDecls;

  //===--------------------------------------------------------------------===//
  // REPL Completion Cache
  //===--------------------------------------------------------------------===//

  /// Cached parser context used for code completion / signature help. Prior
  /// REPL modules are replayed into this context incrementally so that
  /// subsequent completion requests don't re-parse the entire history.
  struct CompletionCache {
    llvm::SourceMgr sourceMgr;
    KGEN::LIT::ParserConfig config;
    std::unique_ptr<MojoParserContext> context;

    /// The last resolved REPL module in the cached context.
    KGEN::LIT::ASTDecl *lastResolvedDecl = nullptr;

    /// How many modules from the main context's replModuleDecls have been
    /// replayed into this cache.
    size_t replayedModuleCount = 0;

    /// The replDecl that was used to build this cache. If the caller switches
    /// between REPL mode (nullptr) and notebook mode (specific cell), the
    /// cache must be invalidated because the target module set changes.
    KGEN::LIT::ASTDecl *replDeclKey = nullptr;

    /// Number of completion requests served since the last full rebuild.
    /// Used to bound memory growth from stale completion modules.
    size_t completionsSinceRebuild = 0;

    CompletionCache(MLIRContext *mlirCtx,
                    const KGEN::CompilationOptions &options,
                    const llvm::SourceMgr &mainSourceMgr)
        : config(mlirCtx, options) {
      // Suppress diagnostics in the completion cache.
      sourceMgr.setDiagHandler([](const llvm::SMDiagnostic &, void *) {});
      // Copy include directories so import resolution works correctly.
      sourceMgr.setIncludeDirs(mainSourceMgr.getIncludeDirs());
      config.maxNotesPerDiagnostic = 0;
      context = std::make_unique<MojoParserContext>(sourceMgr, config);
    }
  };
  std::unique_ptr<CompletionCache> completionCache;
};
} // namespace M

#endif // PARSERDRIVERIMPL_H
