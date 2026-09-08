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

#include "Mojo/MojoTooling/CodeComplete.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/CallOperands.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoParser/ExprNode.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Support/Filesystem/Paths.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;
using namespace M::KGEN::Mojo;

using llvm::SMLoc;
using llvm::SMRange;

/// Returns if the given range, inclusive, contains `loc`.
static bool containsLoc(SMRange range, SMLoc loc) {
  return range.Start.getPointer() <= loc.getPointer() &&
         loc.getPointer() <= range.End.getPointer();
}

//===----------------------------------------------------------------------===//
// BaseCompletionListener
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a base listener for completion or signature help that
/// handles shared listener setup.
struct BaseCompletionListener : public ParserListener {
  BaseCompletionListener(SourceMgr &sourceMgr) : sourceMgr(sourceMgr) {}
  ~BaseCompletionListener() override = default;

  /// The source manager.
  llvm::SourceMgr &sourceMgr;

  /// The range of acceptable locations for the completion.
  llvm::SMRange completionRange;

  /// The current parser context.
  MojoParserContext *parserContext = nullptr;

  /// The buffer ID of the completion buffer within the source manager. In the
  /// in-context path the source manager may contain other buffers (e.g.,
  /// replayed REPL modules), so getMainFileID() is not reliable.
  unsigned completionBufferId = 0;
};
} // namespace

//===----------------------------------------------------------------------===//
// Code Completion: Listener
//===----------------------------------------------------------------------===//

/// Returns true if the given member should be shown during lookup within
/// `decl`. If `isModuleLookup` is true, we are looking up nested modules.
static bool showDeclDuringLookup(MojoASTDeclRef declRef, StringRef &member,
                                 MojoASTDeclRef child,
                                 bool isModuleLookup = false) {
  if (llvm::isa_and_present<PackageOp>(declRef.getIfOperation())) {
    bool childIsPackageOrModule =
        llvm::isa_and_present<FileModuleOp, PackageOp>(child.getIfOperation());
    // If this is a module lookup, we only want to show non-init modules defined
    // within the package.
    if (isModuleLookup)
      return childIsPackageOrModule && member != "__init__";
    // Otherwise, show everything but internally defined modules.
    return !childIsPackageOrModule;
  }
  return true;
}

namespace {
/// This class implements a listener that collects code completion results.
struct CodeCompletionListener : public BaseCompletionListener {
  CodeCompletionListener(std::vector<CodeCompletionResult> &results,
                         llvm::SourceMgr &sourceMgr)
      : BaseCompletionListener(sourceMgr), results(results) {}
  ~CodeCompletionListener() override = default;

  /// Returns true if the listener is interested in being notified for the given
  /// location.
  bool isInterestedInLoc(SMLoc parserLoc) override {
    return containsLoc(completionRange, parserLoc);
  }

  /// Notify the listener that an import is currently being resolved.
  void onImport(SMLoc importLoc) override {
    // Simple helper for adding completion results and dropping duplicates.
    StringSet<> addedImports;
    auto addImportCompletion = [&](const std::filesystem::path &path,
                                   bool isPackage) {
      std::string name = path.stem().string();
      if (!addedImports.insert(name).second)
        return;
      results.emplace_back(name, isPackage ? CodeCompletionResult::kPackage
                                           : CodeCompletionResult::kModule);

      // Grab the documentation for the import. Do this out of the current
      // context to avoid pulling in a bunch of unwanted state.
      MLIRContext ctx{MLIRContext::Threading::DISABLED};
      ParserConfig config(&ctx, parserContext->getCompilationOptions());
      MojoParserContext importContext(sourceMgr, config);
      if (auto module = importContext.parseFileOrPackageNonRecursive(path)) {
        if (auto decl = module.getDecl())
          results.back().documentation =
              decl->getFullMarkdownString(*parserContext);
      }
    };

    // Standard library packages are exposed as top-level imports, even though
    // they are defined inside the 'std' package.
    addedImports.insert("std");
    onImport(
        [&]() {
          return &parserContext->getSharedState().importModule(
              {"std"}, PackageOp(), SMLoc());
        },
        importLoc);
    for (CodeCompletionResult &result : results)
      addedImports.insert(result.label);

    // Compute the viable imports for the given location.
    for (const std::string &dir :
         parserContext->getModuleSearchDirectories(completionBufferId)) {
      std::error_code ec;
      for (const auto &it : std::filesystem::directory_iterator(dir, ec)) {
        if (ec)
          continue;
        if (Filesystem::isMojoSourceFile(it.path()))
          addImportCompletion(it.path(), /*isPackage=*/false);
        else if (Filesystem::isMojoBinaryPackagePath(it.path()) ||
                 Filesystem::isMojoSourcePackagePath(it.path()))
          addImportCompletion(it.path(), /*isPackage=*/true);
        else if (std::filesystem::is_directory(it.path(), ec) && !ec)
          addImportCompletion(it.path(), /*isPackage=*/true);
      }
    }
  }

  /// Notify the listener that an import of a module within the given package is
  /// currently being resolved.
  void onImport(ResolveInputDeclFn getPackageDecl, SMLoc importLoc) override {
    MojoASTDeclRef packageDecl = getPackageDecl();

    // A package's submodules are unlisted children: they live in the
    // module-state cache and the package IR, not in declsInScope (which is what
    // getChildren() iterates below). Thus we enumerate them from the cache
    // instead. This lets a relative import (`from . import X`) complete any
    // sibling module or sub-package.
    if (auto pkgOp =
            dyn_cast_or_null<PackageOp>(packageDecl.getIfOperation())) {
      SharedState &shared = parserContext->getSharedState();
      for (ASTDecl *sub : shared.getNestedModuleDecls(pkgOp)) {
        MojoASTDeclRef subRef(sub);
        StringAttr nameAttr =
            TypeSwitch<Operation *, StringAttr>(subRef.getIfOperation())
                .Case<FileModuleOp, PackageOp>(
                    [](auto m) { return m.getDeclName(); })
                .Default(StringAttr());
        if (!nameAttr || nameAttr.getValue() == "__init__")
          continue;
        addCompletionForOp(nameAttr.getValue(), subRef);
      }
      return;
    }

    for (MojoASTDeclRef::ChildEntry child : packageDecl.getChildren()) {
      StringRef name = child.getName();
      MojoASTDeclRef childDecl = *child.getDecls().begin();
      if (!showDeclDuringLookup(packageDecl, name, childDecl,
                                /*isModuleLookup=*/true))
        continue;

      addCompletionForOp(name, childDecl, [](Operation *op) {
        return isa<FileModuleOp, PackageOp>(op);
      });
    }
  }

  /// Notify the listener that a member within the given decl is being looked
  /// up.
  void onMemberLookup(ResolveInputDeclFn getDeclFn, llvm::SMLoc lookupLoc,
                      bool searchParentScopes) override {
    MojoASTDeclRef decl = getDeclFn();

    auto collectDeclChildren = [&](MojoASTDeclRef decl) {
      // Snapshot the child list: adding a completion for an import placeholder
      // resolves decls lazily, which can grow this scope's decl map (e.g. a
      // module importing from itself) and invalidate a live child iterator.
      SmallVector<std::pair<StringRef, MojoASTDeclRef>> children;
      for (MojoASTDeclRef::ChildEntry child : decl.getChildren())
        children.emplace_back(child.getName(), *child.getDecls().begin());

      for (auto &[name, childDecl] : children) {
        if (!showDeclDuringLookup(decl, name, childDecl))
          continue;

        // TODO: Include information about overloads here and just handle multi
        // decls in general.
        addCompletionForOp(name, childDecl, [](Operation *op) {
          // Witness tables are not user-facing.
          return !isa<ConformanceOp>(op);
        });
      }
    };

    // An import gate's own scope holds only the submodules explicitly imported
    // through it; the imported entity's own members live in its target module.
    // Enumerate the gate's nested children, then resolve to the target.
    if (auto importOp = dyn_cast_or_null<ImportOp>(decl.getIfOperation())) {
      collectDeclChildren(decl);
      SharedState &shared = parserContext->getSharedState();
      SMLoc importLoc = shared.diags.convertLocToSMLoc(importOp->getLoc());
      ASTDecl &target = shared.importModule(
          SharedState::ImportPath::fromAttr(importOp.getModulePath()),
          /*currentPackage=*/PackageOp(), importLoc);
      (void)shared.getDeclResolver().resolveBody(target, importLoc);
      decl = MojoASTDeclRef(&target);
    }

    // A package's own scope is empty; its public surface lives in its
    // __init__. Redirect a package target to its __init__ so member access
    // (`pkg.<...>`) completes exactly what __init__ re-exports.
    if (auto pkgOp = dyn_cast_or_null<PackageOp>(decl.getIfOperation())) {
      SharedState &shared = parserContext->getSharedState();
      SMLoc loc = shared.diags.convertLocToSMLoc(pkgOp->getLoc());
      if (auto initOrFail =
              shared.getDeclResolver().bodyResolvePackageInit(*decl, loc);
          succeeded(initOrFail) && *initOrFail)
        decl = MojoASTDeclRef(*initOrFail);
    }

    // Collect all of the decls in the current scope.
    if (!searchParentScopes)
      return collectDeclChildren(decl);

    // Collect all of the decls in the current scope and all parent scopes.
    do {
      collectDeclChildren(decl);
      decl = decl.getParent();
    } while (
        !llvm::isa_and_present<PackageOp, ModuleOp>(decl.getIfOperation()));
  }

  /// Resolve the decl bound by an unresolved import to the decl it names within
  /// its defining module, or null if it can't be found. Resolution happens in
  /// the imported module's scope; the scope containing `importOp` is not
  /// modified.
  MojoASTDeclRef resolveImportedDecl(UnresolvedImportOp importOp) {
    SharedState &shared = parserContext->getSharedState();
    SMLoc loc = shared.diags.convertLocToSMLoc(importOp->getLoc());
    ASTDecl &module = shared.importModule(
        SharedState::ImportPath::fromAttr(importOp.getModulePathAttr()),
        importOp->getParentOfType<PackageOp>(), loc);
    StringAttr declName = importOp.getDeclNameAttr();
    if (!declName)
      return MojoASTDeclRef(&module);
    LookupResult result =
        shared.lookupAndResolveDecl(declName.getValue(), loc, module,
                                    /*searchParentScopes=*/false,
                                    /*resolveTarget=*/false);
    if (!result.isSuccess()) {
      // The name may be a submodule of a package rather than a symbol in its
      // scope.
      return MojoASTDeclRef(
          shared.tryImportSubModule(module, declName.getValue(), loc));
    }
    return MojoASTDeclRef(result.getIfSuccess().front());
  }

  /// Utility function to add a completion result for the given decl. An
  /// optional filter that returns which operations should be considered.
  void addCompletionForOp(StringRef name, MojoASTDeclRef declRef,
                          function_ref<bool(Operation *)> filter = {}) {
    if (!addedResults.insert({&*declRef, name}).second)
      return;

    Operation *op = declRef.getIfOperation();
    if (!op || (filter && !filter(op)))
      return;

    auto kindForOp = [](Operation *op) {
      return TypeSwitch<Operation *, CodeCompletionResult::Kind>(op)
          .Case([](FileModuleOp) { return CodeCompletionResult::kModule; })
          .Case([](PackageOp) { return CodeCompletionResult::kPackage; })
          .Case([](ImportOp) { return CodeCompletionResult::kModule; })
          .Case([](StructDeclOp) { return CodeCompletionResult::kStruct; })
          .Case([](TraitDeclOp) { return CodeCompletionResult::kTrait; })
          .Case([](FnOp) { return CodeCompletionResult::kFunction; })
          .Case([](StructFieldOp) { return CodeCompletionResult::kField; })
          .Case([](VarDeclOp) { return CodeCompletionResult::kVariable; })
          .Default(CodeCompletionResult::kUnknown);
    };
    CodeCompletionResult::Kind kind = kindForOp(op);

    // An import binding that hasn't been referenced yet is still an unresolved
    // placeholder, which would complete as an unknown kind. Peek at the decl it
    // names in its defining module for the kind.
    if (auto importOp = dyn_cast<UnresolvedImportOp>(op)) {
      if (MojoASTDeclRef target = resolveImportedDecl(importOp);
          target && target.getIfOperation())
        kind = kindForOp(target.getIfOperation());
    }

    CodeCompletionResult result(name, kind);
    if (auto decl = declRef.getDecl())
      result.documentation = decl->getFullMarkdownString(*parserContext);
    results.emplace_back(result);
  }

  /// The (decl, label) pairs that have been collected so far. Keyed by both
  /// because one decl can be bound under several names, each of which is its
  /// own completion (e.g. `from module import name as other_name` binds the
  /// same decl as `from module import name`).
  DenseSet<std::pair<ASTDecl *, StringRef>> addedResults;
  std::vector<CodeCompletionResult> &results;
};
} // namespace

//===----------------------------------------------------------------------===//
// Signature Help: Listener
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a listener that collects signature help results.
struct SignatureHelpListener : public BaseCompletionListener {
  SignatureHelpListener(llvm::SourceMgr &sourceMgr, SignatureHelpResult &result)
      : BaseCompletionListener(sourceMgr), result(result) {}
  ~SignatureHelpListener() override = default;

  /// Returns true if the listener is interested in being notified for the given
  /// location.
  bool isInterestedInLoc(SMLoc loc) override {
    // Filter at a high level for locations in the completion buffer, we'll
    // filter further when examining calls.
    return completionBufferId == sourceMgr.FindBufferContainingLoc(loc);
  }

  void onCall(ArrayRef<ASTDecl *> decls, llvm::SMLoc rparenLoc,
              const CallOperands &operands) override {
    auto findInterestedOperand = [&]() -> std::optional<size_t> {
      for (const auto &[index, operand] : llvm::enumerate(operands.values)) {
        if (containsLoc(completionRange, operand.expr->getRangeStart()))
          return index;
      }

      // Consider the rparen location if it is within the completion range.
      bool noKWArgs = operands.getNumKwOperands() == 0;
      if (noKWArgs && containsLoc(completionRange, rparenLoc))
        return operands.size();

      // TODO: Consider kwargs.
      return std::nullopt;
    };

    // Check if any of the operands are within the completion range.
    std::optional<size_t> operandIndex = findInterestedOperand();
    if (!operandIndex)
      return;
    result.activeParameter = *operandIndex;

    // Collect the signatures for each of the decls.
    for (MojoASTDeclRef decl : decls) {
      std::unique_ptr<PublicDecl> publicDecl = decl.getDecl();
      if (!publicDecl)
        continue;
      if (auto *fnDecl = dyn_cast<PublicFunctionDecl>(publicDecl.get())) {
        if (operands.size() > fnDecl->getArguments().size())
          continue;

        // If this is the first function and it's a method, bump the active
        // parameter past the self argument.
        if (result.signatures.empty() && fnDecl->isMethod())
          ++result.activeParameter;

        SignatureHelpResult::Signature signature;
        SmallVector<std::pair<unsigned, unsigned>> argOffsets;
        signature.label = fnDecl->getDeclarationSnippet(
            *parserContext,
            /*parameterOffsets=*/nullptr, &argOffsets);
        addDeclDocAndParametersToSignature(*parserContext, signature, *fnDecl,
                                           fnDecl->getArguments(), argOffsets);
        result.signatures.emplace_back(std::move(signature));
      }
    }
  }

  void onParameterBinding(ArrayRef<ASTDecl *> decls, llvm::SMLoc rsquareLoc,
                          ArrayRef<ExprNode *> parameters) override {
    auto findInterestedParam = [&]() -> std::optional<size_t> {
      for (const auto &[index, param] : llvm::enumerate(parameters))
        if (containsLoc(completionRange, param->getRangeStart()))
          return index;

      // Consider the rparen location if it is within the completion range.
      if (containsLoc(completionRange, rsquareLoc))
        return parameters.size();
      return std::nullopt;
    };

    // Check if any of the operands are within the completion range.
    std::optional<size_t> paramIndex = findInterestedParam();
    if (!paramIndex)
      return;
    result.activeParameter = *paramIndex;

    // Collect the signatures for each of the decls.
    for (MojoASTDeclRef declRef : decls) {
      std::unique_ptr<PublicDecl> publicDecl = declRef.getDecl();
      if (!publicDecl)
        continue;
      TypeSwitch<PublicDecl *>(publicDecl.get())
          .Case<PublicFunctionDecl, PublicStructDecl>([&](auto *publicDecl) {
            if (parameters.size() > publicDecl->getParameters().size())
              return;
            SignatureHelpResult::Signature signature;
            SmallVector<std::pair<unsigned, unsigned>> paramOffsets;
            signature.label = publicDecl->getDeclarationSnippet(*parserContext,
                                                                &paramOffsets);
            addDeclDocAndParametersToSignature(
                *parserContext, signature, *publicDecl,
                publicDecl->getParameters(), paramOffsets);
            result.signatures.emplace_back(std::move(signature));
          });
    }
  }

  /// Utility function for adding the documentation and parameter information
  /// form the given decl to a signature.
  template <typename RangeT>
  static void addDeclDocAndParametersToSignature(
      MojoParserContext &ctx, SignatureHelpResult::Signature &signature,
      PublicDecl &publicDecl, RangeT &&range,
      ArrayRef<std::pair<unsigned, unsigned>> offsets) {
    signature.documentation = publicDecl.getFullMarkdownString(ctx);
    for (const auto &[arg, offset] : llvm::zip(range, offsets))
      signature.parameters.push_back({offset, arg.getMarkdownDocString()});
  }

  /// The result that has been collected so far.
  SignatureHelpResult &result;
};
} // namespace

//===----------------------------------------------------------------------===//
// Entrypoint
//===----------------------------------------------------------------------===//

/// Compute the completion range for the given buffer and position. The start
/// is trimmed back to the beginning of any partial identifier, and the end is
/// advanced to the start of the next token.
static void computeCompletionRange(SharedState &sharedState,
                                   llvm::MemoryBufferRef buffer,
                                   uint64_t completionPosition,
                                   BaseCompletionListener &listener) {
  // Compute the start completion location. We first trim the buffer to the
  // last non-whitespace, and then to the start of any identifier. We often get
  // completion requests for lookups that are partially formed already (e.g. a
  // completion on `p` to get things like `print`).
  StringRef completionPosStr =
      buffer.getBuffer().take_front(completionPosition).rtrim();
  while (!completionPosStr.empty()) {
    char c = completionPosStr.back();
    if (!(llvm::isAlpha(c) || llvm::isDigit(c) || c == '_' || c == '$'))
      break;
    completionPosStr = completionPosStr.drop_back();
  }
  listener.completionRange.Start =
      SMLoc::getFromPointer(completionPosStr.end());

  // Compute the end completion location by finding the next token from the
  // input completion position.
  completionPosStr = buffer.getBuffer().drop_front(completionPosition);
  listener.completionRange.End =
      findStartOfNextToken(sharedState, completionPosStr);
}

/// Parse the given buffer for completion results using the given listener
/// implementation. Creates a fresh MojoParserContext.
static void parseCompletionImpl(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, const KGEN::CompilationOptions &options,
    function_ref<void(MojoParserContext &, int)> parserCallback,
    BaseCompletionListener &listener, bool disableModuleCaching) {
  if (buffer.getBufferSize() < completionPosition)
    return;
  int bufId = listener.sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(buffer), SMLoc());
  listener.completionBufferId = bufId;

  // Suppress diagnostics during completion.
  listener.sourceMgr.setDiagHandler([](const llvm::SMDiagnostic &, void *) {});

  ParserConfig config(context, options);
  config.parserListener = &listener;
  config.maxNotesPerDiagnostic = 0;

  MojoParserContext parserContext(listener.sourceMgr, config);
  listener.parserContext = &parserContext;

  computeCompletionRange(parserContext.getSharedState(), buffer,
                         completionPosition, listener);

  parserCallback(parserContext, bufId);
}

//===----------------------------------------------------------------------===//
// Code Completion

std::vector<CodeCompletionResult> MojoParserContext::codeComplete(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, const KGEN::CompilationOptions &options) {
  return codeComplete(
      buffer, completionPosition, context, options,
      [](MojoParserContext &ctx, int fileID) { ctx.parseFileForLSP(fileID); });
}

std::vector<CodeCompletionResult> MojoParserContext::codeComplete(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, const KGEN::CompilationOptions &options,
    function_ref<void(MojoParserContext &, int)> parserCallback,
    bool disableModuleCaching) {
  llvm::SourceMgr sourceMgr;
  std::vector<CodeCompletionResult> results;
  CodeCompletionListener listener(results, sourceMgr);
  parseCompletionImpl(buffer, completionPosition, context, options,
                      parserCallback, listener, disableModuleCaching);
  return results;
}

//===----------------------------------------------------------------------===//
// Signature Help

std::optional<SignatureHelpResult>
MojoParserContext::signatureHelp(llvm::MemoryBufferRef buffer,
                                 uint64_t position, MLIRContext *context,
                                 const KGEN::CompilationOptions &options) {
  return signatureHelp(
      buffer, position, context, options,
      [](MojoParserContext &ctx, int fileID) { ctx.parseFileForLSP(fileID); });
}

std::optional<SignatureHelpResult> MojoParserContext::signatureHelp(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, const KGEN::CompilationOptions &options,
    function_ref<void(MojoParserContext &, int)> parserCallback,
    bool disableModuleCaching) {
  llvm::SourceMgr sourceMgr;
  SignatureHelpResult result;
  SignatureHelpListener listener(sourceMgr, result);
  parseCompletionImpl(buffer, completionPosition, context, options,
                      parserCallback, listener, disableModuleCaching);
  return result.signatures.empty() ? std::nullopt : std::optional(result);
}

//===----------------------------------------------------------------------===//
// In-Context Completion (for REPL caching)

/// Set up a completion listener on an existing context, add the buffer, compute
/// the completion range, invoke the callback, and restore the listener.
static void parseCompletionInContextImpl(
    MojoParserContext &context, llvm::MemoryBufferRef buffer,
    uint64_t completionPosition, BaseCompletionListener &listener,
    function_ref<void(int)> parserCallback) {
  if (buffer.getBufferSize() < completionPosition)
    return;

  auto &sourceMgr = context.getSourceMgr();
  int bufId = sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(buffer), SMLoc());
  listener.completionBufferId = bufId;

  // Suppress diagnostics during completion.
  sourceMgr.setDiagHandler([](const llvm::SMDiagnostic &, void *) {});

  // Temporarily override the parser listener.
  auto *oldListener = context.getSharedState().parserListener;
  context.getSharedState().parserListener = &listener;
  listener.parserContext = &context;
  auto restoreListener = llvm::scope_exit(
      [&] { context.getSharedState().parserListener = oldListener; });

  computeCompletionRange(context.getSharedState(), buffer, completionPosition,
                         listener);

  parserCallback(bufId);
}

std::vector<CodeCompletionResult> MojoParserContext::codeCompleteInContext(
    MojoParserContext &context, llvm::MemoryBufferRef buffer,
    uint64_t completionPosition, function_ref<void(int)> parserCallback) {
  std::vector<CodeCompletionResult> results;
  CodeCompletionListener listener(results, context.getSourceMgr());
  parseCompletionInContextImpl(context, buffer, completionPosition, listener,
                               parserCallback);
  return results;
}

std::optional<SignatureHelpResult> MojoParserContext::signatureHelpInContext(
    MojoParserContext &context, llvm::MemoryBufferRef buffer, uint64_t position,
    function_ref<void(int)> parserCallback) {
  SignatureHelpResult result;
  SignatureHelpListener listener(context.getSourceMgr(), result);
  parseCompletionInContextImpl(context, buffer, position, listener,
                               parserCallback);
  return result.signatures.empty() ? std::nullopt : std::optional(result);
}
