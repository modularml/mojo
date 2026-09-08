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

#ifndef KGEN_MOJOPARSER_ENTRYPOINT_H
#define KGEN_MOJOPARSER_ENTRYPOINT_H

#include "Support/Context.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <filesystem>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class TimingScope;
} // namespace mlir
namespace M {
class AutoFixItHandler;
} // namespace M

namespace M::AsyncRT {
class CPUDevice;
} // namespace M::AsyncRT
namespace M::KGEN {
class CompilationOptions;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class ASTDecl;
class CallOperands;
class ExprNode;
class PackageOp;
class ParserListener;
class SharedState;

//===----------------------------------------------------------------------===//
// Diagnostic Utils
//===----------------------------------------------------------------------===//

/// This utility takes a slice of Mojo source code and lexes to the start of the
/// next token, returning it as an SMLoc. This is useful for providing
/// diagnostics in tooling.
llvm::SMLoc findStartOfNextToken(SharedState &shared, StringRef string);

//===----------------------------------------------------------------------===//
// ParserConfig
//===----------------------------------------------------------------------===//

/// This class provides the various configurations used to parse a Mojo file.
struct ParserConfig {
  ParserConfig(MLIRContext *context, const CompilationOptions &options)
      : context(context), options(options) {}

  /// The MLIR context to use when parsing the file.
  MLIRContext *context;

  /// The compilation options to use when parsing the file.
  const CompilationOptions &options;

  /// When non-null, this handler collects and applies fix-its automatically.
  AutoFixItHandler *autoFixItHandler = nullptr;

  /// When non-empty, this prefix is stripped from filenames used for
  /// diagnostics & debugging.
  std::string stripFilePrefix;

  /// When true, this prints diagnostics through MLIR (so MLIR features like
  /// -verify-diagnostics may be used). When false, this prints them through
  /// SourceMgr to get ranges and fixit hints.
  bool useMLIRDiagnostics = false;

  /// If true, this will diagnose missing pieces of documentation strings.
  bool diagnoseMissingDocStrings = false;

  /// If true, ignore any already-compiled `foo.mojoc` that appear in its import
  /// search paths. Doing so results in Mojo source packages named `foo/` being
  /// found instead, and those source packages being parsed anew.
  bool disablePrebuiltPackages = false;

  /// If true, auto-import the builtin package.
  bool useBuiltinModule = true;

  /// An optional listener that is used to inspect certain events of the parser.
  /// For simplicity it is a single item, but it could evolve into a list of
  /// listeners.
  ParserListener *parserListener = nullptr;

  /// Maximum number of notes to print per compiler error or warning.
  int maxNotesPerDiagnostic = 10;

  /// Base library path prefix for generated documentation links.
  std::string docsBasePath;
};

//===----------------------------------------------------------------------===//
// Driver Entry Points
//===----------------------------------------------------------------------===//

/// Parse a single .mojo file and return the MLIR module for it.
///
/// If `includedFiles` is provided, it is set to the list of included files when
/// parsing imports.
OwningOpRef<ModuleOp>
importMojoFile(ContextRef context, llvm::SourceMgr &sourceMgr,
               ParserConfig &config, mlir::TimingScope &ts,
               SmallVectorImpl<std::string> *includedFiles = nullptr);

/// Parse the directory at the given path as a Mojo package. Returns a module op
/// that contains the package, represented as a `lit.package` op, as well as the
/// package op itself.
///
/// If `includedFiles` is provided, it is set to the list of included files when
/// parsing imports.
std::pair<OwningOpRef<ModuleOp>, KGEN::LIT::PackageOp>
importMojoPackage(ContextRef context, StringRef path, StringRef packageName,
                  llvm::SourceMgr &sourceMgr, ParserConfig &config,
                  mlir::TimingScope &ts,
                  SmallVectorImpl<std::string> *includedFiles = nullptr);

/// Clone the module containing the given decl, and prepare it for compilation.
/// This handles stripping out any unused decls, stabilizing value uses, and
/// performing any other necessary transformations.
OwningOpRef<ModuleOp> cloneDeclModuleForCompilation(ASTDecl &decl);
OwningOpRef<ModuleOp> cloneDeclModuleForCompilation(ASTDecl &decl,
                                                    mlir::IRMapping &mapping);

//===----------------------------------------------------------------------===//
// ParserListener
//===----------------------------------------------------------------------===//

/// This class provides an interface for other language tools like LSP servers
/// to inspect specific events in the parser.
class ParserListener {
public:
  virtual ~ParserListener() = default;

  /// A functor type used to resolve an input decl for a listener method.
  using ResolveInputDeclFn = function_ref<ASTDecl *()>;

  /// Returns true if the listener is interested in being notified for the given
  /// location.
  virtual bool isInterestedInLoc(llvm::SMLoc parserLoc);

  /// Notify the listener that a new `alias` declaration has been resolved by
  /// the parser.
  virtual void onAliasDecl(ASTDecl *decl, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `function argument` declaration has been
  /// resolved by the parser.
  ///
  /// It is guaranteed that this listener has been notified of its parent
  /// function decl before this call.
  virtual void onArgumentDecl(ASTDecl *decl, StringRef argName,
                              llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `def` or `fn` function declaration has been
  /// resolved by the parser. This includes struct methods and closures.
  virtual void onFunctionDecl(ASTDecl *decl, llvm::SMLoc identifierLoc);

  /// Notify the listener that an import is currently being resolved.
  virtual void onImport(llvm::SMLoc importLoc);

  /// Notify the listener that an import of a module within the given package is
  /// currently being resolved.
  virtual void onImport(ResolveInputDeclFn getPackageDecl,
                        llvm::SMLoc importLoc);

  /// Notify the listener that a member within the given decl is being looked
  /// up. `searchParentScopes` is true if the lookup is not restricted to just
  /// the given decl.
  virtual void onMemberLookup(ResolveInputDeclFn getDeclFn,
                              llvm::SMLoc lookupLoc, bool searchParentScopes);

  /// Notify the listener that a new `module` decl has been created by the
  /// parser.
  virtual void onModuleDecl(ASTDecl *decl, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new function or struct `parameter` decl has
  /// been resolved by the parser.
  virtual void onParameterDecl(ASTDecl *decl, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new import of the form `from Module [as Alias]`
  /// has been resolved by the parser. The provided location and spelling
  /// correspond to the module name and not to its optional alias.
  virtual void onModuleImport(ASTDecl *decl, StringRef spelling,
                              llvm::SMLoc loc);

  /// Notify the listener that a new `struct` declaration has been resolved by
  /// the parser.
  virtual void onStructDecl(ASTDecl *decl, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `struct field` declaration has been
  /// resolved by the parser.
  virtual void onStructFieldDecl(ASTDecl *decl, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `trait` declaration has been resolved by
  /// the parser.
  virtual void onTraitDecl(ASTDecl *decl, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `let` or `var` declaration has been
  /// resolved by the parser.
  virtual void onVariableDecl(ASTDecl *decl, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new reference has been resolved by the parser,
  /// i.e. its declarations are known.
  virtual void onRef(ArrayRef<ASTDecl *> decls, StringRef spelling,
                     llvm::SMRange range);

  /// Notify the listener that a call is being resolved with the given operands.
  virtual void onCall(ArrayRef<ASTDecl *> decls, llvm::SMLoc rparenLoc,
                      const CallOperands &operands);

  /// Notify the listener that parameters are being bound to one of the given
  /// decls.
  virtual void onParameterBinding(ArrayRef<ASTDecl *> decls,
                                  llvm::SMLoc rsquareLoc,
                                  ArrayRef<ExprNode *> parameters);
};
} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_ENTRYPOINT_H
