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

#ifndef KGEN_MOJOTOOLING_PARSERDRIVER_H
#define KGEN_MOJOTOOLING_PARSERDRIVER_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/MemoryBuffer.h"
#include <filesystem>
#include <string>

namespace llvm {
class SMDiagnostic;
class SourceMgr;
} // namespace llvm

namespace M {
namespace KGEN {
class CompilationOptions;
namespace LIT {
class SharedState;
struct ParserConfig;
} // namespace LIT
namespace Mojo {
struct CodeCompletionResult;
struct SignatureHelpResult;
} // namespace Mojo
} // namespace KGEN

class PublicDecl;
class MojoASTDeclRef;
class MojoASTTypeRef;

//===----------------------------------------------------------------------===//
// MojoParserREPLListener
//===----------------------------------------------------------------------===//

/// This class provides a listener for interacting with the parser for REPL
/// like expressions. It contains various hooks to allow for customizing the
/// behavior of the parser.
class MojoParserREPLListener {
public:
  virtual ~MojoParserREPLListener() = default;

  //===------------------------------------------------------------------===//
  // Notifications

  /// The following methods are called by the parser to notify the listener of
  /// various events during parsing. These can be useful for logging,
  /// debugging, etc.

  /// Notify the listener that the parser has wrapped the input expression
  /// into code capable of being parsed. `wrappedExpr` is the fully wrapped
  /// expression.
  virtual void notifyWrappedExpr(StringRef wrappedExpr) = 0;

  /// Notify the listener that the parser applied fixes the original input
  /// expression.
  virtual void notifyFixedExpr(StringRef fixedExpr) = 0;

  /// Notify the listener that the given set of diagnostics were emitted while
  /// parsing the wrapped expression.
  virtual void notifyDiagnostics(ArrayRef<llvm::SMDiagnostic> diagnostics) = 0;

  //===------------------------------------------------------------------===//
  // Queries

  /// The following methods are called by the parser to query the listener for
  /// various information. These can be useful for customizing the behavior of
  /// the parser.

  /// Query the listener to see if a variable with the given name and type
  /// should be persisted. If this returns true, the variable will be appended
  /// to the list of fields within the struct passed to the expression
  /// function.
  virtual bool shouldPersistVariable(StringRef name, Type type) = 0;
};

//===----------------------------------------------------------------------===//
// MojoParserContext
//===----------------------------------------------------------------------===//

/// This class provides a context for parsing and interacting with Mojo
/// modules.
class MojoParserContext {
public:
  MojoParserContext(llvm::SourceMgr &sourceMgr,
                    KGEN::LIT::ParserConfig &config);
  ~MojoParserContext();

  /// Return the current module being parsed.
  ModuleOp getModule();

  /// Return the source manager used by the parser.
  llvm::SourceMgr &getSourceMgr();

  /// Return the shared state user by the parser.
  KGEN::LIT::SharedState &getSharedState();

  /// Return the full list of directories considered for module lookup from
  /// the given file.
  std::vector<std::string> getModuleSearchDirectories(unsigned fileId);

  /// Return the compilation options used by the parser.
  const KGEN::CompilationOptions &getCompilationOptions();

  /// Parse a SourceMgr file given its id as a module.
  ///
  /// In the case of success, the decl corresponding to the module is returned.
  /// In the case of an error, a null decl is returned. If `eraseUnparsedDecls`
  /// is true, any unparsed decls are removed from the module.
  MojoASTDeclRef parseFile(unsigned fileId, bool eraseUnparsedDecls = true);

  /// Parse a SourceMgr file given its ID as a module. This is a specialized
  /// variant of parseFile that does as little work as possible while still
  /// producing a decl that is usable for the language server.
  MojoASTDeclRef parseFileForLSP(unsigned fileId);

  /// Ensures that all parsed decls have been signature-resolved. This is a
  /// required step to ensure the IR is well-formed.
  void ensureSignaturesResolved();

  /// Parse a package with the given path.
  ///
  /// In the case of success, the decl corresponding to the package is returned.
  /// In the case of an error, a null decl is returned.
  MojoASTDeclRef parsePackage(const std::filesystem::path &path);

  /// Parse a module or package with the given path.
  ///
  /// In the case of success, the corresponding decl is returned.
  /// In the case of an error, a null decl is returned.
  MojoASTDeclRef parseFileOrPackage(const std::filesystem::path &path);

  /// Parse a module or package with the given path, without resolving any of
  /// the nested decls. The returned decl provides only a partial view of the
  /// module or package, and does not contain information for nested decls.
  ///
  /// In the case of success, the corresponding decl is returned.
  /// In the case of an error, a null decl is returned.
  MojoASTDeclRef
  parseFileOrPackageNonRecursive(const std::filesystem::path &path);

  /// Parse a module or package with the given path, without resolving
  /// "external" decls that originate from other modules/packages.
  ///
  /// In the case of success, the corresponding decl is returned.
  /// In the case of an error, a null decl is returned.
  MojoASTDeclRef parseIsolatedFileOrPackage(const std::filesystem::path &path);

  /// Returns true if an error occurred during parsing.
  bool wasErrorEmitted() const;

  //===--------------------------------------------------------------------===//
  // Code Completion

  /// Returns the code completion results for the given buffer at the given
  /// completion position.
  static std::vector<KGEN::Mojo::CodeCompletionResult>
  codeComplete(llvm::MemoryBufferRef buffer, uint64_t completionPosition,
               MLIRContext *context, const KGEN::CompilationOptions &options);

  /// Returns the code completion results for the given buffer at the given
  /// completion position. The given callback is invoked with the parsing
  /// context, and source manager buffer file id, allowing for custom additional
  /// setup and parser invocation.
  static std::vector<KGEN::Mojo::CodeCompletionResult>
  codeComplete(llvm::MemoryBufferRef buffer, uint64_t completionPosition,
               MLIRContext *context, const KGEN::CompilationOptions &options,
               function_ref<void(MojoParserContext &, int)> parserCallback,
               bool disableModuleCaching = false);

  /// Returns the code completion results for the given buffer at the given
  /// completion position, using the provided existing context. The buffer is
  /// added to the context's source manager, and the callback is invoked with
  /// the file id. The context's parser listener is temporarily overridden.
  static std::vector<KGEN::Mojo::CodeCompletionResult> codeCompleteInContext(
      MojoParserContext &context, llvm::MemoryBufferRef buffer,
      uint64_t completionPosition, function_ref<void(int)> parserCallback);

  //===--------------------------------------------------------------------===//
  // Signature Help

  /// Returns the signature help result for the given buffer at the given
  /// position.
  static std::optional<KGEN::Mojo::SignatureHelpResult>
  signatureHelp(llvm::MemoryBufferRef buffer, uint64_t position,
                MLIRContext *context, const KGEN::CompilationOptions &options);

  /// Returns the signature help result for the given buffer at the given
  /// position. The given callback is invoked with the parsing context, and
  /// source manager buffer file id, allowing for custom additional setup and
  /// parser invocation.
  static std::optional<KGEN::Mojo::SignatureHelpResult>
  signatureHelp(llvm::MemoryBufferRef buffer, uint64_t position,
                MLIRContext *context, const KGEN::CompilationOptions &options,
                function_ref<void(MojoParserContext &, int)> parserCallback,
                bool disableModuleCaching = false);

  /// Returns the signature help result for the given buffer at the given
  /// position, using the provided existing context.
  static std::optional<KGEN::Mojo::SignatureHelpResult>
  signatureHelpInContext(MojoParserContext &context,
                         llvm::MemoryBufferRef buffer, uint64_t position,
                         function_ref<void(int)> parserCallback);

  //===--------------------------------------------------------------------===//
  // REPL

  /// The following methods provide functionality for interacting with the
  /// parser context from REPL like environments.

  /// This class provides support for mapping between the input expression to a
  /// REPL expression, and the generated wrapped expression used during the
  /// parsing of the REPL expression.
  class REPLLocMapper {
  public:
    class ExprLocMapper;

    ~REPLLocMapper();

    /// Map the given location in the input expression to the wrapped
    /// expression, or vice versa. Returns an invalid location if the location
    /// is not mapped.
    llvm::SMLoc mapLocation(llvm::SMLoc loc) const;

    /// Map the given range in the input expression to the wrapped expression,
    /// or vice versa. Returns invalid locations in the range if they cannot
    /// be mapped.
    /// This handles exclusive ends that don't belong to the underlying buffers.
    llvm::SMRange mapRange(llvm::SMRange range) const;

    /// Remap the locations in the given diagnostic, returning a newly formed
    /// diagnostic.
    llvm::SMDiagnostic mapDiagnostic(const llvm::SMDiagnostic &diag);

  private:
    REPLLocMapper(llvm::SourceMgr &sourceMgr);

    /// Allow access to the constructor.
    friend class MojoParserContext;

    /// The source manager used for the REPL.
    llvm::SourceMgr &sourceMgr;

    /// The mapper used for each expression within the REPL expression.
    std::vector<std::unique_ptr<ExprLocMapper>> exprMappers;
  };

  /// This class represents the result of a parsed REPL expression.
  struct ParsedREPLExpr {
    /// Return true if the expression was parsed successfully.
    bool isValid() const { return exprFnDecl; }

    /// The module decl corresponding to the REPL expression. This field is
    /// always valid.
    MojoASTDeclRef moduleDecl;

    /// The function decl corresponding to the REPL expression. If the REPL
    /// expression failed to parse, this will be null.
    MojoASTDeclRef exprFnDecl;
  };

  /// Return the location mapper used for REPL expressions.
  REPLLocMapper &getREPLLocMapper();

  /// The following methods allow for interacting with the parser for REPL
  /// like expressions, i.e., in environments like Jupyter notebooks.
  /// `exprFileId` is the buffer id of the expression to parse within the main
  /// source manager. `replExprFnName` is the name of the function to use for
  /// wrapping the expression. `replVariables` is a list of pre-existing
  /// variables to make available to the expression function, these variables
  /// should be used as `Pointer[Pointer[]]` fields within a struct that is
  /// passed by reference to the expression function. For example, given the
  /// following expression:
  ///
  ///   print(a)
  ///
  /// Where `a` is a pre-existing repl variable with type `Int`, the
  /// expression wrapper will effectively emulate the following:
  ///
  ///   struct ReplContext:
  ///     var a: Pointer[Pointer[Int]]
  ///
  ///   def replExprFn(context&: ReplContext):
  ///      print(context.a.load().load())
  ///
  ParsedREPLExpr
  parseREPLExpression(MojoParserREPLListener &listener, unsigned exprFileId,
                      StringRef replExprFnName,
                      ArrayRef<std::pair<StringRef, Type>> replVariables);

  /// The following methods allow for interacting with the parser for REPL
  /// like expressions, i.e., in environments like Jupyter notebooks.
  /// `exprText` is the expression text, and represents a sub-string of the
  /// buffer with id `exprFileId`.
  /// `prevReplExpr` is the decl corresponding to the previously parsed REPL
  /// expression, whose state should be imported into the new REPL expression.
  ///
  /// When `parseForLSP` is true, transitive dependencies (decls referenced by
  /// the expression but not owned by it) are only signature-resolved rather
  /// than fully body-resolved. This is correct for LSP doc-string code blocks,
  /// which need type information but never execute the bodies of library
  /// functions. It must NOT be set for interactive REPL cells or notebook
  /// cells, which compile and run the generated code.
  ParsedREPLExpr
  parseREPLExpression(MojoParserREPLListener &listener, unsigned exprFileId,
                      StringRef exprText, StringRef replExprFnName,
                      ArrayRef<std::pair<StringRef, Type>> replVariables,
                      MojoASTDeclRef prevReplExpr, bool parseForLSP);

  /// Return the code completion results for the given REPL expression.
  std::vector<KGEN::Mojo::CodeCompletionResult> codeCompleteREPLExpression(
      StringRef exprText, uint64_t completionPosition,
      ArrayRef<std::pair<StringRef, Type>> replVariables);
  /// Return the code completion results for the given REPL expression.
  /// `replDecl` corresponds to the decl if a previously parsed repl expression.
  /// Completion results will only consider state before that expression was
  /// parsed.
  std::vector<KGEN::Mojo::CodeCompletionResult>
  codeCompleteREPLExpression(StringRef exprText, uint64_t completionPosition,
                             ArrayRef<std::pair<StringRef, Type>> replVariables,
                             MojoASTDeclRef replDecl);

  /// Return a signature help result for the given REPL expression. `replDecl`
  /// corresponds to the decl of a previously parsed repl expression. Signature
  /// results will only consider state before that expression was parsed.
  std::optional<KGEN::Mojo::SignatureHelpResult> signatureHelpREPLExpression(
      StringRef exprText, uint64_t position,
      ArrayRef<std::pair<StringRef, Type>> replVariables,
      MojoASTDeclRef replDecl);

  /// Remove the previously parsed REPL expression. This allows for removing an
  /// erroneous expression when it is only detected as invalid after it has been
  /// parsed.
  void removeLastREPLExpression();

  /// Return if the given variable name is a hidden persistent variable, that
  /// is generated by the REPL context.
  static bool isHiddenPersistentVariable(StringRef name);

  //===--------------------------------------------------------------------===//
  // Types

  /// Get the declaration that defined an AST type.
  MojoASTDeclRef getDecl(MojoASTTypeRef type);

  /// Get the declaration that defined the trait.
  MojoASTDeclRef getTraitDecl(KGEN::TraitSymbolAttr traitSymbol);

  /// Substitute parameters into a type and resolve them into a different type.
  MojoASTTypeRef concretizeType(MojoASTTypeRef base, ArrayRef<TypedAttr> params,
                                MojoASTTypeRef type);

private:
  /// Ensure the completion cache is up-to-date for a completion request.
  /// Collects the target module list, incrementally updates the cache, and
  /// returns the buffer identifier to use for the completion buffer (derived
  /// from replDecl's source buffer for correct import resolution).
  StringRef prepareCompletionCache(MojoASTDeclRef replDecl);

protected:
  /// A struct representing the internal state of the parser.
  struct Impl;

  /// The internal state of the parser.
  std::unique_ptr<Impl> impl;
};

} // namespace M

#endif // KGEN_MOJOTOOLING_PARSERDRIVER_H
