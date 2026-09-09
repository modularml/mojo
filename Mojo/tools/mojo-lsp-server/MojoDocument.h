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

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_MOJODOCUMENT_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_MOJODOCUMENT_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "Mojo/MojoParser/DocString.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/tools/mojo-lsp-server/LSPTelemetryContext.h"
#include "MojoServer.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/MapVector.h"

/// Define ordering operators for SMLoc for use in IntervalMap.
namespace llvm {
inline bool operator<(const SMLoc &lhs, const SMLoc &rhs) {
  return lhs.getPointer() < rhs.getPointer();
}
inline bool operator<=(const SMLoc &lhs, const SMLoc &rhs) {
  return lhs.getPointer() <= rhs.getPointer();
}
} // namespace llvm

namespace {

class ProgressManager;

}

namespace M::Mojo::LSP {
class MojoDocStrings;
struct SemanticToken;

//===----------------------------------------------------------------------===//
// MojoInlayHint
//===----------------------------------------------------------------------===//

/// This class is used to represent an inlay hint in the document. This is a bit
/// more stripped, optimized, and mojo specific compared to lsp::InlayHint.
struct MojoInlayHint {
  MojoInlayHint(llvm::lsp::InlayHintKind kind, StringRef label, SMLoc loc)
      : label(label), loc(loc), leftIndent(0), kind(kind), paddingLeft(false),
        paddingRight(false) {}

  /// Generate an LSP inlay hint from this inlay hint.
  llvm::lsp::InlayHint toLspInlayHint(SourceMgr &sourceMgr) const;

  /// Order inlay hints by their location.
  bool operator<(const MojoInlayHint &other) const {
    return loc.getPointer() < other.loc.getPointer();
  }

  /// The label of the inlay hint.
  StringRef label;

  /// The location of the inlay hint.
  SMLoc loc;

  /// An optional left indent for the inlay hint.
  unsigned leftIndent : 28;

  /// The kind of the inlay hint.
  llvm::lsp::InlayHintKind kind : 2;

  /// If the hint should be padded to the left.
  bool paddingLeft : 1;

  /// If the hint should be padded to the right.
  bool paddingRight : 1;
};

//===----------------------------------------------------------------------===//
// MojoDocument
//===----------------------------------------------------------------------===//

/// This class represents all of the information pertaining to a specific Mojo
/// document.
struct MojoDocument : public ReferenceCounted<MojoDocument> {
public:
  MojoDocument(const MojoDocument &) = delete;
  MojoDocument &operator=(const MojoDocument &) = delete;
  virtual ~MojoDocument() = default;

  /// Return the version of this document.
  int64_t getVersion() const { return version; }

  /// Return the cpuDevice used for this document.
  AsyncRT::CPUDevice &getRuntime() const { return cpuDevice; }

  /// Return the URIs of this document.
  ArrayRef<llvm::lsp::URIForFile> getURIs() const { return uris; }

  /// Return the source manager used for this document.
  llvm::SourceMgr &getSourceMgr() { return sourceMgr; }

  /// Return the compilation options for this document.
  const KGEN::CompilationOptions &getCompilationOptions() const;

  /// Return the parser context for this document.
  MojoParserContext &getParserContext() const;

  /// Invalidate this document.
  void invalidate();

  /// Returns the current task chain of the document. When this value is readied
  /// all currently outstanding tasks have been completed.
  AnyAsyncValueRef getTaskChain() {
    std::lock_guard<std::mutex> guard(currentTaskMutex);
    return currentTaskChain.copy();
  }

  //===--------------------------------------------------------------------===//
  // RTTI Utilities
  //===--------------------------------------------------------------------===//

  /// The kind of document this is.
  enum class Kind {
    kTextDocument,
    kNotebookDocument,
  };

  /// Return the kind of this document.
  Kind getKind() const { return kind; }

  //===--------------------------------------------------------------------===//
  // Document Utilities
  //===--------------------------------------------------------------------===//

  /// Returns true if the document contains the given location.
  virtual bool containsLocation(llvm::SMLoc loc) = 0;

  /// Returns true if the document contains the given location.
  virtual llvm::SMLoc getLocFromPos(const llvm::lsp::URIForFile &uri,
                                    llvm::lsp::Position position) = 0;

  /// Return the source range from the given LSP range.
  llvm::SMRange getLocFromPos(const llvm::lsp::URIForFile &uri,
                              const llvm::lsp::Range &range) {
    return llvm::SMRange(getLocFromPos(uri, range.start),
                         getLocFromPos(uri, range.end));
  }

  /// Return a location range for the document of the given uri.
  virtual llvm::SMRange
  getFullRangeForURI(const llvm::lsp::URIForFile &uri) = 0;

  /// Translate the given parser location into one usable by the language
  /// server.
  virtual llvm::SMLoc translateParserLoc(llvm::SMLoc loc) { return loc; }
  llvm::SMRange translateParserLoc(llvm::SMRange range) {
    llvm::SMLoc newStart = translateParserLoc(range.Start);
    auto newEnd = llvm::SMLoc::getFromPointer(
        newStart.getPointer() +
        (range.End.getPointer() - range.Start.getPointer()));
    return {newStart, newEnd};
  }

  /// Returns a language server uri for the given source location. `mainFileURI`
  /// corresponds to the uri for the main file of the source manager.
  std::optional<llvm::lsp::URIForFile> getURIFromLoc(llvm::SMLoc loc);

  /// Returns a language server location from the given diagnostic.
  std::optional<llvm::lsp::Location>
  getLocationFromDiag(const llvm::SMDiagnostic &diag);

  /// Get a document symbol with the given ASTDecl, appending it to the given
  /// vector.
  void getDocumentSymbols(MojoASTDeclRef decl,
                          std::vector<llvm::lsp::DocumentSymbol> &symbols);

  /// Get a document symbol with the given ASTDecl, appending it to the given
  /// vector. The provided functor defines whether a decl should be included in
  /// the symbol list.
  void getDocumentSymbols(MojoASTDeclRef decl,
                          std::vector<llvm::lsp::DocumentSymbol> &symbols,
                          function_ref<bool(MojoASTDeclRef)> shouldIncludeDecl);

  /// Recursively process the document strings in decls nested within `decl`.
  /// The provided functor defines whether a decl should be processed. If the
  /// main document represents a REPL module, `curReplDecl` is the AST decl for
  /// the REPL module that contains `decl`. In the case of a normal text
  /// document, `curReplDecl` is null.
  void processDocStrings(MojoDocStrings &docStrings, MojoASTDeclRef decl,
                         unsigned bufferId,
                         function_ref<bool(MojoASTDeclRef)> shouldIncludeDecl,
                         MojoASTDeclRef curReplDecl = {});

  /// Recursively process the document strings in decls nested within `decl`. If
  /// the main document represents a REPL module, `curReplDecl` is the AST decl
  /// for the REPL module that contains `decl`. In the case of a normal text
  /// document, `curReplDecl` is null.
  void processDocStrings(MojoDocStrings &docStrings, MojoASTDeclRef decl,
                         MojoASTDeclRef curReplDecl = {});

  /// Check the given the parsed module decl for high-level semantic issues. Any
  /// errors are reported to the source manager.
  void checkModuleSemantics(MojoASTDeclRef decl);

  /// Starts the document parse task. This must be invoked _after_ construction,
  /// because there is setup code that runs after the constructor that has to be
  /// finished before the parse task can complete successfully.
  void startDocumentParse(LSPTelemetryContext &telemetryCtx,
                          ProgressManager &progressMgr);

  //===--------------------------------------------------------------------===//
  // Asynchronous LSP Queries
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Code Actions

  void
  getCodeActions(const llvm::lsp::URIForFile &uri, const llvm::lsp::Range &pos,
                 const llvm::lsp::CodeActionContext &context,
                 LSPResponder<std::vector<llvm::lsp::CodeAction>> responder);

  //===--------------------------------------------------------------------===//
  // Language Features

  void onCodeCompletion(const llvm::lsp::URIForFile &uri,
                        const llvm::lsp::Position &completePos,
                        llvm::unique_function<bool()> hasPendingUpdate,
                        LSPResponder<llvm::lsp::CompletionList> responder);

  void onDefinition(const llvm::lsp::URIForFile &uri,
                    const llvm::lsp::Position &pos,
                    LSPResponder<std::vector<llvm::lsp::Location>> responder);

  void onDocumentSymbol(
      const llvm::lsp::URIForFile &uri,
      LSPResponder<std::vector<llvm::lsp::DocumentSymbol>> responder);

  void
  onFoldingRange(const llvm::lsp::URIForFile &uri,
                 LSPResponder<std::vector<llvm::lsp::FoldingRange>> responder);

  void onHover(const llvm::lsp::URIForFile &uri, const llvm::lsp::Position &pos,
               LSPResponder<std::optional<llvm::lsp::Hover>> responder);

  void onInlayHint(const llvm::lsp::URIForFile &uri,
                   const llvm::lsp::Range &range,
                   LSPResponder<std::vector<llvm::lsp::InlayHint>> responder);

  void onReferences(const llvm::lsp::URIForFile &uri,
                    const llvm::lsp::Position &position,
                    bool includeDeclaration,
                    LSPResponder<std::vector<llvm::lsp::Location>> responder);

  void onSemanticTokens(
      const llvm::lsp::URIForFile &uri,
      OnSemanticTokensResultFn<std::optional<std::vector<SemanticToken>>>
          onSemanticTokens);

  void onSignatureHelp(const llvm::lsp::URIForFile &uri,
                       const llvm::lsp::Position &pos,
                       LSPResponder<llvm::lsp::SignatureHelp2> responder);

  void onRename(const llvm::lsp::URIForFile &uri,
                const llvm::lsp::Position &pos, StringRef newName,
                LSPResponder<llvm::lsp::WorkspaceEdit> responder);

  //===--------------------------------------------------------------------===//
  // Debugging methods

  void dumpParsedIR();

protected:
  MojoDocument(Kind kind, ArrayRef<llvm::lsp::URIForFile> uris, int64_t version,
               SendDiagnosticsFnRef sendDiagnosticsFn,
               AsyncRT::CPUDevice &cpuDevice,
               ArrayRef<std::string> includeDirs);

  /// A collection of MLIR and Mojo related entities used to invoke the parser.
  /// Its lifetime is tied to that of the AST objects gotten from the parser.
  /// It also sets up a SourceMgr with the given MojoDocument as its main file.
  struct Context;

  //===--------------------------------------------------------------------===//
  // Derived Document Hooks
  //===--------------------------------------------------------------------===//

  /// Hook that is invoked to perform the raw document parsing process. Returns
  /// the number of bytes parsed by the Mojo parser.
  virtual size_t parseDocumentImpl() = 0;

  /// Hook that returns the URI for the given contained location.
  virtual const llvm::lsp::URIForFile &
  getURIFromContainedLoc(llvm::SMLoc loc) = 0;

  //===--------------------------------------------------------------------===//
  // Language Features

  /// Hook that is invoked to perform code completion at the given position.
  virtual std::vector<KGEN::Mojo::CodeCompletionResult>
  onCodeCompletionSyncImpl(llvm::SMLoc completeLoc) = 0;

  /// Hook that returns the symbols within the document.
  virtual std::vector<llvm::lsp::DocumentSymbol>
  onDocumentSymbolSync(const llvm::lsp::URIForFile &uri) = 0;

  /// Hook that returns the folding ranges within the document.
  virtual std::vector<llvm::lsp::FoldingRange>
  onFoldingRangeSync(const llvm::lsp::URIForFile &uri) = 0;

  /// Hook that is invoked to perform signature help at the given position.
  virtual std::optional<KGEN::Mojo::SignatureHelpResult>
  onSignatureHelpSyncImpl(llvm::SMLoc loc) = 0;

private:
  /// Parse the document and populate the index based on the current contents.
  void parseDocument(LSPTelemetryContext &telemetryCtx,
                     ProgressManager &progressMgr);

  /// Enqueue a new sequential task. Returns the previous task's chain (to wait
  /// on) and a chain for the new task to mark as finished.
  std::pair<AsyncValueRef<Chain>, AsyncValueRef<Chain>> enqueueNewTask();

  /// Start a task to be run sequentially, with exclusive access to the
  /// document's parse state.
  template <typename FnT>
  void startTask(FnT &&fn) {
    auto [previous, current] = enqueueNewTask();

    previous.andThenAsync([doc = RCRef<MojoDocument>::copy(this),
                           fn = std::forward<FnT>(fn),
                           current = std::move(current)]() mutable {
      fn(*doc);
      std::move(current).emplace();
    });
  }

  //===--------------------------------------------------------------------===//
  // Synchronous LSP Queries
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Diagnostics

  std::optional<llvm::lsp::Diagnostic>
  buildLspDiagnosticFromSMDiagnostic(llvm::SourceMgr &sourceMgr,
                                     ArrayRef<llvm::SMDiagnostic> diags,
                                     const llvm::lsp::URIForFile &uri);

  //===--------------------------------------------------------------------===//
  // Code Actions

  std::vector<llvm::lsp::CodeAction>
  getCodeActionsSync(llvm::SMRange range,
                     const llvm::lsp::CodeActionContext &context);

  //===--------------------------------------------------------------------===//
  // Language Features

  llvm::lsp::CompletionList onCodeCompletionSync(llvm::SMLoc completeLoc);

  std::vector<llvm::lsp::Location> onDefinitionSync(llvm::SMLoc loc);

  std::optional<llvm::lsp::Hover> onHoverSync(llvm::SMLoc loc);

  std::vector<llvm::lsp::InlayHint> onInlayHintSync(llvm::SMRange range);

  std::vector<llvm::lsp::Location> onReferencesSync(SMLoc smLoc,
                                                    bool includeDeclaration);

  ErrorOr<std::vector<llvm::lsp::TextEdit>>
  onRenameSync(SMLoc loc, const std::string &newName);

  std::optional<std::vector<SemanticToken>>
  onSemanticTokensSync(llvm::SMRange range);

  llvm::lsp::SignatureHelp onSignatureHelpSync(llvm::SMLoc loc);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Static Fields

  /// The following fields are always available for access and don't require
  /// additional synchronization.

  /// The kind of this document.
  Kind kind;

  /// The uri of the file.
  SmallVector<llvm::lsp::URIForFile> uris;

  /// The version of this file.
  int64_t version = 0;

  /// The function used to send diagnostics for this document.
  SendDiagnosticsFnRef sendDiagnosticsFn;

  /// The cpuDevice used when parsing the file.
  AsyncRT::CPUDevice &cpuDevice;

  /// A flag indicating if this document version has been invalidated.
  std::atomic<bool> isInvalidated = false;

  /// The source manager used to parse the document.
  llvm::SourceMgr sourceMgr;

  //===--------------------------------------------------------------------===//
  // Parsed Fields

  /// An async value readied when the document has finished executing all
  /// currently-enqueued tasks.
  AsyncValueRef<Chain> currentTaskChain;

  /// Incremented once when a new chain is enqueued. Used to ensure that the
  /// first task queued is the parse task.
  size_t chainIndex = 0;

  /// Guards access to currentTaskChain and chainIndex.
  std::mutex currentTaskMutex;

  /// The following fields are only available after the document has been
  /// parsed. To access these fields safely, use the startTask method.

  /// A set of fixits for diagnostics emitted for the current version of the
  /// file.
  llvm::StringMap<
      std::map<llvm::lsp::Range, std::vector<llvm::lsp::CodeAction>>>
      fixits;

  /// An ordered set of inlay hints for the current version of the file.
  std::vector<MojoInlayHint> inlayHints;

  /// Indicates if the document produced parser errors.
  bool hasParserErrors = false;

  /// The overall parser context.
  std::unique_ptr<Context> context;
};

using MojoDocumentRef = RCRef<MojoDocument>;

//===----------------------------------------------------------------------===//
// MojoDocStrings
//===----------------------------------------------------------------------===//

/// This class represents all of the doc string state within a Mojo document,
/// including code block state. Code blocks somewhat function as independent
/// documents, as they are parsed and processed separately from the main
/// document, but are still tied to the main document (e.g. for locations,
/// requests, etc.).
class MojoDocStrings {
public:
  MojoDocStrings() : rangeToCodeBlock(allocator) {}

  /// This class represents an individual code block within a doc string.
  struct CodeBlock {
    CodeBlock(StringRef contents,
              SmallVector<std::pair<StringRef, Type>> persistentVariables,
              unsigned contentsIndent)
        : contents(contents),
          persistentVariables(std::move(persistentVariables)),
          contentsIndent(contentsIndent) {}

    /// Attempt to perform code completion at the given location.
    std::vector<KGEN::Mojo::CodeCompletionResult>
    onCodeCompletion(llvm::SMLoc loc, MojoParserContext &ctx);

    /// Attempt to compute signature help at the given location.
    std::optional<KGEN::Mojo::SignatureHelpResult>
    onSignatureHelp(llvm::SMLoc loc, MojoParserContext &ctx);

    /// The contents of the code block.
    StringRef contents;

    /// The persistent REPL variables defined in code blocks defined before this
    /// one in the same doc string.
    SmallVector<std::pair<StringRef, Type>> persistentVariables;

    /// The AST decl for the module containing this code block.
    MojoASTDeclRef decl;

    /// The indent of the code within the contents.
    unsigned contentsIndent;
  };

  /// This class represents an individual doc string within a Mojo document.
  struct DocString {
    DocString(llvm::SMRange range) : range(range) {}

    /// The range of the doc string.
    llvm::SMRange range;
  };

  /// Add the doc string and any code blocks for the given decl. `bufferId` is
  /// the source manager buffer for the main document. If the main document
  /// represents a REPL module, `curReplDecl` is the AST decl for the REPL
  /// module that contains `decl`. In the case of a normal text document,
  /// `curReplDecl` is null.
  void addDocString(MojoDocument &mainDoc, MojoASTDeclRef decl,
                    MojoASTDeclRef curReplDecl, unsigned bufferId);

  /// When true, parse and type-check code blocks inside doc strings. Defaults
  /// to off, since docstring examples need not be fully correct Mojo code.
  /// Text documents override it from the server's `-check-docstrings` flag;
  /// notebook cells use this default.
  bool checkCodeBlocks = false;

  /// Find the code block that contains the given location.
  CodeBlock *findContainingCodeBlock(llvm::SMLoc loc);

  /// Get the folding ranges for held doc strings.
  void getFoldingRanges(SourceMgr &sourceMgr,
                        std::vector<llvm::lsp::FoldingRange> &ranges);

private:
  using MapT = llvm::IntervalMap<
      SMLoc, CodeBlock *,
      llvm::IntervalMapImpl::NodeSizer<SMLoc, CodeBlock *>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<SMLoc>>;

  /// An allocator to use for code blocks.
  llvm::SpecificBumpPtrAllocator<CodeBlock> codeBlockAllocator;

  /// The code blocks within the document.
  SmallVector<CodeBlock *> codeBlocks;

  /// The doc strings within the document.
  SmallVector<DocString> docStrings;

  /// A map of source locations within the main document to code blocks.
  MapT::Allocator allocator;
  MapT rangeToCodeBlock;
};

//===----------------------------------------------------------------------===//
// MojoTextDocument
//===----------------------------------------------------------------------===//

/// This class represents all of the information pertaining to a specific Mojo
/// text document, i.e. a .mojo file.
struct MojoTextDocument : public MojoDocument {
public:
  MojoTextDocument(const llvm::lsp::URIForFile &uri, std::string &&contents,
                   int64_t version, SendDiagnosticsFnRef sendDiagnosticsFn,
                   AsyncRT::CPUDevice &cpuDevice,
                   ArrayRef<std::string> includeDirs,
                   bool checkDocstringCodeBlocks = false);
  MojoTextDocument(const MojoDocument &) = delete;
  MojoTextDocument &operator=(const MojoDocument &) = delete;

  /// Return the contents of this document.
  StringRef getContents() const { return contents; }

  /// Support LLVM RTTI.
  static bool classof(const MojoDocument *doc) {
    return doc->getKind() == Kind::kTextDocument;
  }

private:
  //===--------------------------------------------------------------------===//
  // Derived Document Hooks
  //===--------------------------------------------------------------------===//

  /// Hook that is invoked to perform the raw document parsing process. Returns
  /// the number of bytes parsed by the Mojo parser.
  size_t parseDocumentImpl() override;

  /// Hook that returns the URI for the given contained location.
  const llvm::lsp::URIForFile &getURIFromContainedLoc(llvm::SMLoc loc) override;

  /// Returns true if the document contains the given location.
  bool containsLocation(llvm::SMLoc loc) override;

  /// Translate the given parser location into one usable by the language
  /// server.
  llvm::SMLoc translateParserLoc(llvm::SMLoc loc) override;

  /// Returns true if the document contains the given location.
  llvm::SMLoc getLocFromPos(const llvm::lsp::URIForFile &uri,
                            llvm::lsp::Position position) override;

  /// Return a location range for the document of the given uri.
  llvm::SMRange getFullRangeForURI(const llvm::lsp::URIForFile &uri) override;

  //===--------------------------------------------------------------------===//
  // Language Features

  std::vector<KGEN::Mojo::CodeCompletionResult>
  onCodeCompletionSyncImpl(llvm::SMLoc completeLoc) override;

  std::vector<llvm::lsp::DocumentSymbol>
  onDocumentSymbolSync(const llvm::lsp::URIForFile &uri) override;

  std::vector<llvm::lsp::FoldingRange>
  onFoldingRangeSync(const llvm::lsp::URIForFile &uri) override;

  std::optional<KGEN::Mojo::SignatureHelpResult>
  onSignatureHelpSyncImpl(llvm::SMLoc loc) override;

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The full string contents of the file.
  std::string contents;

  /// The AST decl for the module containing this document.
  MojoASTDeclRef parsedDecl;

  /// The doc strings within this document.
  MojoDocStrings docStrings;
};

using MojoTextDocumentRef = RCRef<MojoTextDocument>;

//===----------------------------------------------------------------------===//
// MojoNotebookDocument
//===----------------------------------------------------------------------===//

/// This class represents all of the information pertaining to a specific Mojo
/// notebook document, e.g. a jupyter notebook file.
struct MojoNotebookDocument : public MojoDocument {
public:
  /// This class represents a cell within the notebook.
  struct Cell {
    Cell(llvm::lsp::URIForFile uri, StringRef contents)
        : uri(std::move(uri)), contents(contents.str()) {}

    /// Return if this cell is a python cell.
    bool isPythonCell() const {
      return StringRef(contents).starts_with("%%python");
    }

    /// The uri of the cell
    llvm::lsp::URIForFile uri;

    /// The contents of the cell.
    std::string contents;

    /// The buffer id of the cell contents within the source manager.
    unsigned bufferId = 0;

    /// The AST decl for the module containing this cell.
    MojoASTDeclRef decl;

    /// The persistent REPL variables defined before this cell.
    SmallVector<std::pair<StringRef, Type>> persistentVariables;

    /// The doc strings within this cell.
    MojoDocStrings docStrings;
  };

  MojoNotebookDocument(ArrayRef<llvm::lsp::URIForFile> notebookAndCellURIs,
                       int64_t version,
                       ArrayRef<llvm::lsp::NotebookCell> cellInfos,
                       ArrayRef<llvm::lsp::TextDocumentItem> cellDocuments,
                       SendDiagnosticsFnRef sendDiagnosticsFn,
                       AsyncRT::CPUDevice &cpuDevice,
                       ArrayRef<std::string> includeDirs);
  MojoNotebookDocument(const MojoDocument &) = delete;
  MojoNotebookDocument &operator=(const MojoDocument &) = delete;

  /// Return the cells within this document.
  auto getCells() { return llvm::make_pointee_range(cells); }

  /// Support LLVM RTTI.
  static bool classof(const MojoDocument *doc) {
    return doc->getKind() == Kind::kNotebookDocument;
  }

private:
  //===--------------------------------------------------------------------===//
  // Derived Document Hooks
  //===--------------------------------------------------------------------===//

  /// Hook that is invoked to perform the raw document parsing process. Returns
  /// the number of bytes parsed by the Mojo parser.
  size_t parseDocumentImpl() override;

  /// Returns true if the document contains the given location.
  bool containsLocation(llvm::SMLoc loc) override;

  /// Translate the given parser location into one usable by the language
  /// server.
  llvm::SMLoc translateParserLoc(llvm::SMLoc loc) override;

  /// Returns true if the document contains the given location.
  llvm::SMLoc getLocFromPos(const llvm::lsp::URIForFile &uri,
                            llvm::lsp::Position position) override;

  /// Return a location range for the document of the given uri.
  llvm::SMRange getFullRangeForURI(const llvm::lsp::URIForFile &uri) override;

  /// Hook that returns the URI for the given contained location.
  const llvm::lsp::URIForFile &getURIFromContainedLoc(llvm::SMLoc loc) override;

  //===--------------------------------------------------------------------===//
  // Language Features

  std::vector<KGEN::Mojo::CodeCompletionResult>
  onCodeCompletionSyncImpl(llvm::SMLoc completeLoc) override;

  std::vector<llvm::lsp::DocumentSymbol>
  onDocumentSymbolSync(const llvm::lsp::URIForFile &uri) override;

  std::vector<llvm::lsp::FoldingRange>
  onFoldingRangeSync(const llvm::lsp::URIForFile &uri) override;

  std::optional<KGEN::Mojo::SignatureHelpResult>
  onSignatureHelpSyncImpl(llvm::SMLoc loc) override;

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Static Fields

  /// The following fields are always available for access and don't require
  /// additional synchronization.

  /// The cells within the document, mapped from the uri of the cell.
  llvm::StringMap<Cell *> uriToCell;
  std::vector<std::unique_ptr<Cell>> cells;
};

using MojoNotebookDocumentRef = RCRef<MojoNotebookDocument>;
} // namespace M::Mojo::LSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJODOCUMENT_H
