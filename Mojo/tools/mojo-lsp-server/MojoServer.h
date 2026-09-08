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

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Transport.h"
#include "llvm/ADT/FunctionExtras.h"

namespace M::AsyncRT {
class WorkQueue;
} // namespace M::AsyncRT

namespace M::Mojo::LSP {
using SendDiagnosticsFn =
    llvm::unique_function<void(const llvm::lsp::PublishDiagnosticsParams &)>;
using SendDiagnosticsFnRef =
    function_ref<void(const llvm::lsp::PublishDiagnosticsParams &)>;
template <typename T>
using OnSemanticTokensResultFn =
    llvm::unique_function<void(T result, bool outdated, bool invalid)>;

template <typename T>
using SendProgressFn =
    llvm::unique_function<void(const llvm::lsp::ProgressParams<T> &)>;

template <typename T>
using SendProgressFnRef =
    function_ref<void(const llvm::lsp::ProgressParams<T> &)>;

/// This class implements all of the Mojo related functionality necessary for a
/// language server. This class allows for keeping the Mojo specific logic
/// separate from the logic that involves LSP server/client communication.
class MojoServer {
  struct Impl;

public:
  MojoServer(MojoServer &) = delete;
  MojoServer(MojoServer &&);
  ~MojoServer();

  /// Create a new MojoServer instance.
  static ErrorOr<MojoServer> create(bool singleThreaded, bool waitOnShutdown,
                                    llvm::lsp::MessageHandler &messageHandler,
                                    ArrayRef<std::string> includeDirs,
                                    bool checkDocstringCodeBlocks = false);

  // Get the telemetry context for this server.
  LSPTelemetryContext &getLSPTelemetryContext();

  /// Begin the shutdown sequence for the server.
  void shutdown();

  /// Receive client capabilities from the LSPServer transport layer.
  void receiveCapabilities(bool workDoneProgress);

  //===--------------------------------------------------------------------===//
  // Document Management
  //===--------------------------------------------------------------------===//

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document will be added to `diagnostics`.
  void addDocument(const llvm::lsp::URIForFile &uri, std::string &&contents,
                   int64_t version);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document will be added to `diagnostics`.
  void
  updateDocument(const llvm::lsp::URIForFile &uri,
                 ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
                 int64_t version);

  /// Remove the document with the given uri.
  void removeDocument(const llvm::lsp::URIForFile &uri);

  /// Returns true if there is a pending (not-yet-parsed) content update for
  /// the given file URI. Used by request handlers to distinguish a transient
  /// stale-document state from a genuinely invalid request: when a position
  /// fails to map in the current document but a newer, unparsed version is
  /// already queued, the request should be reported as ContentModified so
  /// clients retry against the updated document.
  bool hasPendingUpdate(llvm::StringRef file) const;

  //===--------------------------------------------------------------------===//
  // Notebook Document Management
  //===--------------------------------------------------------------------===//

  /// Add the notebook document, with the provided `version`, at the given URI.
  /// Any diagnostics emitted for this document will be added to `diagnostics`.
  void addNotebookDocument(const llvm::lsp::URIForFile &uri,
                           ArrayRef<llvm::lsp::NotebookCell> cells,
                           int64_t version,
                           ArrayRef<llvm::lsp::TextDocumentItem> cellDocuments);

  /// Remove the notebook document with the given uri.
  void removeNotebookDocument(
      const llvm::lsp::URIForFile &uri,
      ArrayRef<llvm::lsp::TextDocumentIdentifier> cellDocuments);

  /// Update the document, with the provided `version`, at the given URI.
  void
  updateNotebookDocument(const llvm::lsp::URIForFile &uri, int64_t version,
                         const llvm::lsp::NotebookDocumentChangeEvent &change);

  //===--------------------------------------------------------------------===//
  // Queries
  //===--------------------------------------------------------------------===//

  /// Get the set of code actions within the file.
  void
  getCodeActions(const llvm::lsp::CodeActionParams &params,
                 LSPResponder<std::vector<llvm::lsp::CodeAction>> responder);

  /// Get the code completion list for the position within the given file.
  void onCodeCompletion(const llvm::lsp::CompletionParams &params,
                        LSPResponder<llvm::lsp::CompletionList> responder);

  /// Get the identifier location of the symbol declarations that contain the
  /// given position.
  void onDefinition(const llvm::lsp::TextDocumentPositionParams &params,
                    LSPResponder<std::vector<llvm::lsp::Location>> responder);

  /// Find all of the document symbols within the given file.
  void onDocumentSymbol(
      const llvm::lsp::DocumentSymbolParams &params,
      LSPResponder<std::vector<llvm::lsp::DocumentSymbol>> responder);

  /// Find all of the folding ranges within the given file.
  void
  onFoldingRange(const llvm::lsp::FoldingRangeParams &params,
                 LSPResponder<std::vector<llvm::lsp::FoldingRange>> responder);

  /// Get a `Hover` element corresponding to the given document position.
  void onHover(const llvm::lsp::TextDocumentPositionParams &params,
               LSPResponder<std::optional<llvm::lsp::Hover>> responder);

  /// Get inlay hints for the given document range.
  void onInlayHint(const llvm::lsp::InlayHintsParams &params,
                   LSPResponder<std::vector<llvm::lsp::InlayHint>> responder);

  // Get the references of the symbol in the given location.
  void onReferences(const llvm::lsp::ReferenceParams &params,
                    LSPResponder<std::vector<llvm::lsp::Location>> responder);

  /// Get the semantic tokens for the given document.
  void onSemanticTokens(
      const llvm::lsp::SemanticTokensParams &params,
      LSPResponder<std::optional<llvm::lsp::SemanticTokens>> responder);

  /// Get the delta of semantic tokens for the given document compared to the
  /// tokens at the given identifier (representing a previous result).
  void onSemanticTokensDelta(
      const llvm::lsp::SemanticTokensDeltaParams &params,
      LSPResponder<std::optional<llvm::lsp::SemanticTokensOrDelta>> responder);

  /// Get the signature help for the position within the given document.
  void getSignatureHelp(const llvm::lsp::TextDocumentPositionParams &params,
                        LSPResponder<llvm::lsp::SignatureHelp2> responder);

  /// Perform a rename operation at the position within the given document.
  void onRename(const llvm::lsp::RenameParams &params,
                LSPResponder<llvm::lsp::WorkspaceEdit> responder);

  /// Dump the parsed MLIR to a file for inspection.
  ///
  /// This is only available in debug builds.
  void dumpParsedIR(const llvm::lsp::TextDocumentIdentifier &params);

private:
  MojoServer(std::unique_ptr<Impl> &&);
  std::unique_ptr<Impl> impl;
};

} // namespace M::Mojo::LSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
