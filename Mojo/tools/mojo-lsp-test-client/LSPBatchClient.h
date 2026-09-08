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

#ifndef KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_LSPBATCHCLIENT_H
#define KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_LSPBATCHCLIENT_H

#include "../common/lsp-protocol/Protocol.h"
#include "../common/lsp-protocol/SemanticTokens.h"
#include "Document.h"
#include "JSONUtils.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ADT/StringMap.h"
#include <deque>

namespace M {

/// Helper struct containing the paths to the files that are used as stdio when
/// invoking the LSP server.
struct LSPServerStdioFiles {
  LSPServerStdioFiles(const std::filesystem::path &parentDir);

  std::string serverStdin;
  std::string serverStdout;
  std::string serverStderr;
};

/// Non-interactive batch client for the Mojo Language Server that collects
/// requests and sends them as a single JSON input to the server at once when
/// the `execute` method is invoked. Each registered request has an assigned
/// callback that is invoked when the corresponding response from the server is
/// received.
///
/// This class aborts if it is destroyed and the `execute` method hasn't been
/// invoked, or if such method is executed twice.
class LSPBatchClient {
private:
  using DiagnosticHandler =
      std::function<void(const std::vector<llvm::lsp::Diagnostic> &)>;

  /// Type-erasure class used to dispatch a response of an LSP request.
  class ResponseHandler {
  public:
    virtual ~ResponseHandler() = default;
    virtual ErrorOrSuccess onResponse(const llvm::json::Value &response) = 0;
  };

  /// Implementation of `ResponseHandler` with a specific Result type.
  template <typename Result>
  class ResponseHandlerImpl : public ResponseHandler {
  public:
    ResponseHandlerImpl(std::function<void(const Result &)> callback,
                        bool allowNull = false)
        : callback(std::move(callback)), allowNull(allowNull) {}

    ErrorOrSuccess onResponse(const llvm::json::Value &response) override {
      // Some requests might just want the raw JSON value.
      if constexpr (std::is_same_v<llvm::json::Value, Result>) {
        callback(response);
      } else {
        if (auto resultOr = llvm::json::parse<Result>(response))
          callback(*resultOr);
        else if (allowNull)
          callback({});
        else
          return toModularErrorOr(resultOr.takeError());
      }
      return success();
    }

  private:
    std::function<void(const Result &)> callback;
    bool allowNull;
  };

public:
  using RequestId = int;

  struct ExecutionResult {
    ErrorOrSuccess err;
    std::optional<LSPServerStdioFiles> serverIOFiles = std::nullopt;
  };

  LSPBatchClient(
      bool attachDebugger = false,
      std::function<void(const ExecutionResult &)> onExecuteCallback =
          [](const ExecutionResult &) {});

  /// When called before execute(), passes -check-docstrings to the server so
  /// that code blocks inside doc strings are parsed and type-checked. Disabled
  /// by default, matching the server.
  LSPBatchClient &setCheckDocstrings(bool value = true);

  ~LSPBatchClient();

  /// textDocument/didOpen
  LSPBatchClient &open(const Document &doc);

  /// notebookDocument/didOpen
  LSPBatchClient &openNotebook(const NotebookDocument &doc);

  /// notebookDocument/didChange
  LSPBatchClient &
  notebookDidChange(const llvm::lsp::DidChangeNotebookDocumentParams &params);

  /// textDocument/definition
  LSPBatchClient &definition(
      const Document &doc, const llvm::lsp::Position &position,
      std::function<void(const std::vector<llvm::lsp::Location> &)> callback);

  /// textDocument/semanticTokens/full
  LSPBatchClient &semanticTokensFull(
      const Document &doc,
      std::function<void(ArrayRef<Mojo::LSP::SemanticToken>)> callback);

  /// textDocument/signatureHelp
  LSPBatchClient &signatureHelp(
      const Document &doc, const llvm::lsp::Position &position,
      std::function<void(const llvm::lsp::SignatureHelp2 &)> callback);

  /// testDocument/codeAction
  LSPBatchClient &codeAction(
      const Document &doc, const llvm::lsp::Range &range,
      std::initializer_list<llvm::lsp::Diagnostic> diags,
      std::function<void(const std::vector<llvm::lsp::CodeAction> &)> callback);

  /// textDocument/hover
  LSPBatchClient &
  hover(const Document &doc, const llvm::lsp::Position &position,
        std::function<void(const llvm::lsp::Hover2 &)> callback);

  LSPBatchClient &hoverNullable(
      const Document &doc, const llvm::lsp::Position &position,
      std::function<void(const std::optional<llvm::lsp::Hover2> &)> callback);

  /// textDocument/rename
  LSPBatchClient &
  rename(const Document &doc, const llvm::lsp::Position &position,
         std::string newName,
         std::function<void(const llvm::lsp::WorkspaceEdit &)> callback);

  /// textDocument/rename (error variant)
  LSPBatchClient &
  renameError(const Document &doc, const llvm::lsp::Position &position,
              std::string newName,
              std::function<void(const llvm::lsp::LSPError2 &)> callback);

  /// textDocument/documentSymbol
  LSPBatchClient &documentSymbol(
      const Document &doc,
      std::function<void(const std::vector<llvm::lsp::DocumentSymbol> &)>
          callback);

  /// textDocument/references
  LSPBatchClient &references(
      const Document &doc, const llvm::lsp::Position &pos,
      bool includeDeclaration,
      std::function<void(const std::vector<llvm::lsp::Location> &)> callback);

  /// textDocument/foldingRange
  LSPBatchClient &
  foldingRange(const Document &doc,
               std::function<void(const std::vector<llvm::lsp::FoldingRange> &)>
                   callback);

  /// textDocument/completion
  LSPBatchClient &
  completion(const Document &doc, const llvm::lsp::Position &position,
             std::function<void(const llvm::lsp::CompletionList &)> callback);

  /// textDocument/diagnostic
  LSPBatchClient &onDiagnostics(const Document &doc, DiagnosticHandler handler);

  /// Dispatch a document update
  LSPBatchClient &update(const Document &doc, const llvm::lsp::Range &range,
                         std::string change);

  /// Actual `execute` logic.
  ErrorOrSuccess doExecute(const LSPServerStdioFiles &ioFiles,
                           StringRef lspServerPath);

  /// Append an arbitrary request to the server input stream. The response will
  /// not be capturable.
  ///
  /// This is used to support replaying recorded LSP sessions.
  void replayRequest(StringRef method, const llvm::json::Value &params);

  /// Append an arbitrary notification to the server input stream.
  ///
  /// This is used to support replaying recorded LSP sessions.
  void replayNotification(StringRef method, const llvm::json::Value &params);

  /// Create a single JSON input for the server, then parse the response and
  /// invoke the corresponding response callbacks.
  ///
  /// This method aborts if it's invoked twice.
  ExecutionResult execute();

private:
  friend class LSPBatchClientExecutor;

  /// Register a notification to be sent to the server.
  void notify(StringRef method, const llvm::json::Value &params);

  /// Register a request to be sent to the server.
  template <typename Result>
  void request(StringRef method, const llvm::json::Value &params,
               std::function<void(const Result &)> callback,
               bool allowNull = false) {
    RequestId id = requestId++;
    requestHandlers.try_emplace(
        id, std::unique_ptr<ResponseHandler>(new ResponseHandlerImpl<Result>(
                std::move(callback), allowNull)));

    appendJSONRequest(id, method, params);
  }

  /// Append a JSON request to the server input stream.
  void appendJSONRequest(RequestId id, StringRef method,
                         const llvm::json::Value &params);

  /// Append to the server input the shutdown and exit messages, which marks
  /// the end of the input.
  void appendShutdownAndExit();

  /// Dispatch a single response given its JSON contents.
  ErrorOrSuccess dispatchResponse(StringRef json);

  /// Parses the JSON output from the server and invoke the corresponding
  /// handlers, erasing them as they are responded.
  ErrorOrSuccess dispatchResponses(StringRef serverStdout);

  std::function<void(const ExecutionResult &)> onExecuteCallback;
  /// Flag that indicates whether `execute` has been invoked.
  bool didExecute = false;
  /// The collected JSON input to send to the server.
  std::string serverJSONInput;
  /// An output stream for `jsonInput`.
  llvm::raw_string_ostream serverJSONInputOS;
  /// A monotonically increasing request id used when collecting requests.
  RequestId requestId = 0;
  /// A map from request id to response handler.
  DenseMap<RequestId, std::unique_ptr<ResponseHandler>> requestHandlers;
  /// A map from doc URL to response handler.
  llvm::StringMap<std::deque<DiagnosticHandler>> diagnosticsHandlers;
  /// A flag indicating that a mojo-lsp-server will be launched with a
  /// debugger attached.
  bool attachDebugger;
  /// When true, pass -check-docstrings to the server, enabling parsing and
  /// type-checking of code blocks inside doc strings.
  bool checkDocstrings;
};

} // namespace M

#endif // KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_LSPBATCHCLIENT_H
