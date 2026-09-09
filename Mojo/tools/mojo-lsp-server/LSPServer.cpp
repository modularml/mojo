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

#include "LSPServer.h"
#include "../common/lsp-protocol/Protocol.h"
#include "../common/lsp-protocol/SemanticTokens.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "MojoServer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Transport.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Transport.h"
#include <optional>

#define DEBUG_TYPE "mojo-lsp-server"

using namespace llvm::lsp;
using namespace M;
using namespace M::Mojo::LSP;

/// Wrapper around MessageHandler for handling requests using an `LSPResponder`.
class LSPRequestHandler {
public:
  LSPRequestHandler(LSPTelemetryContext &lspTelemetryCtx,
                    MessageHandler &messageHandler)
      : messageHandler(messageHandler), lspTelemetryCtx(lspTelemetryCtx) {}

  /// Wrapper around `MessageHandler::method` that provides an `LSPResponder` to
  /// the underlying implementation.
  template <typename Param, typename Result, typename ThisT>
  void method(llvm::StringLiteral method, ThisT *thisPtr,
              void (ThisT::*handler)(const Param &, LSPResponder<Result>)) {
    struct Handler {
      void invoke(const Param &param, Callback<Result> reply) {
        KGEN::CompilerTimeTraceScope traceScope("handleRequest",
                                                [&]() { return method.str(); });

        size_t parentId = 0;

        (thisPtr->*handler)(param,
                            LSPResponder<Result>(lspTelemetryCtx, method,
                                                 std::move(reply), parentId));
      }
      LSPTelemetryContext &lspTelemetryCtx;
      StringRef method;
      ThisT *thisPtr;
      void (ThisT::*handler)(const Param &, LSPResponder<Result>);
    };
    auto *wrappedHandlerPtr = new (allocator.Allocate<Handler>())
        Handler{lspTelemetryCtx, method, thisPtr, handler};
    messageHandler.method(method, wrappedHandlerPtr, &Handler::invoke);
  }

private:
  llvm::BumpPtrAllocator allocator;
  MessageHandler &messageHandler;
  LSPTelemetryContext &lspTelemetryCtx;
};

//===----------------------------------------------------------------------===//
// LSPServer
//===----------------------------------------------------------------------===//

namespace {
struct LSPServer {
  LSPServer(MojoServer &server, JSONTransport &transport,
            std::unique_ptr<KGEN::TraceProfiler> profiler)
      : server(server), transport(transport), profiler(std::move(profiler)) {}

  //===--------------------------------------------------------------------===//
  // Initialization

  void onInitialize(const InitializeParams &params,
                    LSPResponder<llvm::json::Value> responder);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params,
                  LSPResponder<std::nullptr_t> responder);

  //===--------------------------------------------------------------------===//
  // Document Change

  void onDocumentDidOpen(const DidOpenTextDocumentParams &params);
  void onDocumentDidClose(const DidCloseTextDocumentParams &params);
  void onDocumentDidChange(const DidChangeTextDocumentParams &params);

  void onNotebookDocumentDidOpen(const DidOpenNotebookDocumentParams &params);
  void onNotebookDocumentDidClose(const DidCloseNotebookDocumentParams &params);
  void
  onNotebookDocumentDidChange(const DidChangeNotebookDocumentParams &params);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  MojoServer &server;
  JSONTransport &transport;

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool shutdownRequestReceived = false;

  std::unique_ptr<KGEN::TraceProfiler> profiler;
};
} // namespace

//===----------------------------------------------------------------------===//
// Initialization

/// Return the set of supported semantic token types.
static std::vector<StringRef> semanticTokenTypes() {
  std::vector<StringRef> types;
  for (int i : llvm::seq(0, static_cast<int>(SemanticTokenKind::kCount)))
    types.push_back(toLspSemanticTokenType(static_cast<SemanticTokenKind>(i)));
  return types;
}

/// Return the set of supported semantic token modifiers.
static std::vector<StringRef> semanticTokenModifiers() {
  std::vector<StringRef> modifiers;
  for (int i : llvm::seq(0, static_cast<int>(SemanticTokenModifier::kCount)))
    modifiers.push_back(
        toLspSemanticTokenModifier(static_cast<SemanticTokenModifier>(i)));
  return modifiers;
}

void LSPServer::onInitialize(const InitializeParams &params,
                             LSPResponder<llvm::json::Value> responder) {
  using JSONArray = llvm::json::Array;
  using JSONObject = llvm::json::Object;

  // Send a response with the capabilities of this server.
  JSONObject serverCaps{
      {"completionProvider",
       JSONObject{
           {"allCommitCharacters", {"\t"}},
           {"resolveProvider", false},
           {"triggerCharacters", {"."}},
       }},
      {"signatureHelpProvider",
       llvm::json::Object{
           {"triggerCharacters", {"(", "[", ","}},
       }},
      {"definitionProvider", true},
      // Note: foldingRangeProvider is deliberately NOT advertised. The current
      // implementation only returns docstring ranges (see
      // MojoTextDocument::onFoldingRangeSync), and advertising it makes VSCode
      // disable its indentation-based folding entirely — leaving users unable
      // to fold functions, structs, and blocks. Re-add once the server returns
      // structural folding ranges.
      {"hoverProvider", true},
      {"inlayHintProvider", true},
      {"notebookDocumentSync",
       JSONObject{{
           "notebookSelector",
           JSONArray{JSONObject{
               {"notebook", JSONObject{{"scheme", "file"},
                                       {"notebookType", "jupyter-notebook"}}},
               {"cells", JSONArray{JSONObject{{"language", "mojo"}}}},
           }},
       }}},
      {"referencesProvider", true},
      {"renameProvider", true},
      {"semanticTokensProvider",
       llvm::json::Object{
           {"full", llvm::json::Object{{"delta", true}}},
           {"range", false},
           {"legend",
            llvm::json::Object{{"tokenTypes", semanticTokenTypes()},
                               {"tokenModifiers", semanticTokenModifiers()}}},
       }},
      {
          "textDocumentSync",
          JSONObject{
              {"openClose", true},
              {"change", (int)TextDocumentSyncKind::Incremental},
              {"save", true},
          },
      },

      // For now we only support documenting symbols when the client supports
      // hierarchical symbols.
      {"documentSymbolProvider",
       params.capabilities.hierarchicalDocumentSymbol},
  };

  server.receiveCapabilities(params.capabilities.workDoneProgress);

  // Per LSP, codeActionProvider can be either boolean or CodeActionOptions.
  // CodeActionOptions is only valid if the client supports action literal
  // via textDocument.codeAction.codeActionLiteralSupport.
  serverCaps["codeActionProvider"] =
      params.capabilities.codeActionStructure
          ? llvm::json::Object{{"codeActionKinds",
                                {CodeAction::kQuickFix, CodeAction::kRefactor,
                                 CodeAction::kInfo}}}
          : llvm::json::Value(true);

  llvm::json::Object result{
      {{"serverInfo",
        llvm::json::Object{{"name", "mojo-lsp-server"}, {"version", "0.0.1"}}},
       {"capabilities", std::move(serverCaps)}}};
  responder.reply(std::move(result));

  server.getLSPTelemetryContext().reportInitialization(
      params.clientInfo ? std::make_optional<StringRef>(params.clientInfo->name)
                        : std::nullopt);
}
void LSPServer::onInitialized(const InitializedParams &) {}
void LSPServer::onShutdown(const NoParams &,
                           LSPResponder<std::nullptr_t> responder) {
  profiler.reset();

  server.getLSPTelemetryContext().reportShutdown();
  server.getLSPTelemetryContext().flush();
  server.shutdown();
  shutdownRequestReceived = true;
  responder.reply(nullptr);
}

//===----------------------------------------------------------------------===//
// Document Change

void LSPServer::onDocumentDidOpen(const DidOpenTextDocumentParams &params) {
  Logger::debug("--> textDocument/didOpen: uri='{0}', version={1}",
                params.textDocument.uri, params.textDocument.version);
  server.addDocument(params.textDocument.uri,
                     std::string(params.textDocument.text),
                     params.textDocument.version);
}

void LSPServer::onDocumentDidClose(const DidCloseTextDocumentParams &params) {
  server.removeDocument(params.textDocument.uri);
}

void LSPServer::onDocumentDidChange(const DidChangeTextDocumentParams &params) {
  Logger::debug("--> textDocument/didChange: uri='{0}', version={1}",
                params.textDocument.uri, params.textDocument.version);
  server.updateDocument(params.textDocument.uri, params.contentChanges,
                        params.textDocument.version);
}

void LSPServer::onNotebookDocumentDidOpen(
    const DidOpenNotebookDocumentParams &params) {
  Logger::debug("--> notebookDocument/didOpen: uri='{0}', version={1}",
                params.notebookDocument.uri, params.notebookDocument.version);
  server.addNotebookDocument(
      params.notebookDocument.uri, params.notebookDocument.cells,
      params.notebookDocument.version, params.cellTextDocuments);
}

void LSPServer::onNotebookDocumentDidClose(
    const DidCloseNotebookDocumentParams &params) {
  server.removeNotebookDocument(params.notebookDocument.uri,
                                params.cellTextDocuments);
}

void LSPServer::onNotebookDocumentDidChange(
    const DidChangeNotebookDocumentParams &params) {
  Logger::debug("--> notebookDocument/didChange: uri='{0}', version={1}",
                params.notebookDocument.uri, params.notebookDocument.version);
  server.updateNotebookDocument(params.notebookDocument.uri,
                                params.notebookDocument.version, params.change);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

mlir::LogicalResult
M::KGEN::LIT::runMojoLSPServer(JSONTransport &transport, bool singleThreaded,
                               bool waitOnShutdown,
                               ArrayRef<std::string> includeDirs,
                               std::unique_ptr<KGEN::TraceProfiler> profiler,
                               bool checkDocstringCodeBlocks) {
  MessageHandler messageHandler(transport);
  ErrorOr<MojoServer> serverOr =
      MojoServer::create(singleThreaded, waitOnShutdown, messageHandler,
                         includeDirs, checkDocstringCodeBlocks);
  if (serverOr.isError()) {
    auto error = llvm::make_error<llvm::StringError>(
        serverOr.getError(), llvm::inconvertibleErrorCode());
    Logger::error("Server creation error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
  MojoServer server(serverOr.takeValue());
  LSPRequestHandler requestHandler(server.getLSPTelemetryContext(),
                                   messageHandler);
  LSPServer lspServer(server, transport, std::move(profiler));

  // Initialization
  requestHandler.method("initialize", &lspServer, &LSPServer::onInitialize);
  messageHandler.notification("initialized", &lspServer,
                              &LSPServer::onInitialized);
  requestHandler.method("shutdown", &lspServer, &LSPServer::onShutdown);

  // Document Changes
  messageHandler.notification("notebookDocument/didOpen", &lspServer,
                              &LSPServer::onNotebookDocumentDidOpen);
  messageHandler.notification("notebookDocument/didClose", &lspServer,
                              &LSPServer::onNotebookDocumentDidClose);
  messageHandler.notification("notebookDocument/didChange", &lspServer,
                              &LSPServer::onNotebookDocumentDidChange);
  messageHandler.notification("textDocument/didOpen", &lspServer,
                              &LSPServer::onDocumentDidOpen);
  messageHandler.notification("textDocument/didClose", &lspServer,
                              &LSPServer::onDocumentDidClose);
  messageHandler.notification("textDocument/didChange", &lspServer,
                              &LSPServer::onDocumentDidChange);

  // Code Action
  requestHandler.method("textDocument/codeAction", &server,
                        &MojoServer::getCodeActions);

  // Language Features
  requestHandler.method("textDocument/completion", &server,
                        &MojoServer::onCodeCompletion);
  requestHandler.method("textDocument/definition", &server,
                        &MojoServer::onDefinition);
  requestHandler.method("textDocument/documentSymbol", &server,
                        &MojoServer::onDocumentSymbol);
  requestHandler.method("textDocument/foldingRange", &server,
                        &MojoServer::onFoldingRange);
  requestHandler.method("textDocument/hover", &server, &MojoServer::onHover);
  requestHandler.method("textDocument/inlayHint", &server,
                        &MojoServer::onInlayHint);
  requestHandler.method("textDocument/references", &server,
                        &MojoServer::onReferences);
  requestHandler.method("textDocument/semanticTokens/full", &server,
                        &MojoServer::onSemanticTokens);
  requestHandler.method("textDocument/semanticTokens/full/delta", &server,
                        &MojoServer::onSemanticTokensDelta);
  requestHandler.method("textDocument/signatureHelp", &server,
                        &MojoServer::getSignatureHelp);
  requestHandler.method("textDocument/rename", &server, &MojoServer::onRename);

#ifndef MODULAR_PRODUCTION
  messageHandler.notification("mojo/emitParsedIR", &server,
                              &MojoServer::dumpParsedIR);
#endif // MODULAR_PRODUCTION

  // Run the main loop of the transport.
  if (llvm::Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
  return success(lspServer.shutdownRequestReceived);
}
