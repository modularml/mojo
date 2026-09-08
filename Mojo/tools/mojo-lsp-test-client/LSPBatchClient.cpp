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

#include "LSPBatchClient.h"
#include "../common/lsp-protocol/Protocol.h"
#include "Document.h"
#include "Mojo/Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Program.h"
#include <type_traits>

namespace lsp = llvm::lsp;
using namespace M;

/// Helper callback that ignores a response from the server. It helps
/// simplifying the response dispatch code.
static void doNothing(const llvm::json::Value &response) {}

LSPServerStdioFiles::LSPServerStdioFiles(const std::filesystem::path &parentDir)
    : serverStdin(parentDir / "stdin.txt"),
      serverStdout(parentDir / "stdout.txt"),
      serverStderr(parentDir / "stderr.txt") {}

LSPBatchClient::LSPBatchClient(
    bool attachDebugger,
    std::function<void(const ExecutionResult &)> onExecuteCallback)
    : onExecuteCallback(std::move(onExecuteCallback)),
      serverJSONInputOS(serverJSONInput), attachDebugger(attachDebugger),
      checkDocstrings(false) {
  llvm::json::Value initialize =
      llvm::json::Object{{"processId", 123},
                         {"rootPath", "mojo"},
                         {
                             "capabilities",
                             {
                                 {
                                     "window",
                                     {"workDoneProgress", false},
                                 },
                             },
                         },
                         {"trace", "off"}};
  request("initialize", initialize, std::function(doNothing));
}

LSPBatchClient &LSPBatchClient::setCheckDocstrings(bool value) {
  checkDocstrings = value;
  return *this;
}

LSPBatchClient &LSPBatchClient::open(const Document &doc) {
  lsp::DidOpenTextDocumentParams params{lsp::TextDocumentItem{
      doc.getURI(), "mojo", doc.getContents().str(), /*version=*/0}};
  notify("textDocument/didOpen", toJSON(params));
  return *this;
}

template <typename Result, typename Callback>
static std::vector<Result> mapDocuments(ArrayRef<Document> documents,
                                        Callback callback) {
  std::vector<Result> result;
  for (const Document &doc : documents)
    result.push_back(callback(doc));
  return result;
}

LSPBatchClient &LSPBatchClient::openNotebook(const NotebookDocument &doc) {
  lsp::DidOpenNotebookDocumentParams params{
      lsp::NotebookDocument{
          doc.getURI(), "jupyter",
          /*version=*/0,
          mapDocuments<lsp::NotebookCell>(doc.getCells(),
                                          [](const Document &cell) {
                                            return lsp::NotebookCell{
                                                lsp::NotebookCellKind::Code,
                                                cell.getURI()};
                                          })},
      mapDocuments<lsp::TextDocumentItem>(doc.getCells(),
                                          [](const Document &cell) {
                                            return lsp::TextDocumentItem{
                                                cell.getURI(),
                                                "mojo",
                                                cell.getContents().str(),
                                                /*version=*/0,
                                            };
                                          })};
  notify("notebookDocument/didOpen", toJSON(params));
  return *this;
}

LSPBatchClient &LSPBatchClient::notebookDidChange(
    const llvm::lsp::DidChangeNotebookDocumentParams &params) {
  notify("notebookDocument/didChange", toJSON(params));
  return *this;
}

LSPBatchClient &LSPBatchClient::definition(
    const Document &doc, const lsp::Position &position,
    std::function<void(const std::vector<lsp::Location> &)> callback) {
  lsp::TextDocumentPositionParams params{
      lsp::TextDocumentIdentifier{doc.getURI()}, position};
  request("textDocument/definition", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::semanticTokensFull(
    const Document &doc,
    std::function<void(ArrayRef<Mojo::LSP::SemanticToken>)> callback) {
  lsp::SemanticTokensParams params{lsp::TextDocumentIdentifier{doc.getURI()}};
  request("textDocument/semanticTokens/full", toJSON(params),
          std::function<void(const llvm::lsp::SemanticTokens &)>(
              [callback = std::move(callback)](
                  const llvm::lsp::SemanticTokens &tokens) {
                callback(Mojo::LSP::fromLspSemanticTokens(tokens.tokens));
              }));
  return *this;
}

LSPBatchClient &LSPBatchClient::signatureHelp(
    const Document &doc, const lsp::Position &position,
    std::function<void(const lsp::SignatureHelp2 &)> callback) {
  lsp::TextDocumentPositionParams params{
      lsp::TextDocumentIdentifier{doc.getURI()}, position};
  request("textDocument/signatureHelp", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::codeAction(
    const Document &doc, const lsp::Range &range,
    std::initializer_list<lsp::Diagnostic> diags,
    std::function<void(const std::vector<lsp::CodeAction> &)> callback) {
  lsp::CodeActionParams params{
      lsp::TextDocumentIdentifier{doc.getURI()}, range, {diags, /*only=*/{}}};
  request("textDocument/codeAction", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &
LSPBatchClient::hover(const Document &doc, const lsp::Position &position,
                      std::function<void(const lsp::Hover2 &)> callback) {
  lsp::TextDocumentPositionParams params{
      lsp::TextDocumentIdentifier{doc.getURI()}, position};
  request("textDocument/hover", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::hoverNullable(
    const Document &doc, const lsp::Position &position,
    std::function<void(const std::optional<lsp::Hover2> &)> callback) {
  lsp::TextDocumentPositionParams params{
      lsp::TextDocumentIdentifier{doc.getURI()}, position};
  request("textDocument/hover", toJSON(params), std::move(callback),
          /*allowNull=*/true);
  return *this;
}

LSPBatchClient &LSPBatchClient::rename(
    const Document &doc, const lsp::Position &position, std::string newName,
    std::function<void(const lsp::WorkspaceEdit &)> callback) {
  lsp::RenameParams params{lsp::TextDocumentIdentifier{doc.getURI()}, position,
                           std::move(newName)};
  request("textDocument/rename", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::renameError(
    const Document &doc, const lsp::Position &position, std::string newName,
    std::function<void(const llvm::lsp::LSPError2 &)> callback) {
  lsp::RenameParams params{lsp::TextDocumentIdentifier{doc.getURI()}, position,
                           std::move(newName)};
  request("textDocument/rename", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::documentSymbol(
    const Document &doc,
    std::function<void(const std::vector<lsp::DocumentSymbol> &)> callback) {
  lsp::DocumentSymbolParams params{lsp::TextDocumentIdentifier{doc.getURI()}};
  request("textDocument/documentSymbol", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::references(
    const Document &doc, const lsp::Position &pos, bool includeDeclaration,
    std::function<void(const std::vector<lsp::Location> &)> callback) {
  lsp::ReferenceParams params{{lsp::TextDocumentIdentifier{doc.getURI()}, pos},
                              lsp::ReferenceContext{includeDeclaration}};
  request("textDocument/references", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::foldingRange(
    const Document &doc,
    std::function<void(const std::vector<lsp::FoldingRange> &)> callback) {
  lsp::FoldingRangeParams params{{lsp::TextDocumentIdentifier{doc.getURI()}}};
  request("textDocument/foldingRange", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::completion(
    const Document &doc, const lsp::Position &position,
    std::function<void(const lsp::CompletionList &)> callback) {
  lsp::CompletionParams params{
      {lsp::TextDocumentIdentifier{doc.getURI()}, position},
      lsp::CompletionContext{lsp::CompletionTriggerKind::Invoked,
                             /*triggerCharacter*/ ""}};
  request("textDocument/completion", toJSON(params), std::move(callback));
  return *this;
}

LSPBatchClient &LSPBatchClient::update(const Document &doc,
                                       const llvm::lsp::Range &range,
                                       std::string change) {
  lsp::TextDocumentContentChangeEvent event{range, std::nullopt, change};
  lsp::DidChangeTextDocumentParams params{
      lsp::VersionedTextDocumentIdentifier{doc.getURI(), 0}, {event}};
  notify("textDocument/didChange", toJSON(params));
  return *this;
}

void LSPBatchClient::appendJSONRequest(RequestId id, StringRef method,
                                       const llvm::json::Value &params) {
  llvm::json::Value jsonRequest = llvm::json::Object{
      {"jsonrpc", "2.0"}, {"method", method}, {"id", id}, {"params", params}};
  serverJSONInputOS << llvm::formatv("{0:2}\n", jsonRequest) << "// -----\n";
}

void LSPBatchClient::appendShutdownAndExit() {
  request("shutdown", llvm::json::Object{}, std::function(doNothing));
  serverJSONInputOS << llvm::json::Object{{"jsonrpc", "2.0"},
                                          {"method", "exit"}}
                    << "\n";
}

ErrorOrSuccess LSPBatchClient::dispatchResponse(StringRef json) {
  if (llvm::Expected<llvm::json::Value> valueOr = llvm::json::parse(json)) {
    if (llvm::json::Object *obj = valueOr->getAsObject()) {
      // If we get an "id", then this is the response to a request.
      if (std::optional<int64_t> id = obj->getInteger("id")) {
        auto it = requestHandlers.find(*id);

        // There may not be a response handler for replayed requests, which
        // don't try to capture responses for analysis.
        if (it == requestHandlers.end())
          return success();

        // The request may have errored.
        if (auto err = obj->get("error")) {
          // If there's an error from the response handler, it's probably a
          // parse error about converting from the error value to the actually
          // expected type. In this case it's better to report the JSON.
          if (auto errOr = it->second->onResponse(*err)) {
            return Error(llvm::formatv("error while handling an error "
                                       "response: {0}\nJSON response:\n{1:2}",
                                       errOr.getError(), json));
          }
        } else if (auto errOr = it->second->onResponse(*obj->get("result")))
          return errOr;

        requestHandlers.erase(it);

        // Now we try to identify diagnostics.
      } else if (std::optional<StringRef> method = obj->getString("method")) {
        if (*method == "textDocument/publishDiagnostics") {
          llvm::json::Object &params = *obj->getObject("params");
          StringRef uri = *params.getString("uri");
          auto it = diagnosticsHandlers.find(uri);
          if (it != diagnosticsHandlers.end()) {
            auto diags =
                llvm::cantFail(llvm::json::parse<std::vector<lsp::Diagnostic>>(
                    *params.get("diagnostics")));
            it->second.front()(diags);
            it->second.pop_front();
            if (it->second.empty())
              diagnosticsHandlers.erase(it);
          }
        }
      }
      return success();
    }
  } else {
    // Return the entire malformed JSON below and hide this `llvm::Error`
    // because its error message is not very actionable.
    llvm::consumeError(valueOr.takeError());
  }
  return Error("Malformed JSON message returned by the server:\n" + json);
}

ErrorOrSuccess LSPBatchClient::dispatchResponses(StringRef serverStdout) {
  // A Language Server Protocol message starts with a set of headers,
  // delimited by \r\n, and terminated by an empty line (\r\n).

  // The following parser is extremely simple because its intention is not to
  // test the correctness of the JSON printing code.
  auto bufferOr = toModularErrorOr(llvm::MemoryBuffer::getFile(serverStdout));
  if (failed(bufferOr))
    return Error(Twine("Error reading the servers's output: ") +
                 bufferOr.getError());

  llvm::MemoryBuffer &buffer = *bufferOr->get();
  StringRef output = buffer.getBuffer();

  size_t contentLength;
  while (output.consume_front("Content-Length: ") &&
         !output.consumeInteger(10, contentLength) &&
         output.consume_front("\r\n\r\n")) {
    StringRef response = output.substr(0, contentLength);
    output = output.drop_front(contentLength);
    if (auto err = dispatchResponse(response))
      return err;
  }

  if (!output.empty())
    return Error("Malformed server output. Not all data could be parsed");

  if (!requestHandlers.empty()) {
    std::string errorMsg;
    llvm::raw_string_ostream os(errorMsg);
    os << "Not all requests received a response: ";
    llvm::interleave(llvm::make_first_range(requestHandlers), os, " ");
    return Error(errorMsg);
  }

  if (!diagnosticsHandlers.empty()) {
    std::string errorMsg;
    llvm::raw_string_ostream os(errorMsg);
    os << "Not all diagnostic handlers received a corresponding diagnostic "
          "notification:";
    // StringMap doesn't work with make_first_range, so we have to do a classic
    // iteration.
    for (const auto &[key, _] : diagnosticsHandlers)
      os << " " << key;
    return Error(errorMsg);
  }

  return success();
}

LSPBatchClient &LSPBatchClient::onDiagnostics(const Document &doc,
                                              DiagnosticHandler handler) {
  auto &list =
      diagnosticsHandlers.try_emplace(doc.getURI().uri()).first->second;
  list.emplace_back(std::move(handler));
  return *this;
}

void LSPBatchClient::replayRequest(StringRef method,
                                   const llvm::json::Value &params) {
  RequestId id = requestId++;
  appendJSONRequest(id, method, params);
}

void LSPBatchClient::replayNotification(StringRef method,
                                        const llvm::json::Value &params) {
  notify(method, params);
}

ErrorOrSuccess LSPBatchClient::doExecute(const LSPServerStdioFiles &ioFiles,
                                         StringRef lspServerPath) {
  appendShutdownAndExit();

  if (llvm::Error err =
          llvm::writeToOutput(ioFiles.serverStdin, [&](raw_ostream &os) {
            os << serverJSONInput;
            return llvm::Error::success();
          })) {
    return Error(Twine("Error writing the server's input file: ") +
                 toModularError(std::move(err)).get());
  }

  std::string errMsg;
  llvm::SmallVector<StringRef> args = {lspServerPath, "-mojo-test"};
  if (attachDebugger)
    args.push_back("-attach-debugger-on-startup");
  // Pass the value explicitly so the client controls the behavior regardless of
  // the server's default (which is -check-docstrings=false).
  args.push_back(checkDocstrings ? "-check-docstrings=true"
                                 : "-check-docstrings=false");

  int exitCode = llvm::sys::ExecuteAndWait(lspServerPath, args,
                                           /*Env=*/std::nullopt, /*redirects=*/
                                           {
                                               ioFiles.serverStdin,
                                               ioFiles.serverStdout,
                                               ioFiles.serverStderr,
                                           },
                                           /*SecondsToWait=*/0,
                                           /*MemoryLimit=*/0,
                                           /*ErrMsg=*/&errMsg);
  if (exitCode != 0) {
    return Error(llvm::formatv("Server failed with exit code {0}. {1}",
                               exitCode, errMsg));
  }

  return dispatchResponses(ioFiles.serverStdout);
}

LSPBatchClient::ExecutionResult LSPBatchClient::execute() {
  if (std::exchange(didExecute, true))
    llvm::report_fatal_error("`LSPBatchClient::execute` invoked twice");

  // Find the path to the LLDB executable and the MojoLLDB plugin library.
  // Read the mojo configuration.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr))
    return {
        Error(Twine("Failed to parse 'modular.cfg': ") + configOr.getError())};

  StringRef lspServerPath = configOr->getLSPServerPath();

  std::filesystem::path tempDirPath = "lsp-test-client.%%%%%%";
  if (auto bazelTempDir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR"))
    tempDirPath = bazelTempDir / tempDirPath;

  ErrorOr<TempDir> tempDirOr = TempDir::create(tempDirPath.generic_string());
  if (failed(tempDirOr)) {
    llvm::errs() << "tmp dir failed: " << tempDirOr.takeError() << "\n";
    return {tempDirOr.takeError()};
  }
  LSPServerStdioFiles ioFiles(tempDirOr->getPath());

  bool preserveIOFiles = std::getenv("PRESERVE_LSP_IO_FILES");

  // Keep the temp dir and print the file paths so they can be inspected.
  auto keepAndPrintIOFiles = [&] {
    tempDirOr->keep();
    llvm::errs() << "Language server stdin: " << ioFiles.serverStdin << "\n";
    llvm::errs() << "Language server stdout: " << ioFiles.serverStdout << "\n";
    llvm::errs() << "Language server stderr: " << ioFiles.serverStderr << "\n";
  };

  // Print paths before executing so they are visible even if the server
  // crashes hard and doExecute never returns cleanly.
  if (preserveIOFiles)
    keepAndPrintIOFiles();

  LSPBatchClient::ExecutionResult result;
  if (auto err = doExecute(ioFiles, lspServerPath)) {
    result.err = std::move(err);
    // Also print on failure when not already printed above.
    if (!preserveIOFiles)
      keepAndPrintIOFiles();
  }
  onExecuteCallback(result);
  result.serverIOFiles = std::move(ioFiles);
  return result;
}

LSPBatchClient::~LSPBatchClient() {
  if (!didExecute) {
    llvm::report_fatal_error("LSPBatchClient being destroyed without "
                             "`execute` having been invoked.");
  }
}

void LSPBatchClient::notify(StringRef method, const llvm::json::Value &params) {
  llvm::json::Value notification = llvm::json::Object{
      {"jsonrpc", "2.0"}, {"method", method}, {"params", params}};
  serverJSONInputOS << llvm::formatv("{0:2}\n", notification) << "// -----\n";
}
