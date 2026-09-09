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

#include "Protocol.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace lsp = llvm::lsp;
using namespace lsp;
using namespace M;

// Helper that doesn't treat `null` and absent fields as failures.
template <typename T>
static bool mapOptOrNull(const llvm::json::Value &params,
                         llvm::StringLiteral prop, T &out,
                         llvm::json::Path path) {
  const llvm::json::Object *o = params.getAsObject();
  assert(o);

  // Field is missing or null.
  auto *v = o->get(prop);
  if (!v || v->getAsNull())
    return true;
  return fromJSON(*v, out, path.field(prop));
}

//===----------------------------------------------------------------------===//
// SignatureInformation
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         SignatureInformation2 &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("label", result.label) &&
         o.map("parameters", result.parameters) &&
         o.mapOptional("documentation", result.documentation);
}

llvm::json::Value llvm::lsp::toJSON(const SignatureInformation2 &value) {
  assert(!value.label.empty() && "signature information label is required");
  llvm::json::Object result{
      {"label", value.label},
      {"parameters", llvm::json::Array(value.parameters)},
  };
  if (value.documentation)
    result["documentation"] = value.documentation;
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// SignatureHelp
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value, SignatureHelp2 &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("signatures", result.signatures) &&
         o.map("activeSignature", result.activeSignature) &&
         o.map("activeParameter", result.activeParameter);
}

llvm::json::Value llvm::lsp::toJSON(const SignatureHelp2 &value) {
  assert(value.activeSignature >= 0 &&
         "Unexpected negative value for number of active signatures.");
  assert(value.activeParameter >= 0 &&
         "Unexpected negative value for active parameter index");
  return llvm::json::Object{
      {"activeSignature", value.activeSignature},
      {"activeParameter", value.activeParameter},
      {"signatures", llvm::json::Array(value.signatures)},
  };
}

//===----------------------------------------------------------------------===//
// NotebookCell
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value, NotebookCell &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;

  int kind = 0;
  if (!mapOptOrNull(value, "kind", kind, path))
    return false;
  result.kind = (NotebookCellKind)kind;

  return o.map("document", result.document);
}

llvm::json::Value llvm::lsp::toJSON(const NotebookCell &value) {
  return llvm::json::Object{{"kind", static_cast<int64_t>(value.kind)},
                            {"document", value.document}};
}

//===----------------------------------------------------------------------===//
// NotebookDocument
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocument &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri) &&
         o.map("notebookType", result.notebookType) &&
         o.map("version", result.version) && o.map("cells", result.cells);
}

llvm::json::Value llvm::lsp::toJSON(const NotebookDocument &value) {
  return llvm::json::Object{{"uri", value.uri},
                            {"notebookType", value.notebookType},
                            {"version", value.version},
                            {"cells", value.cells}};
}

//===----------------------------------------------------------------------===//
// DidOpenNotebookDocumentParams
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         DidOpenNotebookDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("notebookDocument", result.notebookDocument) &&
         o.map("cellTextDocuments", result.cellTextDocuments);
}

llvm::json::Value
llvm::lsp::toJSON(const DidOpenNotebookDocumentParams &value) {
  return llvm::json::Object{{"notebookDocument", value.notebookDocument},
                            {"cellTextDocuments", value.cellTextDocuments}};
}

//===----------------------------------------------------------------------===//
// NotebookDocumentChangeEvent
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookCellArrayChange &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  o.map("cells", result.cells);
  return o && o.map("start", result.start) &&
         o.map("deleteCount", result.deleteCount);
}

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocumentChangeEvent::CellsStructure &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);

  o.map("didOpen", result.didOpen);
  o.map("didClose", result.didClose);
  return o && o.map("array", result.array);
}

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocumentChangeEvent::CellsTextContent &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("document", result.document) &&
         o.map("changes", result.changes);
}

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocumentChangeEvent::Cells &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;

  o.map("structure", result.structure);
  o.map("data", result.data);
  o.map("textContent", result.textContent);
  return true;
}

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         NotebookDocumentChangeEvent &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("cells", result.cells);
}

llvm::json::Value llvm::lsp::toJSON(const NotebookCellArrayChange &value) {
  return llvm::json::Object{{"start", value.start},
                            {"deleteCount", value.deleteCount},
                            {"cells", value.cells}};
}

llvm::json::Value
llvm::lsp::toJSON(const NotebookDocumentChangeEvent::CellsStructure &value) {
  return llvm::json::Object{{"array", value.array},
                            {"didOpen", value.didOpen},
                            {"didClose", value.didClose}};
}

llvm::json::Value
llvm::lsp::toJSON(const NotebookDocumentChangeEvent::CellsTextContent &value) {
  return llvm::json::Object{{"document", value.document},
                            {"changes", value.changes}};
}

llvm::json::Value
llvm::lsp::toJSON(const NotebookDocumentChangeEvent::Cells &value) {
  return llvm::json::Object{{"structure", value.structure},
                            {"data", value.data},
                            {"textContent", value.textContent}};
}

llvm::json::Value llvm::lsp::toJSON(const NotebookDocumentChangeEvent &value) {
  return llvm::json::Object{{"cells", value.cells}};
}

llvm::json::Value
llvm::lsp::toJSON(const TextDocumentContentChangeEvent &value) {
  return llvm::json::Object{{"range", value.range},
                            {"rangeLength", value.rangeLength},
                            {"text", value.text}};
}

//===----------------------------------------------------------------------===//
// DidChangeNotebookDocumentParams
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         DidChangeNotebookDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("notebookDocument", result.notebookDocument) &&
         o.map("change", result.change);
}

llvm::json::Value
llvm::lsp::toJSON(const DidChangeNotebookDocumentParams &value) {
  return llvm::json::Object{{"notebookDocument", value.notebookDocument},
                            {"change", value.change}};
}

//===----------------------------------------------------------------------===//
// DidCloseNotebookDocumentParams
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         DidCloseNotebookDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("notebookDocument", result.notebookDocument) &&
         o.map("cellTextDocuments", result.cellTextDocuments);
}

//===----------------------------------------------------------------------===//
// Semantic Token
//===----------------------------------------------------------------------===//

/// The encoded size of a single semantic token.
static constexpr unsigned kSemanticTokenEncodingSize = 5;

/// Encode the given list of semantic tokens into a JSON array.
static llvm::json::Value encodeTokens(ArrayRef<SemanticToken> tokens) {
  llvm::json::Array result;
  result.reserve(kSemanticTokenEncodingSize * tokens.size());
  for (const SemanticToken &token : tokens) {
    result.push_back(token.deltaLine);
    result.push_back(token.deltaStart);
    result.push_back(token.length);
    result.push_back(token.tokenType);
    result.push_back(token.tokenModifiers);
  }
  assert(result.size() == (kSemanticTokenEncodingSize * tokens.size()));
  return std::move(result);
}

bool SemanticToken::operator==(const SemanticToken &rhs) const {
  return std::tie(deltaLine, deltaStart, length, tokenType, tokenModifiers) ==
         std::tie(rhs.deltaLine, rhs.deltaStart, rhs.length, rhs.tokenType,
                  rhs.tokenModifiers);
}

bool llvm::lsp::fromJSON(const llvm::json::Value &params,
                         SemanticTokens &result, llvm::json::Path path) {
  std::vector<int64_t> encodedTokens;

  llvm::json::ObjectMapper o(params, path);
  if (!o || !o.mapOptional("resultId", result.resultId) ||
      !o.map("data", encodedTokens))
    return false;

  for (size_t i = 0, e = encodedTokens.size(); i < e; i += 5) {
    result.tokens.push_back(lsp::SemanticToken{
        (unsigned)encodedTokens[i], (unsigned)encodedTokens[i + 1],
        (unsigned)encodedTokens[i + 2], (unsigned)encodedTokens[i + 3],
        (unsigned)encodedTokens[i + 4]});
  }
  return true;
}

llvm::json::Value llvm::lsp::toJSON(const SemanticTokens &value) {
  return llvm::json::Object{{"resultId", value.resultId},
                            {"data", encodeTokens(value.tokens)}};
}

//===----------------------------------------------------------------------===//
// SemanticTokensParams
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &params,
                         SemanticTokensParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(params, path);
  return o && o.map("textDocument", result.textDocument);
}

llvm::json::Value llvm::lsp::toJSON(const SemanticTokensParams &value) {
  return llvm::json::Object{{"textDocument", value.textDocument}};
}

bool llvm::lsp::fromJSON(const llvm::json::Value &params,
                         SemanticTokensDeltaParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(params, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("previousResultId", result.previousResultId);
}

//===----------------------------------------------------------------------===//
// SemanticTokensEdit
//===----------------------------------------------------------------------===//

llvm::json::Value llvm::lsp::toJSON(const SemanticTokensEdit &value) {
  return llvm::json::Object{
      {"start", kSemanticTokenEncodingSize * value.startToken},
      {"deleteCount", kSemanticTokenEncodingSize * value.deleteTokens},
      {"data", encodeTokens(value.tokens)}};
}

//===----------------------------------------------------------------------===//
// SemanticTokensOrDelta
//===----------------------------------------------------------------------===//

llvm::json::Value llvm::lsp::toJSON(const SemanticTokensOrDelta &value) {
  llvm::json::Object result{{"resultId", value.resultId}};
  if (value.edits)
    result["edits"] = *value.edits;
  if (value.tokens)
    result["data"] = encodeTokens(*value.tokens);
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// FoldingRangeParams
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &params,
                         FoldingRangeParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(params, path);
  return o && o.map("textDocument", result.textDocument);
}

//===----------------------------------------------------------------------===//
// FoldingRange
//===----------------------------------------------------------------------===//

const llvm::StringLiteral FoldingRange::kRegionKind = "region";
const llvm::StringLiteral FoldingRange::kCommentKind = "comment";
const llvm::StringLiteral FoldingRange::kImportKind = "import";

llvm::json::Value llvm::lsp::toJSON(const FoldingRange &value) {
  llvm::json::Object result{
      {"startLine", value.startLine},
      {"endLine", value.endLine},
  };
  if (value.startCharacter)
    result["startCharacter"] = value.startCharacter;
  if (value.endCharacter)
    result["endCharacter"] = value.endCharacter;
  if (!value.kind.empty())
    result["kind"] = value.kind;
  return result;
}

//===----------------------------------------------------------------------===//
// RenameParams
//===----------------------------------------------------------------------===//

llvm::json::Value lsp::toJSON(const RenameParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument},
                            {"position", params.position},
                            {"newName", params.newName}};
}

bool lsp::fromJSON(const llvm::json::Value &value, RenameParams &params,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", params.textDocument) &&
         o.map("position", params.position) && o.map("newName", params.newName);
}

//===----------------------------------------------------------------------===//
// Progress reporting
//===----------------------------------------------------------------------===//

llvm::json::Value llvm::lsp::toJSON(const WorkDoneProgressParams &params) {
  return llvm::json::Object{{"token", params.token}};
}

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         WorkDoneProgressParams &params,
                         llvm::json::Path path) {

  llvm::json::ObjectMapper o(value, path);
  return o && o.map("token", params.token);
}

llvm::json::Value llvm::lsp::toJSON(const WorkDoneProgressBeginParams &params) {
  return llvm::json::Object{
      {"kind", "begin"},
      {"title", params.title},
      {"message", params.message},
  };
}

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         WorkDoneProgressBeginParams &params,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("title", params.title) &&
         o.mapOptional("message", params.message);
}

llvm::json::Value llvm::lsp::toJSON(const WorkDoneProgressEndParams &params) {
  return llvm::json::Object{
      {"kind", "end"},
      {"message", params.message},
  };
}

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         WorkDoneProgressEndParams &params,
                         llvm::json::Path path) {

  llvm::json::ObjectMapper o(value, path);
  return o && o.mapOptional("message", params.message);
}

//===----------------------------------------------------------------------===//
// Serialization methods not available in the upstream MLIR code
//===----------------------------------------------------------------------===//

llvm::json::Value lsp::toJSON(const DidChangeTextDocumentParams &params) {
  return llvm::json::Object{
      {"textDocument", params.textDocument},
      {"contentChanges", params.contentChanges},
  };
}

llvm::json::Value lsp::toJSON(const TextDocumentItem &params) {
  return llvm::json::Object{{"uri", params.uri},
                            {"languageId", params.languageId},
                            {"text", params.text},
                            {"version", params.version}};
}

llvm::json::Value lsp::toJSON(const DidOpenTextDocumentParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument}};
}

llvm::json::Value lsp::toJSON(const TextDocumentPositionParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument},
                            {"position", params.position}};
}

llvm::json::Value lsp::toJSON(const ReferenceContext &context) {
  return llvm::json::Object{{"includeDeclaration", context.includeDeclaration}};
}

llvm::json::Value lsp::toJSON(const ReferenceParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument},
                            {"position", params.position},
                            {"context", params.context}};
}

llvm::json::Value lsp::toJSON(const CodeActionContext &context) {
  return llvm::json::Object{{"diagnostics", context.diagnostics},
                            {"only", context.only}};
}

llvm::json::Value lsp::toJSON(const CodeActionParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument},
                            {"range", params.range},
                            {"context", params.context}};
}

llvm::json::Value lsp::toJSON(const Hover2 &hover) {
  return toJSON(static_cast<Hover>(hover));
}

llvm::json::Value lsp::toJSON(const LSPError2 &error) {
  return llvm::json::Object{{"message", error.message},
                            {"code", (int)error.code}};
}

llvm::json::Value lsp::toJSON(const DocumentSymbolParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument}};
}

llvm::json::Value lsp::toJSON(const FoldingRangeParams &params) {
  return llvm::json::Object{{"textDocument", params.textDocument}};
}

bool lsp::fromJSON(const llvm::json::Value &value, Hover2 &range,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("contents", range.contents) &&
         o.mapOptional("range", range.range);
}

bool lsp::fromJSON(const llvm::json::Value &value, LSPError2 &error,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("message", error.message) && o.map("code", error.code);
}

bool lsp::fromJSON(const llvm::json::Value &value, ErrorCode &code,
                   llvm::json::Path path) {
  std::optional<int64_t> intVal = value.getAsInteger();
  if (!intVal)
    return false;
  code = static_cast<ErrorCode>(*intVal);
  return true;
}

bool lsp::fromJSON(const llvm::json::Value &value, MarkupKind &kind,
                   llvm::json::Path path) {
  std::optional<StringRef> str = value.getAsString();
  if (!str)
    return false;
  if (*str == "plaintext") {
    kind = MarkupKind::PlainText;
    return true;
  }
  if (*str == "markdown") {
    kind = MarkupKind::Markdown;
    return true;
  }
  return false;
}

bool lsp::fromJSON(const llvm::json::Value &value, MarkupContent &mc,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("kind", mc.kind) && o.map("value", mc.value);
}

bool lsp::fromJSON(const llvm::json::Value &value, CodeAction &codeAction,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("title", codeAction.title) &&
         o.map("kind", codeAction.kind) &&
         o.map("diagnostics", codeAction.diagnostics) &&
         o.map("isPreferred", codeAction.isPreferred) &&
         o.map("edit", codeAction.edit);
}

bool lsp::fromJSON(const llvm::json::Value &value,
                   CompletionList &completionList, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("isIncomplete", completionList.isIncomplete) &&
         o.map("items", completionList.items);
}

bool lsp::fromJSON(const llvm::json::Value &value,
                   CompletionItem &completionItem, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.mapOptional("label", completionItem.label) &&
         o.mapOptional("documentation", completionItem.documentation) &&
         o.mapOptional("kind", completionItem.kind) &&
         o.mapOptional("sortText", completionItem.sortText);
}

bool lsp::fromJSON(const llvm::json::Value &value,
                   DocumentSymbol &documentSymbol, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("name", documentSymbol.name) &&
         o.mapOptional("detail", documentSymbol.detail) &&
         o.map("kind", documentSymbol.kind) &&
         o.map("range", documentSymbol.range) &&
         o.map("selectionRange", documentSymbol.selectionRange) &&
         o.mapOptional("children", documentSymbol.children);
}

bool lsp::fromJSON(const llvm::json::Value &value, SymbolKind &kind,
                   llvm::json::Path path) {
  std::optional<int64_t> intVal = value.getAsInteger();
  if (!intVal)
    return false;
  kind = static_cast<SymbolKind>(*intVal);
  return true;
}

bool lsp::fromJSON(const llvm::json::Value &value, FoldingRange &foldingRange,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("startLine", foldingRange.startLine) &&
         o.map("endLine", foldingRange.endLine) &&
         o.mapOptional("startCharacter", foldingRange.startCharacter) &&
         o.mapOptional("endCharacter", foldingRange.endCharacter) &&
         o.mapOptional("kind", foldingRange.kind);
}

bool lsp::fromJSON(const llvm::json::Value &value, ParameterInformation &info,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  const llvm::json::Object &obj = *value.getAsObject();
  if (!o)
    return false;
  if (obj.getString("label")) {
    if (!o.map("label", info.labelString))
      return false;
  } else {
    std::vector<int64_t> pair;
    if (!o.map("label", pair) || pair.size() != 2)
      return false;
    info.labelOffsets = {pair[0], pair[1]};
  }
  return o.mapOptional("documentation", info.documentation);
}
