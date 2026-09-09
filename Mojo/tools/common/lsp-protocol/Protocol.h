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
//
// This file contains extensions to llvm/Support/LSP/Protocol.h
// file.
//
// TODO: upstream all the changes in this file, as they are generic.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_COMMON_LSP_PROTOCOL_PROTOCOL_H
#define KGEN_TOOLS_COMMON_LSP_PROTOCOL_PROTOCOL_H

#include "llvm/Support/LSP/Protocol.h"

namespace llvm::lsp {
using NotebookDocumentIdentifier = TextDocumentIdentifier;
using VersionedNotebookDocumentIdentifier = VersionedTextDocumentIdentifier;

//===----------------------------------------------------------------------===//
// SignatureInformation
//===----------------------------------------------------------------------===//

/// Represents the signature of something callable.
struct SignatureInformation2 {
  /// The label of this signature. Mandatory.
  std::string label;

  /// The documentation of this signature. Optional.
  std::optional<MarkupContent> documentation;

  /// The parameters of this signature.
  std::vector<ParameterInformation> parameters;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, SignatureInformation2 &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const SignatureInformation2 &value);

//===----------------------------------------------------------------------===//
// SignatureHelp
//===----------------------------------------------------------------------===//

/// Represents the signature of a callable.
struct SignatureHelp2 {
  /// The resulting signatures.
  std::vector<SignatureInformation2> signatures;

  /// The active signature.
  int activeSignature = 0;

  /// The active parameter of the active signature.
  int activeParameter = 0;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, SignatureHelp2 &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const SignatureHelp2 &value);

//===----------------------------------------------------------------------===//
// NotebookCell
//===----------------------------------------------------------------------===//

enum class NotebookCellKind {
  /// A markup-cell is formatted source that is used for display.
  Markup = 1,

  /// A code-cell is source code.
  Code = 2,
};

/// A notebook cell.
///
/// A cell's document URI must be unique across ALL notebook cells and can
/// therefore be used to uniquely identify a notebook cell or the cell's text
/// document.
struct NotebookCell {
  /// The cell's kind.
  NotebookCellKind kind;

  /// The URI of the cell's text document content.
  URIForFile document;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, NotebookCell &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const NotebookCell &value);

//===----------------------------------------------------------------------===//
// NotebookDocument
//===----------------------------------------------------------------------===//

struct NotebookDocument {
  /// The notebook document's URI.
  URIForFile uri;

  /// The type of the notebook.
  std::string notebookType;

  /// The version number of this document (it will increase after each change,
  /// including undo/redo).
  int version;

  /// The cells of a notebook.
  std::vector<NotebookCell> cells;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, NotebookDocument &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const NotebookDocument &value);

//===----------------------------------------------------------------------===//
// DidOpenNotebookDocumentParams
//===----------------------------------------------------------------------===//

struct DidOpenNotebookDocumentParams {
  /// The notebook document that got opened.
  NotebookDocument notebookDocument;

  /// The text documents that represent the content of a notebook cell.
  std::vector<TextDocumentItem> cellTextDocuments;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              DidOpenNotebookDocumentParams &result, llvm::json::Path path);
llvm::json::Value toJSON(const DidOpenNotebookDocumentParams &value);

//===----------------------------------------------------------------------===//
// NotebookDocumentChangeEvent
//===----------------------------------------------------------------------===//

/// A change describing how to move a `NotebookCell` array from state S to S'.
struct NotebookCellArrayChange {
  /// The start offset of the cell that changed.
  uint64_t start;

  /// The deleted cells.
  uint64_t deleteCount;

  /// The new cells, if any.
  std::vector<NotebookCell> cells;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, NotebookCellArrayChange &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const NotebookCellArrayChange &value);

/// A change event for a notebook document.
struct NotebookDocumentChangeEvent {
  /// Changes to the cell structure to add or remove cells.
  struct CellsStructure {
    /// The change to the cell array.
    NotebookCellArrayChange array;

    /// Additional opened cell text documents.
    std::vector<TextDocumentItem> didOpen;

    /// Additional closed cell text documents.
    std::vector<TextDocumentIdentifier> didClose;
  };

  /// Changes to the text content of notebook cells.
  struct CellsTextContent {
    VersionedTextDocumentIdentifier document;
    std::vector<TextDocumentContentChangeEvent> changes;
  };

  /// Changes to cells.
  struct Cells {
    /// Changes to the cell structure to add or remove cells.
    std::optional<CellsStructure> structure;

    /// Changes to notebook cells properties like its kind, execution summary or
    /// metadata.
    std::vector<NotebookCell> data;

    /// Changes to the text content of notebook cells.
    std::vector<CellsTextContent> textContent;
  };

  /// Changes to cells.
  std::optional<Cells> cells;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              NotebookDocumentChangeEvent::CellsStructure &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value,
              NotebookDocumentChangeEvent::CellsTextContent &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value,
              NotebookDocumentChangeEvent::Cells &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value,
              NotebookDocumentChangeEvent &result, llvm::json::Path path);

llvm::json::Value
toJSON(const NotebookDocumentChangeEvent::CellsStructure &value);
llvm::json::Value
toJSON(const NotebookDocumentChangeEvent::CellsTextContent &value);
llvm::json::Value toJSON(const NotebookDocumentChangeEvent::Cells &value);
llvm::json::Value toJSON(const NotebookDocumentChangeEvent &value);
llvm::json::Value toJSON(const TextDocumentContentChangeEvent &value);

//===----------------------------------------------------------------------===//
// DidChangeNotebookDocumentParams
//===----------------------------------------------------------------------===//

struct DidChangeNotebookDocumentParams {
  /// The notebook document that got opened.
  VersionedNotebookDocumentIdentifier notebookDocument;

  /// The actual changes to the notebook document.
  ///
  /// The change describes single state change to the notebook document.
  /// So it moves a notebook document, its cells and its cell text document
  /// contents from state S to S'.
  ///
  /// To mirror the content of a notebook using change events use the
  /// following approach:
  /// - start with the same initial content
  /// - apply the 'notebookDocument/didChange' notifications in the order
  ///   you receive them.
  NotebookDocumentChangeEvent change;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              DidChangeNotebookDocumentParams &result, llvm::json::Path path);
llvm::json::Value toJSON(const DidChangeNotebookDocumentParams &value);

//===----------------------------------------------------------------------===//
// DidCloseNotebookDocumentParams
//===----------------------------------------------------------------------===//

/// The params sent in a close notebook document notification.
struct DidCloseNotebookDocumentParams {
  /// The notebook document that got closed.
  NotebookDocumentIdentifier notebookDocument;

  /// The text documents that represent the content of a notebook cell that got
  /// closed.
  std::vector<TextDocumentIdentifier> cellTextDocuments;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              DidCloseNotebookDocumentParams &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// Semantic Token
//===----------------------------------------------------------------------===//

/// Specifies a single semantic token in the document.
/// This struct is not part of LSP, which just encodes lists of tokens as
/// arrays of numbers directly.
struct SemanticToken {
  bool operator==(const SemanticToken &rhs) const;

  /// The token line number, relative to the previous token.
  unsigned deltaLine = 0;
  /// The token start character, relative to the previous token.
  /// (relative to 0 or the previous token's start if they are on the same line)
  unsigned deltaStart = 0;
  /// The length of the token. Note, a token cannot be multiline.
  unsigned length = 0;
  /// The type of the token, which will be looked up in
  /// `SemanticTokensLegend.tokenTypes`.
  unsigned tokenType = 0;
  /// The modifiers of the token. Each set bit will be looked up in
  /// `SemanticTokensLegend.tokenModifiers`
  unsigned tokenModifiers = 0;
};

/// A versioned set of tokens.
struct SemanticTokens {
  SemanticTokens() = default;
  SemanticTokens(std::vector<SemanticToken> tokens)
      : tokens(std::move(tokens)) {}

  /// An optional result id. If provided and clients support delta updating
  /// the client will include the result id in the next semantic token request.
  /// A server can then instead of computing all semantic tokens again simply
  /// send a delta.
  std::string resultId;

  /// The actual tokens, encoded as a flat integer array.
  std::vector<SemanticToken> tokens;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &params, SemanticTokens &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const SemanticTokens &value);

//===----------------------------------------------------------------------===//
// SemanticTokensParams
//===----------------------------------------------------------------------===//

/// Body of textDocument/semanticTokens/full request.
struct SemanticTokensParams {
  /// The text document.
  TextDocumentIdentifier textDocument;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &params, SemanticTokensParams &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const SemanticTokensParams &value);

/// Body of textDocument/semanticTokens/full/delta request. Requests the changes
/// in semantic tokens since a previous response.
struct SemanticTokensDeltaParams {
  /// The text document.
  TextDocumentIdentifier textDocument;
  /// The previous result id.
  std::string previousResultId;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &params,
              SemanticTokensDeltaParams &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// SemanticTokensEdit
//===----------------------------------------------------------------------===//

/// Describes a replacement of a contiguous range of semanticTokens.
struct SemanticTokensEdit {
  /// LSP specifies `start` and `deleteCount` which are relative to the array
  /// encoding of the previous tokens.
  /// We use token counts instead, and translate when serializing this struct.
  unsigned startToken = 0;
  unsigned deleteTokens = 0;

  /// The tokens of the edit, encoded as a flat integer array.
  std::vector<SemanticToken> tokens;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const SemanticTokensEdit &value);

//===----------------------------------------------------------------------===//
// SemanticTokensOrDelta
//===----------------------------------------------------------------------===//

/// This models LSP SemanticTokensDelta | SemanticTokens, which is the result of
/// textDocument/semanticTokens/full/delta.
struct SemanticTokensOrDelta {
  std::string resultId;
  /// Set if we computed edits relative to a previous set of tokens.
  std::optional<std::vector<SemanticTokensEdit>> edits;
  /// Set if we computed a fresh set of tokens, encoded as an integer array.
  std::optional<std::vector<SemanticToken>> tokens;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const SemanticTokensOrDelta &value);

//===----------------------------------------------------------------------===//
// FoldingRangeParams
//===----------------------------------------------------------------------===//

struct FoldingRangeParams {
  TextDocumentIdentifier textDocument;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &params, FoldingRangeParams &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const FoldingRangeParams &params);

//===----------------------------------------------------------------------===//
// FoldingRange
//===----------------------------------------------------------------------===//

/// Stores information about a region of code that can be folded.
struct FoldingRange {
  FoldingRange(Range range = {}, StringRef kind = {})
      : startLine(range.start.line), startCharacter(range.start.character),
        endLine(range.end.line), endCharacter(range.end.character),
        kind(kind.str()) {}

  const static llvm::StringLiteral kRegionKind;
  const static llvm::StringLiteral kCommentKind;
  const static llvm::StringLiteral kImportKind;

  int64_t startLine = 0;
  int64_t startCharacter;
  int64_t endLine = 0;
  int64_t endCharacter;
  std::string kind;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const FoldingRange &value);
bool fromJSON(const llvm::json::Value &value, FoldingRange &foldingRange,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// Extension of Hover that has a default constructor to make fromJSON happy.
//===----------------------------------------------------------------------===//

struct Hover2 : Hover {
  using Hover::Hover;
  Hover2() : Hover(Range()) {}
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const Hover2 &hover);
bool fromJSON(const llvm::json::Value &value, Hover2 &range,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// Extension of LSPError that has a default constructor to make fromJSON happy.
//===----------------------------------------------------------------------===//

struct LSPError2 : LSPError {
  using LSPError::LSPError;
  LSPError2() : LSPError("", ErrorCode::UnknownErrorCode) {}
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const LSPError2 &error);
bool fromJSON(const llvm::json::Value &value, LSPError2 &error,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, ErrorCode &code,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// RenameParams
//===----------------------------------------------------------------------===//

/// Contains the parameters of a textDocument/rename LSP request.
struct RenameParams {
  /// The text document.
  TextDocumentIdentifier textDocument;

  /// The position inside the text document.
  Position position;

  /// The new name of the symbol.
  std::string newName;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const RenameParams &params);
bool fromJSON(const llvm::json::Value &value, RenameParams &params,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// Progress reporting
//===----------------------------------------------------------------------===//

template <typename T>
struct ProgressParams {
  std::string token;
  T value;
};

template <typename T>
llvm::json::Value toJSON(const ProgressParams<T> &params) {
  return llvm::json::Object{
      {"token", params.token},
      {"value", params.value},
  };
}

template <typename T>
bool fromJSON(const llvm::json::Value &value, ProgressParams<T> &params,
              llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("token", params.token) && o.map("value", params.value);
}

struct WorkDoneProgressParams {
  std::string token;
};
llvm::json::Value toJSON(const WorkDoneProgressParams &params);
bool fromJSON(const llvm::json::Value &value, WorkDoneProgressParams &params,
              llvm::json::Path path);

struct WorkDoneProgressBeginParams {
  /// The title of the progress operation, used to communicate what kind of
  /// operation is being performed.
  std::string title;

  /// An optional message containing more details.
  std::optional<std::string> message;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const WorkDoneProgressBeginParams &params);
bool fromJSON(const llvm::json::Value &value,
              WorkDoneProgressBeginParams &params, llvm::json::Path path);

struct WorkDoneProgressEndParams {
  /// An optional message describing the final result of the operation.
  std::optional<std::string> message;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const WorkDoneProgressEndParams &params);
bool fromJSON(const llvm::json::Value &value, WorkDoneProgressEndParams &params,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// Serialization methods not available in the upstream MLIR code
//===----------------------------------------------------------------------===//

llvm::json::Value toJSON(const DidChangeTextDocumentParams &params);

llvm::json::Value toJSON(const TextDocumentItem &params);

llvm::json::Value toJSON(const DidOpenTextDocumentParams &params);

llvm::json::Value toJSON(const TextDocumentPositionParams &params);

llvm::json::Value toJSON(const ReferenceContext &context);

llvm::json::Value toJSON(const ReferenceParams &params);

llvm::json::Value toJSON(const CodeActionContext &context);

llvm::json::Value toJSON(const CodeActionParams &params);

llvm::json::Value toJSON(const DocumentSymbolParams &params);

bool fromJSON(const llvm::json::Value &value, MarkupKind &kind,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, MarkupContent &mc,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, CodeAction &codeAction,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, CompletionList &completionList,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, CompletionItem &completionItem,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, DocumentSymbol &documentSymbol,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, SymbolKind &kind,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, ParameterInformation &info,
              llvm::json::Path path);
} // namespace llvm::lsp

#endif // KGEN_TOOLS_COMMON_LSP_PROTOCOL_PROTOCOL_H
