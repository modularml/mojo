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

#ifndef KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_DOCUMENT_H
#define KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_DOCUMENT_H

#include "../common/lsp-protocol/Protocol.h"
#include "Support/LLVMForwardDecls.h"

namespace M {
/// Class representing an in-memory document.
class Document {
public:
  Document(StringRef uri, StringRef text);

  llvm::lsp::URIForFile getURI() const { return uri; }

  StringRef getContents() const { return contents; }

  /// Get the full range of the entire text.
  llvm::lsp::Range getFullRange() const;

  /// Get the position of the first occurrence of the given substring in the
  /// document within a single line.
  /// Skip lines with `# skip` at the end.
  std::optional<llvm::lsp::Position> findFirstPos(StringRef substr) const;

  /// Get the position of the last occurrence of the given substring in the
  /// document within a single line.
  /// Skip lines with `# skip` at the end.
  std::optional<llvm::lsp::Position> findLastPos(StringRef substr) const;

  /// Get the range of the first occurrence of the given substring in the
  /// document within a single line.
  /// Skip lines with `# skip` at the end.
  std::optional<llvm::lsp::Range> findFirstRange(StringRef substr) const;

  /// Get the range of the last occurrence of the given substring in the
  /// document within a single line.
  /// Skip lines with `# skip` at the end.
  std::optional<llvm::lsp::Range> findLastRange(StringRef substr) const;

  /// Get the ranges of all the occurrences of a given substring in the
  /// document. One occurrence per line. Skip lines with `# skip` at the end.
  std::vector<llvm::lsp::Range> findAllRanges(StringRef substr) const;

private:
  llvm::lsp::URIForFile uri;
  std::string contents;
  SmallVector<StringRef> lines;
};

class NotebookDocument {
public:
  NotebookDocument(StringRef uri, ArrayRef<StringRef> cellContents);

  llvm::lsp::URIForFile getURI() const { return uri; }

  ArrayRef<Document> getCells() const { return cells; }

private:
  llvm::lsp::URIForFile uri;
  SmallVector<Document> cells;
};

} // namespace M

#endif // KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_DOCUMENT_H
