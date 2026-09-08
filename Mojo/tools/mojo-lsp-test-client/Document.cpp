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

#include "Document.h"

using namespace M;
namespace lsp = llvm::lsp;

Document::Document(StringRef uri, StringRef text) : contents(text) {
  if (llvm::Expected<lsp::URIForFile> uriOr = lsp::URIForFile::fromURI(uri))
    this->uri = std::move(*uriOr);
  else
    llvm::report_fatal_error(uriOr.takeError());

  StringRef(contents).split(lines, '\n');
}

lsp::Range Document::getFullRange() const {
  return {lsp::Position{0, 0}, lsp::Position{(int)lines.size(), 0}};
}

std::optional<lsp::Position> Document::findFirstPos(StringRef substr) const {
  if (std::optional<lsp::Range> range = findFirstRange(substr))
    return range->start;

  return {};
}

std::optional<lsp::Position> Document::findLastPos(StringRef substr) const {
  if (std::optional<lsp::Range> range = findLastRange(substr))
    return range->start;

  return {};
}

std::optional<llvm::lsp::Range>
Document::findFirstRange(StringRef substr) const {
  for (size_t line = 0, e = lines.size(); line < e; ++line) {
    if (lines[line].ends_with("# skip"))
      continue;
    if (size_t pos = lines[line].find(substr); pos != StringRef::npos)
      return lsp::Range{lsp::Position(line, pos),
                        lsp::Position(line, pos + substr.size())};
  }

  return {};
}

std::vector<llvm::lsp::Range> Document::findAllRanges(StringRef substr) const {
  std::vector<llvm::lsp::Range> ranges;
  for (size_t line = 0, e = lines.size(); line < e; ++line) {
    if (lines[line].ends_with("# skip"))
      continue;
    if (size_t pos = lines[line].find(substr); pos != StringRef::npos)
      ranges.emplace_back(lsp::Position(line, pos),
                          lsp::Position(line, pos + substr.size()));
  }

  return ranges;
}

std::optional<llvm::lsp::Range>
Document::findLastRange(StringRef substr) const {
  if (lines.empty())
    return {};
  for (size_t line = lines.size() - 1; line; --line) {
    if (lines[line].ends_with("# skip"))
      continue;
    if (size_t pos = lines[line].rfind(substr); pos != StringRef::npos)
      return lsp::Range{lsp::Position(line, pos),
                        lsp::Position(line, pos + substr.size())};
  }

  return {};
}

NotebookDocument::NotebookDocument(StringRef uri,
                                   ArrayRef<StringRef> cellContents) {
  if (llvm::Expected<lsp::URIForFile> uriOr = lsp::URIForFile::fromURI(uri))
    this->uri = std::move(*uriOr);
  else
    llvm::report_fatal_error(uriOr.takeError());

  for (auto [idx, cellContent] : llvm::enumerate(cellContents))
    cells.emplace_back((uri + std::to_string(idx)).str(), cellContent);
}
