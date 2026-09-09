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

#include "SemanticTokens.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mojo-lsp-server"

using namespace M;
using namespace M::Mojo::LSP;

//===----------------------------------------------------------------------===//
// SemanticToken Kind
//===----------------------------------------------------------------------===//

StringRef Mojo::LSP::toLspSemanticTokenType(SemanticTokenKind kind) {
  switch (kind) {
  case SemanticTokenKind::kVariable:
    return "variable";
  case SemanticTokenKind::kSpecialVariable:
    // NOTE: This is a non-standard token type.
    return "specialVariable";
  case SemanticTokenKind::kParameter:
    return "parameter";
  case SemanticTokenKind::kFunction:
    return "function";
  case SemanticTokenKind::kMethod:
    return "method";
  case SemanticTokenKind::kField:
    return "property";
  case SemanticTokenKind::kClass:
    return "class";
  case SemanticTokenKind::kType:
    return "type";
  case SemanticTokenKind::kModule:
    return "namespace";
  case SemanticTokenKind::kTrait:
    return "interface";
  default:
    llvm_unreachable("unhandled SemanticTokenKind");
  }
}

//===----------------------------------------------------------------------===//
// SemanticToken Modifier
//===----------------------------------------------------------------------===//

StringRef
Mojo::LSP::toLspSemanticTokenModifier(SemanticTokenModifier modifier) {
  switch (modifier) {
  default:
    llvm_unreachable("unhandled SemanticTokenModifier");
  }
}

//===----------------------------------------------------------------------===//
// SemanticToken Token
//===----------------------------------------------------------------------===//

bool SemanticToken::operator==(const SemanticToken &rhs) const {
  return std::tie(range, kind, modifiers) ==
         std::tie(rhs.range, rhs.kind, rhs.modifiers);
}
bool SemanticToken::operator<(const SemanticToken &rhs) const {
  return std::tie(range, kind, modifiers) <
         std::tie(rhs.range, rhs.kind, rhs.modifiers);
}

std::vector<llvm::lsp::SemanticToken>
Mojo::LSP::toLspSemanticTokens(ArrayRef<SemanticToken> tokens) {
  assert(llvm::is_sorted(tokens) && "expected tokens to be sorted");
  std::vector<llvm::lsp::SemanticToken> result;

  for (auto [index, tok] : llvm::enumerate(tokens)) {
    if (tok.range.start.line != tok.range.end.line)
      llvm::report_fatal_error("expected token to be one line");

    if (tok.range.end.character < tok.range.start.character)
      llvm::report_fatal_error("expected token to have a positive length");

    llvm::lsp::SemanticToken &newTok = result.emplace_back();

    // `deltaStart`/`deltaLine` should be computed relative to the last token if
    // possible.
    if (index) {
      const SemanticToken &lastTok = tokens[index - 1];

      newTok.deltaLine = tok.range.start.line - lastTok.range.end.line;
      if (newTok.deltaLine) {
        newTok.deltaStart = tok.range.start.character;
      } else {
        newTok.deltaStart =
            tok.range.start.character - lastTok.range.start.character;
      }
    } else {
      newTok.deltaLine = tok.range.start.line;
      newTok.deltaStart = tok.range.start.character;
    }

    newTok.tokenType = static_cast<unsigned>(tok.kind);
    newTok.tokenModifiers = tok.modifiers;
    newTok.length = tok.range.end.character - tok.range.start.character;
  }
  return result;
}

std::vector<SemanticToken>
Mojo::LSP::fromLspSemanticTokens(ArrayRef<llvm::lsp::SemanticToken> tokens) {
  std::vector<SemanticToken> result;

  SemanticToken *lastToken = nullptr;
  for (const llvm::lsp::SemanticToken &token : tokens) {
    auto kind = static_cast<SemanticTokenKind>(token.tokenType);

    // Compute the range for the token (relative to the last token if possible).
    int line = token.deltaLine;
    int col = token.deltaStart;
    if (lastToken) {
      line += lastToken->range.end.line;

      // If the line number is 0, we are in the same line as the last token. In
      // that case, we need to add the column offset of the last token.
      if (token.deltaLine == 0)
        col += lastToken->range.start.character;
    }
    llvm::lsp::Range range({line, col},
                           {line, col + static_cast<int>(token.length)});

    lastToken = &result.emplace_back(kind, range, token.tokenModifiers);
  }
  return result;
}

std::vector<llvm::lsp::SemanticTokensEdit>
Mojo::LSP::diffTokens(ArrayRef<llvm::lsp::SemanticToken> before,
                      ArrayRef<llvm::lsp::SemanticToken> after) {
  // For now, just replace everything from the first-last modification.
  // FIXME: We should ideally use a real diff instead. The current behavior
  // isn't great for auto-insertion of imports.
  unsigned offset = 0;
  while (!before.empty() && !after.empty() && before.front() == after.front()) {
    ++offset;
    before = before.drop_front();
    after = after.drop_front();
  }
  while (!before.empty() && !after.empty() && before.back() == after.back()) {
    before = before.drop_back();
    after = after.drop_back();
  }
  if (before.empty() && after.empty())
    return {};

  llvm::lsp::SemanticTokensEdit edit;
  edit.startToken = offset;
  edit.deleteTokens = before.size();
  edit.tokens = after;
  return {std::move(edit)};
}
