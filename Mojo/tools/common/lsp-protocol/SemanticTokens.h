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

#ifndef KGEN_TOOLS_COMMON_LSPPROTOCOL_SEMANTICTOKENS_H
#define KGEN_TOOLS_COMMON_LSPPROTOCOL_SEMANTICTOKENS_H

#include "Protocol.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

namespace M::Mojo::LSP {
//===----------------------------------------------------------------------===//
// SemanticToken Kind
//===----------------------------------------------------------------------===//

/// This enum represents all the different kinds of tokens that can be
/// highlighted.
enum class SemanticTokenKind {
  kVariable = 0,
  kSpecialVariable,
  kParameter,
  kFunction,
  kMethod,
  kField,
  kClass,
  kTrait,
  kType,
  kModule,

  kCount
};

/// Convert the given token kind into a string representing the LSP token type.
StringRef toLspSemanticTokenType(SemanticTokenKind kind);

//===----------------------------------------------------------------------===//
// SemanticToken Modifier
//===----------------------------------------------------------------------===//

/// This enum represents all the different modifiers that can be applied to
/// highlighted tokens.
enum class SemanticTokenModifier {
  kCount,
};

/// Convert the given token modifier into a string representing the LSP token
/// modifier.
StringRef toLspSemanticTokenModifier(SemanticTokenModifier modifier);

//===----------------------------------------------------------------------===//
// SemanticToken Token
//===----------------------------------------------------------------------===//

/// This class represents a highlighted token.
struct SemanticToken {
  SemanticToken() : kind(SemanticTokenKind::kCount) {}
  SemanticToken(SemanticTokenKind kind, llvm::lsp::Range range,
                uint32_t modifiers = 0)
      : kind(kind), modifiers(modifiers), range(range) {}

  bool operator==(const SemanticToken &rhs) const;
  bool operator<(const SemanticToken &rhs) const;

  /// Add a modifier to the token.
  SemanticToken &addModifier(SemanticTokenModifier modifier) {
    modifiers |= 1 << static_cast<unsigned>(modifier);
    return *this;
  }

  /// The kind of token this is.
  SemanticTokenKind kind;

  /// Modifiers that affect the token.
  uint32_t modifiers = 0;

  /// The range of the token.
  llvm::lsp::Range range;
};

/// Convert the given tokens into LSP semantic tokens. LSP semantic tokens need
/// to be constructed at the same time, because the position fields of an LSP
/// token are relative to the previous token.
std::vector<llvm::lsp::SemanticToken>
toLspSemanticTokens(ArrayRef<SemanticToken> tokens);

/// Convert the given LSP semantic tokens into the Mojo equivalent. We process
/// all at once because the position fields of an LSP token are relative to the
/// previous token.
std::vector<SemanticToken>
fromLspSemanticTokens(ArrayRef<llvm::lsp::SemanticToken> tokens);

/// Compute the difference between the two sets of tokens.
std::vector<llvm::lsp::SemanticTokensEdit>
diffTokens(ArrayRef<llvm::lsp::SemanticToken> before,
           ArrayRef<llvm::lsp::SemanticToken> after);

} // namespace M::Mojo::LSP

#endif
