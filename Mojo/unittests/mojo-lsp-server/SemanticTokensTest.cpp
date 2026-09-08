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

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::Mojo::LSP;

TEST(SemanticTokensTest, testSemanticTokens) {
  Document doc("test:///foo.mojo", R"(
import std.builtin
comptime builtin_alias = std.builtin

struct Struct:
  var field: Int

comptime struct_alias = Struct

# `raises` is load-bearing; see MOTO-903.
def foo() raises:
  return

comptime int_alias = 10

trait ATrait:
  def foo(var self, i: Self):
     ...

struct StructWithTrait(ATrait):
    def foo(var self, i: Self):
        pass
)");

  createTestClient()
      .open(doc)
      .semanticTokensFull(
          doc,
          [&](ArrayRef<SemanticToken> tokens) {
            EXPECT_NE((int)tokens.size(), 0);
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("builtin") &&
                     token.kind == SemanticTokenKind::kModule;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("builtin_alias") &&
                     token.kind == SemanticTokenKind::kModule;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("Struct") &&
                     token.kind == SemanticTokenKind::kClass;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("struct_alias") &&
                     token.kind == SemanticTokenKind::kType;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("field") &&
                     token.kind == SemanticTokenKind::kField;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("foo") &&
                     token.kind == SemanticTokenKind::kFunction;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("int_alias") &&
                     token.kind == SemanticTokenKind::kVariable;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("ATrait") &&
                     token.kind == SemanticTokenKind::kTrait;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("Self") &&
                     token.kind == SemanticTokenKind::kTrait;
            }));
            // Check that we didn't add a token for the synthetic methods of the
            // StructWithTrait struct.
            EXPECT_FALSE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range ==
                         *doc.findLastPos("struct StructWithTrait") &&
                     token.kind == SemanticTokenKind::kFunction;
            }));

            EXPECT_TRUE(llvm::all_of(tokens, [&](const SemanticToken &token) {
              return token.range.start.line == token.range.end.line &&
                     token.range.start.character <= token.range.end.character;
            }));
          })
      .execute();
}
