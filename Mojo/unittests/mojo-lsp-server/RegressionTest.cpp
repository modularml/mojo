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

// Test that multiple rapid incremental edits are accumulated correctly.
// This is a regression test for a bug where the debouncer would apply each
// incremental edit to the original document content instead of the accumulated
// content from previous edits. For example, if the document starts with "ab"
// and we send two edits: insert "X" at position 1 (-> "aXb") then insert "Y"
// at position 2 (-> "aXYb"), the bug would cause the second edit to be applied
// to "ab" instead of "aXb", resulting in "abY" instead of "aXYb".
TEST(RegressionTest, IncrementalEditsAccumulateCorrectly) {
  // Start with a simple expression statement.
  Document doc("test:///foo.mojo", R"(def foo():
  1+1)");

  // We'll send two incremental edits:
  // 1. Insert "x" at line 1, col 2 (after the two spaces): "  1+1" -> "  x1+1"
  // 2. Insert " = " at line 1, col 3 (after "x"): "  x1+1" -> "  x = 1+1"
  //
  // If edits accumulate correctly, the final content will be:
  //   def foo():
  //     x = 1+1
  //
  // Which is valid Mojo (x is assigned the expression 1+1).
  //
  // If the bug exists (edits applied to stale content), edit 2 would be
  // applied to the original "  1+1" at position 3, giving "  1 = +1" which
  // is a syntax error (cannot assign to a literal).

  createTestClient()
      .open(doc)
      // Edit 1: Insert "x" at line 1, col 2
      .update(doc, lsp::Range{{1, 2}, {1, 2}}, "x")
      // Edit 2: Insert " = " at line 1, col 3 (right after the "x" we just
      // inserted)
      .update(doc, lsp::Range{{1, 3}, {1, 3}}, " = ")
      .onDiagnostics(
          doc,
          [](auto diags) {
            // With correct accumulation, we should NOT have a syntax
            // error. We might have a warning about unused variable,
            // but the key is no syntax/parse errors indicating
            // corrupted content.
            for (const auto &diag : diags) {
              // Check that we don't have errors about invalid
              // assignment targets, which would indicate the second
              // edit was applied to the original content instead of
              // the accumulated content.
              bool hasAssignmentError =
                  diag.message.find("cannot assign") != std::string::npos ||
                  diag.message.find("invalid assignment") !=
                      std::string::npos ||
                  diag.message.find("assign to literal") != std::string::npos;
              EXPECT_FALSE(hasAssignmentError)
                  << "Unexpected assignment error: " << diag.message
                  << "\nThis suggests incremental edits were not "
                     "accumulated correctly.";
            }
          })
      .execute();
}

TEST(RegressionTest, moto1041) {
  // The original issue was caused by the parser emitting a hidden symbol that,
  // when combined with the language server's range calculations, created a
  // range that slightly exceeded the bounds of the document. This caused us to
  // crash because the range was not contained within the document, which meant
  // SourceMgr lookups failed.

  // The whitespace in this source snippet is deliberate and required to
  // reproduce the original crash.
  Document doc("test:///foo.mojo",
               // clang-format off
               R"(
def main() raises:
  pass)");
  // clang-format on

  // Simply not crashing is sufficient.
  createTestClient()
      .open(doc)
      .semanticTokensFull(doc, [](ArrayRef<Mojo::LSP::SemanticToken>) {})
      .execute();
}

TEST(RegressionTest, moto983) {
  // The original issue was caused by a misinterpretation of the LSP's encoding
  // of column offsets, where we interpreted them as UTF-8 offsets instead of
  // UTF-16 code unit offsets as required by the specification. This caused the
  // server's internal view to diverge from reality as it incorrectly sliced
  // multi-byte code points.

  Document doc("test:///foo.mojo", R"(
def main():
    var str = "Hello 🔥"

    print(str)

    )");

  createTestClient()
      .open(doc)
      .update(doc, llvm::lsp::Range{{0, 24}, {1, 0}}, "")
      .onDiagnostics(doc, [](auto diags) { EXPECT_TRUE(diags.empty()); })
      .execute();
}
