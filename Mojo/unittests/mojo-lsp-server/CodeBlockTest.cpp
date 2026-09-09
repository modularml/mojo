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

TEST(CodeBlockTest, testCodeBlockDiagnostics) {
  Document doc("test:///foo.mojo",
               R"(
def function():
  """Test doc string.

  ```mojo
  var foo = bar
  ```
  """
)");

  createTestClient()
      // Docstring code-block checking is disabled by default; enable it here
      // since these tests exercise diagnostics/hover/completion inside blocks.
      .setCheckDocstrings(true)
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message,
                                 "use of unknown declaration 'bar'");
                     })
      .execute();
}

TEST(CodeBlockTest, testCodeBlockHover) {
  Document doc("test:///foo.mojo",
               R"(
def function():
  """Test doc string.

  ```mojo
  def test():
    var foo: Int = 420
    var bar = 1 + `foo`
    print(bar)
  ```

  """
)");

  createTestClient()
      // Docstring code-block checking is disabled by default; enable it here
      // since these tests exercise diagnostics/hover/completion inside blocks.
      .setCheckDocstrings(true)
      .open(doc)
      .hover(doc, *doc.findFirstPos("foo"),
             [](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(variable) var foo: Int
```)");
               EXPECT_EQ(hover.range, lsp::Range({6, 8}, {6, 11}));
             })
      .execute();
}

TEST(CodeBlockTest, testCodeBlockCompletion) {
  Document doc("test:///foo.mojo",
               R"(
def function():
  """Test doc string.

  ```mojo
  var foo = 10
  ```

  ```mojo
  foo.completion
  ```

  """
)");

  createTestClient()
      // Docstring code-block checking is disabled by default; enable it here
      // since these tests exercise diagnostics/hover/completion inside blocks.
      .setCheckDocstrings(true)
      .open(doc)
      .completion(doc, *doc.findFirstPos("completion"),
                  [](const lsp::CompletionList &completion) {
                    EXPECT_TRUE(llvm::any_of(
                        completion.items, [](const lsp::CompletionItem &item) {
                          return item.label == "_mlir_value" &&
                                 item.kind == lsp::CompletionItemKind::Field;
                        }));
                  })
      .execute();
}

TEST(CodeBlockTest, testCodeBlockEndCompletion) {
  Document doc = createDocumentFromInputFileWithinPackage("doc_strings.mojo");

  createTestClient()
      // Docstring code-block checking is disabled by default; enable it here
      // since these tests exercise diagnostics/hover/completion inside blocks.
      .setCheckDocstrings(true)
      .open(doc)
      .completion(doc, doc.findFirstRange("test_completions.")->end,
                  [](const lsp::CompletionList &completion) {
                    EXPECT_TRUE(llvm::any_of(
                        completion.items, [](const lsp::CompletionItem &item) {
                          return item.label == "completion_test" &&
                                 item.kind == lsp::CompletionItemKind::Function;
                        }));
                  })
      .execute();
}
