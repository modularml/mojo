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

TEST(CodeActionTest, testGenerateDocumentation) {
  Document doc("test:///foo.mojo",
               R"(""""""

def function[value: Int](self: Int) -> Int:
    """"""
    return 10

struct EmptyStruct:
    """"""
    ...

struct ParameterStruct[value: Int]:
    """"""
    ...
)");

  // Build an expected diagnostic for a missing doc string given a start line
  // and position.
  auto buildExpectedDiag = [](int line, int startPos) -> lsp::Diagnostic {
    return {lsp::Range{{line, startPos}, {line, startPos + 6}},
            lsp::DiagnosticSeverity::Warning,
            "mojo",
            "Unexpected empty documentation string",
            /*relatedInformation=*/std::nullopt,
            /*tags*/ {},
            /*category=*/std::nullopt};
  };

  auto checkTemplate = [](const lsp::CodeAction &action, StringRef expected) {
    EXPECT_EQ(action.title, "Generate documentation");
    EXPECT_TRUE(action.edit);
    EXPECT_EQ((int)action.edit->changes.size(), 1);
    EXPECT_EQ(action.edit->changes.begin()->second[0].newText, expected);
  };

  createTestClient()
      .open(doc)
      .codeAction(doc, doc.getFullRange(),
                  {buildExpectedDiag(0, 0), buildExpectedDiag(3, 4),
                   buildExpectedDiag(7, 4), buildExpectedDiag(11, 4)},
                  [&](const std::vector<lsp::CodeAction> &actions) {
                    EXPECT_EQ((int)actions.size(), 4);

                    // Module template
                    checkTemplate(actions[0], "[summary].");

                    // Function template
                    checkTemplate(actions[1],
                                  R"([summary].

    Parameters:
        value: [description].

    Args:
        self: [description].

    Returns:
        [description].
    )");

                    // Empty struct template.
                    checkTemplate(actions[2], "[summary].");

                    // Parameterized struct template.
                    checkTemplate(actions[3],
                                  R"([summary].

    Parameters:
        value: [description].
    )");
                  })
      .execute();
}
