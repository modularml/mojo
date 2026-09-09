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

TEST(FoldingRangeTest, testDocStringFoldingRange) {
  Document doc("test:///foo.mojo",
               R"(
def single_line():
  """This is a single line doc string."""
  return

def multi_line():
  """This is a multi-line doc string.

  It has multiple lines.

  """
)");

  createTestClient()
      .open(doc)
      .foldingRange(
          doc,
          [](const std::vector<lsp::FoldingRange> &ranges) {
            ASSERT_TRUE(!ranges.empty());

            EXPECT_TRUE(
                llvm::any_of(ranges, [](const lsp::FoldingRange &range) {
                  return range.startLine == 2 && range.startCharacter == 5 &&
                         range.endLine == 2 && range.endCharacter == 38 &&
                         range.kind == "comment";
                }));

            EXPECT_TRUE(
                llvm::any_of(ranges, [](const lsp::FoldingRange &range) {
                  return range.startLine == 6 && range.startCharacter == 5 &&
                         range.endLine == 10 && range.endCharacter == 2 &&
                         range.kind == "comment";
                }));
          })
      .execute();
}
