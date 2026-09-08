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
TEST(InitFileTest, testInitModuleIsNotIndexed) {
  Document doc = createDocumentFromInputFileWithinPackage("__init__.mojo");

  createTestClient()
      .open(doc)
      .hoverNullable(doc, {0, 0},
                     [&](const std::optional<lsp::Hover> &hover) {
                       if (hover.has_value()) {
                         EXPECT_NE(hover->contents.value,
                                   "### module `__init__`\n");
                       }
                     })
      .execute();
}
