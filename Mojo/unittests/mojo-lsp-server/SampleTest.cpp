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

TEST(SampleTest, testHoverAndDefinition) {
  // You first need to define a main document, which can be in-memory. The
  // server will build its index upon this document.
  Document doc("test:///foo.mojo",
               R"(
def function():
  var foo: Int = 420
  var bar = 1 + `foo`
  print(bar)
)");

  // Then you need to create a test client, followed by a list of requests to
  // issue. These requests are responded in the order in which they are defined.
  //
  // The test client runs in a non-interactive batch mode, i.e. all the requests
  // are collected into a single input file for the server. Then, the server is
  // invoked with this input and the client waits for the server to terminate
  // and print its entire output. Then, the test client parses this output and
  // responds to each pending request callback one by one. This approach helps
  // with reproducibility.
  createTestClient()
      // Make sure to open the main doc you want to index.
      .open(doc)
      .hover(doc, lsp::Position(3, 17),
             [](const lsp::Hover &response) {
               EXPECT_EQ(response.range, lsp::Range({3, 16}, {3, 21}));
               EXPECT_EQ(response.contents.value, R"(```mojo
(variable) var foo: Int
```)");
             })
      .definition(doc, lsp::Position(3, 17),
                  [](const std::vector<lsp::Location> &response) {
                    EXPECT_EQ(response.size(), 1u);
                    EXPECT_EQ(response[0].range, lsp::Range({2, 6}, {2, 9}));
                  })
      // Execute is needed to invoke the server. If not invoked, a crash will
      // happen when terminating this test.
      .execute();
}
