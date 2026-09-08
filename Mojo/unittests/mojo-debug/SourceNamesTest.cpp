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
using namespace lldb;

// TODO(MOTO-1079): Fix this test.
TEST(SourceNamesTest, DISABLED_testFunctionBeforeStructParsing) {
  // Tests that DWARF parsing is done correctly when LLDB parses a function
  // source name before its owning struct.
  // This happens when the debug session starts with a single breakpoint
  // within a struct method.

  StopContext ctx = buildAndLaunch("point.mojo");
  SBValue x = ctx.frame.FindVariable("x");
  EXPECT_EQ((int)x.GetValueAsSigned(), 1);

  ctx.stepOver();
  SBValue p1 = ctx.frame.FindVariable("p1");
  EXPECT_STREQ(p1.GetTypeName(), "!lit.struct<@point::@Point>");
}
