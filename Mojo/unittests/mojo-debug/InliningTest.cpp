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

llvm::StringRef getFunctionName(StopContext &ctx) {
  return ctx.thread.GetSelectedFrame().GetFunctionName();
}

TEST(InliningTest, testBreakingOnInlinedCallsite) {
  // Tests that setting breakpoints on callsites that were inlined works.

  StopContext ctx = buildAndLaunch("inlined_callsite.mojo");
  std::vector<int> expectedBreakingLines =
      ctx.binary.getSource().findLinesWithText("# breakpoint");

  EXPECT_TRUE(getFunctionName(ctx).contains("main"));
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedBreakingLines[1]);

  ctx.resume();
  EXPECT_TRUE(getFunctionName(ctx).contains("callee_regular"));
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedBreakingLines[0]);

  ctx.resume();
  EXPECT_TRUE(getFunctionName(ctx).contains("main"));
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedBreakingLines[2]);

  ctx.resume();
  EXPECT_TRUE(getFunctionName(ctx).contains("callee_regular"));
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedBreakingLines[0]);
}

TEST(InliningTest, testInlinedVariableCalledFromNoDebug) {
  // Tests that debug functions inlined into no-debug functions are still
  // debuggable after being inlined again into a regular function.
  StopContext ctx = buildAndLaunch("inlined_variable.mojo");

  SBValue number = ctx.frame.FindVariable("nested_var").GetChildAtIndex(0);
  EXPECT_EQ((int)number.GetValueAsSigned(), 2);
}

TEST(InliningTest, testLiftedInlinedInoutArgModification) {
  // Tests that modifications to inlined mut args that are lifted by mem2reg
  // show up.
  StopContext ctx = buildAndLaunch("inlined_argument.mojo");

  SBValue number = ctx.frame.FindVariable("m").GetChildAtIndex(0);
  // Check the value is the updated value from the inlined callee.
  EXPECT_EQ((int)number.GetValueAsSigned(), 42);
}

TEST(InliningTest, testLiftedInlinedInoutArgPartialModification) {
  // Tests that modifications to inlined mut args that are lifted by mem2reg
  // show up when the argument is not the full variable from the caller side.
  StopContext ctx = buildAndLaunch("inlined_partial_argument.mojo");

  SBValue pair = ctx.frame.FindVariable("p");
  // At -O0, trivial types may be spilled to stack for debugger visibility
  // (extended debug lifetimes), so we don't assert register placement here.
  // TODO: Add an -O1 variant to verify mem2reg still promotes register-passable
  // types.  The test harness (MojoBinary / buildAndLaunch) hardcodes -O0, so
  // this requires plumbing an opt-level parameter through the test infra first.
  // Check the value is the updated value from the inlined callee.
  ASSERT_EQ((int)pair.GetNumChildren(), 2);
  SBValue firstField = pair.GetChildAtIndex(0).GetChildAtIndex(0);
  EXPECT_EQ((int)firstField.GetValueAsSigned(), 42);
  SBValue secondField = pair.GetChildAtIndex(1).GetChildAtIndex(0);
  EXPECT_EQ((int)secondField.GetValueAsSigned(), 4);
}
