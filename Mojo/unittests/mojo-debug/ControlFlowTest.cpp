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

TEST(ControlFlowTest, testSteppingIntoInlinedNoDebugInfo) {
  // Checks line info for inlined function with no debuginfo.
  StopContext ctx = buildAndLaunch("step_into_inlined_no_debug_info.mojo");
  ctx.stepInto();

  int expectedLine = ctx.binary.getSource().findFirstLineWithText(
      "# expected after step-into");
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedLine);
}

TEST(ControlFlowTest, testStepStraightLine) {
  // Checks stepping straight line code.
  StopContext ctx = buildAndLaunch("step_straight_line.mojo");
  int functionHeaderLine =
      ctx.binary.getSource().findFirstLineWithText("def main()");

  int line = ctx.frame.GetLineEntry().GetLine();
  int prevLine = line;
  while (line != functionHeaderLine) {
    ASSERT_GE(line, prevLine);

    ctx.stepOver();
    prevLine = line;
    line = ctx.frame.GetLineEntry().GetLine();
  }
}

static void assertIndex(StopContext &ctx, StringRef name, int64_t expected) {
  SBValue var = ctx.frame.FindVariable(name.data());
  EXPECT_STREQ(var.GetChildAtIndex(0).GetValue(),
               std::to_string(expected).c_str());
  EXPECT_STREQ(var.GetTypeName(), "!kgen.scalar<index>");
  EXPECT_STREQ(var.GetDisplayTypeName(), "__mlir_type.`!kgen.scalar<index>`");
  EXPECT_TRUE(var.GetType().GetTypeFlags() | lldb::eTypeIsInteger);
  EXPECT_EQ(var.GetChildAtIndex(0).GetValueAsSigned(expected - 1), expected);
}

TEST(ControlFlowTest, testAssignment) {
  // Make sure basic var mutation assignment is tracked.
  StopContext ctx = buildAndLaunch("var_mutation_assignment.mojo");

  assertIndex(ctx, "i", 5);
  assertIndex(ctx, "j", 7);

  ctx.resume();
  assertIndex(ctx, "i", 15);
  assertIndex(ctx, "j", 7);

  ctx.resume();
  assertIndex(ctx, "i", 15);
  assertIndex(ctx, "j", 13);

  ctx.resume();
  assertIndex(ctx, "i", 2);
  assertIndex(ctx, "j", 13);
}

TEST(ControlFlowTest, testIteration) {
  // Make sure changes to basic loop index variable is tracked.
  StopContext ctx = buildAndLaunch("var_mutation_iteration.mojo");

  assertIndex(ctx, "i", 0);
  ctx.resume();
  assertIndex(ctx, "i", 1);
  ctx.resume();
  assertIndex(ctx, "i", 2);
}
