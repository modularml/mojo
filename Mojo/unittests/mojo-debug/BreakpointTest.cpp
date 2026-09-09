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
using namespace ::testing::internal;

TEST(BreakpointTest, testBreakpoint) {
  // Test the breakpoint() intrinsic: stop and check a variable.

  StopContext ctx = buildAndLaunch("breakpoint.mojo");
  SBValue sum = ctx.frame.FindVariable("sum").GetChildAtIndex(0);
  EXPECT_EQ((int)sum.GetValueAsSigned(), 36);

  // Resuming after breakpoint() should continue normally to exit.
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}

TEST(BreakpointTest, testAdjacentBreakpoints) {
  // Test that three truly adjacent breakpoint() calls (no statements
  // between them) each produce a separate stop.

  StopContext ctx = buildAndLaunch("breakpoint_adjacent.mojo");

  // buildAndLaunch already stopped at the first breakpoint() trap.
  ctx.resume();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateStopped);
  ctx.resume();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateStopped);
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}

TEST(BreakpointTest, testSeparatedBreakpoints) {
  // Test that three breakpoint() calls with statements between them
  // each produce a separate stop at the correct line, and that
  // the interleaved statements execute between stops.

  StopContext ctx = buildAndLaunch("breakpoint_separated.mojo");

  // First stop: x was just initialised to 0.
  auto &src = ctx.binary.getSource();
  EXPECT_EQ(src.findFirstLineWithText("# stop_1"),
            (int)ctx.frame.GetLineEntry().GetLine());
  SBValue x = ctx.frame.FindVariable("x");
  EXPECT_EQ((int)x.GetValueAsSigned(), 0);

  // Second stop: x += 1 executed, so x == 1.
  ctx.resume();
  EXPECT_EQ(src.findFirstLineWithText("# stop_2"),
            (int)ctx.frame.GetLineEntry().GetLine());
  x = ctx.frame.FindVariable("x").GetChildAtIndex(0);
  EXPECT_EQ((int)x.GetValueAsSigned(), 1);

  // Third stop: x += 1 again, so x == 2.
  ctx.resume();
  EXPECT_EQ(src.findFirstLineWithText("# stop_3"),
            (int)ctx.frame.GetLineEntry().GetLine());
  x = ctx.frame.FindVariable("x").GetChildAtIndex(0);
  EXPECT_EQ((int)x.GetValueAsSigned(), 2);

  // Resume to exit.
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}

TEST(BreakpointTest, testBreakOnRaise) {
  // Test the break-on-raise feature.

  StopContext ctx = buildAndLaunch("raise.mojo");
  ctx.runCommand("mojo break-on-raise");
  ctx.resume();

  EXPECT_EQ(ctx.binary.getSource().findFirstLineWithText("# raises"),
            (int)ctx.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testBreakOnRaiseInlined) {
  // Test the break-on-raise feature in an @always_inline function.

  StopContext ctx = buildAndLaunch("raise_inlined.mojo");
  ctx.runCommand("mojo break-on-raise");
  ctx.resume();

  EXPECT_EQ(ctx.binary.getSource().findFirstLineWithText("# raises"),
            (int)ctx.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testBreakOnRaiseSecondTarget) {
  // Test that we break on raise on a second target as well as the first.
  //
  // Test is intended to be equivalent to the below lldb command sequence:
  //
  // target create raise
  // mojo break-on-raise
  // run  # Expect this to break on the raise
  //
  // target create raise
  // mojo break-on-raise
  // run  # Expect this to break on the raise

  StopContext ctx1 = buildAndLaunch("raise.mojo");
  ctx1.runCommand("mojo break-on-raise");
  ctx1.process.Continue();
  EXPECT_EQ(ctx1.binary.getSource().findFirstLineWithText("# raises"),
            (int)ctx1.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());

  StopContext ctx2 = buildAndLaunch("raise.mojo");
  ctx2.runCommand("mojo break-on-raise");
  ctx2.process.Continue();
  EXPECT_EQ(ctx2.binary.getSource().findFirstLineWithText("# raises"),
            (int)ctx2.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testDontAutomaticallyBreakOnRaise) {
  // Test that we don't break on raise without enabling the feature.

  StopContext ctx = buildAndLaunch("raise.mojo");
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}

TEST(BreakpointTest, testDontAutomaticallyBreakOnRaiseSecondTarget) {
  // Test that we don't break on raise on a second target just because we
  // enabled it for a first.
  //
  // Test is intended to be equivalent to the below lldb command sequence:
  //
  // target create raise
  // mojo break-on-raise
  // target create raise
  // run  # Expect this to not break on the raise
  // target select 0
  // run  # Expect this to break on the raise

  StopContext ctx1 = buildAndLaunch("raise.mojo");
  ctx1.runCommand("mojo break-on-raise enable");

  StopContext ctx2 = buildAndLaunch("raise.mojo");
  ctx2.process.Continue();
  EXPECT_EQ(ctx2.process.GetState(), lldb::eStateExited);

  ctx1.process.Continue();
  EXPECT_EQ(ctx1.binary.getSource().findFirstLineWithText("# raises"),
            (int)ctx1.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testDontBreakOnRaiseIfDisabled) {
  // Test that we don't break on raise if we disabled it.

  StopContext ctx = buildAndLaunch("raise.mojo");
  ctx.runCommand("mojo break-on-raise enable");
  ctx.runCommand("mojo break-on-raise disable");
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}

TEST(BreakpointTest, testDontBreakOnRaiseIfDisabledSecondTarget) {
  // Test that we don't break on raise if we disabled it for a single target
  // when we have two.
  //
  // Test is intended to be equivalent to the below lldb command sequence:
  //
  // target create raise
  // mojo break-on-raise
  // target create raise
  // mojo break-on-raise
  // target select 0
  // mojo break-on-raise disable
  // run  # Expect this to not break on the raise
  // target select 1
  // run  # Expect this to break on the raise

  StopContext ctx1 = buildAndLaunch("raise.mojo");
  ctx1.runCommand("mojo break-on-raise enable");

  StopContext ctx2 = buildAndLaunch("raise.mojo");
  ctx2.runCommand("mojo break-on-raise enable");

  ctx1.runCommand("mojo break-on-raise disable");
  ctx1.process.Continue();
  EXPECT_EQ(ctx1.process.GetState(), lldb::eStateExited);

  ctx2.process.Continue();
  EXPECT_EQ(ctx2.binary.getSource().findFirstLineWithText("# raises"),
            (int)ctx2.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testSymbolBreakpoints) {
  StopContext ctx = buildAndLaunch("symbol_breakpoints.mojo");
  ctx.runCommand("b simple_fn");
  ctx.runCommand("b parametrized_fn");
  ctx.runCommand("b parametrized_method");
  ctx.resume();

  int expectedLine =
      ctx.binary.getSource().findFirstLineWithText("# simple_fn stop");
  EXPECT_EQ(ctx.frame.GetLineEntry().GetLine(), expectedLine);

  ctx.resume();

  expectedLine =
      ctx.binary.getSource().findFirstLineWithText("# parametrized_fn stop");
  EXPECT_EQ(ctx.frame.GetLineEntry().GetLine(), expectedLine);

  ctx.resume();

  expectedLine = ctx.binary.getSource().findFirstLineWithText(
      "# parametrized_method stop");
  EXPECT_EQ(ctx.frame.GetLineEntry().GetLine(), expectedLine);
}
