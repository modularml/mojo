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

TEST(LifetimesTest, testInlinedUser) {
  /// Ensures that the lifetime for a variable with an inlined last user is
  /// correct.
  StopContext ctx = buildAndLaunch("eager_destruction_inlined_user.mojo");
  SBValue foo = ctx.frame.FindVariable("foo");
  EXPECT_STREQ(foo.GetSummary(), R"("42")");
}

static void assertVarNotAvailable(StopContext &ctx, StringRef varName) {
  SBValue var = ctx.frame.FindVariable(varName.data());
  ASSERT_TRUE(StringRef(var.GetError().GetCString())
                  .contains("variable not available"));
}

static void assertVarAvailable(StopContext &ctx, StringRef varName) {
  SBValue var = ctx.frame.FindVariable(varName.data());
  ASSERT_TRUE(var.IsValid() && var.GetError().Success())
      << "expected variable '" << varName.data() << "' to be available";
}

TEST(LifetimesTest, testFullEagerDestruction) {
  /// Ensures that if a variable is completely destroyed eagerly, the
  /// lifetime of the value is reflected in DWARF.
  /// Non-trivial types (String) are eagerly destroyed and become unavailable.
  /// Trivial types (Int, SIMD) persist through the scope at -O0.
  StopContext ctx = buildAndLaunch("full_eager_destruction.mojo");
  SBValue text = ctx.frame.FindVariable("text");
  EXPECT_STREQ(text.GetSummary(), R"("hello")");

  for (size_t i = 0; i < 2; ++i) {
    ctx.resume();
    SBValue number = ctx.frame.FindVariable("number").GetChildAtIndex(0);
    EXPECT_EQ((int)number.GetValueAsSigned(), 8);
    assertVarNotAvailable(ctx, "text");
  }

  // Non-trivial types are dead after their last use.
  // Trivial types (number, simd) persist through the scope at -O0.
  ctx.resume();
  assertVarNotAvailable(ctx, "text");
  EXPECT_EQ((int)ctx.frame.FindVariable("number")
                .GetChildAtIndex(0)
                .GetValueAsSigned(),
            8);

  // Past the last use of all variables; non-trivial dead, trivial persists.
  ctx.resume();
  assertVarNotAvailable(ctx, "text");
  EXPECT_EQ((int)ctx.frame.FindVariable("number")
                .GetChildAtIndex(0)
                .GetValueAsSigned(),
            8);
  assertVarAvailable(ctx, "simd");

  // `text_moved` should be alive when breaking on the call.
  ctx.resume();
  SBValue textMoved = ctx.frame.FindVariable("text_moved");
  EXPECT_STREQ(textMoved.GetSummary(), R"("hello")");

  // This breakpoint is inside `take_string`.  `s` should be alive when breaking
  // on the print call.
  ctx.resume();
  SBValue s = ctx.frame.FindVariable("s");
  EXPECT_STREQ(s.GetSummary(), R"("hello")");

  // `text_moved` should be dead now.
  // `text_copied` should be alive when breaking on the call.
  ctx.resume();
  assertVarNotAvailable(ctx, "text_moved");
  SBValue textCopied = ctx.frame.FindVariable("text_copied");
  EXPECT_STREQ(textCopied.GetSummary(), R"("hello")");

  // This breakpoint is inside `take_string`. `s` should be alive when breaking
  // on the print call.
  ctx.resume();
  s = ctx.frame.FindVariable("s");
  EXPECT_STREQ(s.GetSummary(), R"("hello")");

  // `text_before` should be dead after the move.
  ctx.resume();

  // TODO: Why? assertVarNotAvailable(ctx, "text_before");
  SBValue textAfter = ctx.frame.FindVariable("text_after");
  EXPECT_STREQ(textAfter.GetSummary(), R"("hello")");

  // `text_after` should be dead now.
  // `number2` should be alive when breaking on the call.
  ctx.resume();
  assertVarNotAvailable(ctx, "text_after");
  SBValue number2 = ctx.frame.FindVariable("number2").GetChildAtIndex(0);
  EXPECT_EQ(number2.GetValueAsSigned(), 8);
}

TEST(LifetimesTest, testResurrection) {
  /// Ensures that if a variable is killed and re-initialized again, it is
  /// visible.
  StopContext ctx = buildAndLaunch("resurrection.mojo");

  SBValue text2 = ctx.frame.FindVariable("text2");
  EXPECT_STREQ(text2.GetSummary(), R"("hello")");
  // TODO: Why? assertVarNotAvailable(ctx, "text1");

  ctx.resume();
  SBValue text1 = ctx.frame.FindVariable("text1");
  EXPECT_STREQ(text1.GetSummary(), R"("hello")");
  assertVarNotAvailable(ctx, "text2");
}

TEST(LifetimesTest, testAsapDestructionNotAvailable) {
  /// Ensures that ASAP-destroyed non-trivial variables show as "not available"
  /// in the debugger after their last use, rather than showing wrong/garbage
  /// values (MOTO-1424).
  StopContext ctx = buildAndLaunch("asap_destruction_optimized.mojo");

  // First breakpoint: `print(text)` — `text` is alive at its last use.
  SBValue text = ctx.frame.FindVariable("text");
  EXPECT_STREQ(text.GetSummary(), R"("hello")");

  // Second breakpoint: `print("done")` — `text` has been ASAP-destroyed
  // and should show as "not available" (not garbage values).
  ctx.resume();
  assertVarNotAvailable(ctx, "text");
}

TEST(LifetimesTest, testRedefined) {
  /// Ensures that if a variable is redefined, it is visible.
  StopContext ctx = buildAndLaunch("redefined.mojo");

  SBValue x = ctx.frame.FindVariable("x").GetChildAtIndex(0);
  EXPECT_EQ((int)x.GetValueAsSigned(), 468);

  ctx.resume();
  SBValue y = ctx.frame.FindVariable("y");
  EXPECT_STREQ(y.GetSummary(), R"("world")");
}

TEST(LifetimesTest, testTrivialTypePersistence) {
  /// Tests that trivial types (Int, Float64, Bool) remain visible in the
  /// debugger past their last use point through the end of their scope.
  ///
  /// At -O0, the compiler suppresses debuginfo.kill markers for variables
  /// whose type has no destructor.  This keeps them visible in the debugger
  /// through the end of their lexical scope, because the -O0
  /// dbg.value→dbg.declare conversion gives them stable stack slots.
  StopContext ctx = buildAndLaunch("trivial_type_persistence.mojo");

  // Breakpoint 1: "after last use" — past the last use of my_int, my_float,
  // my_bool.  Trivial types persist through the scope.
  SBValue myInt = ctx.frame.FindVariable("my_int").GetChildAtIndex(0);
  EXPECT_EQ((int)myInt.GetValueAsSigned(), 42);
  assertVarAvailable(ctx, "my_float");
  assertVarAvailable(ctx, "my_bool");

  // Breakpoint 2: print(my_string) — my_string should be alive here.
  // Trivial types are still visible.
  ctx.resume();
  SBValue myString = ctx.frame.FindVariable("my_string");
  EXPECT_STREQ(myString.GetSummary(), R"("hello")");

  myInt = ctx.frame.FindVariable("my_int").GetChildAtIndex(0);
  EXPECT_EQ((int)myInt.GetValueAsSigned(), 42);
  assertVarAvailable(ctx, "my_float");
  assertVarAvailable(ctx, "my_bool");

  // Breakpoint 3: "end" — past last use of my_string (non-trivial, should be
  // dead). Trivial types still persist.
  ctx.resume();
  assertVarNotAvailable(ctx, "my_string");

  myInt = ctx.frame.FindVariable("my_int").GetChildAtIndex(0);
  EXPECT_EQ((int)myInt.GetValueAsSigned(), 42);
  assertVarAvailable(ctx, "my_float");
  assertVarAvailable(ctx, "my_bool");
}

TEST(LifetimesTest, testOriginTypesDoNotExtendNonTrivialLifetimes) {
  /// Verifies that extending debug lifetimes of trivial types does NOT
  /// transitively keep non-trivial values alive.  A trivial Int derived
  /// from a non-trivial String (via len()) must not prevent the String
  /// from being ASAP-destroyed.
  StopContext ctx = buildAndLaunch("origin_type_persistence.mojo");

  // Breakpoint 1: print(my_string) — my_string should be alive.
  SBValue myString = ctx.frame.FindVariable("my_string");
  EXPECT_STREQ(myString.GetSummary(), R"("hello")");

  // `length` (trivial Int) persists through the scope.
  SBValue length = ctx.frame.FindVariable("length").GetChildAtIndex(0);
  EXPECT_EQ((int)length.GetValueAsSigned(), 5);

  // Breakpoint 2: "after string use" — my_string (non-trivial) should be dead.
  // The trivial `length` must NOT keep my_string alive.
  ctx.resume();
  assertVarNotAvailable(ctx, "my_string");

  // But `length` (trivial) should still be visible.
  length = ctx.frame.FindVariable("length").GetChildAtIndex(0);
  EXPECT_EQ((int)length.GetValueAsSigned(), 5);
}
