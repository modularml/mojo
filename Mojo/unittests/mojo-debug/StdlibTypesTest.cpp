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

TEST(StdlibTypesTest, testVariant) {
  GTEST_SKIP() << "Disabled due to MOTO-1590";
  StopContext ctx = buildAndLaunch("variant.mojo");

  // v = Variant[Int, String](42) — active type is Int
  SBValue v = ctx.frame.FindVariable("v");
  EXPECT_STREQ(v.GetSummary(), "Int(42)");

  ctx.resume();

  // v.set[String]("hello, world") — heap-encoded String.
  v = ctx.frame.FindVariable("v");
  EXPECT_STREQ(v.GetSummary(), "String(\"hello, world\")");

  ctx.resume();

  // v.set[String]("hi") — inline/small-string form.
  v = ctx.frame.FindVariable("v");
  EXPECT_STREQ(v.GetSummary(), "String(\"hi\")");

  ctx.resume();

  // v.set[String](String("")) — heap path with size == 0.
  v = ctx.frame.FindVariable("v");
  EXPECT_STREQ(v.GetSummary(), "String(\"\")");

  ctx.resume();

  // w = Variant[Int, Bool, String](True) — discriminant > 1 in a 3-way
  // variant, confirming the union's `GetChildAtIndex(discr)` indexing.
  SBValue w = ctx.frame.FindVariable("w");
  EXPECT_STREQ(w.GetSummary(), "Bool(True)");

  ctx.resume();

  // w.set[String]("last arm") — boundary case: last arm of a 3-way variant.
  w = ctx.frame.FindVariable("w");
  EXPECT_STREQ(w.GetSummary(), "String(\"last arm\")");
}

TEST(StdlibTypesTest, testOptional) {
  GTEST_SKIP() << "Disabled due to MOTO-1590";
  StopContext ctx = buildAndLaunch("optional.mojo");

  // opt_int = Optional(42) — Some with an Int payload.
  SBValue opt_int = ctx.frame.FindVariable("opt_int");
  EXPECT_STREQ(opt_int.GetSummary(), "Some(42)");

  ctx.resume();

  // opt_none = Optional[Int](None) — empty Optional.
  SBValue opt_none = ctx.frame.FindVariable("opt_none");
  EXPECT_STREQ(opt_none.GetSummary(), "None");

  ctx.resume();

  // opt_str = Optional(String("hello, world")) — heap-encoded String payload.
  SBValue opt_str = ctx.frame.FindVariable("opt_str");
  EXPECT_STREQ(opt_str.GetSummary(), "Some(\"hello, world\")");

  ctx.resume();

  // opt_str_small = Optional(String("hi")) — inline/small-string form.
  SBValue opt_str_small = ctx.frame.FindVariable("opt_str_small");
  EXPECT_STREQ(opt_str_small.GetSummary(), "Some(\"hi\")");

  ctx.resume();

  // opt_str_none = Optional[String](None) — empty Optional[String].
  SBValue opt_str_none = ctx.frame.FindVariable("opt_str_none");
  EXPECT_STREQ(opt_str_none.GetSummary(), "None");

  ctx.resume();

  // opt_bool = Optional(True) — exercises the GetSummaryAsCString path in
  // renderActivePayload (Bool has its own registered formatter).
  SBValue opt_bool = ctx.frame.FindVariable("opt_bool");
  EXPECT_STREQ(opt_bool.GetSummary(), "Some(True)");
}

TEST(StdlibTypesTest, testList) {
  /// Tests that List can be parsed correctly and its data formatter works
  /// correctly as well.
  StopContext ctx = buildAndLaunch("list.mojo");
  SBValue var = ctx.frame.FindVariable("point_vec");
  EXPECT_STREQ(var.GetSummary(), "(size 3)");
  EXPECT_STREQ(
      var.GetValueForExpressionPath("[0].x").GetChildAtIndex(0).GetValue(),
      "1");
  EXPECT_STREQ(
      var.GetValueForExpressionPath("[1].y").GetChildAtIndex(0).GetValue(),
      "-2");
  EXPECT_STREQ(
      var.GetValueForExpressionPath("[2].x").GetChildAtIndex(0).GetValue(),
      "3");

  ctx.resume();
  var = ctx.frame.FindVariable("int_vec");
  EXPECT_STREQ(var.GetSummary(), "(size 3)[1, 2, 3]");

  ctx.resume();
  var = ctx.frame.FindVariable("int_vec");
  EXPECT_STREQ(var.GetSummary(),
               "(size 103)[1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, ...]");
}

TEST(StdlibTypesTest, testUnsafePointerSummary) {
  StopContext ctx = buildAndLaunch("unsafe_pointer.mojo");

  // p_int = UnsafePointer[Int] pointing to 42
  SBValue p_int = ctx.frame.FindVariable("p_int");
  EXPECT_STREQ(p_int.GetSummary(), "42");

  ctx.resume();

  // p_neg = UnsafePointer[Int] pointing to -5 — exercises correct signed
  // display.
  SBValue p_neg = ctx.frame.FindVariable("p_neg");
  EXPECT_STREQ(p_neg.GetSummary(), "-5");

  ctx.resume();

  // p_bool = UnsafePointer[Bool] pointing to True — exercises Bool summary
  // path.
  SBValue p_bool = ctx.frame.FindVariable("p_bool");
  EXPECT_STREQ(p_bool.GetSummary(), "True");

  ctx.resume();

  // p_float = UnsafePointer[Float64] pointing to 3.125 — exercises scalar
  // path.
  SBValue p_float = ctx.frame.FindVariable("p_float");
  EXPECT_STREQ(p_float.GetSummary(), "3.125");
}

TEST(StdlibTypesTest, testDict) {
  StopContext ctx = buildAndLaunch("dict.mojo");

  // d = Dict[String, Int] with three entries inserted in order.
  SBValue d = ctx.frame.FindVariable("d");
  EXPECT_STREQ(d.GetSummary(), "(size 3)");
  EXPECT_EQ(d.GetNumChildren(), 3);
  // Entries are exposed in insertion order. Entry [0] is "one" → 1.
  EXPECT_STREQ(d.GetValueForExpressionPath("[0].key").GetSummary(), "\"one\"");
  EXPECT_STREQ(
      d.GetValueForExpressionPath("[0].value").GetChildAtIndex(0).GetValue(),
      "1");

  ctx.resume();

  // d2 = Dict[String, Int] with a single entry.
  SBValue d2 = ctx.frame.FindVariable("d2");
  EXPECT_STREQ(d2.GetSummary(), "(size 1)");
  EXPECT_EQ(d2.GetNumChildren(), 1);

  ctx.resume();

  // d3 = empty Dict[String, Int].
  SBValue d3 = ctx.frame.FindVariable("d3");
  EXPECT_STREQ(d3.GetSummary(), "(size 0)");
  EXPECT_EQ(d3.GetNumChildren(), 0);
}
