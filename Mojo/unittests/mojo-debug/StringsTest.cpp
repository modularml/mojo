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

TEST(StringsTest, testStringSpan) {
  // Ensures that StaticString (= StringSpan[False, ImmStaticOrigin]) is
  // shown as a quoted string by the StringSpan summary formatter.
  StopContext ctx = buildAndLaunch("string_slice.mojo");

  SBValue s1 = ctx.frame.FindVariable("s1");
  EXPECT_STREQ(s1.GetSummary(), "\"static_string\"");

  SBValue s2 = ctx.frame.FindVariable("s2");
  EXPECT_STREQ(s2.GetSummary(), "\"\"");
}

TEST(StringsTest, testStrings) {
  // Ensures that String and StringLiteral can be parsed correctly from memory.
  StopContext ctx = buildAndLaunch("strings.mojo");

  SBValue st = ctx.frame.FindVariable("st");
  EXPECT_TRUE(StringRef(st.GetSummary()).contains("\"012345678910111213141"));

  ctx.resume();

  SBValue literal = ctx.frame.FindVariable("literal");
  SBValue s1 = ctx.frame.FindVariable("s1");
  SBValue s2 = ctx.frame.FindVariable("s2");
  SBValue s3 = ctx.frame.FindVariable("s3");
  SBValue s4 = ctx.frame.FindVariable("s4");

  // String, being parsed by a data formatter, provides the
  // underlying string as a Summary, following C++'s convention in LLDB.
  EXPECT_STREQ(s1.GetSummary(), "\"let_string\"");
  EXPECT_TRUE(StringRef(s2.GetSummary()).contains("\"012345678910111213141"));
  EXPECT_STREQ(s3.GetSummary(), "\"\"");
  EXPECT_TRUE(StringRef(s4.GetSummary()).contains("\"012345678910111213141"));
}
