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

TEST(ArtificialsTest, testArtificialArguments) {
  StopContext ctx = buildAndLaunch("artificials.mojo");

  SBValueList visibleVariables = ctx.frame.GetVariables(/*arguments=*/true,
                                                        /*locals=*/true,
                                                        /*statics=*/false,
                                                        /*in_scope_only=*/true);

  EXPECT_EQ((int)visibleVariables.GetSize(), 2);
  EXPECT_TRUE(visibleVariables.GetFirstValueByName("a").IsValid());
  EXPECT_TRUE(visibleVariables.GetFirstValueByName("b").IsValid());

  EXPECT_TRUE(ctx.frame.FindVariable("__result__").IsValid());
  EXPECT_TRUE(ctx.frame.FindVariable("__error__").IsValid());
}
