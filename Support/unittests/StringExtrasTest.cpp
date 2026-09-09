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

#include "Support/StringExtras.h"

#include "gtest/gtest.h"

using namespace M;

TEST(StringExtrasTest, ReplaceAll) {
  std::string str = "hello";
  replaceAll(str, "l", "L");
  EXPECT_EQ(str, "heLLo");

  str = "hello";
  replaceAll(str, "l", "");
  EXPECT_EQ(str, "heo");

  str = "hello";
  replaceAll(str, "ll", "L");
  EXPECT_EQ(str, "heLo");

  str = "hello";
  replaceAll(str, "", "L");
  EXPECT_EQ(str, "hello");
}
