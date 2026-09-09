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

#include "Support/IPRational.h"

#include "gtest/gtest.h"

using namespace M;

/// Check that constructed numbers are simplified.
TEST(IPRational, canonicalization) {
  IPRational three(IPInt(15), IPInt(5));
  EXPECT_EQ(three.getNumerator(), IPInt(3));
  EXPECT_EQ(three.getDenominator(), IPInt(1));
  IPRational oneAndAHalf(IPInt(15), IPInt(10));
  EXPECT_EQ(oneAndAHalf.getNumerator(), IPInt(3));
  EXPECT_EQ(oneAndAHalf.getDenominator(), IPInt(2));
  EXPECT_EQ(IPRational(-2, -1), IPRational(2, 1));
  EXPECT_EQ(IPRational(-2, -1).getNumerator(), IPInt(2));
  EXPECT_EQ(IPRational(-2, 1), IPRational(2, -1));
  EXPECT_EQ(IPRational(-2, 1).getNumerator(), IPInt(-2));
  EXPECT_EQ(IPRational(2, -1).getNumerator(), IPInt(-2));
  EXPECT_EQ(IPRational(0, 1).getNumerator(), IPInt(0));
  EXPECT_EQ(IPRational(0, -1).getNumerator(), IPInt(0));

  EXPECT_EQ(IPRational(0, 77).getDenominator(), IPInt(1));
  EXPECT_EQ(IPRational(0, -77).getDenominator(), IPInt(1));
}

TEST(IPRational, comparison) {
  for (unsigned i = 1; i < 100; ++i) {
    EXPECT_TRUE(IPRational(1, i) > IPRational(1, i + 1));
    EXPECT_TRUE(IPRational(1, i) >= IPRational(1, i + 1));
    EXPECT_FALSE(IPRational(1, i) == IPRational(1, i + 1));
    EXPECT_TRUE(IPRational(1, i) != IPRational(1, i + 1));
    EXPECT_TRUE(IPRational(1, i + 1) < IPRational(1, i));
    EXPECT_TRUE(IPRational(1, i + 1) <= IPRational(1, i));
    EXPECT_TRUE(IPRational(i, i) > IPRational(i, i + 1));
    EXPECT_TRUE(IPRational(i + 1, i) == IPRational((i + 1) * 2, i * 2));
  }
  EXPECT_FALSE(IPRational(1, 2) > IPRational(1, 2));
  EXPECT_FALSE(IPRational(1, 2) < IPRational(1, 2));
}

TEST(IPRational, arithmetic) {
  EXPECT_EQ(IPRational(1, 2) + IPRational(3, 10), IPRational(8, 10));
  EXPECT_EQ(IPRational(1, 2) - (-IPRational(3, 10)), IPRational(8, 10));
  EXPECT_EQ(IPRational(1, 2) + (-IPRational(3, 10)), IPRational(2, 10));
  EXPECT_EQ(IPRational(1, 2) - IPRational(3, 10), IPRational(2, 10));

  EXPECT_EQ(IPRational(2, 3) * IPRational(5, 7), IPRational(10, 21));
  EXPECT_EQ(IPRational(2, 3) * IPRational(-5, 7), IPRational(-10, 21));
  EXPECT_EQ(IPRational(3, 6) * IPRational(9, 12), IPRational(3, 8));
  EXPECT_EQ(IPRational(2, 3) / IPRational(5, 7), IPRational(14, 15));
  EXPECT_EQ(IPRational(3, 6) / IPRational(9, 12), IPRational(2, 3));

  EXPECT_EQ(IPRational(2, 1).abs(), IPRational(2, 1));
  EXPECT_EQ(IPRational(-2, 1).abs(), IPRational(2, 1));
}
