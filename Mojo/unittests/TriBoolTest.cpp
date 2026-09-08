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

#include "Mojo/include/Mojo/Support/TriBool.h"
#include "gtest/gtest.h"

#include <array>

using M::KGEN::allTrue;
using M::KGEN::TriBool;

namespace {

TEST(TriBoolTest, factoriesAndPredicates) {
  EXPECT_TRUE(TriBool::yes().isTrue());
  EXPECT_FALSE(TriBool::yes().isFalse());
  EXPECT_FALSE(TriBool::yes().isUnknown());
  EXPECT_TRUE(TriBool::yes().isDefinite());

  EXPECT_TRUE(TriBool::no().isFalse());
  EXPECT_FALSE(TriBool::no().isTrue());
  EXPECT_FALSE(TriBool::no().isUnknown());
  EXPECT_TRUE(TriBool::no().isDefinite());

  EXPECT_TRUE(TriBool::unknown().isUnknown());
  EXPECT_FALSE(TriBool::unknown().isTrue());
  EXPECT_FALSE(TriBool::unknown().isFalse());
  EXPECT_FALSE(TriBool::unknown().isDefinite());
}

TEST(TriBoolTest, fromBoolAndToOptionalBool) {
  EXPECT_TRUE(TriBool::fromBool(true).isTrue());
  EXPECT_TRUE(TriBool::fromBool(false).isFalse());

  EXPECT_EQ(TriBool::yes().toOptionalBool(), std::optional<bool>(true));
  EXPECT_EQ(TriBool::no().toOptionalBool(), std::optional<bool>(false));
  EXPECT_EQ(TriBool::unknown().toOptionalBool(), std::nullopt);
}

TEST(TriBoolTest, equality) {
  EXPECT_EQ(TriBool::yes(), TriBool::yes());
  EXPECT_NE(TriBool::yes(), TriBool::no());
  EXPECT_NE(TriBool::yes(), TriBool::unknown());
  EXPECT_NE(TriBool::no(), TriBool::unknown());
}

TEST(TriBoolTest, kleeneAndTruthTable) {
  const TriBool y = TriBool::yes(), n = TriBool::no(), u = TriBool::unknown();

  // Any `no` dominates.
  EXPECT_EQ(n & n, n);
  EXPECT_EQ(n & y, n);
  EXPECT_EQ(y & n, n);
  EXPECT_EQ(n & u, n);
  EXPECT_EQ(u & n, n);

  // Otherwise any `unknown` yields `unknown`.
  EXPECT_EQ(u & u, u);
  EXPECT_EQ(y & u, u);
  EXPECT_EQ(u & y, u);

  // All `yes` yields `yes`.
  EXPECT_EQ(y & y, y);
}

TEST(TriBoolTest, kleeneOrTruthTable) {
  const TriBool y = TriBool::yes(), n = TriBool::no(), u = TriBool::unknown();

  // Any `yes` dominates.
  EXPECT_EQ(y | y, y);
  EXPECT_EQ(y | n, y);
  EXPECT_EQ(n | y, y);
  EXPECT_EQ(y | u, y);
  EXPECT_EQ(u | y, y);

  // Otherwise any `unknown` yields `unknown`.
  EXPECT_EQ(u | u, u);
  EXPECT_EQ(n | u, u);
  EXPECT_EQ(u | n, u);

  // All `no` yields `no`.
  EXPECT_EQ(n | n, n);
}

TEST(TriBoolTest, compoundAssignment) {
  TriBool acc = TriBool::yes();
  acc &= TriBool::unknown();
  EXPECT_EQ(acc, TriBool::unknown());
  acc &= TriBool::no();
  EXPECT_EQ(acc, TriBool::no());

  TriBool orAcc = TriBool::no();
  orAcc |= TriBool::unknown();
  EXPECT_EQ(orAcc, TriBool::unknown());
  orAcc |= TriBool::yes();
  EXPECT_EQ(orAcc, TriBool::yes());
}

TEST(TriBoolTest, allTrueFold) {
  // Empty range is vacuously true.
  EXPECT_EQ(allTrue(std::array<TriBool, 0>{}), TriBool::yes());

  EXPECT_EQ(allTrue(std::array{TriBool::yes(), TriBool::yes()}),
            TriBool::yes());
  EXPECT_EQ(allTrue(std::array{TriBool::yes(), TriBool::unknown()}),
            TriBool::unknown());
  EXPECT_EQ(
      allTrue(std::array{TriBool::yes(), TriBool::unknown(), TriBool::no()}),
      TriBool::no());
}

// constexpr usability: the algebra must be evaluable at compile time.
TEST(TriBoolTest, constexprEvaluation) {
  static_assert((TriBool::yes() & TriBool::unknown()).isUnknown());
  static_assert((TriBool::no() | TriBool::yes()).isTrue());
  static_assert(TriBool::fromBool(true).isTrue());
  SUCCEED();
}

} // namespace
