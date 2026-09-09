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

#include "Support/IPInt.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace M;

TEST(IPInt, abs) {
  EXPECT_EQ(IPInt(5).abs(), IPInt(5));
  EXPECT_EQ(IPInt(-5).abs(), IPInt(5));
}

TEST(IPInt, gcd) {
  EXPECT_EQ(IPInt(15).gcd(IPInt(10)), IPInt(5));
  EXPECT_EQ(IPInt(5 * 6 * 7).gcd(2 * 3 * 4 * 5 * 6), IPInt(30));
}

TEST(IPInt, exponentiate) {
  EXPECT_EQ(IPInt(2).exponentiate(9), IPInt(512));
  EXPECT_EQ(IPInt(2).exponentiate(10), IPInt(1024));
  EXPECT_EQ(IPInt(10).exponentiate(0), IPInt(1));
  EXPECT_EQ(IPInt(10).exponentiate(1), IPInt(10));
  EXPECT_EQ(IPInt(10).exponentiate(2), IPInt(100));
  EXPECT_EQ(IPInt(10).exponentiate(3), IPInt(1000));
  EXPECT_EQ(IPInt(10).exponentiate(4), IPInt(10000));
  EXPECT_EQ(IPInt(10).exponentiate(5), IPInt(100000));
  EXPECT_EQ(IPInt(10).exponentiate(6), IPInt(1000000));
  EXPECT_EQ(IPInt(-3).exponentiate(0), IPInt(1));
  EXPECT_EQ(IPInt(-3).exponentiate(1), IPInt(-3));
  EXPECT_EQ(IPInt(-3).exponentiate(2), IPInt(9));
  EXPECT_EQ(IPInt(-3).exponentiate(3), IPInt(-27));
  EXPECT_EQ(IPInt(0).exponentiate(0), IPInt(1));
  EXPECT_EQ(IPInt(0).exponentiate(1), IPInt(0));
}

/// IPInts should always have the minimum bit width to encode the value as a
/// signed two's compliment integer.
TEST(IPInt, bitWidth) {
  EXPECT_EQ(IPInt(0).getAPInt().getBitWidth(), (unsigned)1);
  EXPECT_EQ(IPInt(-1).getAPInt().getBitWidth(), (unsigned)1);
  EXPECT_EQ(IPInt(1).getAPInt().getBitWidth(), (unsigned)2);
  EXPECT_EQ(IPInt(1024).getAPInt().getBitWidth(), (unsigned)12);
  for (unsigned i = 0; i < 100; ++i) {
    EXPECT_EQ((IPInt(1) << i).getAPInt().getBitWidth(), i + 2);
    EXPECT_EQ((-(IPInt(1) << i)).getAPInt().getBitWidth(), i + 1);
  }
}

TEST(IPInt, arithmetic) {
  for (int i = -20; i < 20; ++i) {
    EXPECT_EQ(-IPInt(i), IPInt(-i));
    for (int j = -20; j < 20; ++j) {
      EXPECT_EQ(IPInt(i) + IPInt(j), IPInt(i + j));
      EXPECT_EQ(IPInt(i) - IPInt(j), IPInt(i - j));
      EXPECT_EQ(IPInt(i) * IPInt(j), IPInt(i * j));
      if (j != 0) {
        EXPECT_EQ(IPInt(i) / IPInt(j), IPInt(i / j));
        EXPECT_EQ(IPInt(i) % IPInt(j), IPInt(i % j));
      }
      EXPECT_EQ(IPInt(i) < IPInt(j), i < j);
      EXPECT_EQ(IPInt(i) <= IPInt(j), i <= j);
      EXPECT_EQ(IPInt(i) > IPInt(j), i > j);
      EXPECT_EQ(IPInt(i) >= IPInt(j), i >= j);
      EXPECT_EQ(IPInt(i) == IPInt(j), i == j);
      EXPECT_EQ(IPInt(i) != IPInt(j), i != j);

      if (i >= 0 && j >= 0) {
        // For non-negative numbers, bitwise operations will be the same as
        // for fixed width ints.
        EXPECT_EQ(IPInt(i) & IPInt(j), IPInt(i & j));
        EXPECT_EQ(IPInt(i) | IPInt(j), IPInt(i | j));
        EXPECT_EQ(IPInt(i) ^ IPInt(j), IPInt(i ^ j));

        // For positive RHS, shifting will be the same as for fixed width ints.
        // Left shift on negative integers is undefined.
        // https://stackoverflow.com/questions/3784996/why-does-left-shift-operation-invoke-undefined-behaviour-when-the-left-side-oper
        // It is well defined for IPInt, but we shouldn't use normal integers as
        // a test case for it.
        EXPECT_EQ(IPInt(i) << IPInt(j), IPInt(i << j));
        EXPECT_EQ(IPInt(i) >> IPInt(j), IPInt(i >> j));
      }
    }
  }
}

TEST(IPInt, bitwiseLogicWithNegatives) {
  // Bitwise logic operators do sign extension to be the same width before
  // applying operators.
  // -1 is all ones at whatever bit width.
  EXPECT_EQ(IPInt(-1) & IPInt(0xffff), IPInt(0xffff));
  EXPECT_EQ(IPInt(-2) & IPInt(0xffff), IPInt(0xfffe));
  EXPECT_EQ(IPInt(-1) & IPInt(0x7f), IPInt(0x7f));
  EXPECT_EQ(IPInt(-1) & IPInt(0x7e), IPInt(0x7e));
  EXPECT_EQ(IPInt(-1) & IPInt(-2), IPInt(-2));
  EXPECT_EQ(IPInt(-1) | IPInt(0x7fff), IPInt(-1));
  EXPECT_EQ(IPInt(-2) | IPInt(0x7fff), IPInt(-1));
  EXPECT_EQ(IPInt(-1) | IPInt(-2), IPInt(-1));
  EXPECT_EQ(IPInt(-4) | IPInt(-2), IPInt(-2));
  EXPECT_EQ(IPInt(-1) ^ IPInt(0x7fff), IPInt(-0x7fff - 1));
  EXPECT_EQ(IPInt(-1) ^ IPInt(-4), IPInt(3));
}

TEST(IPInt, ostream) {
  std::string s;
  llvm::raw_string_ostream out(s);
  out << IPInt(0);
  EXPECT_EQ(out.str(), "0");
  s = "";
  out << IPInt(12357);
  EXPECT_EQ(out.str(), "12357");
  s = "";
  out << IPInt(-5678);
  EXPECT_EQ(out.str(), "-5678");
}
