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

#include "Cache/Support/Keys.h"
#include "gtest/gtest.h"

using namespace M::Cache;

namespace {

struct WrapOne {

  static std::string wrapKey(const std::string &key) { return key + "1"; }
};

struct WrapTwo {

  static std::string wrapKey(const std::string &key) { return key + "2"; }
};

using OneWrapped = Keys::WrappedKey<Keys::ReadOnlyKey, WrapOne>;

using OneTwoWrapped = Keys::WrappedKey<Keys::ReadOnlyKey, WrapOne, WrapTwo>;

using TwoOneWrapped = Keys::WrappedKey<Keys::ReadOnlyKey, WrapTwo, WrapOne>;

} // namespace

TEST(KeyTest, WrappedKeys) {

  std::string original = "test";
  EXPECT_EQ(original, Keys::ReadOnlyKey::hashKey(original));

  auto hasher = [](const std::string &input) -> std::string {
    llvm::BLAKE3 hashStateOne{};
    hashStateOne.update(input);
    auto hash = hashStateOne.final();
    return {hash.begin(), hash.end()};
  };

  // Result should be hash of "test1".
  std::string resultOne = hasher("test1");
  EXPECT_EQ(resultOne, OneWrapped::hashKey(original));

  // Result should be hash of "test12".
  std::string resultOneTwo = hasher("test12");
  EXPECT_EQ(resultOneTwo, OneTwoWrapped::hashKey(original));

  // Result should be hash of "test21".
  std::string resultTwoOne = hasher("test21");
  EXPECT_EQ(resultTwoOne, TwoOneWrapped::hashKey(original));
}
