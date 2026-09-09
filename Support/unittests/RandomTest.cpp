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

#include "Support/Random.h"
#include "llvm/ADT/ArrayRef.h"

#include "Support/ErrorOr.h"
#include "gtest/gtest.h"

using namespace M;

TEST(Random, Works) {
  SecureRandomBytesGenerator rng;

  SmallVector<uint8_t> buf;
  buf.resize(32);
  MutableArrayRef<uint8_t> randView(buf.begin(), buf.size());
  auto err = rng.getRandomBytes(randView);
  EXPECT_FALSE(err.isError()) << err.getError();
}
