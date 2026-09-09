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
// Tests for ASSERT_STREAM
//===----------------------------------------------------------------------===//

#include "Support/AssertStream.h"

#include "gtest/gtest.h"

TEST(Assert, Aborts) {
  EXPECT_EXIT(ASSERT_STREAM(false, << "Error message"),
              testing::KilledBySignal(6), "Error message");
}
