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

// Covers the ambient request context on its own. What a log record does with
// it is covered by RequestLogTest, which is a separate binary because this one
// deliberately does not link the logger.

#include "Support/RequestContext.h"

#include <thread>

#include "gtest/gtest.h"

using namespace M::Request;

namespace {

class RequestContextTest : public ::testing::Test {
protected:
  void TearDown() override { clearBatchId(); }
};

TEST_F(RequestContextTest, UnsetByDefault) {
  EXPECT_EQ(currentBatchId(), nullptr);
}

TEST_F(RequestContextTest, SetThenReadRoundTrips) {
  setBatchId(42);
  ASSERT_NE(currentBatchId(), nullptr);
  EXPECT_EQ(*currentBatchId(), 42);
}

TEST_F(RequestContextTest, ClearResetsToUnset) {
  setBatchId(42);
  clearBatchId();
  EXPECT_EQ(currentBatchId(), nullptr);
}

TEST_F(RequestContextTest, ClearIsSafeWhenNothingIsSet) {
  clearBatchId();
  EXPECT_EQ(currentBatchId(), nullptr);
}

TEST_F(RequestContextTest, SecondSetOverwritesTheFirst) {
  setBatchId(1);
  setBatchId(2);
  ASSERT_NE(currentBatchId(), nullptr);
  EXPECT_EQ(*currentBatchId(), 2);
}

// Zero is a legitimate batch id — the scheduler's counter starts there — so it
// has to be distinguishable from no batch at all.
TEST_F(RequestContextTest, ZeroIsDistinctFromUnset) {
  setBatchId(0);
  ASSERT_NE(currentBatchId(), nullptr);
  EXPECT_EQ(*currentBatchId(), 0);
}

TEST_F(RequestContextTest, NegativeIdRoundTrips) {
  setBatchId(-1);
  ASSERT_NE(currentBatchId(), nullptr);
  EXPECT_EQ(*currentBatchId(), -1);
}

TEST_F(RequestContextTest, IdWiderThan32BitsRoundTrips) {
  constexpr int64_t wide = int64_t{1} << 62;
  setBatchId(wide);
  ASSERT_NE(currentBatchId(), nullptr);
  EXPECT_EQ(*currentBatchId(), wide);
}

TEST_F(RequestContextTest, ScopeUnbindsOnExit) {
  {
    BatchScope batch(7);
    ASSERT_NE(currentBatchId(), nullptr);
    EXPECT_EQ(*currentBatchId(), 7);
  }
  EXPECT_EQ(currentBatchId(), nullptr);
}

// The restore is what a bare clear cannot do: it has no way to know an outer
// scope was holding a different id.
TEST_F(RequestContextTest, NestedScopesShadowThenRestore) {
  BatchScope outer(1);
  {
    BatchScope inner(2);
    EXPECT_EQ(*currentBatchId(), 2);
  }
  ASSERT_NE(currentBatchId(), nullptr);
  EXPECT_EQ(*currentBatchId(), 1);
}

TEST_F(RequestContextTest, ScopeRestoresOverAManualSet) {
  setBatchId(1);
  {
    BatchScope batch(2);
    EXPECT_EQ(*currentBatchId(), 2);
  }
  ASSERT_NE(currentBatchId(), nullptr);
  EXPECT_EQ(*currentBatchId(), 1);
}

// The context is per thread, so a sibling thread must not pick it up. This is
// the documented limitation for runtime work that fans out.
TEST_F(RequestContextTest, ContextDoesNotLeakToOtherThreads) {
  BatchScope batch(7);
  bool sawContextOnOtherThread = true;
  std::thread other([&] { sawContextOnOtherThread = currentBatchId(); });
  other.join();
  EXPECT_FALSE(sawContextOnOtherThread);
}

TEST_F(RequestContextTest, EachThreadKeepsItsOwnId) {
  BatchScope batch(7);
  int64_t observedOnWorker = 0;
  std::thread other([&] {
    BatchScope workerBatch(99);
    observedOnWorker = *currentBatchId();
  });
  other.join();
  EXPECT_EQ(observedOnWorker, 99);
  EXPECT_EQ(*currentBatchId(), 7);
}

} // namespace
