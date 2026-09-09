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

#include "Support/MpscRingBuffer.h"

#include <atomic>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

using M::MpscRingBuffer;

namespace {

TEST(MpscRingBufferTest, SingleProducerRoundTrip) {
  MpscRingBuffer<int> ring(4);

  auto pos = ring.claim();
  ASSERT_TRUE(pos.has_value());
  ring.itemAt(*pos) = 42;
  ring.publish(*pos);

  int *item = ring.peek();
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(*item, 42);
  EXPECT_EQ(ring.frontPos(), *pos);
  ring.consume();

  EXPECT_EQ(ring.peek(), nullptr);
}

TEST(MpscRingBufferTest, BufferFullReturnsNullopt) {
  MpscRingBuffer<int> ring(4);

  for (int i = 0; i < 4; ++i) {
    auto pos = ring.claim();
    ASSERT_TRUE(pos.has_value()) << "expected slot " << i << " to be claimable";
    ring.itemAt(*pos) = i;
    ring.publish(*pos);
  }

  EXPECT_FALSE(ring.claim().has_value());
}

TEST(MpscRingBufferTest, DrainAndRefillReusesSlots) {
  MpscRingBuffer<int> ring(4);

  for (int cycle = 0; cycle < 16; ++cycle) {
    for (int i = 0; i < 4; ++i) {
      auto pos = ring.claim();
      ASSERT_TRUE(pos.has_value()) << "cycle " << cycle << " slot " << i;
      ring.itemAt(*pos) = cycle * 4 + i;
      ring.publish(*pos);
    }

    EXPECT_FALSE(ring.claim().has_value())
        << "ring should be full after cycle " << cycle;

    for (int i = 0; i < 4; ++i) {
      int *item = ring.peek();
      ASSERT_NE(item, nullptr) << "cycle " << cycle << " item " << i;
      EXPECT_EQ(*item, cycle * 4 + i);
      ring.consume();
    }

    EXPECT_EQ(ring.peek(), nullptr);
  }
}

TEST(MpscRingBufferTest, EnqueueAndConsumeCountsAreMonotonic) {
  MpscRingBuffer<int> ring(4);

  EXPECT_EQ(ring.enqueueCount(), 0u);
  EXPECT_EQ(ring.consumeCount(), 0u);

  auto pos = ring.claim();
  ASSERT_TRUE(pos.has_value());
  EXPECT_EQ(ring.enqueueCount(), 1u);
  EXPECT_EQ(ring.consumeCount(), 0u);

  ring.itemAt(*pos) = 7;
  ring.publish(*pos);

  EXPECT_NE(ring.peek(), nullptr);
  EXPECT_EQ(ring.consumeCount(), 0u); // held, not yet consumed
  ring.consume();
  EXPECT_EQ(ring.consumeCount(), 1u);
}

TEST(MpscRingBufferTest, FrontPosMatchesClaimedPosition) {
  MpscRingBuffer<int> ring(4);

  auto pos0 = ring.claim();
  ASSERT_TRUE(pos0.has_value());
  ring.itemAt(*pos0) = 10;
  ring.publish(*pos0);

  auto pos1 = ring.claim();
  ASSERT_TRUE(pos1.has_value());
  ring.itemAt(*pos1) = 20;
  ring.publish(*pos1);

  EXPECT_EQ(ring.frontPos(), *pos0);
  ring.consume();
  EXPECT_EQ(ring.frontPos(), *pos1);
  ring.consume();
}

// Concurrent inserts from multiple producers — run under TSan to catch races.
TEST(MpscRingBufferTest, MultiProducerConcurrentInserts) {
  constexpr size_t kCapacity = 2048;
  constexpr int kProducers = 4;
  constexpr int kItemsPerProducer = 256; // total 1024 < kCapacity, so no drops

  MpscRingBuffer<int> ring(kCapacity);
  std::atomic<int> dropped{0};

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kItemsPerProducer; ++i) {
        auto pos = ring.claim();
        if (!pos) {
          dropped.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        ring.itemAt(*pos) = p * kItemsPerProducer + i;
        ring.publish(*pos);
      }
    });
  }

  for (auto &t : producers)
    t.join();

  // Drain on the main thread (single consumer).
  int consumed = 0;
  while (ring.peek()) {
    ring.consume();
    ++consumed;
  }

  EXPECT_EQ(dropped.load(), 0)
      << "ring was sized to hold all items; no drops expected";
  EXPECT_EQ(consumed, kProducers * kItemsPerProducer);
  EXPECT_EQ(ring.enqueueCount(), ring.consumeCount());
}

// Concurrent producers and a concurrent consumer — exercises the full
// producer/consumer protocol under TSan. The consumer runs in its own thread
// draining the ring while the producers are still inserting.
TEST(MpscRingBufferTest, MultiProducerConcurrentConsumer) {
  constexpr size_t kCapacity = 2048;
  constexpr int kProducers = 4;
  constexpr int kItemsPerProducer = 256; // 1024 total < kCapacity, no drops
  constexpr int kTotal = kProducers * kItemsPerProducer;

  MpscRingBuffer<int> ring(kCapacity);
  std::atomic<bool> stop{false};
  std::atomic<int> consumed{0};
  std::atomic<int> dropped{0};

  std::thread consumer([&] {
    while (!stop.load(std::memory_order_acquire) || ring.peek()) {
      if (ring.peek()) {
        ring.consume();
        consumed.fetch_add(1, std::memory_order_relaxed);
      } else {
        std::this_thread::yield();
      }
    }
  });

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kItemsPerProducer; ++i) {
        auto pos = ring.claim();
        if (!pos) {
          dropped.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        ring.itemAt(*pos) = p * kItemsPerProducer + i;
        ring.publish(*pos);
      }
    });
  }

  for (auto &t : producers)
    t.join();
  // All producers done: no new items can arrive. Signal consumer to drain
  // whatever remains and exit.
  stop.store(true, std::memory_order_release);
  consumer.join();

  EXPECT_EQ(dropped.load(), 0)
      << "ring sized to hold all items; no drops expected";
  EXPECT_EQ(consumed.load(), kTotal);
  EXPECT_EQ(ring.enqueueCount(), ring.consumeCount());
}

} // namespace
