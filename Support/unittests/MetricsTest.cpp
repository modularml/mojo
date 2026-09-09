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

#include "Support/Metrics.h"

#include <thread>
#include <vector>

#include "gtest/gtest.h"

using namespace M::Metrics;

namespace {

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------

TEST(MetricsTest, CounterAccumulates) {
  MetricsCollector mc;
  Counter &c = mc.registerCounter("test.counter");
  c.increment();
  c.increment();
  c.increment();
  EXPECT_EQ(c.read(), 3u);
}

// Counter values are cumulative: collect() does not reset them.
TEST(MetricsTest, CounterPreservedAcrossCollect) {
  MetricsCollector mc;
  Counter &c = mc.registerCounter("test.counter");
  c.increment();
  c.increment();

  auto snap1 = mc.collect();
  ASSERT_EQ(snap1.counters.size(), 1u);
  EXPECT_EQ(snap1.counters[0].value, 2u);

  c.increment();

  auto snap2 = mc.collect();
  ASSERT_EQ(snap2.counters.size(), 1u);
  EXPECT_EQ(snap2.counters[0].value, 3u);
}

// Registering the same name twice returns the same instrument.
TEST(MetricsTest, CounterDoubleRegisterIdempotent) {
  MetricsCollector mc;
  Counter &c1 = mc.registerCounter("test.counter");
  Counter &c2 = mc.registerCounter("test.counter");
  EXPECT_EQ(&c1, &c2);
  c1.increment();
  EXPECT_EQ(c2.read(), 1u);
}

// ---------------------------------------------------------------------------
// Gauge
// ---------------------------------------------------------------------------

TEST(MetricsTest, GaugeAccumulates) {
  MetricsCollector mc;
  Gauge &g = mc.registerGauge("test.gauge");
  g.increase(5);
  g.decrease(2);
  EXPECT_EQ(g.read(), 3);
  g.set(100);
  EXPECT_EQ(g.read(), 100);
}

TEST(MetricsTest, GaugePreservedAcrossCollect) {
  MetricsCollector mc;
  Gauge &g = mc.registerGauge("test.gauge");
  g.set(42);

  auto snap1 = mc.collect();
  ASSERT_EQ(snap1.gauges.size(), 1u);
  EXPECT_EQ(snap1.gauges[0].value, 42);

  auto snap2 = mc.collect();
  ASSERT_EQ(snap2.gauges.size(), 1u);
  EXPECT_EQ(snap2.gauges[0].value, 42);
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

TEST(MetricsTest, HistogramAccumulates) {
  MetricsCollector mc;
  Histogram &h = mc.registerHistogram("test.histogram");
  h.record(1.0);
  h.record(3.0);
  h.record(5.0);

  auto snap = mc.collect();
  ASSERT_EQ(snap.histograms.size(), 1u);
  EXPECT_EQ(snap.histograms[0].ss.count, 3u);
  EXPECT_DOUBLE_EQ(snap.histograms[0].ss.sum, 9.0);
  EXPECT_DOUBLE_EQ(snap.histograms[0].ss.min, 1.0);
  EXPECT_DOUBLE_EQ(snap.histograms[0].ss.max, 5.0);
}

// Histogram snapshots reset on each collect(); a second collect() with no new
// records should return a zero snapshot.
TEST(MetricsTest, HistogramResetsOnCollect) {
  MetricsCollector mc;
  Histogram &h = mc.registerHistogram("test.histogram");
  h.record(42.0);

  auto snap1 = mc.collect();
  ASSERT_EQ(snap1.histograms.size(), 1u);
  EXPECT_EQ(snap1.histograms[0].ss.count, 1u);

  auto snap2 = mc.collect();
  ASSERT_EQ(snap2.histograms.size(), 1u);
  EXPECT_EQ(snap2.histograms[0].ss.count, 0u);
  EXPECT_DOUBLE_EQ(snap2.histograms[0].ss.sum, 0.0);
}

// ---------------------------------------------------------------------------
// Concurrent updates
// ---------------------------------------------------------------------------

TEST(MetricsTest, ConcurrentCounterIncrements) {
  constexpr int kThreads = 8;
  constexpr int kIncrementsPerThread = 10'000;

  MetricsCollector mc;
  Counter &c = mc.registerCounter("test.counter");

  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i)
    threads.emplace_back([&] {
      for (int j = 0; j < kIncrementsPerThread; ++j)
        c.increment();
    });
  for (auto &t : threads)
    t.join();

  EXPECT_EQ(c.read(), static_cast<uint64_t>(kThreads * kIncrementsPerThread));
}

TEST(MetricsTest, ConcurrentHistogramRecords) {
  constexpr int kThreads = 8;
  constexpr int kRecordsPerThread = 1'000;

  MetricsCollector mc;
  Histogram &h = mc.registerHistogram("test.histogram");

  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i)
    threads.emplace_back([&] {
      for (int j = 0; j < kRecordsPerThread; ++j)
        h.record(1.0);
    });
  for (auto &t : threads)
    t.join();

  auto snap = mc.collect();
  ASSERT_EQ(snap.histograms.size(), 1u);
  EXPECT_EQ(snap.histograms[0].ss.count,
            static_cast<uint64_t>(kThreads * kRecordsPerThread));
  EXPECT_DOUBLE_EQ(snap.histograms[0].ss.sum,
                   static_cast<double>(kThreads * kRecordsPerThread));
}

TEST(MetricsTest, ConcurrentGaugeUpdates) {
  constexpr int kThreads = 8;
  constexpr int kOpsPerThread = 1'000;

  MetricsCollector mc;
  Gauge &g = mc.registerGauge("test.gauge");

  // Half the threads increase, half decrease by the same total — net must be 0.
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i)
    threads.emplace_back([&, i] {
      for (int j = 0; j < kOpsPerThread; ++j) {
        if (i % 2 == 0)
          g.increase(1);
        else
          g.decrease(1);
      }
    });
  for (auto &t : threads)
    t.join();

  EXPECT_EQ(g.read(), 0);
}

// Multiple threads registering the same name and then updating — exercises
// the registration mutex and confirms idempotent registration under contention.
TEST(MetricsTest, ConcurrentRegistrationAndUpdate) {
  constexpr int kThreads = 4;
  constexpr int kIncrementsPerThread = 1'000;

  MetricsCollector mc;

  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i)
    threads.emplace_back([&] {
      Counter &c = mc.registerCounter("shared.counter");
      for (int j = 0; j < kIncrementsPerThread; ++j)
        c.increment();
    });
  for (auto &t : threads)
    t.join();

  auto snap = mc.collect();
  ASSERT_EQ(snap.counters.size(), 1u);
  EXPECT_EQ(snap.counters[0].value,
            static_cast<uint64_t>(kThreads * kIncrementsPerThread));
}

} // namespace
