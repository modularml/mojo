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
//
// Measures the throughput of concurrent MLOG_INFO calls through the ring
// buffer under two scenarios:
//
//   normal    — message count low enough that the ring rarely saturates;
//               nearly all attempted records are consumed.
//   saturated — message count high enough to fill the ring; producers drop
//               records and the consumer becomes the bottleneck.
//
// Throughput is reported as consumed (not attempted) records per second, so
// the number reflects actual work done rather than the cheap drop path.
// The drop count makes saturation visible.
//
// A /dev/null file sink is used so sink I/O does not artificially throttle
// the consumer. Set MODULAR_LOG_FILE before running to measure with a real
// sink.
//
// Run with:
//   ./bazelw run //Support/benchmarks/Log:LogThroughputBenchmark

#include "Support/Log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace M::Log;
using Clock = std::chrono::steady_clock;

static constexpr size_t kMessagesNormal = 40'000;
static constexpr size_t kMessagesSaturated = 200'000;
static constexpr size_t kWarmupMessages = 10'000;
static constexpr size_t kFlushRounds = 500;
static constexpr size_t kMsgsPerFlush = 100;
// Work units per thread for the mixed workload benchmark. At ~14 ns/unit with
// 16 threads, the wall time per run stays under ~200 ms across all densities.
static constexpr size_t kWorkUnitsPerThread = 2'000'000;
// Number of timed runs per scenario. The median is reported; outliers caused by
// OS scheduling jitter (especially at high thread counts) are suppressed.
static constexpr int kRunsPerScenario = 5;

// ===-----------------------------------------------------------------------===#
// Throughput
// ===-----------------------------------------------------------------------===#

struct ThroughputResult {
  const char *scenario;
  size_t threadCount;
  size_t attempted;
  size_t consumed;
  std::chrono::nanoseconds duration;

  size_t dropped() const {
    return attempted > consumed ? attempted - consumed : 0;
  }

  double messagesPerSecond() const {
    return static_cast<double>(consumed) /
           (static_cast<double>(duration.count()) * 1e-9);
  }

  double nsPerMessage() const {
    if (consumed == 0)
      return 0.0;
    return static_cast<double>(duration.count()) /
           static_cast<double>(consumed);
  }
};

// Runs messagesPerThread * threadCount enqueue attempts against `log`. Records
// processedCount() before and after flush so the result reflects actual work.
static ThroughputResult runThroughput(Logger &log, const char *scenario,
                                      size_t threadCount,
                                      size_t messagesPerThread) {
  std::atomic<bool> go{false};
  std::atomic<size_t> finishedCount{0};
  std::vector<std::thread> threads;
  threads.reserve(threadCount);

  for (size_t t = 0; t < threadCount; ++t) {
    threads.emplace_back([t, &log, &go, &finishedCount, messagesPerThread] {
      while (!go.load(std::memory_order_acquire))
        std::this_thread::yield();
      for (size_t i = 0; i < messagesPerThread; ++i)
        logWriteDispatch(log, LogLevel::INFO, Channel::Default,
                         "bench t={} i={}", t, i);
      finishedCount.fetch_add(1, std::memory_order_release);
    });
  }

  // Snapshot before go so the baseline is clean (previous flush drained ring).
  size_t consumedBefore = log.processedCount();

  auto start = Clock::now();
  go.store(true, std::memory_order_release);
  while (finishedCount.load(std::memory_order_acquire) < threadCount)
    std::this_thread::yield();
  auto end = Clock::now();

  for (auto &th : threads)
    th.join();

  log.flush();
  size_t consumedAfter = log.processedCount();

  return {scenario, threadCount, threadCount * messagesPerThread,
          consumedAfter - consumedBefore,
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)};
}

// Run the scenario kRunsPerScenario times and return the median by msg/sec.
// Sorting by throughput puts scheduler-preempted (slow) runs at the low end
// and CPU-contention-free (fast) runs at the high end; the median is stable.
static ThroughputResult
medianRun(Logger &log, std::function<ThroughputResult(Logger &)> runFn) {
  std::vector<ThroughputResult> runs;
  runs.reserve(kRunsPerScenario);
  for (int i = 0; i < kRunsPerScenario; ++i) {
    runs.push_back(runFn(log));
    log.flush();
  }
  std::nth_element(runs.begin(), runs.begin() + kRunsPerScenario / 2,
                   runs.end(),
                   [](const ThroughputResult &a, const ThroughputResult &b) {
                     return a.messagesPerSecond() < b.messagesPerSecond();
                   });
  return runs[kRunsPerScenario / 2];
}

static void printThroughputHeader() {
  std::printf("scenario,threads,attempted,consumed,dropped,msg_per_sec,"
              "ns_per_msg\n");
}

static void printResult(const ThroughputResult &r) {
  std::printf("%s,%zu,%zu,%zu,%zu,%.0f,%.1f\n", r.scenario, r.threadCount,
              r.attempted, r.consumed, r.dropped(), r.messagesPerSecond(),
              r.nsPerMessage());
  std::fflush(stdout);
}

// ===-----------------------------------------------------------------------===#
// Mixed workload (realistic: compute + infrequent logging)
// ===-----------------------------------------------------------------------===#

static inline double trigWork(double x) {
  return std::atan(std::cos(std::sin(x)));
}

// Each thread does kWorkUnitsPerThread trig ops and logs once every logEvery
// iterations. The scenario name encodes the log density.
static ThroughputResult runWorkload(Logger &log, const char *scenario,
                                    size_t threadCount, size_t logEvery) {
  std::atomic<bool> go{false};
  std::atomic<size_t> finishedCount{0};
  std::vector<std::thread> threads;
  threads.reserve(threadCount);

  for (size_t t = 0; t < threadCount; ++t) {
    threads.emplace_back([t, &log, &go, &finishedCount, logEvery] {
      while (!go.load(std::memory_order_acquire))
        std::this_thread::yield();
      double x = static_cast<double>(t) * 0.001;
      for (size_t i = 0; i < kWorkUnitsPerThread; ++i) {
        x = trigWork(x);
        if ((i + 1) % logEvery == 0)
          logWriteDispatch(log, LogLevel::INFO, Channel::Default,
                           "work t={} x={}", t, x);
      }
      finishedCount.fetch_add(1, std::memory_order_release);
    });
  }

  size_t consumedBefore = log.processedCount();
  auto start = Clock::now();
  go.store(true, std::memory_order_release);
  while (finishedCount.load(std::memory_order_acquire) < threadCount)
    std::this_thread::yield();
  auto end = Clock::now();

  for (auto &th : threads)
    th.join();
  log.flush();
  size_t consumedAfter = log.processedCount();

  size_t attempted = threadCount * (kWorkUnitsPerThread / logEvery);
  return {scenario, threadCount, attempted, consumedAfter - consumedBefore,
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)};
}

// ===-----------------------------------------------------------------------===#
// Flush latency
// ===-----------------------------------------------------------------------===#

static void benchmarkFlush(Logger &log) {
  std::vector<double> samples;
  samples.reserve(kFlushRounds);

  for (size_t r = 0; r < kFlushRounds; ++r) {
    for (size_t i = 0; i < kMsgsPerFlush; ++i)
      logWriteDispatch(log, LogLevel::INFO, Channel::Default, "flush bench {}",
                       i);
    auto t0 = Clock::now();
    log.flush();
    auto t1 = Clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  std::sort(samples.begin(), samples.end());
  double median = samples[samples.size() / 2];
  double p95 = samples[static_cast<size_t>(samples.size() * 0.95)];
  std::printf("flush_latency_us: median=%.2f p95=%.2f\n", median, p95);
  std::fflush(stdout);
}

// ===-----------------------------------------------------------------------===#
// Suite
// ===-----------------------------------------------------------------------===#

static void warmup(Logger &log) {
  for (size_t i = 0; i < kWarmupMessages; ++i)
    logWriteDispatch(log, LogLevel::INFO, Channel::Default, "warmup i={}", i);
  log.flush();
}

static void runSuite(Logger &log) {
  warmup(log);
  benchmarkFlush(log);
  std::printf("\n");
  printThroughputHeader();

  for (size_t threadCount : {1u, 2u, 4u, 8u, 16u}) {
    for (auto [scenario, msgCount] :
         {std::pair{"normal", kMessagesNormal},
          std::pair{"saturated", kMessagesSaturated}}) {
      // One warmup run to get the ring and consumer into steady state, then
      // kRunsPerScenario timed runs; the median is reported.
      runThroughput(log, scenario, threadCount, msgCount);
      log.flush();
      printResult(medianRun(log, [=](Logger &l) {
        return runThroughput(l, scenario, threadCount, msgCount);
      }));
    }
  }

  std::printf("\n");
  std::printf(
      "# Mixed workload: %zu work units/thread, log every N iterations\n",
      kWorkUnitsPerThread);
  printThroughputHeader();

  for (size_t threadCount : {1u, 2u, 4u, 8u, 16u}) {
    for (auto [scenario, logEvery] :
         {std::pair{"workload_every_1", 1u},
          std::pair{"workload_every_10", 10u},
          std::pair{"workload_every_100", 100u},
          std::pair{"workload_every_1000", 1000u}}) {
      runWorkload(log, scenario, threadCount, logEvery); // warmup
      log.flush();
      printResult(medianRun(log, [=](Logger &l) {
        return runWorkload(l, scenario, threadCount, logEvery);
      }));
    }
  }
}

int main() {
  if (!getenv("MODULAR_LOG_FILE"))
    ::setenv("MODULAR_LOG_FILE", "/dev/null", 1);
  ::setenv("MODULAR_LOG_STDOUT", "false", 1);
  ::setenv("MODULAR_LOG_NO_SUMMARY", "1", 1);

  Logger log;
  log.setLogLevel(LogLevel::INFO);
  runSuite(log);

  return EXIT_SUCCESS;
}
