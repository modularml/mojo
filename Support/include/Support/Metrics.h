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
// In-process metrics collection for the Modular runtime.
//
// Callers register named instruments once at startup via MetricsCollector and
// retain a direct reference for the lifetime of the process:
//
//   Counter& reqs = M::Metrics::metrics().registerCounter("my.requests");
//   reqs.increment();
//
// A periodic Python task calls collect() to snapshot all values and forward
// them to the OTel/Prometheus export pipeline. Counter and Gauge values are
// cumulative (the Python layer computes deltas); Histogram snapshots reset on
// each collect() call.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_METRICS_H
#define SUPPORT_METRICS_H

#include "Support/SymbolExport.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace M::Metrics {

// Counters hold a single monotonically-increasing value of type uint64_t.
class Counter {
  std::atomic<uint64_t> count = 0;

public:
  void increment() noexcept { count.fetch_add(1, std::memory_order_relaxed); }

  uint64_t read() const noexcept {
    return count.load(std::memory_order_relaxed);
  }
};

// Gauges are a single int64_t value which can change both up and down,
// as well as discontinuously.
class Gauge {
  std::atomic<int64_t> gauge = 0;

public:
  void decrease(int64_t delta = 1) noexcept {
    gauge.fetch_sub(delta, std::memory_order_relaxed);
  }

  void increase(int64_t delta = 1) noexcept {
    gauge.fetch_add(delta, std::memory_order_relaxed);
  }

  void set(int64_t value) noexcept {
    gauge.store(value, std::memory_order_relaxed);
  }

  int64_t read() const noexcept {
    return gauge.load(std::memory_order_relaxed);
  }
};

// Histograms maintain a count of data points accumulated since the last
// read() call, as well as extrema and sum of all data points. The state
// is reset when read.
class Histogram {
public:
  struct Snapshot {
    uint64_t count = 0;
    double sum = 0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::lowest();
  };

  void record(double value) noexcept {
    std::lock_guard<std::mutex> guard(mu);
    ++ss.count;
    ss.sum += value;
    if (value < ss.min)
      ss.min = value;
    if (value > ss.max)
      ss.max = value;
  }

  Snapshot read() noexcept {
    std::lock_guard<std::mutex> guard(mu);
    Snapshot retVal{ss};
    ss = Snapshot{};
    return retVal;
  }

private:
  std::mutex mu;
  Snapshot ss;
};

// This struct is created by dumping the contents of the metric maps and is
// used by the Nanobind interface.
struct CollatedMetrics {
  struct CounterSample {
    llvm::StringRef name;
    uint64_t value;
  };
  struct GaugeSample {
    llvm::StringRef name;
    int64_t value;
  };
  struct HistogramSample {
    llvm::StringRef name;
    Histogram::Snapshot ss;
  };

  std::vector<CounterSample> counters;
  std::vector<GaugeSample> gauges;
  std::vector<HistogramSample> histograms;
};

// Registers metrics by name, returning the new metric to the caller.
// Responds to requests for new data snapshots by returning the
// contents of the maps.
class MetricsCollector {
  std::mutex mu;
  llvm::StringMap<Counter> counters;
  llvm::StringMap<Gauge> gauges;
  llvm::StringMap<Histogram> histograms;

public:
  CollatedMetrics collect();

  Counter &registerCounter(llvm::StringRef name);
  Gauge &registerGauge(llvm::StringRef name);
  Histogram &registerHistogram(llvm::StringRef name);
};

// The MetricsCollector is a singleton, retrieved through this helper.
MODULAR_VISIBILITY_EXPORT inline MetricsCollector &metrics() {
  // We need to be careful that the metrics collector isn't destroyed while
  // others hold references to its pointees. The OS will reclaim the memory.
  [[clang::no_destroy]] static MetricsCollector metrics;
  return metrics;
}

} // namespace M::Metrics

#endif // SUPPORT_METRICS_H
