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
// Measures the amortised cost of infrequent MLOG_INFO calls on the producing
// thread — the most important optimisation target for production workloads
// where logging is rare relative to the surrounding computation.
//
// Each benchmark iteration is one "work unit" (a sin→cos→atan chain). A log
// call is injected every N iterations. Over millions of iterations the mean
// latency converges to:
//
//   T_mean ≈ T_work + T_log / N
//
// so the per-log overhead visible to the calling thread is:
//
//   T_log ≈ (T_mean − T_baseline) × N
//
// This isolates the cost that logging imposes on the producer: the enqueue
// atomic ops (claim, publish, inFlightEnqueues), the arena copy, and any
// cache disturbance to the surrounding work.
//
// A /dev/null sink is used so sink I/O does not artificially slow the
// consumer. Set MODULAR_LOG_FILE before running to measure with a real sink.
//
// Run with:
//   ./bazelw run //Support/benchmarks/Log:LogWorkloadBenchmark

#include "Support/Log.h"
#include "Support/MicroBenchmark.h"

#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>
#include <cstdlib>

using namespace M;
using namespace M::Log;

static MicroBenchmark::RunOptions makeOpts() {
  MicroBenchmark::RunOptions opts;
  opts.warmupIterations = 5;
  opts.minRuntime = std::chrono::milliseconds(500);
  opts.maxRuntime = std::chrono::seconds(10);
  opts.printWarningIfDebugMode = false;
  return opts;
}

static void printResult(MicroBenchmark &bench) {
  MicroBenchmark::ReportOptions reportOpts;
  reportOpts.timeUnit = MicroBenchmark::TimeUnit::kNanoseconds;
  reportOpts.metrics = {
      MicroBenchmark::ReportMetric::kName,
      MicroBenchmark::ReportMetric::kTimeUnit,
      MicroBenchmark::ReportMetric::kMeanLatency,
      MicroBenchmark::ReportMetric::kMedianLatency,
      MicroBenchmark::ReportMetric::kIterationCount,
  };
  bench.report(llvm::outs(), reportOpts);
  llvm::outs().flush();
}

static inline double trigWork(double x) {
  return std::atan(std::cos(std::sin(x)));
}

// Run the workload with a log call every logEvery iterations.
// logEvery == 0 means never log (the baseline).
static bool runVariant(const char *name, size_t logEvery) {
  double x = 0.0;
  size_t counter = 0;
  MicroBenchmark bench(name, [&](MicroBenchmark::State &st) {
    for (auto _ : st) {
      double result = trigWork(x);
      x += 0.001;
      if (logEvery && ++counter % logEvery == 0)
        MLOG_INFO("result={}", result);
      MicroBenchmark::doNotOptimizeAway(result);
    }
  });
  ErrorOrSuccess err = bench.run(makeOpts());
  if (failed(err)) {
    llvm::errs() << name << " failed: " << err.takeError() << "\n";
    return false;
  }
  printResult(bench);
  return true;
}

static int runBenchmarks() {
  if (!getenv("MODULAR_LOG_FILE"))
    ::setenv("MODULAR_LOG_FILE", "/dev/null", 1);
  ::setenv("MODULAR_LOG_STDOUT", "false", 1);
  ::setenv("MODULAR_LOG_NO_SUMMARY", "1", 1);
  // Force log initialisation now while env vars are set.
  getDefaultLog().setLogLevel(LogLevel::INFO);

  struct Variant {
    const char *name;
    size_t logEvery;
  };
  static constexpr std::array variants{
      Variant{"no_log_baseline", 0},   Variant{"log_every_1", 1},
      Variant{"log_every_10", 10},     Variant{"log_every_100", 100},
      Variant{"log_every_1000", 1000},
  };

  for (auto [name, logEvery] : variants) {
    if (!runVariant(name, logEvery))
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main() { return runBenchmarks(); }
