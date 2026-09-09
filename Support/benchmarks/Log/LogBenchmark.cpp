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
// Measures the overhead of filtered MLOG calls in a tight computation loop.
//
// Two variants run the same workload — a chain of trig operations per element:
//   trig_no_log              — no logging at all
//   trig_mlog_debug_filtered — MLOG_DEBUG on every element, level set to WARN
//
// The delta isolates the cost of the level check (one atomic acquire load +
// compare) on the filtered fast-path.
//
// The mid-batch setLogLevel flip introduces a store to the atomic, defeating
// any hoisting of the acquire load out of the loop. Both WARN and ERROR filter
// DEBUG so no output is emitted in either variant.
//
// Run with:
//   ./bazelw run //Support/benchmarks/Log:LogBenchmark
//

#include "Support/Log.h"
#include "Support/MicroBenchmark.h"

#include "llvm/Support/raw_ostream.h"

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

// Trig chain representative of a small numerical kernel: sin -> cos -> atan.
// The result of each feeds the next to prevent the compiler from reordering or
// collapsing independent operations.
static inline double trigWork(double x) {
  return std::atan(std::cos(std::sin(x)));
}

static int runBenchmarks() {
  double x = 0.0;
  MicroBenchmark noLog("trig_no_log", [&x](MicroBenchmark::State &st) {
    setLogLevel(LogLevel::WARN);
    size_t half = st.getBatchSize() / 2;
    size_t i = 0;
    for (auto _ : st) {
      if (++i == half)
        setLogLevel(LogLevel::ERROR);
      double result = trigWork(x);
      x += 0.001;
      MicroBenchmark::doNotOptimizeAway(result);
    }
  });
  ErrorOrSuccess err = noLog.run(makeOpts());
  if (failed(err)) {
    llvm::errs() << "trig_no_log failed: " << err.takeError() << "\n";
    return EXIT_FAILURE;
  }
  printResult(noLog);

  x = 0.0;
  auto benchFunc = [&x](MicroBenchmark::State &st) {
    setLogLevel(LogLevel::WARN);
    size_t half = st.getBatchSize() / 2;
    size_t i = 0;
    for (auto _ : st) {
      if (++i == half)
        setLogLevel(LogLevel::ERROR);
      double result = trigWork(x);
      x += 0.001;
      MLOG_DEBUG("{}", result);
      MicroBenchmark::doNotOptimizeAway(result);
    }
  };

  MicroBenchmark filtered("trig_mlog_debug_filtered", benchFunc);
  err = filtered.run(makeOpts());
  if (failed(err)) {
    llvm::errs() << "trig_mlog_debug_filtered failed: " << err.takeError()
                 << "\n";
    return EXIT_FAILURE;
  }
  printResult(filtered);

  return EXIT_SUCCESS;
}

int main() { return runBenchmarks(); }
