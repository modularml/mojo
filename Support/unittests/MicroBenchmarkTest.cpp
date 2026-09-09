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

#include "Support/MicroBenchmark.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <chrono>

using namespace M;
using namespace std::chrono_literals;
using ::testing::HasSubstr;

TEST(MicroBenchmarkTest, BenchmarkAllocation) {
  MicroBenchmark::RunOptions runOpts;
  runOpts.printWarningIfDebugMode = false;
  runOpts.maxRuntime = 10ms; // Set to 10ms to avoid long running tests.

  MicroBenchmark bench("vector allocation", [&](MicroBenchmark::State &st) {
    for (auto _ : st) {
      // Allocate a 1M bytes. This is a slightly expensive operation.
      std::vector<std::byte> vec;
      vec.reserve(1'000'000);
      // Tell the compiler to not optimize the unused variable away.
      MicroBenchmark::doNotOptimizeAway(vec);
    }
  });

  ErrorOrSuccess err = bench.run(runOpts);
  EXPECT_FALSE(failed(err)) << err.takeError();

  double meanLatency =
      bench.measurement(MicroBenchmark::ReportMetric::kMeanLatency,
                        /*timeUnit=*/MicroBenchmark::TimeUnit::kNanoseconds);
  EXPECT_GT(meanLatency, 0) << "the mean latency must be positive";

  // Generate the benchmark report.
  MicroBenchmark::ReportOptions reportOpts;
  std::string reportStr;
  llvm::raw_string_ostream os(reportStr);
  bench.report(os, reportOpts);

  // We should expect that the report contains the name of the benchmark.
  EXPECT_THAT(reportStr, HasSubstr("\"vector allocation\""));

  // We should not expect the report to contain some other benchmark name.
  EXPECT_THAT(reportStr, Not(HasSubstr("string creation")));
}
