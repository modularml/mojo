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
// Compare two outputs of mt's --save-timings option.
//
//===----------------------------------------------------------------------===//

#include "Support/Stats.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>

using namespace M;
using namespace M::Stats;

namespace {

struct CompareTimingsCLOptions {
  llvm::cl::opt<std::string> a{
      "a", llvm::cl::desc("Path to timings output for the 'A' side"),
      llvm::cl::init(""), llvm::cl::Required};

  llvm::cl::opt<std::string> b{
      "b", llvm::cl::desc("Path to timings output for the 'B' side"),
      llvm::cl::init(""), llvm::cl::Required};
};

} // namespace

#define BIND(var, expr)                                                        \
  auto var##OrErr = (expr);                                                    \
  if (const auto *err = var##OrErr.getError()) {                               \
    llvm::errs() << err << "\n";                                               \
    return 1;                                                                  \
  }                                                                            \
  auto var = *var##OrErr;

/// Prune samples outside of +/- this factor of the measured mean.
constexpr double kPruneFactor = 0.5;

/// Units and scale for samples.
constexpr const char *kLatencyUnits = "ms";
constexpr double kLatencyScale = 1e-6;

/// In the latency histograms, round bucket boundaries to nearest 0.5ms.
constexpr double kLatencyRounding = 0.5;

/// In the ration histogram, round percentage to nearest 0.5%.
constexpr double kPercentRounding = 0.5;

/// Maximum number of sample when simulating a distribution.
constexpr size_t kMaxSamples = 10000;

/// Limit percentile for Welch t test
constexpr double kWelchLimit = 5.0;

int main(int argc, char **argv) {
  CompareTimingsCLOptions options;

  llvm::cl::ParseCommandLineOptions(argc, argv, "Compare timings tool");

  BIND(aSamples,
       Samples::load(kLatencyUnits, options.a.getValue(), kLatencyScale));
  BIND(aNormal, Normal::fromSamples(aSamples));
  size_t aPruned = aSamples.prune(kPruneFactor * aNormal.mean,
                                  (1.0 + kPruneFactor) * aNormal.mean);
  BIND(aHistogram, Histogram::fromSamples(aSamples, kLatencyRounding));

  BIND(bSamples,
       Samples::load(kLatencyUnits, options.b.getValue(), kLatencyScale));
  BIND(bNormal, Normal::fromSamples(bSamples));
  size_t bPruned = bSamples.prune(kPruneFactor * bNormal.mean,
                                  (1.0 + kPruneFactor) * bNormal.mean);
  BIND(bHistogram, Histogram::fromSamples(bSamples, kLatencyRounding));

  llvm::outs() << "A:\n";
  llvm::outs() << "  pruned:   " << aPruned << "\n";
  aSamples.printSummary();
  aNormal.printSummary();
  llvm::outs() << "  histogram:\n";
  aHistogram.printSummary();
  llvm::outs() << "\n";

  llvm::outs() << "B:\n";
  llvm::outs() << "  pruned:   " << bPruned << "\n";
  bSamples.printSummary();
  bNormal.printSummary();
  llvm::outs() << "  histogram:\n";
  bHistogram.printSummary();
  llvm::outs() << "\n";

  llvm::outs() << "Speedup of B w.r.t. A:\n";
  BIND(ratioSamples, Samples::ratio(aSamples, bSamples, kMaxSamples));
  ratioSamples.printSummary();
  BIND(ratioHistogram, Histogram::fromSamples(ratioSamples, kPercentRounding));
  llvm::outs() << "  histogram:\n";
  ratioHistogram.printSummary();
  llvm::outs() << "\n";

  llvm::outs() << "Welch t-test:\n";
  double welchPercentile =
      welchTTest(aSamples, aNormal, bSamples, bNormal, kMaxSamples);
  llvm::outs() << "  %ile:       " << llvm::format("%.2f", welchPercentile)
               << "%\n";
  if (welchPercentile <= kWelchLimit)
    llvm::outs() << "  <<<B APPEARS FASTER THAN A>>>\n";
  else if (welchPercentile >= (100.0 - kWelchLimit))
    llvm::outs() << "  <<<B APPEARS SLOWER THAN A>>>\n";
  llvm::outs() << "\n";

  return 0;
}
