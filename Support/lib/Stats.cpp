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

#include "Support/Stats.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/ErrorOr.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <system_error>

using namespace M;
using namespace Stats;

static std::mt19937 &getRandomGenerator() {
  static std::random_device dev;
  static std::mt19937 gen(dev());
  return gen;
}

ErrorOr<Samples> Samples::load(StringRef units, StringRef filename,
                               double scale) {
  Samples result;
  result.units = units;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream(filename);
  if (std::error_code ec = errOrBuf.getError())
    return Error("can't open timings file: " + ec.message());

  SmallVector<StringRef> strs;
  (*errOrBuf)->getBuffer().split(strs, "\n", /*MaxSplit=*/-1,
                                 /*KeepEmpty=*/false);
  for (StringRef line : strs) {
    line = line.trim();
    Sample sample;
    if (line.getAsDouble(sample))
      return Error("ill-formed timing entry");
    result.samples.push_back(sample * scale);
  }
  std::sort(result.samples.begin(), result.samples.end());
  return result;
}

ErrorOr<Samples> Samples::ratio(const Samples &lhs, const Samples &rhs,
                                size_t numSamples) {
  Samples result;
  result.units = "%";
  for (size_t i = 0; i < numSamples; ++i) {
    Sample left = lhs.random();
    Sample right = rhs.random();
    if (right == 0.0)
      return Error("divide by zero");
    result.samples.push_back(left * 100.0 / right);
  }
  std::sort(result.samples.begin(), result.samples.end());
  return result;
}

Samples Samples::tDistribution(double degFreedom, size_t numSamples) {
  Samples result;
  result.units = "";
  std::student_t_distribution<double> dist(degFreedom);
  for (size_t i = 0; i < numSamples; ++i)
    result.samples.push_back(dist(getRandomGenerator()));
  std::sort(result.samples.begin(), result.samples.end());
  return result;
}

Sample Samples::atPercentile(double percentile) const {
  assert(!samples.empty());
  percentile = std::max(0.0, percentile);
  percentile = std::min(100.0, percentile);
  size_t index =
      std::lround(percentile / 100.0 * static_cast<double>(samples.size() - 1));
  return samples[index];
}

double Samples::percentileOf(Sample value) const {
  if (samples.empty())
    return NAN;
  if (value <= samples.front())
    return 0.0;
  if (value >= samples.back())
    return 100.0;
  auto itr = std::lower_bound(samples.begin(), samples.end(), value);
  size_t index = std::distance(itr, samples.end());
  return static_cast<double>(index) * 100.0 /
         static_cast<double>(samples.size() - 1);
}

size_t Samples::prune(Sample lowerLimit, Sample upperLimit) {
  auto lowerItr = std::lower_bound(samples.begin(), samples.end(), lowerLimit);
  size_t numPruned = std::distance(samples.begin(), lowerItr);
  samples.erase(samples.begin(), lowerItr);
  auto upperItr = std::upper_bound(samples.begin(), samples.end(), upperLimit);
  numPruned += std::distance(upperItr, samples.end());
  samples.erase(upperItr, samples.end());
  return numPruned;
}

Sample Samples::random() const {
  std::uniform_int_distribution<> distrib(0, numSamples() - 1);
  return samples[distrib(getRandomGenerator())];
}

constexpr const char kSampleFmt[] = "%.3f";
constexpr const char kPercentFmt[] = "%+.2f";
constexpr const char kFixedWidthFmt[] = "%12.3f";

void Samples::printSummary() const {
  llvm::outs() << "  samples:  " << samples.size() << "\n";
  Sample median = atPercentile(50.0);
  auto printWithPercentage = [this, median](StringRef prefix, Sample v) {
    llvm::outs() << prefix << llvm::format(kSampleFmt, v) << units << " ("
                 << llvm::format(kPercentFmt, (v - median) * 100.0 / median)
                 << "%)\n";
  };
  printWithPercentage("  min:      ", atPercentile(0.0));
  printWithPercentage("  1%ile:    ", atPercentile(1.0));
  printWithPercentage("  5%ile:    ", atPercentile(5.0));
  printWithPercentage("  50%ile:   ", atPercentile(50.0));
  printWithPercentage("  95%ile:   ", atPercentile(95.0));
  printWithPercentage("  99%ile:   ", atPercentile(99.0));
  printWithPercentage("  max:      ", atPercentile(100.0));
}

ErrorOr<Normal> Normal::fromSamples(const Samples &samples) {
  if (samples.numSamples() == 0)
    return Error("no samples");
  Normal result;
  result.units = samples.units;
  if (samples.numSamples() == 1) {
    result.mean = samples.samples.front();
    result.stdDev = 0.0;
    return result;
  }
  double total = 0.0;
  for (Sample sample : samples.samples)
    total += sample;
  result.mean = total / static_cast<double>(samples.numSamples());
  double sumSquared = 0.0;
  for (Sample sample : samples.samples)
    sumSquared += std::pow(sample - result.mean, 2.0);
  // Corrected standard deviation
  // https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
  result.stdDev =
      sqrt(sumSquared / static_cast<double>(samples.numSamples() - 1));
  return result;
}

/// Critical value for 99% probability estimated mean is within tolerance
/// of true mean.
constexpr double kCritical = 2.326;

/// Estimate mean is to be within 0.5% of the true mean.
constexpr double kTolerance = 0.005;

void Normal::printSummary() const {
  llvm::outs() << "  mean&sd:  " << llvm::format(kSampleFmt, mean) << units
               << " +/-" << llvm::format(kSampleFmt, stdDev) << units << " ("
               << llvm::format(kPercentFmt, stdDev * 100.0 / mean) << "%)\n";
  // Suggested minimum sample size. Only valid if underlying distribution
  // is Gaussian, hence the simple-minded histogram printing.
  // https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm
  double suggestedMinSamples =
      pow(kCritical * stdDev / (kTolerance * mean), 2.0);
  llvm::outs() << "  min N:    "
               << llvm::format(kSampleFmt, suggestedMinSamples) << "\n";
}

/// Maximum number of buckets in histogram.
constexpr uint64_t kMaxBuckets = 20;

/// Width of histogram in characters.
constexpr size_t kHistogramChars = 50;

ErrorOr<Histogram> Histogram::fromSamples(const Samples &samples,
                                          double rounding) {
  if (samples.samples.empty())
    return Error("no samples");

  Histogram result;
  result.units = samples.units;
  result.rounding = rounding;
  result.n = samples.numSamples();
  result.lower = samples.atPercentile(0.0);
  result.upper = samples.atPercentile(100.0);

  if (result.lower == result.upper) {
    // Degenerate case, one bucket, no variance.
    result.numBuckets = 1;
    result.width = 0.0;
    result.counts.resize(1);
    result.counts[0] = samples.numSamples();
    return result;
  }

  // Choose buckets
  result.lower = result.roundDown(result.lower);
  result.upper = result.roundUp(result.upper);
  result.numBuckets = std::min(kMaxBuckets, result.n);
  result.width = result.roundUp((result.upper - result.lower) /
                                static_cast<double>(result.numBuckets));
  result.numBuckets =
      std::lround(std::ceil((result.upper - result.lower) / result.width));
  result.upper = result.lower + result.width * result.numBuckets;

  // Fill buckets
  result.counts.resize(result.numBuckets, 0);
  size_t lowerIndex = 0;
  for (uint64_t bucket = 0; bucket < result.numBuckets; ++bucket) {
    Sample high = result.lower + (bucket + 1) * result.width;
    size_t upperIndex = lowerIndex;
    while (upperIndex < result.n && samples.samples[upperIndex] < high)
      ++upperIndex;
    result.counts[bucket] = upperIndex - lowerIndex;
    lowerIndex = upperIndex;
  }

  return result;
}

void Histogram::printSummary() {
  if (counts.empty())
    return;

  double maxCount =
      static_cast<double>(*std::max_element(counts.begin(), counts.end()));
  for (size_t bucket = 0; bucket < numBuckets; ++bucket) {
    double bucketLowerBound = lower + bucket * width;
    size_t height = static_cast<size_t>(
        std::lround(std::ceil(static_cast<double>(counts[bucket]) / maxCount *
                              static_cast<double>(kHistogramChars))));
    llvm::outs() << "  " << llvm::format(kFixedWidthFmt, bucketLowerBound)
                 << units << " " << std::string(height, '*') << "\n";
  }
  double upperBound = lower + numBuckets * width;
  llvm::outs() << "  " << llvm::format(kFixedWidthFmt, upperBound) << units
               << "\n";
}

double M::Stats::welchTTest(const Samples &lhsSamples, const Normal &lhsNormal,
                            const Samples &rhsSamples, const Normal &rhsNormal,
                            size_t numSamples) {
  double lhsN = static_cast<double>(lhsSamples.numSamples());
  double rhsN = static_cast<double>(rhsSamples.numSamples());
  double lhsStdErr = lhsNormal.stdDev / std::sqrt(lhsN);
  double rhsStdErr = rhsNormal.stdDev / std::sqrt(rhsN);
  double t = (rhsNormal.mean - lhsNormal.mean) /
             std::sqrt(std::pow(lhsStdErr, 2.0) + std::pow(rhsStdErr, 2.0));
  double degFreedomNum = std::pow(std::pow(lhsNormal.stdDev, 2.0) / lhsN +
                                      std::pow(rhsNormal.stdDev, 2.0) / rhsN,
                                  2.0);
  double degFreedomDen =
      std::pow(lhsNormal.stdDev, 4.0) / (std::pow(lhsN, 2.0) * (lhsN - 1.0)) +
      std::pow(rhsNormal.stdDev, 4.0) / (std::pow(rhsN, 2.0) * (rhsN - 1.0));
  double degFreedom = degFreedomNum / degFreedomDen;
  llvm::outs() << "  t:          " << llvm::format("%.4f", t) << "\n";
  llvm::outs() << "  degFreedom: " << llvm::format("%.2f", degFreedom) << "\n";
  Samples tDistribution = Samples::tDistribution(degFreedom, numSamples);
  return tDistribution.percentileOf(t);
}
