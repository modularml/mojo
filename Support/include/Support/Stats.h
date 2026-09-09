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
// Simple-minded summary statistics to help check the mlperf benchmark
// settings are sensible.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STATS_H
#define SUPPORT_STATS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace M::Stats {

using Sample = double;

/// A collection of raw samples.
struct Samples {
  /// Units for each sample.
  std::string units;
  /// Raw samples in ascending order.
  std::vector<Sample> samples;

  size_t numSamples() const { return samples.size(); }

  /// Returns sample at given percentile.
  Sample atPercentile(double percentile) const;

  /// Returns percentile of given value.
  double percentileOf(Sample value) const;

  /// Returns samples loaded from filename and scaled by given scale.
  static ErrorOr<Samples> load(StringRef units, StringRef filename,
                               double scale);

  /// Returns samples of lhs / rhs as percentage.
  static ErrorOr<Samples> ratio(const Samples &lhs, const Samples &rhs,
                                size_t numSamples);

  /// Returns samples from the t distribution with given degrees of freedom.
  static Samples tDistribution(double degFreedom, size_t numSamples);

  /// Removes samples which are outside of the range (lowerLimit, upperLimit).
  /// Returns number of samples removed.
  size_t prune(Sample lowerLimit, Sample upperLimit);

  /// Returns a random sample.
  Sample random() const;

  /// Prints a description of samples.
  void printSummary() const;
};

/// A normal distribution.
struct Normal {
  std::string units;
  Sample mean;
  double stdDev;

  /// Returns MLE fit for samples.
  static ErrorOr<Normal> fromSamples(const Samples &samples);

  /// Prints a description of normal distribution.
  void printSummary() const;
};

/// Summarize raw timing samples as a histogram with key statistics.
struct Histogram {
  std::string units;
  double rounding;

  // Number of samples.
  uint64_t n;
  // min, max samples.
  Sample lower;
  Sample upper;

  // Histogram buckets.
  uint64_t numBuckets;
  Sample width;
  std::vector<uint64_t> counts;

  /// Constructs histogram.
  static ErrorOr<Histogram> fromSamples(const Samples &samples,
                                        double rounding);

  /// Prints summary of histogram to llvm::outs().
  void printSummary();

private:
  Sample roundDown(Sample v) { return std::floor(v / rounding) * rounding; }
  Sample roundUp(Sample v) { return std::ceil(v / rounding) * rounding; }
};

// Returns the percentile of the t value representing the difference
// lhsNormal.mean - rhsNormal.mean, without assuming lhsNormal.stdDev ==
// rhsNormal.stdDev. A result of >=95% indicates the lhs mean is statistically
// significantly greater than the rhs mean. A result of <=5% indicates the
// converse. All other results indicate the means are not statistically
// significantly different.
// See https://en.wikipedia.org/wiki/Welch%27s_t-test
double welchTTest(const Samples &lhsSamples, const Normal &lhsNormal,
                  const Samples &rhsSamples, const Normal &rhsNormal,
                  size_t numSamples);

} // namespace M::Stats

#endif // SUPPORT_STATS_H
