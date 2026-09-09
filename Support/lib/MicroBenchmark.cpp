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
#include "Support/CPUCache.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MathExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <ratio>
#include <sstream>
#include <utility>
#include <vector>

using namespace M;

static void clearCache(uint8_t level);

MicroBenchmark::MicroBenchmark(StringRef name, std::function<void(State &)> fn)
    : name(name), benchmarkFunction(std::move(fn)) {}

/// Get the name of the benchmark. The name is a description of what is being
/// benchmarked.
StringRef MicroBenchmark::getName() const { return name; }

/// The main benchmark loop which calls the function to be benchmarked and
/// stores the results in the measurements field.
ErrorOrSuccess MicroBenchmark::run(const RunOptions &options) {
#ifdef MODULAR_DEBUG
  if (options.printWarningIfDebugMode) {
    // Show a warning when benchmarking in debug mode. We make sure that we only
    // show the warning a single time per run (this reduces noise).
    static std::once_flag showDebugWarningOnceFlag;
    std::call_once(showDebugWarningOnceFlag, [&]() {
      llvm::errs()
          << "WARNING: Benchmarking in debug mode is not recommended due to "
             "increased overhead. Please use a release build.\n";
      llvm::errs().flush();
    });
  }
#endif // MODULAR_DEBUG
  assert(measurements.empty() &&
         "Measurements should be empty before running. This usually means that "
         "you have invoked the run function twice for the same MicroBenchmark "
         "object.");
  runOptions = options;

  size_t totalIterations = 0;
  uint64_t batchSize =
      options.maxBatchSize ? options.maxBatchSize : options.warmupIterations;
  bool isWarmupPhase = batchSize > 0;
  std::chrono::nanoseconds totalTime(0);
  std::chrono::milliseconds minRuntime = options.minRuntime;

  // Run the benchmark until the time elapsed is greater than the minimum time,
  // the maximum number of iterations is reached, or the total runtime exceeds
  // the maximum runtime.
  while (true) {
    if (totalIterations >= options.maxBenchmarkIterations ||
        totalTime >= std::chrono::duration_cast<std::chrono::nanoseconds>(
                         options.maxRuntime) ||
        totalTime >= minRuntime)
      break;

    std::chrono::nanoseconds duration(0);
    std::chrono::nanoseconds batchDuration(0);

    // Run the batched loop. A zero value for batchSize can occur when the
    // user sets the warmupCount to 0.
    if (batchSize > 0) {
      // Create the state for the current benchmark run.
      State st(batchSize);

      // Run the prologue function if it exists.
      if (options.prologueFunction) {
        std::invoke(options.prologueFunction, st);
        // The user reported an error, so terminate the benchmark run.
        if (st.hasError())
          return st.takeError();
      }

      // Start the timers and execute the body.
      auto tic = clock_type::now();

      // Run the benchmark function.
      std::invoke(benchmarkFunction, st);

      // Stop the timers.
      auto toc = clock_type::now();

      // Compute the duration of the batch along with the time it takes to
      // perform a single run of the function to measure.
      batchDuration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
      duration = batchDuration / batchSize;

      // Store the current duration in the state. This allows the user-passed in
      // epilogue function to query the duration.
      st.duration = duration;

      // Run the epilogue function if it exists.
      if (options.epilogueFunction)
        std::invoke(options.epilogueFunction, st);

      // Check if the benchmark function returned an error. If so, return the
      // error.
      if (st.hasError())
        return st.takeError();

      if (isWarmupPhase) {
        // We only run the warmup phase once, so we toggle the flag so that
        // subsequent iterations collect the time information.
        isWarmupPhase = false;
      } else {
        // When we are not in a warmup phase, we need to record the
        // measurements.
        measurements.push_back({batchSize, batchDuration});

        // We also need to keep track of the total runtime and number of
        // iterations.
        totalTime +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(batchDuration);
        totalIterations += batchSize;
      }
    }

    // Clear the cache if requested.
    if (options.clearCacheLevel > 0)
      clearCache(options.clearCacheLevel);

    // If the maxBatchSize is specified, then prefer that over the subsequent
    // computations.
    if (options.maxBatchSize) {
      batchSize = options.maxBatchSize;
      continue;
    }

    // We now count the next batchSize. A user might run the benchmark with no
    // warmup phase, so we need to make sure the divisor is not zero.
    if (batchDuration.count() == 0)
      batchDuration = std::chrono::nanoseconds(1);
    // Compute the next batch size.
    double nextBatchSize = (minRuntime * (double)batchSize) / batchDuration;
    // We increase the iteration count by 1.2x from the previous loop iteration.
    nextBatchSize *= 1.2;
    // We should not grow too fast, so we cap it to only 10x the growth from
    // the prior iteration. Fast growth can happen when the function is too
    // fast.
    nextBatchSize = std::min(nextBatchSize, 10.0 * batchSize);
    // We have to increase the batchSize each time. So, we make sure we advance
    // the number of iterations regardless of the prior logic.
    nextBatchSize = std::max(nextBatchSize, batchSize + 1.0);
    // The batch size should not be larger than 1.0e9.
    nextBatchSize = std::min(nextBatchSize, 1.0e9);

    // Update the batch size based on the above compute logic.
    batchSize = std::lround(nextBatchSize);
  }

  // Revisit all the measurement entries and determine if they are statistically
  // significant.
  size_t idx = 0;
  for (Measurement &measurement : measurements)
    measurement.isSignificant = isSignificantMeasurement(measurement, idx++);

  return success();
}

bool MicroBenchmark::isSignificantMeasurement(const Measurement &measurement,
                                              size_t idx) {
  // The measurement number of iteration is the same as the requested
  // maxBatchSize and the measurement duration exceeded the requested min
  // runtime.
  if (runOptions.maxBatchSize &&
      measurement.iterations >= runOptions.maxBatchSize &&
      measurement.duration >= runOptions.minRuntime)
    return true;

  // This measurement occurred in the last 10% of the run.
  if ((idx + 1) >= 0.9 * measurements.size())
    return true;

  // Otherwise the result is not significant.
  return false;
}

/// Clear the cache if requested.
static void clearCache(uint8_t level) {
  // There is nothing to do if the level is 0 or -1.
  if (level <= 0)
    return;

  ErrorOr<size_t> cacheSize = getHostCPUCacheSize(level);

  // If we cannot get the cache size, then we cannot clear the cache.
  if (cacheSize.isError())
    return;

  // If the cacheSize is zero, then we cannot clear the cache at that level, so
  // we try to clear the cache at lower-levels.
  if (*cacheSize == 0)
    return clearCache(level - 1);

  // Otherwise, we allocate a buffer of the cache size and clear it. We also
  // mark the buffer so that the compiler does not optimize it away.
  std::vector<char> buffer(*cacheSize, 0);
  MicroBenchmark::doNotOptimizeAway(buffer);
}

/// Formats the time based on the time unit specified.
static double formatTime(MicroBenchmark::TimeUnit timeUnit,
                         std::chrono::nanoseconds time) {
  switch (timeUnit) {
  case MicroBenchmark::TimeUnit::kNanoseconds:
    return std::chrono::duration<double, std::nano>(time).count();
  case MicroBenchmark::TimeUnit::kMicroseconds:
    return std::chrono::duration<double, std::micro>(time).count();
  case MicroBenchmark::TimeUnit::kMilliseconds:
    return std::chrono::duration<double, std::milli>(time).count();
  case MicroBenchmark::TimeUnit::kSeconds:
    return std::chrono::duration<double>(time).count();
  }
  llvm_unreachable("Invalid time unit");
  return -1;
}

/// Gets the time unit name as a string.
static StringRef toString(MicroBenchmark::TimeUnit timeUnit) {
  switch (timeUnit) {
  case MicroBenchmark::TimeUnit::kNanoseconds:
    return "ns";
  case MicroBenchmark::TimeUnit::kMicroseconds:
    return "us";
  case MicroBenchmark::TimeUnit::kMilliseconds:
    return "ms";
  case MicroBenchmark::TimeUnit::kSeconds:
    return "s";
  }
  llvm_unreachable("Invalid time unit");
  return "<unknown time unit>";
}

/// Gets the report metric name as a string.
static StringRef toString(MicroBenchmark::ReportMetric metric) {
  switch (metric) {
  case MicroBenchmark::ReportMetric::kName:
    return "name";
  case MicroBenchmark::ReportMetric::kTimeUnit:
    return "time_unit";
  case MicroBenchmark::ReportMetric::kRaw:
    return "raw";
  case MicroBenchmark::ReportMetric::kMeanLatency:
    return "mean_latency";
  case MicroBenchmark::ReportMetric::kMedianLatency:
    return "median_latency";
  case MicroBenchmark::ReportMetric::kWarmupCount:
    return "warmup_count";
  case MicroBenchmark::ReportMetric::kIterationCount:
    return "iteration_count";
  case MicroBenchmark::ReportMetric::kBatchCount:
    return "batch_count";
  }
  return "<unknown report metric>";
}

/// Print the report header in CSV format.
static void printCSVHeader(raw_ostream &os,
                           ArrayRef<MicroBenchmark::ReportMetric> metrics) {
  static std::once_flag showHeaderOnceFlag;
  std::call_once(showHeaderOnceFlag, [&]() {
    // Note: We do not use llvm::interleaveComma here because we do not want
    // spaces between the comma values.
    llvm::interleave(
        metrics, os,
        [&](MicroBenchmark::ReportMetric metric) { os << toString(metric); },
        ",");
    os << "\n";
    os.flush();
  });
}

/// Gets the timing information from the measurements.
static SmallVector<std::chrono::nanoseconds>
getTimings(ArrayRef<MicroBenchmark::Measurement> measurements) {
  return llvm::map_to_vector(
      llvm::make_filter_range(
          measurements,
          [](auto &measurement) { return measurement.isSignificant; }),
      [](auto &measurement) -> std::chrono::nanoseconds {
        return measurement.duration / measurement.iterations;
      });
}

/// Computes the mean latency and returns the value as a double in the
/// specified time unit.
static double getMeanLatency(ArrayRef<MicroBenchmark::Measurement> measurements,
                             MicroBenchmark::TimeUnit timeUnit) {
  return formatTime(timeUnit, mean(getTimings(measurements)));
}

/// Computes the median latency and returns the value as a double in the
/// specified time unit.
static double
getMedianLatency(ArrayRef<MicroBenchmark::Measurement> measurements,
                 MicroBenchmark::TimeUnit timeUnit) {
  auto timings = getTimings(measurements);
  llvm::sort(timings);
  return formatTime(timeUnit, median(timings));
}

/// Gets the measurements for the given metric as a double value in the
/// specified time unit.
double MicroBenchmark::measurement(MicroBenchmark::ReportMetric metric,
                                   MicroBenchmark::TimeUnit timeUnit) const {
  assert(!measurements.empty() && "no measurements to report");
  switch (metric) {
  case ReportMetric::kName:
  case ReportMetric::kRaw:
  case ReportMetric::kTimeUnit:
    llvm_unreachable("invalid report metric. Only metrics which have a value "
                     "coercible to a double are supported.");
    return 0;
  case ReportMetric::kMeanLatency:
    return getMeanLatency(measurements, timeUnit);
  case ReportMetric::kMedianLatency:
    return getMedianLatency(measurements, timeUnit);
  case ReportMetric::kWarmupCount:
    return runOptions.warmupIterations;
  case ReportMetric::kIterationCount:
    return std::accumulate(measurements.begin(), measurements.end(), 0,
                           [](size_t acc, auto &measurement) {
                             return acc + measurement.iterations;
                           });
  case ReportMetric::kBatchCount:
    return measurements.size();
  }
  return 0;
}

/// Prints the benchmark results to the given output stream.
void MicroBenchmark::report(raw_ostream &os, const ReportOptions &options) {
  assert(options.format == ReportFormat::kCSV &&
         "only CSV format is supported");
  assert(!measurements.empty() && "no measurements to report");
  printCSVHeader(os, options.metrics);
  // Note: We do not use llvm::interleaveComma here because we do not want
  // spaces between the comma values.
  llvm::interleave(
      options.metrics, os,
      [&](auto metric) {
        switch (metric) {
        case ReportMetric::kName: {
          std::stringstream str;
          str << std::quoted(getName().str());
          os << str.str();
          return;
        }
        case ReportMetric::kRaw: {
          // We print the raw measurements as semi-colon separated values.
          llvm::interleave(
              getTimings(measurements), os,
              [&](auto &measurement) {
                os << formatTime(options.timeUnit, measurement);
              },
              ";");
          return;
        }
        case ReportMetric::kTimeUnit: {
          os << toString(options.timeUnit);
          return;
        }
        default:
          os << measurement(metric, options.timeUnit);
        }
      },
      ",");
  os << "\n";
  os.flush();
}
