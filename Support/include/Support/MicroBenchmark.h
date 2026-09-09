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
// This file defines the MicroBenchmark infrastructure used throughout the
// Modular codebase.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MICRO_BENCHMARK_H
#define SUPPORT_MICRO_BENCHMARK_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Compiler.h"

#include <chrono>
#include <functional>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

namespace M {

/// This class is used to run a micro-benchmark. It is designed to be used by
/// any C++ code with no dependency on AsyncRT or GraphRT. It is also designed
/// to be flexible and extensible, so that it can apply to many use cases.
///
/// Usage:
///
///  MicroBenchmark bench("name of my benchmark", [](MicroBenchmark::State
///  &state) {
///     for (auto _ : state) {
///       <<< Do some work that needs to be benchmarked >>>
///     }
///  });
///  MicroBenchmark::RunOptions runOptions;
///  // You can optionally specify a prologue function. The prologue function
///  // usually performs prep work that is needed by the main benchmark loop
///  // body.
///  runOptions.prologueFunction =  [](MicroBenchmark::State &state) {
///     <<< Do some unmeasured prep work that needs to be done before the
///     benchmark >>>
///  };
///  // Similarly an epilogue function can be specified. The epilogue function
///  // usually performs cleanup work that is needed after the main benchmark
///  // loop body.
///  runOptions.epilogueFunction =  [](MicroBenchmark::State &state) {
///     <<< Do some unmeasured cleanup work that needs to be done after the
///     benchmark >>>
///  };
///  // You can then run the function with the specified options. The run
///  // function returns an error if the benchmark failed to run.
///  ErrorOrSuccess runResult = bench.run(runOptions);
///  // You can then print the results of the benchmark.
///  bench.report(llvm::outs(), ReportOptions());
///
struct MicroBenchmark {
  /// The State object is passed to the benchmark function. It is used by the
  /// user when writing the benchmark function.
  struct State {
  private:
    struct [[maybe_unused]] IteratorValue {};

  public:
    /// The iterator type used by the benchmark function. The iterator is used
    /// by the user to loop over the batch size in the state object.
    struct Iterator
        : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                            IteratorValue> {
      /// We reach the end of the iterator when the iteration count is 0 or an
      /// error is observed.
      inline bool operator==(Iterator const &) const {
        return LLVM_UNLIKELY(iterationCount == 0 || state->hasError());
      }

      /// Return a dummy value for the iterator.
      inline IteratorValue operator*() const { return IteratorValue(); }

      /// Advancing the iterator means that we decrement the iteration count.
      inline Iterator &operator++() {
        assert(iterationCount > 0);
        --iterationCount;
        return *this;
      }

    private:
      friend struct State;
      size_t iterationCount = 0;
      State *const state = nullptr;

      inline explicit Iterator(State *st)
          : iterationCount(st->hasError() ? 0 : st->getBatchSize()), state(st) {
      }
      inline explicit Iterator() = default;
    };
    inline Iterator begin() { return Iterator(this); }
    inline Iterator end() { return Iterator(); }
    std::chrono::nanoseconds getDuration() const { return duration; }
    size_t getBatchSize() const { return batchSize; }
    void reportError(const Error &err) { error = err.copy(); }
    inline bool hasError() const { return error.isError(); }

  private:
    friend struct MicroBenchmark;
    State(size_t batchSize) : batchSize(batchSize) {}
    size_t batchSize;
    std::chrono::nanoseconds duration = std::chrono::nanoseconds::zero();
    ErrorOrSuccess error;

    ErrorOrSuccess takeError() { return std::move(error); }
  };

  /// The enum class specifies the time unit used for reporting the benchmark.
  enum class TimeUnit { kNanoseconds, kMicroseconds, kMilliseconds, kSeconds };

  /// The enum class specifies the report fields to be printed.
  enum class ReportMetric {
    kName,
    kRaw,
    kTimeUnit,
    kMeanLatency,
    kMedianLatency,
    kWarmupCount,
    kIterationCount,
    kBatchCount
  };

  /// The report format specifies the output format when generating the
  /// benchmark report.
  enum class ReportFormat { kCSV };

  /// Options which control the benchmark execution.
  struct RunOptions {
    /// The maximum number of iterations to perform per time measurement.
    size_t maxBatchSize = 0;
    /// The number of iterations to run as warmup.
    size_t warmupIterations = 0;
    /// The minimum time to run the benchmark for. The benchmark will run for
    /// at least the minRuntime specified.
    std::chrono::milliseconds minRuntime = std::chrono::seconds(2);
    /// The prologue function to be called before each batch iteration of the
    /// benchmarkFunction. If the prologue function returns an error via the
    /// state, then the benchmark loop is terminated. The prologue function is
    /// not accounted for in the timing.
    std::function<void(State &)> prologueFunction{nullptr};
    /// The epilogue function to be called after each batch iteration of the
    /// benchmarkFunction. If the epilogue function returns an error via the
    /// state, then the benchmark loop is terminated.  The epilogue function is
    /// not accounted for in the timing.
    std::function<void(State &)> epilogueFunction{nullptr};
    /// The maximum number of iterations to run for each benchmark. We assume
    /// that the function will reach the minimum runtime of the benchmark with
    /// 1,000,000,000 iterations.
    static constexpr uint64_t kDefaultMaxBenchmarkIterations = 1'000'000'000;
    uint64_t maxBenchmarkIterations = kDefaultMaxBenchmarkIterations;
    /// The maximum runtime of a benchmarking is measured in seconds. The
    /// maximum runtime is set to 1m.
    std::chrono::milliseconds maxRuntime = std::chrono::seconds(60);
    /// When enabled, the first run of the benchmark will print a warning if you
    /// are in debug mode. This is generally useful to always have enabled,
    /// unless in cases where the benchmark is performed by some internal
    /// function within the Modular stack. When enabled, the warning is printed
    /// at most once in the session.
    bool printWarningIfDebugMode = true;
    /// When positive, the benchmark will clear the level of cache specified
    /// prior to each iteration of the function. The cache level must be between
    /// 1 and 4, where 1 specifies the L1 cache, 2 specifies the L2 cache, etc.
    /// If the cache level is -1, then the cache is not cleared. Note that
    /// specifying a clear cache level will need to be performed in conjunction
    /// with setting the maxBatchSize to 1 to be effective. Also note that this
    /// will slow down the benchmarking process.
    int8_t clearCacheLevel = -1;
  };

  /// Options used to control the report output. The user can specify which
  /// metrics to report and what time unit to report them in.
  struct ReportOptions {
    /// The format of the report.
    ReportFormat format = ReportFormat::kCSV;
    /// The time unit to use for the report.
    TimeUnit timeUnit = TimeUnit::kMicroseconds;
    /// The metrics to be reported.
    SmallVector<ReportMetric, 15> metrics{
        ReportMetric::kName,           ReportMetric::kTimeUnit,
        ReportMetric::kWarmupCount,    ReportMetric::kMedianLatency,
        ReportMetric::kIterationCount, ReportMetric::kBatchCount};
  };

  /// Measurements currently include the number of iterations (i.e. batch size)
  /// and the duration for said batch.
  struct Measurement {
    size_t iterations;
    std::chrono::nanoseconds duration;
    bool isSignificant = false;
  };

  /// The clock type used for the benchmark. We ue the high_resolution_clock to
  /// measure the time if the high_resolution_clock is steady. Otherwise, we use
  /// the steady_clock.
  using clock_type =
      std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                         std::chrono::high_resolution_clock,
                         std::chrono::steady_clock>;

  MicroBenchmark(StringRef name, std::function<void(State &)> fn);

  /// Return the name of the benchmark.
  StringRef getName() const;

  /// Run the benchmark. The run function returns an error if the user reported
  /// an error during execution.
  ErrorOrSuccess run(const RunOptions &options);

  /// Report the the results and print them to the output stream.
  void report(raw_ostream &os, const ReportOptions &options);

  /// Get the measurement value. Only values which can be converted to double
  /// are supported.
  double measurement(ReportMetric metric,
                     MicroBenchmark::TimeUnit timeUnit =
                         MicroBenchmark::TimeUnit::kMicroseconds) const;

  /// Tell the compiler to not optimize away the variable. Otherwise, the
  /// compiler might elide the entire function if the resulting variable is
  /// not used. This code is based on the code in the Google Benchmark library
  /// https://github.com/google/benchmark/blob/d2a8a4ee41b923876c034afb939c4fc03598e622/include/benchmark/benchmark.h#L437-L516
#ifdef _MSC_VER
private:
  /// Tell the compiler to not optimize this function on windows. See
  /// https://docs.microsoft.com/en-us/cpp/preprocessor/optimize
#pragma optimize("", off)
  static inline void doNotOptimizeAwayImpl(void const *p) {}
#pragma optimize("", on)
public:
  template <typename T>
  static inline void doNotOptimizeAway(T const &p) {
    doNotOptimizeAwayImpl(&p);
  }
#else // _MSC_VER
  template <typename T>
  static inline void doNotOptimizeAway(T const &p) {
    // The "r" tells the compiler that the variable must reside in a register
    // for the asm block, therefore it must be computed. The "m" tells that
    // the asm block will read the value from memory.
    if constexpr (std::is_trivially_copyable_v<T> || sizeof(long) < sizeof(T) ||
                  std::is_pointer_v<T>)
      asm volatile("" : : "r,m"(p) : "memory");
    else
      asm volatile("" : : "m"(p) : "memory");
  }

  template <class T>
  static inline void doNotOptimizeAway(T &value) {
#ifdef __clang__
    asm volatile("" : "+r,m"(value) : : "memory");
#else  // __clang__
    asm volatile("" : "+m,r"(value) : : "memory");
#endif // __clang__
  }
#endif // _MSC_VER

private:
  std::string name;
  std::function<void(State &)> benchmarkFunction;
  std::vector<Measurement> measurements;
  RunOptions runOptions;

  bool isSignificantMeasurement(const Measurement &measurement, size_t idx);
};
} // namespace M

#endif // SUPPORT_MICRO_BENCHMARK_H
