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

#ifndef SUPPORT_TELEMETRY_H
#define SUPPORT_TELEMETRY_H

#include "Support/Configuration.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Common.h"
#include "Support/Telemetry/Instruments.h"
#include "Support/Telemetry/Logs.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "opentelemetry/logs/event_logger_provider.h"
#include "opentelemetry/logs/logger_provider.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/meter_provider.h"
#include "opentelemetry/sdk/common/exporter_utils.h"
#include "opentelemetry/sdk/logs/exporter.h"
#include "opentelemetry/sdk/metrics/push_metric_exporter.h"

namespace M::Telemetry {

// TODO: Support some of these in config file.
/// When the TelemetryContext is destroyed, it does a synchronous flush to
/// ensure that any telemetry that hasn't yet been exported is exported. This
/// timeout is how long it waits for the export to complete before the
/// destructor returns.
constexpr auto kShutdownFlushTimeout = std::chrono::milliseconds(500);
/// Periodically export metrics every kExportInterval duration.
constexpr auto kExportInterval = std::chrono::seconds(600);
/// Timeout for periodic metric exports. Note that periodic exports happen
/// asynchronously and this timeout is for the worker thread that does them
/// (OTel-managed thread). NOTE: this value must be smaller than the export
/// interval.
constexpr auto kExportTimeout = std::chrono::milliseconds(1000);

/// Timeout for OTLP HTTP export requests. Without this, when the telemetry
/// endpoint is unreachable (e.g. firewall silently drops packets), TCP SYN
/// retries can block for ~20s per attempt. This sets CURLOPT_TIMEOUT_MS on the
/// underlying libcurl handle, providing a hard deadline on the entire HTTP
/// request (including connection establishment).
constexpr auto kOtlpRequestTimeout = std::chrono::milliseconds(3000);

/// Emit a one-time warning to stderr when a telemetry export fails. The warned
/// flag is injectable so that tests can create isolated instances. Production
/// code shares a single flag across all exporters in TelemetryContext.
inline void warnOnExportFailure(std::shared_ptr<std::atomic<bool>> warned,
                                const std::string &endpoint) {
  if (warned && !warned->exchange(true)) {
    llvm::errs() << "Warning: telemetry export to "
                 << (endpoint.empty() ? "<unknown>" : endpoint)
                 << " failed (endpoint may be unreachable). "
                    "Set MODULAR_TELEMETRY_ENABLED=0 to disable telemetry "
                    "if you are in a restricted network environment.\n";
  }
}

/// Metric exporter wrapper that emits a one-time warning on export failure.
/// Delegates all calls to the underlying exporter.
class WarningMetricExporter
    : public opentelemetry::sdk::metrics::PushMetricExporter {
public:
  WarningMetricExporter(
      std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter> delegate,
      std::string endpoint, std::shared_ptr<std::atomic<bool>> warned)
      : delegate_(std::move(delegate)), endpoint_(std::move(endpoint)),
        warned_(std::move(warned)) {}

  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::sdk::metrics::ResourceMetrics &data) noexcept
      override {
    auto result = delegate_->Export(data);
    if (result != opentelemetry::sdk::common::ExportResult::kSuccess)
      warnOnExportFailure(warned_, endpoint_);
    return result;
  }

  opentelemetry::sdk::metrics::AggregationTemporality GetAggregationTemporality(
      opentelemetry::sdk::metrics::InstrumentType instrumentType)
      const noexcept override {
    return delegate_->GetAggregationTemporality(instrumentType);
  }

  bool ForceFlush(std::chrono::microseconds timeout) noexcept override {
    return delegate_->ForceFlush(timeout);
  }

  bool Shutdown(std::chrono::microseconds timeout) noexcept override {
    return delegate_->Shutdown(timeout);
  }

private:
  std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter> delegate_;
  std::string endpoint_;
  std::shared_ptr<std::atomic<bool>> warned_;
};

/// Log record exporter wrapper that emits a one-time warning on export failure.
/// Delegates all calls to the underlying exporter.
class WarningLogRecordExporter
    : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  WarningLogRecordExporter(
      std::unique_ptr<opentelemetry::sdk::logs::LogRecordExporter> delegate,
      std::string endpoint, std::shared_ptr<std::atomic<bool>> warned)
      : delegate_(std::move(delegate)), endpoint_(std::move(endpoint)),
        warned_(std::move(warned)) {}

  std::unique_ptr<opentelemetry::sdk::logs::Recordable>
  MakeRecordable() noexcept override {
    return delegate_->MakeRecordable();
  }

  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::nostd::span<
         std::unique_ptr<opentelemetry::sdk::logs::Recordable>>
             &records) noexcept override {
    auto result = delegate_->Export(records);
    if (result != opentelemetry::sdk::common::ExportResult::kSuccess)
      warnOnExportFailure(warned_, endpoint_);
    return result;
  }

  bool ForceFlush(std::chrono::microseconds timeout) noexcept override {
    return delegate_->ForceFlush(timeout);
  }

  bool Shutdown(std::chrono::microseconds timeout) noexcept override {
    return delegate_->Shutdown(timeout);
  }

private:
  std::unique_ptr<opentelemetry::sdk::logs::LogRecordExporter> delegate_;
  std::string endpoint_;
  std::shared_ptr<std::atomic<bool>> warned_;
};

/// Fire-and-forget log record exporter wrapper: dispatches the delegate's
/// Export call on a detached thread so emit never blocks on the delegate's
/// HTTP/network I/O, at the cost of losing visibility into whether the
/// export eventually succeeded. Intended for low-volume diagnostic events
/// (e.g. `program.initialized`) where emit-time
/// latency is the primary concern and per-record delivery guarantees are
/// not; do NOT route high-volume or delivery-critical telemetry through
/// this wrapper — pair the synchronous delegate with a
/// `BatchLogRecordProcessor` instead.
///
/// Background: OTel's curl HTTP exporter uses the synchronous system name
/// resolver, so its nominal 3s `CURLOPT_TIMEOUT_MS` does not cap DNS
/// resolution or TCP SYN retries. In environments where the OTel endpoint
/// is unreachable, every synchronous `Export` can stall for tens of
/// seconds, blocking `emitL0Event` on the emit path. See SDLC-3618.
///
/// `emit` stays non-blocking — `Export` only spawns the worker and returns —
/// but the wrapper now *tracks* in-flight workers so it can drain them at
/// well-defined points. This matters for correctness, not just delivery:
/// each record carries a raw pointer to the LoggerProvider's `Resource`, so a
/// worker that outlives the provider would read freed Resource strings and
/// crash (observed as `std::length_error` in `basic_string::_M_create`).
/// Draining in `ForceFlush` (and the destructor backstop) — driven from
/// `TelemetryContext::flush()` / `~TelemetryContext()` while the Resource is
/// still alive — closes that use-after-free window.
///
/// Semantic notes on the base `LogRecordExporter` contract:
///
///   - `Export` always returns `kSuccess` and never blocks; the caller has no
///     signal about whether the off-thread export eventually succeeded.
///   - `ForceFlush` waits up to the supplied `timeout` for in-flight workers to
///     finish and returns whether the drain completed. `flush()` is always
///     called before teardown, so this is the primary UAF guard.
///   - `Shutdown` stays non-blocking (returns immediately) by design: awaiting
///     it could stall process exit on the delegate's unreachable-endpoint curl
///     retries (SDLC-3618). It does not forward to the delegate.
///   - If a drain times out (e.g. endpoint unreachable at process exit) the
///     worker is left running. It can never touch freed *wrapper* memory — it
///     only races the `SharedState`, kept alive by its own `shared_ptr`. The
///     records it still holds, however, point at the provider's `Resource`,
///     which teardown can then free out from under it. So a drain timeout
///     *narrows* the use-after-free window to that case rather than
///     eliminating it. Fully closing it would require making each record
///     self-contained (deep-copying the Resource) before dispatch.
class FireAndForgetLogRecordExporter
    : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  explicit FireAndForgetLogRecordExporter(
      std::unique_ptr<opentelemetry::sdk::logs::LogRecordExporter> delegate)
      : state_(std::make_shared<SharedState>()) {
    state_->delegate = std::move(delegate);
  }

  // Backstop: drain any export still in flight before this wrapper goes away.
  // In the normal teardown path TelemetryContext::~TelemetryContext() already
  // calls flush() (-> ForceFlush) then Shutdown() before the LoggerProvider —
  // and the Resource the records point into — is destroyed, so by here the
  // count is usually already zero and this returns immediately. Reuses the
  // budget from the most recent ForceFlush so teardown doesn't pile an
  // independent timeout on top of the flush() that just ran.
  ~FireAndForgetLogRecordExporter() override {
    std::chrono::microseconds budget;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      budget = state_->drainBudget;
    }
    drain(budget);
  }

  std::unique_ptr<opentelemetry::sdk::logs::Recordable>
  MakeRecordable() noexcept override {
    return state_->delegate->MakeRecordable();
  }

  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::nostd::span<
         std::unique_ptr<opentelemetry::sdk::logs::Recordable>>
             &records) noexcept override {
    // Take ownership of the records so the worker thread can own them.
    // Our only caller, `SimpleLogRecordProcessor::OnEmit`, does not touch
    // the span after `Export` returns — it destroys the local `unique_ptr`
    // at scope exit — so moving out is safe. Do NOT use this wrapper
    // behind a processor that reuses records post-Export.
    std::vector<std::unique_ptr<opentelemetry::sdk::logs::Recordable>> owned;
    owned.reserve(records.size());
    for (auto &record : records)
      owned.push_back(std::move(record));

    // Mark one export in flight, then run the (potentially slow / blocking)
    // delegate export off-thread so `emit` never blocks on network I/O.
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      ++state_->inFlight;
    }
    // The worker co-owns `state_` via the shared_ptr capture, so the mutex /
    // condition variable / delegate it touches stay alive even if this
    // wrapper is destroyed first — the worker only ever races the SharedState,
    // never freed exporter memory.
    std::thread([state = state_, owned = std::move(owned)]() mutable {
      opentelemetry::nostd::span<
          std::unique_ptr<opentelemetry::sdk::logs::Recordable>>
          span{owned.data(), owned.size()};
      (void)state->delegate->Export(span);
      std::lock_guard<std::mutex> lock(state->mutex);
      if (--state->inFlight == 0)
        state->done.notify_all();
    }).detach();
    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }

  // ForceFlush now honestly waits (up to `timeout`) for in-flight exports to
  // finish. This is what keeps a detached export from reading the
  // LoggerProvider's Resource after the provider is torn down:
  // TelemetryContext::flush() -> ForceFlush runs from both the explicit flush
  // path and ~TelemetryContext while the Resource is still alive. Returns false
  // if the drain timed out with work still pending.
  bool ForceFlush(std::chrono::microseconds timeout) noexcept override {
    {
      // Remember the budget so the destructor backstop reuses it rather than
      // adding an independent one.
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->drainBudget = timeout;
    }
    return drain(timeout);
  }

  // Shutdown stays non-blocking by design (SDLC-3618): forwarding/awaiting here
  // could stall process exit on the delegate's unreachable-endpoint curl
  // retries. The UAF window is already closed by ForceFlush (always called
  // first in teardown) and the destructor backstop, so Shutdown need not wait.
  bool Shutdown(std::chrono::microseconds /*timeout*/) noexcept override {
    return true;
  }

private:
  // Shared between this wrapper and every in-flight worker so the
  // synchronization primitives and delegate outlive the wrapper if needed.
  struct SharedState {
    std::shared_ptr<opentelemetry::sdk::logs::LogRecordExporter> delegate;
    std::mutex mutex;
    std::condition_variable done;
    std::size_t inFlight = 0;
    // Budget the destructor backstop drains with, set to the most recent
    // ForceFlush timeout. The default is the fallback for standalone use that
    // never routes through TelemetryContext::flush() (e.g. unit tests); guarded
    // by `mutex`.
    std::chrono::microseconds drainBudget =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::seconds(5));
  };

  bool drain(std::chrono::microseconds timeout) noexcept {
    // Cap the wait so a sentinel like `microseconds::max()` can't overflow the
    // steady_clock time_point inside wait_for (which would make it return
    // immediately instead of waiting). A day is far past any real flush budget.
    constexpr auto kMaxWait =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::hours(24));
    const auto waitFor = timeout < kMaxWait ? timeout : kMaxWait;
    std::unique_lock<std::mutex> lock(state_->mutex);
    return state_->done.wait_for(lock, waitFor,
                                 [&] { return state_->inFlight == 0; });
  }

  std::shared_ptr<SharedState> state_;
};

// TODO: Add ways to organize instruments (e.g. Meters/instrumentation scope)
// later if needed.

/// Local identifiers reported alongside telemetry events.
struct LocalIDs {
  /// Invariant within a given container.
  std::string machine;
  /// Invariant within a given process.
  std::string session;
};

/// createLocalIDs creates the local machineid/sessionid pair.
///
/// The result is computed once and memoized (it may be quite expensive), so
/// every caller in the process observes identical values. The crash reporting
/// and usage telemetry lanes rely on this to carry the same IDs.
const LocalIDs &createLocalIDs();

/// Reports whether the usage telemetry lane is enabled,
/// Users opt out, so this defaults to enabled for production builds.
bool isTelemetryEnabled(Config &settings);

/// Reports whether the crash reporting lane is enabled,
/// Users opt out, so this defaults to isTelemetryEnabled.
bool isCrashReportingEnabled(Config &settings);

/// A TelemetryContext provides access to instruments (e.g. Counter, Histogram)
/// to instrument the code and generate metrics. These metrics will be exported
/// by the TelemetryContext based on the options passed to it during creation.
///
/// Right now we are assuming that the TelemetryContext will collect Resource
/// attributes (e.g. CPU info, OS info, version of software components) without
/// this information being passed to it explicitly through its API, but this is
/// subject to change.
class TelemetryContext {
public:
  /// This is just a copy of the OTel MetricAttributeValue - we can use this to
  /// provide resources to the telemetry context. We don't support the lists
  /// yet, we can add those as necessary.
  using AttributeValue =
      std::variant<bool, int32_t, int64_t, uint32_t, double, StringRef,
                   ArrayRef<bool>, ArrayRef<int32_t>, ArrayRef<int64_t>,
                   ArrayRef<uint32_t>, ArrayRef<double>, uint64_t,
                   ArrayRef<uint64_t>, ArrayRef<uint8_t>>;

  /// Set up a TelemetryContext from a Config object. If programName is
  /// non-empty, it is recorded as the "program.name" resource attribute. When
  /// crash reporting is also enabled (its default follows the
  /// telemetry.enabled setting), a "program.initialized" event is emitted at
  /// L0 once per process.
  /// If subCommand is non-empty (e.g. "run", "build", "debug"), it is
  /// included as the "program.sub_command" attribute on that event.
  TelemetryContext(Config &settings, StringRef programName = "",
                   StringRef subCommand = "");

  TelemetryContext(TelemetryContext &&other) = default;

  virtual ~TelemetryContext();

  // XXX: not sure if it's better to allocate Counter and Histogram on the heap
  // or not. For OTel, the Counter struct will basically just contain a pointer
  // to the OTel counter, and so returning the struct seems appropriate.

  /// Returns true if an instrument will be enabled based on its level and the
  /// configured telemetry level.
  bool isInstrumentEnabled(Level instrumentLevel) const {
    return instrumentLevel <= telemetryLevel;
  }

  bool isUserMetric(Level instrumentLevel) const {
    return instrumentLevel == Level::USER;
  }

  Counter<uint64_t> createUInt64Counter(
      StringRef name, Level instrumentLevel,
      const llvm::StringMap<MetricAttributeValue> &attributes = {},
      StringRef description = "", StringRef unit = "") {
    return createCounter<uint64_t>(name, instrumentLevel, attributes,
                                   description, unit);
  }

  /// Create a Histogram<uint64_t>.
  Histogram<uint64_t> createUInt64Histogram(
      StringRef name, Level instrumentLevel,
      const llvm::StringMap<MetricAttributeValue> &attributes = {},
      StringRef description = "", StringRef unit = "") {
    return createHistogram<uint64_t>(name, instrumentLevel, attributes,
                                     description, unit);
  }

  /// Create a Timer. If unit is omitted, the method will implicitly set
  /// it to one of {"ns", "us", "ms", "s"} based on the DurationT template
  /// parameter (e.g. std::chrono::microseconds).
  template <typename DurationT>
  Timer<uint64_t, DurationT> createUInt64Timer(
      StringRef name, Level instrumentLevel,
      const llvm::StringMap<MetricAttributeValue> &attributes = {},
      StringRef description = "", StringRef unit = "") {
    if (unit.empty()) {
      if constexpr (std::is_same_v<DurationT, std::chrono::nanoseconds>)
        unit = "ns";
      else if constexpr (std::is_same_v<DurationT, std::chrono::microseconds>)
        unit = "us";
      else if constexpr (std::is_same_v<DurationT, std::chrono::milliseconds>)
        unit = "ms";
      else if constexpr (std::is_same_v<DurationT, std::chrono::seconds>)
        unit = "s";
    }
    if (isUserMetric(instrumentLevel)) {
      if (userMeter)
        return Timer<uint64_t, DurationT>(
            userMeter->CreateUInt64Histogram(name, description, unit),
            attributes);
      else
        return Timer<uint64_t, DurationT>(
            noopMeter->CreateUInt64Histogram(name, description, unit),
            attributes);
    }
    if (isInstrumentEnabled(instrumentLevel))
      return Timer<uint64_t, DurationT>(
          meter->CreateUInt64Histogram(name, description, unit), attributes);
    else
      return Timer<uint64_t, DurationT>(
          noopMeter->CreateUInt64Histogram(name, description, unit),
          attributes);
  }

  /// Create a Logger with given domain (see
  /// https://opentelemetry.io/docs/specs/otel/logs/semantic_conventions/events/).
  virtual std::shared_ptr<Logs::Logger> getLogger(StringRef eventDomain) {
    auto otelLogger = loggerProvider->GetLogger("modular_logger");
    auto otelEventLogger =
        eventLoggerProvider->CreateEventLogger(otelLogger, eventDomain);
    return std::shared_ptr<Logs::Logger>(
        new Logs::Logger(otelEventLogger, telemetryLevel));
  }

  /// Flush all the collected metrics. Blocks until the flush completes
  /// or the timeout elapses, whichever comes first.
  /// NOTE: TelemetryContext flushes periodically asynchronously. Manual
  /// flushing is not recommended except where needed (for example the
  /// TelemetryContext flushes itself at shutdown).
  void
  flush(std::chrono::microseconds timeout = std::chrono::microseconds::max());

private:
  /// Configured telemetry level for this telemetry context.
  Level telemetryLevel;
  StringRef machineId;
  // Metrics.
  std::unique_ptr<opentelemetry::metrics::MeterProvider> userMetricsProvider;
  std::shared_ptr<opentelemetry::metrics::Meter> userMeter;
  std::unique_ptr<opentelemetry::metrics::MeterProvider> metricsProvider;
  std::shared_ptr<opentelemetry::metrics::Meter> meter;
  std::unique_ptr<opentelemetry::metrics::MeterProvider> noopMetricsProvider;
  std::shared_ptr<opentelemetry::metrics::Meter> noopMeter;
  //  Logs.
  std::shared_ptr<opentelemetry::logs::LoggerProvider> loggerProvider;
  std::shared_ptr<opentelemetry::logs::EventLoggerProvider> eventLoggerProvider;

  bool isValidInstrumentName(StringRef name) {
    // TODO: SERV-138 - If the name is invalid, it looks like OTel logs the
    // error and returns a NOOP counter. Instead, we should probably try to
    // assert that the name is valid or that the returned counter is not NOOP.
    // Same for other instruments.
    return !name.empty();
  }

  template <typename T>
  Counter<T>
  createCounter(StringRef name, Level instrumentLevel,
                const llvm::StringMap<MetricAttributeValue> &attributes = {},
                StringRef description = "", StringRef unit = "") {
    assert(isValidInstrumentName(name) && "instrument name is invalid");
    if (isUserMetric(instrumentLevel) && userMeter)
      return createCounterImpl<T>(userMeter, name, description, unit,
                                  attributes);
    if (isInstrumentEnabled(instrumentLevel))
      return createCounterImpl<T>(meter, name, description, unit, attributes);
    else
      return createCounterImpl<T>(noopMeter, name, description, unit,
                                  attributes);
  }

  // Utility function to help make code cleaner
  template <typename T>
  Counter<T>
  createCounterImpl(std::shared_ptr<opentelemetry::metrics::Meter> m,
                    StringRef name, StringRef description, StringRef unit,
                    const llvm::StringMap<MetricAttributeValue> &attributes) {
    if constexpr (std::is_same_v<T, uint64_t>) {
      return Counter<uint64_t>(
          m->CreateUInt64Counter(name.data(), description.data(), unit.data()),
          attributes);
    } else if constexpr (std::is_same_v<T, double>) {
      return Counter<double>(
          m->CreateDoubleCounter(name.data(), description.data(), unit.data()),
          attributes);
    }
  }

  /// Create a Histogram
  template <typename T>
  Histogram<T>
  createHistogram(StringRef name, Level instrumentLevel,
                  const llvm::StringMap<MetricAttributeValue> &attributes = {},
                  StringRef description = "", StringRef unit = "") {
    assert(isValidInstrumentName(name) && "instrument name is invalid");
    if (isUserMetric(instrumentLevel) && userMeter)
      return createHistogramImpl<T>(userMeter, name, description, unit,
                                    attributes);
    if (isInstrumentEnabled(instrumentLevel))
      return createHistogramImpl<T>(meter, name, description, unit, attributes);
    return createHistogramImpl<T>(noopMeter, name, description, unit,
                                  attributes);
  }

  // Utility function to help make code cleaner
  template <typename T>
  Histogram<T>
  createHistogramImpl(std::shared_ptr<opentelemetry::metrics::Meter> m,
                      StringRef name, StringRef description, StringRef unit,
                      const llvm::StringMap<MetricAttributeValue> &attributes) {
    if constexpr (std::is_same_v<T, uint64_t>) {
      return Histogram<uint64_t>(m->CreateUInt64Histogram(name.data(),
                                                          description.data(),
                                                          unit.data()),
                                 attributes);
    } else if constexpr (std::is_same_v<T, double>) {
      return Histogram<double>(m->CreateDoubleHistogram(name.data(),
                                                        description.data(),
                                                        unit.data()),
                               attributes);
    }
  }
};

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_H
