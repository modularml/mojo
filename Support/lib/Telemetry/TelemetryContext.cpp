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

#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Common.h"
#include "Support/Telemetry/Telemetry.h"

#include "Config/Version.h"
#include "Support/Base64.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Random.h"
#include "Support/Telemetry/Exporters/FileLogExporter.h"
#include "Support/Telemetry/Exporters/FileMetricExporter.h"
#include "Support/Threading/HWInfo.h"
#include "opentelemetry/metrics/noop.h"
#include "opentelemetry/sdk/metrics/aggregation/aggregation_config.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_options.h"
#include "opentelemetry/sdk/metrics/instruments.h"
#include "opentelemetry/sdk/metrics/view/instrument_selector.h"
#include "opentelemetry/sdk/metrics/view/meter_selector.h"
#include "opentelemetry/sdk/metrics/view/view.h"
#include "opentelemetry/sdk/metrics/view/view_registry.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include "llvm/TargetParser/Host.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter_options.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/common/global_log_handler.h"
#include "opentelemetry/sdk/logs/event_logger_provider_factory.h"
#include "opentelemetry/sdk/logs/logger_provider.h"
#include "opentelemetry/sdk/logs/logger_provider_factory.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"

#include "opentelemetry/sdk/logs/processor.h"
#include "opentelemetry/sdk/logs/simple_log_record_processor_factory.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/resource/resource.h"

// enable TEST_UDS to use unix domain sockets for log/metrics. Do set the config
// value for `telemetry.exporters.metrics.uds_name` to the right socket from the
// server
#define TEST_UDS 0

#include <algorithm> // For std::sort.
#include <cstdlib>   // For std::getenv

#define DEBUG_TYPE "telemetry-context"

using namespace M;
using namespace Telemetry;
using namespace Exporter;

static bool isModularEmployee() {
  const char *home = std::getenv("HOME");
  if (home == nullptr)
    return false;

  return std::filesystem::exists(std::filesystem::path(home) /
                                 ".modular-internal");
}

static Level levelFromString(StringRef levelStr) {
  if (levelStr.empty())
    return Level::L1;

  int level = 0;
  if (levelStr.getAsInteger(10, level))
    assert(false && "Non-integer telemetry level specified");
  assert((level >= 0 && level < 3) && "Telemetry level outside [0,2] range");
  if (level == 0)
    return Level::L0;
  if (level == 1)
    return Level::L1;
  if (level == 2)
    return Level::L2;
  llvm_unreachable("unknown telemetry level");
}

static void configureInternalLogging(StringRef internalLogConfig) {
  // OTel internal logging (e.g. warnings and errors related to OTel's
  // operation) is off by default and controlled with `telemetry.internal_log`
  // config key (or equivalently with `TELEMETRY_INTERNAL_LOG` env var).
  bool internalLogsOff = false;
  opentelemetry::sdk::common::internal_log::LogLevel logLevel;
  if (internalLogConfig.empty() || internalLogConfig == "off") {
    internalLogsOff = true;
  } else {
    if (internalLogConfig == "error") {
      logLevel = opentelemetry::sdk::common::internal_log::LogLevel::Error;
    } else if (internalLogConfig.starts_with("warn")) {
      logLevel = opentelemetry::sdk::common::internal_log::LogLevel::Warning;
    } else if (internalLogConfig == "info") {
      logLevel = opentelemetry::sdk::common::internal_log::LogLevel::Info;
    } else if (internalLogConfig == "debug") {
      logLevel = opentelemetry::sdk::common::internal_log::LogLevel::Debug;
    } else {
      LDBG() << "Unrecognized log level for telemetry.internal_log: "
             << internalLogConfig;
      internalLogsOff = true;
    }
  }
  if (internalLogsOff) {
    // Use NOOP log handler to disable all OTel internal logs.
    auto noopHandler = std::make_shared<
        opentelemetry::sdk::common::internal_log::NoopLogHandler>();
    opentelemetry::sdk::common::internal_log::GlobalLogHandler::SetLogHandler(
        noopHandler);
  } else {
    opentelemetry::sdk::common::internal_log::GlobalLogHandler::SetLogLevel(
        logLevel);
  }
}

/// creates local identifiers; see Telemetry.h.
const M::Telemetry::LocalIDs &M::Telemetry::createLocalIDs() {
  // Memoized so every caller in the process observes identical values: the
  // crash reporting and usage telemetry lanes must share machineid/sessionid
  // for their events to be joinable.
  static const LocalIDs ids = [] {
    std::vector<std::string> macs = localMACs();
    std::sort(std::begin(macs), std::end(macs));

    llvm::BLAKE3 hashState{};
    for (const auto &mac : macs)
      hashState.update(StringRef(mac));

    auto hash = hashState.result();
    std::string machineID =
        encodeURLSafeBase64(StringRef((char *)hash.begin(), hash.size()));

    // Mix in some random bytes in order to construct a local session
    // identifier. This may suffer from a cardinality explosion (and we may
    // choose to rely on the machineid in the future instead), but we can make
    // that decision in the backend separately.
    SecureRandomBytesGenerator rng;
    std::array<uint8_t, 32> scratchBuf = {};
    auto err = rng.getRandomBytes(scratchBuf);
    assert(!err.isError());
    hashState.update(scratchBuf);
    hash = hashState.result();
    std::string sessionID =
        encodeURLSafeBase64(StringRef((char *)hash.begin(), hash.size()));

    return LocalIDs{machineID, sessionID};
  }();
  return ids;
}

bool M::Telemetry::isTelemetryEnabled(Config &settings) {
  return settings.getValueAsBool("telemetry.enabled",
#ifdef MODULAR_PRODUCTION
                                 true
#else
                                 false
#endif // MODULAR_PRODUCTION
  );
}

bool M::Telemetry::isCrashReportingEnabled(Config &settings) {
  return settings.getValueAsBool("crash_reporting.enabled",
                                 isTelemetryEnabled(settings));
}

static size_t getMaxProcessors(const HostMachineInfo &hostInfo) {
  auto limitsOr = CPULimits::get();
  if (!limitsOr.isError()) {
    auto millicores = limitsOr->millicores;
    if (millicores)
      return *millicores / 1000;
  }
  return hostInfo.numPhysicalCores;
}

TelemetryContext::~TelemetryContext() {
  flush(kShutdownFlushTimeout);
  // Flush metrics.
  auto metricsProviderImpl =
      static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
          metricsProvider.get());
  metricsProviderImpl->Shutdown();
  if (userMetricsProvider) {
    auto userMetricsProviderImpl =
        static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
            userMetricsProvider.get());
    userMetricsProviderImpl->Shutdown();
  }

  // Flush logs.
  auto loggerProviderImpl =
      std::static_pointer_cast<opentelemetry::sdk::logs::LoggerProvider>(
          loggerProvider);
  loggerProviderImpl->Shutdown();
}

void TelemetryContext::flush(std::chrono::microseconds timeout) {
  // Flush metrics.
  auto metricsProviderImpl =
      static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
          metricsProvider.get());
  metricsProviderImpl->ForceFlush(timeout);
  if (userMetricsProvider) {
    auto userMetricsProviderImpl =
        static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
            userMetricsProvider.get());
    userMetricsProviderImpl->ForceFlush(timeout);
  }

  // Flush logs.
  auto loggerProviderImpl =
      std::static_pointer_cast<opentelemetry::sdk::logs::LoggerProvider>(
          loggerProvider);
  loggerProviderImpl->ForceFlush(timeout);
}

TelemetryContext::TelemetryContext(Config &settings, StringRef programName,
                                   StringRef subCommand) {
  using namespace opentelemetry::sdk::resource;
  // -------- Resources --------
  // Get the map of resources for the full host info.
  ResourceAttributes attrs;
  std::string programNameStr = programName.str();
  std::string subCommandStr = subCommand.str();
  auto hostInfoOr = getHostMachineInfo();
  // May fail in sandboxed or containerized environments, but do not print a
  // warning as some Telemetry data not initializing does not help the user.
  if (!hostInfoOr.isError()) {
    // Set the CPU and architecture.
    attrs.SetAttribute("cpu.description", hostInfoOr->cpuModelName);
    // WARNING: Metering & billing depends on cpu.arch. Do not remove!
    attrs.SetAttribute("cpu.arch", hostInfoOr->cpuArch);
    // Set the CPU features.
    std::vector<std::string_view> featuresView;
    for (auto &f : hostInfoOr->cpuFeatures)
      featuresView.emplace_back(f);
    attrs.SetAttribute("cpu.features", featuresView);
    // Set some of the other useful features, like number of cores and operating
    // system.
    attrs.SetAttribute("cpu.cores", hostInfoOr->numPhysicalCores);
    attrs.SetAttribute("cpu.max_cores", getMaxProcessors(*hostInfoOr));
    attrs.SetAttribute("cpu.model_name", hostInfoOr->cpuModelName);
    attrs.SetAttribute("os.type", hostInfoOr->osName);
    attrs.SetAttribute("os.version", hostInfoOr->osVersion);
  } else {
    LDBG() << "getHostMachineInfo() failed: " << hostInfoOr.getError()
           << "; falling back to getHostCPUName() for cpu.arch";
    // WARNING: Metering & billing depends on cpu.arch. Do not remove!
    // getHostCPUName() always succeeds, though it may return "generic"
    // on unrecognized CPUs.
    attrs.SetAttribute("cpu.arch", llvm::sys::getHostCPUName().str());
  }

  // Get total memory.
  auto memoryOr = getHostTotalMemoryKB();
  if (!memoryOr.isError()) {
    attrs.SetAttribute("memory", memoryOr.takeValue());
  }

  // Check if we are running in a container
  auto isInContainer = getHostIsInContainer();
  if (!isInContainer.isError())
    attrs.SetAttribute("system.in.container", isInContainer.takeValue());

  // Set the underlying Modular version.
  auto version = getModularVersion();
  attrs.SetAttribute("modular.version.major", version.major);
  attrs.SetAttribute("modular.version.minor", version.minor);
  attrs.SetAttribute("modular.version.patch", version.patch);
  attrs.SetAttribute("modular.version.label", version.label);
  attrs.SetAttribute("modular.version.revision", version.revision);
  attrs.SetAttribute("modular.version.buildtype", version.buildType);

  // Set the local machineid.
  const auto &localIDs = createLocalIDs();
  // WARNING: Metering & billing depends on machineid. Do not remove!
  attrs.SetAttribute("machineid", localIDs.machine);
  attrs.SetAttribute("sessionid", localIDs.session);
  machineId = localIDs.machine;

  auto webId = settings.getValue("web.id");
  if (webId.empty()) {
    auto homeDir = llvm::sys::Process::GetEnv("HOME");
    if (homeDir) {
      auto webIdFile =
          std::filesystem::path(*homeDir) / ".modular" / "webUserId";
      if (std::filesystem::exists(webIdFile)) {
        auto mBufOr = llvm::MemoryBuffer::getFile(webIdFile.string(),
                                                  /*IsText=*/true);
        if (mBufOr) {
          std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mBufOr);
          auto buffer = mbuf->getBuffer();
          size_t newlineLoc = buffer.find_first_of("\n\r\f\v");
          webId = buffer.take_front(newlineLoc);
          if (!webId.empty())
            attrs.SetAttribute("web.user.id", webId);
        }
      }
    }
  } else {
    attrs.SetAttribute("web.user.id", webId);
  }

  attrs.SetAttribute("modular.employee", isModularEmployee());

  bool enabled = isTelemetryEnabled(settings);

  // Get telemetry level.
  auto level = settings.getValue("telemetry.level");
  telemetryLevel = levelFromString(level);

  // Configure OTel internal logging.
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() {
    configureInternalLogging(settings.getValue("telemetry.internal_log"));
  });

  // Get the user ID if we have one.
  attrs.SetAttribute("enduser.id", settings.getValue("user.id"));

  // Set the program name if provided.
  if (!programNameStr.empty())
    attrs.SetAttribute("program.name", programNameStr);

  // Get the resource object we can give to OTel.
  auto otelResources = Resource::Create(attrs).Merge(Resource::GetDefault());

  // -------- Metrics --------
  // Initialize the MeterProvider.
  auto provider = std::make_unique<opentelemetry::sdk::metrics::MeterProvider>(
      std::make_unique<opentelemetry::sdk::metrics::ViewRegistry>(),
      otelResources);

  opentelemetry::sdk::metrics::PeriodicExportingMetricReaderOptions options;
  options.export_interval_millis = kExportInterval;
  options.export_timeout_millis = kExportTimeout;

  // Extend the histogram buckets for our timers. The default's max bucket is
  // 10000 ms.
  auto instrumentSelector =
      std::make_unique<opentelemetry::sdk::metrics::InstrumentSelector>(
          opentelemetry::sdk::metrics::InstrumentType::kHistogram, ".*\\.time$",
          "ms");
  auto meterSelector =
      std::make_unique<opentelemetry::sdk::metrics::MeterSelector>("", "", "");
  auto histConfig = std::make_shared<
      opentelemetry::sdk::metrics::HistogramAggregationConfig>();
  histConfig->boundaries_ = {0,     50,    100,   250,   500,   750,   1000,
                             2500,  5000,  7500,  10000, 12500, 15000, 17500,
                             20000, 25000, 30000, 40000, 50000};
  auto view = std::make_unique<opentelemetry::sdk::metrics::View>(
      "", "", "", opentelemetry::sdk::metrics::AggregationType::kHistogram,
      histConfig);

  provider->AddView(std::move(instrumentSelector), std::move(meterSelector),
                    std::move(view));

  // Shared one-time warning flag for all HTTP exporters in this context.
  auto exportWarned = std::make_shared<std::atomic<bool>>(false);

  // Get metrics exporter config.
  auto httpEndpoint =
      settings.getValue("telemetry.exporters.metrics.http_endpoint");
  std::filesystem::path filePath =
      settings.getValue("telemetry.exporters.metrics.file_path").str();

  // Create metric readers, one for each exporter.

  if (enabled && !filePath.empty()) {
    // File exporter.
    auto exporter = std::make_unique<FileMetricExporter>(filePath);
    auto reader = std::make_shared<
        opentelemetry::sdk::metrics::PeriodicExportingMetricReader>(
        std::move(exporter), options);
    provider->AddMetricReader(reader);
  }

  if (enabled && !httpEndpoint.empty()) {
#if TEST_UDS
    std::filesystem::path udsName =
        settings.getValue("telemetry.exporters.metrics.uds_name").str();
    auto exporter = std::make_unique<UDSMetricExporter>(udsName);
#else

    // HTTP OTLP exporter.
    opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions otlpOptions;

    otlpOptions.url = (httpEndpoint + "/v1/metrics").str();
    otlpOptions.timeout = kOtlpRequestTimeout;
    auto exporter =
        opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(
            otlpOptions);
#endif
    // Wrap to detect and report export failures.
    auto warningExporter = std::make_unique<WarningMetricExporter>(
        std::move(exporter), httpEndpoint.str(), exportWarned);
    auto reader = std::make_shared<
        opentelemetry::sdk::metrics::PeriodicExportingMetricReader>(
        std::move(warningExporter), options);
    provider->AddMetricReader(reader);
  }

  metricsProvider = std::unique_ptr<opentelemetry::metrics::MeterProvider>(
      provider.release());
  meter = metricsProvider->GetMeter("modular");

  noopMetricsProvider =
      std::make_unique<opentelemetry::metrics::NoopMeterProvider>();
  noopMeter = noopMetricsProvider->GetMeter("modular");

  // -------- Logs --------
  // Get logs exporter config.
  httpEndpoint = settings.getValue("telemetry.exporters.logs.http_endpoint");
  if (httpEndpoint.empty())
    httpEndpoint = MODULAR_TELEMETRY_URL;
  filePath = settings.getValue("telemetry.exporters.logs.file_path").str();

  // Create log processors for each exporter.
  std::vector<std::unique_ptr<opentelemetry::sdk::logs::LogRecordProcessor>>
      processors;

  if (enabled && !filePath.empty()) {

    auto logExporter = std::make_unique<FileLogExporter>(filePath);
    processors.emplace_back(
        opentelemetry::sdk::logs::SimpleLogRecordProcessorFactory::Create(
            std::move(logExporter)));
  }

  if (enabled && !httpEndpoint.empty()) {
#if TEST_UDS
    auto logExporter = std::make_unique<UDSLogExporter>(udsName, "/v1/logs");
#else

    // HTTP OTLP exporter.
    opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterOptions
        otlpLogOptions;

    otlpLogOptions.url = (httpEndpoint + "/v1/logs").str();
    otlpLogOptions.timeout = kOtlpRequestTimeout;
    auto logExporter =
        opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterFactory::Create(
            otlpLogOptions);
#endif
    // Wrap to detect and report export failures.
    auto warningExporter = std::make_unique<WarningLogRecordExporter>(
        std::move(logExporter), httpEndpoint.str(), exportWarned);
    // Run the HTTP export on a detached thread so emit is not blocked by
    // the delegate's network I/O. OTel's curl HTTP exporter relies on the
    // system's synchronous name resolver, so its 3s CURLOPT_TIMEOUT_MS does
    // not cap DNS resolution or TCP SYN retries, and synchronous emit can
    // stall for tens of seconds when the endpoint is unreachable — see
    // SDLC-3618.
    auto asyncExporter = std::make_unique<FireAndForgetLogRecordExporter>(
        std::move(warningExporter));
    processors.emplace_back(
        opentelemetry::sdk::logs::SimpleLogRecordProcessorFactory::Create(
            std::move(asyncExporter)));
  }

  loggerProvider = opentelemetry::sdk::logs::LoggerProviderFactory::Create(
      std::move(processors), otelResources);
  eventLoggerProvider =
      opentelemetry::sdk::logs::EventLoggerProviderFactory::Create();

  // Emit the once-per-process program.initialized event: the usage-lane record
  // of this session, used for adoption counts and as the failure-rate
  // denominator. It reports whether the crash lane was on rather than being
  // gated on it, so a session that opted out of crash reporting still counts
  // toward adoption while the failure-rate mart excludes it — a session that
  // could not have reported a crash must never read as a crash-free one.
  if (!programNameStr.empty() && enabled) {
    auto logger = getLogger("program");
    llvm::StringMap<Logs::AttributeValue> eventAttrs;
    eventAttrs["crash_reporting.enabled"] = isCrashReportingEnabled(settings);
    if (!subCommandStr.empty())
      eventAttrs["program.sub_command"] = subCommandStr;
    logger->emitL0Event("program.initialized", eventAttrs);
  }
}
