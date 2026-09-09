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

#include "Support/Telemetry/Telemetry.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "Support/Telemetry/Logs.h"
#include "llvm/Support/MemoryBuffer.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <thread>

#include <stdlib.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using namespace Telemetry;

/// RAII-style way to restore Modular config after each test.
struct LogFileSetup {
public:
  LogFileSetup(StringRef signalType) {
    EXPECT_THAT(signalType.str(), testing::AnyOf("metrics", "logs"));

    filePathKey = ("telemetry.exporters." + signalType + ".file_path").str();
    httpUrlKey = ("telemetry.exporters." + signalType + ".http_endpoint").str();
  }

  /// NOTE: This config stays in-memory and is passed to the TelemetryContext
  /// constructor. This is done to isolate the telemetry config used by these
  /// unit tests from Modular's centralized config. There might be other
  /// processes running at the same time as these tests and emitting telemetry,
  /// and we don't want them to write to the same files.
  Config getConfig() { return std::move(cfg); }

  TempFile getLogFile(StringRef prefix, StringRef level) {
    EXPECT_THAT(level.str(), testing::AnyOf("0", "1", "2"));
    auto tmpOr =
        TempFile::create((prefix + "-test-telemetry-%%%%%%%.log").str());
    EXPECT_FALSE(tmpOr.isError()) << tmpOr.getError();

    // Set the config value.
    cfg.setValue(filePathKey, tmpOr->getPath().string());
    // These tests don't need to send to any HTTP endpoint.
    cfg.setValue(httpUrlKey, "");

    // Set telemetry level.
    cfg.setValue("telemetry.level", level);

    // Return the temp file so it'll automatically get destroyed.
    return std::move(*tmpOr);
  }

private:
  Config cfg;
  std::string filePathKey;
  std::string httpUrlKey;
  std::string filePathOriginalValue;
  std::string httpUrlOriginalValue;
};

/// This function parses an OTel message, and provides visitor-style access to
/// the fields in a message. The message looks like this:
///
///  {
///    scope name	: modular
///    schema url	:
///    version	:
///    start time	: Wed Jul 26 21:48:59 2023
///    end time	: Wed Jul 26 21:48:59 2023
///    instrument name	: basic.counter
///    description	:
///    unit		:
///    type		: SumPointData
///    value		: 32
///    attributes		:
///    resources	:
///  	arch: apple-m1
///  	cores: 10
///  	cpu: Apple M1 Max\0
///  	features: []
///  	operating system: darwin
///  	service.name: unknown_service
///  	telemetry.sdk.language: cpp
///  	telemetry.sdk.name: opentelemetry
///  	telemetry.sdk.version: 1.9.1
///  }
///
static void iterateFields(StringRef message,
                          function_ref<void(StringRef, StringRef)> callback) {
  StringRef line;
  while (!message.empty()) {
    line = message.take_until([](char c) { return c == '\n' || c == '}'; });
    if (line.empty())
      return;
    StringRef key, value;
    std::tie(key, value) = line.split(':');
    callback(key.trim(), value.trim());
    // Drop the line we just handled.
    message = message.drop_front(line.size());
    message.consume_front("\n");
  }
}

/// This function provides visitor-style access to all the fields of every
/// message. The OTel text stream looks like this (many new-line delimited
/// messages):
///
/// {
///   scope name: modular
///   ...
/// }
/// {
///   scope name: modular
///   ...
/// }
/// ...
///
/// See above for an example message.
static void iterateMessages(StringRef log,
                            function_ref<void(StringRef, StringRef)> callback) {
  StringRef message;
  while (!log.empty()) {
    log.consume_front("{\n");
    message = log.take_until([](char c) { return c == '}'; });
    if (message == "}")
      return;
    iterateFields(message, callback);
    log = log.drop_front(message.size());
    log.consume_front("}\n");
  }
}

/// This function provides visitor-style access to every message.
static void iterateMessages(StringRef log,
                            function_ref<void(StringRef)> callback) {
  StringRef message;
  while (!log.empty()) {
    log.consume_front("{\n");
    message = log.take_until([](char c) { return c == '}'; });
    if (message == "}")
      return;
    callback(message);
    log = log.drop_front(message.size());
    log.consume_front("}\n");
  }
}

/// Returns the value of the "event.name" field in a log message, or an empty
/// StringRef if not present.  Assumes the log format "event.name: <value>\n".
static StringRef getEventName(StringRef message) {
  auto pos = message.find("event.name: ");
  if (pos == StringRef::npos)
    return "";
  return message.substr(pos + strlen("event.name: "))
      .take_until([](char c) { return c == '\n'; })
      .trim();
}

/// RAII guard that saves and restores an environment variable.
struct EnvGuard {
  std::string key;
  std::optional<std::string> prev;
  EnvGuard(const char *k, const char *v) : key(k) {
    if (auto *p = ::getenv(k))
      prev = p;
    ::setenv(k, v, 1);
  }
  ~EnvGuard() {
    if (prev)
      ::setenv(key.c_str(), prev->c_str(), 1);
    else
      ::unsetenv(key.c_str());
  }
};

/// This test ensures that when we create and increment a counter, we get the
/// values we expect in the log file, in the order we expect.
TEST(Telemetry, Counter) {
  LogFileSetup logFileSetup("metrics");
  TempFile tmpFile = logFileSetup.getLogFile("counter", "0");
  Config settings(logFileSetup.getConfig());
  TelemetryContext ctx(settings);

  auto counter = ctx.createUInt64Counter("basic.test.counter", Level::L0);
  counter.add(32);
  ctx.flush();
  counter.add(10);
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool found32 = false;
        bool foundPlus10 = false;
        StringRef currentInstrument = "";
        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          int i;
          if (key == "instrument name") {
            currentInstrument = value;
            return;
          }
          // consumeInteger returns *false* on success.
          if (key == "value" && currentInstrument == "basic.test.counter" &&
              !value.consumeInteger(10, i)) {
            if (i == 32) {
              EXPECT_FALSE(foundPlus10) << "expected to find 32 first";
              found32 = true;
            } else if (i == 42) {
              EXPECT_TRUE(found32)
                  << "expected to find 32 first, found 32 + 10 first?";
              foundPlus10 = true;
            }
          }
        });

        EXPECT_TRUE(found32 && foundPlus10)
            << "expected to find both counter values";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that if we create a histogram and add some records, we get
/// the values we expect in the log file.
/// FIXME(SVCS-218): This test is flaky.
TEST(Telemetry, DISABLED_Histogram) {
  LogFileSetup logFileSetup("metrics");
  TempFile tmpFile = logFileSetup.getLogFile("histogram", "1");
  Config settings(logFileSetup.getConfig());
  TelemetryContext ctx(settings);

  std::string value = "ATTRIBUTE";
  llvm::StringMap<MetricAttributeValue> attributes = {{"TELEMETRY", value}};
  auto hist =
      ctx.createUInt64Histogram("basic.test.histogram", Level::L0, attributes);
  hist.record(32);
  hist.record(10);

  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        auto getLineStartingAt = [&](auto pos) {
          StringRef str = mbuf->getBuffer().substr(pos);
          return str.take_until([](char c) { return c == '\n'; });
        };

        bool instrumentFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          auto instrumentPos = message.find("instrument name");
          StringRef instrumentLine = getLineStartingAt(instrumentPos);
          if (instrumentLine.split(':').second.trim() != "basic.test.histogram")
            return;

          instrumentFound = true;

          auto countPos = message.find("count");
          StringRef countLine = getLineStartingAt(countPos);
          EXPECT_EQ(countLine.split(':').second.trim(), "2");

          auto minPos = message.find("min");
          StringRef minLine = getLineStartingAt(minPos);
          EXPECT_EQ(minLine.split(':').second.trim(), "10");

          auto maxPos = message.find("max");
          StringRef maxLine = getLineStartingAt(maxPos);
          EXPECT_EQ(maxLine.split(':').second.trim(), "32");

          auto bucketsPos = message.find("buckets");
          StringRef bucketsLine = getLineStartingAt(bucketsPos);
          EXPECT_EQ(bucketsLine.split(':').second.trim(),
                    "[0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, "
                    "5000, 7500, 10000, ]");

          auto countsPos = message.find("counts");
          StringRef countsLine = getLineStartingAt(countsPos);
          EXPECT_EQ(countsLine.split(':').second.trim(),
                    "[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]");
        });

        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          if (key == "TELEMETRY") {
            EXPECT_EQ(value, "ATTRIBUTE");
          }
        });

        EXPECT_TRUE(instrumentFound) << "expected to find histogram in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that measurements for L1 instruments are not emitted
/// when the telemetry level is L0.
TEST(Telemetry, HistogramL1) {
  LogFileSetup logFileSetup("metrics");
  TempFile tmpFile = logFileSetup.getLogFile("histogram", "0");
  Config settings(logFileSetup.getConfig());
  TelemetryContext ctx(settings);

  auto hist = ctx.createUInt64Histogram("optional.histogram", Level::L1);
  hist.record(32);
  hist.record(10);
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool instrumentFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          if (key == "instrument name" && value == "optional.histogram")
            instrumentFound = true;
        });

        EXPECT_FALSE(instrumentFound)
            << "expected not to find histogram in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

// This check tests that our Timer object works, and properly tags attributes
// when it goes out of scope
TEST(Telemetry, Timer) {
  LogFileSetup logFileSetup("metrics");
  TempFile tmpFile = logFileSetup.getLogFile("timer", "0");
  Config settings(logFileSetup.getConfig());
  TelemetryContext ctx(settings);

  auto lambdaTest = [&]() {
    std::string value = "ATTRIBUTE";
    llvm::StringMap<MetricAttributeValue> attrs = {{"TELEMETRY", value}};
    auto timer = ctx.createUInt64Timer<std::chrono::milliseconds>(
        "basic.test.timer", Level::L0, attrs);
    value[0] = 'C';
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  };

  lambdaTest();
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        auto getLineStartingAt = [&](auto pos) {
          StringRef str = mbuf->getBuffer().substr(pos);
          return str.take_until([](char c) { return c == '\n'; });
        };

        bool instrumentFound = false;
        std::cerr << mbuf->getBuffer().str() << "\n";
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          auto instrumentPos = message.find("instrument name");
          StringRef instrumentLine = getLineStartingAt(instrumentPos);
          if (instrumentLine.split(':').second.trim() != "basic.test.timer")
            return;

          instrumentFound = true;

          auto maxPos = message.find("max");
          StringRef maxLine = getLineStartingAt(maxPos);
          EXPECT_NE(maxLine.split(':').second.trim().compare_numeric("100"),
                    -1);
        });

        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          if (key == "TELEMETRY") {
            EXPECT_EQ(value, "ATTRIBUTE");
          }
        });

        EXPECT_TRUE(instrumentFound) << "expected to find timer in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that logs are properly flushed to the log file, escapes and
/// all.
TEST(Telemetry, Logger) {
  LogFileSetup logFileSetup("logs");
  TempFile tmpFile = logFileSetup.getLogFile("log", "1");
  Config settings(logFileSetup.getConfig());
  TelemetryContext ctx(settings);

  auto logger = ctx.getLogger("basic.log");
  logger->emitL1Event("test.Logger", {{"attr1", "hello"}, {"attr2", "world"}});
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        auto getLineStartingAt = [&](auto pos) {
          StringRef str = mbuf->getBuffer().substr(pos);
          return str.take_until([](char c) { return c == '\n'; });
        };

        bool eventFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          auto eventNamePos = message.find("event.name");
          StringRef eventNameLine = getLineStartingAt(eventNamePos);
          if (eventNameLine.split(':').second.trim() != "test.Logger")
            return;

          eventFound = true;

          // There should be no body
          auto bodyPos = message.find("body");
          StringRef bodyLine = getLineStartingAt(bodyPos);
          EXPECT_EQ(bodyLine.split(':').second.trim(), "");

          auto severityPos = message.find("severity_text");
          StringRef severityLine = getLineStartingAt(severityPos);
          EXPECT_EQ(severityLine.split(':').second.trim(), "INFO");

          auto attribute1Pos = message.find("attr1");
          StringRef attr1Line = getLineStartingAt(attribute1Pos);
          EXPECT_EQ(attr1Line.split(':').second.trim(), "hello");

          auto attribute2Pos = message.find("attr2");
          StringRef attr2Line = getLineStartingAt(attribute2Pos);
          EXPECT_EQ(attr2Line.split(':').second.trim(), "world");
        });

        EXPECT_TRUE(eventFound) << "expected to find event in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that L2 events are not emitted when telemetry
/// level is L1.
TEST(Telemetry, LoggerL2) {
  LogFileSetup logFileSetup("logs");
  TempFile tmpFile = logFileSetup.getLogFile("log", "1");
  Config settings(logFileSetup.getConfig());
  TelemetryContext ctx(settings);

  auto logger = ctx.getLogger("basic.log");
  logger->emitL2Event("test.LoggerL2");
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool eventFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          if (key == "event.name" && value == "test.LoggerL2")
            eventFound = true;
        });

        EXPECT_FALSE(eventFound) << "expected not to find event in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

TEST(Telemetry, Resources) {
  LogFileSetup logFileSetup("logs");
  TempFile tmpFile = logFileSetup.getLogFile("log", "1");
  Config settings(logFileSetup.getConfig());

  TelemetryContext ctx(settings);

  auto logger = ctx.getLogger("basic.log");
  logger->emitL0Event("test.Resources");
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool eventFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          if (getEventName(message) != "test.Resources")
            return;
          eventFound = true;
        });

        EXPECT_TRUE(eventFound) << "expected to find event in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test verifies that when a TelemetryContext is created with a
/// programName and crash reporting enabled, the "program.name" resource
/// attribute is present and a "program.initialized" event is emitted at L0.
TEST(Telemetry, DISABLED_ProgramInvocation) {
  LogFileSetup logFileSetup("logs");
  TempFile tmpFile = logFileSetup.getLogFile("log", "0");
  Config settings(logFileSetup.getConfig());
  // Override env vars that disable telemetry/crash reporting in test builds.
  // EnvGuard restores the original values when the test exits.
  EnvGuard telemetryGuard("MODULAR_TELEMETRY_ENABLED", "true");
  EnvGuard crashGuard("MODULAR_CRASH_REPORTING_ENABLED", "true");

  TelemetryContext ctx(settings, "test-program", "test-sub-command");
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool eventFound = false;
        bool programNameFound = false;
        bool subCommandFound = false;
        bool crashReportingAttested = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          if (getEventName(message) != "program.initialized")
            return;
          eventFound = true;
          programNameFound = message.contains("program.name: test-program");
          subCommandFound =
              message.contains("program.sub_command: test-sub-command");
          // The ostream exporter renders bool attributes numerically.
          crashReportingAttested =
              message.contains("crash_reporting.enabled: 1");
        });

        EXPECT_TRUE(eventFound) << "expected to find program.initialized event";
        EXPECT_TRUE(programNameFound)
            << "expected to find program.name resource attribute";
        EXPECT_TRUE(subCommandFound)
            << "expected to find program.sub_command event attribute";
        EXPECT_TRUE(crashReportingAttested)
            << "expected crash_reporting.enabled=true attesting that this "
               "session belongs in the failure-rate denominator";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// Recording exporter for the SDLC-3618 regression test below. Captures
/// the thread ID that Export was called on and signals completion.
class RecordingLogRecordExporter
    : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  std::unique_ptr<opentelemetry::sdk::logs::Recordable>
  MakeRecordable() noexcept override {
    // Not invoked by this test — the wrapper is exercised with an empty
    // record span, which is sufficient to verify the dispatch path.
    return nullptr;
  }

  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::nostd::span<
         std::unique_ptr<opentelemetry::sdk::logs::Recordable>> &) noexcept
      override {
    {
      std::lock_guard<std::mutex> lock(mu);
      exportThreadId = std::this_thread::get_id();
      exportCalls += 1;
    }
    cv.notify_all();
    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }

  bool ForceFlush(std::chrono::microseconds) noexcept override { return true; }
  bool Shutdown(std::chrono::microseconds) noexcept override { return true; }

  std::mutex mu;
  std::condition_variable cv;
  int exportCalls = 0;
  std::thread::id exportThreadId{};
};

/// SDLC-3618 regression test: FireAndForgetLogRecordExporter must run the
/// delegate's Export on a different thread than the caller, and its own
/// Export / Shutdown must return without waiting on that work. Before
/// this wrapper existed, the program invocation event was pushed
/// synchronously through the OTel curl HTTP exporter, whose 3 s
/// `CURLOPT_TIMEOUT_MS` does not cap DNS resolution or TCP SYN retries;
/// with an unreachable endpoint (e.g. the Mac CI runners, which can't
/// reach telemetry.modular.com) each emit stalled for tens of seconds,
/// blocking every mojo invocation until the caller's test-harness
/// timeout fired.
///
/// We verify the non-blocking property structurally (delegate's thread
/// ID differs from the caller's) rather than via a long-sleep delegate:
/// a hung detached thread lingering into subsequent tests is both
/// wasteful and risks amplifying unrelated failures when the gtest
/// process is already under resource pressure.
TEST(Telemetry,
     DISABLED_FireAndForgetLogRecordExporterDispatchesOnDetachedThread) {
  auto recording = std::make_unique<RecordingLogRecordExporter>();
  auto *recordingPtr = recording.get();
  FireAndForgetLogRecordExporter async(std::move(recording));

  // An empty record span is enough — the wrapper only cares that Export
  // is invoked and dispatched, not what's in the span.
  std::vector<std::unique_ptr<opentelemetry::sdk::logs::Recordable>> records;
  opentelemetry::nostd::span<
      std::unique_ptr<opentelemetry::sdk::logs::Recordable>>
      span{records.data(), records.size()};

  const auto callerThreadId = std::this_thread::get_id();
  auto start = std::chrono::steady_clock::now();
  auto result = async.Export(span);
  auto exportElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);

  EXPECT_EQ(result, opentelemetry::sdk::common::ExportResult::kSuccess);
  EXPECT_LT(exportElapsed.count(), 1000)
      << "FireAndForgetLogRecordExporter::Export took " << exportElapsed.count()
      << " ms — it must not wait for the delegate's Export to complete "
         "(SDLC-3618 regression).";

  // Shutdown must not wait on the detached work either.
  start = std::chrono::steady_clock::now();
  EXPECT_TRUE(async.Shutdown(std::chrono::microseconds::max()));
  auto shutdownElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  EXPECT_LT(shutdownElapsed.count(), 1000)
      << "FireAndForgetLogRecordExporter::Shutdown took "
      << shutdownElapsed.count()
      << " ms — it must not wait for the delegate's Shutdown to "
         "complete (SDLC-3618 regression).";

  // Wait for the detached thread to actually invoke the delegate, so we
  // can inspect which thread ran it and so the test doesn't leave work
  // in flight after the body returns.
  {
    std::unique_lock<std::mutex> lock(recordingPtr->mu);
    ASSERT_TRUE(recordingPtr->cv.wait_for(lock, std::chrono::seconds(5), [&] {
      return recordingPtr->exportCalls > 0;
    })) << "Detached thread never invoked the delegate's Export";
    EXPECT_EQ(recordingPtr->exportCalls, 1)
        << "Delegate's Export should have been dispatched exactly once";
    EXPECT_NE(recordingPtr->exportThreadId, callerThreadId)
        << "Delegate's Export ran on the caller's thread — the wrapper "
           "is no longer fire-and-forget (SDLC-3618 regression).";
  }
}

// A delegate whose Export blocks until released, so a test can prove
// FireAndForgetLogRecordExporter::ForceFlush actually waits for the in-flight
// export to finish rather than returning while it is still running.
class BlockingLogRecordExporter
    : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  std::unique_ptr<opentelemetry::sdk::logs::Recordable>
  MakeRecordable() noexcept override {
    return nullptr;
  }

  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::nostd::span<
         std::unique_ptr<opentelemetry::sdk::logs::Recordable>> &) noexcept
      override {
    {
      std::unique_lock<std::mutex> lock(mu);
      started = true;
      startedCv.notify_all();
      releaseCv.wait(lock, [&] { return released; });
    }
    exportReturned.store(true);
    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }

  bool ForceFlush(std::chrono::microseconds) noexcept override { return true; }
  bool Shutdown(std::chrono::microseconds) noexcept override { return true; }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mu);
      released = true;
    }
    releaseCv.notify_all();
  }

  std::mutex mu;
  std::condition_variable startedCv;
  std::condition_variable releaseCv;
  bool started = false;
  bool released = false;
  std::atomic<bool> exportReturned{false};
};

// ForceFlush must block until the in-flight export finishes — that drain is
// what keeps a worker from reading the LoggerProvider's Resource after
// teardown (SDLC-3618). Verified deterministically with a delegate that blocks
// inside Export until released.
TEST(Telemetry, FireAndForgetLogRecordExporterForceFlushDrainsInFlight) {
  auto blocking = std::make_unique<BlockingLogRecordExporter>();
  auto *blockingPtr = blocking.get();
  FireAndForgetLogRecordExporter async(std::move(blocking));

  std::vector<std::unique_ptr<opentelemetry::sdk::logs::Recordable>> records;
  opentelemetry::nostd::span<
      std::unique_ptr<opentelemetry::sdk::logs::Recordable>>
      span{records.data(), records.size()};
  async.Export(span);

  // Wait until the worker is actually inside the delegate's (blocked) Export.
  {
    std::unique_lock<std::mutex> lock(blockingPtr->mu);
    ASSERT_TRUE(blockingPtr->startedCv.wait_for(
        lock, std::chrono::seconds(5), [&] { return blockingPtr->started; }))
        << "worker never entered the delegate's Export";
  }

  // While the export is blocked, a short ForceFlush must report the drain did
  // NOT complete, and the export must still be in flight.
  EXPECT_FALSE(async.ForceFlush(std::chrono::milliseconds(50)))
      << "ForceFlush reported drained while an export was still blocked";
  EXPECT_FALSE(blockingPtr->exportReturned.load());

  // Release the delegate; a generous ForceFlush must now wait for the export
  // to finish before returning success.
  blockingPtr->release();
  EXPECT_TRUE(async.ForceFlush(std::chrono::seconds(5)))
      << "ForceFlush did not drain after the export was released";
  EXPECT_TRUE(blockingPtr->exportReturned.load())
      << "ForceFlush returned before the in-flight export completed";
}

/// This test verifies that the program.initialized event is still emitted when
/// crash reporting is explicitly disabled, carrying
/// crash_reporting.enabled=false so the session counts toward adoption but is
/// excluded from the failure-rate denominator downstream.
TEST(Telemetry, DISABLED_ProgramInvocationNoCrashReporting) {
  LogFileSetup logFileSetup("logs");
  TempFile tmpFile = logFileSetup.getLogFile("log", "0");
  Config settings(logFileSetup.getConfig());
  // Enable telemetry but disable crash reporting.
  EnvGuard telemetryGuard("MODULAR_TELEMETRY_ENABLED", "true");
  EnvGuard crashGuard("MODULAR_CRASH_REPORTING_ENABLED", "false");

  TelemetryContext ctx(settings, "test-program");
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool eventFound = false;
        bool crashReportingDisabled = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          if (getEventName(message) != "program.initialized")
            return;
          eventFound = true;
          // The ostream exporter renders bool attributes numerically.
          crashReportingDisabled =
              message.contains("crash_reporting.enabled: 0");
        });

        EXPECT_TRUE(eventFound)
            << "program.initialized should still be emitted for adoption when "
               "only crash reporting is disabled";
        EXPECT_TRUE(crashReportingDisabled)
            << "expected crash_reporting.enabled=false so the failure-rate "
               "denominator excludes this session";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

TEST(Telemetry, DISABLED_LocalIDs) {
  auto origIDs = createLocalIDs();
  // The IDs are memoized: every caller in the process must observe identical
  // values, since the crash reporting and usage telemetry lanes rely on
  // sharing the same machineid/sessionid.
  auto newIDs = createLocalIDs();
  EXPECT_EQ(origIDs.machine, newIDs.machine);
  EXPECT_EQ(origIDs.session, newIDs.session);
}

// RAII utility for overriding an environment variable.
struct EnvSetter {
  EnvSetter(StringRef name, StringRef newValue)
      : name(name), oldValue(std::getenv(name.data())) {
    setenv(name.data(), newValue.data(), true);
  }

  ~EnvSetter() {
    if (oldValue)
      setenv(name.data(), oldValue, true);
    else
      unsetenv(name.data());
  }

private:
  StringRef name;
  const char *oldValue = nullptr;
};

TEST(Telemetry, ModularEmployee) {
  ErrorOr<TempDir> tempHome =
      TempDir::create("telemetry-modularemployee-%%%%%%%%");
  EXPECT_FALSE(tempHome.isError()) << tempHome.getError();
  EnvSetter homeOverride("HOME", tempHome->getPath().c_str());

  auto checkTelemetry = [&](bool expectedValue) {
    LogFileSetup logFileSetup("logs");
    TempFile tmpFile = logFileSetup.getLogFile("log", "1");
    Config settings(logFileSetup.getConfig());

    TelemetryContext ctx(settings);

    auto logger = ctx.getLogger("basic.log");
    logger->emitL0Event("test.ModularEmployee");
    ctx.flush();

    auto err = readFileUnderLock(
        tmpFile.getPath(), [&](const std::filesystem::path &path) {
          auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                    /*IsText=*/true);
          EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
          std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

          auto getLineStartingAt = [&](auto pos) {
            StringRef str = mbuf->getBuffer().substr(pos);
            return str.take_until([](char c) { return c == '\n'; });
          };

          bool eventFound = false;
          iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
            auto eventNamePos = message.find("event.name");
            StringRef eventNameLine = getLineStartingAt(eventNamePos);
            if (eventNameLine.split(':').second.trim() !=
                "test.ModularEmployee")
              return;

            eventFound = true;

            auto resourcePos = message.find("modular.employee");
            StringRef resourceLine = getLineStartingAt(resourcePos);
            EXPECT_EQ(resourceLine.split(':').second.trim(),
                      expectedValue ? StringRef("1") : StringRef("0"));
          });

          EXPECT_TRUE(eventFound) << "expected to find event in file";
        });
    EXPECT_FALSE(err.isError()) << err.getError();
  };

  checkTelemetry(false);

  // Create the marker file that signals Modular employee status.
  std::ofstream(tempHome->getPath() / ".modular-internal");

  checkTelemetry(true);
}

/// Verify that the OTLP request timeout constant is positive and within a
/// reasonable range (under 30 seconds).
TEST(Telemetry, OtlpRequestTimeout) {
  EXPECT_GT(kOtlpRequestTimeout.count(), 0);
  EXPECT_LE(kOtlpRequestTimeout.count(), 30000);
}

/// Verify that warnOnExportFailure emits a one-time warning to stderr with
/// the endpoint URL, stays silent on subsequent calls with the same flag,
/// and that independent flags produce independent warnings.
TEST(Telemetry, ExportFailureWarning) {
  // Each test gets its own warned flag (no process-global state).
  auto warned = std::make_shared<std::atomic<bool>>(false);

  // First call should produce a warning on stderr with the endpoint URL.
  testing::internal::CaptureStderr();
  warnOnExportFailure(warned, "https://telemetry.example.com:443");
  std::string firstStderr = testing::internal::GetCapturedStderr();
  EXPECT_THAT(firstStderr, testing::HasSubstr("telemetry export to"));
  EXPECT_THAT(firstStderr,
              testing::HasSubstr("https://telemetry.example.com:443"));
  EXPECT_THAT(firstStderr, testing::HasSubstr("MODULAR_TELEMETRY_ENABLED=0"));
  EXPECT_TRUE(warned->load());

  // Second call with the same flag should NOT produce another warning.
  testing::internal::CaptureStderr();
  warnOnExportFailure(warned, "https://telemetry.example.com:443");
  std::string secondStderr = testing::internal::GetCapturedStderr();
  EXPECT_EQ(secondStderr, "");

  // A fresh flag should produce a new warning (flags are independent).
  auto warned2 = std::make_shared<std::atomic<bool>>(false);
  testing::internal::CaptureStderr();
  warnOnExportFailure(warned2, "https://other.endpoint.com:443");
  std::string thirdStderr = testing::internal::GetCapturedStderr();
  EXPECT_THAT(thirdStderr,
              testing::HasSubstr("https://other.endpoint.com:443"));
}
