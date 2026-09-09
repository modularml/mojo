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

#include "Support/Log.h"
#include "Support/Configuration.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <fmt/args.h>
#include <fmt/chrono.h>
#include <fmt/format.h>

#include "Support/Threading/SpinWaiter.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>

namespace M::Log {

namespace ConfigEntry {
static constexpr llvm::StringLiteral LOG_STDOUT = "log.stdout";
static constexpr llvm::StringLiteral LOG_FILE = "log.file";
static constexpr llvm::StringLiteral LOG_ISO_TIME = "log.iso_time";
static constexpr llvm::StringLiteral LOG_LEVEL = "log.level";
static constexpr llvm::StringLiteral LOG_MICROSECONDS = "log.microseconds";
static constexpr llvm::StringLiteral LOG_NO_ENHANCED = "log.no_enhanced";
static constexpr llvm::StringLiteral LOG_NO_TIMESTAMP = "log.no_timestamp";
static constexpr llvm::StringLiteral LOG_JSON = "log.json";
static constexpr llvm::StringLiteral LOG_CHANNELS = "log.enabled_channels";
static constexpr llvm::StringLiteral LOG_NO_SUMMARY = "log.no_summary";
} // namespace ConfigEntry

static LogLevel parseLogLevelFromString(llvm::StringRef levelStr) {
  if (levelStr == "0")
    return LogLevel::DEBUG;
  if (levelStr == "1")
    return LogLevel::INFO;
  if (levelStr == "2")
    return LogLevel::WARN;
  if (levelStr == "3")
    return LogLevel::ERROR;
  if (levelStr == "4")
    return LogLevel::FATAL;
  if (levelStr.equals_insensitive("DEBUG"))
    return LogLevel::DEBUG;
  if (levelStr.equals_insensitive("INFO"))
    return LogLevel::INFO;
  if (levelStr.equals_insensitive("WARN"))
    return LogLevel::WARN;
  if (levelStr.equals_insensitive("ERROR"))
    return LogLevel::ERROR;
  if (levelStr.equals_insensitive("FATAL"))
    return LogLevel::FATAL;

  return LogLevel::WARN;
}

static llvm::raw_ostream::Colors getLogLevelColor(LogLevel level) {
  using enum llvm::raw_ostream::Colors;
  switch (level) {
  case LogLevel::DEBUG:
    return BRIGHT_BLACK;
  case LogLevel::INFO:
    return BRIGHT_CYAN;
  case LogLevel::WARN:
    return BRIGHT_YELLOW;
  case LogLevel::ERROR:
    return BRIGHT_RED;
  case LogLevel::FATAL:
    return RED;
  }
}

static llvm::StringRef getChannelName(Channel::Channels c) {
  switch (c) {
#define CHANNEL_CASE(ch, cfgName)                                              \
  case Channel::ch:                                                            \
    return cfgName;
    MLOG_CHANNELS(CHANNEL_CASE)
#undef CHANNEL_CASE
  case Channel::NChannels:
    break;
  }
  llvm_unreachable("invalid Channel::Channels value");
}

static llvm::StringLiteral getLogLevelPrefix(LogLevel level) {
  switch (level) {
  case LogLevel::DEBUG:
    return " DBG";
  case LogLevel::INFO:
    return "INFO";
  case LogLevel::WARN:
    return "WARN";
  case LogLevel::ERROR:
    return " ERR";
  case LogLevel::FATAL:
    return "FATL";
  }
}

// Returns us in string form with a leading '.' and zero-padded to six digits
static llvm::SmallString<8> formatMicroseconds(LogRecord::Timestamp timePoint) {
  llvm::SmallString<8> result;
  // Only count microseconds by modulo'ing by 1 million
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                    timePoint.time_since_epoch()) %
                1'000'000;
  // us are 6 digits long, plus the single '.'
  result.resize(7, 0);
  fmt::format_to_n(result.data(), result.size(), ".{:06}", micros.count());
  return result;
}

// Returns a full ISO 8601 UTC timestamp, e.g. "2026-12-25T12:00:00.123456Z".
// The result is always 20 chars (without microseconds) or 27 chars (with).
// SmallString<32> is sized to hold either with room to spare.
static llvm::SmallString<32> buildISOFormatString(LogRecord::Timestamp now,
                                                  bool includeMicroseconds) {
  std::tm utc = fmt::gmtime(std::chrono::system_clock::to_time_t(now));
  constexpr size_t nChars = 32;
  llvm::SmallString<nChars> result;
  // Pre-size the buffer before format_to_n. Sizing after would call append()
  // from size 0, which overwrites the just-formatted data with zeros.
  result.resize(nChars, '\0');
  auto fmtResult =
      fmt::format_to_n(result.data(), nChars, "{:%Y-%m-%dT%H:%M:%S}", utc);
  result.resize(fmtResult.size);
  if (includeMicroseconds)
    result += formatMicroseconds(now);
  result += "Z";
  return result;
}

static bool isStringArg(const LogArg &arg) {
  return arg.tag == LogArg::Type::SmallString ||
         arg.tag == LogArg::Type::String;
}

// Valid only for SmallString- and String-tagged args. A SmallString that fills
// the buffer has no null terminator, hence strnlen over strlen.
static std::string_view argAsStringView(const LogArg &arg) {
  if (arg.tag == LogArg::Type::SmallString)
    return {arg.data.ssoStr.data(),
            strnlen(arg.data.ssoStr.data(), arg.data.ssoStr.size())};
  return {arg.data.str.ptr, arg.data.str.len};
}

static llvm::json::Value argAsJSON(const LogArg &arg) {
  switch (arg.tag) {
  case LogArg::Type::Bool:
    return arg.data.b;
  case LogArg::Type::Int64:
    return arg.data.i64;
  case LogArg::Type::UInt64:
    return arg.data.ui64;
  case LogArg::Type::Fp32:
    return static_cast<double>(arg.data.fp32);
  case LogArg::Type::Fp64:
    return arg.data.fp64;
  case LogArg::Type::SmallString:
  case LogArg::Type::String: {
    auto sv = argAsStringView(arg);
    return llvm::StringRef(sv.data(), sv.size());
  }
  case LogArg::Type::Pointer:
    return fmt::format("{}", fmt::ptr(arg.data.ptr));
  }
  llvm_unreachable("invalid LogArg::Type value");
}

static void appendArgText(std::string &out, const LogArg &arg) {
  auto append = [&out](auto &&value) {
    fmt::format_to(std::back_inserter(out), "{}", value);
  };
  switch (arg.tag) {
  case LogArg::Type::Bool:
    return append(arg.data.b);
  case LogArg::Type::Int64:
    return append(arg.data.i64);
  case LogArg::Type::UInt64:
    return append(arg.data.ui64);
  case LogArg::Type::Fp32:
    return append(arg.data.fp32);
  case LogArg::Type::Fp64:
    return append(arg.data.fp64);
  case LogArg::Type::SmallString:
  case LogArg::Type::String: {
    auto sv = argAsStringView(arg);
    out.append(sv.data(), sv.size());
    return;
  }
  case LogArg::Type::Pointer:
    return append(fmt::ptr(arg.data.ptr));
  }
  llvm_unreachable("invalid LogArg::Type value");
}

// Renders a key-value record as space-separated key=value tokens, the
// non-JSON counterpart to buildJSONKVLogLine.
static std::string renderKVRecord(const LogRecord &record) {
  std::string out;
  for (size_t i = 0; i + 1 < record.argCount; i += 2) {
    // MLOG_KV static-asserts that keys are strings, but a hand-built record
    // could carry something else, and reading an int64 as a pointer would
    // fault on the consumer thread. Drop the pair rather than risk that.
    if (!isStringArg(record.args[i]))
      continue;
    if (!out.empty())
      out += ' ';
    auto key = argAsStringView(record.args[i]);
    out.append(key.data(), key.size());
    out += '=';
    appendArgText(out, record.args[i + 1]);
  }
  return out;
}

static llvm::SmallString<512> buildJSONLogLine(LogLevel level,
                                               Channel::Channels channel,
                                               llvm::StringRef msg,
                                               LogRecord::Timestamp ts) {
  // Always use full ISO 8601 with microseconds in JSON mode regardless of
  // the showTimeStamp / useIsoTimestamps / showMicroseconds config flags.
  auto timestamp = buildISOFormatString(ts, /*includeMicroseconds=*/true);

  llvm::SmallString<512> jsonLogLine;
  llvm::raw_svector_ostream svOstream(jsonLogLine);
  llvm::json::OStream json(svOstream);
  json.object([&] {
    json.attribute("timestamp", timestamp);
    json.attribute("level", getLogLevelPrefix(level).trim());
    json.attribute("channel", getChannelName(channel));
    json.attribute("message", msg);
  });
  return jsonLogLine;
}

// Same envelope as buildJSONLogLine, but the record's pairs are promoted to
// top-level fields in place of "message" so they land as Datadog facets. A key
// colliding with an envelope field emits a duplicate JSON key; keep keys clear
// of "timestamp", "level" and "channel".
static llvm::SmallString<512> buildJSONKVLogLine(const LogRecord &record) {
  auto timestamp =
      buildISOFormatString(record.timestamp, /*includeMicroseconds=*/true);

  llvm::SmallString<512> jsonLogLine;
  llvm::raw_svector_ostream svOstream(jsonLogLine);
  llvm::json::OStream json(svOstream);
  json.object([&] {
    json.attribute("timestamp", timestamp);
    json.attribute("level", getLogLevelPrefix(record.level).trim());
    json.attribute("channel", getChannelName(record.channel));
    for (size_t i = 0; i + 1 < record.argCount; i += 2) {
      if (!isStringArg(record.args[i]))
        continue;
      auto key = argAsStringView(record.args[i]);
      json.attribute(llvm::StringRef(key.data(), key.size()),
                     argAsJSON(record.args[i + 1]));
    }
  });
  return jsonLogLine;
}

static std::string renderRecord(const LogRecord &record) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.reserve(record.argCount, /*new_cap_named=*/0);
  for (const auto &arg : std::span(record.args.data(), record.argCount)) {
    switch (arg.tag) {
    case LogArg::Type::Bool:
      store.push_back(arg.data.b);
      break;
    case LogArg::Type::Int64:
      store.push_back(arg.data.i64);
      break;
    case LogArg::Type::UInt64:
      store.push_back(arg.data.ui64);
      break;
    case LogArg::Type::Fp32:
      store.push_back(arg.data.fp32);
      break;
    case LogArg::Type::Fp64:
      store.push_back(arg.data.fp64);
      break;
    case LogArg::Type::SmallString:
      store.push_back(fmt::string_view(
          arg.data.ssoStr.data(),
          strnlen(arg.data.ssoStr.data(), arg.data.ssoStr.size())));
      break;
    case LogArg::Type::String:
      store.push_back(fmt::string_view(arg.data.str.ptr, arg.data.str.len));
      break;
    case LogArg::Type::Pointer:
      store.push_back(fmt::ptr(arg.data.ptr));
      break;
    }
  }
  return fmt::vformat(record.fmtString, store);
}

class Sink {
public:
  virtual ~Sink() = default;
  virtual void write(llvm::StringRef msg) = 0;
  // Called by the consumer thread when the ring drains and before shutdown,
  // and by Logger::flush() from arbitrary caller threads — implementations
  // that buffer must synchronize flush() against concurrent write().
  // Default is a no-op for sinks that don't buffer.
  virtual void flush() {}
};

class FileSink : public Sink {
  std::error_code ec;
  llvm::raw_fd_ostream ostream;
  std::mutex outputMutex;

  FileSink(llvm::StringRef path)
      : ostream{path, ec, llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::FA_Write,
                llvm::sys::fs::OF_Append} {}

public:
  static llvm::ErrorOr<std::unique_ptr<FileSink>> create(llvm::StringRef path) {
    auto fileSink = std::unique_ptr<FileSink>(new FileSink(path));
    if (fileSink->ec)
      return fileSink->ec;
    return fileSink;
  }

  void write(llvm::StringRef msg) override {
    std::lock_guard<std::mutex> lock(outputMutex);
    ostream.write(msg.begin(), msg.size());
    ostream.write('\n');
  }

  // Locked so Logger::flush() callers can flush concurrently with the
  // consumer thread's write() — raw_fd_ostream is not internally
  // synchronized.
  void flush() override {
    std::lock_guard<std::mutex> lock(outputMutex);
    ostream.flush();
  }
};

class StdoutSink : public Sink {
  std::mutex outputMutex;

  // Note: outputMutex only serializes accesses made through this sink.
  // llvm::outs() is a process-global stream, so third-party writers that
  // use it directly are outside this synchronization.
  void write(llvm::StringRef msg) override {
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::outs() << msg << "\n";
  }

  void flush() override {
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::outs().flush();
  }
};

void Logger::initFromConfig(Config cfg) {
  using namespace ConfigEntry;
  formatState.useEnhancedFormat = !cfg.getValueAsBool(LOG_NO_ENHANCED, false);
  formatState.showTimeStamp = !cfg.getValueAsBool(LOG_NO_TIMESTAMP, false);
  formatState.useIsoTimestamps = cfg.getValueAsBool(LOG_ISO_TIME, false);
  formatState.showMicroseconds = cfg.getValueAsBool(LOG_MICROSECONDS, false);
  formatState.emitJSON = cfg.getValueAsBool(LOG_JSON, false);
  formatState.noShutdownSummary = cfg.getValueAsBool(LOG_NO_SUMMARY, false);
  auto logToStdout = cfg.getValueAsBool(LOG_STDOUT, true);

  this->setLogLevel(
      parseLogLevelFromString(cfg.getValueOr(ConfigEntry::LOG_LEVEL, "WARN")));

  // If stdout logging is requested or if no log file present, log to stdout.
  // The stdout variable is default-true, but can be overridden.
  auto logFilePath = cfg.getValueOr(ConfigEntry::LOG_FILE, "");
  if (logToStdout)
    sinks.push_back(std::make_unique<StdoutSink>());
  if (!logFilePath.empty()) {
    auto fileSinkOrErr = FileSink::create(logFilePath);
    if (fileSinkOrErr) {
      sinks.push_back(std::move(*fileSinkOrErr));
    } else {
      llvm::errs() << "Failed to open log file '" << logFilePath
                   << "': " << fileSinkOrErr.getError().message()
                   << (logToStdout ? "\nLog messages only going to stdout.\n"
                                   : "\nNo log messages will be emitted.\n");
    }
  }
  auto channelsList = cfg.getValueOr(LOG_CHANNELS, "");
  llvm::SmallVector<llvm::StringRef> names;
  llvm::StringRef(channelsList).split(names, ';');
  for (auto name : names) {
    name = name.trim();
    if (name.empty())
      continue;
    if (name == "all") {
      channelsEnabled.enableAll();
      continue;
    }
#define MATCH(channel, cfgName)                                                \
  if (name == cfgName) {                                                       \
    channelsEnabled.enable(Channel::channel);                                  \
    continue;                                                                  \
  }
    MLOG_CHANNELS(MATCH)
#undef MATCH
    llvm::errs() << "Unknown channel name \"" << name
                 << "\" in configuration file parsing\n";
  }
}

Logger::Logger() {
  // Respect the standard NO_COLOR env var, any value (even empty) disables
  // color - does not depend on config object.
  formatState.showColors = !llvm::sys::Process::GetEnv("NO_COLOR").has_value();

  // Check if the terminal does not support colors.
  if (formatState.showColors) {
    auto term = llvm::sys::Process::GetEnv("TERM");
    if ((term && *term == "dumb") ||
        !llvm::sys::Process::StandardOutIsDisplayed())
      formatState.showColors = false;
  }

  auto cfgOr = Config::open();
  if (cfgOr.isError()) {
    // If we can't read the config, the default is to log to stdout.
    sinks.push_back(std::make_unique<StdoutSink>());
  } else {
    initFromConfig(cfgOr.takeValue());
  }

  // Must run after all other optional state has been initialised.
  consumer = std::thread(&Logger::run, this);
}

Logger::~Logger() {
  stopConsumer.store(true, std::memory_order_release);
  drainCv.notify_all();
  consumer.join();

  auto written = ring.consumeCount();
  size_t dropped = droppedRecords.load(std::memory_order_relaxed);
  // Only print when the logger was actually used, and allow opt-out.
  // std::printf (C stdout) is safe here; llvm::outs() and file sinks may
  // already be torn down at static-destructor time.
  if ((written > 0 || dropped > 0) && !formatState.noShutdownSummary)
    std::printf("[Logger] shutdown: %zu records written, %zu dropped\n",
                written, dropped);
}

void Logger::log(LogRecord record) { enqueue(std::move(record)); }

bool Logger::enqueue(LogRecord record) {
  inFlightEnqueues.fetch_add(1, std::memory_order_relaxed);
  auto posOpt = ring.claim();
  if (!posOpt) {
    inFlightEnqueues.fetch_sub(1, std::memory_order_release);
    droppedRecords.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  size_t pos = *posOpt;

  // Copy String-tagged args into the slot's arena so the consumer thread can
  // safely read them after the caller's stack is gone. If the arena fills up,
  // strings are clipped to the remaining space rather than left dangling.
  char *arena = strArena[pos & ring.getMask()];
  size_t arenaUsed = 0;
  for (size_t i = 0; i < record.argCount; ++i) {
    auto &arg = record.args[i];
    if (arg.tag != LogArg::Type::String)
      continue;
    auto &sv = arg.data.str;
    size_t toCopy = std::min(sv.len, kStrBufPerSlot - arenaUsed);
    if (toCopy > 0) {
      std::memcpy(arena + arenaUsed, sv.ptr, toCopy);
      sv.ptr = arena + arenaUsed;
      sv.len = toCopy; // may be clipped if arena was nearly full
      arenaUsed += toCopy;
    } else {
      // Arena exhausted; render as empty string rather than dangle a ptr.
      sv = {"", 0};
    }
  }

  ring.itemAt(pos) = std::move(record);
  ring.publish(pos);
  inFlightEnqueues.fetch_sub(1, std::memory_order_release);
  return true;
}

void Logger::flushSinks() {
  for (const auto &sink : sinks)
    sink->flush();
}

void Logger::run() {
  while (true) {
    if (LogRecord *item = ring.peek()) {
      // Process while holding the slot so strArena_ bytes remain valid.
      processRecord(*item);
      ring.consume();
    } else {
      if (stopConsumer.load(std::memory_order_acquire) &&
          inFlightEnqueues.load(std::memory_order_acquire) == 0 &&
          ring.enqueueCount() <= ring.consumeCount())
        break;
      // Flush buffered sink writes to the OS before sleeping — cheaper than
      // flushing after every record, and ensures output appears promptly when
      // the ring drains.
      flushSinks();
      std::unique_lock<std::mutex> lock(drainMutex);
      // The consumer polls every 100 us to drain the ring buffer. Messages
      // will therefore wait up to this long to be seen in the log. Flushing
      // will force the thread to wake up and drain the buffer as normal.
      drainCv.wait_for(lock, std::chrono::microseconds(100));
    }
  }
  flushSinks();
}

void Logger::flush() {
  if (!consumer.joinable())
    return;
  size_t target = ring.enqueueCount();
  drainCv.notify_one();
  SpinWaiter<> waiter;
  while (ring.consumeCount() < target)
    waiter.wait();
  // The consume-count wait only guarantees the records were rendered into
  // the sinks' userspace buffers; the consumer pushes those buffers to the
  // OS on its next idle iteration, so a caller reading the log file right
  // after flush() could still miss the data. Flush the sinks here so the
  // documented "written to all sinks" contract holds when we return. Safe
  // from this thread: each sink's flush() takes the same mutex as its
  // write().
  flushSinks();
}

void Logger::processRecord(const LogRecord &record) {
  bool isKV = record.kind == RecordKind::KeyValue;
  auto level = record.level;
  llvm::SmallString<512> enhancedOrJSONMsg;
  if (formatState.emitJSON) {
    enhancedOrJSONMsg =
        isKV ? buildJSONKVLogLine(record)
             : buildJSONLogLine(level, record.channel, renderRecord(record),
                                record.timestamp);
  } else {
    auto msg = isKV ? renderKVRecord(record) : renderRecord(record);
    if (formatState.useEnhancedFormat) {
      enhancedOrJSONMsg =
          buildLogPrefix(level, record.channel, record.timestamp);
      enhancedOrJSONMsg += msg;
    } else {
      enhancedOrJSONMsg = msg;
    }
  }
  for (const auto &sink : sinks)
    sink->write(enhancedOrJSONMsg);
}

// Depends on logger state to access formatting options
// TODO(dmcbain): change how format options are used to move this out-of-line
llvm::SmallString<32> Logger::buildTimestampString(LogRecord::Timestamp ts) {
  llvm::SmallString<32> result;
  if (!formatState.showTimeStamp)
    return result;

  if (formatState.useIsoTimestamps)
    return buildISOFormatString(ts, formatState.showMicroseconds);

  // Simple format: 16:21:14
  std::tm utc = fmt::gmtime(std::chrono::system_clock::to_time_t(ts));
  constexpr size_t nChars = 32;
  llvm::SmallString<nChars> time;
  // Pre-size the buffer before format_to_n. Sizing after would call append()
  // from size 0, which overwrites the just-formatted data with zeros.
  time.resize(nChars, '\0');
  auto formatResult = fmt::format_to_n(time.data(), nChars, "{:%H:%M:%S}", utc);
  time.resize(formatResult.size);
  result += time;
  if (formatState.showMicroseconds)
    result += formatMicroseconds(ts);
  return result;
}

// TODO(dmcbain): change how format options are used to move this out-of-line
llvm::SmallString<128> Logger::buildLogPrefix(LogLevel level,
                                              Channel::Channels channel,
                                              LogRecord::Timestamp ts) {
  using enum llvm::raw_ostream::Colors;
  llvm::SmallString<128> prefix;

  if (formatState.showTimeStamp) {
    prefix += "[";
    prefix += buildTimestampString(ts);
    prefix += "] ";
  }

  if (formatState.showLogLevel) {
    if (formatState.showColors)
      prefix += llvm::sys::Process::OutputColor(
          static_cast<char>(getLogLevelColor(level)), /*bold=*/false,
          /*bg=*/false);
    prefix += "[";
    prefix += getLogLevelPrefix(level);
    prefix += "] ";
    if (formatState.showColors)
      prefix += llvm::sys::Process::ResetColor();
  }

  prefix += "[";
  prefix += getChannelName(channel);
  prefix += "] ";

  return prefix;
}

} // namespace M::Log
