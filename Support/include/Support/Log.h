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

// MLOG is a macro that takes a variable number of arguments and logs them to
// a file or the console.

// If the single arg is a string, it is printed to the INFO level with a
// newline. MLOG(""); // "\n" is printed to the INFO level.
// MLOG("hello");
// "hello" is printed to the INFO level.
//
// Two args or more is a format string and a value, or values
// MLOG("{}", "hello"); // hello
// MLOG("{} {}", "hello", "world"); // hello world
// MLOG("{} {} {}", "hello", "world", "!"); // hello world!
//
// ...unless the first arg is a LogLevel, then everything above
// shifts left
// MLOG(LogLevel::DEBUG, "hi"); // "hi" printed at debug level
// MLOG(LogLevel::DEBUG, "{}", "hello"); // hello printed at debug level
// MLOG(LogLevel::DEBUG, "{} {}", "hello", 42); // hello 42
//
// MLOG_KV logs structured data instead of a formatted message. It takes a
// level followed by alternating key, value pairs, up to four pairs:
// MLOG_KV(LogLevel::INFO, "event", "span_start", "batch_id", 42);
// In JSON mode each pair becomes a top-level field; otherwise the pairs
// render as key=value tokens.
//
// The library goals are as follows:
// 1. Log in as few cycles as we can muster
// 2. Output messages reasonably quickly after they are logged
// 3. Reliably output all messages
//
// These are in order of importance, and design decisions have been taken
// reflecting this (e.g. messages are dropped on the floor if throughput
// isn't high enough).

#ifndef SUPPORT_LOG_H
#define SUPPORT_LOG_H

#include "LogChannels.h"
#include "MpscRingBuffer.h"

#include "llvm/ADT/SmallString.h"

#define FMT_EXCEPTIONS 0
#include <fmt/base.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>

namespace M {
class Config; // avoids including "Configuration.h"
} // namespace M

#define MLOG(...) M::Log::log(__VA_ARGS__)

#define MLOG_DEBUG(...) MLOG(::M::Log::LogLevel::DEBUG, __VA_ARGS__)
#define MLOG_INFO(...) MLOG(::M::Log::LogLevel::INFO, __VA_ARGS__)
#define MLOG_WARN(...) MLOG(::M::Log::LogLevel::WARN, __VA_ARGS__)
#define MLOG_ERROR(...) MLOG(::M::Log::LogLevel::ERROR, __VA_ARGS__)
#define MLOG_FATAL(...)                                                        \
  do {                                                                         \
    MLOG(::M::Log::LogLevel::FATAL, __VA_ARGS__);                              \
    getDefaultLog().flush();                                                   \
    std::abort();                                                              \
  } while (0)

#define MLOG_KV(level, ...)                                                    \
  do {                                                                         \
    const ::M::Log::LogLevel mlogKVLevel = (level);                            \
    if (::M::Log::getDefaultLog().getLogLevel() <= mlogKVLevel)                \
      ::M::Log::logKV(mlogKVLevel, __VA_ARGS__);                               \
  } while (0)

namespace M::Log {

enum class LogLevel : uint8_t {
  // Logs detailed debugging information for development and troubleshooting.
  DEBUG = 0,
  // Logs general informational messages about normal program operation.
  INFO = 1,
  // Logs warning messages about potential issues that don't prevent execution.
  WARN = 2,
  // Logs error messages about problems that affect functionality but allow
  // continuation.
  ERROR = 3,
  // Logs critical error messages that indicate severe problems requiring
  // immediate attention.
  FATAL = 4,
};

// Holds the pre-formatted data for a log event. The tag and the union should be
// kept in sync.
struct LogArg {
  enum class Type : uint8_t {
    Bool,
    Int64,
    UInt64,
    Fp32,
    Fp64,
    SmallString,
    String,
    Pointer
  };

  // Plain POD string view: pointer + length with a guaranteed, stable layout
  // that does not depend on any string_view-like implementation.
  struct StrView {
    const char *ptr;
    size_t len;
  };

  union {
    bool b;
    int64_t i64;
    uint64_t ui64;
    float fp32;
    double fp64;
    // for short strings, stored inline (16 chars max)
    std::array<char, sizeof(StrView)> ssoStr;
    StrView str{};
    const void *ptr;
  } data;
  Type tag;
};

namespace Detail {

template <typename T>
LogArg toLogArg(T &&val) {
  using D = std::decay_t<T>;
  using enum LogArg::Type;
  if constexpr (std::is_same_v<D, bool>) {
    return LogArg{.data = {.b = val}, .tag = Bool};
  } else if constexpr (std::is_integral_v<D> && std::is_signed_v<D> &&
                       sizeof(D) <= sizeof(int64_t)) {
    return LogArg{.data = {.i64 = static_cast<int64_t>(val)}, .tag = Int64};
  } else if constexpr (std::is_integral_v<D> && std::is_unsigned_v<D> &&
                       sizeof(D) <= sizeof(uint64_t)) {
    return LogArg{.data = {.ui64 = static_cast<uint64_t>(val)}, .tag = UInt64};
  } else if constexpr (std::is_same_v<D, float>) {
    return LogArg{.data = {.fp32 = val}, .tag = Fp32};
  } else if constexpr (std::is_same_v<D, double>) {
    return LogArg{.data = {.fp64 = val}, .tag = Fp64};
  } else if constexpr (std::is_convertible_v<D, std::string_view>) {
    std::string_view sv(val);
    if (sv.size() <= sizeof(LogArg::data.ssoStr)) {
      LogArg arg;
      arg.tag = LogArg::Type::SmallString;
      std::copy(sv.data(), sv.data() + sv.size(), arg.data.ssoStr.data());
      // Null-terminate only when there is room; a full 16-byte string is
      // identified by the absence of a null within the buffer.
      if (sv.size() < sizeof(LogArg::data.ssoStr))
        arg.data.ssoStr[sv.size()] = '\0';
      return arg;
    } else {
      return LogArg{.data = {.str = {sv.data(), sv.size()}}, .tag = String};
    }
  } else if constexpr (std::is_pointer_v<D>) {
    return LogArg{.data = {.ptr = val}, .tag = Pointer};
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported log argument type.");
  }
}

template <typename T, size_t I>
using is_sv_convertible =
    std::is_convertible<std::tuple_element_t<I, T>, std::string_view>;

template <typename Tuple, size_t... Is>
constexpr bool keysAreStringsImpl(std::index_sequence<Is...>) {
  return ((Is % 2 == 1 || is_sv_convertible<Tuple, Is>::value) && ...);
}

// Checks that the even-indexed arguments of an MLOG_KV pair list can be
// rendered as field names. Without this an int key compiles and silently
// produces an unusable JSON object key.
template <typename... Args>
constexpr bool keysAreStrings() {
  return keysAreStringsImpl<std::tuple<Args...>>(
      std::index_sequence_for<Args...>{});
}
} // namespace Detail

// Selects how a record's args are interpreted: positional arguments to
// fmtString, or alternating key, value pairs with no format string.
enum class RecordKind : uint8_t {
  Formatted,
  KeyValue,
};

struct LogRecord {
  constexpr static size_t maxArgs = 8;
  constexpr static size_t maxKVPairs = maxArgs / 2;
  using Timestamp = std::chrono::time_point<std::chrono::system_clock>;
  // Mojo FFI mirrors Timestamp as Int64. If this fires, the platform's
  // system_clock uses a different rep type and the Mojo bindings need
  // revisiting.
  static_assert(std::is_same_v<Timestamp::clock::duration::rep, int64_t>);
  // Four key-value pairs, since a pair occupies two arg slots.
  Timestamp timestamp;
  std::string_view fmtString;
  std::array<LogArg, maxArgs> args;
  uint8_t argCount;
  LogLevel level;
  RecordKind kind;
  Channel::Channels channel;

  // Disambiguates the key-value constructor, which shares the args array with
  // the formatted one but has no format string to key off.
  struct KeyValueTag {};

  template <typename... Args>
  LogRecord(Timestamp ts, LogLevel lvl, Channel::Channels c,
            std::string_view fmt, Args &&...args)
      : timestamp(ts), fmtString(fmt),
        args{Detail::toLogArg(std::forward<Args>(args))...},
        argCount(sizeof...(Args)), level(lvl), kind(RecordKind::Formatted),
        channel(c) {
    static_assert(sizeof...(Args) <= maxArgs, "Too many log arguments");
  }

  template <typename... Args>
  LogRecord(KeyValueTag, Timestamp ts, LogLevel lvl, Channel::Channels c,
            Args &&...args)
      : timestamp(ts), fmtString(""),
        args{Detail::toLogArg(std::forward<Args>(args))...},
        argCount(sizeof...(Args)), level(lvl), kind(RecordKind::KeyValue),
        channel(c) {
    static_assert(sizeof...(Args) <= maxArgs, "Too many log arguments");
  }

  // Constructs from a pre-built args array. Used by the C FFI shim
  // (LogFFI.cpp), which serializes typed arguments across the boundary itself.
  // The format string must have static duration (the Mojo API enforces this,
  // as the format string is a parameter).
  LogRecord(Timestamp ts, LogLevel lvl, Channel::Channels c,
            std::string_view fmt, std::array<LogArg, maxArgs> prebuiltArgs,
            uint8_t count)
      : timestamp(ts), fmtString(fmt), args(std::move(prebuiltArgs)),
        argCount(count), level(lvl), channel(c) {}

  // Required for ring-buffer slot pre-allocation.
  LogRecord() = default;
};

// Receives formatted log lines and writes to destinations (stdout, file &c).
class Sink;

class Logger {
  /// Number of slots in the logger's Ring Buffer.
  static constexpr size_t kRingCapacity = 1u << 12;
  /// How much char data can be stored by each log message.
  static constexpr size_t kStrBufPerSlot = 256;

  struct LogFormatState {
    bool useEnhancedFormat = true;
    bool showTimeStamp = true;
    bool showColors = true;
    bool showLogLevel = true;
    bool useIsoTimestamps = false;
    bool showMicroseconds = false;
    bool emitJSON = false;
    bool noShutdownSummary = false;
  } formatState;

  std::atomic<LogLevel> level = LogLevel::WARN;
  std::vector<std::unique_ptr<Sink>> sinks;
  ChannelState channelsEnabled;
  // Async ring buffer (Vyukov MPSC protocol). String-tagged LogArgs point into
  // strArena_; the slot's sequence number guarantees arena bytes live until
  // the consumer calls ring_.consume() after processing.
  M::MpscRingBuffer<LogRecord> ring{kRingCapacity};
  /// Defaults to 128 MiB.
  char strArena[kRingCapacity][kStrBufPerSlot];
  std::mutex drainMutex;
  std::condition_variable drainCv;
  std::atomic<bool> stopConsumer{false};
  std::thread consumer;
  std::atomic<uint64_t> droppedRecords{0};
  // Counts producers currently between claim() and publish(). The consumer
  // only exits when this is zero so it doesn't miss a slot whose sequence
  // number hasn't been written yet.
  std::atomic<size_t> inFlightEnqueues{0};

  llvm::SmallString<32> buildTimestampString(LogRecord::Timestamp);
  llvm::SmallString<128> buildLogPrefix(LogLevel, Channel::Channels,
                                        LogRecord::Timestamp);
  void run();
  bool enqueue(LogRecord record);
  void initFromConfig(Config cfg);
  void processRecord(const LogRecord &record);
  void flushSinks();

public:
  Logger();
  ~Logger();
  void log(LogRecord record);

  // Blocks until all records enqueued before this call have been written to
  // all sinks. Primarily useful in tests before inspecting sink output.
  void flush();

  void enableChannel(Channel::Channels c) { channelsEnabled.enable(c); }
  void disableChannel(Channel::Channels c) { channelsEnabled.disable(c); }
  void enableAllChannels() { channelsEnabled.enableAll(); }
  void disableAllChannels() { channelsEnabled.disableAll(); }

  bool isEnabled(Channel::Channels c) const {
    return channelsEnabled.isEnabled(c);
  }

  LogLevel getLogLevel() const {
    return level.load(std::memory_order::acquire);
  }

  void setLogLevel(LogLevel newLevel) {
    level.store(newLevel, std::memory_order::release);
  }

  /// Total records successfully written to sinks (monotonically increasing).
  size_t processedCount() const { return ring.consumeCount(); }

  /// Total records dropped because the ring was full (monotonically
  /// increasing).
  uint64_t droppedCount() const {
    return droppedRecords.load(std::memory_order_relaxed);
  }
};

inline Logger &getDefaultLog() {
  static Logger defaultLog;
  return defaultLog;
}

inline void setLogLevel(LogLevel level) { getDefaultLog().setLogLevel(level); }

inline void enableChannel(Channel::Channels c) {
  getDefaultLog().enableChannel(c);
}

inline void disableChannel(Channel::Channels c) {
  getDefaultLog().disableChannel(c);
}

inline void enableAllChannels() { getDefaultLog().enableAllChannels(); }

inline void disableAllChannels() { getDefaultLog().disableAllChannels(); }

inline void logWrite(Logger &log, LogRecord record) {
  log.log(std::move(record));
}

// Checks the log level to see if it should emit a message, and, if so,
// dispatches to the Logger object.
template <typename... Args>
inline void logWriteDispatch(Logger &log, LogLevel level,
                             Channel::Channels channel,
                             fmt::format_string<Args...> fmt, Args &&...args) {
  if (log.getLogLevel() > level || !log.isEnabled(channel))
    return;

  LogRecord record(std::chrono::system_clock::now(), level, channel,
                   {fmt.get().data(), fmt.get().size()},
                   std::forward<Args>(args)...);
  logWrite(log, std::move(record));
}

// Same shape as logWriteDispatch, but the args are alternating key, value
// pairs rather than positional arguments to a format string.
template <typename... Args>
inline void logKVDispatch(Logger &log, LogLevel level,
                          Channel::Channels channel, Args &&...args) {
  static_assert(sizeof...(Args) >= 2,
                "MLOG_KV needs at least one key, value pair");
  static_assert(sizeof...(Args) % 2 == 0,
                "MLOG_KV takes alternating key, value pairs");
  static_assert(sizeof...(Args) <= LogRecord::maxKVPairs * 2,
                "MLOG_KV takes at most four key, value pairs");
  static_assert(Detail::keysAreStrings<Args...>(),
                "MLOG_KV keys must be convertible to std::string_view");

  if (log.getLogLevel() > level || !log.isEnabled(channel))
    return;

  LogRecord record(LogRecord::KeyValueTag{}, std::chrono::system_clock::now(),
                   level, channel, std::forward<Args>(args)...);
  logWrite(log, std::move(record));
}

template <typename... Args>
void logKV(LogLevel level, Args &&...args) {
  logKVDispatch(getDefaultLog(), level, Channel::Default,
                std::forward<Args>(args)...);
}

template <typename... Args>
void log(fmt::format_string<Args...> fmt, Args &&...args) {
  logWriteDispatch(getDefaultLog(), LogLevel::INFO, Channel::Default, fmt,
                   std::forward<Args>(args)...);
}

template <typename... Args>
void log(LogLevel level, fmt::format_string<Args...> fmt, Args &&...args) {
  logWriteDispatch(getDefaultLog(), level, Channel::Default, fmt,
                   std::forward<Args>(args)...);
}

template <typename... Args>
void log(Channel::Channels channel, fmt::format_string<Args...> fmt,
         Args &&...args) {
  logWriteDispatch(getDefaultLog(), LogLevel::INFO, channel, fmt,
                   std::forward<Args>(args)...);
}

template <typename... Args>
void log(LogLevel level, Channel::Channels channel,
         fmt::format_string<Args...> fmt, Args &&...args) {
  logWriteDispatch(getDefaultLog(), level, channel, fmt,
                   std::forward<Args>(args)...);
}

} // namespace M::Log

#endif // SUPPORT_LOG_H
