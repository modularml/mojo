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

#include <cstdlib>
#include <fstream>
#include <regex>
#include <string>

#ifndef _WIN32
#include <stdlib.h>
#endif

#include "gtest/gtest.h"

#include "llvm/Support/FileSystem.h"

using namespace M::Log;

namespace {

static std::string gLogFilePath;

// Configures the logging environment before any test runs. This is critical
// because writeStringToLogFileOrStdout and getLogFormatState both use
// one-time statics that capture env vars on first call. Setting them here
// guarantees our configuration is in effect when those statics initialize.
class LogTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    // Unset MODULAR_LOG_LEVEL so initLogLevel() defaults to INFO.
#ifndef _WIN32
    unsetenv("MODULAR_LOG_LEVEL");
#else
#error "Test requires modification for Windows"
#endif
    // Route all log output to a temp file so tests can inspect it.
    char tmpPath[] = "/tmp/modular-log-test-%%%%%%";
    int fd = 0;
    llvm::SmallString<128> realPath;
    auto ec = llvm::sys::fs::createUniqueFile(tmpPath, fd, realPath);
    ASSERT_FALSE(ec) << "Failed to create temp log file: " << ec.message();
    gLogFilePath = realPath.str().str();
#ifndef _WIN32
    setenv("MODULAR_LOG_FILE", gLogFilePath.c_str(), /*overwrite=*/1);
#else
#error "Test requires modification for Windows"
#endif
    // Runs after writing the config file, as getDefaultLog() initializes
    // the Logger class.
    auto &log = getDefaultLog();
    (void)log;
  }

  void TearDown() override { llvm::sys::fs::remove(gLogFilePath); }
};

// Register before main() so SetUp() runs before any test case.
static ::testing::Environment *const kLogEnv =
    ::testing::AddGlobalTestEnvironment(new LogTestEnvironment);

std::streampos currentLogEnd() {
  std::ifstream f(gLogFilePath);
  f.seekg(0, std::ios::end);
  return f.tellg();
}

std::string readLogSince(std::streampos offset) {
  std::ifstream f(gLogFilePath);
  f.seekg(offset);
  return {std::istreambuf_iterator<char>(f), {}};
}

class LogLevelTest : public ::testing::Test {
protected:
  void SetUp() override { setLogLevel(LogLevel::INFO); }
  void TearDown() override { setLogLevel(LogLevel::INFO); }
};

// Extends LogLevelTest with per-test output capture via file position tracking.
class LogOutputTest : public LogLevelTest {
protected:
  void SetUp() override {
    LogLevelTest::SetUp();
    startPos_ = currentLogEnd();
  }

  std::string capturedOutput() const {
    getDefaultLog().flush();
    return readLogSince(startPos_);
  }

private:
  std::streampos startPos_{};
};

TEST_F(LogLevelTest, SetGetRoundTrip) {
  for (auto level : {LogLevel::DEBUG, LogLevel::INFO, LogLevel::WARN,
                     LogLevel::ERROR, LogLevel::FATAL}) {
    setLogLevel(level);
    EXPECT_EQ(M::Log::getDefaultLog().getLogLevel(), level);
  }
}

TEST_F(LogLevelTest, LevelsAreOrdered) {
  EXPECT_LT(LogLevel::DEBUG, LogLevel::INFO);
  EXPECT_LT(LogLevel::INFO, LogLevel::WARN);
  EXPECT_LT(LogLevel::WARN, LogLevel::ERROR);
  EXPECT_LT(LogLevel::ERROR, LogLevel::FATAL);
}

TEST_F(LogOutputTest, MessageAtCurrentLevelIsWritten) {
  setLogLevel(LogLevel::WARN);
  MLOG(LogLevel::WARN, "at-level message");
  EXPECT_NE(capturedOutput().find("at-level message"), std::string::npos);
}

TEST_F(LogOutputTest, MessagesAboveCurrentLevelAreWritten) {
  setLogLevel(LogLevel::WARN);
  MLOG(LogLevel::ERROR, "error message");
  MLOG(LogLevel::FATAL, "fatal message");
  auto out = capturedOutput();
  EXPECT_NE(out.find("error message"), std::string::npos);
  EXPECT_NE(out.find("fatal message"), std::string::npos);
}

TEST_F(LogOutputTest, MessagesBelowCurrentLevelAreFiltered) {
  setLogLevel(LogLevel::WARN);
  MLOG(LogLevel::DEBUG, "debug message");
  MLOG(LogLevel::INFO, "info message");
  EXPECT_TRUE(capturedOutput().empty());
}

TEST_F(LogOutputTest, DebugLevelPassesAllMessages) {
  setLogLevel(LogLevel::DEBUG);
  MLOG(LogLevel::DEBUG, "debug msg");
  MLOG(LogLevel::INFO, "info msg");
  MLOG(LogLevel::WARN, "warn msg");
  MLOG(LogLevel::ERROR, "error msg");
  MLOG(LogLevel::FATAL, "fatal msg");
  auto out = capturedOutput();
  EXPECT_NE(out.find("debug msg"), std::string::npos);
  EXPECT_NE(out.find("info msg"), std::string::npos);
  EXPECT_NE(out.find("warn msg"), std::string::npos);
  EXPECT_NE(out.find("error msg"), std::string::npos);
  EXPECT_NE(out.find("fatal msg"), std::string::npos);
}

TEST_F(LogOutputTest, FatalLevelFiltersAllOtherLevels) {
  setLogLevel(LogLevel::FATAL);
  MLOG(LogLevel::DEBUG, "debug msg");
  MLOG(LogLevel::INFO, "info msg");
  MLOG(LogLevel::WARN, "warn msg");
  MLOG(LogLevel::ERROR, "error msg");
  EXPECT_TRUE(capturedOutput().empty());
}

TEST_F(LogOutputTest, LevelPrefixDebug) {
  setLogLevel(LogLevel::DEBUG);
  MLOG(LogLevel::DEBUG, "msg");
  EXPECT_NE(capturedOutput().find("[ DBG]"), std::string::npos);
}

TEST_F(LogOutputTest, LevelPrefixInfo) {
  MLOG(LogLevel::INFO, "msg");
  EXPECT_NE(capturedOutput().find("[INFO]"), std::string::npos);
}

TEST_F(LogOutputTest, LevelPrefixWarn) {
  MLOG(LogLevel::WARN, "msg");
  EXPECT_NE(capturedOutput().find("[WARN]"), std::string::npos);
}

TEST_F(LogOutputTest, LevelPrefixError) {
  MLOG(LogLevel::ERROR, "msg");
  EXPECT_NE(capturedOutput().find("[ ERR]"), std::string::npos);
}

TEST_F(LogOutputTest, ChannelPrefixDefault) {
  MLOG(LogLevel::INFO, "msg");
  EXPECT_NE(capturedOutput().find("[default]"), std::string::npos);
}

TEST_F(LogOutputTest, ChannelPrefixNonDefault) {
  enableChannel(Channel::Mojo);
  log(Channel::Mojo, "msg");
  disableChannel(Channel::Mojo);
  EXPECT_NE(capturedOutput().find("[mojo]"), std::string::npos);
}

// Disabled: FATAL aborts the process
// TEST_F(LogOutputTest, LevelPrefixFatal) {
//  MLOG(LogLevel::FATAL, "msg");
//  EXPECT_NE(capturedOutput().find("[FATL]"), std::string::npos);
// }

// flush() must make records visible to an independent reader of the log
// file before it returns — push the sinks' userspace buffers to the OS,
// not just wait for the ring to drain. The race window is microseconds
// wide, so loop to make a regression likely to be caught.
TEST_F(LogOutputTest, FlushMakesRecordVisibleToReader) {
  for (int i = 0; i < 500; ++i) {
    auto pos = currentLogEnd();
    MLOG(LogLevel::INFO, "flush-visibility {}", i);
    getDefaultLog().flush();
    auto out = readLogSince(pos);
    ASSERT_NE(out.find("flush-visibility " + std::to_string(i)),
              std::string::npos)
        << "record not visible after flush() on iteration " << i;
  }
}

TEST_F(LogOutputTest, TimestampMatchesSimpleFormat) {
  MLOG(LogLevel::INFO, "ts test");
  // Default format: [HH:MM:SS] [INFO] ts test
  std::regex tsPattern(R"(\[\d{2}:\d{2}:\d{2}\])");
  EXPECT_TRUE(std::regex_search(capturedOutput(), tsPattern));
}

// ── Primitive Type Formatting ──────────────────────────────────────────────

TEST_F(LogOutputTest, LogsBool) {
  MLOG("{}", true);
  EXPECT_NE(capturedOutput().find("true"), std::string::npos);
}

TEST_F(LogOutputTest, LogsInt) {
  MLOG("{}", 42);
  EXPECT_NE(capturedOutput().find("42"), std::string::npos);
}

TEST_F(LogOutputTest, LogsNegativeInt) {
  MLOG("{}", -7);
  EXPECT_NE(capturedOutput().find("-7"), std::string::npos);
}

TEST_F(LogOutputTest, LogsFloat) {
  // 3.14f is formatted using shortest-round-trip representation.
  MLOG("{}", 3.14f);
  EXPECT_NE(capturedOutput().find("3.14"), std::string::npos);
}

TEST_F(LogOutputTest, LogsDouble) {
  MLOG("{}", 2.718281828);
  EXPECT_NE(capturedOutput().find("2.718281828"), std::string::npos);
}

TEST_F(LogOutputTest, LogsUnsignedInt) {
  MLOG("{}", 42u);
  EXPECT_NE(capturedOutput().find("42"), std::string::npos);
}

TEST_F(LogOutputTest, LogsMultipleArgs) {
  MLOG("{} {} {}", 1, "hello", 3.14f);
  auto out = capturedOutput();
  EXPECT_NE(out.find("1"), std::string::npos);
  EXPECT_NE(out.find("hello"), std::string::npos);
  EXPECT_NE(out.find("3.14"), std::string::npos);
}

TEST_F(LogOutputTest, LogsCString) {
  MLOG("{}", "c-string value, long to avoid small string in LogArg");
  EXPECT_NE(capturedOutput().find(
                "c-string value, long to avoid small string in LogArg"),
            std::string::npos);
}

TEST_F(LogOutputTest, LogsStdString) {
  std::string s("std-string value, longer than LogArg SSO");
  MLOG("{}", s);
  EXPECT_NE(capturedOutput().find("std-string value, longer than LogArg SSO"),
            std::string::npos);
}

TEST_F(LogOutputTest, LogsStdStringSmall) {
  std::string s("small");
  MLOG("{}", s);
  EXPECT_NE(capturedOutput().find("small"), std::string::npos);
}

TEST_F(LogOutputTest, LogsStdStringTemporary) {
  MLOG("{}", std::string("temporary std-string value"));
  EXPECT_NE(capturedOutput().find("temporary std-string value"),
            std::string::npos);
}

TEST_F(LogOutputTest, LogsSmallStringInSso) {
  MLOG("{}", "small str");
  EXPECT_NE(capturedOutput().find("small str"), std::string::npos);
}

TEST_F(LogOutputTest, LogsVoidPointer) {
  int x = 0;
  void *ptr = &x;
  MLOG("{}", ptr);
  // Non-string pointers are formatted as hex addresses (e.g. 0x7fff12345678).
  std::regex ptrPattern(R"(0x[0-9a-f]+)");
  EXPECT_TRUE(std::regex_search(capturedOutput(), ptrPattern));
}

} // namespace
