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

// Covers MLOG_KV's plain-text rendering. The JSON rendering of the same
// records is covered by LogJSONOutputTest.

#include "Support/Log.h"

#include <cstdlib>
#include <fstream>
#include <string>

#ifndef _WIN32
#include <stdlib.h>
#endif

#include "gtest/gtest.h"

#include "llvm/Support/FileSystem.h"

using namespace M::Log;

namespace {

static std::string gLogFilePath;

class LogKeyValueTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    char tmpPath[] = "/tmp/modular-log-kv-test-%%%%%%";
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
    // Forces Logger construction while the env above is in effect.
    (void)getDefaultLog();
  }

  void TearDown() override { llvm::sys::fs::remove(gLogFilePath); }
};

static ::testing::Environment *const kLogEnv =
    ::testing::AddGlobalTestEnvironment(new LogKeyValueTestEnvironment);

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

class LogKeyValueTest : public ::testing::Test {
protected:
  void SetUp() override {
    setLogLevel(LogLevel::DEBUG);
    // The logger is async, so a previous test's last record may still be in
    // the ring. Drain it before marking where this test's output starts.
    getDefaultLog().flush();
    startPos_ = currentLogEnd();
  }

  void TearDown() override { setLogLevel(LogLevel::INFO); }

  std::string capturedOutput() const {
    getDefaultLog().flush();
    return readLogSince(startPos_);
  }

private:
  std::streampos startPos_{};
};

TEST_F(LogKeyValueTest, SinglePairRendersAsKeyEqualsValue) {
  MLOG_KV(LogLevel::INFO, "event", "span_start");
  EXPECT_NE(capturedOutput().find("event=span_start"), std::string::npos);
}

TEST_F(LogKeyValueTest, PairsAreSpaceSeparatedInOrder) {
  MLOG_KV(LogLevel::INFO, "event", "span_start", "operation", "prefill",
          "batch_id", 42, "request_id", "a1b2c3");
  EXPECT_NE(
      capturedOutput().find("event=span_start operation=prefill batch_id=42 "
                            "request_id=a1b2c3"),
      std::string::npos);
}

TEST_F(LogKeyValueTest, IntegerValueRendersUnquoted) {
  MLOG_KV(LogLevel::INFO, "batch_id", 42);
  EXPECT_NE(capturedOutput().find("batch_id=42"), std::string::npos);
}

TEST_F(LogKeyValueTest, NegativeIntegerValue) {
  MLOG_KV(LogLevel::INFO, "delta", -17);
  EXPECT_NE(capturedOutput().find("delta=-17"), std::string::npos);
}

TEST_F(LogKeyValueTest, BoolValue) {
  MLOG_KV(LogLevel::INFO, "cached", true);
  EXPECT_NE(capturedOutput().find("cached=true"), std::string::npos);
}

TEST_F(LogKeyValueTest, DoubleValue) {
  MLOG_KV(LogLevel::INFO, "duration_ms", 1.5);
  EXPECT_NE(capturedOutput().find("duration_ms=1.5"), std::string::npos);
}

// Values longer than LogArg's inline buffer take the arena path, so this
// exercises different code from the short-string cases above.
TEST_F(LogKeyValueTest, ValueLongerThanInlineBuffer) {
  std::string longValue(64, 'x');
  MLOG_KV(LogLevel::INFO, "trace_id", longValue);
  EXPECT_NE(capturedOutput().find("trace_id=" + longValue), std::string::npos);
}

TEST_F(LogKeyValueTest, KeyLongerThanInlineBuffer) {
  std::string longKey(40, 'k');
  MLOG_KV(LogLevel::INFO, longKey, 7);
  EXPECT_NE(capturedOutput().find(longKey + "=7"), std::string::npos);
}

TEST_F(LogKeyValueTest, LevelPrefixIsStillApplied) {
  MLOG_KV(LogLevel::WARN, "event", "evicted");
  EXPECT_NE(capturedOutput().find("[WARN]"), std::string::npos);
}

TEST_F(LogKeyValueTest, FilteredLevelWritesNothing) {
  setLogLevel(LogLevel::ERROR);
  MLOG_KV(LogLevel::INFO, "event", "suppressed");
  EXPECT_TRUE(capturedOutput().empty());
}

// The macro's level guard must short-circuit before the arguments run, which
// is the property that makes MLOG_KV safe on hot paths.
TEST_F(LogKeyValueTest, FilteredLevelDoesNotEvaluateArguments) {
  setLogLevel(LogLevel::ERROR);
  int callCount = 0;
  auto expensiveValue = [&callCount] { return ++callCount; };
  MLOG_KV(LogLevel::INFO, "value", expensiveValue());
  EXPECT_EQ(callCount, 0);

  setLogLevel(LogLevel::DEBUG);
  MLOG_KV(LogLevel::INFO, "value", expensiveValue());
  EXPECT_EQ(callCount, 1);
}

// The macro names the level twice, so it has to bind it to a local first.
TEST_F(LogKeyValueTest, LevelExpressionIsEvaluatedOnce) {
  int callCount = 0;
  auto level = [&callCount] {
    ++callCount;
    return LogLevel::INFO;
  };
  MLOG_KV(level(), "k", 1);
  EXPECT_EQ(callCount, 1);

  setLogLevel(LogLevel::ERROR);
  callCount = 0;
  MLOG_KV(level(), "k", 1);
  EXPECT_EQ(callCount, 1);
}

// A key over 16 bytes no longer fits LogArg's inline buffer and is copied into
// the record's 256-byte arena, which is shared with every other string in the
// record and clips rather than grows. A clipped key is a silently renamed
// field, so this pins where the safe boundary is.
TEST_F(LogKeyValueTest, KeysWithinTheInlineBufferNeverTouchTheArena) {
  std::string maxInlineKey(16, 'k');
  std::string bigValue(240, 'v');
  MLOG_KV(LogLevel::INFO, maxInlineKey, bigValue);
  // The key survives intact even though the value alone nearly fills the
  // arena, because a 16-byte key is stored inline.
  EXPECT_NE(capturedOutput().find(maxInlineKey + "="), std::string::npos);
}

TEST_F(LogKeyValueTest, FormattedRecordsStillRenderNormally) {
  MLOG(LogLevel::INFO, "plain {} message", "formatted");
  auto out = capturedOutput();
  EXPECT_NE(out.find("plain formatted message"), std::string::npos);
  EXPECT_EQ(out.find('='), std::string::npos);
}

} // namespace
