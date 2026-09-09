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
#include <sstream>
#include <string>

#ifndef _WIN32
#include <stdlib.h>
#endif

#include "gtest/gtest.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace M::Log;

namespace {

// Validates a single log line against the JSON schema in
// Support/docs/Logging.md. Returns "" on success, or an error description on
// failure.
std::string validateLogLineSchema(const std::string &line) {
  auto trimmed = line.substr(0, line.find_last_not_of("\n") + 1);

  auto parsed = llvm::json::parse(trimmed);
  if (!parsed)
    return llvm::toString(parsed.takeError());

  auto *obj = parsed->getAsObject();
  if (!obj)
    return "top-level value is not a JSON object";

  // envelope fields, present on both record variants
  for (const char *key : {"timestamp", "level", "channel"})
    if (!obj->get(key))
      return std::string("missing required field: ") + key;

  // A formatted record carries "message" and nothing else; a key-value record
  // carries its pairs as additional top-level fields instead.
  bool isKeyValue = !obj->get("message");
  if (!isKeyValue) {
    static const llvm::StringSet<> allowed = {
        {"timestamp"}, {"level"}, {"channel"}, {"message"}};
    for (auto &[k, v] : *obj)
      if (!allowed.contains(k))
        return "unexpected field: " + k.str();
    if (!obj->getString("message"))
      return R"("message" must be a string)";
  } else if (obj->size() <= 3) {
    return "key-value record has no pair fields";
  }

  // type checks
  if (!obj->getString("timestamp"))
    return R"("timestamp" must be a string)";
  if (!obj->getString("level"))
    return R"("level" must be a string)";
  if (!obj->getString("channel"))
    return R"("channel" must be a string)";

  // enum check on "level"
  static const llvm::StringSet<> validLevels = {
      {"DBG"}, {"INFO"}, {"WARN"}, {"ERR"}, {"FATL"}};
  auto lvl = *obj->getString("level");
  if (!validLevels.contains(lvl))
    return ("\"level\" value not in enum: " + lvl).str();

  return "";
}

static std::string gLogFilePath;
static std::string gConfigDirRoot;

class LogJSONOutputTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    // Create a unique temp directory: /tmp/modular-log-json-test-XXXXXX
    llvm::SmallString<128> tmpDir;
    auto ec =
        llvm::sys::fs::createUniqueDirectory("modular-log-json-test", tmpDir);
    ASSERT_FALSE(ec) << "Failed to create temp config dir: " << ec.message();
    gConfigDirRoot = tmpDir.str().str();

    // Create /tmp/<random>/.modular/ subdirectory
    llvm::SmallString<128> configDir = tmpDir;
    llvm::sys::path::append(configDir, ".modular");
    ec = llvm::sys::fs::create_directory(configDir);
    ASSERT_FALSE(ec) << "Failed to create .modular dir: " << ec.message();

    // Set TEST_TMPDIR so Config::open() searches <tmpDir>/.modular for config.
#ifndef _WIN32
    setenv("TEST_TMPDIR", tmpDir.c_str(), /*overwrite=*/1);
#else
#error "Test requires modification for Windows"
#endif

    // Create a unique temp file for log output.
    char tmpPath[] = "/tmp/modular-log-json-test-%%%%%%";
    int fd = 0;
    llvm::SmallString<128> realPath;
    ec = llvm::sys::fs::createUniqueFile(tmpPath, fd, realPath);
    ASSERT_FALSE(ec) << "Failed to create temp log file: " << ec.message();
    gLogFilePath = realPath.str().str();

    // Build and write modular.cfg into the .modular directory.
    llvm::SmallString<128> configFilePath = configDir;
    llvm::sys::path::append(configFilePath, "modular.cfg");

    llvm::SmallString<256> configContent;
    configContent += R"(
[log]
file = )";
    configContent += gLogFilePath;
    configContent += "\n";

    llvm::raw_fd_ostream cfgFile(configFilePath, ec);
    ASSERT_FALSE(ec) << "Failed to open modular.cfg for writing: "
                     << ec.message();
    cfgFile << configContent;
  }

  void TearDown() override {
    llvm::sys::fs::remove(gLogFilePath);
    llvm::sys::fs::remove_directories(gConfigDirRoot);
  }
};

static ::testing::Environment *const kLogEnv =
    ::testing::AddGlobalTestEnvironment(new LogJSONOutputTestEnvironment);

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

class LogJSONTest : public ::testing::Test {
protected:
  void SetUp() override {
    // The logger is async, so a previous test's last record may still be in
    // the ring. Drain it before marking where this test's output starts.
    getDefaultLog().flush();
    startPos_ = currentLogEnd();
  }

  std::string capturedOutput() const {
    getDefaultLog().flush();
    return readLogSince(startPos_);
  }

private:
  std::streampos startPos_{};
};

TEST_F(LogJSONTest, OutputIsOnOneLine) {
  MLOG(LogLevel::INFO, "single line");
  auto out = capturedOutput();
  // Exactly one newline, at the end.
  EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 1);
}

TEST_F(LogJSONTest, TimestampIsISO8601WithMicroseconds) {
  MLOG(LogLevel::INFO, "ts check");
  std::regex tsPattern(
      R"("timestamp"\s*:\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")");
  EXPECT_TRUE(std::regex_search(capturedOutput(), tsPattern));
}

TEST_F(LogJSONTest, LevelFieldDebug) {
  MLOG(LogLevel::DEBUG, "msg");
  EXPECT_NE(capturedOutput().find(R"("level":"DBG")"), std::string::npos);
}

TEST_F(LogJSONTest, LevelFieldInfo) {
  MLOG(LogLevel::INFO, "msg");
  EXPECT_NE(capturedOutput().find(R"("level":"INFO")"), std::string::npos);
}

TEST_F(LogJSONTest, LevelFieldWarn) {
  MLOG(LogLevel::WARN, "msg");
  EXPECT_NE(capturedOutput().find(R"("level":"WARN")"), std::string::npos);
}

TEST_F(LogJSONTest, LevelFieldError) {
  MLOG(LogLevel::ERROR, "msg");
  EXPECT_NE(capturedOutput().find(R"("level":"ERR")"), std::string::npos);
}

TEST_F(LogJSONTest, ChannelFieldIsPresent) {
  MLOG(LogLevel::INFO, "channel check");
  EXPECT_NE(capturedOutput().find(R"("channel":"default")"), std::string::npos);
}

TEST_F(LogJSONTest, MessageFieldIsPresent) {
  MLOG(LogLevel::INFO, "hello world");
  EXPECT_NE(capturedOutput().find(R"("message":"hello world")"),
            std::string::npos);
}

TEST_F(LogJSONTest, MessageEscapesDoubleQuote) {
  MLOG(LogLevel::INFO, R"(say "hi")");
  EXPECT_NE(capturedOutput().find(R"("message":"say \"hi\"")"),
            std::string::npos);
}

TEST_F(LogJSONTest, MessageEscapesBackslash) {
  MLOG(LogLevel::INFO, R"(a\b)");
  EXPECT_NE(capturedOutput().find(R"("message":"a\\b")"), std::string::npos);
}

TEST_F(LogJSONTest, MessageEscapesNewline) {
  MLOG(LogLevel::INFO, "line1\nline2");
  EXPECT_NE(capturedOutput().find(R"("message":"line1\nline2")"),
            std::string::npos);
}

TEST_F(LogJSONTest, NoANSIColorCodes) {
  MLOG(LogLevel::WARN, "colorless");
  // ESC character should not appear anywhere in the output.
  EXPECT_EQ(capturedOutput().find('\x1b'), std::string::npos);
}

TEST_F(LogJSONTest, LevelFilteringSuppressesOutput) {
  auto level = getDefaultLog().getLogLevel();
  setLogLevel(LogLevel::ERROR);
  MLOG(LogLevel::DEBUG, "should be suppressed");
  MLOG(LogLevel::INFO, "also suppressed");
  MLOG(LogLevel::WARN, "also suppressed");
  EXPECT_TRUE(capturedOutput().empty());
  setLogLevel(level);
}

TEST_F(LogJSONTest, KeyValuePairsBecomeTopLevelFields) {
  MLOG_KV(LogLevel::INFO, "event", "span_start", "operation", "prefill");
  auto out = capturedOutput();
  EXPECT_NE(out.find(R"("event":"span_start")"), std::string::npos);
  EXPECT_NE(out.find(R"("operation":"prefill")"), std::string::npos);
}

TEST_F(LogJSONTest, KeyValueRecordHasNoMessageField) {
  MLOG_KV(LogLevel::INFO, "event", "span_end");
  EXPECT_EQ(capturedOutput().find(R"("message")"), std::string::npos);
}

TEST_F(LogJSONTest, KeyValueRecordKeepsEnvelopeFields) {
  MLOG_KV(LogLevel::WARN, "event", "evicted");
  auto out = capturedOutput();
  EXPECT_NE(out.find(R"("level":"WARN")"), std::string::npos);
  EXPECT_NE(out.find(R"("channel":"default")"), std::string::npos);
}

// Datadog facets are typed, so an integer must not arrive quoted.
TEST_F(LogJSONTest, IntegerValueIsAJSONNumber) {
  MLOG_KV(LogLevel::INFO, "batch_id", 42);
  EXPECT_NE(capturedOutput().find(R"("batch_id":42)"), std::string::npos);
}

TEST_F(LogJSONTest, UnsignedValueIsAJSONNumber) {
  MLOG_KV(LogLevel::INFO, "duration_us", 1234u);
  EXPECT_NE(capturedOutput().find(R"("duration_us":1234)"), std::string::npos);
}

TEST_F(LogJSONTest, BoolValueIsAJSONBool) {
  MLOG_KV(LogLevel::INFO, "cached", true);
  EXPECT_NE(capturedOutput().find(R"("cached":true)"), std::string::npos);
}

TEST_F(LogJSONTest, StringValueIsEscaped) {
  MLOG_KV(LogLevel::INFO, "detail", R"(say "hi")");
  EXPECT_NE(capturedOutput().find(R"("detail":"say \"hi\"")"),
            std::string::npos);
}

TEST_F(LogJSONTest, ValueLongerThanInlineBufferSurvivesTheArena) {
  std::string longValue(64, 'x');
  MLOG_KV(LogLevel::INFO, "trace_id", longValue);
  EXPECT_NE(capturedOutput().find(R"("trace_id":")" + longValue + R"(")"),
            std::string::npos);
}

TEST_F(LogJSONTest, KeyValueOutputIsOnOneLine) {
  MLOG_KV(LogLevel::INFO, "event", "span_start", "batch_id", 7);
  auto out = capturedOutput();
  EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 1);
}

TEST_F(LogJSONTest, KeyValueLevelFilteringSuppressesOutput) {
  auto level = getDefaultLog().getLogLevel();
  setLogLevel(LogLevel::ERROR);
  MLOG_KV(LogLevel::INFO, "event", "suppressed");
  EXPECT_TRUE(capturedOutput().empty());
  setLogLevel(level);
}

// Validate each emitted line against the schema in Support/docs/Logging.md.
TEST_F(LogJSONTest, OutputMatchesSchema) {
  MLOG(LogLevel::DEBUG, "schema check debug");
  MLOG(LogLevel::INFO, "schema check info");
  MLOG(LogLevel::WARN, "schema check warn");
  MLOG(LogLevel::ERROR, "schema check error");
  MLOG_KV(LogLevel::INFO, "event", "schema check kv", "batch_id", 1);
  auto out = capturedOutput();
  ASSERT_FALSE(out.empty());
  std::istringstream stream(out);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty())
      continue;
    auto err = validateLogLineSchema(line);
    EXPECT_TRUE(err.empty())
        << "Schema violation in line: " << line << "\nError: " << err;
  }
}

} // namespace
