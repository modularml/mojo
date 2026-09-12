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

// Covers MLOG_KV_REQ, the seam between the logger and the request context. The
// context's own behaviour is covered by RequestContextTest.

#include "Support/RequestLog.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

#ifndef _WIN32
#include <stdlib.h>
#endif

#include "gtest/gtest.h"

#include "llvm/Support/FileSystem.h"

using namespace M::Log;

namespace {

static std::string gLogFilePath;

class RequestLogTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    char tmpPath[] = "/tmp/modular-request-log-test-%%%%%%";
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
    ::testing::AddGlobalTestEnvironment(new RequestLogTestEnvironment);

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

class RequestLogTest : public ::testing::Test {
protected:
  void SetUp() override {
    setLogLevel(LogLevel::DEBUG);
    // The logger is async, so a previous test's last record may still be in
    // the ring. Drain it before marking where this test's output starts.
    getDefaultLog().flush();
    startPos_ = currentLogEnd();
  }

  void TearDown() override {
    M::Request::clearBatchId();
    setLogLevel(LogLevel::INFO);
  }

  std::string capturedOutput() const {
    getDefaultLog().flush();
    return readLogSince(startPos_);
  }

private:
  std::streampos startPos_{};
};

TEST_F(RequestLogTest, AppendsBatchIdInsideAScope) {
  M::Request::BatchScope batch(7);
  MLOG_KV_REQ(LogLevel::INFO, "event", "kv_evict");
  EXPECT_NE(capturedOutput().find("event=kv_evict batch_id=7"),
            std::string::npos);
}

TEST_F(RequestLogTest, OmitsBatchIdOutsideAScope) {
  MLOG_KV_REQ(LogLevel::INFO, "event", "kv_evict");
  auto out = capturedOutput();
  EXPECT_NE(out.find("event=kv_evict"), std::string::npos);
  EXPECT_EQ(out.find("batch_id"), std::string::npos);
}

TEST_F(RequestLogTest, OmitsBatchIdAfterTheScopeEnds) {
  {
    M::Request::BatchScope batch(7);
  }
  MLOG_KV_REQ(LogLevel::INFO, "event", "after");
  EXPECT_EQ(capturedOutput().find("batch_id"), std::string::npos);
}

// Zero is a real batch id, and the branch turns on a null pointer rather than
// on the value, so it has to survive.
TEST_F(RequestLogTest, BatchIdZeroIsStillEmitted) {
  M::Request::BatchScope batch(0);
  MLOG_KV_REQ(LogLevel::INFO, "event", "first_batch");
  EXPECT_NE(capturedOutput().find("batch_id=0"), std::string::npos);
}

TEST_F(RequestLogTest, CallerPairsKeepTheirOrderAheadOfBatchId) {
  M::Request::BatchScope batch(7);
  MLOG_KV_REQ(LogLevel::INFO, "event", "kv_evict", "blocks", 3, "reason",
              "pressure");
  EXPECT_NE(capturedOutput().find(
                "event=kv_evict blocks=3 reason=pressure batch_id=7"),
            std::string::npos);
}

TEST_F(RequestLogTest, NestedScopeIdIsTheOneEmitted) {
  M::Request::BatchScope outer(1);
  {
    M::Request::BatchScope inner(2);
    MLOG_KV_REQ(LogLevel::INFO, "event", "inner");
  }
  MLOG_KV_REQ(LogLevel::INFO, "event", "outer");
  auto out = capturedOutput();
  EXPECT_NE(out.find("event=inner batch_id=2"), std::string::npos);
  EXPECT_NE(out.find("event=outer batch_id=1"), std::string::npos);
}

TEST_F(RequestLogTest, FilteredLevelWritesNothing) {
  M::Request::BatchScope batch(7);
  setLogLevel(LogLevel::ERROR);
  MLOG_KV_REQ(LogLevel::INFO, "event", "suppressed");
  EXPECT_TRUE(capturedOutput().empty());
}

// The macro expands the caller's arguments into two mutually exclusive
// branches, so an argument with a side effect must still run exactly once.
TEST_F(RequestLogTest, ArgumentsAreEvaluatedOnce) {
  int callCount = 0;
  auto value = [&callCount] { return ++callCount; };

  MLOG_KV_REQ(LogLevel::INFO, "value", value());
  EXPECT_EQ(callCount, 1);

  M::Request::BatchScope batch(7);
  MLOG_KV_REQ(LogLevel::INFO, "value", value());
  EXPECT_EQ(callCount, 2);
}

// The level guard lives in MLOG_KV, which the macro expands into both branches;
// neither may evaluate the caller's arguments when the record is filtered out.
TEST_F(RequestLogTest, FilteredLevelDoesNotEvaluateArguments) {
  M::Request::BatchScope batch(7);
  setLogLevel(LogLevel::ERROR);
  int callCount = 0;
  auto value = [&callCount] { return ++callCount; };
  MLOG_KV_REQ(LogLevel::INFO, "value", value());
  EXPECT_EQ(callCount, 0);
}

// The macro names `level` in both branches, so it must not be an expression
// that is evaluated twice.
TEST_F(RequestLogTest, LevelExpressionIsEvaluatedOnce) {
  M::Request::BatchScope batch(7);
  int callCount = 0;
  auto level = [&callCount] {
    ++callCount;
    return LogLevel::INFO;
  };
  MLOG_KV_REQ(level(), "k", 1);
  EXPECT_EQ(callCount, 1);
}

// A worker thread has no batch of its own, which is the documented gap for
// runtime work that fans out mid-request.
TEST_F(RequestLogTest, DispatchedWorkLogsWithoutABatchId) {
  M::Request::BatchScope batch(7);
  std::thread worker(
      [] { MLOG_KV_REQ(LogLevel::INFO, "event", "fanned_out"); });
  worker.join();
  auto out = capturedOutput();
  EXPECT_NE(out.find("event=fanned_out"), std::string::npos);
  EXPECT_EQ(out.find("batch_id"), std::string::npos);
}

// The macro is sugar over MLOG_KV, so it has to compose with an if/else that
// does not brace its branches.
TEST_F(RequestLogTest, ExpandsAsASingleStatement) {
  M::Request::BatchScope batch(7);
  bool condition = true;
  if (condition)
    MLOG_KV_REQ(LogLevel::INFO, "event", "taken");
  else
    MLOG_KV_REQ(LogLevel::INFO, "event", "not_taken");
  auto out = capturedOutput();
  EXPECT_NE(out.find("event=taken batch_id=7"), std::string::npos);
  EXPECT_EQ(out.find("not_taken"), std::string::npos);
}

} // namespace
