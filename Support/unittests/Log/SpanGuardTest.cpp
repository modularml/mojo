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

#include "Support/SpanGuard.h"

#include <cstdlib>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <stdlib.h>
#endif

#include "gtest/gtest.h"

#include "llvm/Support/FileSystem.h"

using namespace M::Log;

namespace {

static std::string gLogFilePath;

class SpanGuardTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    char tmpPath[] = "/tmp/modular-spanguard-test-%%%%%%";
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
    ::testing::AddGlobalTestEnvironment(new SpanGuardTestEnvironment);

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

// Returns the value of `key` from each record that has it, in order.
std::vector<std::string> fieldValues(const std::string &out,
                                     const std::string &key) {
  std::vector<std::string> values;
  std::regex pattern(key + "=([^ \n]+)");
  for (std::sregex_iterator it(out.begin(), out.end(), pattern), end; it != end;
       ++it)
    values.push_back((*it)[1].str());
  return values;
}

class SpanGuardTest : public ::testing::Test {
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

TEST_F(SpanGuardTest, EmitsStartAndEndRecords) {
  {
    SpanGuard span("prefill");
  }
  auto out = capturedOutput();
  EXPECT_NE(out.find("event=span_start operation=prefill"), std::string::npos);
  EXPECT_NE(out.find("event=span_end operation=prefill"), std::string::npos);
}

TEST_F(SpanGuardTest, StartAndEndShareASpanId) {
  {
    SpanGuard span("decode");
  }
  auto ids = fieldValues(capturedOutput(), "span_id");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], ids[1]);
}

TEST_F(SpanGuardTest, GetSpanIdMatchesTheEmittedRecords) {
  uint64_t observed = 0;
  {
    SpanGuard span("decode");
    observed = span.getSpanId();
  }
  auto ids = fieldValues(capturedOutput(), "span_id");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], std::to_string(observed));
}

TEST_F(SpanGuardTest, EndRecordCarriesAPositiveDuration) {
  {
    SpanGuard span("kernel_dispatch");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto durations = fieldValues(capturedOutput(), "duration_us");
  ASSERT_EQ(durations.size(), 1u);
  auto us = std::stoll(durations[0]);
  EXPECT_GT(us, 0);
  // Generous ceiling; this only catches a unit or clock-source mistake.
  EXPECT_LT(us, 10'000'000);
}

TEST_F(SpanGuardTest, DurationIsAtLeastTheSleptTime) {
  {
    SpanGuard span("prefill");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  auto durations = fieldValues(capturedOutput(), "duration_us");
  ASSERT_EQ(durations.size(), 1u);
  EXPECT_GE(std::stoll(durations[0]), 5000);
}

// The destructor is what closes the span, so any path out of the scope must
// still emit span_end. The toolchain builds with -fno-exceptions, so early
// return is the abnormal exit that actually occurs here.
TEST_F(SpanGuardTest, EndRecordIsEmittedOnEarlyReturn) {
  auto bailsOutEarly = [](bool bail) {
    SpanGuard span("prefill");
    if (bail)
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  };
  bailsOutEarly(true);
  auto out = capturedOutput();
  EXPECT_NE(out.find("event=span_start"), std::string::npos);
  EXPECT_NE(out.find("event=span_end"), std::string::npos);
}

TEST_F(SpanGuardTest, NestedSpansGetDistinctIds) {
  {
    SpanGuard outer("prefill");
    SpanGuard inner("kernel_dispatch");
    EXPECT_NE(outer.getSpanId(), inner.getSpanId());
  }
  auto ids = fieldValues(capturedOutput(), "span_id");
  ASSERT_EQ(ids.size(), 4u);
  // Emission order is outer-start, inner-start, inner-end, outer-end.
  EXPECT_EQ(ids[0], ids[3]);
  EXPECT_EQ(ids[1], ids[2]);
  EXPECT_NE(ids[0], ids[1]);
}

TEST_F(SpanGuardTest, SequentialSpansGetDistinctIds) {
  {
    SpanGuard a("prefill");
  }
  {
    SpanGuard b("prefill");
  }
  auto ids = fieldValues(capturedOutput(), "span_id");
  ASSERT_EQ(ids.size(), 4u);
  EXPECT_NE(ids[0], ids[2]);
}

TEST_F(SpanGuardTest, SpanIdsAreUniqueAcrossThreads) {
  constexpr int kThreads = 8;
  constexpr int kSpansPerThread = 16;
  std::vector<std::vector<uint64_t>> perThread(kThreads);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&perThread, t] {
      for (int i = 0; i < kSpansPerThread; ++i) {
        SpanGuard span("decode");
        perThread[t].push_back(span.getSpanId());
      }
    });
  }
  for (auto &thread : threads)
    thread.join();

  std::set<uint64_t> unique;
  for (const auto &ids : perThread)
    unique.insert(ids.begin(), ids.end());
  EXPECT_EQ(unique.size(), size_t{kThreads} * kSpansPerThread);
}

TEST_F(SpanGuardTest, FilteredLevelEmitsNothing) {
  setLogLevel(LogLevel::ERROR);
  {
    SpanGuard span("prefill");
  }
  EXPECT_TRUE(capturedOutput().empty());
}

} // namespace
