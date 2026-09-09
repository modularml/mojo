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
//
// Tests that verify the logger handles potentially dangerous argument patterns
// safely. Each test documents a class of hazard and asserts the logger produces
// correct output rather than reading garbage or crashing.

#include "Support/Log.h"

#include <fstream>
#include <string>

#include "gtest/gtest.h"

using namespace M::Log;

namespace {

static std::string gLogFilePath;

class LogEdgeCasesEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    char tmp[] = "/tmp/modular-log-edge-cases-XXXXXX";
    int fd = mkstemp(tmp);
    ASSERT_GE(fd, 0);
    close(fd);
    gLogFilePath = tmp;
    ::setenv("MODULAR_LOG_FILE", gLogFilePath.c_str(), 1);
    ::setenv("MODULAR_LOG_STDOUT", "false", 1);
    ::setenv("MODULAR_LOG_NO_SUMMARY", "1", 1);
    // Force default log initialisation now, while env vars are set.
    (void)getDefaultLog();
  }

  void TearDown() override { ::unlink(gLogFilePath.c_str()); }
};

static ::testing::Environment *const kEnv =
    ::testing::AddGlobalTestEnvironment(new LogEdgeCasesEnvironment);

std::streampos currentLogEnd() {
  std::ifstream f(gLogFilePath);
  f.seekg(0, std::ios::end);
  return f.tellg();
}

std::string readLogSince(std::streampos offset) {
  getDefaultLog().flush();
  std::ifstream f(gLogFilePath);
  f.seekg(offset);
  return {std::istreambuf_iterator<char>(f), {}};
}

class LogEdgeCasesTest : public ::testing::Test {
protected:
  std::streampos logStart;
  void SetUp() override {
    setLogLevel(LogLevel::INFO);
    logStart = currentLogEnd();
  }

  std::string capturedOutput() { return readLogSince(logStart); }
};

// A local const char[N] array longer than the SSO buffer (16 bytes) must be
// arena-copied at enqueue time, not held by pointer. The array goes out of
// scope before the consumer thread processes the record.
//
// The junk block allocates at the same stack address to corrupt name's memory
// before the consumer reads it. Without the arena copy the test would fail —
// though it might accidentally pass if the stack happens not to be reused.
TEST_F(LogEdgeCasesTest, LocalCharArrayArenaCopied) {
  {
    const char name[64] = "a name longer than sixteen chars";
    MLOG_INFO("user={}", name);
  }
  {
    const volatile char junk[64] = "a name longer than\0\0\0\0";
    (void)junk[0]; // ensure the stack write is not elided
  }
  EXPECT_NE(capturedOutput().find("sixteen chars"), std::string::npos);
}

// When the ring buffer fills, producers drop records rather than blocking.
// Snapshot the counter before the flood so the test is independent of any
// drops accumulated by earlier tests in this process.
TEST_F(LogEdgeCasesTest, RingFullDropsRecords) {
  uint64_t dropsBefore = getDefaultLog().droppedCount();
  for (size_t i = 0; i < 20'000; ++i)
    logWriteDispatch(getDefaultLog(), LogLevel::INFO, Channel::Default,
                     "flood {}", i);
  getDefaultLog().flush();
  EXPECT_GT(getDefaultLog().droppedCount(), dropsBefore);
}

// String arguments are copied into a per-slot arena of 256 bytes. Content
// beyond that limit is silently clipped. The marker byte placed just past
// the 256-byte boundary should be absent from the output.
TEST_F(LogEdgeCasesTest, LongStringArgClipped) {
  std::string arg(257, 'x');
  arg[256] = 'y'; // one byte past the 256-byte arena limit
  logWriteDispatch(getDefaultLog(), LogLevel::INFO, Channel::Default, "s={}",
                   arg);
  std::string out = capturedOutput();
  EXPECT_NE(out.find(std::string(256, 'x')), std::string::npos);
  EXPECT_EQ(out.find('y'), std::string::npos);
}

// flush() blocks until all records enqueued before the call have been written
// to sinks. Records must be visible in the output file immediately after
// flush() returns.
TEST_F(LogEdgeCasesTest, FlushDrainsBeforeReturn) {
  logWriteDispatch(getDefaultLog(), LogLevel::INFO, Channel::Default,
                   "flush-marker");
  getDefaultLog().flush();
  // Read directly without going through capturedOutput() so we are not
  // relying on its internal flush() call to make the record visible.
  std::ifstream f(gLogFilePath);
  f.seekg(logStart);
  std::string out(std::istreambuf_iterator<char>(f), {});
  EXPECT_NE(out.find("flush-marker"), std::string::npos);
}

} // namespace
