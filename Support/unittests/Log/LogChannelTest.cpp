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
#include "Support/LogChannels.h"

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

class LogChannelTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    char tmpPath[] = "/tmp/modular-log-channel-test-%%%%%%";
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
    auto &log = getDefaultLog();
    (void)log;
  }

  void TearDown() override { llvm::sys::fs::remove(gLogFilePath); }
};

static ::testing::Environment *const kLogEnv =
    ::testing::AddGlobalTestEnvironment(new LogChannelTestEnvironment);

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

TEST(ChannelStateTest, DefaultChannelEnabledOnConstruction) {
  ChannelState cs;
  EXPECT_TRUE(cs.isEnabled(Channel::Default));
}

TEST(ChannelStateTest, NonDefaultChannelsDisabledOnConstruction) {
  ChannelState cs;
  EXPECT_FALSE(cs.isEnabled(Channel::Mojo));
  EXPECT_FALSE(cs.isEnabled(Channel::MLRT));
}

TEST(ChannelStateTest, EnableChannel) {
  ChannelState cs;
  cs.enable(Channel::Mojo);
  EXPECT_TRUE(cs.isEnabled(Channel::Mojo));
}

TEST(ChannelStateTest, DisableChannel) {
  ChannelState cs;
  cs.enable(Channel::Mojo);
  cs.disable(Channel::Mojo);
  EXPECT_FALSE(cs.isEnabled(Channel::Mojo));
}

TEST(ChannelStateTest, DisableDefaultChannel) {
  ChannelState cs;
  cs.disable(Channel::Default);
  EXPECT_FALSE(cs.isEnabled(Channel::Default));
}

TEST(ChannelStateTest, EnableAllFlagMakesAllChannelsVisible) {
  ChannelState cs;
  cs.enableAll();
  EXPECT_TRUE(cs.isEnabled(Channel::Default));
  EXPECT_TRUE(cs.isEnabled(Channel::Mojo));
  EXPECT_TRUE(cs.isEnabled(Channel::MLRT));
}

TEST(ChannelStateTest, ClearAllFlagDisablesAllChannels) {
  ChannelState cs;
  cs.enableAll();
  cs.disableAll();
  EXPECT_FALSE(cs.isEnabled(Channel::Default));
  EXPECT_FALSE(cs.isEnabled(Channel::Mojo));
  EXPECT_FALSE(cs.isEnabled(Channel::MLRT));
}

TEST(ChannelStateTest, ClearAllFlagAlsoClearsIndividuallyEnabledChannels) {
  ChannelState cs;
  cs.enable(Channel::Mojo);
  cs.enableAll();
  cs.disableAll();
  EXPECT_FALSE(cs.isEnabled(Channel::Mojo));
}

class LogChannelOutputTest : public ::testing::Test {
protected:
  void SetUp() override {
    setLogLevel(LogLevel::DEBUG);
    // Reset channels: disable all, then re-enable Default only.
#define DISABLE_CHANNEL(ch, cfg) disableChannel(Channel::ch);
    MLOG_CHANNELS(DISABLE_CHANNEL)
#undef DISABLE_CHANNEL
    enableChannel(Channel::Default);
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

TEST_F(LogChannelOutputTest, DefaultChannelWrittenByDefault) {
  log(Channel::Default, "default channel msg");
  EXPECT_NE(capturedOutput().find("default channel msg"), std::string::npos);
}

TEST_F(LogChannelOutputTest, MojoChannelFilteredByDefault) {
  log(Channel::Mojo, "mojo channel msg");
  EXPECT_TRUE(capturedOutput().empty());
}

TEST_F(LogChannelOutputTest, MLRTChannelFilteredByDefault) {
  log(Channel::MLRT, "mlrt channel msg");
  EXPECT_TRUE(capturedOutput().empty());
}

TEST_F(LogChannelOutputTest, EnabledChannelMessageWritten) {
  enableChannel(Channel::Mojo);
  log(Channel::Mojo, "mojo enabled msg");
  EXPECT_NE(capturedOutput().find("mojo enabled msg"), std::string::npos);
}

TEST_F(LogChannelOutputTest, DisabledDefaultChannelFilters) {
  disableChannel(Channel::Default);
  log(Channel::Default, "should be filtered");
  EXPECT_TRUE(capturedOutput().empty());
}

TEST_F(LogChannelOutputTest, EnableAllFlagWritesAllChannels) {
  enableAllChannels();
  log(Channel::Mojo, "mojo via all-flag");
  log(Channel::MLRT, "mlrt via all-flag");
  auto out = capturedOutput();
  EXPECT_NE(out.find("mojo via all-flag"), std::string::npos);
  EXPECT_NE(out.find("mlrt via all-flag"), std::string::npos);
}

TEST_F(LogChannelOutputTest, LevelFilteringAppliesWithinEnabledChannel) {
  setLogLevel(LogLevel::WARN);
  enableChannel(Channel::Mojo);
  log(LogLevel::INFO, Channel::Mojo, "below threshold on mojo");
  EXPECT_TRUE(capturedOutput().empty());
}

TEST_F(LogChannelOutputTest, ChannelFilteringAppliesRegardlessOfLevel) {
  // Level passes (WARN >= WARN threshold) but channel is disabled.
  setLogLevel(LogLevel::WARN);
  log(LogLevel::WARN, Channel::Mojo, "mojo disabled but level ok");
  EXPECT_TRUE(capturedOutput().empty());
}

TEST_F(LogChannelOutputTest, LevelAndChannelBothPassMessageWritten) {
  setLogLevel(LogLevel::WARN);
  enableChannel(Channel::Mojo);
  log(LogLevel::ERROR, Channel::Mojo, "mojo error msg");
  EXPECT_NE(capturedOutput().find("mojo error msg"), std::string::npos);
}

TEST_F(LogChannelOutputTest, MultipleEnabledChannelsEachWritten) {
  enableChannel(Channel::Mojo);
  enableChannel(Channel::MLRT);
  log(Channel::Mojo, "mojo msg");
  log(Channel::MLRT, "mlrt msg");
  auto out = capturedOutput();
  EXPECT_NE(out.find("mojo msg"), std::string::npos);
  EXPECT_NE(out.find("mlrt msg"), std::string::npos);
}

TEST_F(LogChannelOutputTest, EnableThenDisableChannelFilters) {
  enableChannel(Channel::Mojo);
  disableChannel(Channel::Mojo);
  log(Channel::Mojo, "mojo re-disabled msg");
  EXPECT_TRUE(capturedOutput().empty());
}

} // namespace
