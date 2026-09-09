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
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace M::Log;

namespace {

static std::string gLogFilePath;
static std::string gConfigDirRoot;

class LogConfigFileTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    // Create a unique temp directory: /tmp/modular-log-cfg-test-XXXXXX
    llvm::SmallString<128> tmpDir;
    auto ec =
        llvm::sys::fs::createUniqueDirectory("modular-log-cfg-test", tmpDir);
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
    char tmpPath[] = "/tmp/modular-log-test-%%%%%%";
    int fd = 0;
    llvm::SmallString<128> realPath;
    ec = llvm::sys::fs::createUniqueFile(tmpPath, fd, realPath);
    ASSERT_FALSE(ec) << "Failed to create temp log file: " << ec.message();
    gLogFilePath = realPath.str().str();

    // Build and write modular.cfg into the .modular directory.
    llvm::SmallString<128> configFilePath = configDir;
    llvm::sys::path::append(configFilePath, "modular.cfg");

    llvm::SmallString<256> configContent;
    llvm::StringRef input = R"(
[log]
iso_time = true
level = ERROR
microseconds = true
file = )";
    configContent += input;
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
    ::testing::AddGlobalTestEnvironment(new LogConfigFileTestEnvironment);

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

class LogOutputTest : public ::testing::Test {
protected:
  void SetUp() override { startPos_ = currentLogEnd(); }

  std::string capturedOutput() const {
    getDefaultLog().flush();
    return readLogSince(startPos_);
  }

private:
  std::streampos startPos_{};
};

TEST_F(LogOutputTest, CheckLogLevel) {
  EXPECT_EQ(M::Log::getDefaultLog().getLogLevel(), M::Log::LogLevel::ERROR);
}

TEST_F(LogOutputTest, TimestampIsISOWithMicroseconds) {
  M::Log::setLogLevel(M::Log::LogLevel::WARN);
  MLOG(LogLevel::WARN, "ts test");
  // ISO 8601 format with microseconds: 2026-12-25T12:00:00.123456Z
  std::regex tsPattern(R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z)");
  EXPECT_TRUE(std::regex_search(capturedOutput(), tsPattern));
}

TEST_F(LogOutputTest, WarnLevelFiltersInfoMessages) {
  MLOG(LogLevel::INFO, "info message");
  EXPECT_TRUE(capturedOutput().empty());
}

TEST_F(LogOutputTest, WarnLevelPassesWarnMessages) {
  MLOG(LogLevel::WARN, "warn message");
  EXPECT_NE(capturedOutput().find("warn message"), std::string::npos);
}

} // namespace
