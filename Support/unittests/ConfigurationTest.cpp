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

#include "Support/Configuration.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

#ifdef LLVM_ON_UNIX
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

using namespace M;

/// Get a source manager for a given buffer so we can check diagnostics.
static llvm::SourceMgr getSourceMgr(StringRef buffer) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(buffer);
  llvm::SourceMgr mgr;
  (void)mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());
  return mgr;
}

TEST(Configuration, SectionParse) {
  StringRef input = R"(
[section]
key = value
# this is a comment
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  EXPECT_EQ(cfg.getValue("section.key"), "value");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "value3");
}

TEST(Configuration, Globals) {
  StringRef input = R"(
key = value
# this is a comment
[section]
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  EXPECT_EQ(cfg.getValue("key"), "value");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "value3");
}

TEST(Configuration, EnvOverride) {
  // Check that the env override works as expected.
  setenv("MODULAR_AKEY", "foo", 0);
  setenv("MODULAR_SECTION_SUBSECTION_KEY3", "bar", 0);

  auto unsetEnv = llvm::scope_exit([]() {
    unsetenv("MODULAR_KEY");
    unsetenv("MODULAR_SECTION_SUBSECTION_KEY");
  });

  StringRef input = R"(
akey = value
# this is a comment
[section]
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  EXPECT_EQ(cfg.getValue("akey"), "foo");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "bar");
  EXPECT_EQ(cfg.getValueOr("only.default.value", "default"), "default");
}

TEST(Configuration, SetValue) {
  StringRef input = R"(
akey = value
# this is a comment
[section]
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  cfg.setValue("akey", "foo");
  EXPECT_EQ(cfg.getValue("akey"), "foo");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  cfg.setValue("section.subsection.key3", "bar");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "bar");
}

TEST(Configuration, MalformedLine) {
  StringRef input = R"(
[section]
key = value
# this is a comment
key2 = value2 # with a comment
; another comment
malformed line here

[section.subsection] # yet another comment
key3 = value3
)";
  llvm::SourceMgr mgr = getSourceMgr(input);

  Config cfg;
  auto err = cfg.parseFrom(input, &mgr);
  ASSERT_TRUE(err.isError());
  EXPECT_EQ(StringRef(err.getError()),
            StringRef(R"(error: malformed line: expected `key = value`
malformed line here
^
)"));
}

#ifdef LLVM_ON_UNIX
// Regression test for GEX-3526: mojo terminated with an uncaught
// `std::filesystem::filesystem_error` when HOME was not traversable by the
// running UID (the directory stat returned EACCES, and `getSearchPaths` used
// the throwing overload of `std::filesystem::exists`). The entry points that
// walk the search paths must swallow permission errors and fall through to the
// next candidate rather than aborting the process.
TEST(Configuration, SearchPathsSurvivesNonTraversableHome) {
  // Create a directory and strip all permissions so that `stat` on any path
  // inside returns EACCES for non-root UIDs. Root bypasses permission bits,
  // so the assertion below only works from a non-root process — we fork and
  // drop privileges in the child.
  std::error_code ec;
  auto lockedHome = std::filesystem::temp_directory_path(ec) /
                    ("gex3526-locked-home-" + std::to_string(::getpid()));
  ASSERT_FALSE(ec) << ec.message();
  std::filesystem::create_directories(lockedHome, ec);
  ASSERT_FALSE(ec) << "create_directories: " << ec.message();
  auto cleanup = llvm::scope_exit([&] {
    (void)::chmod(lockedHome.c_str(), 0700);
    std::error_code rmEc;
    std::filesystem::remove_all(lockedHome, rmEc);
  });
  ASSERT_EQ(::chmod(lockedHome.c_str(), 0), 0)
      << "chmod 000: "
      << std::error_code(errno, std::generic_category()).message();

  // The child exercises the code path under test. We use distinct exit codes
  // so the parent can tell "fix works" (0) from "couldn't drop privileges"
  // (101) from "something else went wrong." An abort (signal) from the old
  // bug would show up as `WIFSIGNALED`, not `WIFEXITED`.
  pid_t pid = ::fork();
  ASSERT_NE(pid, -1) << "fork: " << std::strerror(errno);
  if (pid == 0) {
    if (::geteuid() == 0) {
      // Drop to nobody (65534). If we can't (user-namespace restrictions,
      // etc.), skip — signal that via exit code 101.
      if (::setuid(65534) != 0)
        _exit(101);
    }
    // `getSearchPaths` consults MODULAR_HOME, MODULAR_DERIVED_PATH, and
    // TEST_TMPDIR before HOME. Bazel sets TEST_TMPDIR, so we must clear it.
    ::setenv("HOME", lockedHome.c_str(), 1);
    ::unsetenv("MODULAR_HOME");
    ::unsetenv("MODULAR_DERIVED_PATH");
    ::unsetenv("TEST_TMPDIR");
    // Before the fix, these calls aborted the child (std::terminate from an
    // uncaught filesystem_error) and the parent saw SIGABRT.
    auto folder = Config::getModularConfigFolderPath(/*create=*/false);
    (void)folder;
    auto configFile = Config::getConfigFilePath(/*create=*/false);
    (void)configFile;
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(::waitpid(pid, &status, 0), pid)
      << "waitpid: " << std::strerror(errno);
  if (WIFSIGNALED(status)) {
    FAIL() << "child was killed by signal " << WTERMSIG(status)
           << " (regression: uncaught filesystem_error from getSearchPaths)";
    return;
  }
  ASSERT_TRUE(WIFEXITED(status)) << "child did not exit normally";
  int code = WEXITSTATUS(status);
  if (code == 101)
    GTEST_SKIP() << "unable to drop to a non-root UID in this sandbox";
  EXPECT_EQ(code, 0) << "child exited with unexpected code " << code;
}

// Regression test for a bug in Configuration
TEST(Configuration, PageBoundary) {
  auto pageSize = llvm::cantFail(llvm::sys::Process::getPageSize());
  std::error_code ec;
  llvm::sys::OwningMemoryBlock block(llvm::sys::Memory::allocateMappedMemory(
      2 * pageSize, nullptr,
      llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE, ec));
  ASSERT_FALSE(ec) << "Failed to allocate memory block: " << ec.message();
  // Can't use llvm::sys::Memory::protectMappedMemory because it considers
  // PFlags = 0 to be invalid.
  int mprotectResult = ::mprotect(
      reinterpret_cast<char *>(block.base()) + pageSize, pageSize, PROT_NONE);
  ASSERT_NE(mprotectResult, -1)
      << std::error_code(errno, std::generic_category()).message();
  // Note: No trailing newline!
  StringRef content("key = value");
  char *contentPlacePtr =
      reinterpret_cast<char *>(block.base()) + pageSize - content.size();
  StringRef contentInPlace(contentPlacePtr, content.size());
  memcpy(contentPlacePtr, content.data(), content.size());
  ASSERT_EQ(content, contentInPlace);
  // OK, now try to parse!
  Config cfg;
  auto err = cfg.parseFrom(contentInPlace);
  ASSERT_FALSE(err.isError()) << err.getError();
  EXPECT_EQ(cfg.getValue("key"), "value");
}
#endif // LLVM_ON_UNIX

TEST(Configuration, GetAllValues) {
  StringRef input = R"(
key = value
#maybe a comment in between these guys
key4 = value4

# this is a comment
[section]
key2 = value2 # with a comment
; another comment
key3 = value
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  const llvm::StringMap<std::string> &allVals = cfg.getAllValues();
  EXPECT_TRUE(allVals.contains("key"));
  EXPECT_TRUE(allVals.contains("key4"));
  EXPECT_EQ(allVals.at("key"), "value");
  EXPECT_TRUE(allVals.contains("section.key2"));
  EXPECT_EQ(allVals.at("section.key3"), "value");
}

TEST(Configuration, BooleanValues) {
  using R = bool;
  Config cfg;
  EXPECT_EQ(R(false), cfg.getValueAsBool("example", false));
  EXPECT_EQ(R(true), cfg.getValueAsBool("example", true));
  for (auto value : {"0", "false", "no", "FaLsE"}) {
    cfg.setValue("example", value);
    EXPECT_EQ(R(false), cfg.getValueAsBool("example", false));
    EXPECT_EQ(R(false), cfg.getValueAsBool("example", true));
  }
  for (auto value : {"1", "true", "yes", "TrUe"}) {
    cfg.setValue("example", value);
    EXPECT_EQ(R(true), cfg.getValueAsBool("example", false));
    EXPECT_EQ(R(true), cfg.getValueAsBool("example", true));
  }
  cfg.setValue("example", "maybe");
  bool result = cfg.getValueAsBool("example", false);
  EXPECT_FALSE(result);
}

TEST(Configuration, ListValues) {
  StringRef input = R"(
values = one,two,three
)";
  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  SmallVector<StringRef, 3> values;
  cfg.getValueAsList("values", values);
  EXPECT_EQ(3, values.size());
  EXPECT_TRUE(cfg.isValueInList("values", "one"));
  EXPECT_TRUE(cfg.isValueInList("values", "two"));
  EXPECT_TRUE(cfg.isValueInList("values", "three"));
  EXPECT_FALSE(cfg.isValueInList("values", "four"));

  cfg.setValue("empty", "");
  SmallVector<StringRef, 3> empty;
  cfg.getValueAsList("empty", empty);
  EXPECT_EQ(0, empty.size());
}

TEST(Configuration, GlobalOverride) {
  // Clean up after ourselves — the override map is process-global.
  auto cleanup = llvm::scope_exit([]() {
    Config::unsetGlobalValue("override.bool-key");
    Config::unsetGlobalValue("override.string-key");
    Config::unsetGlobalValue("override.env-win");
    unsetenv("MODULAR_OVERRIDE_ENV_WIN");
  });

  // A fresh Config with nothing else configured should see the override.
  Config::setGlobalValue("override.bool-key", "true");
  Config::setGlobalValue("override.string-key", "hello");

  Config cfg;
  EXPECT_TRUE(cfg.getValueAsBool("override.bool-key", false));
  EXPECT_EQ(cfg.getValue("override.string-key"), "hello");
  EXPECT_EQ(Config::getGlobalValueIfSet("override.bool-key"), "true");

  // Override wins over an env var on the same key.
  setenv("MODULAR_OVERRIDE_ENV_WIN", "from-env", 1);
  Config::setGlobalValue("override.env-win", "from-override");
  Config cfg2;
  EXPECT_EQ(cfg2.getValue("override.env-win"), "from-override");

  // A second Config instance after the override is set also sees it.
  Config cfg3;
  EXPECT_TRUE(cfg3.getValueAsBool("override.bool-key", false));

  // Clearing the override reverts to the underlying source (env var here).
  Config::unsetGlobalValue("override.env-win");
  EXPECT_EQ(Config::getGlobalValueIfSet("override.env-win"), std::nullopt);
  Config cfg4;
  EXPECT_EQ(cfg4.getValue("override.env-win"), "from-env");
}
