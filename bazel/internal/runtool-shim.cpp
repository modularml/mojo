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

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string_view>

static std::string_view getRequiredEnv(const char *key) {
  char *value = std::getenv(key);
  if (!value) {
    std::cerr << "Missing required env var: " << key << std::endl;
    abort();
  }

  return value;
}

// Convert comma-separated relative paths to absolute paths based on the
// current working directory. This must be called before changing directories.
static void makeImportPathsAbsolute() {
  char *importPath = std::getenv("MODULAR_MOJO_MAX_IMPORT_PATH");
  if (!importPath || importPath[0] == '\0')
    return;

  std::filesystem::path cwd = std::filesystem::current_path();
  std::string result;
  std::istringstream iss(importPath);
  std::string path;

  while (std::getline(iss, path, ',')) {
    if (!result.empty())
      result += ',';

    std::filesystem::path fsPath(path);
    if (fsPath.is_relative()) {
      // Convert relative path to absolute based on current runfiles directory
      result += (cwd / fsPath).lexically_normal().generic_string();
    } else {
      result += path;
    }
  }

  setenv("MODULAR_MOJO_MAX_IMPORT_PATH", result.c_str(), 1);
}

__attribute__((visibility("default"))) __attribute__((constructor)) void
fix_bazel_paths() {
  if (std::getenv("RUNNING_DIRECTLY") == nullptr) {
    // Either not running through bazel or being run transitively, in which
    // case the main target being run is responsible for configuration
    return;
  }

  std::filesystem::path workspaceDir =
      getRequiredEnv("BUILD_WORKSPACE_DIRECTORY");
  std::filesystem::path derivedDir = workspaceDir / ".derived";

  // Find modular.cfg in derived for runtime dependencies
  setenv("MODULAR_HOME", derivedDir.generic_string().c_str(), 0);
  auto pwd = std::filesystem::current_path();
  if (pwd.filename() == "_main") {
    // Convert import paths to absolute before changing directories, since they
    // are relative to the runfiles directory.
    makeImportPathsAbsolute();
    std::filesystem::current_path(getRequiredEnv("BUILD_WORKING_DIRECTORY"));
  }
}
