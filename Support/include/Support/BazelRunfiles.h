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

#ifndef SUPPORT_BAZELRUNFILES_H
#define SUPPORT_BAZELRUNFILES_H

#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace M {

/// Try to find a path for a modular.cfg config key using Bazel runfiles.
///
/// This function maps configuration keys (e.g., "mojo-max.driver_path") to
/// their corresponding Bazel runfile paths and resolves them using the Bazel
/// runfiles library.
///
/// Returns std::nullopt if:
/// - The key is not expected to be loaded via runfiles (no mapping exists)
/// - Runfiles is not available (not running under Bazel)
std::optional<std::string> findConfigWithRunfiles(llvm::StringRef key);

/// Get the environment variables a subprocess needs to resolve our runfiles,
/// as name/value pairs. Empty when not running under Bazel.
///
/// A process finds its runfiles relative to its own executable, so a tool we
/// spawn resolves nothing of ours unless it inherits these. Tools staged from
/// another repository have no runfiles tree of their own at all.
const std::vector<std::pair<std::string, std::string>> *getRunfilesEnvVars();

} // namespace M

#endif // SUPPORT_BAZELRUNFILES_H
