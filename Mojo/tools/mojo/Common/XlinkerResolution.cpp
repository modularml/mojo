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

#include "XlinkerResolution.h"

#include "Support/Driver/DriverSupport.h"
#include "llvm/ADT/StringRef.h"

#include <filesystem>
#include <optional>
#include <system_error>

using namespace M;
using namespace llvm;

namespace {

/// True if `path`'s filename has a shared-library extension recognized by
/// `dlopen()`: `.so`, `.dylib`, or a versioned suffix like `libfoo.so.1`.
bool hasSharedLibraryExtension(StringRef path) {
  return path.ends_with(".so") || path.ends_with(".dylib") ||
         path.contains(".so.");
}

/// Search `searchDirs` for a shared library named `libName`, trying each
/// platform-specific extension. Returns the first existing path, or
/// std::nullopt if no match is found.
std::optional<std::string> findLibraryByName(StringRef libName,
                                             ArrayRef<std::string> searchDirs) {
  static constexpr StringLiteral kExtensions[] = {
#if defined(__APPLE__)
      ".dylib",
      ".so",
#else
      ".so",
#endif
  };

  std::error_code ec;
  for (StringRef dir : searchDirs) {
    for (StringRef ext : kExtensions) {
      auto candidate = std::filesystem::path(dir.str()) /
                       ("lib" + libName.str() + ext.str());
      if (std::filesystem::exists(candidate, ec) && !ec)
        return candidate.string();
    }
  }
  return std::nullopt;
}

} // namespace

SmallVector<std::string>
M::resolveXlinkerLibraries(const State &state,
                           ArrayRef<std::string> xlinkerArgs) {
  SmallVector<std::string> searchDirs;
  SmallVector<std::string> libraries;
  // Library-by-name flags (`-l<name>`) are resolved after the full pass so
  // they pick up `-L` directories that appear later in the argument list.
  SmallVector<std::string> deferredLibs;

  auto consumeNext = [&](size_t &i) -> std::optional<StringRef> {
    if (i + 1 >= xlinkerArgs.size())
      return std::nullopt;
    return StringRef(xlinkerArgs[++i]);
  };

  for (size_t i = 0; i < xlinkerArgs.size(); ++i) {
    StringRef arg = xlinkerArgs[i];

    if (arg.consume_front("-L") || arg.consume_front("--library-path=")) {
      if (!arg.empty())
        searchDirs.emplace_back(arg.str());
      else if (auto next = consumeNext(i))
        searchDirs.emplace_back(next->str());
      continue;
    }
    if (arg == "--library-path") {
      if (auto next = consumeNext(i))
        searchDirs.emplace_back(next->str());
      continue;
    }

    if (arg.consume_front("-l")) {
      if (!arg.empty())
        deferredLibs.emplace_back(arg.str());
      else if (auto next = consumeNext(i))
        deferredLibs.emplace_back(next->str());
      continue;
    }

    if (arg.consume_front("-rpath=") || arg.consume_front("--rpath=")) {
      searchDirs.emplace_back(arg.str());
      continue;
    }
    if (arg == "-rpath" || arg == "--rpath") {
      if (auto next = consumeNext(i))
        searchDirs.emplace_back(next->str());
      continue;
    }

    // Direct shared-library path. Require the filename to actually look like
    // a shared library before attempting to dlopen it: an existence check
    // alone would happily swallow object files, response files, or any other
    // path the user happened to pass through `-Xlinker`.
    if (hasSharedLibraryExtension(arg)) {
      std::error_code ec;
      if (std::filesystem::exists(arg.str(), ec) && !ec) {
        libraries.emplace_back(arg.str());
        continue;
      }
      state.reportWarning(
          "shared library does not exist for `-Xlinker` argument: '" +
          arg.str() + "'");
      continue;
    }

    state.reportWarning("-Xlinker argument has no effect on `mojo run`: '" +
                        arg.str() + "'");
  }

  for (StringRef libName : deferredLibs) {
    if (auto resolved = findLibraryByName(libName, searchDirs)) {
      libraries.emplace_back(std::move(*resolved));
    } else {
      state.reportWarning("could not locate shared library for '-l" +
                          libName.str() + "' on the linker search path");
    }
  }
  return libraries;
}
