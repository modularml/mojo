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

#include "Support/BazelRunfiles.h"
#include "rules_cc/cc/runfiles/runfiles.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace M;
using llvm::StringRef;
using rules_cc::cc::runfiles::Runfiles;

struct RunfileMapping {
  llvm::StringLiteral
      configKey; // The config key (e.g., "mojo-max.driver_path")
  llvm::StringLiteral workspace; // Empty for _main, otherwise external
                                 // workspace (e.g., "llvm-project")
  llvm::StringLiteral path;    // Path within workspace (without lib prefix/ext)
  bool isSharedLibrary;        // If true, add lib prefix and platform extension
  llvm::StringLiteral libName; // Shared library name (without lib prefix/ext).
  bool searchExecroot = false; // If true, also resolvable from the execroot,
                               // which is how a build action sees its inputs.
};

#ifdef __APPLE__
static constexpr llvm::StringLiteral kSharedLibExt = ".dylib";
#else
static constexpr llvm::StringLiteral kSharedLibExt = ".so";
#endif

// NVIDIA ships libnvptxcompiler as a static archive only, so libNVPTX.so is
// built per host arch from the matching redistributable.
#if defined(__aarch64__)
static constexpr llvm::StringLiteral kNVPTXWorkspace = "nvptxcompiler_aarch64";
#else
static constexpr llvm::StringLiteral kNVPTXWorkspace = "nvptxcompiler_x86_64";
#endif

static constexpr RunfileMapping kRunfileMappings[] = {
    {"crash_reporting.handler_path", "crashpad", "modular-crashpad-handler",
     false, ""},
    {"mojo-max.driver_path", "", "Mojo/tools/mojo/mojo", false, ""},
    {"mojo-max.lld_path", "llvm-project", "lld/lld", false, ""},
    {"mojo-max.lldb_path", "llvm-project", "lldb/lldb", false, ""},
    {"mojo-max.lsp_server_path", "",
     "Mojo/tools/mojo-lsp-server/mojo-lsp-server", false, ""},
    {"mojo-max.repl_entry_point", "",
     "Mojo/tools/mojo-repl-entry-point/mojo-repl-entry-point", false, ""},

    // Shared libraries
    {"mojo-max.mgprt_path", "", "GraphCompiler", true, "MGPRT"},
    {"mojo-max.compilerrt_path", "", "Mojo", true, "KGENCompilerRTShared"},
    {"mojo-max.lldb_plugin_path", "", "Mojo", true, "MojoLLDB"},

    {"max.lib_path", "", "max/internal", true, "max"},

    // Directory paths. The upstream NIXL transport plugins live in per-vendor
    // cpu/, cuda/, cuda-verbs/, rocm/, rocm-verbs/, and rocm-uccl/
    // subdirectories of the @nixl_upstream repo;
    // resolve one plugin file and let the caller derive the vendor directory
    // to use as NIXL_PLUGIN_DIR (see Support/NixlPluginDir.h). Anchor on the
    // cpu flavor: it is the only one staged unconditionally — the GPU flavors
    // are select()ed per vendor in consumers' data deps.
    {"nixl_plugin_dir", "nixl_upstream", "cpu/libplugin_UCX.so", false, ""},

    // The NVPTX compiler (PTX->cubin) is dlopened on demand rather than linked,
    // so no RPATH points at it and it has to be located explicitly. Not a
    // shared-library mapping: the name is fixed by the third-party BUILD file
    // and never takes the .dylib form.
    {"mojo-max.nvptx_path", kNVPTXWorkspace, "libNVPTX.so", false, "",
     /*searchExecroot=*/true},
};

/// Returns nullptr if runfiles cannot be initialized (not running under Bazel)
static Runfiles *getRunfiles() {
  static std::unique_ptr<Runfiles> runfiles =
      []() -> std::unique_ptr<Runfiles> {
    auto rf = std::unique_ptr<Runfiles>(
        Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, nullptr));
    if (rf)
      return rf;

    std::string execPath =
        llvm::sys::fs::getMainExecutable(nullptr, (void *)&getRunfiles);
    return std::unique_ptr<Runfiles>(
        Runfiles::Create(execPath, BAZEL_CURRENT_REPOSITORY, nullptr));
  }();

  return runfiles.get();
}

static std::string buildRunfilePath(const RunfileMapping &mapping) {
  std::string result;

  if (mapping.workspace.empty()) {
    result = "_main/";
  } else {
    result = mapping.workspace.str() + "/";
  }

  result += mapping.path.str();

  if (mapping.isSharedLibrary) {
    result += "/lib";
    result += mapping.libName.str();
    result += kSharedLibExt.str();
  }

  return result;
}

/// Locates a mapping's file at the path a build action stages it to.
///
/// A build action gets its tool's runfiles as plain action inputs, with no
/// runfiles tree and no manifest, so the runfiles library resolves nothing.
/// The files are staged all the same, and an action's working directory is the
/// execroot, so the location is fixed by the output-tree layout.
///
/// It is fixed rather than per-configuration because path mapping
/// (`--experimental_output_paths=strip`, set for every build) rewrites an
/// action's inputs to a configuration-agnostic `bazel-out/cfg/bin/...`, so that
/// otherwise-identical actions can share a cache entry. External repositories
/// sit under that by their bzlmod-canonical name -- the apparent name prefixed
/// with the repository rule's.
///
/// Only build actions lay the tree out this way. Anything with real runfiles --
/// a test, an installed toolchain -- resolves through them before reaching
/// here.
///
/// This cannot be handed down from the Mojo toolchain instead: path mapping
/// rewrites command lines and inputs, but not environment variables, so a path
/// passed that way names an unmapped output directory that the action never
/// stages.
static std::optional<std::string>
findInExecroot(const RunfileMapping &mapping) {
  if (mapping.workspace.empty())
    return std::nullopt;

  llvm::SmallString<128> candidate("bazel-out/cfg/bin/external");
  llvm::sys::path::append(candidate,
                          llvm::Twine("+http_archive+") + mapping.workspace,
                          mapping.path);
  if (!llvm::sys::fs::exists(candidate))
    return std::nullopt;
  return std::string(candidate);
}

std::optional<std::string> M::findConfigWithRunfiles(StringRef key) {
  std::string lowerKey = key.lower();
  const RunfileMapping *mapping = nullptr;
  for (const auto &m : kRunfileMappings) {
    if (m.configKey == lowerKey) {
      mapping = &m;
      break;
    }
  }

  if (!mapping)
    return std::nullopt;

  auto fallback = [&]() -> std::optional<std::string> {
    if (!mapping->searchExecroot)
      return std::nullopt;
    return findInExecroot(*mapping);
  };

  Runfiles *rf = getRunfiles();
  if (!rf)
    return fallback();

  std::string runfilePath = buildRunfilePath(*mapping);
  std::string rlocation = rf->Rlocation(runfilePath);
  if (rlocation.empty())
    return fallback();

  // If the file isn't part of the runfiles, return nothing so looks
  // fallthrough. It might still fail later.
  std::error_code ec;
  if (!std::filesystem::exists(rlocation, ec))
    return fallback();

  return rlocation;
}

const std::vector<std::pair<std::string, std::string>> *
M::getRunfilesEnvVars() {
  Runfiles *rf = getRunfiles();
  if (!rf)
    return nullptr;
  return &rf->EnvVars();
}
