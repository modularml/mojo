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

#include "LLDB.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Mojo/Support/Configuration.h"
#include "Support/BazelRunfiles.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/Process.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include <filesystem>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#define DRIVER_OPTIONS_PATH "Debug/DebugOptions.inc"
#include "Support/Driver/OptTable.inc"

using namespace M;

/// Returns the path to the `lldb` executable, or an error if not found.
static ErrorOr<std::string> getLLDB(KGEN::MojoConfig &config) {
  std::error_code ec;
  StringRef lldb = config.getLLDBPath();
  if (!std::filesystem::exists(lldb.str(), ec) || ec)
    return Error("unable to resolve the lldb path, try installing `mojo`");
  return lldb.str();
}

/// Returns the path to the MojoLLDB shared library, or an error if not found.
/// This library implements Mojo's LLDB plugin.
static ErrorOr<std::string> getMojoLLDB(KGEN::MojoConfig &config) {
  std::error_code ec;
  StringRef mojoLLDB = config.getLLDBPluginPath();
  if (!std::filesystem::exists(mojoLLDB.str(), ec) || ec)
    return Error("unable to resolve the MojoLLDB plugin path");
  return mojoLLDB.str();
}

int M::invokeLLDB(const State &state, ArrayRef<std::string> lldbArgs,
                  ArrayRef<std::string> runArgs, bool dryRun) {
  // Find the path to the LLDB executable and the MojoLLDB plugin library.
  // Read the mojo configuration.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr)) {
    return state.reportError(Twine("failed to parse 'modular.cfg': ") +
                             configOr.getError());
  }

  KGEN::MojoConfig config = std::move(*configOr);
  ErrorOr<std::string> lldb = getLLDB(config);
  if (failed(lldb))
    return state.reportError(lldb.getError());
  ErrorOr<std::string> mojoLLDB = getMojoLLDB(config);
  if (failed(mojoLLDB))
    return state.reportError(mojoLLDB.getError());

  std::string loadCommand = llvm::formatv("plugin load \"{0}\"", *mojoLLDB);
  SmallVector<StringRef> subprocessArgs = {
      lldb.get(), "-Q", "--one-line-before-file", loadCommand};

  llvm::append_range(subprocessArgs, lldbArgs);

  // LLDB guarantees that any arguments that come after `--` are considered
  // run arguments of the debuggee.
  if (!runArgs.empty()) {
    subprocessArgs.push_back("--");
    llvm::append_range(subprocessArgs, runArgs);
  }

  if (dryRun) {
    llvm::interleave(
        subprocessArgs, llvm::outs(),
        [](StringRef arg) {
          // This is just a simple shell quoting mechanism. We don't use
          // anything fancy because the dry-run command is intended for
          // internal development.
          if (arg.contains(' '))
            llvm::outs() << "'" << arg << "'";
          else
            llvm::outs() << arg;
        },
        " ");
    llvm::outs() << "\n";
    return 0;
  }

  // The MojoLLDB plugin resolves its paths through the runfiles tree, but in
  // order to discover them we need to pass through our environment. Otherwise
  // those lookups fall through to the package root and name files that don't
  // exist.
  const auto *env = M::getRunfilesEnvVars();
  if (env)
    for (const auto &[name, value] : *env)
      (void)M::setProcessEnv(name, value);

  return llvm::sys::ExecuteAndWait(lldb.get(), subprocessArgs);
}
