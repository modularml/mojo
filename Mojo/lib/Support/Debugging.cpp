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

#include "Mojo/Support/Debugging.h"
#include "Mojo/Support/Configuration.h"
#include "Support/Debugger.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <csignal>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace M;

void M::attachToNewRemoteDebugSession(bool quiet) {
  StringRef initializationError =
      "couldn't initiate the debug session. You might want to attach manually "
      "to this process";

  // Find the path to the mojo executable.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr)) {
    llvm::errs() << "error: " << initializationError << ": "
                 << configOr.takeError() << "\n";
  } else {
    std::error_code ec;
    StringRef mojo = configOr->getDriverPath();
    if (!std::filesystem::exists(mojo.str(), ec) || ec) {
      llvm::errs()
          << "error: " << initializationError
          << ": unable to resolve the mojo path from the modular.cfg\n";
    } else {
      std::string pidStr = std::to_string(llvm::sys::Process::getProcessId());
      SmallVector<StringRef> args{mojo, "debug", "--vscode", "--pid", pidStr};

      SmallVector<std::optional<StringRef>> redirects;
      if (quiet) {
        redirects = {
            "",
            "",
            "",
        };
      }

      // `mojo debug --vscode` succeeds if lldb-dap was launched, but it might
      // still be possible that the actual attach failed.
      int exitCode = llvm::sys::ExecuteAndWait(mojo, args, /*Env=*/std::nullopt,
                                               /*Redirects=*/redirects);
      if (exitCode != 0) {
        llvm::errs()
            << "error: the remote debugger seems to have failed to attach. "
               "You might need attach manually to this process\n";
      }
    }
  }
  waitForDebuggerToAttach();
}
