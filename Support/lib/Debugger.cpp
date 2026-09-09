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

#include "Support/Debugger.h"
#include "Support/Process.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"

#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <csignal>
#include <thread>

using namespace M;

void M::waitForDebuggerToAttach(int timeoutSeconds) {
  int pid = llvm::sys::Process::getProcessId();
  llvm::errs() << "======== Pausing for debugger attach ========\n";
  llvm::errs() << "Process name: " << getProcessExecutablePath() << "\n";
  llvm::errs() << "Process ID: " << pid << "\n";
  llvm::errs() << "\n";
  llvm::errs() << "Attach to this process using one of these options:\n";
  llvm::errs() << "  * vmojo debug --pid " << pid
               << " (recommended for VS Code users)\n";
  llvm::errs() << "  * br //:lldb -- -p " << pid
               << " (to use monorepo build of lldb)\n";
  llvm::errs() << "  * lldb -p " << pid << " (to use installed lldb)\n";
  llvm::errs() << "Waiting for " << timeoutSeconds << " seconds...\n";
#ifdef _WIN32
  while (!IsDebuggerPresent())
    Sleep(1000);
#elif defined(__APPLE__)
  raise(SIGSTOP);
#else
  llvm::errs() << "Waiting for " << timeoutSeconds << " seconds...\n";
  std::this_thread::sleep_for(std::chrono::seconds(timeoutSeconds));
#endif
  llvm::errs() << "======= Resuming execution ========\n";
}
