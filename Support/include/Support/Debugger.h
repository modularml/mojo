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
//
// Utilities to help C++ debuggers
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGGER_H
#define SUPPORT_DEBUGGER_H

namespace M {

/// Print the executable name and process ID to standard output, then pause
/// for the specified number of seconds on Linux (default to 120 seconds) and
/// wait. On Mac or Windows, it waits indefinitely.
///
/// The reason behind the differences in behavior for each platform are due to
/// how the debugger resumes after a wait or SIGSTOP pauses the program. For
/// example, on Mac, after resuming a process stopped with `sleep_for`, you
/// continue the wait. However, on Linux, you don't wait anymore and simply
/// continue the rest of the execution of the process.
void waitForDebuggerToAttach(int timeoutSeconds = 120);

} // namespace M

#define DEBUGGER_ATTACH_IF_NOT(condition)                                      \
  if (!(condition))                                                            \
    waitForDebuggerToAttach(30);

#endif // SUPPORT_DEBUGGER_H
