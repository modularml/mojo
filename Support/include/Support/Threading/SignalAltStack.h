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

#ifndef SUPPORT_THREADING_SIGNALALTSTACK_H
#define SUPPORT_THREADING_SIGNALALTSTACK_H

#include <cstddef>

namespace M {
//===----------------------------------------------------------------------===//
// Alternate signal stack
//===----------------------------------------------------------------------===//

/// Installs an alternate signal stack for the calling thread, and removes it
/// again when destroyed. A crash handler registered with `SA_ONSTACK` (as
/// LLVM's is) then runs on intact memory even when the thread has exhausted
/// its own stack, which is what makes a stack overflow reportable rather than
/// an immediate second fault.
///
/// `sigaltstack()` is per-thread state, and LLVM establishes one only on the
/// thread that first registers the crash handlers. Any other thread that runs
/// deeply recursive work needs one of these alive for the duration of that
/// work, so construct it as a local at the top of the thread body.
///
/// Does nothing when the thread already has an alternate stack, on platforms
/// without `sigaltstack()`, or on sanitizer builds.
class ScopedSignalAltStack {
public:
  ScopedSignalAltStack();
  ~ScopedSignalAltStack();

  ScopedSignalAltStack(const ScopedSignalAltStack &) = delete;
  ScopedSignalAltStack &operator=(const ScopedSignalAltStack &) = delete;

private:
  /// The allocation given to `sigaltstack()`, or null if none was installed.
  void *stack = nullptr;
  /// Size of `stack`. Darwin needs it again to remove the stack.
  size_t size = 0;
};

/// Returns true if the calling thread has an alternate signal stack installed,
/// no matter if installed by us or by any other tool (e.g. ASAN).
bool hasSignalAltStack();

/// Returns true if `ScopedSignalAltStack` installs anything on this build.
/// Returns false in sanitizer builds, whose runtimes own the crash-reporting
/// path, and on platforms without `sigaltstack()`.
bool signalAltStackEnabled();

} // namespace M

#endif // SUPPORT_THREADING_SIGNALALTSTACK_H
