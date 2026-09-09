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

#include "Support/Threading/SignalAltStack.h"
#include "llvm/Support/Compiler.h"

#if defined(_WIN32) || defined(_WIN64)
#define MODULAR_HAVE_SIGALTSTACK 0
#else
#define MODULAR_HAVE_SIGALTSTACK 1
#include <cstddef>
#include <cstdlib>
#include <signal.h>
#endif

/// Sanitizer runtimes install SA_ONSTACK handlers of their own and own the
/// crash-reporting path. Installing a stack underneath them would decide where
/// their handlers run, so leave their signal-stack policy alone.
#define MODULAR_INSTALL_SIGNAL_ALT_STACK                                       \
  (MODULAR_HAVE_SIGALTSTACK && !LLVM_ADDRESS_SANITIZER_BUILD &&                \
   !LLVM_HWADDRESS_SANITIZER_BUILD && !LLVM_MEMORY_SANITIZER_BUILD &&          \
   !LLVM_THREAD_SANITIZER_BUILD)

using namespace M;

#if MODULAR_INSTALL_SIGNAL_ALT_STACK
/// Matches llvm::CreateSigAltStack: room for the crash handler to symbolize a
/// backtrace on top of the platform minimum. MINSIGSTKSZ is not a compile-time
/// constant on glibc 2.34 and later.
static size_t getAltStackSize() { return MINSIGSTKSZ + 64 * 1024; }
#endif

bool M::signalAltStackEnabled() { return MODULAR_INSTALL_SIGNAL_ALT_STACK; }

bool M::hasSignalAltStack() {
#if MODULAR_HAVE_SIGALTSTACK
  stack_t current = {};
  if (sigaltstack(nullptr, &current) != 0)
    return false;
  return current.ss_sp != nullptr && !(current.ss_flags & SS_DISABLE);
#else
  return false;
#endif
}

ScopedSignalAltStack::ScopedSignalAltStack() {
#if MODULAR_INSTALL_SIGNAL_ALT_STACK
  // Leave an existing stack alone rather than resizing it the way LLVM does.
  if (hasSignalAltStack())
    return;

  size_t altStackSize = getAltStackSize();
  void *memory = std::malloc(altStackSize);
  if (!memory)
    return;

  stack_t altStack = {};
  altStack.ss_sp = memory;
  altStack.ss_size = altStackSize;
  if (sigaltstack(&altStack, nullptr) != 0) {
    std::free(memory);
    return;
  }

  stack = memory;
  size = altStackSize;
#endif
}

ScopedSignalAltStack::~ScopedSignalAltStack() {
#if MODULAR_INSTALL_SIGNAL_ALT_STACK
  if (!stack)
    return;

  // The constructor only installs when the thread had no stack, so disabling
  // restores the entry state. Darwin rejects SS_DISABLE unless ss_sp and
  // ss_size still describe the stack being removed.
  stack_t altStack = {};
  altStack.ss_flags = SS_DISABLE;
  altStack.ss_sp = stack;
  altStack.ss_size = size;
  sigaltstack(&altStack, nullptr);
  std::free(stack);
#endif
}
