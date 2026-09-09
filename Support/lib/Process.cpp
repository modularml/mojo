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

#include "Support/Process.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include <cstddef>
#include <memory>
#include <stdlib.h>
#include <system_error>
#include <vector>

#ifdef __APPLE__
#include <mach/mach_init.h>
#include <mach/task.h>
#endif // __APPLE__

#ifdef _WIN32
#include <windows.h>
#endif

using namespace M;

LogicalResult M::setProcessEnv(StringRef name, StringRef value,
                               bool overwrite) {
#ifdef _WIN32
  if (!overwrite && llvm::sys::Process::GetEnv(name))
    return success();
  int result = SetEnvironmentVariableA(name.str().data(), value.str().data());
  return success(result != 0);
#else
  int result = setenv(name.str().data(), value.str().data(), overwrite);
  return success(result == 0);
#endif
}

#if defined(_WIN32)
extern char **_environ;
#else
extern char **environ;
#endif

std::vector<StringRef> M::getEnv() {
#ifdef _WIN32
  static char **envp = _environ;
#else
  static char **envp = environ;
#endif
  std::vector<StringRef> env;
  for (char **entry = envp; *entry; ++entry)
    env.emplace_back(*entry);
  return env;
}

//===----------------------------------------------------------------------===//
// Memory usage
//===----------------------------------------------------------------------===//

size_t M::getProcessPhysicalMemUsage() {
#if defined(__linux__)
  // On linux we'll use the (approximate) process resident number of pages.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream("/proc/self/statm");
  if (std::error_code ec = errOrBuf.getError())
    return 0;
  StringRef buffer = (*errOrBuf)->getBuffer();
  // Buffer will be "size resident shared text lib data dt", all as num pages.
  SmallVector<StringRef, 7> strs;
  buffer.split(strs, " ");
  if (strs.size() != 7)
    return 0;
  size_t value;
  if (strs[1].getAsInteger(10, value))
    return 0;
  // Convert from pages to bytes.
  return value * llvm::sys::Process::getPageSizeEstimate();
#elif defined(__APPLE__)
  struct task_basic_info info;
  unsigned count = TASK_BASIC_INFO_COUNT;
  kern_return_t result =
      task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count);
  if (result != KERN_SUCCESS)
    return 0;
  return info.resident_size;
#else
  return 0;
#endif
}

std::string M::getProcessExecutablePath() {
  std::string executable = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  if (executable.empty())
    return "<unknown>";
  return executable;
}
