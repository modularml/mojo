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

#ifndef SUPPORT_PROCESS_H
#define SUPPORT_PROCESS_H

#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"

#include <cstddef>
#include <string>
#include <vector>

namespace M {
/// Set the environment variable `name` to `value`. If `overwrite` is false, the
/// variable will not be set if it already exists.
/// TODO: This should be upstreamed to llvm::sys::Process to match the GetEnv
///       method.
LogicalResult setProcessEnv(StringRef name, StringRef value,
                            bool overwrite = true);

/// Get a list of all the environment variables of the current process.
std::vector<StringRef> getEnv();

//===----------------------------------------------------------------------===//
// Memory usage
//===----------------------------------------------------------------------===//

/// Returns the current process' physical memory usage, or 0 if value is
/// not available. Generally determined from the OS's reported resident
/// page value, and may not very reliable.
size_t getProcessPhysicalMemUsage();

// Returns the full path to the current executable, or "<unknown>" if the path
// cannot be obtained.
std::string getProcessExecutablePath();
} // namespace M

#endif // SUPPORT_PROCESS_H
