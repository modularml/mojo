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

#ifndef ASYNCRT_SUPPORT_THREADAFFINITY_H
#define ASYNCRT_SUPPORT_THREADAFFINITY_H

#include "Support/ErrorOr.h"
#include "Support/Threading/HWInfo.h"

#include "llvm/ADT/FunctionExtras.h"

#include <cstddef>
#include <vector>

namespace M::AsyncRT {

/// Determine the number of threads to use (based on the existing suggestion),
/// and return a vector of CPU IDs for every such thread. The CPU ids may be
/// kNoAffinity, indicating no affinity should be set. On error attempt to
/// fallback to defaults, and return error to the caller if the attempt
/// fails. If withAffinity is false, then expected result is a vector
/// containing all entries with kNoAffinity.
M::ErrorOr<std::vector<size_t>> getThreadAffinityCpuIds(bool withAffinity,
                                                        size_t numThreads,
                                                        size_t maxThreads);

/// Executes workFn with thread execution affinity to the specified CPU core and
/// sets the memory policy to the NUMA node that CPU core resides in if
/// possible.
void runWithThreadAffinity(size_t cpuID, llvm::function_ref<void()> workFn);

/// Sets the execution affinity to the specified CPU core and sets the memory
/// policy to the NUMA node that CPU core resides in if possible.
void setThreadAffinity(size_t cpuID);

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_THREADAFFINITY_H
