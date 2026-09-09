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

#ifndef SUPPORT_THREADING_THREADAFFINITY_H
#define SUPPORT_THREADING_THREADAFFINITY_H

#include "Support/ForwardDecls.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>

namespace M {
//===----------------------------------------------------------------------===//
// Thread affinity
//===----------------------------------------------------------------------===//

/// Returns true if thread affinity is available on this target.
bool haveThreadAffinity();

/// Sets the execution affinity to the specified CPU core and sets the memory
/// policy to the NUMA node that CPU core resides in if possible.
ErrorOrSuccess setThreadAffinity(size_t cpuID);

/// Executes workFn with thread execution affinity to the specified CPU core and
/// sets the memory policy to the NUMA node that CPU core resides in if
/// possible.
ErrorOrSuccess runWithThreadAffinity(size_t cpuID,
                                     llvm::function_ref<void()> workFn);
} // namespace M

#endif // SUPPORT_THREADING_THREADAFFINITY_H
