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

#include "Support/Profiling/TimeProfiler.h"
#include "Support/SymbolExport.h"
#include <atomic>

using namespace M;

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TimeTraceProfilerBeginTask(const char *namePtr, size_t nameLen,
                                           size_t parentId, size_t taskId) {
  // NOTE: Must be always enabled.
  return ProfilerEntry<true, Trace::kMojo>::createWithParent(
             parentId, InternableString(namePtr, nameLen), (uint64_t)taskId)
      .getId();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TimeTraceProfilerBeginDetail(const char *namePtr,
                                             size_t nameLen,
                                             const char *detailPtr,
                                             size_t detailLen,
                                             size_t parentId) {
  // NOTE: Must be always enabled.
  return ProfilerEntry<true, Trace::kMojo>::createWithParent(
             parentId, InternableString(namePtr, nameLen),
             StringRef(detailPtr, detailLen))
      .getId();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TimeTraceProfilerBegin(const char *namePtr, size_t nameLen,
                                       size_t parentId) {
  // NOTE: Must be always enabled.
  return ProfilerEntry<true, Trace::kMojo>::createWithParent(
             parentId, InternableString(namePtr, nameLen))
      .getId();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerEnd(size_t id) {
  // NOTE: Must be always enabled.
  ProfilerEntry<true, Trace::kMojo>(id).record();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TimeTraceProfilerGetCurrentId() {
  // NOTE: Must be always enabled.
  return ProfilerEntry<true, Trace::kMojo>::getCurrentId();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerSetCurrentId(size_t id) {
  // NOTE: Must be always enabled.
  ProfilerEntry<true, Trace::kMojo>(id).setAsCurrentId();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_GetNextOpId() {
  static std::atomic<size_t> opIdCounter{0};
  return opIdCounter.fetch_add(1);
}
