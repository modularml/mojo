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

#ifndef SUPPORT_GLOBALS_GLOBALS_H
#define SUPPORT_GLOBALS_GLOBALS_H

#include "Support/SymbolExport.h"

#include "llvm/ADT/StringMap.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace M {
namespace Detail {
class TypeInfoTable;
}

namespace ProfilingDetail {
struct GlobalProfilerContext;
}

namespace Profiling::Detail {
struct RangeSink;
}

namespace Globals {

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
getGlobalProfilerContext();

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
exchangeGlobalProfilerContext(M::ProfilingDetail::GlobalProfilerContext *ctx);

extern MODULAR_CXX_EXPORT Detail::TypeInfoTable &
getTypeInfoTableSingleton(const std::function<Detail::TypeInfoTable *()> &ctor);

// Process-wide storage for `Config::setGlobalValue()` overrides. These
// live in `libMSupportGlobals.so` so that writes from one shared library
// (e.g. the `max._core` Python extension that wraps `DebugConfig`) are
// visible to reads from another (e.g. `libmax.so` which contains
// `GraphCompiler/FrameworkFrontend`). A function-local static in
// `Support/lib/Configuration.cpp` would give each consumer a distinct
// copy, breaking cross-library propagation.
//
// Callers must hold `getConfigOverridesMutex()` while reading or writing
// `getConfigOverrides()`.
extern MODULAR_CXX_EXPORT std::mutex &getConfigOverridesMutex();
extern MODULAR_CXX_EXPORT llvm::StringMap<std::string> &getConfigOverrides();

// Process-wide hot-path state for //Support:ProfilingRanges. The range shim
// is statically linked into many shared libraries, so this state lives here
// for the same reason as the config overrides above: a gate flip or sink
// attachment performed through one shim copy must be visible to every other
// copy's hot path, and per-copy function-local statics would break that.
// Only the shim (Ranges.cpp) and the profiler integration write these —
// the integration receives the gate addresses through the plugin ABI's
// registerShim (Support/Profiling/PluginABI.h); everything else reads them
// through the inline accessors in Support/Profiling/Ranges.h.
struct ProfilingRangeGlobals {
  // Explicit enable()/disable() intent.
  std::atomic<bool> enabled{false};
  // True while a trace of either origin is live and ranges record.
  std::atomic<bool> recordingActive{false};
  // The attached profiler integration's function table, set when a shim
  // copy's acquire first succeeds (racing copies may each store their own
  // equivalent table; last write wins); valid for the process lifetime.
  std::atomic<const M::Profiling::Detail::RangeSink *> cachedSink{nullptr};
};

extern MODULAR_CXX_EXPORT ProfilingRangeGlobals &getProfilingRangeGlobals();

} // namespace Globals

} // namespace M

#endif // SUPPORT_GLOBALS_GLOBALS_H
