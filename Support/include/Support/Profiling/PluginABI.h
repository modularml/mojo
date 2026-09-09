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
// C ABI between the host glue (ProfilerHostGlue.cpp, the strong RangeSink
// provider behind Support/Profiling/Ranges.h) and the runtime-loaded profiler
// plugin, libMAXProfilerPlugin.so. The glue resolves the single exported entry
// point, M_profilerPluginGetAPI, and forwards every profiling call through
// the returned function-pointer table.
//
// Backend neutrality: nothing in this ABI names any profiling backend or
// vendor library. Hardware backends live entirely inside the plugin (behind
// its internal backend table, RangeBackend.h), so adding a backend never
// changes this interface.
//
// Lifetime contract: the shim never closes the plugin, and a DSO hosting
// a registered shim copy must never be unloaded — registerShim hands the
// plugin raw pointers into the host DSO's storage (M_ProfilerShimGates),
// which the plugin writes for the remaining life of the process. A host
// DSO that could be dynamically unloaded must therefore not link the shim.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_PLUGINABI_H
#define SUPPORT_PROFILING_PLUGINABI_H

#include <stdbool.h> // NOLINT(modernize-deprecated-headers): C ABI header.
#include <stddef.h>  // NOLINT(modernize-deprecated-headers)
#include <stdint.h>  // NOLINT(modernize-deprecated-headers)

// Version negotiation: the shim passes its major to M_profilerPluginGetAPI,
// which returns NULL on a major it does not support (the shim then treats
// the plugin as absent). Bump MAJOR on any breaking layout or semantic
// change; bump MINOR when appending fields to M_ProfilerPluginAPI. A shim
// must not read fields beyond the plugin's structSize — the 1.0 shim reads
// no appended fields, so it carries no such check yet — which is what lets
// newer shims degrade gracefully against older plugins and vice versa.
#define M_PROFILER_ABI_MAJOR 1
#define M_PROFILER_ABI_MINOR 0

// Filename the shim probes when MODULAR_PROFILER_PLUGIN does not name an
// explicit path: next to the host DSO first, then the default dlopen search.
#define M_PROFILER_PLUGIN_SONAME "libMAXProfilerPlugin.so"
#define M_PROFILER_PLUGIN_ENTRY_SYMBOL "M_profilerPluginGetAPI"

#ifdef __cplusplus
extern "C" {
#endif

// The hot-path gates a shim copy asks the plugin to drive. Both pointers
// are std::atomic<bool>* (typed void* to keep the struct C-clean); they
// must stay valid for the life of the process (see the lifetime contract
// above). The plugin is the single source of truth: it stores the
// authoritative values through every registered pointer whenever they
// change, so the shim's inline isEnabled()/isRangeRecordingActive() stay
// one relaxed atomic load with no cross-DSO call. The shim's gates live in
// the process-global block in libMSupportGlobals.so, so every registered
// copy's pointers alias the same two atomics and the mirror writes all land
// on one object; the ABI keeps the addresses explicit so a shim build with
// per-copy gates would still work.
typedef struct M_ProfilerShimGates {
  // Mirrors the session API's enable intent (enable()/disable()).
  void *enabledGate;
  // Mirrors "a trace of either origin is live and ranges record".
  void *recordingActiveGate;
} M_ProfilerShimGates;

// The plugin's call table. All functions are process-global operations with
// the exact semantics of the M::Profiling API (Range.h); strings cross the
// boundary as (pointer, length) or copy-out buffers so no allocator or
// std::string layout is shared between shim and plugin.
typedef struct M_ProfilerPluginAPI {
  uint32_t abiMajor;
  uint32_t abiMinor;
  // sizeof(*this) on the plugin side; a shim may only read fields that fit
  // inside structSize.
  uint32_t structSize;

  // Registers one shim copy's gates. Idempotent (deduplicated by pointer).
  // Seeds the copy's gates with the current authoritative values under the
  // plugin's gate lock, so a copy registering mid-trace observes it.
  void (*registerShim)(const M_ProfilerShimGates *gates);

  // Lifecycle — 1:1 with the M::Profiling functions of the same name.
  void (*enable)(void);
  void (*disable)(void);
  void (*activatePendingTrace)(void);
  void (*waitForTrace)(void);
  void (*step)(void);
  // Returns an M::Profiling::ProfilerState value.
  int (*state)(void);
  bool (*canRecord)(void);
  // Copies the message (NUL-terminated) into buf, truncating to bufSize;
  // returns the full untruncated length so the shim can retry with a larger
  // buffer.
  size_t (*lastTraceError)(char *buf, size_t bufSize);

  // Range recording. `name` need not be NUL-terminated; `nameLen` governs.
  void (*rangeBegin)(const char *name, size_t nameLen, uint32_t color);
  void (*rangeBeginWithId)(uint64_t correlationId, const char *name,
                           size_t nameLen, uint32_t color);
  void (*rangeEnd)(void);
  // Names the calling thread's track in the trace; must run on that thread.
  void (*registerCurrentThread)(void);
} M_ProfilerPluginAPI;

// The single symbol libMAXProfilerPlugin.so exports. Returns NULL when
// shimABIMajor is unsupported. The first successful call performs the
// plugin's one-time backend hookup, so a returned table is ready to use.
typedef const M_ProfilerPluginAPI *(*M_ProfilerPluginGetAPIFn)(
    uint32_t shimABIMajor);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SUPPORT_PROFILING_PLUGINABI_H
