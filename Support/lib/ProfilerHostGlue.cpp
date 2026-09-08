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
// Host-side glue between the minimal range API (//Support:ProfilingRanges) and
// the runtime-loaded profiler plugin, libMAXProfilerPlugin.so.
//
// Provides the STRONG definition of M::Profiling::Detail::acquireRangeSink()
// overriding the weak nullptr default in Ranges.cpp — linking
// //Support:ProfilerHostGlue into a binary is the only wiring there is (the
// target is alwayslink so the override is never dropped from the archive).
// DSOs that do not link the glue keep the inert weak default.
//
// Loading policy (one attempt state per glue copy — one per host DSO):
//
//   * SinkRequest::Attach (enable, haveProfiler, canRecord) performs a full
//     load attempt: MODULAR_PROFILER_PLUGIN (explicit path) → next to the
//     host DSO (dladdr) → default dlopen search.
//   * SinkRequest::DeviceInit (activatePendingTrace) performs a full
//     attempt only when something asked for a profiler without calling
//     enable() first: daemon-style setups (KINETO_USE_DAEMON or
//     MODULAR_PROFILER_PLUGIN set, preserving Dynolog on-demand
//     registration) and the max-debug.profiling-enabled auto-start knob.
//     The knob must make the plugin loadable here and not only at
//     InferenceSession construction: on AMD the profiler can only attach
//     to the GPU runtime BEFORE the first HIP call, which device creation
//     (e.g. a Python Accelerator()) may issue before any session exists.
//     Loading is all that happens early — the trace still starts at the
//     session's construction-time enable(), as on NVIDIA.
//   * SinkRequest::Observe (disable, state, waitForTrace, lastTraceError)
//     only ADOPTS a plugin some other copy in the process already loaded,
//     via a cheap RTLD_NOLOAD probe. Adoption is what keeps the several
//     statically linked copies in one process (max._core, libmax, the mach
//     shim) attached to the single plugin instance.
//
// On success the glue registers the shim's gate addresses with the plugin
// (registerShim), which then stores every authoritative gate transition
// through them; the gates live in the process-global block in
// libMSupportGlobals.so, so every copy registers pointers to the same two
// atomics. The plugin handle is deliberately never dlclose()d
// (rationale in PluginABI.h). In Tracy builds the loader compiles to
// "always unavailable": Tracy owns CUPTI, and the two GPU profilers are
// mutually exclusive.
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "Support/Configuration.h"
#include "Support/Profiling/PluginABI.h"
#include "Support/Profiling/Ranges.h"

#include "llvm/ADT/StringRef.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace M::Profiling::Detail {
namespace {

// Lock-free mirror of the loader's success state for the sink's recording
// entries: set exactly once, after registerShim has completed.
std::atomic<const M_ProfilerPluginAPI *> &getCachedApi() {
  static std::atomic<const M_ProfilerPluginAPI *> cachedApi{nullptr};
  return cachedApi;
}

// MODULAR_PROFILER_PLUGIN, normalized: nullptr when unset OR empty, so every
// consult of the variable (the loader, the daemon-mode full-load trigger,
// the Tracy-build warning) agrees on what "set" means.
const char *pluginEnvPath() {
  const char *path = std::getenv("MODULAR_PROFILER_PLUGIN");
  return (path != nullptr && *path != '\0') ? path : nullptr;
}

// The max-debug.profiling-enabled auto-start knob (settable via the
// MODULAR_MAX_DEBUG_PROFILING_ENABLED / MODULAR_DEBUG env vars or
// modular.cfg) — the same knob InferenceSession's construction-time
// enable() reads. Consulted only on the rare DeviceInit path, never per
// range.
bool profilingConfigEnabled() {
  auto configOr = M::Config::open();
  if (configOr.isError())
    return false;
  return configOr->getValueAsBool("max-debug.profiling-enabled",
                                  /*defaultValue=*/false);
}

#if !MODULAR_TRACY_BUILD && !defined(_WIN32)

struct PluginLoader {
  std::mutex mutex;
  // Terminal states: Loaded, Unavailable. NotAttempted allows later
  // adoption/full-load retries (an adopt-only probe that misses must not
  // block a subsequent enable() from loading for real).
  enum class State { NotAttempted, Loaded, Unavailable };
  State state = State::NotAttempted;
  const M_ProfilerPluginAPI *api = nullptr;
  bool warned = false;
};

PluginLoader &getLoader() {
  static PluginLoader loader;
  return loader;
}

// Directory (with trailing '/') of the DSO or executable this glue copy is
// linked into — the first place a plugin dropped next to a deployment is
// looked for. Empty when unresolvable.
std::string selfDirectory() {
  Dl_info info;
  if (::dladdr(reinterpret_cast<void *>(&getLoader), &info) == 0 ||
      info.dli_fname == nullptr)
    return {};
  std::string path = info.dli_fname;
  const auto slash = path.rfind('/');
  if (slash == std::string::npos)
    return {};
  path.resize(slash + 1);
  return path;
}

void warnOnce(PluginLoader &loader, const char *fmt, const char *a,
              const char *b) {
  if (loader.warned)
    return;
  loader.warned = true;
  std::fprintf(stderr, fmt, a, b);
}

// dlopen `path`, remembering which candidate produced the handle so the
// ABI-mismatch diagnostic can name the offending file.
void *probePlugin(const char *path, int flags, const char *&loadedFrom) {
  void *handle = ::dlopen(path, flags);
  if (handle != nullptr)
    loadedFrom = path;
  return handle;
}

// One load attempt for this glue copy. `allowFullLoad` distinguishes the
// triggers that may pull the plugin into the process from the adopt-only
// calls that merely attach to a copy something else already loaded.
const M_ProfilerPluginAPI *attemptLoad(bool allowFullLoad) {
  PluginLoader &loader = getLoader();
  std::lock_guard<std::mutex> lock(loader.mutex);
  if (loader.state == PluginLoader::State::Loaded)
    return loader.api;
  if (loader.state == PluginLoader::State::Unavailable)
    return nullptr;

  const char *envPath = pluginEnvPath();
  constexpr int kFlags = RTLD_NOW | RTLD_LOCAL;

  // Adopt-first: RTLD_NOLOAD returns a handle only if the file is already
  // mapped, so this never pulls the plugin in — it just attaches this copy
  // to the process's single instance.
  void *handle = nullptr;
  const char *loadedFrom = nullptr;
  if (envPath != nullptr)
    handle = probePlugin(envPath, kFlags | RTLD_NOLOAD, loadedFrom);
  std::string sibling = selfDirectory();
  if (!sibling.empty())
    sibling += M_PROFILER_PLUGIN_SONAME;
  if (handle == nullptr && !sibling.empty())
    handle = probePlugin(sibling.c_str(), kFlags | RTLD_NOLOAD, loadedFrom);
  if (handle == nullptr)
    handle =
        probePlugin(M_PROFILER_PLUGIN_SONAME, kFlags | RTLD_NOLOAD, loadedFrom);

  if (handle == nullptr && allowFullLoad) {
    if (envPath != nullptr) {
      // An explicit path is an explicit request: failing to honor it gets a
      // (one-time) warning rather than a silent no-op profiler.
      handle = probePlugin(envPath, kFlags, loadedFrom);
      if (handle == nullptr)
        warnOnce(loader,
                 "warning: M::Profiling could not load the profiler plugin "
                 "'%s' (MODULAR_PROFILER_PLUGIN): %s; profiling is "
                 "unavailable in this process\n",
                 envPath, ::dlerror());
    } else {
      if (!sibling.empty())
        handle = probePlugin(sibling.c_str(), kFlags, loadedFrom);
      if (handle == nullptr)
        handle = probePlugin(M_PROFILER_PLUGIN_SONAME, kFlags, loadedFrom);
    }
  }

  if (handle == nullptr) {
    // A failed full load is terminal; a missed adopt-only probe is not — a
    // later enable() may still load, or another copy may load first.
    if (allowFullLoad)
      loader.state = PluginLoader::State::Unavailable;
    return nullptr;
  }

  const auto getApi = reinterpret_cast<M_ProfilerPluginGetAPIFn>(
      ::dlsym(handle, M_PROFILER_PLUGIN_ENTRY_SYMBOL));
  const M_ProfilerPluginAPI *api =
      getApi != nullptr ? getApi(M_PROFILER_ABI_MAJOR) : nullptr;
  if (api == nullptr) {
    // Wrong library or incompatible ABI major: treat as absent. The handle
    // is intentionally leaked — another copy may legitimately hold it.
    warnOnce(loader,
             "warning: M::Profiling found a profiler plugin at '%s' but it "
             "does not support ABI major %s; profiling is unavailable in "
             "this process\n",
             loadedFrom, std::to_string(M_PROFILER_ABI_MAJOR).c_str());
    loader.state = PluginLoader::State::Unavailable;
    return nullptr;
  }

  // Hand the plugin the shim's gate addresses (process-global, so every
  // copy registers the same two atomics); it seeds them with the current
  // authoritative values, so adopting mid-trace flips the gates before the
  // first post-adoption read.
  static const M_ProfilerShimGates kGates = {&getEnabledGate(),
                                             &getRecordingGate()};
  api->registerShim(&kGates);
  loader.api = api;
  loader.state = PluginLoader::State::Loaded;
  getCachedApi().store(api, std::memory_order_release);
  return api;
}

#else // MODULAR_TRACY_BUILD || _WIN32

// Tracy owns CUPTI (the two GPU profilers are mutually exclusive), so Tracy
// builds never load the plugin; there is no Windows host either way.
const M_ProfilerPluginAPI *attemptLoad(bool allowFullLoad) {
  if (allowFullLoad && pluginEnvPath() != nullptr) {
    static std::once_flag once;
    std::call_once(once, [] {
      std::fprintf(stderr,
                   "warning: MODULAR_PROFILER_PLUGIN is set but this build "
                   "cannot host the profiler plugin (Tracy owns CUPTI); "
                   "profiling is unavailable in this process\n");
    });
  }
  return nullptr;
}

#endif // MODULAR_TRACY_BUILD || _WIN32

// The RangeSink handed back to //Support:ProfilingRanges — thin forwarders into
// the plugin's C call table. Only invoked after a successful attach, so the
// cached api pointer is always set (the recording entries additionally ride
// behind the gates, which only the plugin raises).
const M_ProfilerPluginAPI *api() {
  return getCachedApi().load(std::memory_order_acquire);
}

void sinkEnable() { api()->enable(); }
void sinkDisable() { api()->disable(); }
void sinkActivatePendingTrace() { api()->activatePendingTrace(); }
void sinkWaitForTrace() { api()->waitForTrace(); }
void sinkStep() { api()->step(); }
int sinkState() { return api()->state(); }
bool sinkCanRecord() { return api()->canRecord(); }

std::string sinkLastTraceError() {
  char buf[512];
  const size_t len = api()->lastTraceError(buf, sizeof(buf));
  if (len < sizeof(buf))
    return std::string(buf, len);
  // Rare long message: retry with a right-sized buffer. The message can
  // change between the two calls; the min() keeps the copy in bounds.
  std::string big(len + 1, '\0');
  const size_t len2 = api()->lastTraceError(big.data(), big.size());
  big.resize(len2 < len ? len2 : len);
  return big;
}

void sinkRangeBegin(StringRef name, uint32_t color) {
  api()->rangeBegin(name.data(), name.size(), color);
}

void sinkRangeBeginWithId(uint64_t correlationId, StringRef name,
                          uint32_t color) {
  api()->rangeBeginWithId(correlationId, name.data(), name.size(), color);
}

void sinkRangeEnd() { api()->rangeEnd(); }

void sinkRegisterCurrentThread() { api()->registerCurrentThread(); }

constexpr RangeSink kSink = {
    .enable = sinkEnable,
    .disable = sinkDisable,
    .activatePendingTrace = sinkActivatePendingTrace,
    .waitForTrace = sinkWaitForTrace,
    .step = sinkStep,
    .state = sinkState,
    .canRecord = sinkCanRecord,
    .lastTraceError = sinkLastTraceError,
    .rangeBegin = sinkRangeBegin,
    .rangeBeginWithId = sinkRangeBeginWithId,
    .rangeEnd = sinkRangeEnd,
    .registerCurrentThread = sinkRegisterCurrentThread,
};

} // namespace

// Strong override of the weak default in Ranges.cpp (see the file header).
const RangeSink *acquireRangeSink(SinkRequest request) {
  bool allowFullLoad = false;
  switch (request) {
  case SinkRequest::Attach:
    allowFullLoad = true;
    break;
  case SinkRequest::DeviceInit:
    // Daemon-style setups attach at device initialization even though
    // nothing enabled profiling, so Dynolog on-demand capture works with no
    // flags; the auto-start knob attaches here too because on AMD device
    // init is the last moment a profiler can hook the GPU runtime (see the
    // file header). Otherwise device init only adopts.
    allowFullLoad = std::getenv("KINETO_USE_DAEMON") != nullptr ||
                    pluginEnvPath() != nullptr || profilingConfigEnabled();
    break;
  case SinkRequest::Observe:
    break;
  }
  return attemptLoad(allowFullLoad) != nullptr ? &kSink : nullptr;
}

} // namespace M::Profiling::Detail
