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
// Host-side glue between the external profiler annotation sink
// (Support/Profiling/ExternalProfilerAnnotation.h) and the runtime-loaded
// external profiler annotation shim, libmax_profiler_shim.so.
//
// Provides the STRONG definition of
// M::Profiling::Detail::acquireExternalProfilerAnnotationSink() overriding
// the weak nullptr default in ExternalProfilerAnnotation.cpp — linking
// //Support:ExternalProfilerAnnotationHostGlue into a binary is the only
// wiring there is (the target is alwayslink so the override is never
// dropped from the archive). DSOs that do not link the glue keep the
// inert weak default.
//
// The shim is only dlopen()ed once annotation has actually been requested
// (MODULAR_ENABLE_PROFILING truthy or "detailed", or an explicit
// setEnabled(true) override), from the first emission attempt after that —
// mirroring the shim's own "enabling does not itself load" contract. An
// explicit MODULAR_PROFILER_SHIM path is honored exclusively — a load
// failure warns and disables, with no fallback; otherwise the shim is
// looked for next to this DSO (dladdr), then on the default dlopen search
// path. The handle is deliberately never dlclose()d. An explicit setEnabled
// override recorded before the shim loads is forwarded to
// m_ext_profiler_set_enabled right after loading, so the two master
// switches cannot disagree.
//
// Unlike ProfilerHostGlue there is no Tracy gate: vendor annotation
// libraries do not contend for CUPTI, so Tracy builds keep this loader.
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "Support/Profiling/ExternalProfilerAnnotation.h"
#include "Support/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace M::Profiling::Detail {

#ifndef _WIN32

namespace {

/// The subset of the shim's m_ext_profiler_* C ABI this glue drives
/// (max/internal/profiler/lib/ProfilerShim.h; resolved with dlsym, additive
/// changes only).
struct ShimAPI {
  int (*isEnabled)(void);
  void (*setEnabled)(int enabled);
  int (*rangePush)(const char *message, uint32_t colorARGB);
  void (*rangePop)(void);
  void (*mark)(const char *message, uint32_t colorARGB);
};

struct ShimLoader {
  std::mutex mutex;
  /// Both outcomes are terminal: the shim either loads on the first
  /// requested emission or is unavailable for the life of the process.
  enum class State { NotAttempted, Loaded, Unavailable };
  State state = State::NotAttempted;
  ShimAPI api = {};
};

} // namespace

/// Tri-state master switch mirroring the shim's own: -1 = take the default
/// from the environment on first use, 0/1 = explicit setEnabled override.
static std::atomic<int> &getEnableOverride() {
  static std::atomic<int> enableOverride{-1};
  return enableOverride;
}

static bool envRequestsProfiling() {
  // Mirrors MODULAR_ENABLE_PROFILING semantics from the Python profiler
  // bindings: isTrueLike or exactly "detailed". The shim applies the same
  // rule through its own dependency-free copy.
  const char *state = std::getenv("MODULAR_ENABLE_PROFILING");
  if (state == nullptr)
    return false;
  return isTrueLike(state) || strcmp(state, "detailed") == 0;
}

/// Latches the environment default into the tri-state on first use so the
/// per-span fast path is one relaxed atomic load, never a getenv().
static bool annotationRequestedNow() {
  std::atomic<int> &enableOverride = getEnableOverride();
  int state = enableOverride.load(std::memory_order_relaxed);
  if (state == -1) {
    state = envRequestsProfiling() ? 1 : 0;
    // A concurrent setEnabled wins over the environment default.
    int expected = -1;
    if (!enableOverride.compare_exchange_strong(expected, state,
                                                std::memory_order_relaxed))
      state = expected;
  }
  return state == 1;
}

/// Lock-free mirror of the loader's success state for the per-span paths:
/// set exactly once, after symbol resolution and override forwarding have
/// completed.
static std::atomic<const ShimAPI *> &getCachedApi() {
  static std::atomic<const ShimAPI *> cachedApi{nullptr};
  return cachedApi;
}

static constexpr const char *kShimSoname = "libmax_profiler_shim.so";

/// MODULAR_PROFILER_SHIM, normalized: nullptr when unset OR empty.
static const char *shimEnvPath() {
  const char *path = std::getenv("MODULAR_PROFILER_SHIM");
  return (path != nullptr && *path != '\0') ? path : nullptr;
}

static ShimLoader &getLoader() {
  static ShimLoader loader;
  return loader;
}

/// Directory (with trailing '/') of the DSO or executable this glue copy is
/// linked into — where a shim shipped next to a deployment is looked for.
/// Empty when unresolvable.
static std::string selfDirectory() {
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

template <typename Fn>
static bool resolve(void *lib, const char *name, Fn &slot) {
  slot = reinterpret_cast<Fn>(::dlsym(lib, name));
  return slot != nullptr;
}

/// One load attempt for this glue copy; only reached once annotation has
/// been requested.
static const ShimAPI *attemptLoad() {
  ShimLoader &loader = getLoader();
  std::lock_guard<std::mutex> lock(loader.mutex);
  if (loader.state == ShimLoader::State::Loaded)
    return &loader.api;
  if (loader.state == ShimLoader::State::Unavailable)
    return nullptr;
  loader.state = ShimLoader::State::Unavailable;

  constexpr int kFlags = RTLD_NOW | RTLD_LOCAL;
  void *handle = nullptr;
  if (const char *envPath = shimEnvPath()) {
    // An explicit path is an explicit request: failing to honor it gets a
    // warning rather than silently missing annotations.
    handle = ::dlopen(envPath, kFlags);
    if (handle == nullptr) {
      std::fprintf(stderr,
                   "warning: M::Profiling could not load the "
                   "external profiler annotation shim '%s' "
                   "(MODULAR_PROFILER_SHIM): %s; external profiler "
                   "annotations are unavailable in this process\n",
                   envPath, ::dlerror());
      return nullptr;
    }
  } else {
    std::string sibling = selfDirectory();
    if (!sibling.empty()) {
      sibling += kShimSoname;
      handle = ::dlopen(sibling.c_str(), kFlags);
    }
    if (handle == nullptr)
      handle = ::dlopen(kShimSoname, kFlags);
    if (handle == nullptr)
      return nullptr;
  }

  ShimAPI api = {};
  if (!resolve(handle, "m_ext_profiler_is_enabled", api.isEnabled) ||
      !resolve(handle, "m_ext_profiler_set_enabled", api.setEnabled) ||
      !resolve(handle, "m_ext_profiler_range_push", api.rangePush) ||
      !resolve(handle, "m_ext_profiler_range_pop", api.rangePop) ||
      !resolve(handle, "m_ext_profiler_mark", api.mark)) {
    // Wrong library: treat as absent. The handle is intentionally leaked —
    // another component may legitimately hold it.
    std::fprintf(stderr,
                 "warning: M::Profiling found an "
                 "external profiler annotation shim but it does not export "
                 "the m_ext_profiler_* surface; external profiler "
                 "annotations are unavailable in this process\n");
    return nullptr;
  }

  loader.api = api;
  loader.state = ShimLoader::State::Loaded;
  getCachedApi().store(&loader.api, std::memory_order_release);

  // Forward an explicit override recorded before the shim was loaded, so
  // the shim's own master switch agrees with ours (its environment default
  // already does). The loader mutex serializes this forward with
  // sinkSetEnabled's store-and-forward, so no override is lost and the
  // shim's flag always ends at the latest override.
  const int overrideState = getEnableOverride().load(std::memory_order_relaxed);
  if (overrideState != -1)
    loader.api.setEnabled(overrideState);

  return &loader.api;
}

/// Gate for emission paths: one relaxed atomic load when annotation was
/// never requested, one acquire load once the shim is up.
static const ShimAPI *apiIfRequested() {
  if (!annotationRequestedNow())
    return nullptr;
  if (const ShimAPI *api = getCachedApi().load(std::memory_order_acquire))
    return api;
  return attemptLoad();
}

static bool sinkRangePush(StringRef name, uint32_t colorARGB) {
  const ShimAPI *api = apiIfRequested();
  if (api == nullptr)
    return false;
  // The shim takes a NUL-terminated message; the copy only happens on the
  // enabled path (the vendor annotation APIs copy the message anyway).
  const std::string message(name);
  return api->rangePush(message.c_str(), colorARGB) != 0;
}

static void sinkRangePop() {
  // Callers pop iff their push returned true, so the cached pointer is
  // normally set; a bare pop before any load is a no-op. Bypasses the
  // request gate so a range opened while enabled still closes after
  // setEnabled(false) — the shim's pop has the same contract.
  if (const ShimAPI *api = getCachedApi().load(std::memory_order_acquire))
    api->rangePop();
}

static void sinkMark(StringRef name, uint32_t colorARGB) {
  const ShimAPI *api = apiIfRequested();
  if (api == nullptr)
    return;
  const std::string message(name);
  api->mark(message.c_str(), colorARGB);
}

static void sinkSetEnabled(bool enabled) {
  // Store and forward under the loader mutex: serialized with attemptLoad's
  // load-time forward and with concurrent setEnabled calls, so the shim's
  // flag always ends at the latest override. Enabling does not itself load
  // the shim (the next emission attempt does); before the load the store
  // alone records the override for attemptLoad to forward.
  ShimLoader &loader = getLoader();
  std::lock_guard<std::mutex> lock(loader.mutex);
  getEnableOverride().store(enabled ? 1 : 0, std::memory_order_relaxed);
  if (loader.state == ShimLoader::State::Loaded)
    loader.api.setEnabled(enabled ? 1 : 0);
}

static bool sinkIsEnabled() {
  const ShimAPI *api = apiIfRequested();
  return api != nullptr && api->isEnabled() != 0;
}

static constexpr ExternalProfilerAnnotationSink kSink = {
    .rangePush = sinkRangePush,
    .rangePop = sinkRangePop,
    .mark = sinkMark,
    .setEnabled = sinkSetEnabled,
    .isEnabled = sinkIsEnabled,
};

#endif // !_WIN32

/// Strong override of the weak default in ExternalProfilerAnnotation.cpp
/// (see the file header).
const ExternalProfilerAnnotationSink *acquireExternalProfilerAnnotationSink() {
#ifdef _WIN32
  return nullptr;
#else
  return &kSink;
#endif
}

} // namespace M::Profiling::Detail
