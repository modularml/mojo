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
// Minimal range-annotation and profiler-control API.
//
// This header carries no profiler implementation: it declares the hot-path
// gates, the RAII range helper, and control entry points that forward to a
// RangeSink — a function table an optional host-side profiler integration
// provides by overriding the weak Detail::acquireRangeSink() at link time.
// When no sink is linked (e.g. the open-source compiler stack) every call
// degrades to a safe no-op, so any binary in the repo can link
// //Support:ProfilingRanges unconditionally: no build config, nothing
// profiler-specific in this tree, and the only runtime dependency is
// libMSupportGlobals.so — the process-global singleton library every
// runtime host already carries — which holds the gates so all statically
// linked copies of the shim observe the same state.
//
// Off-cost: when disabled, every call checks a single relaxed atomic bool
// and returns. Branch predictors eliminate the cost in steady state, so the
// API is safe to use on the hot path.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_RANGES_H
#define SUPPORT_PROFILING_RANGES_H

#include <atomic>
#include <cstdint>
#include <string>

#include "Support/LLVMForwardDecls.h"

namespace M::Profiling {

namespace Detail {
// Implementation-detail master enable flag. Read via the inline
// ``isEnabled()`` accessor below — direct callers must not store() to it;
// doing so bypasses ``enable()`` / ``disable()``'s serialization and will
// desync the profiler state.  Lives behind a function-local-static accessor
// (rather than an extern global) so there is no static-initialization-order
// hazard.
//
// The library is always linked statically, but the flag itself lives in the
// process-global block in libMSupportGlobals.so, so every DSO's copy of
// this accessor returns the same object: a gate flip performed through any
// copy is immediately visible to every other copy's hot path — coherence
// needs no per-copy registration or mirroring (the plugin ABI's
// registerShim still registers gate addresses; see PluginABI.h).
std::atomic<bool> &getEnabledGate() noexcept;
} // namespace Detail

// Cheap read-only query: returns true while the profiler is enabled. Single
// relaxed atomic load — branch predictor eliminates the cost when disabled,
// so this is safe on hot paths.
inline bool isEnabled() noexcept {
  return Detail::getEnabledGate().load(std::memory_order_relaxed);
}

namespace Detail {
// The hot gate for range recording: true while a trace of either origin —
// explicit enable or an externally requested on-demand capture — is live.
// Distinct from getEnabledGate(), which tracks only the explicit enable
// intent and stays false during externally driven traces. Same
// process-global storage story as getEnabledGate().
std::atomic<bool> &getRecordingGate() noexcept;
} // namespace Detail

// True while a trace of either origin is live and rangeBegin/rangeEnd record
// spans. Single relaxed atomic load, safe on hot paths.
inline bool isRangeRecordingActive() noexcept {
  return Detail::getRecordingGate().load(std::memory_order_relaxed);
}

namespace Detail {
// Per-thread sticky flag: set once this thread has registered with the
// profiler, so registerCurrentThreadIfEnabled's fast path short-circuits
// forever after on this thread.  Written only by
// registerCurrentThreadSlow(). Per copy (unlike the gates, which are
// process-global): re-running the registration once per copy per thread is
// harmless (it is idempotent on the profiler side).
extern thread_local bool gThreadRegistered;

// Slow path of registerCurrentThreadIfEnabled: registers the calling thread
// with the profiler and sets gThreadRegistered.
void registerCurrentThreadSlow();

// The optional profiler integration, as a function table. All entries have
// the exact semantics of the same-named public functions below; strings
// cross as std::string/StringRef because sink and caller always compile
// into the same link unit with one toolchain.
struct RangeSink {
  void (*enable)();
  void (*disable)();
  void (*activatePendingTrace)();
  void (*waitForTrace)();
  void (*step)();
  // Returns a ProfilerState value.
  int (*state)();
  bool (*canRecord)();
  std::string (*lastTraceError)();
  void (*rangeBegin)(StringRef name, uint32_t color);
  void (*rangeBeginWithId)(uint64_t correlationId, StringRef name,
                           uint32_t color);
  void (*rangeEnd)();
  void (*registerCurrentThread)();
};

// Why a sink is being requested; the integration decides per request kind
// whether to bring a profiler up or only attach to one that already exists
// in the process.
enum class SinkRequest {
  // A passive query/teardown call (disable, state, waitForTrace,
  // lastTraceError): attach only if a profiler is already active.
  Observe,
  // An explicit activation call (enable, haveProfiler, canRecord): the
  // integration may bring the profiler up.
  Attach,
  // A device-initialization notification (activatePendingTrace): the
  // integration applies its own policy (e.g. environment-driven on-demand
  // capture setups attach here without any explicit enable).
  DeviceInit,
};

// Returns the profiler sink for this copy, or nullptr when no profiler is
// available (yet). DEFINED WEAK in Ranges.cpp with a nullptr result; a
// host-side profiler integration overrides it with a strong definition —
// linking that integration into a binary is all the wiring there is. Only
// control-plane paths call this; hot paths read the gates and a cached sink
// pointer. A non-null result must stay valid for the life of the process.
const RangeSink *acquireRangeSink(SinkRequest request);
} // namespace Detail

// Idempotent per-thread registration with the profiler.  When enabled, the
// first call from a given thread labels that thread's track in the trace
// with its OS-level pthread name; subsequent calls return after one
// thread-local read, and calls while disabled return after one relaxed
// atomic load, so it is safe in per-task hot loops.
//
// Registration must run on the thread being named: without it, the profiler
// captions host-side activity tracks from its own processing thread, so the
// threads that actually issue the device calls show up unnamed in the trace.
inline void registerCurrentThreadIfEnabled() noexcept {
  if (Detail::gThreadRegistered)
    return;
  if (!isRangeRecordingActive())
    return;
  Detail::registerCurrentThreadSlow();
}

// Begin / end a CPU semantic range. The dispatcher does NOT need to call
// these per kernel — the profiler captures kernel launches automatically.
// Use these for higher-level spans (e.g. "graph compile", "scheduler
// step"). Each rangeEnd() must run on the same thread as its matching
// rangeBegin(): ranges nest per thread (the profiler's correlation stack is
// thread-local), so a cross-thread pair cannot be expressed. Prefer
// RangeScope, which guarantees this.
//
// While a trace of either origin is live, each pair is recorded as a named
// annotation span correlated to the GPU kernels launched inside it; pairs
// still open when the trace stops are dropped. Outside a live trace both
// calls reduce to one predicted branch.
void rangeBegin(StringRef name, uint32_t color);
void rangeEnd();

// As rangeBegin, but with a CALLER-SUPPLIED user correlation id instead of an
// internally generated one. An external consumer that stamps the same id on
// its own records (e.g. a scheduler batch id also carried on an OTLP span) can
// then join the trace's GPU kernels to that id exactly — the kernel's
// ``External id`` equals ``correlationId`` — rather than only by the span's
// name or a time-window heuristic. The id is truncated to 32 bits by the
// trace format, so keep it under 2^31 to avoid aliasing. The matching
// rangeEnd() is shared with rangeBegin (per-thread LIFO).
void rangeBeginWithId(uint64_t correlationId, StringRef name, uint32_t color);

// Advance the step counter. Drives the warmup/active state machine for
// fixed-window profiling. MAX calls this internally at the end of
// Model::execute(); user code should not call it directly.
void step();

// Enable / disable the profiler. enable() attaches the profiler integration
// on first use (see the file header); when none is available the enable
// intent is still honored — isEnabled() reports true and every recording
// path stays a no-op, matching the behavior of a profiler-present host that
// cannot record.  Both calls are process-local.  In a multi-process
// multi-rank deployment (one OS process per rank) each rank enables itself
// independently.
void enable();
void disable();

// Activate a deferred trace subscription: enable() only records intent,
// because the profiler backend has a vendor-specific readiness ordering that
// only device initialization can satisfy.  MLRT calls this from each
// device-init path at the moment its vendor requires: on CUDA right after it
// retains the device's primary context (CUDADeviceContext), with that
// context current (subscribing CUPTI without one segfaults later); on AMD
// right before its first HIP call (HIPDriver), because rocprofiler-sdk only
// accepts in-process tool registration until the HIP/HSA runtime
// initializes.  Idempotent and cheap; safe on any thread.
// Environment-driven on-demand capture setups also attach the profiler here
// even when profiling was never enabled (SinkRequest::DeviceInit);
// otherwise this is a no-op when profiling is disabled, already active, or
// the backend is not yet ready.
void activatePendingTrace();

// Block until the most recent disable()'s trace has been serialized.
void waitForTrace();

// Returns the message recorded by the most recent trace stop — disable() or
// a daemon-driven stop — or empty when the trace was written successfully
// (or no trace has stopped yet).  A trace that was written but is incomplete
// — the profiler dropped range spans past its recording cap — is reported
// here too; the message says how many spans were dropped and what the cap
// is.  Cleared by enable() (even when no trace actually starts) and by a
// daemon-driven trace start.
std::string lastTraceError();

// Runtime predicate: true iff a profiler integration is attached to this
// copy (i.e. the recording paths in enable()/disable() are real, not
// no-ops). Calling it attempts the attachment, so it can be used as an
// explicit warm-up.
bool haveProfiler() noexcept;

// Runtime predicate: true iff this process can actually record a trace right
// now.  Stricter than ``haveProfiler()``: the profiler backend must also be
// able to attach (for GPU backends, a live device context on the calling
// thread).  Tests that assert on a produced trace must skip when this
// returns false.
bool canRecord();

// Profiler lifecycle state, returned by state() and named by stateName().
enum class ProfilerState { Idle, Warmup, Active, Flushing };

// Returns the current profiler lifecycle state.
ProfilerState state();

// Maps a ProfilerState to a stable lowercase string view:
//   "idle"     — not enabled, no pending trace
//   "warmup"   — enabled, skipping the configured warmup steps
//   "active"   — enabled, recording
//   "flushing" — disabled, trace still being serialized
StringRef stateName(ProfilerState s);

// RAII helper. Disabled scopes pay one predicted branch in the constructor
// (and zero in the destructor when no begin happened).
//
// ``enabled_`` only decides whether the dtor calls ``rangeEnd``; the
// push/pop balance is tracked by a per-thread open-range stack in the
// profiler, NOT by this latch or the live trace flags (which can flip
// between a begin and its end). That is what lets a span close correctly
// after ``disable()`` raced in, and keeps raw FFI callers balanced without
// their own latch.
struct RangeScope {
  explicit RangeScope(StringRef name, uint32_t color = 0);
  ~RangeScope();

  RangeScope(const RangeScope &) = delete;
  RangeScope &operator=(const RangeScope &) = delete;
  RangeScope(RangeScope &&) = delete;
  RangeScope &operator=(RangeScope &&) = delete;

private:
  // True iff the ctor called rangeBegin. The dtor pairs the call.
  bool enabled_;
};

} // namespace M::Profiling

#endif // SUPPORT_PROFILING_RANGES_H
