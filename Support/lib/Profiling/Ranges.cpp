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
// Implementation of the minimal range-annotation API
// (Support/Profiling/Ranges.h): the hot-path gates and thin forwarders
// into the optional RangeSink. The gates and the cached sink pointer live in
// one process-global block (Globals::getProfilingRangeGlobals, in
// libMSupportGlobals.so), so every statically linked copy of this file
// observes the same state — see the gate comments in Ranges.h. The sink is
// obtained through Detail::acquireRangeSink, defined WEAK here with a
// nullptr result — the host-side profiler integration overrides it with a
// strong definition, so linking that integration is the only wiring. This
// file deliberately knows nothing about how a profiler is found, loaded, or
// implemented.
//
// Sink-absent semantics: enable()/disable() honor the intent locally (so
// isEnabled()/state() behave exactly like a profiler-attached host that
// cannot record), and every other call is a no-op. Hot paths never request
// a sink: they read the gates — which only an attached profiler can raise —
// plus a lock-free cached sink pointer.
//
//===----------------------------------------------------------------------===//

#include "Support/Profiling/Ranges.h"

#include "Support/Globals/Globals.h"

#include <atomic>
#include <cstdint>
#include <string>

#include "llvm/ADT/StringRef.h"

namespace M::Profiling {

namespace Detail {

namespace {

// One process-global state block (gates + cached sink), shared by every
// statically linked copy of this file through libMSupportGlobals.so. The
// function-local reference caches the cross-library lookup per copy.
Globals::ProfilingRangeGlobals &globals() noexcept {
  static Globals::ProfilingRangeGlobals &g =
      M::Globals::getProfilingRangeGlobals();
  return g;
}

} // namespace

std::atomic<bool> &getEnabledGate() noexcept { return globals().enabled; }

std::atomic<bool> &getRecordingGate() noexcept {
  return globals().recordingActive;
}

thread_local bool gThreadRegistered = false;

// Weak default: no profiler integration linked. The strong override lives
// in the host glue; `weak` (not a config define) so a single build of this
// library serves both kinds of binaries and the wiring is purely link-time.
__attribute__((weak)) const RangeSink *acquireRangeSink(SinkRequest request) {
  (void)request;
  return nullptr;
}

namespace {

// Lock-free mirror of the last successful acquisition, for the recording
// paths (step, rangeBegin/End, registerCurrentThreadSlow). Racing first
// attachments may each store their own copy's sink; every stored value
// forwards to the same plugin and stays valid for the process lifetime, so
// last-write-wins is benign. Living in the process-global block, one copy's
// attachment serves every copy — including copies (and their control-plane
// calls) that never attach themselves. Control-plane calls go through
// acquire() below instead.
std::atomic<const RangeSink *> &getCachedSink() { return globals().cachedSink; }

const RangeSink *acquire(SinkRequest request) {
  if (const RangeSink *sink = getCachedSink().load(std::memory_order_acquire))
    return sink;
  const RangeSink *sink = acquireRangeSink(request);
  if (sink != nullptr)
    getCachedSink().store(sink, std::memory_order_release);
  return sink;
}

} // namespace

void registerCurrentThreadSlow() {
  // Only reachable while the recording gate is up, which implies some copy
  // attached a sink — read the lock-free cache, not the acquire path.
  if (const RangeSink *sink = getCachedSink().load(std::memory_order_acquire))
    sink->registerCurrentThread();
  gThreadRegistered = true;
}

} // namespace Detail

void rangeBegin(StringRef name, uint32_t color) {
  // Gate first (contract in Ranges.h): outside a live trace this is one
  // predicted branch. enable()-driven traces raise the gate only after
  // acquire() stored the process-global sink, so the cache read below
  // observes it; an externally driven raise can race the first attachment's
  // store, in which case the null cache read drops the span (never a torn
  // pair).
  if (!isRangeRecordingActive())
    return;
  if (const Detail::RangeSink *sink =
          Detail::getCachedSink().load(std::memory_order_acquire))
    sink->rangeBegin(name, color);
}

void rangeBeginWithId(uint64_t correlationId, StringRef name, uint32_t color) {
  if (!isRangeRecordingActive())
    return;
  if (const Detail::RangeSink *sink =
          Detail::getCachedSink().load(std::memory_order_acquire))
    sink->rangeBeginWithId(correlationId, name, color);
}

void rangeEnd() {
  // Deliberately NOT gated on the recording flag: the profiler tracks
  // begin/end balance on a per-thread stack that must see every end, even
  // one whose trace stopped in between. With no sink attached no begin was
  // ever forwarded, so forwarding nothing is balanced.
  if (const Detail::RangeSink *sink =
          Detail::getCachedSink().load(std::memory_order_acquire))
    sink->rangeEnd();
}

void step() {
  // Disabled fast path: one predicted branch (the idle-path contract the
  // disabled-hot-path perf gate enforces).
  if (!isEnabled())
    return;
  if (const Detail::RangeSink *sink =
          Detail::getCachedSink().load(std::memory_order_acquire))
    sink->step();
}

void enable() {
  if (const Detail::RangeSink *sink =
          Detail::acquire(Detail::SinkRequest::Attach)) {
    sink->enable();
    return;
  }
  // No profiler available: honor the enable intent so the observable state
  // machine (isEnabled/state) behaves exactly like a profiler-attached host
  // that cannot record — callers and tests need not care which they are on.
  Detail::getEnabledGate().store(true, std::memory_order_release);
}

void disable() {
  if (const Detail::RangeSink *sink =
          Detail::acquire(Detail::SinkRequest::Observe)) {
    sink->disable();
    return;
  }
  Detail::getEnabledGate().store(false, std::memory_order_release);
}

void activatePendingTrace() {
  if (const Detail::RangeSink *sink =
          Detail::acquire(Detail::SinkRequest::DeviceInit))
    sink->activatePendingTrace();
}

void waitForTrace() {
  if (const Detail::RangeSink *sink =
          Detail::acquire(Detail::SinkRequest::Observe))
    sink->waitForTrace();
}

std::string lastTraceError() {
  if (const Detail::RangeSink *sink =
          Detail::acquire(Detail::SinkRequest::Observe))
    return sink->lastTraceError();
  return {};
}

bool haveProfiler() noexcept {
  return Detail::acquire(Detail::SinkRequest::Attach) != nullptr;
}

bool canRecord() {
  if (const Detail::RangeSink *sink =
          Detail::acquire(Detail::SinkRequest::Attach))
    return sink->canRecord();
  return false;
}

ProfilerState state() {
  if (const Detail::RangeSink *sink =
          Detail::acquire(Detail::SinkRequest::Observe))
    return static_cast<ProfilerState>(sink->state());
  // No profiler: mirror the attached-but-cannot-record state machine —
  // enable() jumps straight to Active, disable() back to Idle (Flushing is
  // only ever observable mid-disable()).
  return isEnabled() ? ProfilerState::Active : ProfilerState::Idle;
}

StringRef stateName(ProfilerState s) {
  switch (s) {
  case ProfilerState::Idle:
    return "idle";
  case ProfilerState::Warmup:
    return "warmup";
  case ProfilerState::Active:
    return "active";
  case ProfilerState::Flushing:
    return "flushing";
  }
  return "idle";
}

RangeScope::RangeScope(StringRef name, uint32_t color)
    : enabled_(isRangeRecordingActive()) {
  if (enabled_)
    rangeBegin(name, color);
}

RangeScope::~RangeScope() {
  if (enabled_)
    rangeEnd();
}

} // namespace M::Profiling
