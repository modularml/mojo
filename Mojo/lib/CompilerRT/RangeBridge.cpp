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
// FFI bridge between Mojo and the M::Profiling Range API. The Mojo callers
// are `max.runtime.tracing.Trace` (op-level spans, routed here from its
// __enter__/__exit__ on every production trace site) and the
// `profiling_range.Range` struct (profiler-only spans; no in-tree
// production sites yet), both via external_call. The enable/disable
// control surface is driven by
// InferenceSession construction auto-start (max-debug.profiling-enabled).
//
//===----------------------------------------------------------------------===//

#include "Support/SymbolExport.h"

#if MODULAR_KGEN_PROFILING_ENABLED
#include "Support/Profiling/Ranges.h"
#include "llvm/ADT/StringRef.h"
#endif

#include <cstddef>
#include <cstdint>

extern "C" {

// Mojo-callable range begin / end. Both are cheap when no trace is live —
// RangeBegin branches on M::Profiling::isRangeRecordingActive() and RangeEnd
// on this thread's pairing state — so they are safe to call from hot
// kernel-launch paths.
//
// Precondition: `namePtr` must be non-null even when `nameLen == 0`.
// Constructing an `llvm::StringRef` from a null pointer asserts in debug
// builds and is UB for non-zero lengths. The Mojo callers pass
// String/StaticString buffer pointers, which are non-null even for empty
// strings and so satisfy this contract; any new C-ABI caller must uphold
// it itself.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeBegin(const char *namePtr, size_t nameLen,
                           uint32_t color) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::rangeBegin(llvm::StringRef(namePtr, nameLen), color);
#endif
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeEnd(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::rangeEnd();
#endif
}

// Step counter advance, driving the warmup/active step-window state
// machine. The production step boundary is C++-side — MAX's
// ModelHandle::executeDeviceTensors() calls M::Profiling::step() directly
// once per model execute (InferenceSession.cpp), not through this export.
// This FFI entry exists so a Mojo-side runtime entry point that constitutes
// its own step boundary can drive the machine too; it has no in-tree caller.
// Same disabled fast path as step() itself: one predicted branch.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeStep(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::step();
#endif
}

// Enable / disable control surface. Driven by InferenceSession's
// construction-time auto-start (max-debug.profiling-enabled).
// TODO(MXTOOLS-190): Dynolog's IPC listener will drive enable/disable
// through these same entry points.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeEnable(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::enable();
#endif
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RangeDisable(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::Profiling::disable();
#endif
}

// Returns 1 if the profiler is currently enabled, 0 otherwise. Useful from
// Mojo to elide expensive name materialization on the disabled fast path —
// but note this reflects only the session API's enable intent: it stays 0
// during Dynolog daemon-driven on-demand traces, when ranges DO record, so
// eliding on it opts the caller out of daemon-trace annotation (RangeBegin
// itself is one predicted branch when idle, so unconditional calls are fine).
// Returns `size_t` to match the existing `KGEN_CompilerRT_TracyIsEnabled`
// shape so Mojo callers can treat both predicates uniformly.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_RangeIsEnabled(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  return M::Profiling::isEnabled() ? 1 : 0;
#else
  return 0;
#endif
}

// Returns 1 while a trace of either origin — an explicit session enable or an
// externally requested on-demand capture — is live and RangeBegin/RangeEnd
// record. Unlike RangeIsEnabled this is the gate hot-path callers should use
// to elide per-span argument setup: it covers on-demand traces too, and it is
// one relaxed atomic load behind the FFI call. The gate is process-global
// (libMSupportGlobals.so), so a trace started through any other statically
// linked copy of the range shim is visible here immediately.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_RangeIsRecording(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  return M::Profiling::isRangeRecordingActive() ? 1 : 0;
#else
  return 0;
#endif
}

} // extern "C"
