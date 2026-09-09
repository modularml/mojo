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
// FFI bridge between Mojo and the M::Profiling
// external profiler annotation sink (Support/Profiling/
// ExternalProfilerAnnotation.h). Where RangeBridge.cpp records spans into
// MAX's own trace, these entry points annotate an EXTERNAL profiler's
// trace: the intended Mojo caller is `max.runtime.tracing.Trace`, routing
// its OP-level spans here from __enter__/__exit__. When no integration is
// linked, every entry point is a no-op and Push reports that nothing was
// emitted.
//
//===----------------------------------------------------------------------===//

#include "Support/SymbolExport.h"

#if MODULAR_KGEN_PROFILING_ENABLED
#include "Support/Profiling/ExternalProfilerAnnotation.h"
#include "llvm/ADT/StringRef.h"
#endif

#include <cstddef>
#include <cstdint>

#if MODULAR_KGEN_PROFILING_ENABLED
using M::Profiling::Detail::ExternalProfilerAnnotationSink;

/// One acquisition per process copy; the sink (or its absence) is immutable
/// for the life of the process, so a function-local static is the whole
/// cache.
static const ExternalProfilerAnnotationSink *externalSink() {
  static const ExternalProfilerAnnotationSink *sink =
      M::Profiling::Detail::acquireExternalProfilerAnnotationSink();
  return sink;
}
#endif

extern "C" {

/// Mojo-callable annotation push. Returns 1 iff an annotation was actually
/// emitted; the caller must pair each such push with exactly one
/// KGEN_CompilerRT_ExternalProfilerAnnotationPop on the same thread
/// (pop-iff-pushed). Cheap when nothing will be emitted — a cached pointer
/// check when no integration is linked, plus one relaxed atomic load inside
/// the integration while annotation is disabled — so it is safe to call
/// from hot paths.
///
/// Precondition: `namePtr` must be non-null even when `nameLen == 0` (same
/// contract, for the same llvm::StringRef reason, as
/// KGEN_CompilerRT_RangeBegin).
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_ExternalProfilerAnnotationPush(const char *namePtr,
                                               size_t nameLen, uint32_t color) {
#if MODULAR_KGEN_PROFILING_ENABLED
  if (const ExternalProfilerAnnotationSink *sink = externalSink())
    return sink->rangePush(llvm::StringRef(namePtr, nameLen), color) ? 1 : 0;
#endif
  return 0;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_ExternalProfilerAnnotationPop(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  if (const ExternalProfilerAnnotationSink *sink = externalSink())
    sink->rangePop();
#endif
}

/// Instantaneous marker; no pairing.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_ExternalProfilerAnnotationMark(const char *namePtr,
                                               size_t nameLen, uint32_t color) {
#if MODULAR_KGEN_PROFILING_ENABLED
  if (const ExternalProfilerAnnotationSink *sink = externalSink())
    sink->mark(llvm::StringRef(namePtr, nameLen), color);
#endif
}

/// Programmatic override of the integration's environment default, so a
/// host-language control surface (e.g. max._core's set_gpu_profiling_state)
/// and the integration's own gate cannot disagree.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_ExternalProfilerAnnotationSetEnabled(size_t enabled) {
#if MODULAR_KGEN_PROFILING_ENABLED
  if (const ExternalProfilerAnnotationSink *sink = externalSink())
    sink->setEnabled(enabled != 0);
#endif
}

/// Returns 1 when pushes will actually emit. May perform the integration's
/// lazy setup, so Mojo callers can use it to elide expensive message
/// materialization (e.g. for markers) before an emission attempt. Returns
/// `size_t` to match the existing KGEN_CompilerRT_*IsEnabled predicates.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_ExternalProfilerAnnotationIsEnabled(void) {
#if MODULAR_KGEN_PROFILING_ENABLED
  if (const ExternalProfilerAnnotationSink *sink = externalSink())
    return sink->isEnabled() ? 1 : 0;
#endif
  return 0;
}

} // extern "C"
