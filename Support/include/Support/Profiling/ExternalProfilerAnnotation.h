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
// External profiler annotation sink: the seam through which runtime spans
// annotate an EXTERNAL profiler's trace (e.g. NVTX/roctx/Tracy), as opposed
// to the RangeSink in Ranges.h, which records into MAX's own trace.
//
// Same wiring model as RangeSink: this header carries no implementation,
// Detail::acquireExternalProfilerAnnotationSink() is DEFINED WEAK (in
// ExternalProfilerAnnotation.cpp) with a nullptr result, and an optional
// host-side integration overrides it with a strong definition — linking
// that integration is all the wiring there is. When no sink is linked
// (e.g. the open-source compiler stack) every caller degrades to a safe
// no-op.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_EXTERNALPROFILERANNOTATION_H
#define SUPPORT_PROFILING_EXTERNALPROFILERANNOTATION_H

#include <cstdint>

#include "Support/LLVMForwardDecls.h"

namespace M::Profiling::Detail {

/// The optional external profiler annotation integration, as a function
/// table. Strings cross as StringRef because sink and caller always compile
/// into the same link unit with one toolchain.
struct ExternalProfilerAnnotationSink {
  /// Pushes a nested, thread-local annotation range. Returns true iff an
  /// annotation was actually emitted; the caller must call rangePop exactly
  /// once, on the same thread, for each push that returned true (popping
  /// for a suppressed push would close an unrelated outer range).
  bool (*rangePush)(StringRef name, uint32_t colorARGB);
  /// Pops the innermost range pushed on this thread. Runs even after
  /// setEnabled(false), so a range opened while enabled still closes.
  void (*rangePop)();
  /// Emits an instantaneous marker.
  void (*mark)(StringRef name, uint32_t colorARGB);
  /// Programmatically requests or suppresses annotation, overriding the
  /// integration's environment default.
  void (*setEnabled)(bool enabled);
  /// Returns true when pushes will actually emit. May perform the
  /// integration's lazy setup, so callers can use it to elide argument
  /// materialization before an emission attempt.
  bool (*isEnabled)();
};

/// Returns the external profiler annotation sink for this copy, or nullptr
/// when no integration is linked. DEFINED WEAK in
/// ExternalProfilerAnnotation.cpp with a nullptr result; a host-side
/// integration overrides it with a strong definition. Only called once per
/// copy (callers cache the result); a non-null result must stay valid for
/// the life of the process.
const ExternalProfilerAnnotationSink *acquireExternalProfilerAnnotationSink();

} // namespace M::Profiling::Detail

#endif // SUPPORT_PROFILING_EXTERNALPROFILERANNOTATION_H
