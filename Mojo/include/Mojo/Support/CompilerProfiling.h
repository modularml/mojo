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

#ifndef KGEN_SUPPORT_COMPILERPROFILING_H
#define KGEN_SUPPORT_COMPILERPROFILING_H

#include "Support/Profiling/TimeProfiler.h"
#include <filesystem>

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// TimeTraceScope
//===----------------------------------------------------------------------===//

constexpr bool kIsTracingEnabled = Trace::EnableTrace(Trace::kCompiler, 1);

using InterpreterProfilerEntry = ProfilerEntry<true, Trace::kCompiler>;

/// Profiler entry for Mojo compilation passes.
using CompilerProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 2), Trace::kCompiler>;

/// Verbose profiler entry for Mojo compilation passes.
using VerboseCompilerProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 3), Trace::kCompiler>;

struct InterpreterTimeTraceScope
    : public TimeTraceScope<Trace::kCompiler,
                            InterpreterProfilerEntry::isEnabled()> {
  using TimeTraceScope::TimeTraceScope;

  InterpreterTimeTraceScope(StringRef name, StringRef detail = {})
      : TimeTraceScope(InterpreterProfilerEntry::create(name, detail)) {}
  InterpreterTimeTraceScope(StringRef name, ProfilerPrintFn detailFn)
      : TimeTraceScope(InterpreterProfilerEntry::create(name, detailFn)) {}
};

struct CompilerTimeTraceScope
    : public TimeTraceScope<Trace::kCompiler,
                            CompilerProfilerEntry::isEnabled()> {
  using TimeTraceScope::TimeTraceScope;

  CompilerTimeTraceScope(StringRef name, StringRef detail = {})
      : TimeTraceScope(CompilerProfilerEntry::create(name, detail)) {}
  CompilerTimeTraceScope(StringRef name, ProfilerPrintFn detailFn)
      : TimeTraceScope(CompilerProfilerEntry::create(name, detailFn)) {}
};

struct VerboseCompilerTimeTraceScope
    : public TimeTraceScope<Trace::kCompiler,
                            VerboseCompilerProfilerEntry::isEnabled()> {
  using TimeTraceScope::TimeTraceScope;

  VerboseCompilerTimeTraceScope(StringRef name, StringRef detail = {})
      : TimeTraceScope(VerboseCompilerProfilerEntry::create(name, detail)) {}
  VerboseCompilerTimeTraceScope(StringRef name, ProfilerPrintFn detailFn)
      : TimeTraceScope(VerboseCompilerProfilerEntry::create(name, detailFn)) {}
};

//===----------------------------------------------------------------------===//
// TraceProfiler
//===----------------------------------------------------------------------===//

/// Common trace profiler setup.
struct TraceProfiler {
  TraceProfiler(bool enabled, int timeTraceGranularity) {
    if (enabled)
      initialize(timeTraceGranularity);
  }
  ~TraceProfiler();

private:
  void initialize(int timeTraceGranularity);

  std::optional<TimeTraceProfiler> profiler;
  std::filesystem::path outputFilePath;
};

} // namespace M::KGEN

#endif // KGEN_SUPPORT_COMPILERPROFILING_H
