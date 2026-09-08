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
// This file splits the timing reports. Each pipeline that a compiler command
// runs gets one part. A command runs the host pipeline. A command also runs
// one more pipeline for each `kgen.compile_offload` target. The LLVM report
// and the MLIR report keep the parts apart, but each report uses a different
// method. The two timing libraries hold their times in a different
// structure.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_PIPELINETIMING_H
#define KGEN_TOOLCOMMON_PIPELINETIMING_H

#include "mlir/Support/Timing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M::KGEN {
class CompilationOptions;

/// Gives the name of the pipeline that `options` describes. Both reports use
/// this name. Then a reader can compare a part of one report with a part of
/// the other report.
std::string pipelineTimingLabel(const CompilationOptions &options);

/// One part of the LLVM pass timing report.
struct LLVMTimingReportPart {
  /// Names the pipeline that made this part, such as `host x86_64-linux-gnu`.
  std::string label;
  /// The report of LLVM for this pipeline, in the format that LLVM writes.
  std::string report;
};

/// Starts to collect the report in parts. The driver calls this for the
/// `--llvm-timing` option, before the compilation starts.
///
/// The timers of LLVM are global to the process, and LLVM gives no way to hold
/// a separate set for each pipeline. The collection instead reads the timers
/// and clears them each time the pipeline changes, which gives the times of
/// only the pipeline that ran since the previous read. This is correct because
/// the pipelines do not overlap: the option also sets the compilation to one
/// thread, and the compiler runs each offload pipeline inside elaboration,
/// which is before the host pipeline reaches LLVM.
void enableLLVMTimingRegions();

/// Reads the parts and forgets them. The parts come in the order in which the
/// pipelines ran. A pipeline whose timers hold nothing, such as one that the
/// compilation cache answered, gives no part.
llvm::SmallVector<LLVMTimingReportPart> takeLLVMTimingReport();

/// Marks the LLVM work of one pipeline. The object compiler puts one of these
/// around each piece of work that reaches LLVM. Two regions that follow each
/// other and have the same label give one part, so that a target with many
/// kernel groups does not give one part for each group. A region inside
/// another region belongs to the outer region.
///
/// The object does nothing unless `enableLLVMTimingRegions` ran first.
class LLVMTimingRegion {
public:
  explicit LLVMTimingRegion(const CompilationOptions &options);
  ~LLVMTimingRegion();

  LLVMTimingRegion(const LLVMTimingRegion &) = delete;
  LLVMTimingRegion &operator=(const LLVMTimingRegion &) = delete;

private:
  bool active;
};

//===----------------------------------------------------------------------===//
// MLIR
//===----------------------------------------------------------------------===//

/// Holds the root scope of the MLIR timing for the command. The driver sets
/// the root for the `--mlir-timing` option. The driver clears the root after
/// it prints the report.
///
/// MLIR holds its times in a tree. Thus a pipeline does not need a report of
/// its own. A pipeline needs only a scope of its own under the root. The pass
/// manager of an offload target is deep inside the elaboration of the host.
/// The function that makes this pass manager gets no scope. Thus the root
/// moves through this function.
void setMLIRTimingRoot(mlir::TimingScope *root);

/// Gives a scope for the offload pipeline that `options` describes. The scope
/// is under the root that `setMLIRTimingRoot` holds. The scope is empty if the
/// timing is off. A pass manager that gets an empty scope records no times.
///
/// The time of the scope is also part of the host pass that runs the offload,
/// because the host pass contains the offload pipeline.
mlir::TimingScope nestMLIROffloadScope(const CompilationOptions &options);

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_PIPELINETIMING_H
