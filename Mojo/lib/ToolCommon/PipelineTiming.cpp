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

#include "Mojo/ToolCommon/PipelineTiming.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace KGEN;

namespace {
/// The state of the collection. The state is global because the timers of LLVM
/// that it reads are global, and one thread owns both.
struct RegionState {
  bool enabled = false;
  /// The label of the pipeline whose times the timers hold now. LLVM work
  /// outside every region, which no path does today, keeps the first value.
  std::string currentLabel = "other";
  /// How many regions are open. Only the outer region names a part.
  unsigned depth = 0;
  llvm::SmallVector<LLVMTimingReportPart> parts;
#ifndef NDEBUG
  /// The thread that opened the first region. The collection reads global
  /// timers, so a second thread would mix two pipelines into one part.
  uint64_t threadId = 0;
#endif
};

RegionState &getRegionState() {
  static RegionState state;
  return state;
}

/// Reads the timers into a part under the current label, and clears them.
void closeCurrentPart() {
  RegionState &state = getRegionState();
  std::string report;
  llvm::raw_string_ostream os(report);
  // `printAll` drains the queued records of the timers that already ended, and
  // it reads, without clearing, the totals of the timers that are still alive.
  llvm::TimerGroup::printAll(os);
  // `clearAll` resets the live totals, so that a group does not write them
  // again when the process deletes it, and so that the next part holds only
  // the times of the next pipeline.
  llvm::TimerGroup::clearAll();

  if (!report.empty())
    state.parts.push_back({state.currentLabel, std::move(report)});
}

} // namespace

std::string M::KGEN::pipelineTimingLabel(const CompilationOptions &options) {
  llvm::Triple triple(options.targetTriple);
  std::string label = isGPUTriple(triple) ? "offload " : "host ";
  label += options.targetTriple;
  if (!options.targetCpu.empty()) {
    label += " ";
    label += options.targetCpu;
  }
  // The `emission_option` and `emission_link_option` of
  // `kgen.compile_offload` reach the pipeline as these two fields, and
  // `OffloadInfo` keys its groups by the pair. One target thus holds one group
  // for each pair, and each group runs its own pipeline.
  if (!options.emissionOptions.empty()) {
    label += " [";
    label += options.emissionOptions;
    label += "]";
  }
  if (!options.emissionLinkOptions.empty()) {
    label += " [link ";
    label += options.emissionLinkOptions;
    label += "]";
  }
  return label;
}

void M::KGEN::enableLLVMTimingRegions() { getRegionState().enabled = true; }

llvm::SmallVector<LLVMTimingReportPart> M::KGEN::takeLLVMTimingReport() {
  RegionState &state = getRegionState();
  if (!state.enabled)
    return {};
  // The last pipeline has no region after it to close its part.
  closeCurrentPart();
  state.enabled = false;
  // Reset the label and the thread pin, so that a later collection in the
  // same process starts fresh.
  state.currentLabel = "other";
#ifndef NDEBUG
  state.threadId = 0;
#endif
  return std::move(state.parts);
}

LLVMTimingRegion::LLVMTimingRegion(const CompilationOptions &options)
    : active(getRegionState().enabled) {
  if (!active)
    return;

  RegionState &state = getRegionState();
#ifndef NDEBUG
  uint64_t thisThread = llvm::get_threadid();
  if (state.threadId == 0)
    state.threadId = thisThread;
  assert(state.threadId == thisThread &&
         "LLVM timing regions read global timers, so the compilation must run "
         "on one thread; see LLVMPassTiming::configure");
#endif

  // An inner region belongs to the region that holds it.
  if (state.depth++ > 0)
    return;

  std::string label = pipelineTimingLabel(options);
  if (label != state.currentLabel) {
    closeCurrentPart();
    state.currentLabel = std::move(label);
  }
}

LLVMTimingRegion::~LLVMTimingRegion() {
  if (active)
    --getRegionState().depth;
}

//===----------------------------------------------------------------------===//
// MLIR
//===----------------------------------------------------------------------===//

namespace {
/// The width of the rule around the name of an offload scope.
constexpr size_t kOffloadScopeRuleWidth = 72;

/// The root scope of the MLIR timing. The value is null if the timing is off.
/// The value is global because the function that makes the pass manager of an
/// offload target gets no scope. That function is also deep inside the
/// elaboration of the host.
mlir::TimingScope *&getMLIRTimingRoot() {
  static mlir::TimingScope *root = nullptr;
  return root;
}
} // namespace

void M::KGEN::setMLIRTimingRoot(mlir::TimingScope *root) {
  getMLIRTimingRoot() = root;
}

mlir::TimingScope
M::KGEN::nestMLIROffloadScope(const CompilationOptions &options) {
  mlir::TimingScope *root = getMLIRTimingRoot();
  if (!root)
    return mlir::TimingScope();
  // The scope is at the same depth as a host pass, and the scope holds many
  // rows. The marks around the name show that the scope is not a host pass.
  // The marks also show where one offload pipeline stops and the next offload
  // pipeline starts.
  //
  // The host pass that runs the offload contains this scope. Thus the time of
  // this scope is also part of that host pass. The two rows are siblings in
  // the report, but the times are not separate. The words in the label tell
  // the reader not to add the two rows together.
  std::string name =
      "===--- " + pipelineTimingLabel(options) + " (also in host) ";
  // Each scope gets a rule of the same width. Thus the rules look like
  // dividers between the parts of the report. A label that is longer than the
  // rule keeps its own length.
  if (name.size() < kOffloadScopeRuleWidth - 3)
    name.append(kOffloadScopeRuleWidth - 3 - name.size(), '-');
  name += "===";
  // `nest` takes a lock if the thread is not the thread of the root.
  // Thus elaboration can run this function on a worker thread.
  //
  // The compiler compiles one offload group at a time today. Take care if the
  // compiler compiles the groups in parallel later. MLIR joins two scopes that
  // have the same name into one row. Thus the label must give a different name
  // to each pipeline that needs a row of its own. The LLVM part of this file
  // has a stronger rule, because it reads global timers. See `RegionState`.
  return root->nest(name);
}
