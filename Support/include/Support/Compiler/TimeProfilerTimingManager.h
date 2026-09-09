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

#ifndef SUPPORT_COMPILER_TIMEPROFILERTIMINGMANAGER_H
#define SUPPORT_COMPILER_TIMEPROFILERTIMINGMANAGER_H

#include "mlir/Support/Timing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <memory>
#include <optional>
#include <string>

namespace M {

//===----------------------------------------------------------------------===//
// TimeProfilerTimingManager
//===----------------------------------------------------------------------===//

/// This class represents an MLIR timing manager that hooks into the
/// TimeTraceProfiler functionality.
///
/// CAUTION: These profiling entries are always enabled at compile time.
class TimeProfilerTimingManager : public mlir::TimingManager {
public:
  TimeProfilerTimingManager();
  TimeProfilerTimingManager(TimeProfilerTimingManager &&rhs);
  ~TimeProfilerTimingManager() override;

  // Disable copying of the `TimeProfilerTimingManager`.
  TimeProfilerTimingManager(const TimeProfilerTimingManager &rhs) = delete;
  TimeProfilerTimingManager &
  operator=(const TimeProfilerTimingManager &rhs) = delete;

protected:
  // `TimingManager` callbacks
  std::optional<void *> rootTimer() override;
  void startTimer(void *handle) override;
  void stopTimer(void *handle) override;
  void *nestTimer(void *handle, const void *id,
                  llvm::function_ref<std::string()> nameBuilder) override;
  void hideTimer(void *handle) override;

  /// The internal implementation of the `TimeProfilerTimingManager`.
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M

#endif // SUPPORT_COMPILER_TIMEPROFILERTIMINGMANAGER_H
