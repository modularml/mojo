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

#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Profiling/TimeProfiler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

using namespace M;

//===----------------------------------------------------------------------===//
// TimeProfilerTimingManager
//===----------------------------------------------------------------------===//

namespace {
struct Timer {
  /// Nest a timer with the given ID and name.
  Timer *nestTimer(const void *id,
                   llvm::function_ref<std::string()> nameBuilder) {
    std::lock_guard<std::mutex> lock(mutex);
    auto &child = children[id];
    if (!child) {
      child = std::make_unique<Timer>();
      child->name = nameBuilder();
    }
    return child.get();
  }

  /// Returns true if this timer is hidden.
  bool isHidden() const { return name.empty(); }

  /// The name of the timer.
  std::string name;

  /// Mutex for the async access.
  std::mutex mutex;

  /// The children of this timer.
  llvm::DenseMap<const void *, std::unique_ptr<Timer>> children;
};
} // namespace

struct TimeProfilerTimingManager::Impl {
  /// The fake root timer.
  Timer rootTimer;
};

TimeProfilerTimingManager::TimeProfilerTimingManager() : impl(new Impl()) {}
TimeProfilerTimingManager::TimeProfilerTimingManager(
    TimeProfilerTimingManager &&rhs)
    : impl(std::move(rhs.impl)) {}
TimeProfilerTimingManager::~TimeProfilerTimingManager() = default;

std::optional<void *> TimeProfilerTimingManager::rootTimer() {
  if (!impl)
    return std::nullopt;
  return &impl->rootTimer;
}

void TimeProfilerTimingManager::startTimer(void *handle) {
  auto *timer = reinterpret_cast<Timer *>(handle);
  if (!timer->isHidden())
    ProfilerEntry<true, Trace::kOther>::createAndPush(timer->name);
}

void TimeProfilerTimingManager::stopTimer(void *handle) {
  // Time trace profilers are timeline based, so we just stop the last timer.
  auto *timer = reinterpret_cast<Timer *>(handle);
  if (!timer->isHidden())
    ProfilerEntry<true, Trace::kOther>::endAndPop();
}

void *
TimeProfilerTimingManager::nestTimer(void *handle, const void *id,
                                     function_ref<std::string()> nameBuilder) {
  auto *timer = reinterpret_cast<Timer *>(handle);
  return timer->nestTimer(id, nameBuilder);
}

void TimeProfilerTimingManager::hideTimer(void *handle) {
  // No-op hiding timers for the trace view. The hide functionality is for
  // certain types of scopes, but showing those isn't as noisy as other views
  // and preserves information.
}
