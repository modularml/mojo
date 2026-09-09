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

#include "AsyncRT/CompilerSupport/LLVMThreadPool.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"

using namespace M;
using namespace AsyncRT;
using llvm::ThreadPoolTaskGroup;
using AsyncChain = AsyncValueRef<Chain>;

// The `llvm::ThreadPoolInterface` admittedly asks for quite an awkward API.
// Looking at the `StdThreadPool` implementation (the default LLVM threadpool),
// it doesn't lend itself to being very efficient. Among other things, a hash
// map is required, because the `ThreadPoolTaskGroup` provided by the API only
// serves as a pointer ID for the task group.

bool LLVMThreadPool::TurnStile::taskComplete() {
  if (--counter == 0) {
    chain.copy().emplace();
    return true;
  }
  return false;
}

void LLVMThreadPool::TurnStile::waitAndReset(AsyncRT::CPUDevice &cpuDevice) {
  // Decrement the counter, and if there are still tasks running, wait.
  if (!taskComplete())
    await(chain);
  counter = 1;
  chain = AsyncRT::AsyncValueRef<Chain>::allocate(cpuDevice);
}

LLVMThreadPool::~LLVMThreadPool() {
  poolTurnStile.waitAndReset(cpuDevice);
  std::move(poolTurnStile.chain).emplace();
}

void LLVMThreadPool::asyncEnqueue(llvm::unique_function<void()> task,
                                  ThreadPoolTaskGroup *group) {
  // Grab the turnstile for this task group, or create one.
  TurnStile *turnstile = groupTurnStiles.modify([this, group](auto &map) {
    std::unique_ptr<TurnStile> &turnstile = map[group];
    if (!turnstile)
      turnstile = std::make_unique<TurnStile>(cpuDevice);
    return turnstile.get();
  });

  // Increment the number of active tasks.
  ++turnstile->counter;

  cpuDevice.getWorkQueue()->addTask(
      [this, turnstile, func = std::move(task)]() mutable {
        // Run the task.
        func();
        // If this is the last task in the taskgroup to complete, erase it from
        // the map to keep the map size bounded.
        turnstile->taskComplete();
        // Indicate that a threadpool task has completed.
        poolTurnStile.taskComplete();
      });
}

void LLVMThreadPool::wait() { poolTurnStile.waitAndReset(cpuDevice); }

void LLVMThreadPool::wait(ThreadPoolTaskGroup &group) {
  TurnStile *turnstile =
      groupTurnStiles.read([group = &group](auto &map) -> TurnStile * {
        if (auto it = map.find(group); it != map.end())
          return it->second.get();
        return nullptr;
      });
  // No tasks scheduled for this group.
  if (!turnstile)
    return;
  if (!turnstile->taskComplete())
    await(turnstile->chain);
  groupTurnStiles.modify([group = &group](auto &map) { map.erase(group); });
}

unsigned LLVMThreadPool::getMaxConcurrency() const {
  return cpuDevice.getWorkQueue()->getParallelismLevel();
}
