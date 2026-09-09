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

#ifndef ASYNCRT_COMPILERSUPPORT_RUNTIME_H
#define ASYNCRT_COMPILERSUPPORT_RUNTIME_H

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Support/Chain.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Threading/Shared.h"
#include "llvm/Support/ThreadPool.h"

namespace M::AsyncRT {
class CPUDevice;

/// This is an implement of the LLVM thread pool interface that wraps an AsyncRT
/// cpuDevice. This can be used to inject an AsyncRT cpuDevice as the thread
/// pool implementation inside an MLIR context.
class LLVMThreadPool : public llvm::ThreadPoolInterface {
public:
  LLVMThreadPool(AsyncRT::CPUDevice &cpuDevice)
      : cpuDevice(cpuDevice), poolTurnStile(cpuDevice) {}
  ~LLVMThreadPool();

  void asyncEnqueue(llvm::unique_function<void()> task,
                    llvm::ThreadPoolTaskGroup *Group) override;
  void wait() override;
  void wait(llvm::ThreadPoolTaskGroup &group) override;
  unsigned getMaxConcurrency() const override;

private:
  /// The wrapped AsyncRT cpuDevice.
  AsyncRT::CPUDevice &cpuDevice;

  /// Turnstile for a task group or for the whole queue that can be waited on.
  /// This operates under the assumption that tasks cannot be added to a queue
  /// or group while it is being waited on.
  struct TurnStile {
    TurnStile(AsyncRT::CPUDevice &cpuDevice)
        : counter(1),
          chain(AsyncRT::AsyncValueRef<Chain>::allocate(cpuDevice)) {}

    bool taskComplete();
    void waitAndReset(AsyncRT::CPUDevice &cpuDevice);

    std::atomic<unsigned> counter;
    AsyncRT::AsyncValueRef<Chain> chain;
  };

  /// Shared table of turnstiles for all active task groups. The elements are
  /// allocated in unique pointers so that tasks functions can carry the
  /// reference safely.
  Shared<DenseMap<llvm::ThreadPoolTaskGroup *, std::unique_ptr<TurnStile>>>
      groupTurnStiles;
  /// Turnstile for the whole task group.
  TurnStile poolTurnStile;
};
} // namespace M::AsyncRT

#endif // ASYNCRT_COMPILERSUPPORT_RUNTIME_H
