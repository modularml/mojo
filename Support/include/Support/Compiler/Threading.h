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

#ifndef SUPPORT_COMPILER_THREADING_H
#define SUPPORT_COMPILER_THREADING_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ThreadPool.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <vector>

namespace M {
/// Invoke the given function on the elements in the provided range
/// asynchronously, if threading is enabled in the MLIR context, otherwise the
/// elements are processed sequentially. This function takes a reference to a
/// cache type that should be sharded to each worker.
template <typename RangeT, typename FuncT, typename CacheT,
          typename ConsolidateCacheFnT>
LogicalResult failableParallelForEach(MLIRContext *ctx, RangeT &&range,
                                      FuncT &&func, CacheT &cache,
                                      ConsolidateCacheFnT &&consolidate) {
  unsigned numElements = llvm::size(range);
  if (!numElements)
    return llvm::success();

  // If threading is not enabled, there is a single element, or the threadpool
  // is single-threaded, process the elements sequentially. Pass in the cache
  // directly.
  auto begin = std::begin(range);
  if (!ctx->isMultithreadingEnabled() || numElements == 1 ||
      ctx->getThreadPool().getMaxConcurrency() == 1) {
    for (auto e = std::end(range); begin != e; ++begin)
      if (failed(func(cache, *begin)))
        return llvm::failure();
    return llvm::success();
  }

  // Otherwise, process the elements in parallel.
  llvm::ThreadPoolInterface &threadPool = ctx->getThreadPool();
  llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
  size_t numActions = std::min(numElements, threadPool.getMaxConcurrency());

  // Each worker gets a copy of the cache.
  std::vector<CacheT> workerCaches(numActions - 1, cache);

  // Build a wrapper processing function that properly initializes a parallel
  // diagnostic handler.
  mlir::ParallelDiagnosticHandler handler(ctx);
  std::atomic<unsigned> curIndex = 0;
  std::atomic<bool> processingFailed = false;
  auto workFn = [&](CacheT &cache) {
    unsigned index;
    while (!processingFailed && (index = curIndex++) < numElements) {
      handler.setOrderIDForThread(index);
      if (failed(func(cache, *std::next(begin, index))))
        processingFailed = true;
      handler.eraseOrderIDForThread();
    }
  };

  // Save 1 copy of the cache.
  tasksGroup.async([&] { workFn(cache); });
  for (CacheT &cache : workerCaches)
    tasksGroup.async([&] { workFn(cache); });
  // If the current thread is a worker thread from the pool, then waiting for
  // the task group allows the current thread to also participate in processing
  // tasks from the group, which avoid any deadlock/starvation.
  tasksGroup.wait();

  // Consolidate the caches.
  consolidate(cache, workerCaches);
  return llvm::failure(processingFailed);
}

/// This version of the function has a no-op consolidate function.
template <typename RangeT, typename FuncT, typename CacheT>
LogicalResult failableParallelForEach(MLIRContext *ctx, RangeT &&range,
                                      FuncT &&func, CacheT &cache) {
  return failableParallelForEach(ctx, std::forward<RangeT>(range),
                                 std::forward<FuncT>(func), cache,
                                 [](auto &&...) {});
}

/// This version of the function is not failable.
template <typename RangeT, typename FuncT, typename CacheT>
void parallelForEach(MLIRContext *ctx, RangeT &&range, FuncT &&func,
                     CacheT &cache) {
  (void)failableParallelForEach(
      ctx, std::forward<RangeT>(range),
      [&](CacheT &cache, auto &&arg) {
        return func(cache, std::forward<decltype(arg)>(arg)), llvm::success();
      },
      cache, [](auto &&...) {});
}

} // namespace M

#endif // SUPPORT_COMPILER_THREADING_H
