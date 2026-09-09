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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/Threading.h"
#include <cstdint>

#ifndef SUPPORT_THREADING_THREADLOCALCACHE_H
#define SUPPORT_THREADING_THREADLOCALCACHE_H

namespace M {

//===----------------------------------------------------------------------===//
// ThreadLocalCache
//===----------------------------------------------------------------------===//

/// This class manages a set of thread-local caches. This class is useful when
/// an in-memory analysis cache needs to be shared across multiple threads,
/// where each thread needs to update the cache (deterministically), but it is
/// more efficient to let threads mutate their own local copies of the cache
/// which can be merged back into the main cache.
template <typename CacheT>
class ThreadLocalCache {
public:
  /// Construct a thread-local cache manager given the maximum number of threads
  /// that may try to access the cache. This ensures the map does not resize.
  explicit ThreadLocalCache(CacheT &original, unsigned maxNumThreads)
      : original(original) {
    // Reserve the thread-local cache map so that it never resizes.
    threadCaches.reserve(maxNumThreads);
  }

  /// Get the cache instance for the current thread.
  CacheT &getThreadLocalCache() {
    int64_t threadId = llvm::get_threadid();
    {
      llvm::sys::SmartScopedReader<true> lock(mutex);
      if (auto it = threadCaches.find(threadId); it != threadCaches.end())
        return it->second;
    }
    llvm::sys::SmartScopedWriter<true> lock(mutex);
    // Each thread gets a copy of the saved cache.
    return threadCaches.try_emplace(threadId, original).first->second;
  }

  /// Consolidate the thread-local caches back into the main cache instance, if
  /// desired.
  void consolidate(llvm::function_ref<void(CacheT &, const CacheT &)> unionFn) {
    for (const CacheT &cache : llvm::make_second_range(threadCaches))
      unionFn(original, cache);
  }

private:
  /// The original cache to make copies from.
  CacheT &original;
  /// Thread-local instances of the cache.
  llvm::DenseMap<uint64_t, CacheT> threadCaches;
  /// The mutex guarding the thread-local cache instances.
  llvm::sys::SmartRWMutex<true> mutex;
};

} // namespace M

#endif // SUPPORT_THREADING_THREADLOCALCACHE_H
