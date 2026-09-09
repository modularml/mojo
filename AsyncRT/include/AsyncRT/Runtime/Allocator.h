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
// This file declares the M::AsyncRT::Allocator interface, which allows clients
// of AsyncRT to implement custom allocation and other fancy policies.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_ALLOCATOR_H
#define ASYNCRT_RUNTIME_ALLOCATOR_H

#include "Support/AlignedAlloc.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/Threading/HWInfo.h"

#include <memory>

namespace M::AsyncRT {

/// Profiling entry for uses of memcpy via profiledMemcpy.
using MemCopyProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kMem, 1), Trace::kMem>;

/// Profiling entry for all allocs and frees.
using MemAllocFreeProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kMem, 2), Trace::kMem>;

/// This class defines an abstract interface for custom allocators to implement.
/// This is intended for use by large object allocations (e.g. tensor data), not
/// for use by every single small allocation that happens in the execution of a
/// program (don't route std::string's through this!).
///
class Allocator {
public:
  virtual ~Allocator() = default;

  /// Allocate the specified number of bytes with the specified alignment.
  virtual void *allocateBytes(size_t size, size_t alignment) = 0;

  /// Deallocate the specified pointer that had the specified size.
  virtual void deallocateBytes(void *ptr, size_t size = 0) = 0;

  /// Returns the NUMA node this allocator is bound to, and will therefore place
  /// memory in, or returns `kAnyNumaNode` if the allocator has no NUMA
  /// affinity.
  int getNumaPlacement() const { return numaPlacement; }

  /// Allocate memory for one or more entries of type T.
  template <typename T>
  T *allocate(size_t numElements = 1) {
    return static_cast<T *>(
        allocateBytes(sizeof(T) * numElements, kPreferredMemoryAlignment));
  }

  /// Deallocate the memory for one or more entries of type T.
  template <typename T>
  void deallocate(T *ptr, size_t numElements = 1) {
    deallocateBytes(ptr, sizeof(T) * numElements);
  }

  /// Allocate and initialize an object of type T.
  template <typename T, typename... Args>
  T *construct(Args &&...args) {
    T *buf = allocate<T>();
    return new (buf) T(std::forward<Args>(args)...);
  }

  /// Destruct and deallocate space for an object of type T.
  template <typename T>
  void destroy(T *t) {
    t->~T();
    deallocate(t);
  }

  /// Destruct and deallocate space for one or more object of type T.
  template <typename T>
  void destroyAndDeallocate(T *ptr, size_t numElements = 1) {
    for (size_t i = 0; i != numElements; ++i)
      ptr[i].~T();
    deallocate(ptr, numElements);
  }

protected:
  Allocator() = default;
  explicit Allocator(int numaPlacement) : numaPlacement(numaPlacement) {}
  Allocator(const Allocator &) = delete;
  void operator=(const Allocator &) = delete;

private:
  int numaPlacement = kAnyNumaNode;
  virtual void vtableAnchor();
};

/// Create an allocator that just calls malloc/free.
std::unique_ptr<Allocator> createMallocAllocator();

/// Create an allocator that uses tcmalloc
std::unique_ptr<Allocator> createTCMallocAllocator();

/// Create a TCMalloc-backed allocator bound to a specified NUMA node.
std::unique_ptr<Allocator> createTCMallocAllocator(int numaPlacement);

/// Create a wrapper allocator that checks to make sure all memory is
/// deallocated when the allocator itself is destroyed.
std::unique_ptr<Allocator>
createLeakCheckAllocator(std::unique_ptr<Allocator> baseAllocator);

/// Create a wrapper allocator that prints memory profiling information when it
/// is destroyed.  This also performs leak checks.
std::unique_ptr<Allocator>
createProfilingAllocator(std::unique_ptr<Allocator> baseAllocator);

#if !defined(_WIN64) && !defined(_WIN32)
#define HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR 1
#else
#define HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR 0
#endif

#if HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR
/// Returns an allocator which will read/write protect every allocated
/// block to detect use-after-free errors as soon as they occur
/// (without depending on ASAN). Allocated blocks are freed only when
/// the allocator is freed.
///
/// For use when ASAN build is not available. Expensive!
std::unique_ptr<Allocator> createUseAfterFreeAllocator();
#endif

/// As for std::memcpy, but wrapped by profiling if enabled. There's no
/// alignment constraints on dst and src, and they need not have been
/// allocated by one of our allocators.
void profiledMemcpy(void *dst, const void *src, size_t size);

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_ALLOCATOR_H
