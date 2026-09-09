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
// This file defines wrapper allocators that keep track of extra metadata.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Allocator.h"
#include "Support/ADT/ConcurrentAppendingVector.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Process.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/Threading/Atomics.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>

/// Whether to capture system resident memory profiling samples in the
/// ProfilingAllocator.
constexpr bool kCaptureSysMem = true;

/// Whether to capture malloc outstanding memory usage (in blocks) in
/// the ProfilingAllocator.
constexpr bool kCaptureMalloc = true;

#if HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR
#include <sys/mman.h>
#endif

using namespace M;
using namespace AsyncRT;

//===----------------------------------------------------------------------===//
// Leak Checking Allocator
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "leak-check"

namespace {
class LeakCheckAllocator : public Allocator {
public:
  explicit LeakCheckAllocator(std::unique_ptr<Allocator> baseAllocator)
      : Allocator(baseAllocator->getNumaPlacement()),
        baseAllocator(std::move(baseAllocator)) {}
  ~LeakCheckAllocator() override { checkLeak(); }

  void *allocateBytes(size_t size, size_t alignment) override {
    ++numAllocations;
    numBytesAllocated.fetch_add(size);
    auto ptr = baseAllocator->allocateBytes(size, alignment);
    {
      std::lock_guard<std::mutex> lock(mu);
      sizeMap[ptr] = size;
    }
    LLVM_DEBUG(llvm::dbgs()
               << llvm::Twine("allocateBytes(") + llvm::Twine(size) +
                      "B) = 0x" +
                      llvm::Twine::utohexstr(reinterpret_cast<intptr_t>(ptr)) +
                      "\n");
    return ptr;
  }

  void deallocateBytes(void *ptr, size_t size) override {
    // Tolerate deallocating null.
    if (!ptr)
      return;

    [[maybe_unused]] ssize_t num = numAllocations--;
    assert(num > 0 && "calls to deallocateBytes not balanced by allocateBytes");
    size_t storedSize;
    {
      std::lock_guard<std::mutex> lock(mu);
      auto itr = sizeMap.find(ptr);
      assert(itr != sizeMap.end() && "deallocating ptr which is not allocated");
      storedSize = itr->second;
      sizeMap.erase(itr);
    }
    assert((size == 0 || size == storedSize) && "size mismatch at dealloc");
    size = storedSize;
    [[maybe_unused]] ssize_t bytes = numBytesAllocated.fetch_sub(size);
    assert(bytes >= static_cast<ssize_t>(size) &&
           "deallocating more bytes than currently have outstanding");
    baseAllocator->deallocateBytes(ptr, size);
    LLVM_DEBUG(llvm::dbgs()
               << llvm::Twine("deallocateBytes(0x") +
                      llvm::Twine::utohexstr(reinterpret_cast<intptr_t>(ptr)) +
                      ", " + llvm::Twine(size) + "B)\n");
  }

  /// Print a message and exit(1) when memory leak is detected.
  void checkLeak() {
    if (numAllocations.load() == 0 && numBytesAllocated.load() == 0)
      return;
    llvm::errs() << "Memory leak detected: " << numAllocations.load()
                 << " alive allocations, " << numBytesAllocated.load()
                 << " alive bytes:\n";
    for (auto [ptr, size] : sizeMap) {
      llvm::errs() << "  0x"
                   << llvm::Twine::utohexstr(reinterpret_cast<intptr_t>(ptr))
                   << " (" << size << "B)\n";
    }
    llvm::report_fatal_error("Memory leak detected");
  }

protected:
  /// This keeps track of how many bytes/allocations are currently alive.
  /// Uses ssize_t so we can track double deallocation.
  std::atomic<ssize_t> numBytesAllocated{0}, numAllocations{0};
  std::mutex mu;
  llvm::DenseMap<void *, size_t> sizeMap;

private:
  std::unique_ptr<Allocator> baseAllocator;
};
} // namespace

/// Create a wrapper allocator that checks to make sure all memory is
/// deallocated when the allocator itself is destroyed.
std::unique_ptr<Allocator>
M::AsyncRT::createLeakCheckAllocator(std::unique_ptr<Allocator> baseAllocator) {
  return std::make_unique<LeakCheckAllocator>(std::move(baseAllocator));
}

//===----------------------------------------------------------------------===//
// Profiling Allocator
//===----------------------------------------------------------------------===//

namespace {

/// Profiling entry for sampling outstanding bytes allocated on every alloc
/// and free.
using MemProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kMem, 3), Trace::kMem>;

class ProfilingAllocator : public LeakCheckAllocator {
public:
  explicit ProfilingAllocator(std::unique_ptr<Allocator> baseAllocator)
      : LeakCheckAllocator(std::move(baseAllocator)) {}

  void *allocateBytes(size_t size, size_t alignment) override {
    void *result = LeakCheckAllocator::allocateBytes(size, alignment);
    ++totalAllocations;
    atomicAdd(totalBytesAllocated, size);
    atomicMax(maxAllocations, numAllocations.load());
    atomicMax(maxBytesAllocated, numBytesAllocated.load());

    recordProfilingSamples();

    return result;
  }

  void deallocateBytes(void *ptr, size_t size) override {
    LeakCheckAllocator::deallocateBytes(ptr, size);

    recordProfilingSamples();
  }

  ~ProfilingAllocator() override {
    llvm::errs() << "-----------------------------------------------------\n";
    llvm::errs() << "M::AsyncRT::Allocator profile:\n";
    llvm::errs() << "  Total number of allocations:           "
                 << totalAllocations.load() << "\n";
    llvm::errs() << "  Total bytes allocated:                 "
                 << totalBytesAllocated.load() << "\n";
    llvm::errs() << "  Max number of outstanding allocations: "
                 << maxAllocations.load() << "\n";
    llvm::errs() << "  Max outstanding bytes allocated:       "
                 << maxBytesAllocated.load() << "\n";
    llvm::errs() << "-----------------------------------------------------\n";
    llvm::errs().flush();

    // If we still have active memory alive, print an error.
    checkLeak();
  }

  void recordProfilingSamples() {
    MemProfilerEntry::sample(numBytesAllocated.load(),
                             StringLiteral("mem.outstanding"));
    if constexpr (kCaptureMalloc) {
      MemProfilerEntry::sample(llvm::sys::Process::GetMallocUsage(),
                               StringLiteral("mem.malloc_outstanding"));
    }
    if constexpr (kCaptureSysMem) {
      MemProfilerEntry::sample(getProcessPhysicalMemUsage(),
                               StringLiteral("mem.sys_resident"));
    }
  }

  /// High-water marks for numAllocations and numBytesAllocated.
  /// Use ssize_t for consistency with LeakCheckAllocator.
  std::atomic<ssize_t> maxAllocations{0}, maxBytesAllocated{0};
  /// Total number of bytes allocated.
  std::atomic<size_t> totalBytesAllocated{0};
  /// Total number of calls to allocateBytes.
  std::atomic<size_t> totalAllocations{0};
};
} // namespace

/// Create a wrapper allocator that prints memory profiling information when it
/// is destroyed.  This also performs leak checks.
std::unique_ptr<Allocator>
M::AsyncRT::createProfilingAllocator(std::unique_ptr<Allocator> baseAllocator) {
  return std::make_unique<ProfilingAllocator>(std::move(baseAllocator));
}

//===----------------------------------------------------------------------===//
// UseAfterFree Allocator
//===----------------------------------------------------------------------===//

#if HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR

namespace {
class UseAfterFreeAllocator : public Allocator {
public:
  UseAfterFreeAllocator() : allocations(1024) {}

  void *allocateBytes(size_t size, size_t alignment) override {
    if (!size)
      return nullptr;
    void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    allocations.emplace_back(ptr, size);
    return ptr;
  }

  void deallocateBytes(void *ptr, size_t size) override {
    mprotect(ptr, size, PROT_NONE);
  }

  ~UseAfterFreeAllocator() override {
    size_t size = allocations.size();
    for (size_t i = 0; i < size; ++i)
      munmap(allocations[i].first, allocations[i].second);
  }

private:
  ConcurrentAppendingVector<std::pair<void *, size_t>> allocations;
};
} // namespace

std::unique_ptr<Allocator> M::AsyncRT::createUseAfterFreeAllocator() {
  return std::make_unique<UseAfterFreeAllocator>();
}

#endif
