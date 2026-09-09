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

#ifndef ASYNCRT_SUPPORT_LOCKFREERINGBUFFER_H
#define ASYNCRT_SUPPORT_LOCKFREERINGBUFFER_H

#include "Support/Threading/Atomics.h"
#include "Support/Threading/SpinWaiter.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <memory>

namespace M::AsyncRT {

/// This class provides a lock-free ring buffer for concurrent access.
/// NOTE: Currently the size of the ring buffer is fixed at the construction
/// time. We may want to implement resize.
template <typename ItemType>
class LockFreeRingBuffer {

public:
  LockFreeRingBuffer(size_t size)
      : size(llvm::NextPowerOf2(size)),
        buffer(std::make_unique<ItemType[]>(this->size)) {
    assert(llvm::isPowerOf2_64(this->size) &&
           "Ring buffer size is not power of 2.");
  }

  ~LockFreeRingBuffer() {
    assert(readIndex.load() == writeIndex.load() &&
           "Cannot destroy a non-empty ring buffer!");
  }

  /// Enqueue adds the object to the circular buffer and returns true, or
  /// returns false if the buffer is full.
  ///
  /// On success, it takes ownership of the object, std::move'ing from it.
  bool enqueue(ItemType &item) {
    // Make sure that the buffer is not full.
    uint64_t curConsumed = consumed.load(std::memory_order_acquire);
    uint64_t curWriteIndex = writeIndex.load(std::memory_order_acquire);
    if (curWriteIndex - curConsumed >= size)
      return false;

#ifdef MODULAR_DEBUG
    // This is not technically an overflow yet, but will result overflow by
    // the next enqueue. This is close enough to invalidate the underlying
    // assumption that the total number of enqueues will never exceed the
    // uint64_t max value. We check the overflow for `curWriteIndex` as this
    // moves ahead of all other atomic variables.
    assert(curWriteIndex < std::numeric_limits<uint64_t>::max() &&
           "Index overflow.");
#endif

    // Claim the ownership of `curWriteIndex`. If `compare_exchange_weak`
    // succeeds, we can make sure that 1) writing to `buffer[curWriteIndex %
    // size] does not overwrite any unconsumed item, and 2) no other threads
    // will write to `buffer[curWriteIndex % size]` simultaneously.
    SpinWaiter<> indexWaiter;
    while (!writeIndex.compare_exchange_weak(curWriteIndex, curWriteIndex + 1,
                                             std::memory_order_acq_rel)) {
      // New `curWriteIndex` needs to be compared against `consumed` again.
      if (curWriteIndex - consumed.load(std::memory_order_acquire) >= size)
        return false;

      // Wait a bit and retry.
      indexWaiter.wait();
    }

    // Now we can safely write to `buffer[curWriteIndex & (size - 1)]`, which is
    // effectively `buffer[curWriteIndex % size]` when size is power of 2.
    buffer[curWriteIndex & (size - 1)] = std::move(item);

    // Update `published` to indicate that the value is actually written and
    // ready to consume. Check that the value of `published` is same as
    // `curWriteIndex`, to make sure that the values are published in order.
    SpinWaiter<> publishWaiter;
    while (published.load(std::memory_order_acquire) != curWriteIndex)
      publishWaiter.wait();
    published.store(curWriteIndex + 1, std::memory_order_release);
    return true;
  }

  /// Dequeue returns the stored item to the caller and release the ownership of
  /// the item. Returns a value-initialized `ItemType` if the buffer is empty.
  ItemType dequeue() {
    // Make sure that the buffer is not empty.
    uint64_t curPublished = published.load(std::memory_order_acquire);
    uint64_t curReadIndex = readIndex.load(std::memory_order_acquire);
    if (curPublished <= curReadIndex)
      return ItemType();

    // Claim the ownership of `consumed`. If `compare_exchange_weak` succeeds,
    // we can make sure that 1) `buffer[curConsumed % size]` contains a valid
    // item, and 2) no other threads is taking the item from `buffer[curConsumed
    // % size]`.
    SpinWaiter<> indexWaiter;
    while (!readIndex.compare_exchange_weak(curReadIndex, curReadIndex + 1,
                                            std::memory_order_acq_rel)) {
      // Check again if we have enough values published.
      if (published.load(std::memory_order_acquire) <= curReadIndex)
        return ItemType();

      // Wait a bit and retry.
      indexWaiter.wait();
    }

    // Now we can safely read from `buffer[curReadIndex & (size - 1)]`, which
    // is effectively `buffer[curReadIndex % size]` when size is power of 2.
    auto ret = std::move(buffer[curReadIndex & (size - 1)]);

    // Update `consumed` to tell writing threads that the slot
    // `buffer[consumed % size]` can be overwritten. Check the value of
    // `consumed` is same as `curReadIndex`, to make sure that the slots became
    // available in order.
    SpinWaiter<> consumeWaiter;
    while (consumed.load(std::memory_order_acquire) != curReadIndex)
      consumeWaiter.wait();
    consumed.store(curReadIndex + 1, std::memory_order_release);
    return ret;
  }

private:
  LockFreeRingBuffer(const LockFreeRingBuffer &other) = delete;
  LockFreeRingBuffer &operator=(const LockFreeRingBuffer &other) = delete;

  const size_t size;
  std::unique_ptr<ItemType[]> buffer;

  /// The ring buffer is implemented with 4 atomic variables. `writeIndex`
  /// maintains the next slot to be written while `readIndex` maintains the next
  /// slot to be read. `published` follows `writeIndex` when an item is actually
  /// written to `buffer[writeIndex % size]`, and `consumed` follows `readIndex`
  /// when an item is finished read from `buffer[readIndex % size]`. So the
  /// logic for `enqueue` is
  ///
  /// 1. Take the ownership of next `writeIndex` and atomically increases
  /// `writeIndex` by 1.
  /// 2. Write an item to `writeIndex % size`.
  /// 3. Update `published` to the new `writeIndex`.
  ///
  /// If we only have `writeIndex` but not `published`, we cannot block other
  /// threads to access the slot during the transient state between a thread
  /// having an ownership of `writeIndex` but not finished writing the item
  /// yet. Same applies to `dequeue` operation.
  ///
  /// The atomic variables are monotonically increasing. This may
  /// encounter overflow issue, but with uint64_t, even when we add 2^20
  /// elements per second, it takes ~557844 years for the overflow to happen.
  /// They are aligned with cache line size to avoid false sharing.
  /// TODO: Make the implementation handle the overflow.
  AlignedAtomic<uint64_t> readIndex = 0;
  AlignedAtomic<uint64_t> writeIndex = 0;
  AlignedAtomic<uint64_t> consumed = 0;
  AlignedAtomic<uint64_t> published = 0;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_LOCKFREERINGBUFFER_H
