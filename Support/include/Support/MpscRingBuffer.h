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

#ifndef SUPPORT_MPSCRINGBUFFER_H
#define SUPPORT_MPSCRINGBUFFER_H

#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>

namespace M {

/// Lock-free multi-producer, single-consumer ring buffer using the Vyukov
/// sequence-number protocol.
///
/// ## Producer protocol (two-phase)
///
///   auto pos = ring.claim();      // claim a slot; nullopt if full
///   if (!pos) { /* drop */ return; }
///   ring.itemAt(*pos) = myItem;      // write into the slot
///   ring.publish(*pos);              // make visible to consumer
///
/// Between claim() and publish() the caller may write into any auxiliary
/// per-slot storage keyed on `*pos & getMask()` — e.g. a parallel string arena.
/// That storage is guaranteed to remain live until the consumer calls
/// consume().
///
/// ## Consumer protocol (two-phase)
///
///   while (T* item = ring.peek()) {
///     size_t pos = ring.frontPos();  // key into auxiliary storage if needed
///     process(*item);                // slot is held open here
///     ring.consume();                // release slot for reuse
///   }
///
/// The slot is not released until consume(), so any auxiliary storage
/// keyed on `pos & getMask()` stays valid for the entire processing window.
///
/// Capacity must be a power of two.
template <typename T>
class MpscRingBuffer {
  struct Slot {
    std::atomic<size_t> sequence{0};
    T item{};
  };

  const size_t capacity;
  const size_t mask;
  std::unique_ptr<Slot[]> slots;
  // Separate cache lines to avoid false sharing between producers and consumer.
  alignas(64) std::atomic<size_t> enqueuePos{0};
  alignas(64) std::atomic<size_t> dequeuePos{0};

public:
  explicit MpscRingBuffer(size_t capacity)
      : capacity(capacity), mask(capacity - 1),
        slots(std::make_unique<Slot[]>(capacity)) {
    assert((capacity & (capacity - 1)) == 0 &&
           "MpscRingBuffer capacity must be a power of 2");
    for (size_t i = 0; i < capacity; ++i)
      slots[i].sequence.store(i, std::memory_order_relaxed);
  }

  MpscRingBuffer(const MpscRingBuffer &) = delete;
  MpscRingBuffer &operator=(const MpscRingBuffer &) = delete;

  /// Claim a slot for writing. Returns the slot position on success, or
  /// nullopt if the buffer is full (caller should drop the item).
  std::optional<size_t> claim() {
    size_t pos = enqueuePos.load(std::memory_order_relaxed);
    for (;;) {
      Slot &slot = slots[pos & mask];
      size_t seq = slot.sequence.load(std::memory_order_acquire);
      auto diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
      if (diff == 0) {
        if (enqueuePos.compare_exchange_weak(pos, pos + 1,
                                             std::memory_order_relaxed))
          return pos;
      } else if (diff < 0) {
        return std::nullopt; // buffer full
      } else {
        pos = enqueuePos.load(std::memory_order_relaxed);
      }
    }
  }

  /// Returns a reference to the item storage at a claimed slot position.
  /// Valid between claim() and publish() for the same pos.
  T &itemAt(size_t pos) { return slots[pos & mask].item; }

  /// Publishes a claimed slot to the consumer.
  void publish(size_t pos) {
    slots[pos & mask].sequence.store(pos + 1, std::memory_order_release);
  }

  /// Returns a pointer to the next available item, or nullptr if empty. The
  /// slot remains held until consume() is called; auxiliary storage keyed on
  /// frontPos() & getMask() is valid for the same window.
  T *peek() {
    size_t pos = dequeuePos.load(std::memory_order_relaxed);
    Slot &slot = slots[pos & mask];
    size_t seq = slot.sequence.load(std::memory_order_acquire);
    if (static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1) == 0)
      return &slot.item;
    return nullptr;
  }

  /// Returns the position of the currently peeked slot. Valid when
  /// peek() returned non-null; use `frontPos() & getMask()` to index into
  /// auxiliary per-slot storage.
  size_t frontPos() const { return dequeuePos.load(std::memory_order_relaxed); }

  /// Releases the current front slot and advances the consumer position.
  void consume() {
    size_t pos = dequeuePos.load(std::memory_order_relaxed);
    slots[pos & mask].sequence.store(pos + capacity, std::memory_order_release);
    dequeuePos.store(pos + 1, std::memory_order_release);
  }

  /// Total positions claimed by producers (monotonically increasing).
  size_t enqueueCount() const {
    return enqueuePos.load(std::memory_order_acquire);
  }

  /// Total positions consumed (monotonically increasing).
  size_t consumeCount() const {
    return dequeuePos.load(std::memory_order_acquire);
  }

  size_t getMask() const { return mask; }
};

} // namespace M

#endif // SUPPORT_MPSCRINGBUFFER_H
