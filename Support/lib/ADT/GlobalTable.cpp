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

#include "Support/ADT/GlobalTable.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

using namespace M;

/// LockFreeGlobalEntry is used by GlobalTable to store global values
/// in a lock-free hash table.
struct GlobalTable::LockFreeGlobalEntry {
  std::atomic<void *> value{nullptr};
  std::atomic<void (*)(void *)> destroyFn{nullptr};
  std::atomic<LockFreeGlobalEntry *> nextInOrder{nullptr};
  std::string name;

  LockFreeGlobalEntry(StringRef nameRef, void *val, void (*destroyer)(void *))
      : value(val), destroyFn(destroyer), name(nameRef.str()) {}

  void destroy() {
    if (void (*destroyer)(void *) =
            destroyFn.exchange(nullptr, std::memory_order_acq_rel))
      if (void *val = value.exchange(nullptr, std::memory_order_acq_rel))
        destroyer(val);
  }
  void *getValue() const { return value.load(std::memory_order_acquire); }
}; // struct LockFreeGlobalEntry

void *GlobalTable::getOrCreate(StringRef name, void *(*initFn)(),
                               void (*destroyFn)(void *)) {
  uint64_t hash = llvm::hash_value(name);
  size_t startIndex = hash & (kTableSize - 1);

  // This uses linear probing to handle hash collisions.
  for (size_t probe = 0; probe < kMaxProbes; ++probe) {
    size_t index = (startIndex + probe) & (kTableSize - 1);
    LockFreeGlobalEntry *entry =
        hashTable[index].load(std::memory_order_acquire);

    if (entry) {
      if (name == entry->name)
        // This found an existing entry.
        return entry->getValue();
      // This is a hash collision, so continue probing.
      continue;
    }

    // This found an empty slot, so check if we can create new entry.
    if (!initFn)
      return nullptr;

    // This creates entry outside critical section.
    void *newValue = initFn();
    if (!newValue)
      return nullptr;

    LockFreeGlobalEntry *newEntry =
        new LockFreeGlobalEntry(name, newValue, destroyFn);
    if (!newEntry) {
      if (destroyFn)
        destroyFn(newValue);
      return nullptr;
    }

    // This uses atomic CAS to claim the slot.
    LockFreeGlobalEntry *expected = nullptr;
    if (hashTable[index].compare_exchange_strong(expected, newEntry,
                                                 std::memory_order_acq_rel)) {
      // This won the race, so add to destruction order list.
      insertIntoOrderList(newEntry);
      return newEntry->getValue();
    }

    // This lost the race, so cleanup our entry.
    if (destroyFn)
      destroyFn(newValue);
    delete newEntry;

    // This checks if the winner has our name.
    if (expected && name == expected->name)
      return expected->getValue();

    // The winner had different name, so continue probing from next slot.
    // The entry we want might have been placed at a later position.
  }

  // The hash table probe limit exhausted, so try overflow container.
  // But first check if we need to look there to avoid mutex on hot path.
  if (!initFn) {
    // This is lookup only, so check overflow if it has entries.
    if (hasOverflowEntries.load(std::memory_order_acquire))
      return getFromOverflow(name);
    return nullptr;
  }

  // This is creation case, so must use overflow container.
  return getOrCreateInOverflow(name, initFn, destroyFn);
}

void GlobalTable::clear() {
  // For proper LIFO destruction order, we need to destroy entries in this
  // order:
  // 1. Overflow entries first since they are most recent and came after the
  // hash tabled reached capacity.
  // 2. Then hash table entries in LIFO order.

  {
    std::lock_guard<std::mutex> lock(overflowMutex);
    // This destroys in reverse order for LIFO since MapVector stores in
    // insertion order.
    for (auto it = overflowTable.rbegin(), end = overflowTable.rend();
         it != end; ++it)
      it->second.destroy();
    overflowTable.clear();
    hasOverflowEntries.store(false, std::memory_order_relaxed);
  }

  // We atomically take ownership of the hash table destruction list.
  LockFreeGlobalEntry *current =
      orderHead.exchange(nullptr, std::memory_order_acq_rel);

  // Then we clear the hash table's pointers;.
  for (std::atomic<LockFreeGlobalEntry *> &slot : hashTable)
    slot.store(nullptr, std::memory_order_relaxed);

  // Finally we destroy the hash table entries in LIFO order.
  while (current) {
    LockFreeGlobalEntry *next =
        current->nextInOrder.load(std::memory_order_relaxed);
    current->destroy();
    delete current;
    current = next;
  }
}

void GlobalTable::insertIntoOrderList(LockFreeGlobalEntry *entry) {
  // This inserts at head of list for LIFO destruction order.
  LockFreeGlobalEntry *oldHead = orderHead.load(std::memory_order_relaxed);
  do {
    entry->nextInOrder.store(oldHead, std::memory_order_relaxed);
  } while (!orderHead.compare_exchange_weak(oldHead, entry,
                                            std::memory_order_release));
}

void *GlobalTable::getFromOverflow(StringRef name) const {
  std::lock_guard<std::mutex> lock(overflowMutex);
  auto it = overflowTable.find(name.str());
  return (it != overflowTable.end()) ? it->second.value : nullptr;
}

void *GlobalTable::getOrCreateInOverflow(StringRef name, void *(*initFn)(),
                                         void (*destroyFn)(void *)) {
  std::lock_guard<std::mutex> lock(overflowMutex);

  // This checks if already exists.
  if (auto it = overflowTable.find(name.str()); it != overflowTable.end())
    return it->second.value;

  if (!initFn)
    return nullptr;

  // Try to create a new value.
  void *value = initFn();
  if (!value)
    return nullptr;

  // Wrap the value and the destructor in an Overflow entry and place it
  // in the MapVector-based overflow table.
  overflowTable.insert({name.str(), OverflowGlobalEntry(value, destroyFn)});

  // Always set the flag as the overflow table is guaranteed to have entries.
  hasOverflowEntries.store(true, std::memory_order_relaxed);

  return value;
}
