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

#ifndef SUPPORT_ADT_GLOBALTABLE_H
#define SUPPORT_ADT_GLOBALTABLE_H

#include "Support/ADT/DenseStringMap.h" // IWYU pragma: keep
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>

namespace M {

/// OverflowGlobalEntry represents an entry for the GlobalTable's
/// overflow MapVector when the hash table is full.
struct OverflowGlobalEntry {
  void *value;
  void (*destroyFn)(void *);

  OverflowGlobalEntry(void *value, void (*destroyFn)(void *))
      : value(value), destroyFn(destroyFn) {}

  void destroy() {
    if (destroyFn && value)
      destroyFn(value);
  }
};

/// GlobalTable is a generic typeless storage used by Mojo's CompilerRT
/// interface to ffi._Globals.
//
/// This implementation uses a hybrid approach with fixed size lock-free hash
/// map for the main table and a mutex-protected overflow container when the
/// hash table exceeds capacity.
struct GlobalTable {
  void *getOrCreate(llvm::StringRef name, void *(*initFn)(),
                    void (*destroyFn)(void *));

  void clear();

private:
  struct LockFreeGlobalEntry;
  void insertIntoOrderList(LockFreeGlobalEntry *entry);
  void *getFromOverflow(llvm::StringRef name) const;
  void *getOrCreateInOverflow(llvm::StringRef name, void *(*initFn)(),
                              void (*destroyFn)(void *));

  static constexpr size_t kTableSize = 4096;
  static constexpr size_t kMaxProbes = 12;

  std::array<std::atomic<LockFreeGlobalEntry *>, kTableSize> hashTable{};
  std::atomic<LockFreeGlobalEntry *> orderHead{nullptr};

  // The overflowTable container is used when the hashTable capacity is reached.
  std::atomic<bool> hasOverflowEntries{false};
  mutable std::mutex overflowMutex;
  llvm::MapVector<std::string, OverflowGlobalEntry> overflowTable;
};

} // namespace M

#endif // SUPPORT_ADT_GLOBALTABLE_H
