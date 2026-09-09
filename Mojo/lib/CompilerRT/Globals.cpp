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
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>

using namespace M;

static GlobalTable &getGlobalTable() {
  static GlobalTable globalTable;
  return globalTable;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetOrCreateGlobal(llvm::StringRef name, void *(*initFn)(),
                                  void (*destroyFn)(void *)) {
  auto &globalTable = getGlobalTable();
  return globalTable.getOrCreate(name, initFn, destroyFn);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetGlobalOrNull(llvm::StringRef name) {
  return KGEN_CompilerRT_GetOrCreateGlobal(name, nullptr, nullptr);
}

/// getInsertValue provides thread-local storage for InsertGlobal value passing.
static void *&getInsertValue() {
  static thread_local void *gInsertValue = nullptr;
  return gInsertValue;
}

static void *insertGlobalInitFn() { return getInsertValue(); }

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_InsertGlobal(llvm::StringRef name, void *value) {
  auto &globalTable = getGlobalTable();

  getInsertValue() = value;
  globalTable.getOrCreate(name, insertGlobalInitFn, nullptr);
}

//===----------------------------------------------------------------------===//
// Indexed globals for well known constants.
//===----------------------------------------------------------------------===//

namespace {
struct AtomicGlobalEntry {
  std::atomic<void *> value;
  std::atomic<void (*)(void *)> destroyFn;

  void destroy() {
    if (auto loadedValue = value.load()) {
      value.store(nullptr);
      if (auto loadedDestroyFn = destroyFn.load()) {
        destroyFn.store(nullptr);
        loadedDestroyFn(loadedValue);
      }
    }
  }
};
} // namespace

/// Keep this as big as the indexed globals in ffi.mojo.
#define NUM_INDEXED_GLOBALS 3
static AtomicGlobalEntry indexedTable[NUM_INDEXED_GLOBALS];

/// A faster version of GetOrCreateGlobal that doesn't need to lock the table
/// or hash the name.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetOrCreateGlobalIndexed(size_t index, void *(*initFn)(),
                                         void (*destroyFn)(void *)) {
  assert(index < NUM_INDEXED_GLOBALS && "Unsupported indexed global #");

  // Most accesses will be initialized.
  auto entry = indexedTable[index].value.load();
  if (entry)
    return entry;

  // If not, create a value.
  auto newValue = initFn();
  // Try to swap it in, replacing a nullptr.
  if (!indexedTable[index].value.compare_exchange_strong(entry, newValue)) {
    // If we raced and someone else won, delete whatever we just created.
    destroyFn(newValue);
    return entry;
  }
  // Unconditionally set the destroy function. It should always be the same for
  // anyone racing on this.
  indexedTable[index].destroyFn.store(destroyFn);
  return newValue;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_DestroyGlobals() {
  auto &globalTable = getGlobalTable();
  globalTable.clear();

  // Destroy indexed globals last.
  for (auto &entry : indexedTable)
    entry.destroy();
}
