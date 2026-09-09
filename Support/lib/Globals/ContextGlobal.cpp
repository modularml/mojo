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

#include "Support/ContextGlobal.h"

#include <cassert>
#include <mutex>

namespace M {

/// Accessor for the global context mutex. Function-local static avoids
/// -Wglobal-constructors (and global destructor) on macOS/Clang.
std::mutex &getGlobalContextMutex() {
  static std::mutex m;
  return m;
}

/// Accessor for the global context pointer. Kept in the same TU as the mutex.
static Context *&getGlobalContextPtr() {
  static Context *ptr = nullptr;
  return ptr;
}

Context *getCurrentMaxContextPointerOrNull() { return getGlobalContextPtr(); }

void setCurrentMaxContextPointer(Context *ptr) { getGlobalContextPtr() = ptr; }

void clearGlobalContextPointerIfEquals(Context *ptr) {
  if (getGlobalContextPtr() == ptr) {
    getGlobalContextPtr() = nullptr;
  }
}

} // namespace M
