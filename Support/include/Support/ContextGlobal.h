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
// Global current M::Context pointer. Implemented in the Globals shared library
// so there is a single definition per process. The public API lives in
// Support/Context.h; this header declares the internal pointer-based accessors.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CONTEXT_GLOBAL_H
#define SUPPORT_CONTEXT_GLOBAL_H

#include <mutex>

#include "Support/SymbolExport.h"

namespace M {

class Context;

/// Mutex serializing access to the global context pointer and related TLS
/// coordination. Defined in the Globals shared library (single definition per
/// process).
MODULAR_CXX_EXPORT std::mutex &getGlobalContextMutex();

/// Returns the current global context pointer, or nullptr if none set.
MODULAR_CXX_EXPORT Context *getCurrentMaxContextPointerOrNull();

/// Sets the global context pointer.
MODULAR_CXX_EXPORT void setCurrentMaxContextPointer(Context *ptr);

/// If the global context pointer equals \p ptr, clears it. Called from
/// Context::~Context() so the global is cleared when the last ref is destroyed.
MODULAR_CXX_EXPORT void clearGlobalContextPointerIfEquals(Context *ptr);

} // namespace M

#endif // SUPPORT_CONTEXT_GLOBAL_H
