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

#ifndef SUPPORT_CONTEXT_H
#define SUPPORT_CONTEXT_H

#include "Support/ADT/GenericRCRef.h"
#include "Support/ADT/GenericUniquePtrSet.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/SymbolExport.h"
#include "Support/TypeID.h"
#include "llvm/ADT/FunctionExtras.h"
#include <memory>

namespace M {

class Context : public ReferenceCounted<Context> {
public:
  /// Transfers ptr into the context object set.
  template <typename T>
  void set(std::unique_ptr<T> ptr) {
    storage.set(std::move(ptr));
  }

  /// Sets the cpuDevice for this context, called at context creation. Uses
  /// GenericRCRef because of a circular dependency (Context -> AsyncRT/Runtime
  /// -> AsyncRT/Support -> MArchTarget -> MDialect -> Context). See GEX-3400.
  void setRuntime(GenericRCRef ref) { cpuDeviceRef = std::move(ref); }

  /// Emplaces a new object of type T into the context object set and returns a
  /// reference to it.
  template <typename T, typename... Args>
  T &emplace(Args &&...args) {
    return storage.emplace<T, Args...>(std::forward<Args>(args)...);
  }

  /// Returns a reference to the object of type T held by the context object
  /// set. If it does not contain such an object, emplaces a new object and
  /// returns a reference to it.
  template <typename T, typename... Args>
  T &emplaceIfMissing(Args &&...args) {
    return storage.emplaceIfMissing<T, Args...>(std::forward<Args>(args)...);
  }

  /// Returns a pointer to the object of type T held by the context object set.
  /// If it does not contain such an object, calls the creator function to
  /// create one and install. Returns any error the creator function returns.
  template <typename T>
  ErrorOr<T *> createIfMissing(
      llvm::unique_function<ErrorOr<std::unique_ptr<T>>()> creator) {
    return storage.createIfMissing<T>(std::move(creator));
  }

  /// Returns a pointer to the context object of type T held by the context
  /// object set, or nullptr if no such object exists. The CPUDeviceRef is
  /// stored as a separate member so is checked first.
  template <typename T>
  T *get() {
    if (cpuDeviceRef && cpuDeviceRef.getTypeID() == TypeID::get<T>())
      return cpuDeviceRef.get<T>();
    return storage.get<T>();
  }

  MODULAR_CXX_EXPORT ~Context();

private:
  GenericRCRef cpuDeviceRef;
  GenericUniquePtrSet storage;
};

/// Convenience definitions.
using ContextRef = RCRef<Context>;

} // namespace M

#endif // SUPPORT_CONTEXT_H
