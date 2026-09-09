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

#ifndef SUPPORT_ADT_GENERICUNIQUEPTRSET_H
#define SUPPORT_ADT_GENERICUNIQUEPTRSET_H

#include "Support/ADT/GenericUniquePtr.h"
#include "Support/ErrorOr.h"
#include "Support/ReferenceCounted.h"

#include "llvm/ADT/MapVector.h"

#include "llvm/ADT/FunctionExtras.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace M {

/// A set of type-erased GenericUniquePtrs. Pointers may be added and retrieved.
/// The set may contain at most one pointer per concrete pointer type.
///
/// Thread safe, though the caller is responsible for thread safe access to
/// the held objects themselves.
class GenericUniquePtrSet {
public:
  GenericUniquePtrSet() = default;

  /// Tears down in reverse order.
  ~GenericUniquePtrSet() {
    while (map.size() > 0)
      map.pop_back();
  }

  /// Transfers ptr into set. The set may hold at most one object per T.
  template <typename T>
  void set(std::unique_ptr<T> ptr) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    GenericUniquePtr genericPtr;
    genericPtr.reset(std::move(ptr));
    auto denseIndex = genericPtr.getTypeID().getDenseIndex();
    assert(!map.contains(denseIndex) && "set already holds object of type");
    map.insert({denseIndex, std::move(genericPtr)});
  }

  /// Emplaces a new object of type T into the set and returns a reference to
  /// it. The set may hold at most one object object per T. The returned
  /// reference is stable for the lifetime of the set.
  template <typename T, typename... Args>
  T &emplace(Args &&...args) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    auto genericPtr = makeGenericUniquePtr<T>(std::forward<Args>(args)...);
    auto denseIndex = genericPtr.getTypeID().getDenseIndex();
    assert(!map.contains(denseIndex) && "set already holds object of type");
    T &result = *genericPtr.template get<T>();
    map.insert({denseIndex, std::move(genericPtr)});
    return result;
  }

  /// Returns a reference to the object of type T held by the set. If it does
  /// not exist, emplaces a new object and returns a reference to it. The
  /// returned reference is stable for the lifetime of the set.
  template <typename T, typename... Args>
  T &emplaceIfMissing(Args &&...args) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    auto denseIndex = TypeID::get<T>().getDenseIndex();
    auto itr = map.find(denseIndex);
    if (itr == map.end()) {
      auto genericPtr = makeGenericUniquePtr<T>(std::forward<Args>(args)...);
      T &result = *genericPtr.template get<T>();
      map.insert({denseIndex, std::move(genericPtr)});
      return result;
    } else {
      return *(itr->second.template get<T>());
    }
  }

  /// Returns a pointer to the object of type T held by the set. If it does
  /// not exist, calls the creator function to create one and install it.
  /// The function must always succeed.
  template <typename T>
  T *createIfMissing(llvm::unique_function<std::unique_ptr<T>()> creator) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    auto denseIndex = TypeID::get<T>().getDenseIndex();
    auto itr = map.find(denseIndex);
    if (itr != map.end())
      return itr->second.template get<T>();
    std::unique_ptr<T> owner = creator();
    T *result = owner.get();
    GenericUniquePtr genericPtr;
    genericPtr.reset(std::move(owner));
    map.insert({denseIndex, std::move(genericPtr)});
    return result;
  }

  /// Returns a pointer to the object of type T held by the set. If it does
  /// not exist, calls the creator function to create one and install it.
  /// Returns any error the creator function returns.
  template <typename T>
  ErrorOr<T *> createIfMissing(
      llvm::unique_function<ErrorOr<std::unique_ptr<T>>()> creator) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    auto denseIndex = TypeID::get<T>().getDenseIndex();
    auto itr = map.find(denseIndex);
    if (itr != map.end())
      return itr->second.template get<T>();
    ErrorOr<std::unique_ptr<T>> errOr = creator();
    if (errOr.isError())
      return errOr.takeError();
    T *result = errOr->get();
    GenericUniquePtr genericPtr;
    genericPtr.reset(std::move(*errOr));
    map.insert({denseIndex, std::move(genericPtr)});
    return result;
  }

  /// Returns a pointer to the object of type T held by the set, or nullptr
  /// if no such object exists.
  template <typename T>
  T *get() {
    std::lock_guard<std::recursive_mutex> lock(mu);
    auto denseIndex = TypeID::get<T>().getDenseIndex();
    auto itr = map.find(denseIndex);
    if (itr == map.end())
      return nullptr;
    return itr->second.template get<T>();
  }

private:
  /// Protects map. Recursive so that the creator in createIfMissing may also
  /// add pointer objects.
  mutable std::recursive_mutex mu;

  /// A map from globally unique type identifiers TypeID::get<T>() (using
  /// their 'dense index' form) to GenericUniquePtr holding an object of
  /// type T.
  llvm::MapVector<size_t, GenericUniquePtr> map;
};

/// An RCRef-compatible version of GenericUniquePtrSet.
class SharedGenericUniquePtrSet
    : public GenericUniquePtrSet,
      public ReferenceCounted<SharedGenericUniquePtrSet> {};
using GenericUniquePtrSetRef = RCRef<SharedGenericUniquePtrSet>;

} // namespace M

#endif // SUPPORT_ADT_GENERICUNIQUEPTRSET_H
