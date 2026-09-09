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

#ifndef SUPPORT_ADT_GENERICUNIQUEPTR_H
#define SUPPORT_ADT_GENERICUNIQUEPTR_H

#include "Support/TypeID.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <memory>
#include <utility>

namespace M {

/// A type-erased version of std::unique_ptr<T>, where T is recorded using
/// a TypeID::get<T>() and cast (with type safety check) on access. Not
/// thread safe.
///
/// See also GenericRCRef for the reference counting equivalent.
class GenericUniquePtr {
public:
  /// Constructs the null pointer.
  GenericUniquePtr() = default;

  /// Transfers ownership of newPayload into pointer, releasing any
  /// previously held payload.
  template <typename T>
  void reset(std::unique_ptr<T> newPayload) {
    assert(newPayload && "GenericUniquePtr new payload cannot be null");
    // Mimic std::unique_ptr in deleting existing payload only after fields
    // have been updated.
    void *oldPayload = payload;
    llvm::unique_function<void(void *)> oldDeletor = std::move(deletor);

    typeId = M::TypeID::get<T>();
    deletor = [](void *ptr) {
      typename std::unique_ptr<T>::deleter_type defaultDeletor;
      defaultDeletor(static_cast<T *>(ptr));
    };
    payload = newPayload.release();

    if (oldPayload)
      oldDeletor(oldPayload);
  }

  /// Destroys any payload.
  ~GenericUniquePtr() { reset(); }

  // No implicit copying is allowed.
  GenericUniquePtr(const GenericUniquePtr &rhs) = delete;
  GenericUniquePtr &operator=(const GenericUniquePtr &rhs) = delete;

  // Move is supported.
  GenericUniquePtr(GenericUniquePtr &&rhs) { swap(*this, rhs); }
  GenericUniquePtr &operator=(GenericUniquePtr &&rhs) {
    swap(*this, rhs);
    rhs.reset();
    return *this;
  }

  /// Returns true if pointer has a payload.
  operator bool() const { return payload; }

  /// Returns the type id of the payload, or the invalid type id if pointer
  /// has no payload.
  TypeID getTypeID() const { return typeId; }

  /// Returns the payload as type T*, or null if pointer has no payload.
  /// If pointer has a payload then the static type id for T must match the
  /// type id held by the pointer.
  template <typename T>
  T *get() {
    if (payload) {
      typeId.assertEqual(M::TypeID::get<T>(), "GenericUniquePtr::get");
      return static_cast<T *>(payload);
    } else {
      return nullptr;
    }
  }

  template <typename T>
  const T *get() const {
    if (payload) {
      typeId.assertEqual(M::TypeID::get<T>(), "GenericUniquePtr::get");
      return static_cast<const T *>(payload);
    } else {
      return nullptr;
    }
  }

  /// Releases and returns the payload as type T*, or returns null if pointer
  /// has no payload. If the pointer has a payload then the static type id for
  /// T must match the type id held by the pointer.
  template <typename T>
  T *release() {
    if (payload) {
      typeId.assertEqual(M::TypeID::get<T>(), "GenericUniquePtr::release");
      T *result = static_cast<T *>(payload);
      payload = nullptr;
      typeId = TypeID();
      deletor = nullptr;
      return result;
    } else {
      return nullptr;
    }
  }

private:
  static void swap(GenericUniquePtr &lhs, GenericUniquePtr &rhs) {
    std::swap(lhs.typeId, rhs.typeId);
    std::swap(lhs.payload, rhs.payload);
    std::swap(lhs.deletor, rhs.deletor);
  }

  /// Delete any existing payload and reset pointer to the null state.
  void reset() {
    if (payload) {
      deletor(payload);
      payload = nullptr;
      typeId = TypeID();
      deletor = nullptr;
    }
  }

  /// The payload. We own this object. Is actually a T* where
  /// typeID == TypeID::get<T>().
  void *payload = nullptr;

  /// The deletor for the above.
  llvm::unique_function<void(void *)> deletor;

  /// Type id uniquely describing the type of payload.
  TypeID typeId;
};

/// Returns a GenericUniquePtr holding a T constructed from args.
template <typename T, typename... Args>
inline GenericUniquePtr makeGenericUniquePtr(Args &&...args) {
  GenericUniquePtr ptr;
  ptr.reset(std::make_unique<T>(std::forward<Args>(args)...));
  return ptr;
}

} // namespace M

namespace llvm {
template <typename T>
struct CastInfo<T, M::GenericUniquePtr> {
  static bool isPossible(const M::GenericUniquePtr &ptr) {
    return ptr.getTypeID() == M::TypeID::get<T>();
  }

  static T *doCast(M::GenericUniquePtr &f) { return f.get<T>(); }

  static T *castFailed() { return nullptr; }

  static T *doCastIfPossible(M::GenericUniquePtr &f) {
    if (!isPossible(f))
      return nullptr;
    return doCast(f);
  }
};

// Provide casting for const pointers.
template <typename T>
struct CastInfo<T, const M::GenericUniquePtr>
    : public ConstStrippingForwardingCast<T, const M::GenericUniquePtr,
                                          CastInfo<T, M::GenericUniquePtr>> {};

template <>
struct ValueIsPresent<M::GenericUniquePtr> {
  using UnwrappedType = M::GenericUniquePtr;
  static inline bool isPresent(const M::GenericUniquePtr &t) { return bool(t); }
  static inline decltype(auto) unwrapValue(M::GenericUniquePtr &t) { return t; }
};
} // namespace llvm

#endif // SUPPORT_ADT_GENERICUNIQUEPTR_H
