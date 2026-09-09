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

#ifndef SUPPORT_ADT_GENERICRCREF_H
#define SUPPORT_ADT_GENERICRCREF_H

#include "Support/RCRef.h"
#include "Support/TypeID.h"
#include "llvm/ADT/FunctionExtras.h"
#include <utility>

namespace M {

/// A type-erased version of RCRef<T>, where T is recorded using
/// a TypeID::get<T>() and cast (with type safety check) on access. As usual,
/// the referenced object must implement (typically thread-safe) addRef() and
/// dropRef() methods to track outstanding references. The payload may
/// have other GenericRCRef and RCRef<T> references to it.
///
/// See also GenericUniquePtr for the unique-pointer equivalent.
class GenericRCRef {
public:
  /// Constructs the null reference.
  GenericRCRef() = default;

  /// Constructs a generic reference from an RCRef<T>.
  template <typename T>
  static GenericRCRef fromRCRef(RCRef<T> &&ref) {
    return take(ref.release());
  }

  /// Forms a new reference to payload, incrementing its reference count.
  template <typename T>
  static GenericRCRef copy(T *payload) {
    if (payload)
      RCRef<T>::lowLevelAddRef(payload);
    return take(payload);
  }

  /// Forms a reference to payload without incrementing its reference count.
  template <typename T>
  static GenericRCRef take(T *payload) {
    GenericRCRef result;
    result.payload = payload;
    result.typeId = M::TypeID::get<T>();
    result.dropRef = [](void *ptr) {
      RCRef<T>::lowLevelDropRef(static_cast<T *>(ptr));
    };
    return result;
  }

  /// Creates a new instance of T with the given constructor arguments, and
  /// returns it as a GenericRCRef. We assume the payload is constructed with
  /// one reference.
  template <typename T, typename... Args>
  static GenericRCRef create(Args &&...args) {
    return take(new T(std::forward<Args>(args)...));
  }

  /// Drops a reference to the payload, if any.
  ~GenericRCRef() { reset(); }

  // No implicit copying is allowed.
  GenericRCRef(const GenericRCRef &rhs) = delete;
  GenericRCRef &operator=(const GenericRCRef &rhs) = delete;

  // Move is supported.
  GenericRCRef(GenericRCRef &&rhs) { swap(*this, rhs); }
  GenericRCRef &operator=(GenericRCRef &&rhs) {
    swap(*this, rhs);
    rhs.reset();
    return *this;
  }

  /// Returns a copy of this GenericRCRef as an RCRef<T>.
  template <typename T>
  RCRef<T> copy() const {
    return RCRef<T>::copy(get<T>());
  }

  /// Returns true if reference has a payload.
  operator bool() const { return payload; }

  /// Returns the type id of the payload, or the invalid type id if reference
  /// has no payload.
  M::TypeID getTypeID() const { return typeId; }

  /// Returns the payload as type T*, or null if reference has no payload.
  /// If reference has a payload then the static type id for T must match the
  /// type id held by the reference.
  template <typename T>
  T *get() {
    if (payload) {
      typeId.assertEqual(M::TypeID::get<T>(), "GenericRCRef::get");
      return static_cast<T *>(payload);
    } else {
      return nullptr;
    }
  }

  template <typename T>
  const T *get() const {
    if (payload) {
      typeId.assertEqual(M::TypeID::get<T>(), "GenericRCRef::get");
      return static_cast<const T *>(payload);
    } else {
      return nullptr;
    }
  }

  /// Returns a void* opaque pointer to the underlying payload
  void *getOpaquePointer() const { return payload; }

  /// Releases and returns the payload as type T*, or returns null if reference
  /// has no payload. If the reference has a payload then the static type id for
  /// T must match the type id held by the reference.
  template <typename T>
  T *release() {
    if (payload) {
      typeId.assertEqual(M::TypeID::get<T>(), "GenericRCRef::release");
      T *result = static_cast<T *>(payload);
      payload = nullptr;
      typeId = TypeID();
      dropRef = nullptr;
      return result;
    } else {
      return nullptr;
    }
  }

private:
  static void swap(GenericRCRef &lhs, GenericRCRef &rhs) {
    std::swap(lhs.typeId, rhs.typeId);
    std::swap(lhs.payload, rhs.payload);
    std::swap(lhs.dropRef, rhs.dropRef);
  }

  /// Drops a reference to any existing payload and resets reference to the
  /// null state.
  void reset() {
    if (payload) {
      dropRef(payload);
      payload = nullptr;
      typeId = TypeID();
      dropRef = nullptr;
    }
  }

  /// The payload. We have one outstanding reference count on this object.
  /// Is actually a T* where typeID == TypeID::get<T>().
  void *payload = nullptr;

  /// The dropRef method for the above.
  llvm::unique_function<void(void *)> dropRef;

  /// Type id uniquely describing the type of payload.
  TypeID typeId;
};

} // namespace M

#endif // SUPPORT_ADT_GENERICRCREF_H
