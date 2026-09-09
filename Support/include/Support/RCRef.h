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

#ifndef SUPPORT_RCREF_H
#define SUPPORT_RCREF_H

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace M {

/// This is a smart pointer that keeps the specified reference counted value
/// around.  It is move-only to avoid accidental copies, but it can be copied
/// explicitly.
template <typename T>
class RCRef {
public:
  RCRef() : pointer(nullptr) {}
  RCRef(RCRef &&other) : pointer(other.pointer) { other.pointer = nullptr; }

  // See below; prefer the `copy` method.
  RCRef(const RCRef &other) : RCRef(std::move(other.copy())) {}

  /// This constructor forms a reference to the specified pointer, increasing
  /// the underlying reference count by 1.
  static RCRef copy(T *pointer) {
    if (pointer)
      pointer->addRef();
    return take(pointer);
  }

  /// This constructor forms a reference to the specified pointer, taking
  /// ownership it, and thus not increasing the reference count.
  static RCRef take(T *pointer) {
    RCRef<T> ref;
    ref.pointer = pointer;
    return ref;
  }

  /// Create an instance of T with the specified constructor arguments and
  /// return it as an RCRef.
  template <typename... Args>
  static RCRef create(Args &&...args) {
    return take(new T(std::forward<Args>(args)...));
  }

  /// Support implicit conversion from RCRef<Derived> to RCRef<Base>.
  template <typename U,
            typename = std::enable_if_t<std::is_base_of<T, U>::value>>
  RCRef(RCRef<U> &&u) : pointer(u.release()) {}

  ~RCRef() {
    if (pointer) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
      pointer->dropRef();
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    }
  }

  RCRef &operator=(RCRef &&other) {
    if (pointer)
      pointer->dropRef();
    pointer = other.pointer;
    other.pointer = nullptr;
    return *this;
  }

  /// Manually drop the reference in this RCRef, setting it to null.
  void reset() {
    if (pointer)
      pointer->dropRef();
    pointer = nullptr;
  }

  /// Take ownership of the underlying pointer away from the RCRef and reset it
  /// to null.
  T *release() {
    T *tmp = pointer;
    pointer = nullptr;
    return tmp;
  }

  T &operator*() const {
    assert(pointer && "null RCRef");
    return *pointer;
  }

  T *operator->() const {
    assert(pointer && "null RCRef");
    return pointer;
  }

  /// Return a raw pointer.
  T *getPointer() const { return pointer; }

  void *getOpaquePointer() const { return reinterpret_cast<void *>(pointer); }

  /// Make an explicit copy of this RCRef, increasing the refcount by one.
  RCRef<T> copy() const { return RCRef<T>::copy(pointer); }

  /// Test for null.
  explicit operator bool() const { return pointer != nullptr; }

  void swap(RCRef &other) {
    using std::swap;
    swap(pointer, other.pointer);
  }

  /// Low level access for manipulating reference counts.  This allows the
  /// addRef/dropRef to be private on the classes themselves, forcing clients to
  /// go through RCRef.
  static void lowLevelAddRef(T *pointer) { pointer->addRef(); }
  static void lowLevelDropRef(T *pointer) { pointer->dropRef(); }
  static void lowLevelAddRef(T *pointer, size_t amount) {
    pointer->addRef(amount);
  }
  static void lowLevelDropRef(T *pointer, size_t amount) {
    pointer->dropRef(amount);
  }

private:
  // Per above, this type *is* implicitly copyable, as this is required for use
  // in certain contexts where the parent type is copied. However, this is
  // strongly discouraged, and we can disable at least the implicit copy
  // assignment operator. The `copy` method is strongly preferred.
  RCRef &operator=(const RCRef &) = delete;

  T *pointer;
};

// These global functions help make type inference work better.

/// Forms a reference to the specified pointer, increasing the underlying
/// reference count by 1.
template <typename T>
inline RCRef<T> copyRCRef(T *ptr) {
  return RCRef<T>::copy(ptr);
}

/// Form a reference to the specified pointer, taking ownership it, and thus not
/// increasing the reference count.
template <typename T>
inline RCRef<T> takeRCRef(T *ptr) {
  return RCRef<T>::take(ptr);
}

/// For ADL style swap.
template <typename T>
inline void swap(RCRef<T> &a, RCRef<T> &b) {
  a.swap(b);
}

} // namespace M

#endif // SUPPORT_RCREF_H
