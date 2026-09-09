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

#ifndef SUPPORT_REFERENCECOUNTED_H
#define SUPPORT_REFERENCECOUNTED_H

#include <atomic>
#include <cassert>
#include <cstdint>

namespace M {
template <typename T>
class RCRef;

/// This class is a convenience base class for things that need an atomic
/// intrusive reference count for ownership management.  It is implemented with
/// a CRTP pattern where the subclass is specified as a template argument.  It
/// implicitly makes any derived classes non-copyable and non-assignable.
///
/// Subclasses of this are allowed to implement a destroy() instance method,
/// which allows custom allocation/deallocation logic.  By default, the object
/// is deallocated with `delete` when the refcount drops to zero.
///
/// This class intentionally doesn't have a virtual destructor or anything that
/// would require a vtable, but subclasses can have one if they choose.
///
template <typename SubClass>
class ReferenceCounted {
public:
  explicit ReferenceCounted(); // Initialize with refcount = 1.
  ~ReferenceCounted();

  // Return reference count. This should be used for testing and debugging only.
  uint32_t getNumReferences() const { return refCount.load(); }

  /// Return true if reference count is 1.
  bool isUnique() const {
    return refCount.load(std::memory_order_acquire) == 1;
  }

private:
  // Reference counting, only accessible to RCRef<>.
  template <typename T>
  friend class RCRef;

  // Add a new reference to this object.
  void addRef(uint32_t count = 1) const {
    // It is OK to use std::memory_order_relaxed here as it does not affect the
    // ownership state of the object.
    refCount.fetch_add(count, std::memory_order_relaxed);
  }

  // Drop a reference to this object, potentially deallocating it.
  void dropRef(uint32_t count = 1) const {
    // If refCount == count, this object is owned only by the caller. Bypass a
    // locked op in that case.
    if (refCount.fetch_sub(count, std::memory_order_acq_rel) == count) {
      // Make assert in ~ReferenceCounted happy
      assert((refCount.store(0, std::memory_order_relaxed), true));
      static_cast<const SubClass *>(this)->destroy();
    }
  }

protected:
  // Subclasses are allowed to customize this, but the default implementation of
  // destroy() just deletes the pointer.
  void destroy() const { delete static_cast<const SubClass *>(this); }

private:
  // Not copyable or movable.
  ReferenceCounted(const ReferenceCounted &) = delete;
  ReferenceCounted &operator=(const ReferenceCounted &) = delete;

  mutable std::atomic<uint32_t> refCount;
};

#ifdef MODULAR_DEBUG
/// In debug builds we keep track of the number of reference counted objects,
/// which enables clients to check that none are alive at key moments.  This is
/// a low-tech way to find certain classes of memory leaks.
extern std::atomic<std::size_t> currentReferenceCountedObjects;
#endif // MODULAR_DEBUG

/// Verify that there are no live ReferenceCounted objects that are currently
/// alive and print the specified message and abort if there are.
void verifyNoLiveReferenceCountedObjects(const char *errorMessage);

template <typename SubClass>
inline ReferenceCounted<SubClass>::ReferenceCounted() : refCount(1) {
#ifdef MODULAR_DEBUG
  ++currentReferenceCountedObjects;
#endif // MODULAR_DEBUG
}

template <typename SubClass>
inline ReferenceCounted<SubClass>::~ReferenceCounted() {
  assert(refCount.load() == 0 &&
         "Shouldn't destroy a reference counted object with references!");
#ifdef MODULAR_DEBUG
  --currentReferenceCountedObjects;
#endif // MODULAR_DEBUG
}

/// This class is a convenience base class for things that need non-atomic
/// intrusive reference count for ownership management.  It is implemented with
/// a CRTP pattern where the subclass is specified as a template argument.  It
/// implicitly makes any derived classes non-copyable and non-assignable.
///
/// Subclasses of this are allowed to implement a destroy() instance method,
/// which allows custom allocation/deallocation logic.  By default, the object
/// is deallocated with `delete` when the refcount drops to zero.
///
/// This class intentionally doesn't have a virtual destructor or anything that
/// would require a vtable, but subclasses can have one if they choose.
///
template <typename SubClass>
class NonAtomicallyReferenceCounted {
public:
  explicit NonAtomicallyReferenceCounted() : refCount(1) {}
  ~NonAtomicallyReferenceCounted() {
    assert(refCount == 0 &&
           "Shouldn't destroy a reference counted object with references!");
  }

  // Return reference count. This should be used for testing and debugging only.
  uint32_t getNumReferences() const { return refCount; }

  /// Return true if reference count is 1.
  bool isUnique() const { return refCount == 1; }

private:
  // Reference counting, only accessible to RCRef<>.
  template <typename T>
  friend class RCRef;

  // Add a new reference to this object.
  void addRef(uint32_t count = 1) const { refCount += count; }

  // Drop a reference to this object, potentially deallocating it.
  void dropRef(uint32_t count = 1) const {
    refCount -= count;
    if (refCount == 0)
      static_cast<const SubClass *>(this)->destroy();
  }

protected:
  // Subclasses are allowed to customize this, but the default implementation of
  // destroy() just deletes the pointer.
  void destroy() const { delete static_cast<const SubClass *>(this); }

private:
  // Not copyable or movable.
  NonAtomicallyReferenceCounted(const NonAtomicallyReferenceCounted &) = delete;
  NonAtomicallyReferenceCounted &
  operator=(const NonAtomicallyReferenceCounted &) = delete;

  mutable uint32_t refCount;
};

} // namespace M

#endif // SUPPORT_REFERENCECOUNTED_H
