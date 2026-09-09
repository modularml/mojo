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

#ifndef ASYNCRT_RUNTIME_ANYASYNCVALUEREF_H
#define ASYNCRT_RUNTIME_ANYASYNCVALUEREF_H

#include "AsyncRT/Runtime/AsyncValue.h"
#include "Support/RCRef.h"

namespace M::AsyncRT {

/// This class holds an (untyped) smart pointer to an AsyncValue, and is the
/// primary API for working with AsyncValues in the Modular cpuDevice.
///
/// References can be moved and (explicitly) copied. The AsyncValue is
/// automatically reference counted and deleted when the last reference is
/// destroyed.
///
/// See also AsyncValueRef<T> for the corresponding smart pointer to an
/// AsyncValue intended to hold a T.
///
/// TODO(#7532): Could store immediately available small values (eg uint32_t)
/// directly in this class with usual low-order pointer bit tagging.
class AnyAsyncValueRef {
public:
  //===--------------------------------------------------------------------===//
  // Smart Pointer operations
  //===--------------------------------------------------------------------===//

  AnyAsyncValueRef() = default;
  ~AnyAsyncValueRef() = default;

  AnyAsyncValueRef(RCRef<AsyncValue> &&rhs) : value(std::move(rhs)) {}

  AnyAsyncValueRef(AnyAsyncValueRef &&rhs) : value(std::move(rhs.value)) {}

  AnyAsyncValueRef &operator=(AnyAsyncValueRef &&rhs) {
    value = std::move(rhs.value);
    return *this;
  }

  RCRef<AsyncValue> releaseRCRef() { return std::move(value); }

  /// This constructor forms a reference to the specified AsyncValue, increasing
  /// the underlying reference count by 1.
  static AnyAsyncValueRef copy(AsyncValue *pointer) {
    return AnyAsyncValueRef(copyRCRef(pointer));
  }

  /// This constructor forms a reference to the specified AsyncValue, taking
  /// ownership it, and thus not increasing the reference count.
  static AnyAsyncValueRef take(AsyncValue *pointer) {
    return AnyAsyncValueRef(takeRCRef(pointer));
  }

  void *getPointerToData() const { return value->getUnderlyingPtr(); }

  /// Return a raw pointer to the AsyncValue. Will be null if reference is null.
  AsyncValue *getPointer() const { return value.getPointer(); }

  /// Take ownership of the underlying pointer away from the AnyAsyncValueRef
  /// and reset it to null. Result will be null if reference is null.
  AsyncValue *releasePointer() { return value.release(); }

  /// Manually drop the reference in this AnyAsyncValueRef, setting it to null.
  void reset() { value.reset(); }

  /// Create an AsyncValue for the specified type in "unconstructed" state.
  /// This should be `emplace`'d, `construct`'d, or finalized with an error.
  template <typename T>
  static AnyAsyncValueRef allocate(CompactCPUDevicePtr cpuDevice) {
    return take(AsyncValue::allocate<T>(cpuDevice));
  }

  /// Create an AsyncValue for the specified type in "available" and ready
  /// state. This is a terminal state for an AsyncValue, it can never change out
  /// of this state.
  template <typename T, typename... Args>
  static AnyAsyncValueRef createReady(CompactCPUDevicePtr cpuDevice,
                                      Args &&...args) {
    return take(
        AsyncValue::createReady<T>(cpuDevice, std::forward<Args>(args)...));
  }

  /// Create an AsyncValue that has already been turned into an error with the
  /// specified message.
  static AnyAsyncValueRef createError(CompactCPUDevicePtr cpuDevice,
                                      EncodedDiagnostic diagnostic) {
    return take(AsyncValue::createError(cpuDevice, std::move(diagnostic)));
  }

  /// Create an IndirectAsyncValue that may be filled in with any AsyncValue in
  /// the future.
  static AnyAsyncValueRef createIndirect(CompactCPUDevicePtr cpuDevice) {
    return take(AsyncValue::createIndirect(cpuDevice));
  }

  // Make a copy of this AnyAsyncValueRef, increasing the AsyncValue's refcount
  // by one.
  AnyAsyncValueRef copy() const { return AnyAsyncValueRef(value.copy()); }

  /// Returns true if this reference is non-null and holds a ConcreteAsyncValue
  /// or a resolved IndirectAsyncValue for values of type T.
  template <typename T>
  bool isType() const {
    return value && value->isType<T>();
  }

  /// Returns true if this reference is null, an unresolved indirect, an error,
  /// or holds a concrete AsyncValue primed for values of type T.
  template <typename T>
  bool isCompatible() const {
    return !value || value->isIndirect() || value->isError() ||
           value->isType<T>();
  }

  /// Test for null.
  explicit operator bool() const { return getPointer() != nullptr; }

  CompactCPUDevicePtr getCPUDevice() const { return value->getCPUDevice(); }

  //===--------------------------------------------------------------------===//
  // Core AsyncValue operations
  //===--------------------------------------------------------------------===//

  /// Sets the referenced AsyncValue to the error state.
  ///
  /// Consumes this reference just before any waiters are triggered. This
  /// ensures waiters which wish to use move-vs-copy or copy-on-write
  /// optimizations on the emplaced value will not see a stray additional
  /// reference due to the producer.
  void setToError(EncodedDiagnostic diagnostic) && {
    AsyncValue *pointer = releasePointer(); // our ref count will be removed by
                                            // setToErrorAndDecRef.
    pointer->setToErrorAndDecRef(std::move(diagnostic));
  }

  /// Return true if this AsyncValue is "ready" and filled with a concrete
  /// value. get() will return a value in this state.
  bool isValueAvailable() const { return value && value->isValueAvailable(); }

  /// Return true if the AsyncValue is "ready" and either filled with a concrete
  /// value or an error.
  bool isReady() const { return value && value->isReady(); }

  /// Return true if the AsyncValue is fulfilled with an error state.
  bool isError() const { return value && value->isError(); }

  /// Constructs the payload of the referenced AsyncValue in place.
  ///
  /// Consumes this reference just before any waiters are triggered. This
  /// ensures waiters which wish to use move-vs-copy or copy-on-write
  /// optimizations on the emplaced value will not see a stray additional
  /// reference due to the producer.
  ///
  /// Synchronous producers will generally require a copy on this references:
  ///
  ///    auto ref = AnyAsyncValueRef::allocate<Foo>(cpuDevice);
  ///    ...
  ///    ref.copy().emplace(...);
  ///    ...
  ///    return ref;
  ///
  /// However asynchronous producers will generally have taken the copy within
  /// their addTask closure:
  ///
  ///    auto ref = AnyAsyncValueRef::allocate<Foo>(cpuDevice);
  ///    addTask(cpuDevice, [ref = ref.copy()]() mutable {
  ///                        ...
  ///                        ref.emplace(...);
  ///                     });
  ///    return ref;
  template <typename T, typename... Args>
  void emplace(Args &&...args) && {
    AsyncValue *pointer =
        releasePointer(); // our ref count will be removed by emplaceAndDecRef.
    pointer->template emplaceAndDecRef<T, Args...>(std::forward<Args>(args)...);
  }

  /// Resolve the referenced IndirectAsyncValue to contain a concrete
  /// AsyncValue with a newly initialized value, resolving any waiters.
  template <typename T, typename... Args>
  void emplaceIndirect(Args &&...args) && {
    AsyncValue *pointer = releasePointer(); // our ref count will be removed by
                                            // emplaceIndirectAndDecRef.
    pointer->template emplaceIndirectAndDecRef<T, Args...>(
        std::forward<Args>(args)...);
  }

  /// Return the stored value in the available referenced AsyncValue.
  template <typename T>
  T &get() const {
    return getPointer()->template get<T>();
  }

  /// Return the moved diagnostic from the referenced AsyncValue, aborting if it
  /// isn't an error.
  EncodedDiagnostic takeDiagnostic() { return value->takeDiagnostic(); }

  /// Return the diagnostic in the referenced AsyncValue, aborting if it isn't
  /// an error.
  const EncodedDiagnostic &getDiagnostic() const {
    return value->getDiagnostic();
  }

  /// If the referenced AsyncValue holds an error, return its diagnostic.
  /// If not, return nullptr.
  EncodedDiagnostic *getDiagnosticIfPresent() const {
    return value->getDiagnosticIfPresent();
  }

  using Waiter = AsyncValue::Waiter;

  /// Register that waiter should be run when the referenced AsyncValue is
  /// ready (with an emplaced value or an error). The waiter should not depend
  /// on the continued existence of the AsyncValue itself. Prefer the consuming
  /// version of andThen if the waiter needs access back to the AsyncValue.
  ///
  /// If `IsAsync` is true, the waiter will be run as an asynchronous task when
  /// triggered, using the work queue for the AsyncValue's cpuDevice. Otherwise,
  /// the waiter will be run either on the caller's thread or the thread of the
  /// emplace or setError call.
  template <bool IsAsync>
  void andThen(Waiter &&waiter) const {
    getPointer()->andThen<IsAsync>(std::move(waiter));
  }

  void andThenSync(Waiter &&waiter) const {
    andThen</*IsAsync=*/false>(std::move(waiter));
  }

  void andThenAsync(Waiter &&waiter) const {
    andThen</*IsAsync=*/true>(std::move(waiter));
  }

  using ConsumingWaiter = llvm::unique_function<void(AnyAsyncValueRef &&ref)>;

  /// Register that waiter should be run when the referenced AsyncValue is
  /// ready (with an emplaced value or an error). This reference will be
  /// consumed by this method, captured by a closure, and passed to the waiter
  /// when triggered. Prefer the non-consuming version of andThen
  /// if the waiter does not need access to the underlying AsyncValue.
  ///
  /// If `IsAsync` is true, the waiter will be run as an asynchronous task when
  /// triggered, using the work queue for the AsyncValue's cpuDevice. Otherwise,
  /// the waiter will be run either on the caller's thread or the thread of the
  /// emplace or setError call.
  template <bool IsAsync>
  void andThen(ConsumingWaiter &&waiter) && {
    AsyncValue *ptr = getPointer();
    ptr->andThen<IsAsync>(
        [ref = std::move(*this), waiter = std::move(waiter)]() mutable {
          waiter(std::move(ref));
        });
  }

  void andThenSync(ConsumingWaiter &&waiter) && {
    std::move(*this).andThen</*IsAsync=*/false>(std::move(waiter));
  }

  void andThenAsync(ConsumingWaiter &&waiter) && {
    std::move(*this).andThen</*IsAsync=*/true>(std::move(waiter));
  }

  /// Resolves the referenced IndirectAsyncValue to point to the specified new
  /// value, resolving any waiters whenever newValue becomes ready.
  ///
  /// Consumes this reference just before any waiters are triggered. This
  /// ensures waiters which wish to use move-vs-copy or copy-on-write
  /// optimizations on the emplaced value will not see a stray additional
  /// reference due to the producer.
  void resolveIndirect(AnyAsyncValueRef &&newValue) && {
    AsyncValue *pointer =
        releasePointer(); // our ref count will be removed by emplaceAndDecRef.
    pointer->resolveIndirectAndDecRef(newValue.releaseRCRef());
  }

private:
  // Not implicitly copyable, use the copy() method for an explicit copy of
  // this reference.
  AnyAsyncValueRef(const AnyAsyncValueRef &) = delete;
  AnyAsyncValueRef &operator=(const AnyAsyncValueRef &) = delete;

  RCRef<AsyncValue> value;
};

} // namespace M::AsyncRT

namespace llvm {

/// Supports the LLVM casting idiom from AnyAsyncValueRefs to their
/// containing value.
template <typename To>
struct CastInfo<To, const ::M::AsyncRT::AnyAsyncValueRef> {
  using From = ::M::AsyncRT::AnyAsyncValueRef;

  static inline bool isPossible(const From &f) { return f.isType<To>(); }
  static inline To *doCast(const From &t) { return &t.get<To>(); }
  static inline To *castFailed() { return nullptr; }
  static inline To *doCastIfPossible(const From &f) {
    if (isa<To>(f))
      return doCast(f);
    return castFailed();
  }
};

} // namespace llvm

#endif // ASYNCRT_RUNTIME_ANYASYNCVALUEREF_H
