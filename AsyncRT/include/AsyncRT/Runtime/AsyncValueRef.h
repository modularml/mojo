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

#ifndef ASYNCRT_RUNTIME_ASYNCVALUEREF_H
#define ASYNCRT_RUNTIME_ASYNCVALUEREF_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValue.h"

namespace M::AsyncRT {

/// This class specializes AnyAsyncValueRef to assume the target AsyncValue
/// is (intended to hold) a T. Thus the get() and emplace() methods don't
/// require additional template parameters.
template <typename T>
class AsyncValueRef : public AnyAsyncValueRef {
public:
  //===--------------------------------------------------------------------===//
  // Smart Pointer operations
  //===--------------------------------------------------------------------===//

  AsyncValueRef() = default;
  ~AsyncValueRef() = default;

  AsyncValueRef(RCRef<AsyncValue> &&rhs) : AnyAsyncValueRef(std::move(rhs)) {
    assert(isCompatible<T>() &&
           "Constructing AsyncValueRef<T> from incompatible RCRef<AsyncValue>");
  }

  AsyncValueRef(AnyAsyncValueRef &&rhs) : AnyAsyncValueRef(rhs.releaseRCRef()) {
    assert(isCompatible<T>() &&
           "Constructing AsyncValueRef<T> from incompatible AnyAsyncValueRef");
  }

  AsyncValueRef(AsyncValueRef<T> &&rhs) : AnyAsyncValueRef(std::move(rhs)) {}

  AsyncValueRef<T> &operator=(AsyncValueRef<T> &&rhs) {
    AnyAsyncValueRef::operator=(std::move(rhs));
    return *this;
  }

  // Allow implicit conversion from AsyncValueRef<Derived> to
  // AsyncValueRef<Base>.
  template <typename DerivedT,
            std::enable_if_t<std::is_base_of<T, DerivedT>::value, int> = 0>
  AsyncValueRef(AsyncValueRef<DerivedT> &&rhs)
      : AnyAsyncValueRef(std::move(rhs)) {}

  /// This constructor forms a reference to the specified pointer, increasing
  /// the underlying reference count by 1.
  static AsyncValueRef<T> copy(AsyncValue *pointer) {
    auto res = AsyncValueRef<T>(copyRCRef(pointer));
    assert(res.template isCompatible<T>() &&
           "Constructing AsyncValueRef<T> from incompatible AsyncValue*");
    return res;
  }

  /// This constructor forms a reference to the specified pointer, taking
  /// ownership of it, and thus not increasing the reference count.
  static AsyncValueRef<T> take(AsyncValue *pointer) {
    auto res = AsyncValueRef<T>(takeRCRef(pointer));
    assert(res.template isCompatible<T>() &&
           "Constructing AsyncValueRef<T> from incompatible AsyncValue*");
    return res;
  }

  /// Create an AsyncValue for the specified type in "unconstructed" state.
  /// This should be `emplace`'d, `construct`'d, or finalized with an error.
  static AsyncValueRef<T> allocate(CompactCPUDevicePtr cpuDevice) {
    return take(AsyncValue::allocate<T>(cpuDevice));
  }

  /// Create an AsyncValue for the specified type in "available" and ready
  /// state. This is a terminal state for an AsyncValue, it can never change out
  /// of this state.
  template <typename... Args>
  static AsyncValueRef<T> createReady(CompactCPUDevicePtr cpuDevice,
                                      Args &&...args) {
    return take(
        AsyncValue::createReady<T>(cpuDevice, std::forward<Args>(args)...));
  }

  /// Create an AsyncValue that has already been turned into an error with the
  /// specified message.
  static AsyncValueRef<T> createError(CompactCPUDevicePtr cpuDevice,
                                      EncodedDiagnostic diagnostic) {
    return take(AsyncValue::createError(cpuDevice, std::move(diagnostic)));
  }

  T &operator*() const { return getPointer()->template get<T>(); }

  T *operator->() const { return &getPointer()->template get<T>(); }

  // Make an explicit copy of this AsyncValueRef, increasing the AsyncValue's
  // refcount by one.
  AsyncValueRef<T> copy() const { return AnyAsyncValueRef::copy(); }

  //===--------------------------------------------------------------------===//
  // Core AsyncValue operations
  //===--------------------------------------------------------------------===//

  /// Constructs the payload of the referenced AsyncValue in place, and changes
  /// its state to "available". See AnyAsyncValueRef::emplace for more details.
  template <typename... Args>
  void emplace(Args &&...args) && {
    AsyncValue *pointer =
        releasePointer(); // our ref count will be removed by emplaceAndDecRef.
    pointer->template emplaceAndDecRef<T, Args...>(std::forward<Args>(args)...);
  }

  /// Return the stored value in the referenced available AsyncValue.
  T &get() const { return getPointer()->template get<T>(); }

  /// Register that waiter should be run when the referenced AsyncValue is
  /// ready (with an emplaced value or an error). See AnyAsyncValueRef::andThen
  /// for more details.
  template <bool IsAsync>
  void andThen(Waiter &&waiter) const {
    getPointer()->template andThen<IsAsync>(std::move(waiter));
  }

  void andThenSync(Waiter &&waiter) const {
    andThen</*IsAsync=*/false>(std::move(waiter));
  }

  void andThenAsync(Waiter &&waiter) const {
    andThen</*IsAsync=*/true>(std::move(waiter));
  }

  using ConsumingWaiter = llvm::unique_function<void(AsyncValueRef<T> &&ref)>;

  /// Register that waiter should be run when the referenced AsyncValue is
  /// ready (with an emplaced value or an error). See AnyAsyncValueRef::andThen
  /// for more details.
  template <bool IsAsync>
  void andThen(ConsumingWaiter &&waiter) && {
    AsyncValue *ptr = getPointer();
    ptr->andThen<IsAsync>(
        [ref = std::move(*this), waiter = std::move(waiter)]() mutable {
          waiter(std::move(ref));
        });
  }

  void andThenSync(ConsumingWaiter &&waiter) && {
    std::move(*this).template andThen</*IsAsync=*/false>(std::move(waiter));
  }

  void andThenAsync(ConsumingWaiter &&waiter) && {
    std::move(*this).template andThen</*IsAsync=*/true>(std::move(waiter));
  }
};

//===----------------------------------------------------------------------===//
// AsyncValueRefWithEncodedLocation
//===----------------------------------------------------------------------===//

/// This template may be used where it is useful to bundle together a reference
/// to an AsyncValue (either AnyAsyncValueRef or AsyncValueRef<T>) with an
/// EncodedLocation.
///
/// This value is larger than an AsyncValue reference (3 words instead of 1) and
/// involves more reference counting (EncodedLocations need to keep their
/// decoder alive), so it should only be used where needed.
template <typename AVRefType>
class AsyncValueRefWithEncodedLocation : public AVRefType {
public:
  AsyncValueRefWithEncodedLocation(AVRefType refValue, EncodedLocation loc)
      : AVRefType(std::move(refValue)), loc(std::move(loc)) {}

  AsyncValueRefWithEncodedLocation(AsyncValueRefWithEncodedLocation &&) =
      default;

  /// Fill the referenced AsyncValue with an error that has the specified
  /// message.
  void setToError(Error message) && {
    std::move(*this).AVRefType::setToError({std::move(message), loc.copy()});
  }

  // Make an explicit copy of this AsyncValueRefWithEncodedLocation.
  AsyncValueRefWithEncodedLocation copy() const {
    return {AVRefType::copy(), loc.copy()};
  }

  /// Provide access to the location.
  const EncodedLocation &getLocation() const { return loc; }

private:
  EncodedLocation loc;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_ASYNCVALUEREF_H
