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
// This file declares AsyncValue, a lightweight and generic "future" type that
// can be fulfilled by an asynchronously provided value or an error.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_ASYNCVALUE_H
#define ASYNCRT_RUNTIME_ASYNCVALUE_H

#include "AsyncRT/Runtime/CompactCPUDevicePtr.h"
#include "AsyncRT/Runtime/Globals/Globals.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/Diagnostic.h"
#include "Support/AlignedAlloc.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/TypeID.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>

namespace M::AsyncRT {
class CPUDevice;
class WaiterListNode;
class AsyncValue;
namespace Detail {
template <typename T>
class ConcreteAsyncValue;

} // namespace Detail

/// This is a future of the specified value type. Arbitrary C++ types may be
/// used here, even non-copyable types and expensive ones like "your database".
/// All AsyncValues are allocated out of a specific `Runtime` instance and can
/// identify them with `getCPUDevice()`.
///
/// An AsyncValue is in one of four states (unconstructed, constructed,
/// available, error), where the first two are considered "non-ready" and the
/// last two are considered "ready" (waiters are notified).  If it is in the
/// non-ready state, it may have a list of waiters which are notified when the
/// value transitions to a ready state.
//
/// AsyncValue has two possible representations, depending on whether the
/// creator knows the ultimate payload type or not.  If so, we use the
/// ConcreteAsyncValue<T> subclass, which stores the metadata and payload data
/// consecutively (reducing allocations and improving cache effectiveness).  If
/// not, the more general IndirectAsyncValue class adds a level of indirection
/// that allows the payload type to be resolved later.
///
/// The primary API for AsyncValues is provided by AnyAsyncValueRef and
/// AsyncValueRef<T> smart pointer classes. The public methods in this class
/// are intended for advanced users who which to explicitly manage reference
/// counting.
class AsyncValue {
public:
  //===--------------------------------------------------------------------===//
  // Static creation methods for AsyncValue's
  //===--------------------------------------------------------------------===//

  /// Create an AsyncValue for the specified type in an "unconstructed" state.
  /// This should be `emplace`'d, `construct`'d, or finalized with an error.
  /// The result will have ref count 1.
  template <typename T>
  static AsyncValue *allocate(CompactCPUDevicePtr cpuDevice);

  /// Create an AsyncValue for the specified type in "available" and ready
  /// state. This is a terminal state for an AsyncValue, it can never change out
  /// of this state. The result will have ref count 1.
  template <typename T, typename... Args>
  static AsyncValue *createReady(CompactCPUDevicePtr cpuDevice, Args &&...args);

  /// Create an IndirectAsyncValue that may be filled in with any AsyncValue in
  /// the future. The result will have ref count 1.
  static AsyncValue *createIndirect(CompactCPUDevicePtr cpuDevice);

  /// Create an AsyncValue that has already been turned into an error with the
  /// specified message. The result will have ref count 1.
  static AsyncValue *createError(CompactCPUDevicePtr cpuDevice,
                                 EncodedDiagnostic diagnostic);

  //===--------------------------------------------------------------------===//
  // State change methods.
  //===--------------------------------------------------------------------===//

  /// Constructs the payload of the AsyncValue in place, and changes its state
  /// to kAvailable. Requires that the AsyncValue is a ConcreteAsyncValue in
  /// an unconstructed state. All pending waiters will be notified the value
  /// is ready.
  ///
  /// Waiters are never called directly by this method.
  ///
  /// One ref count will be removed from the AsyncValue just before any
  /// existing waiters are triggered. It is valid for the AsyncValue to have
  /// only a single remaining reference, and thus the waiters may be triggered
  /// after the AsyncValue is deleted.
  template <typename T, typename... Args>
  void emplaceAndDecRef(Args &&...args);

  /// Sets the AsyncValue to the kError state. The AsyncValue may be a
  /// ConcreteAsyncValue or IndirectAsyncValue in an unconstructed state.
  /// All pending waiters will be notified the value is ready.
  ///
  /// Waiters are never called directly by this method.
  ///
  /// One ref count will be removed from the AsyncValue just before any
  /// existing waiters are triggered. It is valid for the AsyncValue to have
  /// only a single remaining reference, and thus the waiters may be triggered
  /// after the AsyncValue is deleted.
  void setToErrorAndDecRef(EncodedDiagnostic diagnostic);

  /// Resolve an IndirectAsyncValue to point to the specified new value.
  /// All pending waiters on this value will be notified when the new value
  /// becomes ready.
  ///
  /// Waiters are never called directly by this method.
  ///
  /// One ref count will be removed from the AsyncValue just before any
  /// existing waiters are triggered. It is valid for the AsyncValue to have
  /// only a single remaining reference, and thus the waiters may be triggered
  /// after the AsyncValue is deleted.
  void resolveIndirectAndDecRef(RCRef<AsyncValue> &&newValue);

  /// Resolves an IndirectAsyncValue to contain a concrete AsyncValue with a
  /// newly initialized value. All pending waiters will be notified the value
  /// is ready.
  ///
  /// Waiters are never called directly by this method.
  ///
  /// One ref count will be removed from the AsyncValue just before any
  /// existing waiters are triggered. It is valid for the AsyncValue to have
  /// only a single remaining reference, and thus the waiters may be triggered
  /// after the AsyncValue is deleted.
  template <typename T, typename... Args>
  void emplaceIndirectAndDecRef(Args &&...args);

  //===--------------------------------------------------------------------===//
  // Primary interface to AsyncValue for clients to use.
  //===--------------------------------------------------------------------===//

  /// Return the `Runtime` instance this is part of.
  CompactCPUDevicePtr getCPUDevice() const { return cpuDevice; }

  /// AsyncValue maintains a list of waiters that are waiting for notification
  /// that this value transitioned to Available or Error.
  using Waiter = llvm::unique_function<void()>;

  /// Register that waiter should be run when this AsyncValue becomes ready
  /// (either with an emplaced value, or with an error).
  ///
  /// It is possible for this AsyncValue to have been deleted by the time
  /// the waiter is executed. Prefer the 'ConsumingWaiter' versions of andThen
  /// if the waiter needs access to this AsyncValue.
  ///
  /// If `IsAsync` is true, the waiter will be scheduled as an independent
  /// task, using the work queue for this object's cpuDevice. Otherwise, the
  /// waiter will generally be run on the 'triggering' thread, ie the thread
  /// which caused this AsyncValue to become ready. However, if the triggering
  /// thread is a 'foreign' thread not in an await loop then the waiters will
  /// be run as if `IsAsync` was true.
  template <bool IsAsync>
  void andThen(Waiter &&waiter);

  void andThenSync(Waiter &&waiter) {
    andThen</*IsAsync=*/false>(std::forward<Waiter>(waiter));
  }

  void andThenAsync(Waiter &&waiter) {
    andThen</*IsAsync=*/true>(std::forward<Waiter>(waiter));
  }

  /// Return the stored value as type T.
  ///
  /// This requires that the AsyncValue is either constructed or is a fully
  /// concrete value, and that T be the exact type (or a base type) of the
  /// actual payload type. When T is a base type of the payload type, the
  /// following additional conditions are required:
  ///
  /// 1) Both the payload type and T are polymorphic (have virtual function)
  ///    or neither are.
  /// 2) The payload type does not use multiple inheritance.
  ///
  /// The above conditions are required since we store the value at a fixed
  /// offset from the start of AsyncValue. Violation of either 1) or 2) requires
  /// additional pointer adjustments to get the proper pointer for the base
  /// type, which we do not have sufficient information to perform at runtime.
  template <typename T>
  const T &get() const;

  // Same as the const overload of get(), for mutable use-cases.
  template <typename T>
  T &get() {
    return const_cast<T &>(static_cast<const AsyncValue *>(this)->get<T>());
  }

  void *getUnderlyingPtr() const;

  void *getUnderlyingPtr() {
    return static_cast<const AsyncValue *>(this)->getUnderlyingPtr();
  }

  /// Return true if this AsyncValue is "Ready" and filled with a concrete
  /// value.   get() will return a value in this state.
  bool isValueAvailable() const { return getState() == State::kAvailable; }

  /// Return true if the AsyncValue is "Ready" and either filled with a concrete
  /// value or an error.
  bool isReady() const { return isReady(getState()); }

  /// Return true if the AsyncValue is fulfilled with an error state.
  bool isError() const { return getState() == State::kError; }

  /// Return the Diagnostic in this AsyncValue, aborting if it isn't an error.
  const EncodedDiagnostic &getDiagnostic() const {
    auto *result = getDiagnosticIfPresent();
    assert(result && "AsyncValue doesn't hold an error");
    return *result;
  }

  /// Return the Diagnostic in this AsyncValue, aborting if it isn't an error.
  EncodedDiagnostic takeDiagnostic() {
    auto *result = getDiagnosticIfPresent();
    assert(result && "AsyncValue doesn't hold an error");
    return std::move(*result);
  }

  /// If this AsyncValue holds an error, return its diagnostic.  If not, return
  /// nullptr.
  const EncodedDiagnostic *getDiagnosticIfPresent() const {
    return const_cast<AsyncValue *>(this)->getDiagnosticIfPresent();
  }

  /// If this AsyncValue holds an error, return its diagnostic.  If not, return
  /// nullptr.
  EncodedDiagnostic *getDiagnosticIfPresent();

  //===--------------------------------------------------------------------===//
  // Type Related functionality
  //===--------------------------------------------------------------------===//

  /// Return a type identifier for the payload held by this AsyncValue.  This is
  /// not set for IndirectAsyncValue's until they are resolved to a value.
  TypeID getTypeID() const { return typeID; }

  /// Returns true if this AsyncValue is a ConcreteAsyncValue for type T, or
  /// an IndirectAsyncValue which has been resolved to a ConcreteAsyncValue
  /// of type T.
  template <typename T>
  bool isType() const {
    return typeID == TypeID::get<T>();
  }

  /// Return true if this AsyncValue is a unresolved IndirectAsyncValue.
  bool isIndirect() const {
    return getSubclassKind() == SubclassKind::kIndirect &&
           getTypeID() == TypeID();
  }

  //===--------------------------------------------------------------------===//
  // Low Level Interfaces
  //===--------------------------------------------------------------------===//

  /// This enum indicates whether the AsyncValue was created as a
  /// ConcreteAsyncValue or IndirectAsyncValue.  It is never mutable.
  enum class SubclassKind : uint8_t {
    kConcrete = 0, // ConcreteAsyncValue
    kIndirect = 1, // IndirectAsyncValue
  };

  SubclassKind getSubclassKind() const { return subclassKind; }

  // The state of AsyncValue.  This is mutable as the value evolves.
  // It must fit within 3 bits.
  enum class State : uint8_t {
    /// Initial state.
    /// The payload's constructor has not been invoked so the value is not ready
    /// for consumption. This state can transition to
    /// `kUnconstructedInitializingInlineWaiter`, `kAvailable` and `kError`.
    kUnconstructed = 0,

    /// This is a transient state when the first waiter is added to a
    /// `kUnconstructed` AsyncValue.  It is used for a few cycles when the
    /// inline waiter is being initialized.  Any state-aware internal
    /// implementation details of AsyncValue that encounters this should just
    /// spin until the state changes to `kUnconstructed4ValidOOLWaiterSlots`
    /// (ie +4).
    kUnconstructedInitializingInlineWaiter = 1,

    /// These states are used to keep track of how many entries are used in the
    /// first out-of-line WaiterListNode, if any, which is pointed to by
    /// `waitersAndState`.  Encoding this information in the state integer in
    /// `waitersAndState` allows us to use compare/xchg to atomically allocate
    /// entries in the out of line waiter.  The enum values here are encoded
    /// this way to make certain operations against the state enum value more
    /// efficient.
    ///
    /// For the `kUnconstructed{1,2,3}ValidOOLWaiterSlots` states we know we
    /// have both an inline waiter in the ConcreteAsyncValue or
    /// IndirectAsyncValue subclass, and a WaiterListNode pointed to by the
    /// waitersAndState pointer with that number of valid waiters.
    ///
    /// For the `kUnconstructed4ValidOOLWaiterSlots` state however there are
    /// three possible sub-states:
    ///  - We are in the process of emplacing a value which may or may not
    ///    already have an inline waiter. This is used for interlock of
    ///    emplacing and andThen'ing.
    ///  - We have established the first inline waiter, but the
    ///    `waitersAndState` pointer is null. This is the just-one-waiter state.
    ///  - The `waitersAndState` points to a WaiterListNode which indeed has
    ///    all four entries set to waiters.
    kUnconstructed1ValidOOLWaiterSlots = 2, //< 1 valid, 3 free slots.
    kUnconstructed2ValidOOLWaiterSlots = 3, //< 2 valid, 2 free slots.
    kUnconstructed3ValidOOLWaiterSlots = 4, //< 3 valid, 1 free slots.
    kUnconstructed4ValidOOLWaiterSlots = 5, //< 4 valid, 0 free slots.

    /// Terminal state.
    /// The underlying value is constructed and ready for consumption by
    /// waiters and contains an initialized value. This state can not transition
    /// to any other state.
    kAvailable = 6,

    /// Terminal state.
    /// This AsyncValue is ready and contains an error, along with an
    /// uninitialized value. This state can not transition to any other
    /// state.
    kError = 7,
  };

  /// Return the current state of this AsyncValue.
  State getState() const { return loadWaitersAndState().getInt(); }

  /// Return true if the specified AsyncValue state is ready, which means the
  /// waiters have all been notified.
  static bool isReady(State state) {
    return state == State::kAvailable || state == State::kError;
  }

  /// Return true if this is a state with a finalized inline waiter.
  static bool hasInlineWaiter(State state) {
    return state >= State::kUnconstructed1ValidOOLWaiterSlots &&
           state <= State::kUnconstructed4ValidOOLWaiterSlots;
  }

  /// Returns true if the caller holds the only reference which could change
  /// the AsyncValue.
  ///  - For concrete AsyncValues, the ref count must be 1.
  ///  - For *completed* indirect AsyncValues, both this and the target
  ///    AsyncValue must have a ref count of 1.
  ///
  /// We don't allow yet-to-be completed indirect AsyncValues to be tested
  /// for uniqueness since it is racy w.r.t. completion.
  bool isUnique() const {
    if (refcount.load() > 1)
      return false;
    if (getSubclassKind() == SubclassKind::kIndirect)
      return isUniqueSlow();
    return true;
  }

  /// Returns the current reference count.
  size_t getRefCountForDebugging() const { return refcount.load(); }

  /// Return true if we tracking of live AsyncValue instances is enabled.
  static constexpr bool isAllocationTrackingEnabled() {
#ifdef MODULAR_DEBUG
    return true;
#else  // MODULAR_DEBUG
    // Only track the number of alive AsyncValue instances in debug builds.
    return false;
#endif // MODULAR_DEBUG
  }

  /// Return the total number of async values that are currently live in the
  /// process. This is intended for debugging/assertions only, and shouldn't be
  /// used for mainline logic in the cpuDevice.
  static ssize_t getNumAllocatedInstances() {
    assert(isAllocationTrackingEnabled() &&
           "AsyncValue instance tracking disabled!");
    return Globals::totalAllocatedAsyncValues.load(std::memory_order_relaxed);
  }

  /// Print an internal representation of the AsyncValue. For debugging only.
  /// CAUTION: Not thread safe! The printed state may appear torn.
  void printDebug(raw_ostream &os) const;

private:
  // Reference counting, only accessible to RCRef<>.
  template <typename T>
  friend class M::RCRef;

  // Most of the API is accessible via AnyAsyncValueRef and its subclass
  // AsyncValueRef<T>.
  friend class AnyAsyncValueRef;

  /// Increase the reference count.
  void addRef();
  void addRef(uint32_t count);

  /// Decrease the reference count of this object, potentially deallocating it.
  void dropRef(uint32_t count = 1);

  /// Slow path for isUnique for IndirectAsyncValues.
  bool isUniqueSlow() const;

  //===--------------------------------------------------------------------===//
  // State held by an AsyncValue
  //===--------------------------------------------------------------------===//

  /// This is the number of individual users of the AsyncValue, when it drops
  /// to zero, the AsyncValue is deallocated.
  std::atomic<uint32_t> refcount{1};

  /// This is a compact (8-bit) pointer to the enclosing Runtime instance.
  const CompactCPUDevicePtr cpuDevice;

  /// Whether this is an indirect or concrete AsyncValue.
  const SubclassKind subclassKind : 1;

  /// hasVTable has the same value for a given payload type T.
  const bool hasVTable : 1;

  // NOTE: 6 unused padding bits.

  /// This is a 16-bit value that identifies the type. This is dynamically set
  /// for IndirectAsyncValue's when they get resolved.
  TypeID typeID;

protected:
  struct WaiterListNodePointerTraits {
    static inline void *getAsVoidPointer(WaiterListNode *ptr) { return ptr; }
    static inline WaiterListNode *getFromVoidPointer(void *ptr) {
      return static_cast<WaiterListNode *>(ptr);
    }
    enum { NumLowBitsAvailable = 3 };
  };

  /// The waiter list and the state are compacted into a single atomic word,
  /// since the fields need to be accessed at the same time for state changes.
  ///
  /// Invariant: If the state is ready, then the waiter list must be nullptr.
  using WaitersAndState = llvm::PointerIntPair<WaiterListNode *, 3, State,
                                               WaiterListNodePointerTraits>;

  WaitersAndState loadWaitersAndState() const {
    return WaitersAndState::getFromOpaqueValue(
        (void *)waitersAndState.load(std::memory_order_acquire));
  }

  /// Compare the current value of waiterAndState with the specified `oldValue`.
  /// If equal, replace it with `newValue` and return true (`oldValue` will also
  /// be updated to match).  If the value changed underneath us, return false
  /// and update `oldValue` to what is currently in memory.
  bool compareExchangeWaiterAndState(WaitersAndState &oldValue,
                                     WaitersAndState newValue) {
    intptr_t oldValueInt = (intptr_t)oldValue.getOpaqueValue();
    intptr_t newValueInt = (intptr_t)newValue.getOpaqueValue();
    bool result = waitersAndState.compare_exchange_weak(
        oldValueInt, newValueInt, std::memory_order_acq_rel,
        std::memory_order_acquire);
    oldValue = WaitersAndState::getFromOpaqueValue((void *)oldValueInt);
    return result;
  }

  /// Replace the current waiterAndState value with the new value and return the
  /// old one.
  WaitersAndState exchangeWaiterAndState(WaitersAndState newValue) {
    auto result = waitersAndState.exchange((intptr_t)newValue.getOpaqueValue());
    return WaitersAndState::getFromOpaqueValue((void *)result);
  }

private:
  std::atomic<intptr_t> waitersAndState;

protected:
  LogicalResult moveState(WaitersAndState &oldValue, State newState);
  static void runWaitersAndDeallocate(WaiterListNode *list,
                                      size_t numEntriesValid,
                                      WorkQueue *workQueue);
  void andThenOutOfLine(Waiter waiter, WaitersAndState oldValue);
  void andThenAsyncOutOfLine(Waiter waiter);
  void destroyWithRefCountZero();

  /// Transitions the AsyncValue to a ready state, and notifies all waiters.
  /// Returns the original state. One ref count to the AsyncValue will be
  /// removed just before the first waiter is triggered.
  State notifyReadyAndDecRef(State newState,
                             std::optional<Waiter> &&extraWaiter);

  void removeAnyInlineWaiter(std::optional<Waiter> &inlineWaiter);
  Waiter *getInlineWaiterPointer();

  /// Invoke a single waiter immediately.
  ///
  /// This method is used when an 'andThen' is called on a ready AsyncValue.
  /// Since the 'andThen' waiter closure is constructed at the 'andThen'
  /// call site, it seems reasonable to execute it on the callers stack.
  static void runWaiterNow(Waiter &&waiter) { waiter(); }

  /// Schedule a single waiter to be invoked later, but on the current thread.
  ///
  /// This method is used when an 'emplace' has triggered waiters. Since the
  /// each waiter's closure is arbitrary and remote from the emplace call,
  /// it seems prudent to avoid executing the waiter on the callers stack.
  ///
  /// However the waiter may need to be run immediately and on the callers
  /// stack if there's no place to enqueue the waiter onto. For example,
  /// a 'foreign' thread which is not currently within an await run loop may
  /// emplace an async value.
  static void runWaiterLater(Waiter &&waiter, WorkQueue *workQueue) {
    workQueue->addLocalTask(std::move(waiter));
  }

protected:
  /// This layout of this class is designed very carefully to ensure alignment
  /// of the payload to 16 bytes and we don't want to change this.  That said,
  /// we do put the 16 bytes to work (including metadata about the concrete
  /// type of the value, whether vtables exist or not, etc) in order to detect
  /// common programmer mistakes quickly.
  static constexpr int kAsyncValueSize = 16;

  AsyncValue(SubclassKind subclassKind, State state, bool hasVTable,
             TypeID typeID, CompactCPUDevicePtr cpuDevice)
      : cpuDevice(cpuDevice), subclassKind(subclassKind), hasVTable(hasVTable),
        typeID(typeID),
        waitersAndState(
            (intptr_t)WaitersAndState(nullptr, state).getOpaqueValue()) {
    assert((subclassKind == SubclassKind::kIndirect || typeID != TypeID()) &&
           "require valid type ID when constructing a ConcreteAsyncValue");
    if constexpr (isAllocationTrackingEnabled())
      ++Globals::totalAllocatedAsyncValues;
  }

  ~AsyncValue() {
    if constexpr (isAllocationTrackingEnabled())
      --Globals::totalAllocatedAsyncValues;
  }

private:
  AsyncValue(const AsyncValue &) = delete;
  void operator=(const AsyncValue &) = delete;
};

//===----------------------------------------------------------------------===//
// ConcreteAsyncValue implementation.
//===----------------------------------------------------------------------===//

namespace Detail {
// Subclass for storing the payload of the AsyncValue inline.  This should
/// never be directly accessed by users - always use AsyncValue methods instead.
class SomeConcreteAsyncValue : public AsyncValue {
  friend class AsyncValue;
  template <typename T>
  friend class ConcreteAsyncValue;
  using AsyncValue::AsyncValue;

private:
  // Only invoked by destroyWithRefCountZero.
  ~SomeConcreteAsyncValue();

  /// The error value is always first thing in our derived class.
  EncodedDiagnostic *getDiagnosticPointer() {
    return reinterpret_cast<EncodedDiagnostic *>(this + 1);
  }

  /// The waiter value is always the first thing in our derived class.
  Waiter *getInlineWaiterPointer() {
    return reinterpret_cast<Waiter *>(this + 1);
  }

  /// Return the address of the (potentially uninitialized) payload.
  void *getPayloadPointer() {
    /// The payload in a ConcreteAsyncValue always immediately follows the
    /// AsyncValue.  This is guaranteed by static_asserts in ConcreteAsyncValue
    /// below.
    return this + 1;
  }
};

/// Subclass for storing the payload of the AsyncValue inline.  This should
/// never be directly accessed by users - always use AsyncValue methods instead.
template <typename T>
class ConcreteAsyncValue : public SomeConcreteAsyncValue {
  friend class AsyncValue;
  friend class SomeConcreteAsyncValue;
  /// Allocate an instance of ConcreteAsyncValue in the specified state, but
  /// with the payload uninitialized. The result will have ref count 1.
  static ConcreteAsyncValue<T> *allocate(State state,
                                         CompactCPUDevicePtr cpuDevice) {
    auto *ptr = (ConcreteAsyncValue<T> *)alignedAlloc(
        alignof(ConcreteAsyncValue<T>), sizeof(ConcreteAsyncValue<T>));
    new (ptr) ConcreteAsyncValue<T>(state, std::is_polymorphic_v<T>,
                                    TypeID::get<T>(), cpuDevice);
    return ptr;
  }

private:
  ConcreteAsyncValue(State state, bool hasVTable, TypeID typeID,
                     CompactCPUDevicePtr cpuDevice)
      : SomeConcreteAsyncValue(SubclassKind::kConcrete, state, hasVTable,
                               typeID, cpuDevice) {
// This static assert verifies that no padding is inserted between the
// AsyncValue in the base class and the union containing T below.
// Since ConcreteAsyncValue<T> is not a standard-layout class, the use of
// offsetof() is "conditionally-supported" rather than fully portable.
// Both GCC and Clang raise a warning here, but offsetof() still works and
// gives us the result we expect, so we accept that this use is justified and
// suppress the warning.
//
// See also:
// https://en.cppreference.com/w/cpp/types/offsetof
// https://en.cppreference.com/w/cpp/language/classes#Standard-layout_class
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#endif
    static_assert(offsetof(ConcreteAsyncValue<T>, payload) ==
                      AsyncValue::kAsyncValueSize,
                  "Offset of ConcreteAsyncValue::payload needs to be aligned");
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif
  }

  // NOTE: destruction of this state is handled by ~SomeConcreteAsyncValue.
  union {
    /// When in a `kError` state, this includes location and diagnostic
    /// information.  Both this field and Waiter are 3 words.
    EncodedDiagnostic diagnostic;
    /// When unconstructed, this can hold an inline copy of the first waiter,
    /// avoiding having to heap allocate a waiter node for it.  The state will
    /// be considered "unconstructed" but not `kUnconstructed`.
    Waiter waiter;
    /// When in the `kAvailable` state, this is the payload of the AsyncValue.
    T payload;
  };

  // AsyncValue needs to be 16 bytes in size to assure that payloads in
  // ConcreteAsyncValue are 16-byte aligned.  This simplifies all clients.
  static_assert(sizeof(AsyncValue) == kAsyncValueSize,
                "Unexpected size for AsyncValue");
};

} // namespace Detail.

namespace Detail {
/// IndirectAsyncValue represents an uncomputed AsyncValue of unspecified type.
/// This is used when an AsyncValue must be returned, but the value it holds is
/// not ready and the producer of the value doesn't know what type it will
/// ultimately be, or whether it will be an error.
class IndirectAsyncValue : public AsyncValue {
  friend class AsyncValue;
  IndirectAsyncValue(CompactCPUDevicePtr cpuDevice)
      : AsyncValue(SubclassKind::kIndirect, State::kUnconstructed,
                   /*hasVTable=*/false,
                   /*typeID=*/{}, cpuDevice) {}
  ~IndirectAsyncValue() {
    // If the IndirectAsyncValue is ready, then the RCRef to the resolved
    // pointer has been constructed, destroy it.
    if (isReady(getState())) {
      value.reset();
    } else {
      // Otherwise, we must be in a plain unconstructed case with no waiters.
      // This can happen when an IndirectAsyncValue is created and immediately
      // destroyed.
      // NOTE: It isn't currently valid to destroy an IndirectAsyncValue with
      // waiters - we'd have to propagate error into them, and would have to
      // check for resurrection.  Don't do this until we have a known use-case.
      assert(getState() == State::kUnconstructed &&
             "destroying an IndirectAsyncValue with waiters that never got "
             "resolved?");
    }
  }

  union {
    // This field is present when in kUnconstructedInlineWaiter* state.
    // TODO: kUnconstructedInlineWaiter* does not exist any more
    Waiter waiter;
    // This field is present when resolved to another AsyncValue.
    RCRef<AsyncValue> value;
  };
};
} // namespace Detail

//===----------------------------------------------------------------------===//
// AsyncValue inline method implementations.
//===----------------------------------------------------------------------===//

/// Create an AsyncValue for the specified type in "unconstructed" state.
template <typename T>
inline AsyncValue *AsyncValue::allocate(CompactCPUDevicePtr cpuDevice) {
  assert(cpuDevice && "AsyncValue::allocate requires valid cpuDevice");
  return Detail::ConcreteAsyncValue<T>::allocate(State::kUnconstructed,
                                                 cpuDevice);
}

/// Create an AsyncValue for the specified type in "available" and ready state.
/// This is a terminal state for an AsyncValue, it can never change out of this
/// state.
template <typename T, typename... Args>
inline AsyncValue *AsyncValue::createReady(CompactCPUDevicePtr cpuDevice,
                                           Args &&...args) {
  auto *result =
      Detail::ConcreteAsyncValue<T>::allocate(State::kAvailable, cpuDevice);
  new (&result->payload) T(std::forward<Args>(args)...);
  return result;
}

inline void AsyncValue::addRef() {
  assert(refcount.load() > 0);
  ++refcount;
}

inline void AsyncValue::addRef(uint32_t count) {
  if (count > 0) {
    assert(refcount.load() > 0);
    // Increasing the reference counter can always be done with
    // memory_order_relaxed: New references to an object can only be formed
    // from an existing reference, and passing an existing reference from one
    // thread to another must already provide any required synchronization.
    refcount.fetch_add(count, std::memory_order_relaxed);
  }
}

inline void AsyncValue::dropRef(uint32_t count) {
  assert(refcount.load() > 0);
  // We expect that `count` argument will often equal the actual reference count
  // here; optimize for that.  If `count` == reference count, only an acquire
  // barrier is needed to prevent the effects of the deletion from leaking
  // before this point.
  //
  // TODO: Measure and evaluate whether this is a useful optimization on all
  // systems.  On X86 systems for example, this is probably not actually a win.
  bool isLastRef = refcount.load(std::memory_order_acquire) == count;
  if (!isLastRef) {
    // If `count` != reference count, a release barrier is needed in
    // addition to an acquire barrier so that prior changes by this thread
    // cannot be seen to occur after this decrement.
    isLastRef = refcount.fetch_sub(count, std::memory_order_acq_rel) == count;
  }

  // Destroy this value if the refcount drops to zero.
  if (isLastRef)
    destroyWithRefCountZero();
}

template <bool IsAsync>
void AsyncValue::andThen(Waiter &&waiter) {
  if constexpr (IsAsync)
    andThenAsyncOutOfLine(std::forward<Waiter>(waiter));
  else {
    // Clients generally want to use andThenSync without them each having to
    // check to see if the value is present. Check for them, and immediately
    // run the lambda if it is already here.
    auto waitersAndStateValue = loadWaitersAndState();
    if (isReady(waitersAndStateValue.getInt())) {
      assert(waitersAndStateValue.getPointer() == nullptr &&
             "cannot have waiter nodes when ready");
      runWaiterNow(std::move(waiter));
      return;
    }

    andThenOutOfLine(std::move(waiter), waitersAndStateValue);
  }
}

template <typename T, typename... Args>
inline void AsyncValue::emplaceAndDecRef(Args &&...args) {
  assert(getSubclassKind() == SubclassKind::kConcrete &&
         "Cannot 'emplaceAndDecRef' an IndirectValue, use "
         "'emplaceIndirectAndDecRef' instead");
  typeID.assertEqual(TypeID::get<T>(), "AsyncValue::emplaceAndDecRef");

  // NOTE: At this point we could stop tracking the ref and instead just inc/dec
  // our own ref count. However the explicit ref passing makes the chain of
  // ownership more explicit.

  // Take any inline waiters out of the payload area so we can construct it.
  std::optional<Waiter> inlineWaiter;
  removeAnyInlineWaiter(inlineWaiter);

  // Initialize the payload.
  auto *concrete = static_cast<Detail::ConcreteAsyncValue<T> *>(this);
  new (&concrete->payload) T(std::forward<Args>(args)...);

  // Change state and notify the waiters.
  auto oldState =
      notifyReadyAndDecRef(State::kAvailable, std::move(inlineWaiter));

  // ---------- only static methods from here on ----------

  // This must have been in one of the unconstructed states, but couldn't have
  // been in kUnconstructed because that would allow a race for another inline
  // waiter to be added. `removeAnyInlineWaiter` ensures this isn't possible.
  assert(hasInlineWaiter(oldState) &&
         "AsyncValue transitioned to while we're emplacing?");
  (void)oldState;
}

/// Construct the payload of the AsyncValue in place and change its state to
/// kConcrete. Requires that this is a ConcreteAsyncValue that have state
/// `kUnconstructed`.
template <typename T, typename... Args>
inline void AsyncValue::emplaceIndirectAndDecRef(Args &&...args) {
  assert(getSubclassKind() == SubclassKind::kIndirect);
  resolveIndirectAndDecRef(takeRCRef(
      createReady<T, Args...>(getCPUDevice(), std::forward<Args>(args)...)));
}

template <typename T>
const T &AsyncValue::get() const {
  assert(getState() == State::kAvailable &&
         "Cannot call get() when AsyncValue isn't available");
  if (getSubclassKind() == SubclassKind::kConcrete) {
    auto *thisConcrete =
        static_cast<const Detail::ConcreteAsyncValue<T> *>(this);
    typeID.assertEqual(TypeID::get<T>(), "AsyncValue::get");
    // Make sure T and the stored type agree on whether they have a vtable.
    // (Not strictly necessary given exact equality test above, but retaining
    // in case we allow subtyping relation between T and the true held type.)
    assert(std::is_polymorphic_v<T> == hasVTable &&
           "mismatched static and dynamic AsyncValue type polymorphism");
    return thisConcrete->payload;
  }

  auto *thisIndirect = static_cast<const Detail::IndirectAsyncValue *>(this);
  assert(thisIndirect->value &&
         "indirect can't be constructed without being resolved");
  return thisIndirect->value->get<T>();
}

inline void *AsyncValue::getUnderlyingPtr() const {
  if (getSubclassKind() == SubclassKind::kConcrete) {
    auto *thisConcrete =
        static_cast<const Detail::ConcreteAsyncValue<size_t> *>(this);
    return reinterpret_cast<void *>(
        const_cast<size_t *>(&thisConcrete->payload));
  }

  auto *thisIndirect = static_cast<const Detail::IndirectAsyncValue *>(this);
  return thisIndirect->value->getUnderlyingPtr();
}

inline AsyncValue::Waiter *AsyncValue::getInlineWaiterPointer() {
  if (getSubclassKind() == SubclassKind::kConcrete)
    return static_cast<Detail::SomeConcreteAsyncValue *>(this)
        ->getInlineWaiterPointer();
  return &static_cast<Detail::IndirectAsyncValue *>(this)->waiter;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const M::AsyncRT::AsyncValue &value) {
  value.printDebug(os);
  return os;
}

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_ASYNCVALUE_H
