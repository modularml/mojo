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
// This file declares global functions that help implement parallel algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_ALGORITHMS_H
#define ASYNCRT_RUNTIME_ALGORITHMS_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Support/Chain.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include <utility>

namespace M::AsyncRT {

/// Profiling entry for sub-tasks launched by the parallelization helpers.
using AlgorithmProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kAsyncRT, 2), Trace::kAsyncRT>;

//===----------------------------------------------------------------------===//
// Helpers that wait for values.
//===----------------------------------------------------------------------===//

/// Returns only when all of the given async values are available.
inline static void await(ArrayRef<AnyAsyncValueRef> values) {
  if (!values.empty())
    values[0].getCPUDevice()->getWorkQueue()->await(values);
}

inline static void await(const AnyAsyncValueRef &value) {
  await(ArrayRef<AnyAsyncValueRef>(value));
}

/// Error propagating variant of await functions.
inline ErrorOrSuccess awaitOrError(MutableArrayRef<AnyAsyncValueRef> values) {
  AsyncRT::await(values);

  for (AnyAsyncValueRef &value : values)
    if (value.isError())
      return value.getDiagnostic().getMessage().copy();

  return success();
}

template <typename T>
inline static ErrorOrSuccess awaitOrError(const AsyncValueRef<T> &value) {
  await(ArrayRef<AnyAsyncValueRef>(value));

  if (value.isError())
    return value.getDiagnostic().getMessage().copy();

  return success();
}

//===----------------------------------------------------------------------===//
// 'andThenSync/Async' for multiple values with heterogenous types.
//===----------------------------------------------------------------------===//

namespace Detail {
template <bool IsAsync, typename CompletionFn, typename... ValueTys>
inline static void andThenImpl(std::tuple<ValueTys...> &&values,
                               CompletionFn &&completionFn) {
  struct AndThenState {
    /// This is the number of values we're waiting on.  When this drops to zero,
    /// the completion handler is run.
    std::atomic<size_t> numElementsLeft;

    /// This is the function to execute once all elements are done.
    CompletionFn completionFn;

    /// These are the async values we're waiting on.  They are passed into the
    /// completion function once they all become ready.
    std::tuple<ValueTys...> values;
  };

  // Allocate the parallel state on the heap since it will out-live the call to
  // this function.  We will deallocate it after invoking the completion
  // handler when the last element completes.
  auto state = new AndThenState{sizeof...(ValueTys),
                                std::forward<CompletionFn>(completionFn),
                                std::forward<decltype(values)>(values)};

  // This function is invoked on every async value to wait for it to complete.
  auto processAsyncValue = [state](AsyncValue *value) {
    WorkQueue *asyncWorkQueue = nullptr;
    if constexpr (IsAsync)
      asyncWorkQueue = value->getCPUDevice()->getWorkQueue();
    value->andThenSync([state, asyncWorkQueue]() {
      // Once that is done we can decrement the count and trigger completion
      // when the last element is done.
      if (--state->numElementsLeft != 0)
        return;

      auto handler = [state]() {
        // Invoke the completion function, since we're done.
        std::apply(state->completionFn, std::move(state->values));

        // All uses of the state are done, so we can deallocate it.
        delete state;
      };

      if (asyncWorkQueue)
        asyncWorkQueue->addTask(handler);
      else
        handler();
    });
  };

  // Invoke `processAsyncValue` on each value in state
  std::apply([&](auto &...elt) { (processAsyncValue(elt.getPointer()), ...); },
             state->values);
}
} // namespace Detail

/// This version of andThen takes a tuple of values to wait on, and passes the
/// elements into the completion handler as individual values.  It can be used
/// like this:
///
/// void example(AsyncValueRef<int32_t> lhs, AsyncValueRef<int32_t> rhs) {
///   ...
///   andThenSync(std::make_tuple(std::move(lhs), std::move(rhs)),
///           [... any captures...](AsyncValueRef<int32_t> lhs,
///                                 AsyncValueRef<int32_t> rhs) {
///     ... stuff that uses lhs/rhs ...
///   });
/// }
///
/// The "Sync" version runs the completion handler as soon as the last value is
/// fulfilled by the thread that fulfills the value. The "Async" version adds
/// the completion handler to the work queue when all the async values are
/// fulfilled.

template <typename CompletionFn, typename... ValueTys>
inline static void andThenSync(std::tuple<ValueTys...> &&values,
                               CompletionFn &&completionFn) {
  Detail::andThenImpl</*IsAsync=*/false>(
      std::forward<decltype(values)>(values),
      std::forward<CompletionFn>(completionFn));
}

template <typename CompletionFn, typename... ValueTys>
inline static void andThenAsync(std::tuple<ValueTys...> &&values,
                                CompletionFn &&completionFn) {
  Detail::andThenImpl</*IsAsync=*/true>(
      std::forward<decltype(values)>(values),
      std::forward<CompletionFn>(completionFn));
}

//===----------------------------------------------------------------------===//
// 'andThenSync/Async' for an array of values
//===----------------------------------------------------------------------===//

namespace Detail {
template <bool IsAsync, typename ArrayRefType, typename CompletionFn,
          typename CopyOrMoveFn>
inline static void andThenArrayImpl(ArrayRefType values,
                                    CompletionFn &&completionFn,
                                    CopyOrMoveFn &&copyOrMoveFn) {
  // Avoid malloc overhead for trivial cases.
  if (values.empty()) {
    completionFn(values);
    return;
  }

  if (values.size() == 1) {
    copyOrMoveFn(values[0]).template andThen<IsAsync>(
        [completionFn = std::forward<CompletionFn>(completionFn)](
            AnyAsyncValueRef &&value) mutable {
          AnyAsyncValueRef mutableValue = value.copy();
          completionFn(mutableValue);
        });
    return;
  }

  struct AndThenState {
    /// This is the number of values we're waiting on.  When this drops to zero,
    /// the completion handler is run.
    std::atomic<size_t> numElementsLeft;

    /// This is the function to execute once all elements are done.
    CompletionFn completionFn;

    /// These are the async values we're waiting on.  They are passed into the
    /// completion function once they all become ready.
    llvm::SmallVector<AnyAsyncValueRef> values;
  };

  // Allocate the parallel state on the heap since it will out-live the call to
  // this function.  We will deallocate it after invoking the completion
  // handler when the last element completes.
  auto state = new AndThenState{
      values.size(), std::forward<CompletionFn>(completionFn), {}};

  state->values.reserve(values.size());

  // For each value, wait for completion and then run the completion function on
  // the last one.
  WorkQueue *asyncWorkQueue = nullptr;
  if constexpr (IsAsync)
    asyncWorkQueue = values[0].getCPUDevice()->getWorkQueue();
  for (auto &v : values) {
    state->values.push_back(copyOrMoveFn(v));
    state->values.back().andThenSync([state, asyncWorkQueue]() {
      // Once that is done we can decrement the count and trigger completion
      // when the last element is done.
      if (--state->numElementsLeft != 0)
        return;

      auto handler = [state]() {
        // Invoke the completion function, since we're done.
        state->completionFn(state->values);

        // All uses of the state are done, so we can deallocate it.
        delete state;
      };

      if constexpr (IsAsync)
        asyncWorkQueue->addTask(handler);
      else {
        (void)(asyncWorkQueue); // To suppress "-Wunused-lambda-capture"
        handler();
      }
    });
  }
}
} // namespace Detail

/// This version of andThen takes an array of homogenous AsyncValue references
/// to wait on, and passes the elements into the completion handler as an
/// ArrayRef.  It can be used like this:
///
/// void example() {
///   AsyncValueRef<int32_t> elements[4] = { ... };
///   ...
///   andThenSyncCopying(elements,
///     [... any captures...](MutableArrayRef<AsyncValueRef<int32_t>> elts) {
///     ... stuff that uses elts ...
///   });
/// }
///
/// There are "copying" version and "moving" versions, as well as the "sync" and
/// "async" versions.  The "copying" version doesn't move the AsyncValue's from
/// the input array, while the "moving" version destructively takes the
/// elements out of the array passed in.  The "Sync" version runs the
/// completion handler as soon as the last value is fulfilled on the thread
/// that fulfills that value. The "Async" version adds the completion handler to
/// the work queue when all the values are fulfilled.

template <typename CompletionFn>
inline static void andThenSyncCopying(ArrayRef<AnyAsyncValueRef> values,
                                      CompletionFn &&completionFn) {
  Detail::andThenArrayImpl</*IsAsync=*/false>(
      values, std::forward<CompletionFn>(completionFn),
      [](const AnyAsyncValueRef &ref) -> AnyAsyncValueRef {
        return ref.copy();
      });
}

template <typename CompletionFn>
inline static void andThenAsyncCopying(ArrayRef<AnyAsyncValueRef> values,
                                       CompletionFn &&completionFn) {
  Detail::andThenArrayImpl</*IsAsync=*/true>(
      values, std::forward<CompletionFn>(completionFn),
      [](const AnyAsyncValueRef &ref) -> AnyAsyncValueRef {
        return ref.copy();
      });
}

template <typename CompletionFn>
inline static void
andThenSyncMoving(llvm::MutableArrayRef<AnyAsyncValueRef> values,
                  CompletionFn &&completionFn) {
  Detail::andThenArrayImpl</*IsAsync=*/false>(
      values, std::forward<CompletionFn>(completionFn),
      [](AnyAsyncValueRef &ref) -> AnyAsyncValueRef { return std::move(ref); });
}

template <typename CompletionFn>
inline static void
andThenAsyncMoving(llvm::MutableArrayRef<AnyAsyncValueRef> values,
                   CompletionFn &&completionFn) {
  Detail::andThenArrayImpl</*IsAsync=*/true>(
      values, std::forward<CompletionFn>(completionFn),
      [](AnyAsyncValueRef &ref) -> AnyAsyncValueRef { return std::move(ref); });
}

//===----------------------------------------------------------------------===//
// Helpers to add tasks to the cpuDevice's work queue
//===----------------------------------------------------------------------===//

/// Add some non-blocking work to the WorkQueue managed by the specified
/// Runtime.
inline static void addTask(CPUDevice &cpuDevice, WorkItem &&workItem,
                           int taskId = -1) {
  cpuDevice.getWorkQueue()->addTask(std::move(workItem), taskId);
}

/// Overload of addTask that returns AsyncValueRef<R> for work that returns R
/// (when R is not void).
///
/// Example:
/// int a = 1, b = 2;
/// AsyncValueRef<int> r = addTask(cpuDevice, [a, b] { return a + b; });
///
template <typename FnTy, typename ResultTy = Detail::ResultType<FnTy>,
          std::enable_if_t<!std::is_void<ResultTy>(), int> = 0>
[[nodiscard]] inline static AsyncValueRef<ResultTy>
addTask(CPUDevice &cpuDevice, FnTy work, int taskId = -1) {
  auto result = AsyncValueRef<ResultTy>::allocate(cpuDevice);

  addTask(
      cpuDevice,
      [result = result.copy(), work = std::forward<FnTy>(work)]() mutable {
        std::move(result).template emplace<ResultTy>(work());
      },
      taskId);
  return result;
}

//===----------------------------------------------------------------------===//
// parallelForEachN
//===----------------------------------------------------------------------===//

namespace Detail {
/// Struct containing various utilities used by the implementation of
/// parallelForEachN.
struct ParallelForEachNUtils {
  /// A utility to build a tuple type containing the decay'd capture arguments
  /// of the element function of a parallelForEachN. The only non-general aspect
  /// of this is that it skips the first argument, which is the index of the
  /// element.
  template <typename... Ts>
  struct ElementFnCapturesImplT;
  template <typename FnTraitsT, size_t... Ns>
  struct ElementFnCapturesImplT<FnTraitsT, std::index_sequence<Ns...>> {
    using type =
        std::tuple<std::decay_t<typename FnTraitsT::template arg_t<Ns + 1>>...>;
  };
  template <typename FnT>
  using ElementFnCapturesT = typename ElementFnCapturesImplT<
      llvm::function_traits<FnT>,
      std::make_index_sequence<llvm::function_traits<FnT>::num_args - 1>>::type;
};

} // namespace Detail

namespace Detail {

/// Core implementation of parallelForEachN. Accepts an explicit task-ID
/// mapping function so callers can pin each element to a specific worker
/// thread (e.g. for GPU device affinity). Pass `[](size_t) { return
/// kDefaultTaskId; }` for the default round-robin scheduling.
template <typename... CaptureTys, typename ElementFn, typename CompletionFn,
          typename TaskIdFn>
static inline void
parallelForEachNImpl(CPUDevice &cpuDevice, size_t totalCount,
                     ElementFn &&elementFn, CompletionFn &&completionFn,
                     TaskIdFn &&taskIdFn, CaptureTys &&...captures) {
  // If there is nothing to do, then we're already done.
  if (totalCount == 0)
    return;

  struct ParallelState {
    /// This is the number of elements left to finish executing.  When this
    /// drops to zero, the completion handler is run.
    std::atomic<size_t> numElementsLeft;

    /// This is the function to execute on each element.
    ElementFn elementFn;

    /// This is the function to execute once all elements are done.
    CompletionFn completionFn;

    /// This is the state captured by the computation, it is passed to both the
    /// per-element computation as well as to the completion function.
    ParallelForEachNUtils::ElementFnCapturesT<ElementFn> capturesList;
  };

  // Allocate the parallel state on the heap since it will out-live the call to
  // this function.  We will deallocate it after invoking the completion
  // handler when the last element completes.
  auto state = new ParallelState{totalCount,
                                 std::forward<ElementFn>(elementFn),
                                 std::forward<CompletionFn>(completionFn),
                                 {std::forward<CaptureTys>(captures)...}};

  // Enqueue each element of work!
  for (size_t elementIdx = 0; elementIdx != totalCount; ++elementIdx) {
    addTask(
        cpuDevice,
        [state, elementIdx]() {
          TimeTraceScope scope(AlgorithmProfilerEntry::create(
              "asyncrt.parallelForEach", (uint64_t)elementIdx));
          // Invoke the per-element function with the index and all of the
          // captured state.
          std::apply(
              [&](auto &&...args) {
                (void)state->elementFn(elementIdx, args...);
              },
              state->capturesList);
          // Once that is done we can decrement the count and trigger completion
          // when the last element is done.
          if (--state->numElementsLeft != 0)
            return;

          // Invoke the completion function, since we're done.
          std::apply(state->completionFn, state->capturesList);

          // All uses of the state are done, so we can deallocate it.
          delete state;
        },
        taskIdFn(elementIdx));
  }
}

} // namespace Detail

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// When all of the elements have finished, a completion handler is invoked.
///
template <typename... CaptureTys, typename ElementFn, typename CompletionFn>
static inline void parallelForEachNCustomCompletion(CPUDevice &cpuDevice,
                                                    size_t totalCount,
                                                    ElementFn &&elementFn,
                                                    CompletionFn &&completionFn,
                                                    CaptureTys &&...captures) {
  Detail::parallelForEachNImpl(
      cpuDevice, totalCount, std::forward<ElementFn>(elementFn),
      std::forward<CompletionFn>(completionFn),
      [](size_t) { return kDefaultTaskId; },
      std::forward<CaptureTys>(captures)...);
}

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// When all of the elements have finished, the `readyChain` is completed,
/// unblocking any computation `andThenSync`d on it.
///
template <typename... CaptureTys, typename ElementFn>
static inline void
parallelForEachNCompleteChain(CPUDevice &cpuDevice, size_t totalCount,
                              AsyncValueRef<Chain> readyChain,
                              ElementFn &&elementFn, CaptureTys &&...captures) {
  parallelForEachNCustomCompletion(
      cpuDevice, totalCount, std::forward<ElementFn>(elementFn),
      [readyChain = std::move(readyChain)](auto &&...args) mutable {
        // When all the elements are ready, complete the `readyChain`,
        // unblocking other work.
        std::move(readyChain).emplace();
      },
      std::forward<CaptureTys...>(captures)...);
}

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// This helper takes an initialized `EltTy` value, and when complete it
/// emplaces it into the resultAV value.
///
template <typename EltTy, typename... CaptureTys, typename ElementFn>
static inline void
parallelForEachNFinishing(CPUDevice &cpuDevice, size_t totalCount,
                          EltTy &&initialResultValue,
                          AsyncValueRef<EltTy> resultAV, ElementFn &&elementFn,
                          CaptureTys &&...captures) {
  parallelForEachNCustomCompletion(
      cpuDevice, totalCount, std::forward<ElementFn>(elementFn),
      [resultAV = std::move(resultAV)](EltTy &result, auto &&...args) mutable {
        // When all the elements are ready, emplace the result value into the
        // result AV.
        std::move(resultAV).emplace(std::move(result));
      },
      std::forward<EltTy>(initialResultValue),
      std::forward<CaptureTys...>(captures)...);
}

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// When all of the elements have finished, the chain result is marked as ready.
/// provides a convenient way to chain together work with `.andThenSync` on the
/// chain.
///
template <typename... CaptureTys, typename ElementFn>
static inline AsyncValueRef<Chain>
parallelForEachNChain(CPUDevice &cpuDevice, size_t totalCount,
                      ElementFn &&elementFn, CaptureTys &&...captures) {
  auto result = AsyncValueRef<Chain>::allocate(cpuDevice);
  parallelForEachNCompleteChain(cpuDevice, totalCount, result.copy(),
                                std::forward<ElementFn>(elementFn),
                                std::forward<CaptureTys...>(captures)...);
  return result;
}

/// Like parallelForEachNChain, but pins each element to the worker thread
/// returned by `taskIdFn(elementIdx)`. Use this when elements have device
/// affinity (e.g. one element per GPU, pinned to the GPU's NUMA-local core via
/// taskIdForDevice). Returning kDefaultTaskId (-1) from taskIdFn falls back to
/// the global shared queue for that element.
///
/// taskIdFn must be callable as (size_t elementIdx) -> int.
template <typename... CaptureTys, typename ElementFn, typename TaskIdFn>
static inline AsyncValueRef<Chain>
parallelForEachNChainWithTaskIds(CPUDevice &cpuDevice, size_t totalCount,
                                 TaskIdFn &&taskIdFn, ElementFn &&elementFn,
                                 CaptureTys &&...captures) {
  auto result = AsyncValueRef<Chain>::allocate(cpuDevice);
  Detail::parallelForEachNImpl(
      cpuDevice, totalCount, std::forward<ElementFn>(elementFn),
      [chain = result.copy()](auto &&...) mutable {
        std::move(chain).emplace();
      },
      std::forward<TaskIdFn>(taskIdFn), std::forward<CaptureTys>(captures)...);
  return result;
}

/// This method invokes the specified element function "N" times with indexes
/// from [0, N). Elements [0..N-1) are processed as tasks using the Runtime's
/// WorkQueue. Element N-1 is processed on the caller's thread. It returns when
/// all elements are completed.
///
/// Each call to the element function should have roughly the same latency.
/// The callers thread will not be donated to work on any other work items,
/// and may sleep waiting for elements [0..N-1) to complete.
///
/// Because this doesn't return until the elements are done, it is ok for the
/// element function to capture things on the caller's stack by reference.
template <typename... CaptureTys, typename ElementFn>
static inline void parallelForEachN(CPUDevice &cpuDevice, size_t totalCount,
                                    ElementFn &&elementFn,
                                    CaptureTys &&...captures) {

  if (totalCount == 0)
    return;

  // Execute N-1 elements for elements on background threads.
  AsyncValueRef<Chain> chainResult;
  if (totalCount > 1) {
    chainResult = parallelForEachNChain(
        cpuDevice, totalCount - 1, std::forward<ElementFn>(elementFn),
        std::forward<CaptureTys...>(captures)...);
  }

  // Execute the last element on this thread since we'll be blocking otherwise.
  // This thread just spent a bunch of time kicking off work for other threads,
  // so it may be the straggler and a bit behind the rest of the pack. That
  // said, there is a reasonable likelihood that the last element will be
  // smaller than the rest, so this thread can catch up with the others.
  {
    TimeTraceScope scope(AlgorithmProfilerEntry::create(
        "asyncrt.parallelForEach", (uint64_t)(totalCount - 1)));
    elementFn(totalCount - 1, captures...);
  }

  // Wait for the chain to become available. We'll assume our sharding of
  // the work to elementFn was well-balanced, and so won't attempt to run
  // any additional work items while waiting.
  if (chainResult)
    await(chainResult);
}

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_ALGORITHMS_H
