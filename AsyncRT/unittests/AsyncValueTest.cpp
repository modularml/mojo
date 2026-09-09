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
// Some unit tests for AsyncValue and friends.
//
// See also GraphRT/lib/Primitives/TestPrimitives.cpp
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Support/Semaphore.h"
#include "llvm/Support/Threading.h"

#include "gtest/gtest.h"
#include <thread>

using namespace M::AsyncRT;

namespace {

enum WorkQueueType { kSingleThread = 0, kThreadPool = 1 };

class AsyncValueTest : public testing::TestWithParam<WorkQueueType> {
protected:
  CPUDeviceRef getOrCreateCPUDevice(int numThreads = 4,
                                    bool mainWillDonate = true) {
    CPUDeviceOptions cpuDeviceOptions;
    cpuDeviceOptions.leakCheckedAllocator = true;
    cpuDeviceOptions.withSingleThreaded(GetParam() == kSingleThread);
    cpuDeviceOptions.numThreads = numThreads;
    cpuDeviceOptions.mainWillDonate = mainWillDonate;
    return M::AsyncRT::getOrCreateCPUDevice(M::AsyncRT::CPUDeviceSource::Test,
                                            cpuDeviceOptions);
  }
};

INSTANTIATE_TEST_SUITE_P(ManyRuntimes, AsyncValueTest,
                         testing::Values(kSingleThread, kThreadPool));

//===----------------------------------------------------------------------===//
// Async emplace
//===----------------------------------------------------------------------===//

struct Foo {
  int v = 0;

  Foo(int v) : v(v) {}
  Foo(Foo &&that) { std::swap(v, that.v); }
};

TEST_P(AsyncValueTest, EmplaceWithMove) {
  auto cpuDevice = getOrCreateCPUDevice();
  Foo foo(42);
  auto ref = AsyncValueRef<Foo>::allocate(*cpuDevice);
  ref.copy().emplace(std::move(foo));
  EXPECT_EQ(foo.v, 0);
  EXPECT_EQ(ref->v, 42);
}

//===----------------------------------------------------------------------===//
// Idiomatic async producer/consumer
//===----------------------------------------------------------------------===//

AsyncValueRef<int> typedProducer(CPUDevice &cpuDevice) {
  auto result = AsyncValueRef<int>::allocate(cpuDevice);
  addTask(cpuDevice,
          [result = result.copy()]() mutable { std::move(result).emplace(1); });
  return result;
}

int typedConsumer(AsyncValueRef<int> result) { return *result + 1; }

TEST_P(AsyncValueTest, TypedProducerConsumer) {
  auto cpuDevice = getOrCreateCPUDevice();
  AsyncValueRef<int> finished = AsyncValueRef<int>::allocate(*cpuDevice);
  AsyncValueRef<int> result = typedProducer(*cpuDevice);
  std::move(result).andThenSync(
      [finished = finished.copy()](AsyncValueRef<int> &&result) mutable {
        std::move(finished).emplace(typedConsumer(std::move(result)));
      });
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

AnyAsyncValueRef anyProducer(CPUDevice &cpuDevice) {
  auto result = AnyAsyncValueRef::allocate<int>(cpuDevice);
  addTask(cpuDevice, [result = result.copy()]() mutable {
    std::move(result).emplace<int>(1);
  });
  return result;
}

int anyConsumer(AnyAsyncValueRef result) { return result.get<int>() + 1; }

TEST_P(AsyncValueTest, AnyProducerConsumer) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  AnyAsyncValueRef result = anyProducer(*cpuDevice);
  std::move(result).andThenSync(
      [finished = finished.copy()](AnyAsyncValueRef &&result) mutable {
        std::move(finished).emplace(anyConsumer(std::move(result)));
      });
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

//===----------------------------------------------------------------------===//
// No stray references
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, IsUnique) {
  auto cpuDevice = getOrCreateCPUDevice();

  {
    auto ref = AsyncValueRef<int>::allocate(*cpuDevice);
    ASSERT_TRUE(ref.getPointer()->isUnique());
    std::move(ref).emplace(42);
  }

  {
    auto ref = AsyncValueRef<int>::allocate(*cpuDevice);
    auto ref2 = ref.copy();
    ASSERT_FALSE(ref.getPointer()->isUnique());
    std::move(ref).emplace(42);
  }

  {
    auto indirectRef = AnyAsyncValueRef::createIndirect(*cpuDevice);
    auto concreteRef = AsyncValueRef<int>::createReady(*cpuDevice, 42);
    indirectRef.copy().resolveIndirect(std::move(concreteRef));
    ASSERT_TRUE(indirectRef.getPointer()->isUnique());
  }

  {
    auto indirectRef = AnyAsyncValueRef::createIndirect(*cpuDevice);
    auto concreteRef = AsyncValueRef<int>::allocate(*cpuDevice);
    indirectRef.copy().resolveIndirect(concreteRef.copy());
    ASSERT_FALSE(indirectRef.getPointer()->isUnique());
    std::move(concreteRef).emplace(42);
  }

  {
    auto indirectRef = AnyAsyncValueRef::createIndirect(*cpuDevice);
    auto concreteRef = AsyncValueRef<int>::createReady(*cpuDevice, 42);
    auto concreteRef2 = concreteRef.copy();
    auto concreteRef3 = concreteRef.copy();
    indirectRef.copy().resolveIndirect(std::move(concreteRef));
    ASSERT_FALSE(indirectRef.getPointer()->isUnique());
  }

  {
    auto indirectRef = AnyAsyncValueRef::createIndirect(*cpuDevice);
    auto indirectRef2 = indirectRef.copy();
    auto concreteRef = AsyncValueRef<int>::createReady(*cpuDevice, 42);
    indirectRef.copy().resolveIndirect(std::move(concreteRef));
    ASSERT_FALSE(indirectRef.getPointer()->isUnique());
  }
}

TEST_P(AsyncValueTest, SyncConsuming) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  auto ref = AnyAsyncValueRef::allocate<int>(*cpuDevice);
  ref.andThenSync([ref = ref.copy(), finished = finished.copy()]() mutable {
    // At this point r is the only remaining reference due to the use
    // of AsyncValue::emplace below.
    EXPECT_TRUE(ref.getPointer()->isUnique());
    EXPECT_EQ(ref.get<int>(), 1);
    std::move(finished).emplace(2);
  });
  EXPECT_EQ(ref.getPointer()->getRefCountForDebugging(), 2u);
  std::move(ref).emplace<int>(1);
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

TEST_P(AsyncValueTest, AsyncConsuming) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  auto ref = AnyAsyncValueRef::allocate<int>(*cpuDevice);
  ref.andThenAsync([ref = ref.copy(), finished = finished.copy()]() mutable {
    // At this point r is the only remaining reference due to the use
    // of AsyncValue::emplace below.
    EXPECT_TRUE(ref.getPointer()->isUnique());
    EXPECT_EQ(ref.get<int>(), 1);
    std::move(finished).emplace(2);
  });
  EXPECT_EQ(ref.getPointer()->getRefCountForDebugging(), 2u);
  std::move(ref).emplace<int>(1);
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

//===----------------------------------------------------------------------===//
// Waiters run off stack
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, EmplacingFromTask_DeadlockOnFailure) {
  if (GetParam() != kThreadPool)
    // Can only observe this behaviour with the thread pool workqueue.
    return;

  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  Semaphore testReady;
  Semaphore testDone;
  Semaphore canaryProceed;
  addTask(*cpuDevice, [&testReady, &testDone, &canaryProceed,
                       cpuDevice = cpuDevice.getPointer(),
                       finished = finished.copy()]() mutable {
    testReady.post();
    // Run the test inside an AsyncRT task. Waiter can be scheduled on the
    // same thread.
    auto ref = AsyncValueRef<Chain>::allocate(*cpuDevice);
    ref.andThenSync([&canaryProceed, finished = finished.copy()]() mutable {
      canaryProceed.wait();
      std::move(finished).emplace(1);
    });
    // We'll deadlock if the continuation is run now.
    std::move(ref).emplace();
    testDone.post();
  });
  // Make sure test task is running.
  testReady.wait();
  testDone.wait();
  canaryProceed.post();
  await(finished);
  EXPECT_EQ(finished.get(), 1);
}

TEST_P(AsyncValueTest, EmplaceOnForeignThread_DeadlockOnFailure) {
  if (GetParam() != kThreadPool)
    // Can only observe this behaviour with the thread pool workqueue.
    return;

  // Run the test inside the main (ie 'foreign') thread. Waiter will be
  // scheduled as an AsyncRT task.
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  Semaphore canRun;
  auto ref = AsyncValueRef<Chain>::allocate(*cpuDevice);
  ref.andThenSync([&canRun, finished = finished.copy()]() mutable {
    canRun.wait();
    std::move(finished).emplace(1);
  });
  // We'll deadlock if the continuation is run now.
  std::move(ref).emplace();
  canRun.post();
  await(finished);
  EXPECT_EQ(finished.get(), 1);
}

//===----------------------------------------------------------------------===//
// Recursive 'await'
//===----------------------------------------------------------------------===//

// This rather convoluted test forces an inner run loop to run recursively
// within an outer run loop, with both loops needing to dispatch a waiter.
// It works for both single- and multi-threaded work queues. In the
// multi-threaded case in particular it tests that the machinery to collect
// and execute waiters treats the waiter list as a true queue which may
// grow and shrink while executing a waiter.
TEST_P(AsyncValueTest, RecursiveAsync) {
  auto cpuDevice = getOrCreateCPUDevice(/*numThreads=*/2);

  // If needed, force the worker thread to be occupied with a 'dummy' task.
  Semaphore dummyRunning;
  Semaphore dummyContinue;
  auto dummyFinished = AsyncValueRef<int>::allocate(*cpuDevice);
  if (GetParam() == kThreadPool) {
    addTask(*cpuDevice, [&dummyRunning, &dummyContinue,
                         dummyFinished = dummyFinished.copy()]() mutable {
      dummyRunning.post();
      dummyContinue.wait();
      std::move(dummyFinished).emplace(1);
    });
    dummyRunning.wait();
  }

  // The main thread will run the 'test' task, but trigger from a waiter.
  auto testFinished = AsyncValueRef<int>::allocate(*cpuDevice);
  auto testTrigger = AsyncValueRef<Chain>::allocate(*cpuDevice);
  testTrigger.andThenSync([cpuDevice = cpuDevice.getPointer(),
                           testFinished = testFinished.copy()]() {
    addTask(*cpuDevice, [cpuDevice,
                         testFinished = testFinished.copy()]() mutable {
      // The main thread will also run the 'nested' task, again triggered from
      // a waiter.
      auto nestedFinished = AsyncValueRef<int>::allocate(*cpuDevice);
      auto nestedTrigger = AsyncValueRef<Chain>::allocate(*cpuDevice);
      nestedTrigger.andThenSync([cpuDevice,
                                 nestedFinished = nestedFinished.copy()]() {
        addTask(*cpuDevice, [nestedFinished = nestedFinished.copy()]() mutable {
          std::move(nestedFinished).emplace(3);
        });
      });
      std::move(nestedTrigger).emplace();

      // This await will start the inner run loop.
      await(nestedFinished);
      EXPECT_EQ(nestedFinished.get(), 3);
      std::move(testFinished).emplace(2);
    });
  });
  std::move(testTrigger).emplace();

  // This await will start the outer run loop.
  await(testFinished);
  EXPECT_EQ(testFinished.get(), 2);

  if (GetParam() == kThreadPool) {
    // Dummy can now proceed to completion.
    dummyContinue.post();
    await(dummyFinished);
    EXPECT_EQ(dummyFinished.get(), 1);
  } else {
    std::move(dummyFinished).emplace();
  }
}

//===----------------------------------------------------------------------===//
// Special andThen{Sync,Async}s from Algorithms
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, TupleAndThenSync) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  auto ref1 = AnyAsyncValueRef::allocate<int>(*cpuDevice);
  auto ref2 = AnyAsyncValueRef::allocate<char>(*cpuDevice);
  andThenSync(std::make_tuple(ref1.copy(), ref2.copy()),
              [finished = finished.copy()](AnyAsyncValueRef ref1,
                                           AnyAsyncValueRef ref2) mutable {
                // Confirm that the closure is running after the original
                // `ref` is destroyed.
                EXPECT_TRUE(ref1.getPointer()->isUnique());
                EXPECT_TRUE(ref2.getPointer()->isUnique());
                EXPECT_EQ(ref1.get<int>(), 1);
                EXPECT_EQ(ref2.get<char>(), 'a');
                std::move(finished).emplace(2);
              });
  std::move(ref1).emplace<int>(1);
  std::move(ref2).emplace<char>('a');
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

TEST_P(AsyncValueTest, TupleAndThenAsync) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  auto ref1 = AnyAsyncValueRef::allocate<int>(*cpuDevice);
  auto ref2 = AnyAsyncValueRef::allocate<char>(*cpuDevice);
  andThenAsync(std::make_tuple(ref1.copy(), ref2.copy()),
               [finished = finished.copy()](AnyAsyncValueRef ref1,
                                            AnyAsyncValueRef ref2) mutable {
                 // Confirm that the closure is running after the original
                 // `ref` is destroyed.
                 EXPECT_TRUE(ref1.getPointer()->isUnique());
                 EXPECT_TRUE(ref2.getPointer()->isUnique());
                 EXPECT_EQ(ref1.get<int>(), 1);
                 EXPECT_EQ(ref2.get<char>(), 'a');
                 std::move(finished).emplace(2);
               });
  std::move(ref1).emplace<int>(1);
  std::move(ref2).emplace<char>('a');
  await(finished);
  EXPECT_EQ(finished.get(), 2);
}

TEST_P(AsyncValueTest, ArrayCopyingSync) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*cpuDevice));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*cpuDevice));
  refs[0].copy().emplace<int>(1);
  refs[1].copy().emplace<int>(2);
  andThenSyncCopying(
      llvm::ArrayRef(refs), [finished = finished.copy()](
                                llvm::ArrayRef<AnyAsyncValueRef> elts) mutable {
        // `refs` is copied, so each element has refcount 2 when
        // the completion function is executed.
        EXPECT_EQ(elts[0].getPointer()->getRefCountForDebugging(), 2u);
        EXPECT_EQ(elts[1].getPointer()->getRefCountForDebugging(), 2u);
        EXPECT_EQ(elts[0].get<int>(), 1);
        EXPECT_EQ(elts[1].get<int>(), 2);
        std::move(finished).emplace(3);
      });
  await(finished);
  EXPECT_EQ(finished.get(), 3);
}

TEST_P(AsyncValueTest, ArrayCopyingAsync) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*cpuDevice));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*cpuDevice));
  refs[0].copy().emplace<int>(1);
  refs[1].copy().emplace<int>(2);
  andThenAsyncCopying(
      llvm::ArrayRef(refs), [finished = finished.copy()](
                                llvm::ArrayRef<AnyAsyncValueRef> elts) mutable {
        // `refs` is copied, so each element has refcount 2 when
        // the completion function is executed.
        EXPECT_EQ(elts[0].getPointer()->getRefCountForDebugging(), 2u);
        EXPECT_EQ(elts[1].getPointer()->getRefCountForDebugging(), 2u);
        EXPECT_EQ(elts[0].get<int>(), 1);
        EXPECT_EQ(elts[1].get<int>(), 2);
        std::move(finished).emplace(3);
      });
  await(finished);
  EXPECT_EQ(finished.get(), 3);
}

TEST_P(AsyncValueTest, ArrayMovingSync) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*cpuDevice));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*cpuDevice));
  refs[0].copy().emplace<int>(1);
  refs[1].copy().emplace<int>(2);
  andThenSyncMoving(llvm::MutableArrayRef(refs),
                    [finished = finished.copy()](
                        llvm::MutableArrayRef<AnyAsyncValueRef> elts) mutable {
                      // `refs` is moved, so each element has refcount 1 when
                      // the completion function is executed.
                      EXPECT_TRUE(elts[0].getPointer()->isUnique());
                      EXPECT_TRUE(elts[1].getPointer()->isUnique());
                      EXPECT_EQ(elts[0].get<int>(), 1);
                      EXPECT_EQ(elts[1].get<int>(), 2);
                      std::move(finished).emplace(3);
                    });
  await(finished);
  EXPECT_EQ(finished.get(), 3);
}

TEST_P(AsyncValueTest, ArrayMovingAsync) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::allocate(*cpuDevice);
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*cpuDevice));
  refs.emplace_back(AnyAsyncValueRef::allocate<int>(*cpuDevice));
  refs[0].copy().emplace<int>(1);
  refs[1].copy().emplace<int>(2);
  andThenAsyncMoving(llvm::MutableArrayRef(refs),
                     [finished = finished.copy()](
                         llvm::MutableArrayRef<AnyAsyncValueRef> elts) mutable {
                       // `refs` is moved, so each element has refcount 1 when
                       // the completion function is executed.
                       EXPECT_TRUE(elts[0].getPointer()->isUnique());
                       EXPECT_TRUE(elts[1].getPointer()->isUnique());
                       EXPECT_EQ(elts[0].get<int>(), 1);
                       EXPECT_EQ(elts[1].get<int>(), 2);
                       std::move(finished).emplace(3);
                     });
  await(finished);
  EXPECT_EQ(finished.get(), 3);
}

//===----------------------------------------------------------------------===//
// 'Stress' tests (despite the name these tests currently run in about 30ms)
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, StressAndThen) {
  // Deliberately over-subscribe threads to try to tickle any races.
  auto cpuDevice =
      getOrCreateCPUDevice(/*numThreads=*/M::getNumLogicalCores() * 2);

  const size_t nRounds = 5;
  const size_t nValues = 500;
  // Root AsyncValue.
  auto start = AsyncValueRef<size_t>::allocate(*cpuDevice);
  // Intermediate AsyncValues.
  llvm::SmallVector<llvm::SmallVector<AsyncValueRef<size_t>>> refs;
  for (size_t i = 0; i < nRounds; ++i) {
    refs.emplace_back();
    for (size_t j = 0; j < nValues; ++j)
      refs.back().emplace_back(AsyncValueRef<size_t>::allocate(*cpuDevice));
  }
  // Final AsyncValue.
  auto finish = AsyncValueRef<Chain>::allocate(*cpuDevice);

  // Intermediate dependencies.
  for (size_t i = 0; i < nRounds; ++i) {
    for (size_t j = 0; j < nValues; ++j) {
      const AsyncValueRef<size_t> &prev = i == 0 ? start : refs[i - 1][j];
      const AsyncValueRef<size_t> &next = refs[i][j];
      prev.copy().andThenAsync(
          [next = next.copy()](AsyncValueRef<size_t> &&prev) mutable {
            std::this_thread::sleep_for(std::chrono::microseconds(rand() % 50));
            std::move(next).emplace(prev.get() + 1);
          });
    }
  }

  // Final values and join condition.
  std::atomic<size_t> sum = 0;
  std::atomic<size_t> waiting = nValues;
  for (size_t j = 0; j < nValues; ++j) {
    std::move(refs[nRounds - 1][j])
        .andThenAsync([&sum, &waiting, finish = finish.copy()](
                          AsyncValueRef<size_t> &&prev) mutable {
          sum += prev.get();
          if (--waiting == 0)
            std::move(finish).emplace();
        });
  }

  std::move(start).emplace(1);
  await(finish);

  EXPECT_EQ(sum, (nRounds + 1) * nValues);
}

TEST_P(AsyncValueTest, StressParallelForEachN) {
  // Deliberately over-subscribe threads to try to tickle any races.
  auto cpuDevice = getOrCreateCPUDevice(
      /*numThreads=*/M::getNumLogicalCores() * 2);

  const size_t nShards = 500;
  std::vector<std::unique_ptr<std::atomic<bool>>> doneFlags;
  for (size_t i = 0; i < nShards; ++i)
    doneFlags.emplace_back(std::make_unique<std::atomic<bool>>());

  auto elementFn = [&doneFlags](size_t index) {
    std::this_thread::sleep_for(std::chrono::microseconds(200 + rand() % 50));
    EXPECT_FALSE(*doneFlags[index]);
    *doneFlags[index] = true;
  };

  parallelForEachN(*cpuDevice, nShards, std::move(elementFn));

  for (size_t i = 0; i < nShards; ++i)
    EXPECT_TRUE(*doneFlags[i]) << "(index " << i << ")";
}

//===----------------------------------------------------------------------===//
// Awaiting
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, AwaitFromForeign) {
  if (GetParam() != kThreadPool)
    // Can only observe this behaviour with the thread pool workqueue.
    return;

  auto cpuDevice = getOrCreateCPUDevice();

  constexpr size_t nTasks = 20;
  Semaphore canRun[nTasks];
  llvm::SmallVector<AnyAsyncValueRef> finished;
  finished.reserve(nTasks);
  for (size_t i = 0; i < nTasks; ++i) {
    finished.emplace_back(AsyncValueRef<Chain>::allocate(*cpuDevice));
    addTask(*cpuDevice, [i, &canRun, finished = finished[i].copy()]() mutable {
      canRun[i].wait();
      std::move(finished).emplace<Chain>();
    });
  }

  std::thread foreign[nTasks];
  for (size_t i = 0; i < nTasks; ++i)
    foreign[i] = std::thread([i, &finished]() { await(finished[i]); });

  for (size_t i = 0; i < nTasks; ++i)
    canRun[i].post();
  for (size_t i = 0; i < nTasks; ++i)
    foreign[i].join();

  await(finished);
}

TEST_P(AsyncValueTest, AwaitWithoutDonating) {
  if (GetParam() != kThreadPool)
    // Can only observe this behaviour with the thread pool workqueue.
    return;

  auto cpuDevice =
      getOrCreateCPUDevice(/*numThreads=*/4, /*mainWillDonate=*/false);

  constexpr size_t nTasks = 20;
  Semaphore canRun[nTasks];
  llvm::SmallVector<AnyAsyncValueRef> finished;
  finished.reserve(nTasks);
  for (size_t i = 0; i < nTasks; ++i) {
    finished.emplace_back(AsyncValueRef<Chain>::allocate(*cpuDevice));
    addTask(*cpuDevice, [i, &canRun, finished = finished[i].copy()]() mutable {
      canRun[i].wait();
      std::move(finished).emplace<Chain>();
    });
  }

  std::thread foreign[nTasks];
  for (size_t i = 0; i < nTasks; ++i)
    foreign[i] = std::thread([i, &finished]() { await(finished[i]); });

  for (size_t i = 0; i < nTasks; ++i)
    canRun[i].post();
  for (size_t i = 0; i < nTasks; ++i)
    foreign[i].join();

  await(finished);
}

//===----------------------------------------------------------------------===//
// addTask
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, AddTaskOverflow_DeadlockOnFailure) {
  if (GetParam() != kThreadPool)
    // Can only observe with the thread pool workqueue.
    return;

  auto cpuDevice = getOrCreateCPUDevice(/*numThreads=*/2);

  // Keep worker 1 occupied so it won't be able to execute any 'extra' tasks.
  Semaphore workerIsWaiting;
  Semaphore workerCanRun;
  auto workerFinished = AsyncValueRef<Chain>::allocate(*cpuDevice);
  addTask(*cpuDevice, [&workerIsWaiting, &workerCanRun,
                       workerFinished = workerFinished.copy()]() mutable {
    workerIsWaiting.post();
    workerCanRun.wait();
    std::move(workerFinished).emplace();
  });
  workerIsWaiting.wait();

  // Prepare for the 'extra' tasks.
  constexpr size_t nExtraTasks = 1000; // More than the task queue capacity.
  llvm::SmallVector<AnyAsyncValueRef> extraFinished;
  extraFinished.reserve(nExtraTasks);
  for (size_t i = 0; i < nExtraTasks; ++i)
    extraFinished.emplace_back(AsyncValueRef<Chain>::allocate(*cpuDevice));

  // Add the main task so the inner addTasks will appear to come from
  // a 'foreign awaiting' thread.
  auto mainFinished = AsyncValueRef<Chain>::allocate(*cpuDevice);
  Semaphore extraCanRun[nExtraTasks];
  addTask(*cpuDevice, [&cpuDevice, mainFinished = mainFinished.copy(),
                       &extraFinished, &extraCanRun]() mutable {
    // Flood the task queue with 'extra' tasks.
    for (size_t i = 0; i < nExtraTasks; ++i) {
      addTask(*cpuDevice, [i, extraFinished = extraFinished[i].copy(),
                           &extraCanRun]() mutable {
        // Will deadlock if addTask runs this item immediately.
        extraCanRun[i].wait();
        std::move(extraFinished).emplace<Chain>();
      });
    }

    // The 'extra' tasks can now proceed.
    for (size_t i = 0; i < nExtraTasks; ++i)
      extraCanRun[i].post();

    std::move(mainFinished).emplace();
  });

  // Run the main task.
  await(mainFinished);

  // Cleanup.
  workerCanRun.post();
  await(workerFinished);
  // Will deadlock if an extra task was dropped.
  await(extraFinished);
}

//===----------------------------------------------------------------------===//
// Large refcount add
//===----------------------------------------------------------------------===//

TEST_P(AsyncValueTest, LargeAsyncRefCount) {
  auto cpuDevice = getOrCreateCPUDevice();
  auto finished = AsyncValueRef<int>::createReady(*cpuDevice, 0);

  // Refcount is initialized to 1.
  AsyncValue *value = finished.getPointer();
  EXPECT_EQ(value->getRefCountForDebugging(), 1);

  // Increase it.
  M::RCRef<AsyncValue>::lowLevelAddRef(value, 33);
  EXPECT_EQ(value->getRefCountForDebugging(), 34);

  // Increase it with a large number to ensure there is no overflow.
  M::RCRef<AsyncValue>::lowLevelAddRef(value, 123457);
  EXPECT_EQ(value->getRefCountForDebugging(), 123491);

  // Decrease it back to 1
  M::RCRef<AsyncValue>::lowLevelDropRef(value, 123490);
  EXPECT_EQ(value->getRefCountForDebugging(), 1);
}

} // namespace
