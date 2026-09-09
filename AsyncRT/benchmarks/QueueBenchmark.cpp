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

#include "AsyncRT/Support/ConcurrentQueue.h"
#include "AsyncRT/Support/LockFreeRingBuffer.h"

#include "benchmark/benchmark.h"

#include <chrono>
#include <thread>

using namespace M;
using namespace M::AsyncRT;

class ConcurrentQueueTest {
public:
  void enqueue() {
    void *val = (void *)&queue;
    queue.enqueue(std::move(val));
  }
  void dequeue() {
    while (!queue.dequeue())
      ;
  }

private:
  ConcurrentQueue<void *> queue;
};

class LockFreeRingBufferTest {
public:
  LockFreeRingBufferTest() : queue(1024) {}
  void enqueue() {
    void *val = (void *)&queue;
    while (!queue.enqueue(val))
      ;
  }
  void dequeue() {
    while (!queue.dequeue())
      ;
  }

private:
  LockFreeRingBuffer<void *> queue;
};

template <class T>
void BM_SingleProducer(benchmark::State &state) {
  static T *queue;
  int consumers = state.threads() - 1;
  if (state.thread_index() == 0) {
    queue = new (T);
    for (auto _ : state) {
      for (int i = 0; i < consumers; ++i)
        queue->enqueue();
    }
    delete queue;
  } else {
    for (auto _ : state) {
      queue->dequeue();
    }
  }
}

BENCHMARK(BM_SingleProducer<ConcurrentQueueTest>)->ThreadRange(2, 16);
BENCHMARK(BM_SingleProducer<LockFreeRingBufferTest>)->ThreadRange(2, 16);

template <class T>
void BM_SingleConsumer(benchmark::State &state) {
  static T *queue;
  int producers = state.threads() - 1;
  if (state.thread_index() == 0) {
    queue = new (T);
    for (auto _ : state) {
      for (int i = 0; i < producers; ++i)
        queue->dequeue();
    }
    delete queue;
  } else {
    for (auto _ : state) {
      queue->enqueue();
    }
  }
}

BENCHMARK(BM_SingleConsumer<ConcurrentQueueTest>)->ThreadRange(2, 16);
BENCHMARK(BM_SingleConsumer<LockFreeRingBufferTest>)->ThreadRange(2, 16);

template <class T>
void BM_Balanced(benchmark::State &state) {
  static T *queue;
  int consumers = state.threads() / 2;
  if (state.thread_index() == 0)
    queue = new (T);
  if (state.thread_index() < consumers) {
    for (auto _ : state) {
      queue->dequeue();
    }
  } else {
    for (auto _ : state) {
      queue->enqueue();
    }
  }
  if (state.thread_index() == 0)
    delete queue;
}

BENCHMARK(BM_Balanced<ConcurrentQueueTest>)->ThreadRange(2, 16);
BENCHMARK(BM_Balanced<LockFreeRingBufferTest>)->ThreadRange(2, 16);
