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

#ifndef ASYNCRT_SUPPORT_TIMERHEAP_H
#define ASYNCRT_SUPPORT_TIMERHEAP_H

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Support/Chain.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace M::AsyncRT {

class TimerHeap {
public:
  using deadline = std::chrono::time_point<std::chrono::steady_clock>;
  TimerHeap() : running(true), thread(&TimerHeap::run, this) {}
  ~TimerHeap() { stop(); }

  /// Push a new function which will be executed at the given deadline.
  void push(const deadline expiration, AsyncValueRef<Chain> &chain);
  void cancel(AsyncValueRef<Chain> &chain);

private:
  void stop();
  void run();

  /// Pending entries, managed as a heap.
  class Entry {
  public:
    Entry(const deadline expiration, AsyncValueRef<Chain> &&chain)
        : expiration(expiration), expired(std::move(chain)) {}
    deadline expiration;
    AsyncValueRef<Chain> expired; // Null if no longer needed.

    /// Something is *less* than something else if it expires further into the
    /// future. This is backwards, but useful for the priority queue. Think of
    /// this as the priority of the entry with respect to expiration.
    bool operator<(const Entry &other) const {
      return expiration > other.expiration;
    }
  };
  std::vector<Entry> entries;
  bool running;

  /// Protects `running` and `entries`.
  std::mutex mu;
  std::condition_variable cv;

  /// Exclusively runs `run` until stopped.
  std::thread thread;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_TIMERHEAP_H
