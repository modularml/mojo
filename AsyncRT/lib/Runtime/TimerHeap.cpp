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

#include "AsyncRT/Runtime/TimerHeap.h"
#include "AsyncRT/Runtime/Algorithms.h"

using namespace M::AsyncRT;

void TimerHeap::push(const deadline expiration, AsyncValueRef<Chain> &chain) {
  std::unique_lock<std::mutex> lk(mu);
  bool notify = entries.empty() || entries[0].expiration > expiration;
  entries.emplace_back(Entry(expiration, chain.copy()));
  std::push_heap(entries.begin(), entries.end());
  if (notify)
    cv.notify_one();
}

void TimerHeap::cancel(AsyncValueRef<Chain> &chain) {
  std::unique_lock<std::mutex> lk(mu);
  bool reheap = false;
  unsigned long i = 0;
  while (i < entries.size()) {
    auto &entry = entries[i];
    if (entry.expired.getPointer() == chain.getPointer()) {
      entry.expired.copy().emplace();
      if (i < entries.size() - 1) {
        std::swap(entries[i], entries.back());
        reheap = true;
      }
      entries.pop_back();
      continue;
    }
    ++i;
  }
  if (reheap)
    std::make_heap(entries.begin(), entries.end());
}

void TimerHeap::stop() {
  {
    std::unique_lock<std::mutex> lk(mu);
    if (!running)
      return;
    running = false;
    cv.notify_one();
  }
  thread.join();
}

void TimerHeap::run() {
  std::unique_lock<std::mutex> lk(mu);
  while (running) {
    if (entries.empty()) {
      cv.wait(lk, [&] { return !running || !entries.empty(); });
      continue; // Recheck running.
    }
    const Entry &next = entries[0];
    auto now = std::chrono::steady_clock::now();
    if (now >= next.expiration) {
      // The entry has expired, trigger the chain and drop it. This only
      // happens if the reference hasn't been canceled/clear. See above.
      if (next.expired.getPointer())
        next.expired.copy().emplace();
      std::pop_heap(entries.begin(), entries.end());
      entries.pop_back();
    } else {
      // Wait until this point, or we are signalled.
      auto delta = next.expiration - now;
      cv.wait_for(lk, delta, [&] { return !running; });
    }
  }
}
