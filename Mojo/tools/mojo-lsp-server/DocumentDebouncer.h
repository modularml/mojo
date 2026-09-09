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

#ifndef MOJO_LSP_SERVER_DOCUMENT_DEBOUNCER_H
#define MOJO_LSP_SERVER_DOCUMENT_DEBOUNCER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LSP/Protocol.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace M::Mojo::LSP {

/// Manages debounced document updates to avoid parsing on every keystroke.
/// When a document update is received, it waits for a configurable delay before
/// triggering the actual parse. If another update arrives before the delay
/// expires, the timer is reset.
template <typename CallbackT>
class DocumentDebouncer {
public:
  /// The default delay to wait after the last keystroke before parsing.
  static constexpr std::chrono::milliseconds kDefaultDebounceDelay{150};

  explicit DocumentDebouncer(
      CallbackT callback,
      std::chrono::milliseconds debounceDelay = kDefaultDebounceDelay)
      : callback(std::move(callback)), debounceDelay(debounceDelay),
        running(true) {
    workerThread = std::thread([this] { workerLoop(); });
  }

  ~DocumentDebouncer() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      running = false;
      cv.notify_one();
    }
    if (workerThread.joinable())
      workerThread.join();
  }

  /// Schedule a document update. If there's already a pending update for this
  /// document, it will be replaced and the timer reset. The generation
  /// parameter is passed through to the callback to detect stale updates.
  void scheduleUpdate(const llvm::lsp::URIForFile &uri, std::string contents,
                      int64_t version, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex);
    pendingUpdates[uri.file()] =
        PendingUpdate{uri, std::move(contents), version, generation,
                      std::chrono::steady_clock::now()};
    cv.notify_one();
  }

  /// Cancel any pending update for the given URI.
  void cancelUpdate(llvm::StringRef filename) {
    std::lock_guard<std::mutex> lock(mutex);
    pendingUpdates.erase(filename);
  }

  /// Flush all pending updates immediately (useful for shutdown).
  /// Note: This only processes updates currently in the pending queue. If the
  /// worker thread is mid-callback, that callback will complete independently.
  /// For shutdown, callers should call flush() then destroy the debouncer
  /// (which joins the worker thread) before iterating over results - this
  /// ensures any in-flight callback completes before the results are read.
  void flush() {
    std::vector<PendingUpdate> updates;
    {
      std::lock_guard<std::mutex> lock(mutex);
      for (auto &entry : pendingUpdates)
        updates.push_back(std::move(entry.second));
      pendingUpdates.clear();
    }
    for (auto &update : updates)
      callback(update.uri, std::move(update.contents), update.version,
               update.generation);
  }

private:
  struct PendingUpdate {
    llvm::lsp::URIForFile uri;
    std::string contents;
    int64_t version;
    uint64_t generation;
    std::chrono::steady_clock::time_point lastUpdate;
  };

  void workerLoop() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);

      cv.wait(lock, [this] { return !running || !pendingUpdates.empty(); });
      if (!running)
        return;

      // Find the earliest deadline among all pending updates.
      auto now = std::chrono::steady_clock::now();
      auto earliestDeadline = std::chrono::steady_clock::time_point::max();
      for (auto &entry : pendingUpdates)
        if (auto deadline = entry.second.lastUpdate + debounceDelay;
            deadline < earliestDeadline)
          earliestDeadline = deadline;

      // Wait until the earliest deadline or until we're notified of new
      // updates.
      if (earliestDeadline > now) {
        cv.wait_until(lock, earliestDeadline);
        if (!running)
          return;
        now = std::chrono::steady_clock::now();
      }

      // Process all updates whose deadlines have passed.
      std::vector<PendingUpdate> readyUpdates;
      for (auto it = pendingUpdates.begin(); it != pendingUpdates.end();) {
        if (it->second.lastUpdate + debounceDelay <= now) {
          readyUpdates.push_back(std::move(it->second));
          pendingUpdates.erase(it++);
        } else {
          ++it;
        }
      }

      // Release the lock while processing updates to avoid blocking new
      // updates.
      lock.unlock();

      for (auto &update : readyUpdates)
        callback(update.uri, std::move(update.contents), update.version,
                 update.generation);
    }
  }

  CallbackT callback;
  std::chrono::milliseconds debounceDelay;
  std::thread workerThread;
  std::mutex mutex;
  std::condition_variable cv;
  llvm::StringMap<PendingUpdate> pendingUpdates;
  bool running;
};

// Deduction guide for DocumentDebouncer.
template <typename CallbackT>
DocumentDebouncer(CallbackT) -> DocumentDebouncer<CallbackT>;

} // namespace M::Mojo::LSP

#endif // MOJO_LSP_SERVER_DOCUMENT_DEBOUNCER_H
