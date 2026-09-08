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

#include "Mojo/tools/mojo-lsp-server/DocumentDebouncer.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <thread>

using namespace M::Mojo::LSP;

namespace {

/// Test helper to create URIs for testing.
llvm::lsp::URIForFile makeURI(llvm::StringRef path) {
  auto uri = llvm::lsp::URIForFile::fromURI("test://" + path.str());
  EXPECT_TRUE(static_cast<bool>(uri));
  return *uri;
}

/// Test callback that records calls for verification.
struct TestCallback {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::tuple<std::string, std::string, int64_t, uint64_t>> calls;

  void operator()(const llvm::lsp::URIForFile &uri, std::string contents,
                  int64_t version, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex);
    calls.emplace_back(uri.file().str(), std::move(contents), version,
                       generation);
    cv.notify_all();
  }

  /// Wait for at least n calls to be recorded.
  bool waitForCalls(size_t n, std::chrono::milliseconds timeout =
                                  std::chrono::milliseconds(1000)) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, timeout, [&] { return calls.size() >= n; });
  }

  size_t callCount() {
    std::lock_guard<std::mutex> lock(mutex);
    return calls.size();
  }

  std::tuple<std::string, std::string, int64_t, uint64_t>
  getCall(size_t index) {
    std::lock_guard<std::mutex> lock(mutex);
    return calls.at(index);
  }
};

} // namespace

TEST(DocumentDebouncerTest, BasicDebounce) {
  TestCallback callback;
  // Use a short delay for faster tests.
  constexpr auto delay = std::chrono::milliseconds(50);
  DocumentDebouncer debouncer(std::ref(callback), delay);

  auto uri = makeURI("/foo.mojo");
  debouncer.scheduleUpdate(uri, "content1", 1, /*generation=*/1);

  // Wait for the callback to be invoked.
  ASSERT_TRUE(callback.waitForCalls(1));
  EXPECT_EQ(callback.callCount(), 1u);

  auto [path, contents, version, gen] = callback.getCall(0);
  EXPECT_EQ(path, "/foo.mojo");
  EXPECT_EQ(contents, "content1");
  EXPECT_EQ(version, 1);
  EXPECT_EQ(gen, 1u);
}

TEST(DocumentDebouncerTest, CoalescesRapidUpdates) {
  TestCallback callback;
  constexpr auto delay = std::chrono::milliseconds(50);
  DocumentDebouncer debouncer(std::ref(callback), delay);

  auto uri = makeURI("/foo.mojo");

  // Send multiple rapid updates - only the last should be processed.
  debouncer.scheduleUpdate(uri, "content1", 1, /*generation=*/1);
  debouncer.scheduleUpdate(uri, "content2", 2, /*generation=*/1);
  debouncer.scheduleUpdate(uri, "content3", 3, /*generation=*/1);

  // Wait for callback.
  ASSERT_TRUE(callback.waitForCalls(1));

  // Give a bit more time to ensure no additional calls come through.
  std::this_thread::sleep_for(delay * 2);

  EXPECT_EQ(callback.callCount(), 1u);

  auto [path, contents, version, gen] = callback.getCall(0);
  EXPECT_EQ(path, "/foo.mojo");
  EXPECT_EQ(contents, "content3");
  EXPECT_EQ(version, 3);
}

TEST(DocumentDebouncerTest, IndependentDocuments) {
  TestCallback callback;
  constexpr auto delay = std::chrono::milliseconds(50);
  DocumentDebouncer debouncer(std::ref(callback), delay);

  auto uri1 = makeURI("/foo.mojo");
  auto uri2 = makeURI("/bar.mojo");

  // Schedule updates for two different documents.
  debouncer.scheduleUpdate(uri1, "content1", 1, /*generation=*/1);
  debouncer.scheduleUpdate(uri2, "content2", 2, /*generation=*/2);

  // Both should be processed independently.
  ASSERT_TRUE(callback.waitForCalls(2));
  EXPECT_EQ(callback.callCount(), 2u);
}

TEST(DocumentDebouncerTest, CancelPreventsCallback) {
  TestCallback callback;
  constexpr auto delay = std::chrono::milliseconds(100);
  DocumentDebouncer debouncer(std::ref(callback), delay);

  auto uri = makeURI("/foo.mojo");
  debouncer.scheduleUpdate(uri, "content1", 1, /*generation=*/1);

  // Cancel before the delay expires.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  debouncer.cancelUpdate(uri.file());

  // Wait longer than the delay.
  std::this_thread::sleep_for(delay * 2);

  // Callback should not have been invoked.
  EXPECT_EQ(callback.callCount(), 0u);
}

TEST(DocumentDebouncerTest, FlushProcessesImmediately) {
  TestCallback callback;
  constexpr auto delay = std::chrono::milliseconds(500); // Long delay.
  DocumentDebouncer debouncer(std::ref(callback), delay);

  auto uri = makeURI("/foo.mojo");
  debouncer.scheduleUpdate(uri, "content1", 1, /*generation=*/1);

  // Flush should process immediately without waiting for delay.
  debouncer.flush();

  EXPECT_EQ(callback.callCount(), 1u);

  auto [path, contents, version, gen] = callback.getCall(0);
  EXPECT_EQ(contents, "content1");
}

TEST(DocumentDebouncerTest, DestructorCleansUp) {
  TestCallback callback;
  constexpr auto delay = std::chrono::milliseconds(500);

  {
    DocumentDebouncer debouncer(std::ref(callback), delay);
    auto uri = makeURI("/foo.mojo");
    debouncer.scheduleUpdate(uri, "content1", 1, /*generation=*/1);
    // Destructor should clean up without hanging.
  }

  // If we get here without hanging, the test passes.
  // The callback may or may not have been called depending on timing.
  SUCCEED();
}

TEST(DocumentDebouncerTest, CustomDelay) {
  TestCallback callback;
  constexpr auto shortDelay = std::chrono::milliseconds(20);
  constexpr auto longDelay = std::chrono::milliseconds(200);

  // Test with short delay - should complete quickly.
  {
    DocumentDebouncer debouncer(std::ref(callback), shortDelay);
    auto uri = makeURI("/foo.mojo");

    auto start = std::chrono::steady_clock::now();
    debouncer.scheduleUpdate(uri, "content1", 1, /*generation=*/1);
    ASSERT_TRUE(callback.waitForCalls(1));
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should complete in roughly the delay time (with some tolerance).
    EXPECT_LT(elapsed, longDelay);
  }
}

TEST(DocumentDebouncerTest, ResetsTimerOnUpdate) {
  TestCallback callback;
  constexpr auto delay = std::chrono::milliseconds(80);
  DocumentDebouncer debouncer(std::ref(callback), delay);

  auto uri = makeURI("/foo.mojo");

  // Schedule initial update.
  debouncer.scheduleUpdate(uri, "content1", 1, /*generation=*/1);

  // Wait half the delay time.
  std::this_thread::sleep_for(delay / 2);
  EXPECT_EQ(callback.callCount(), 0u);

  // Update again - should reset the timer.
  debouncer.scheduleUpdate(uri, "content2", 2, /*generation=*/1);

  // Wait half the delay time again - still shouldn't fire.
  std::this_thread::sleep_for(delay / 2);
  EXPECT_EQ(callback.callCount(), 0u);

  // Now wait for full delay - should fire with content2.
  ASSERT_TRUE(callback.waitForCalls(1));

  auto [path, contents, version, gen] = callback.getCall(0);
  EXPECT_EQ(contents, "content2");
  EXPECT_EQ(version, 2);
}
