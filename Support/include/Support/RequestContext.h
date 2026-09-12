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

// Which request the calling thread is currently serving, for telemetry that
// wants to attribute its output to one request without every call site passing
// the id along.
//
//   {
//     M::Request::BatchScope batch(batchId);
//     runForwardPass();
//   }
//
// This knows nothing about logging, and nothing about it is in the logger:
// metrics and traces want the same answer, and code that only logs should not
// have to link it. RequestLog.h is the layer that joins the two.
//
// The context is thread-local and does not cross a dispatch boundary, so work
// fanned out to a thread pool does not inherit it.

#ifndef SUPPORT_REQUESTCONTEXT_H
#define SUPPORT_REQUESTCONTEXT_H

#include <cstdint>
#include <optional>

namespace M::Request {

namespace Detail {
inline thread_local std::optional<int64_t> batchId;
} // namespace Detail

// The batch this thread is serving, or nullptr outside one. Callers that render
// it are expected to omit their field entirely when this is null rather than
// emit a sentinel, so that a reader can tell "no batch" from "batch 0".
inline const int64_t *currentBatchId() {
  return Detail::batchId ? &*Detail::batchId : nullptr;
}

// Binds the batch id for a scope and restores what the thread held before, so
// scopes nest and leaving an inner one cannot strand an outer one's id.
class BatchScope {
public:
  explicit BatchScope(int64_t batchId) : saved(Detail::batchId) {
    Detail::batchId = batchId;
  }

  ~BatchScope() { Detail::batchId = saved; }

  BatchScope(const BatchScope &) = delete;
  BatchScope &operator=(const BatchScope &) = delete;
  BatchScope(BatchScope &&) = delete;
  BatchScope &operator=(BatchScope &&) = delete;

private:
  std::optional<int64_t> saved;
};

// For boundaries a scope object cannot span, such as the Python bindings, where
// the two ends are separate calls. Prefer BatchScope anywhere it fits: it
// cannot be left unbalanced.
inline void setBatchId(int64_t id) { Detail::batchId = id; }
inline void clearBatchId() { Detail::batchId.reset(); }

} // namespace M::Request

#endif // SUPPORT_REQUESTCONTEXT_H
