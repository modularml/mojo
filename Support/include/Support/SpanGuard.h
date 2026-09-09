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

// SpanGuard measures how long a scope takes and reports it as a pair of
// MLOG_KV records:
//
//   SpanGuard span("prefill");
//
//   event=span_start operation=prefill span_id=8027...
//   event=span_end   operation=prefill span_id=8027... duration_us=1423
//
// Downstream, the two records sharing a span_id are what lets Datadog Log
// Management render time-in-operation per batch. The end record is emitted
// from the destructor, so an early return still closes its span.

#ifndef SUPPORT_SPANGUARD_H
#define SUPPORT_SPANGUARD_H

#include "Log.h"

#include <chrono>
#include <cstdint>
#include <random>
#include <string_view>

namespace M::Log {

namespace Detail {

// Span ids only need to be distinct among the spans a reader is comparing, so
// each thread draws one random 64-bit base and counts up from it. That keeps
// generation off any shared cache line, which a process-wide atomic would not.
inline uint64_t nextSpanId() {
  static thread_local uint64_t counter = [] {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }();
  return counter++;
}

} // namespace Detail

class SpanGuard {
public:
  // `operation` must outlive the guard; call sites are expected to pass a
  // literal. It is stored rather than copied to keep construction allocation
  // free on the hot path.
  explicit SpanGuard(std::string_view operation)
      : operation(operation), spanId(Detail::nextSpanId()),
        start(std::chrono::steady_clock::now()) {
    MLOG_KV(LogLevel::INFO, "event", "span_start", "operation", operation,
            "span_id", spanId);
  }

  ~SpanGuard() {
    MLOG_KV(LogLevel::INFO, "event", "span_end", "operation", operation,
            "span_id", spanId, "duration_us", elapsedUs());
  }

  SpanGuard(const SpanGuard &) = delete;
  SpanGuard &operator=(const SpanGuard &) = delete;
  SpanGuard(SpanGuard &&) = delete;
  SpanGuard &operator=(SpanGuard &&) = delete;

  // Identifies the two records this guard emits. Callers that log inside the
  // scope can pass it as their own `span_id` to tie their records to the span.
  uint64_t getSpanId() const { return spanId; }

  int64_t elapsedUs() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
  }

private:
  std::string_view operation;
  uint64_t spanId;
  // Monotonic, so a wall-clock adjustment mid-span cannot produce a negative
  // or wildly wrong duration.
  std::chrono::steady_clock::time_point start;
};

} // namespace M::Log

#endif // SUPPORT_SPANGUARD_H
