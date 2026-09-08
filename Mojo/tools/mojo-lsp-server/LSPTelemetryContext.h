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

#ifndef KGEN_LIB_MOJO_LSP_LSPTELEMETRYCONTEXT_H
#define KGEN_LIB_MOJO_LSP_LSPTELEMETRYCONTEXT_H

#include "Support/Telemetry/Telemetry.h"

namespace M::AsyncRT {
class CPUDevice;
}

namespace M::Mojo::LSP {
/// Utility that manages the objects used to perform telemetry related to the
/// LSP.
class LSPTelemetryContext {
public:
  LSPTelemetryContext(Telemetry::TelemetryContext &ctx);

  /// Record a metric after a successful response of a specific request.
  void recordResponseTime(StringRef request,
                          std::chrono::microseconds microseconds);

  /// Record a metric signaling a request with an invalid input.
  void recordInvalidRequest(StringRef request);

  /// Record a metric signaling a request that has gone stale, e.g. the
  /// document has changed while the request was in the queue.
  void recordOutdatedRequest(StringRef request);

  /// Report initialization of the server, including the name the client
  /// reported, if provided.
  void reportInitialization(std::optional<StringRef> clientName);

  /// Report an orderly shutdown of the server. This is not reported if there is
  /// a crash.
  void reportShutdown();

  /// Flushes the underlying telemetry context.
  void flush();

  /// Record parse time, size in bytes, and whether the document is a notebook
  /// or not.
  void recordParseTime(std::chrono::microseconds duration, size_t byteSize,
                       bool notebook);

private:
  Telemetry::Histogram<uint64_t> responseTimeHistogram;
  Telemetry::Counter<uint64_t> outdatedRequestCounter;
  Telemetry::Counter<uint64_t> invalidRequestCounter;

  Telemetry::TelemetryContext &ctx;
};
} // namespace M::Mojo::LSP

#endif // KGEN_LIB_MOJO_LSP_LSPTELEMETRYCONTEXT_H
