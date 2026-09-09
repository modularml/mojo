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

#include "LSPTelemetryContext.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace M::Mojo;
using namespace M::Mojo::LSP;

LSPTelemetryContext::LSPTelemetryContext(Telemetry::TelemetryContext &ctx)
    : responseTimeHistogram(ctx.createUInt64Histogram(
          "mojo.lsp.request.time", Telemetry::Level::L1,
          /*attributes=*/{},
          "Time it took to respond valid LSP requests that were effectively "
          "computed.",
          /*unit=*/"microsecond")),
      outdatedRequestCounter(ctx.createUInt64Counter(
          "mojo.lsp.request.outdated", Telemetry::Level::L1, /*attributes=*/{},
          "Number of outdated LSP requests.")),
      invalidRequestCounter(ctx.createUInt64Counter(
          "mojo.lsp.request.invalid", Telemetry::Level::L1, /*attributes=*/{},
          "Number of invalid LSP requests.")),
      ctx(ctx) {}

void LSPTelemetryContext::recordResponseTime(
    StringRef request, std::chrono::microseconds microseconds) {
  responseTimeHistogram.record(microseconds.count(),
                               {{"request", request.str()}});
}

void LSPTelemetryContext::recordInvalidRequest(StringRef request) {
  invalidRequestCounter.add(1, {{"request", request.str()}});
}

void LSPTelemetryContext::recordOutdatedRequest(StringRef request) {
  outdatedRequestCounter.add(1, {{"request", request.str()}});
}

void LSPTelemetryContext::reportInitialization(
    std::optional<StringRef> clientName) {
  ctx.getLogger("mojo")->emitL0Event(
      "lsp.initialized", {{"client_name", clientName.value_or("").str()}});
}

void LSPTelemetryContext::reportShutdown() {
  ctx.getLogger("mojo")->emitL0Event("lsp.shutdown");
}

void LSPTelemetryContext::flush() { ctx.flush(); }

void LSPTelemetryContext::recordParseTime(std::chrono::microseconds duration,
                                          size_t byteSize, bool notebook) {
  ctx.getLogger("mojo")->emitL1Event(
      "lsp.parse", {
                       {"duration", duration.count()},
                       {"size", byteSize},
                       {"documentType", notebook ? "notebook" : "text"},
                   });
}
