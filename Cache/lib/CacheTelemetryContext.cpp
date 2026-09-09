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

#include "Cache/CacheTelemetryContext.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "Support/Context.h"
#include "Support/Telemetry/Common.h"
#include "Support/Telemetry/Instruments.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <functional>
#include <string>

using namespace M;
using namespace M::Cache;

CacheTelemetryContext::CacheTelemetryContext(Telemetry::TelemetryContext &ctx)
    : cacheHitCounter(ctx.createUInt64Counter(
          "mojo.compile.cache.hit", Telemetry::Level::L2,
          /*attributes=*/{}, "Number of compilation cache hits.")),
      cacheMissCounter(ctx.createUInt64Counter(
          "mojo.compile.cache.miss", Telemetry::Level::L2,
          /*attributes=*/{}, "Number of compilation cache misses.")) {}

CacheTelemetryContext &
CacheTelemetryContext::getCacheTelemetryContext(Context *context) {
  auto &telemetryCtx = *context->get<M::Telemetry::TelemetryContext>();
  return context->emplaceIfMissing<CacheTelemetryContext>(telemetryCtx);
}

CacheTelemetryContext &
CacheTelemetryContext::getCacheTelemetryContext(ContextRef context) {
  return getCacheTelemetryContext(context.getPointer());
}

void CacheTelemetryContext::recordCacheHit(llvm::StringRef pipelineName) {
  cacheHitCounter.add(1, {{"pipeline", pipelineName.str()}});
}

void CacheTelemetryContext::recordCacheMiss(llvm::StringRef pipelineName) {
  cacheMissCounter.add(1, {{"pipeline", pipelineName.str()}});
}

std::function<void(mlir::Operation *)>
CacheTelemetryContext::getTelemetryOnMissLambda(
    const std::string &counterName, const std::string &timerName,
    const llvm::StringMap<M::Telemetry::MetricAttributeValue> &attrs) {
  return [counterName, timerName, attrs](mlir::Operation *op) {
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheMiss(counterName);

    [[maybe_unused]] auto timeScope =
        loadContext(op->getContext())
            ->get<M::Telemetry::TelemetryContext>()
            ->createUInt64Timer<std::chrono::milliseconds>(
                timerName, M::Telemetry::Level::L2, attrs);
  };
}

std::function<void(mlir::Operation *)>
CacheTelemetryContext::getTelemetryOnHitLambda(const std::string &counterName) {
  return [counterName](mlir::Operation *op) {
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheHit(counterName);
  };
}
