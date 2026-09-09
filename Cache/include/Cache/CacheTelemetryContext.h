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

#ifndef CACHE_CACHE_TELEMETRY_CONTEXT_H
#define CACHE_CACHE_TELEMETRY_CONTEXT_H

#include "Support/Context.h"
#include "Support/Telemetry/Instruments.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <functional>
#include <string>

namespace M::AsyncRT {
class CPUDevice;
}

namespace M::Cache {
/// Utility that manages the objects used to perform telemetry related to the
/// KGEN compiler.
class CacheTelemetryContext {
public:
  CacheTelemetryContext(Telemetry::TelemetryContext &ctx);

  static CacheTelemetryContext &getCacheTelemetryContext(ContextRef context);
  static CacheTelemetryContext &getCacheTelemetryContext(Context *context);

  /// Record a cache hit event.
  void recordCacheHit(llvm::StringRef pipelineName);

  /// Record a cache miss event.
  void recordCacheMiss(llvm::StringRef pipelineName);

  static std::function<void(mlir::Operation *)> getTelemetryOnMissLambda(
      const std::string &counterName, const std::string &timerName,
      const llvm::StringMap<M::Telemetry::MetricAttributeValue> &attributes =
          {});

  static std::function<void(mlir::Operation *)>
  getTelemetryOnHitLambda(const std::string &counterName);

private:
  Telemetry::Counter<uint64_t> cacheHitCounter;
  Telemetry::Counter<uint64_t> cacheMissCounter;
};

} // namespace M::Cache

#endif // CACHE_CACHE_TELEMETRY_CONTEXT_H
