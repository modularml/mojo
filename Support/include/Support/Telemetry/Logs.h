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

#ifndef SUPPORT_TELEMETRY_LOGS_H
#define SUPPORT_TELEMETRY_LOGS_H

#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Common.h"
#include "Support/Telemetry/ForwardDecls.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/logs/event_logger.h"
#include "opentelemetry/logs/severity.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

namespace M::Telemetry::Logs {

/// Severity levels for logs.
/// See
/// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/data-model.md#field-severitynumber

using Severity = opentelemetry::logs::Severity;
using AttributeValue = opentelemetry::common::AttributeValue;

/// A Logger to emit logs. Logger's methods are thread-safe.
/// Usage examples:
/// - logger->getError("my-event") << "my log error message";
/// - logger->emitEvent("my-event", Severity::kError, "my log error message");
/// - logger->emitEvent("billing", Severity::kInfo);
class Logger : public std::enable_shared_from_this<Logger> {
public:
  virtual ~Logger() = default;

  virtual void
  emitL0Event(StringRef eventName,
              const llvm::StringMap<AttributeValue> &attributes = {}) {
    return emitEvent(eventName, Severity::kInfo, M::Telemetry::Level::L0,
                     attributes);
  }
  virtual void
  emitL1Event(StringRef eventName,
              const llvm::StringMap<AttributeValue> &attributes = {}) {
    return emitEvent(eventName, Severity::kInfo, M::Telemetry::Level::L1,
                     attributes);
  }
  virtual void
  emitL2Event(StringRef eventName,
              const llvm::StringMap<AttributeValue> &attributes = {}) {
    return emitEvent(eventName, Severity::kInfo, M::Telemetry::Level::L2,
                     attributes);
  }

  /// Returns true if an event will be emitted based on its level and the
  /// configured telemetry level.
  bool eventEnabled(Level eventLevel) const {
    return eventLevel <= telemetryLevel;
  }

protected:
  Logger(std::shared_ptr<opentelemetry::logs::EventLogger> logger,
         M::Telemetry::Level level)
      : logger(std::move(logger)), telemetryLevel(level) {}

  std::shared_ptr<opentelemetry::logs::EventLogger> logger;

private:
  friend class M::Telemetry::TelemetryContext;

  /// Emit event with given name, severity, body and attributes.
  void emitEvent(StringRef eventName, Severity severity,
                 M::Telemetry::Level level,
                 const llvm::StringMap<AttributeValue> &attributes) {
    if (eventEnabled(level)) {
      // Convert the attributes to unordered_map to pass to OTel.
      std::unordered_map<std::string, AttributeValue> attrs;
      for (auto &attr : attributes) {
        std::visit([&](auto v) { attrs[attr.first().str()] = v; }, attr.second);
      }
      // Use structured attributes rather than unstructured body
      logger->EmitEvent(eventName,
                        static_cast<opentelemetry::logs::Severity>(severity),
                        "", attrs);
    }
  }

  // Configured level for Telemetry.
  M::Telemetry::Level telemetryLevel;
};

} // namespace M::Telemetry::Logs

#endif // SUPPORT_TELEMETRY_LOGS_H
