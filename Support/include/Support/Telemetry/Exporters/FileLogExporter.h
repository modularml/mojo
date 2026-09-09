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

#ifndef SUPPORT_TELEMETRY_EXPORTERS_FILELOGEXPORTER_H
#define SUPPORT_TELEMETRY_EXPORTERS_FILELOGEXPORTER_H

#include "opentelemetry/exporters/ostream/log_record_exporter.h"
#include "opentelemetry/nostd/span.h"
#include "opentelemetry/sdk/common/exporter_utils.h"
#include "opentelemetry/sdk/logs/exporter.h"
#include "opentelemetry/sdk/logs/recordable.h"
#include <chrono>
#include <filesystem>
#include <memory>
#include <sstream>
#include <utility>

namespace M::Telemetry::Exporter {

/// The FileLogExporter exports log data to a file, leveraging
/// OTel's OStreamLogRecordExporter.
class FileLogExporter : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  explicit FileLogExporter(std::filesystem::path filePath)
      : filePath(std::move(filePath)), ostreamExporter(outputStream) {}

  virtual ~FileLogExporter() = default;

  std::unique_ptr<opentelemetry::sdk::logs::Recordable>
  MakeRecordable() noexcept override {
    return ostreamExporter.MakeRecordable();
  }

  /// Exports a span of logs sent from the processor to a file.
  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::nostd::span<std::unique_ptr<
             opentelemetry::sdk::logs::Recordable>> &records) noexcept override;

  /// Force flush the exporter.
  bool ForceFlush(std::chrono::microseconds timeout =
                      std::chrono::microseconds::max()) noexcept override {
    return ostreamExporter.ForceFlush(timeout);
  }

  /// Shut down the exporter, with optional timeout.
  bool Shutdown(std::chrono::microseconds timeout =
                    std::chrono::microseconds::max()) noexcept override {
    return ostreamExporter.Shutdown(timeout);
  }

private:
  /// Logs are exported to this file.
  std::filesystem::path filePath;
  /// Buffer OTel's outputs in a string and flush it atomically to a file every
  /// time we export.
  std::stringstream outputStream;
  /// Delegate printing of telemetry data to OTel's OStreamLogRecordExporter.
  opentelemetry::exporter::logs::OStreamLogRecordExporter ostreamExporter;
};

} // namespace M::Telemetry::Exporter

#endif // SUPPORT_TELEMETRY_EXPORTERS_FILELOGEXPORTER_H
