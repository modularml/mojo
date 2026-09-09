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

#ifndef SUPPORT_TELEMETRY_EXPORTERS_FILEMETRICEXPORTER_H
#define SUPPORT_TELEMETRY_EXPORTERS_FILEMETRICEXPORTER_H

#include "opentelemetry/exporters/ostream/metric_exporter.h"
#include "opentelemetry/sdk/common/exporter_utils.h"
#include "opentelemetry/sdk/metrics/export/metric_producer.h"
#include "opentelemetry/sdk/metrics/instruments.h"
#include "opentelemetry/sdk/metrics/push_metric_exporter.h"
#include <chrono>
#include <filesystem>
#include <sstream>
#include <utility>

namespace M::Telemetry::Exporter {

/// The FileMetricExporter exports metric data to a file, leveraging
/// OTel's OStreamMetricExporter.
class FileMetricExporter
    : public opentelemetry::sdk::metrics::PushMetricExporter {
public:
  explicit FileMetricExporter(
      std::filesystem::path filePath,
      opentelemetry::sdk::metrics::AggregationTemporality
          aggregation_temporality =
              opentelemetry::sdk::metrics::AggregationTemporality::kCumulative)
      : filePath(std::move(filePath)), ostreamExporter(outputStream) {}

  virtual ~FileMetricExporter() = default;

  /// Export metrics data.
  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::sdk::metrics::ResourceMetrics &data) noexcept
      override;

  /// Get the AggregationTemporality for the exporter.
  opentelemetry::sdk::metrics::AggregationTemporality GetAggregationTemporality(
      opentelemetry::sdk::metrics::InstrumentType instrument_type)
      const noexcept override {
    return ostreamExporter.GetAggregationTemporality(instrument_type);
  }

  /// Force flush the exporter.
  bool ForceFlush(std::chrono::microseconds timeout =
                      (std::chrono::microseconds::max)()) noexcept override {
    return ostreamExporter.ForceFlush(timeout);
  }

  /// Shut down the exporter, with optional timeout.
  bool Shutdown(std::chrono::microseconds timeout =
                    (std::chrono::microseconds::max)()) noexcept override {
    return ostreamExporter.Shutdown(timeout);
  }

private:
  /// Metrics are exported to this file.
  std::filesystem::path filePath;
  /// Buffer OTel's outputs in a string and flush it atomically to a file every
  /// time we export.
  std::stringstream outputStream;
  /// Delegate printing of telemetry data to OTel's OStreamMetricExporter.
  opentelemetry::exporter::metrics::OStreamMetricExporter ostreamExporter;
};

} // namespace M::Telemetry::Exporter

#endif // SUPPORT_TELEMETRY_EXPORTERS_FILEMETRICEXPORTER_H
