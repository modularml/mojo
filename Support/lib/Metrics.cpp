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

#include "Support/Metrics.h"

namespace M::Metrics {

CollatedMetrics MetricsCollector::collect() {
  CollatedMetrics result;
  std::lock_guard<std::mutex> guard(mu);
  for (auto &e : counters)
    result.counters.push_back({e.getKey(), e.second.read()});
  for (auto &e : gauges)
    result.gauges.push_back({e.getKey(), e.second.read()});
  for (auto &e : histograms)
    result.histograms.push_back({e.getKey(), e.second.read()});
  return result;
}

Counter &MetricsCollector::registerCounter(llvm::StringRef name) {
  std::lock_guard<std::mutex> guard(mu);
  return counters.try_emplace(name).first->second;
}

Gauge &MetricsCollector::registerGauge(llvm::StringRef name) {
  std::lock_guard<std::mutex> guard(mu);
  return gauges.try_emplace(name).first->second;
}

Histogram &MetricsCollector::registerHistogram(llvm::StringRef name) {
  std::lock_guard<std::mutex> guard(mu);
  return histograms.try_emplace(name).first->second;
}

} // namespace M::Metrics
