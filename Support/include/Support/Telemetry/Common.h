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

#ifndef SUPPORT_TELEMETRY_COMMON_H
#define SUPPORT_TELEMETRY_COMMON_H

#include <cstdint>

namespace M::Telemetry {

/// Telemetry levels. We emit more information with increasing telemetry level,
/// such that if the configured telemetry level is X, we emit all signals
/// (metrics, logs) tagged with level <= X.
enum class Level : uint8_t { L0, L1, L2, USER = 255 };

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_COMMON_H
