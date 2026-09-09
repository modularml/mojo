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

#include "Support/LogFFI.h"
#include "Support/Log.h"
#include "Support/SymbolExport.h"

#include <array>
#include <chrono>
#include <cstring>

using namespace M::Log;

static_assert(std::is_trivially_copyable_v<LogArg>);
static_assert(Channel::Mojo == 1,
              "Mojo logging implementation hardcodes Mojo channel to 1");

MODULAR_EXPORT int64_t MLog_now(void) {
  return std::chrono::system_clock::now().time_since_epoch().count();
}

MODULAR_EXPORT uint8_t MLog_get_level(void) {
  return static_cast<uint8_t>(getDefaultLog().getLogLevel());
}

MODULAR_EXPORT void MLog_set_level(uint8_t level) {
  getDefaultLog().setLogLevel(static_cast<LogLevel>(level));
}

MODULAR_EXPORT void MLog_flush(void) { getDefaultLog().flush(); }

MODULAR_EXPORT void MLog_write(uint8_t level, uint64_t channel,
                               int64_t timestamp, const char *fmt,
                               size_t fmtLen, const void *args,
                               uint8_t argCount) {
  auto &log = getDefaultLog();
  if (log.getLogLevel() > static_cast<LogLevel>(level) ||
      !log.isEnabled(static_cast<Channel::Channels>(channel)))
    return;

  std::array<LogArg, LogRecord::maxArgs> argArr{};
  std::memcpy(argArr.data(), args, argCount * sizeof(LogArg));

  LogRecord::Timestamp ts{LogRecord::Timestamp::clock::duration(timestamp)};
  LogRecord record(ts, static_cast<LogLevel>(level),
                   static_cast<Channel::Channels>(channel), {fmt, fmtLen},
                   std::move(argArr), argCount);
  logWrite(log, std::move(record));
}
