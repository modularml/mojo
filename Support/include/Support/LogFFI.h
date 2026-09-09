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

// C interface to the logging library, intended for consumption via
// external_call from Mojo. The caller is responsible for serializing
// arguments into a LogArg array (mirroring M::Log::LogArg's layout) and
// passing it alongside the other record fields. LogFFI.cpp constructs the
// LogRecord and dispatches to the logger.

#ifndef SUPPORT_LOG_C_H
#define SUPPORT_LOG_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the current time as a system_clock tick count. Call this at the
/// log site to capture the timestamp before argument serialization.
int64_t MLog_now(void);

/// Returns the current minimum log level as a uint8_t. Use this for an
/// early-out before serializing arguments.
uint8_t MLog_get_level(void);

/// Constructs a LogRecord from the provided fields and submits it to the
/// default logger. args must point to argCount elements whose layout matches
/// M::Log::LogArg (verified by static_asserts in LogFFI.cpp). timestamp
/// must be a value returned from MLog_now to reconstruct time values.
void MLog_write(uint8_t level, uint64_t channel, int64_t timestamp,
                const char *fmt, size_t fmtLen, const void *args,
                uint8_t argCount);

/// Sets the minimum log level on the default logger.
void MLog_set_level(uint8_t level);

/// Blocks until all records enqueued before this call have been written to all
/// sinks. Use in tests before reading sink output.
void MLog_flush(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SUPPORT_LOG_C_H
