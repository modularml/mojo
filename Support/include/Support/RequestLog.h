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

// MLOG_KV_REQ logs a key-value record and, when the calling thread is inside a
// request, appends that request's batch id as one more pair:
//
//   MLOG_KV_REQ(LogLevel::INFO, "event", "kv_evict");
//
//   event=kv_evict batch_id=42    inside a M::Request::BatchScope
//   event=kv_evict                outside one
//
// MLOG_KV takes at most four pairs, so this takes at most three. Exceeding that
// fails to compile with MLOG_KV's own four-pair diagnostic.

#ifndef SUPPORT_REQUESTLOG_H
#define SUPPORT_REQUESTLOG_H

#include "Log.h"
#include "RequestContext.h"

#define MLOG_KV_REQ(level, ...)                                                \
  do {                                                                         \
    if (const int64_t *mlogReqBatchId = ::M::Request::currentBatchId())        \
      MLOG_KV(level, __VA_ARGS__, "batch_id", *mlogReqBatchId);                \
    else                                                                       \
      MLOG_KV(level, __VA_ARGS__);                                             \
  } while (0)

#endif // SUPPORT_REQUESTLOG_H
