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

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_NVPTXADAPTER_H
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_NVPTXADAPTER_H

#include "TargetAdapter.h"

namespace M::DebugInfo {
/// Adapter for NVPTX backend.
TargetAdapter getNVPTXAdapter(bool tradeoffPerfForVariableDI);
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_NVPTXADAPTER_H
