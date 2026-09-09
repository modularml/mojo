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

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DEBUGINFOTOLLVM_H
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DEBUGINFOTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
#define GEN_PASS_DECL_DEBUGINFOTOLLVM
#define GEN_PASS_REGISTRATION
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h.inc"
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DEBUGINFOTOLLVM_H
