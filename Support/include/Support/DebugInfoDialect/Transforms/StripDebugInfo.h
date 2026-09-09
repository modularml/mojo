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
//
// This file provides support for stripping debug information from IR.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_TRANSFORMS_STRIPDEBUGINFO_H
#define SUPPORT_DEBUGINFODIALECT_TRANSFORMS_STRIPDEBUGINFO_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::DebugInfo {

/// Parse a source file contained within the given source manager, and attach
/// artificial debug information that describes the input IR.
void stripDebugInfo(Operation *scope, bool preserveLineTables);

} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_TRANSFORMS_STRIPDEBUGINFO_H
