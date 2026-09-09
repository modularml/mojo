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

#ifndef SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOINTERFACES_H
#define SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOINTERFACES_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "mlir/IR/OpDefinition.h"

namespace M::DebugInfo {
class SubprogramScoped;
class InlinedSubprogramScoped;

/// Return true if constants should be materialized into a subprogram scoped
/// region.
bool shouldMaterializeConstantsInto(Region &region);

namespace Impl {
LogicalResult verifySubprogramScoped(SubprogramScoped op);

Location getLocNoInlined(InlinedSubprogramScoped iss);
LocationAttr getCallLocAttr(InlinedSubprogramScoped iss);
void setCallLocAttr(InlinedSubprogramScoped iss, LocationAttr attr);
} // namespace Impl
} // namespace M::DebugInfo

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h.inc"

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOINTERFACES_H
