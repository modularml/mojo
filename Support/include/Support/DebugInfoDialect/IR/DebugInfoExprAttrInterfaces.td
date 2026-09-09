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

#ifndef DEBUGINFOEXPRINTERFACES_TD
#define DEBUGINFOEXPRINTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// DIExprAttrInterface
//===----------------------------------------------------------------------===//

/// The base class of all debug info expression attributes. Expression
/// attributes are all typed attributes.
def DebugInfo_DIExprAttrInterface
  : AttrInterface<"DIExprAttr", [TypedAttrInterface]> {
  let description = [{
    This interface is used for attributes that represent DebugInfo expression
    operators.

    Each operator is typed so that dialects may type-check DI expressions.
    However, since neither LLVM nor DWARF expects typed DI expressions, these
    types will only be used for legalization during compilation, and are
    discarded during LLVM lowering. Thus, the type of an expression operator is
    not expected to be a DIType. This also allows dialects to use their native
    types to track the optimizations their passes performed on them.
  }];
  let cppNamespace = "::M::DebugInfo";
}

#endif // DEBUGINFOEXPRINTERFACES_TD
