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

#ifndef SUPPORT_COMPILER_ATTRBASE_TD
#define SUPPORT_COMPILER_ATTRBASE_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"

class AttrOfType<Type type> : ConfinedAttr<TypedAttrInterface, [
  AttrConstraint<SubstLeaves<"$_self", "llvm::cast<TypedAttr>($_self).getType()",
                             type.predicate>, type.summary>]>;

#endif // SUPPORT_COMPILER_ATTRBASE_TD
