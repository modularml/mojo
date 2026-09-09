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

#ifndef KGEN_LITDIALECT_LITATTRS_H
#define KGEN_LITDIALECT_LITATTRS_H

#include "Mojo/KGENDialect/KGENAttrInterfaces.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Regex.h"

namespace M::KGEN {
class NoneType;
namespace LIT {
class ModuleType;
class StructMetaType;
class StructType;
class OriginType;
class OriginSetType;
class RefPackType;
class TraitType;
class StructFieldOp;
class FnMetaOriginDataAttr;
} // namespace LIT
} // namespace M::KGEN

#include "Mojo/LITDialect/LITEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Mojo/LITDialect/LITAttrs.h.inc"

namespace M::KGEN::LIT {

/// Given a list of operations, create an array of bools (as a mask) indicating
/// variadic parameters in their concatenated list of parameter declarations.
/// The given operations must all implement DeclInterface.
SmallVector<VariadicKind>
getContextualVariadicParams(ArrayRef<Operation *> ops);

/// This digs in and unpacks all of the origin references in the specified
/// TypedAttr, unpacking unions.
///
/// This invokes the specified closure on each origin element.
template <typename T>
static inline void processOriginUnionElts(TypedAttr origin, T &&fn) {
  if (auto sugar = dyn_cast<SugarAttr>(origin))
    origin = sugar.getCanonical();

  // Expand origin unions into their members, we know they will canonicalize
  // nested unions into a single one.
  if (auto unionAttr = dyn_cast<OriginUnionAttr>(origin)) {
    for (auto elt : unionAttr.getOperands())
      fn(elt);
    return;
  }

  fn(origin);
}

} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITATTRS_H
