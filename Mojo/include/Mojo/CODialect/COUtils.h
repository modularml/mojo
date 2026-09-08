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

#ifndef KGEN_CODIALECT_COUTILS_H
#define KGEN_CODIALECT_COUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"

namespace M {
class TypeArrayAttr;
namespace KGEN {
ParseResult parseCoroutineTypes(AsmParser &p, TypeArrayAttr &typeAttr);
void printCoroutineTypes(AsmPrinter &p, Operation *op, TypeArrayAttr types);
} // namespace KGEN
} // namespace M

#endif // KGEN_CODIALECT_COUTILS_H
