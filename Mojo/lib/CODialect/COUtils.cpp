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

#include "Mojo/CODialect/COUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;

ParseResult KGEN::parseCoroutineTypes(AsmParser &p, TypeArrayAttr &typeAttr) {
  SmallVector<Type> types;
  if (succeeded(p.parseOptionalColon()) && p.parseCommaSeparatedList([&] {
        return p.parseType(types.emplace_back());
      }))
    return failure();
  typeAttr = TypeArrayAttr::get(p.getContext(), types);
  return success();
}

void KGEN::printCoroutineTypes(AsmPrinter &p, Operation *op,
                               TypeArrayAttr types) {
  if (types.empty())
    return;
  p << " : ";
  llvm::interleaveComma(types, p);
}
