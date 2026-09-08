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

#include "Mojo/HLCFDialect/HLCFAttrs.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"

using namespace M;
using namespace HLCF;

//===----------------------------------------------------------------------===//
// UnrollLevel
//===----------------------------------------------------------------------===//

llvm::hash_code UnrollLevel::hash() const { return llvm::hash_value(value); }

static ParseResult parseUnrollLevel(AsmParser &p,
                                    FailureOr<UnrollLevel> &unrollLevel) {
  if (succeeded(p.parseOptionalKeyword("none"))) {
    unrollLevel = UnrollLevel::none();
    return success();
  }
  if (succeeded(p.parseOptionalKeyword("full"))) {
    unrollLevel = UnrollLevel::full();
    return success();
  }
  llvm::SMLoc loc = p.getCurrentLocation();
  int32_t value = 0;
  OptionalParseResult result = p.parseOptionalInteger(value);
  if (!result.has_value())
    return p.emitError(loc, "expected 'none', 'full', or a positive integer");
  if (result.has_value() && failed(*result))
    return failure();
  if (value <= 0)
    return p.emitError(loc, "expected a positive integer for unroll factor");
  unrollLevel = UnrollLevel(value);
  return success();
}

static void printUnrollLevel(AsmPrinter &p, UnrollLevel level) {
  if (level.isNone())
    p << "none";
  else if (level.isFull())
    p << "full";
  else
    p << level.getFactor();
}

namespace M::HLCF {
static llvm::hash_code hash_value(UnrollLevel level) { return level.hash(); }

UnrollLevelAttr UnrollLevelAttr::getNone(MLIRContext *context) {
  return UnrollLevelAttr::get(context, 0);
}

UnrollLevelAttr UnrollLevelAttr::getFull(MLIRContext *context) {
  return UnrollLevelAttr::get(context, -1);
}

} // namespace M::HLCF

//===----------------------------------------------------------------------===//
// HLCFDialect
//===----------------------------------------------------------------------===//

void HLCFDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Mojo/HLCFDialect/HLCFAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Mojo/HLCFDialect/HLCFAttrs.cpp.inc"
