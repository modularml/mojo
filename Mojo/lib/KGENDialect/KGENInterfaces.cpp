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

#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/OperationUtils.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

LogicalResult impl::verifyGeneratorUser(GeneratorUserOpInterface op) {
  if (!op.getCallee())
    return success();

  // Disallow calls from within a concrete function from calling anything with
  // input or output parameters.
  if (auto func = op->getParentOfType<FuncOp>()) {
    auto symbolCst = dyn_cast<SymbolConstantAttr>(op.getCallee());
    if (!symbolCst || !symbolCst.getParamValues().empty()) {
      return op.emitOpError("cannot reference generator with input parameters "
                            "from within a concrete 'kgen.func'")
                 .attachNote(func.getLoc())
             << "within 'kgen.func' @" << func.getName();
    }

    if (!op.isAllowedInFunc())
      return op.emitOpError("is only allowed in generators pre-elaboration");
  }

  return success();
}

Operation *impl::lookupConformance(Operation *op, ParameterEvaluator &evaluator,
                                   TraitSymbolAttr traitSymbol) {
  assert(op->getRegions().size() == 1 && "expected an one-region ops");
  for (ConformanceOp conformance : op->getRegion(0).getOps<ConformanceOp>()) {
    if (evaluator.replace(conformance.getTraitSymbol()) == traitSymbol)
      return conformance;
  }
  return nullptr;
}
Operation *lookupConformance(Operation *op, ParameterEvaluator &evaluator,
                             TraitSymbolAttr traitSymbol);

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENInterfaces.cpp.inc"
