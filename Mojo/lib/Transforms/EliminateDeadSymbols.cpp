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

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/EliminateDeadSymbolUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_ELIMINATEDEADSYMBOLS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct EliminateDeadSymbolsPass
    : M::KGEN::impl::EliminateDeadSymbolsBase<EliminateDeadSymbolsPass> {
  void runOnOperation() override;
};
} // namespace

void EliminateDeadSymbolsPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();

  DenseSet<StringAttr> usedSymbols = KGEN::getUsedSymbols(analysis, theModule);
  // OK, we have all the used symbols. Now, just erase ones that aren't in
  // there.
  unsigned numErased = 0;
  for (auto sym : llvm::make_early_inc_range(
           theModule.getOps<mlir::SymbolOpInterface>())) {
    if (!usedSymbols.contains(sym.getNameAttr())) {
      analysis.getTopLevelSymbolTable().erase(sym);
      ++numErased;
    }
  }
  this->numErased = numErased;
}
