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

#include "Support/Config.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

void M::configureMLIRContext(mlir::MLIRContext &ctx) {
#ifdef MODULAR_PRODUCTION
  ctx.printOpOnDiagnostic(false);
#endif
}

void M::configurePassManager(mlir::PassManager &mgr) {
#ifdef MODULAR_PRODUCTION
  mgr.enableVerifier(false);
#endif
}
