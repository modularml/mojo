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

#include "Support/Compiler/DiagnosticHandler.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Threading.h"
#include <utility>

using namespace M;

DiagnosticHandler::DiagnosticHandler(MLIRContext *ctx, bool capturePerThread)
    : ctx(ctx), capturePerThread(capturePerThread) {
  threadID = llvm::get_threadid();
  handlerID = ctx->getDiagEngine().registerHandler([this](Diagnostic &diag) {
    if (!this->capturePerThread || this->threadID == llvm::get_threadid()) {
      diagnostics.push_back(std::move(diag));
      return mlir::success();
    }
    return mlir::failure();
  });
}

DiagnosticHandler::~DiagnosticHandler() { release(); }

void DiagnosticHandler::release() {
  ctx->getDiagEngine().eraseHandler(handlerID);
}

void DiagnosticHandler::emitDiagnostics(
    function_ref<void(Diagnostic &)> emitFn) {
  for (Diagnostic &diag : diagnostics)
    emitFn(diag);
}

mlir::DiagnosticEngine::HandlerID DiagnosticHandler::getHandlerID() {
  return handlerID;
}
