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

#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include <cstdint>
#include <utility>

using namespace M;

/// Utility function to insert a new asm blob into the context.
static DenseResourceElementsAttr insertAsmBlob(MLIRContext *ctx,
                                               RankedTensorType attrType,
                                               const Twine &name,
                                               mlir::AsmResourceBlob blob) {
  using HandleTy = mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>;
  auto resourceManager =
      mlir::DenseResourceElementsHandle::getManagerInterface(ctx);
  auto *dialect = cast<mlir::BuiltinDialect>(resourceManager.getDialect());
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.getBlobManager().insert<HandleTy>(
                    dialect, name.str(), std::move(blob)));
}

DenseResourceElementsAttr M::createResourceAttr(MLIRContext *ctx,
                                                ArrayRef<char> data,
                                                const Twine &name) {
  // Pretend this is a "tensor" of data.
  auto attrType = RankedTensorType::get(
      {(int64_t)data.size()}, IntegerType::get(ctx, 8, IntegerType::Unsigned));
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
      ArrayRef<char>(data.begin(), data.size()),
      /*align=*/kAsmResourceBlobAlignment);
  return insertAsmBlob(ctx, attrType, name, std::move(blob));
}

DenseResourceElementsAttr
M::createResourceAttr(MLIRContext *ctx, BufferRef data, const Twine &name) {
  // Pretend this is a "tensor" of data.
  auto attrType =
      RankedTensorType::get({(int64_t)data->getBufferSize()},
                            IntegerType::get(ctx, 8, IntegerType::Unsigned));

  // Build an unmanaged blob to represent the data, using a deleter that holds
  // on to the reference (to avoid copying).
  StringRef buffer = data->getBuffer();
  auto blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
      ArrayRef<char>(buffer.data(), buffer.size()),
      /*align=*/kAsmResourceBlobAlignment,
      [data = std::move(data)](void *, unsigned, unsigned) {});
  return insertAsmBlob(ctx, attrType, name, std::move(blob));
}
