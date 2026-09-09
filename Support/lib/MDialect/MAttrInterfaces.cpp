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

#include "Support/MDialect/MAttrInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "Support/MDialect/MTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstdint>

using namespace M;

//===----------------------------------------------------------------------===//
// HasAlignedBytesInterface for DenseResourceElementsAttr
//===----------------------------------------------------------------------===//

namespace {
struct DenseResourceElementsAttrHasAlignedBytesInterface
    : public HasAlignedBytesInterface::ExternalModel<
          DenseResourceElementsAttrHasAlignedBytesInterface,
          mlir::DenseResourceElementsAttr> {
  AlignedBytesType getAlignedBytesType(Attribute attr) const {
    auto denseAttr = dyn_cast<mlir::DenseResourceElementsAttr>(attr);
    if (auto alignedBytesType =
            llvm::dyn_cast<AlignedBytesType>(denseAttr.getType()))
      return alignedBytesType;
    mlir::AsmResourceBlob *blob = denseAttr.getRawHandle().getBlob();
    assert(blob && "dense_resource has not been initialized");
    uint64_t size = static_cast<uint64_t>(blob->getData().size());
    uint64_t align = static_cast<uint64_t>(blob->getDataAlignment());
    return AlignedBytesType::get(denseAttr.getContext(), size, align);
  }
};
} // namespace

void MDialect::injectAttrInterfaces() {
  mlir::DenseResourceElementsAttr::attachInterface<
      DenseResourceElementsAttrHasAlignedBytesInterface>(*getContext());
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MAttrInterfaces.cpp.inc"
