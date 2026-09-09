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

#ifndef SUPPORT_COMPILER_MLIRDENSEATTR_H
#define SUPPORT_COMPILER_MLIRDENSEATTR_H

#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include <cstddef>

namespace M {

constexpr size_t kAsmResourceBlobAlignment = 8;

/// Returns true if an array with the given number of elements is sufficiently
/// large that out-of-line storage should be used. This indicates to the caller
/// that the data is big enough to treat specially, e.g. that it shouldn't be
/// stored in the MLIRContext, folded unconditionally, etc.
inline bool shouldUseOutOfLineAttrStorage(size_t numElements) {
  // A sufficiently large element threshold is used to avoid treating large
  // arrays as "free". The storage, constant folding, etc. of large arrays
  // should be treated specially to ensure we don't bloat generated code, memory
  // use, and more.
  static constexpr size_t kLargeDataThreshold = 512;
  return numElements > kLargeDataThreshold;
}

/// Returns an attribute with the given `name` that represents the serialized
/// `data`. The data is always copied into the MLIR context.
DenseResourceElementsAttr
createResourceAttr(MLIRContext *ctx, ArrayRef<char> data, const Twine &name);

/// Returns an attribute with the given `name` that represents the serialized
/// `data`.
DenseResourceElementsAttr createResourceAttr(MLIRContext *ctx, BufferRef data,
                                             const Twine &name);
} // namespace M

#endif // SUPPORT_COMPILER_MLIRDENSEATTR_H
