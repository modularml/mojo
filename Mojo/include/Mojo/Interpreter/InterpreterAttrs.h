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

#ifndef KGEN_INTERPRETER_INTERPRETERATTRS_H
#define KGEN_INTERPRETER_INTERPRETERATTRS_H

#include "Mojo/Interpreter/InterpreterDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

//===----------------------------------------------------------------------===//
// MemoryBlob
//===----------------------------------------------------------------------===//

namespace M {
enum class MemoryKind : uint8_t { Heap, Stack, ConstGlobal, Persistent };

struct OwnedAlignedBlob {
  uint64_t align;
  std::string data;
  bool isString;
};

struct AlignedBlob {
  AlignedBlob(uint64_t align = 0, ArrayRef<char> data = {},
              bool isString = false)
      : align(align), data(data), isString(isString) {}
  AlignedBlob(OwnedAlignedBlob blob)
      : align(blob.align), data(blob.data.data(), blob.data.size()),
        isString(blob.isString) {}

  bool operator==(const AlignedBlob &other) const {
    return std::tie(align, data, isString) ==
           std::tie(other.align, other.data, other.isString);
  }

  uint64_t align;
  ArrayRef<char> data;
  bool isString;
};

/// A pointer region is a chunk of memory in the reference blob that
/// represents a pointer.
struct PointerRegion {
  /// The location of the region within the current blob.
  int64_t offset;
  /// The index of the referenced blob.
  int64_t blobIndex;
  /// The offset into the reference blob.
  int64_t blobOffset;
};
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Mojo/Interpreter/InterpreterAttrs.h.inc"

namespace M {
/// Blobs in `ConstGlobal` or `Persistent` with non-generic address space are
/// globally allocated.
bool isGlobalBlob(MemoryBlobAttr blob);
} // namespace M

#endif // KGEN_INTERPRETER_INTERPRETERATTRS_H
