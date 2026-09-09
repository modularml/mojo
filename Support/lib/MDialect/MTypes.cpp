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

#include "Support/MDialect/MTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"
#include <cassert>
#include <cstdint>
#include <optional>

using namespace M;

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Parse rank 1 dimension followed by an 'x'.
static ParseResult parseSizeX(AsmParser &p, int64_t &size) {
  SmallVector<int64_t> dims;
  llvm::SMLoc curLoc = p.getCurrentLocation();
  if (p.parseDimensionList(dims, /*allowDynamic=*/false))
    return failure();
  if (dims.size() != 1)
    return p.emitError(curLoc, "expected a single dimension");
  size = dims.front();
  return success();
}

static void printSizeX(AsmPrinter &p, int64_t size) { p << size << 'x'; }

//===----------------------------------------------------------------------===//
// MDialect
//===----------------------------------------------------------------------===//

void MDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Support/MDialect/MTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Ensure the array size is non-negative.
LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                int64_t size, Type elementType) {
  if (size < 0)
    return emitError() << "invalid array size: " << size;
  return success();
}

/// An array type always has rank 1.
bool ArrayType::hasRank() const { return true; }

/// Clone the type. Expect the shape to always be rank 1.
ShapedType ArrayType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                Type elementType) const {
  assert(!shape || shape->size() == 1);
  if (shape)
    return get(shape->front(), elementType);
  return get(getSize(), elementType);
}

ArrayType ArrayType::get(int64_t size, Type elementType) {
  return get(elementType.getContext(), size, elementType);
}

ArrayType ArrayType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                int64_t size, Type elementType) {
  if (failed(verify(emitError, size, elementType)))
    return {};
  return get(size, elementType);
}

// We defer ArrayType::getShape till after ODS import since depends on
// impl.

//===----------------------------------------------------------------------===//
// AlignedBytesType
//===----------------------------------------------------------------------===//

/// Implements ShapedType::cloneWith
ShapedType AlignedBytesType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                       Type elementType) const {
  assert(!shape && "cannot change shape");
  assert(elementType.isUnsignedInteger(8) && "elementType must be ui8");
  return AlignedBytesType::get(getContext(), getSize(), getAlign());
}

/// Implements ShapedType::getElementType
Type AlignedBytesType::getElementType() const {
  return IntegerType::get(getContext(), /*width=*/8, IntegerType::Unsigned);
}

/// Implements ShapedType::hasRank
bool AlignedBytesType::hasRank() const { return true; }

// We defer AlignedBytesType::getShape till after ODS import since depends on
// impl.

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Support/MDialect/MTypes.cpp.inc"
#include "Support/ML/SizeUtils.h"

//===----------------------------------------------------------------------===//
// ArrayType (cont.)
//===----------------------------------------------------------------------===//

/// The shape of an array is always [size].
ArrayRef<int64_t> ArrayType::getShape() const {
  // We need to return a const reference.
  return static_cast<detail::ArrayTypeStorage *>(getImpl())->size;
}

//===----------------------------------------------------------------------===//
// AlignedBytesType (cont.)
//===----------------------------------------------------------------------===//

/// Implements ShapedType::getShape
ArrayRef<int64_t> AlignedBytesType::getShape() const {
  uint64_t &size = getImpl()->size;
  // Convert to raw signed size for assertion side effect.
  (void)optSizeToRawSignedSize(size);
  // We'll just reinterpret the uint64_t size as an int64_t since it's in range.
  return ArrayRef<int64_t>(reinterpret_cast<const int64_t *>(&size), 1);
}
