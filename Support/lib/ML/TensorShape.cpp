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

#include "Support/ML/TensorShape.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <utility>

using namespace M;

//===----------------------------------------------------------------------===//
// TensorShapeStorage
//===----------------------------------------------------------------------===//

bool Detail::TensorShapeStorage::equalsIncludingAuxOOL(
    const TensorShapeStorage &rhs) const {
  return getAuxiliary() == rhs.getAuxiliary() && equalsExcludingAuxOOL(rhs);
}

bool Detail::TensorShapeStorage::equalsExcludingAuxOOL(
    const TensorShapeStorage &rhs) const {
  return SmallVector<ssize_t, kMaxRank>(begin(), end()) ==
         SmallVector<ssize_t, kMaxRank>(rhs.begin(), rhs.end());
}

bool Detail::TensorShapeStorage::isStatic() const {
  if (!hasRank())
    return false;
  switch (getRepKind()) {
  case RepKind::k32:
    for (size_t i = 0; i < 3; ++i) { // encourage unrolling by checking all dims
      if (representation.rep32.dims[i] < 0)
        return false;
    }
    if (representation.rep32.dim3 < 0)
      return false;
    break;
  case RepKind::k16:
    for (size_t i = 0; i < 6; ++i) { // encourage unrolling by checking all dims
      if (representation.rep16.dims[i] < 0)
        return false;
    }
    break;
  case RepKind::kOutOfLine:
    for (size_t i = 0, n = representation.repOutOfLine.rank; i < n; ++i) {
      if (representation.repOutOfLine.dims[i] < 0)
        return false;
    }
    break;
  }
  return true;
}

void Detail::TensorShapeStorage::assignDynamic() {
  if (getRepKind() == RepKind::kOutOfLine)
    delete[] representation.repOutOfLine.dims;

  // Zero-initialize to ensure the representation value is deterministic.
  // We do not zero out the auxiliary field.
  memset(&representation, 0, sizeof(representation) - 1);

  representation.rep16.rank = kDynamicRank;
}

void Detail::TensorShapeStorage::assign(ArrayRef<ssize_t> elements) {
  if (getRepKind() == RepKind::kOutOfLine)
    delete[] representation.repOutOfLine.dims;

  // Zero-initialize to ensure the representation value is deterministic.
  // We do not zero out the auxiliary field.
  memset(&representation, 0, sizeof(representation) - 1);

  // Get and set the rank, regardless of the representation.
  const size_t rank = elements.size();
  assert(rank <= kMaxRank && "requested shape has too high a rank");
  representation.rep16.rank = static_cast<uint8_t>(rank);

  // Decide which representation we can use and initialize the elements.  The
  // most common case should fit into 4 dimensions.
  if (rank <= 4) {
    ssize_t dim;
    // Copy the iterator in case things don't work out.
    auto endIt = elements.end();
    switch (rank) {
    default:
      llvm_unreachable("invalid rank");
    case 4:
      dim = *--endIt;
      representation.rep32.dim3 = dim;
      if (representation.rep32.dim3 != dim)
        break; // Check for dimension too large.
      [[fallthrough]];
    case 3:
      dim = *--endIt;
      representation.rep32.dims[2] = dim;
      if (representation.rep32.dims[2] != dim)
        break; // Check for dimension too large.
      [[fallthrough]];
    case 2:
      dim = *--endIt;
      representation.rep32.dims[1] = dim;
      if (representation.rep32.dims[1] != dim)
        break; // Check for dimension too large.
      [[fallthrough]];
    case 1:
      dim = *--endIt;
      representation.rep32.dims[0] = dim;
      if (representation.rep32.dims[0] != dim)
        break; // Check for dimension too large.
      [[fallthrough]];
    case 0:
      representation.rep32.kind = RepKind::k32;
      return; // Success
    }
  }

  // Virtually everything else will fit into 6 dimensions.
  if (rank <= 6) {
    size_t i;
    // Copy the iterator in case things don't work out.
    auto beginIt = elements.begin();
    for (i = 0; i < rank; ++i) {
      ssize_t dim = *beginIt++;
      representation.rep16.dims[i] = dim;
      if (representation.rep16.dims[i] != dim)
        break;
    }
    if (i == rank) {
      representation.rep16.kind = RepKind::k16;
      return; // Success
    }
  }

  // Otherwise go out of line.
  representation.repOutOfLine.kind = RepKind::kOutOfLine;
  representation.repOutOfLine.dims = new ssize_t[rank];
  std::copy(elements.begin(), elements.end(), representation.repOutOfLine.dims);
}

//===----------------------------------------------------------------------===//
// TensorShape
//===----------------------------------------------------------------------===//

ErrorOrSuccess TensorShape::isRefinedBy(const TensorShape &staticShape) const {
  if (!staticShape.hasRank())
    return Error(Twine("Specified shape ") + staticShape.getAsString() +
                 " must have a statically known rank.");
  if (hasRank()) {
    if (getRank() != staticShape.getRank())
      return Error(Twine("Specified shape ") + staticShape.getAsString() +
                   " doesn't match the rank of the required shape " +
                   getAsString() + ".");
    for (size_t i = 0, n = getRank(); i < n; ++i) {
      int64_t thisDim = this->operator[](i);
      int64_t thatDim = staticShape[i];
      if (thatDim < 0)
        return Error(Twine("Specified shape ") + staticShape.getAsString() +
                     " must have statically known dimension at index " +
                     Twine(i) + ".");
      if (thisDim >= 0 && thisDim != thatDim)
        return Error(Twine("Specified shape ") + staticShape.getAsString() +
                     " has dimension which doesn't match the required shape " +
                     getAsString() + " at index " + Twine(i) + ".");
    }
  } else {
    for (size_t i = 0, n = staticShape.getRank(); i < n; ++i) {
      int64_t thatDim = staticShape[i];
      if (thatDim < 0)
        return Error(Twine("Specified shape ") + staticShape.getAsString() +
                     " must have statically known dimension at index " +
                     Twine(i) + ".");
    }
  }
  return success();
}

void TensorShape::print(raw_ostream &os) const {
  if (hasRank()) {
    llvm::interleave(
        *this, os,
        [&](ssize_t dim) {
          if (dim < 0)
            os << "?";
          else
            os << dim;
        },
        "x");
  } else {
    os << "*";
  }
}

std::string TensorShape::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

ErrorOr<TensorShape> TensorShape::parseFromString(StringRef str) {
  // Empty strings gum up the rest of the function since splitStr would still
  // have one (empty) element, so early-out in this case.
  if (str.empty())
    return TensorShape();

  if (str.size() == 1 && str[0] == '*')
    return TensorShape(kDynamicallyRanked);

  SmallVector<StringRef, kMaxRank> splitStr;
  str.split(splitStr, 'x');
  size_t rank = splitStr.size();
  if (rank > kMaxRank) {
    return Error(Twine("could not parse tensor shape from string: ") + str +
                 " because it is larger that the maximum supported rank " +
                 Twine(kMaxRank));
  }

  SmallVector<ssize_t, kMaxRank> shape;
  shape.reserve(splitStr.size());
  for (auto &it : splitStr) {
    int64_t value;
    if (it == "?") {
      // Follow the MLIR convention for representing dynamic dimensions, though
      // we interpret any negative value to denote dynamic elsewhere.
      shape.emplace_back(mlir::ShapedType::kDynamic);
    } else if (it.getAsInteger(10, value)) {
      return Error(Twine("could not parse dimension integer from string: ") +
                   str + " because " + it + " cannot be parsed as an integer");
    } else if (value < 0) {
      return Error(Twine("could not parse dimension integer from string: ") +
                   str + " because " + it + " is negative");
    } else {
      shape.emplace_back(value);
      if (shape.back() != value)
        return Error(Twine("could not parse dimension integer from string: ") +
                     str + " because " + it + " cannot be represented");
    }
  }

  return TensorShape(shape);
}

void llvm::yaml::ScalarTraits<TensorShape>::output(const M::TensorShape &value,
                                                   void *ctxt,
                                                   llvm::raw_ostream &out) {
  value.print(out);
}

StringRef llvm::yaml::ScalarTraits<TensorShape>::input(StringRef scalar,
                                                       void *ctxt,
                                                       M::TensorShape &value) {
  M::ErrorOr<TensorShape> shapeOr = TensorShape::parseFromString(scalar);
  if (shapeOr.isError())
    // Can't return shapeOr.getError() because that has a lifetime coinciding
    // with shapeOr, whose lifetime ends at the end of this function (can't
    // safely return a StringRef to it, since it would be used after lifetime
    // end).  Unfortunately this means we discard error details, but we don't
    // have the mechanism to preserve them while being safe about lifetime.
    return "Unable to parse tensor shape";
  value = std::move(*shapeOr);
  return StringRef();
}

llvm::yaml::QuotingType
llvm::yaml::ScalarTraits<TensorShape>::mustQuote(StringRef) {
  return llvm::yaml::QuotingType::None;
}
