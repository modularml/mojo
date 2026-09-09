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

#include "Support/ML/Fill.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/ML/DType.h"
#include "Support/MathExtras.h"
#include "llvm/ADT/ArrayRef.h"

#include "llvm/ADT/APFloat.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <sys/types.h>
#include <type_traits>

using namespace M;
using llvm::MutableArrayRef;

/// This kernel fills the specified generic buffer with a single "1" or "1.0"
/// real value.  Complex numbers have their imaginary component set to zero.
ErrorOrSuccess M::getScalarOne(void *destPtr, DType eltType) {
  if (eltType.isComplex()) {
    unsigned widthInBytes = eltType.getWidthInBits() / 8;
    // Set the imaginary component to zero.
    memset((char *)destPtr + widthInBytes, 0, widthInBytes);
    eltType = eltType.stripComplex();
  }

  return eltType.dispatch<ErrorOrSuccess>(destPtr)
      .when([&](bool *ptr) {
        *ptr = true;
        return success();
      })
      .when<DType::bf16>([&](void *ptr) {
        *(static_cast<uint16_t *>(ptr)) = 0x3F80;
        return success();
      })
      .when<DType::f16>([&](void *ptr) {
        *(static_cast<uint16_t *>(ptr)) = 0x3C00;
        return success();
      })
      .whenCXXInt([&](auto *ptr) { // Standard C++ integers.
        *ptr = 1;
        return success();
      })
      .whenCXXFP([&](auto *ptr) { // float and double.
        *ptr = 1.0;
        return success();
      })
      .otherwise([&]() {
        return Error("getScalarOne: cannot initialize " +
                     eltType.getAsString() + " to 1");
      });
}

/// This kernel fills the specified generic buffer with a single "-1" or "-1.0"
/// real value.  Complex numbers have their imaginary component set to zero.
ErrorOrSuccess M::getScalarNegativeOne(void *destPtr, DType eltType) {
  if (eltType.isComplex()) {
    unsigned widthInBytes = eltType.getWidthInBits() / 8;
    // Set the imaginary component to zero.
    memset((char *)destPtr + widthInBytes, 0, widthInBytes);
    eltType = eltType.stripComplex();
  }

  return eltType.dispatch<ErrorOrSuccess>(destPtr)
      .when([&](bool *ptr) {
        *ptr = true;
        return success();
      })
      .when<DType::bf16>([&](void *ptr) {
        *(static_cast<uint16_t *>(ptr)) = 0xBF80;
        return success();
      })
      .when<DType::f16>([&](void *ptr) {
        *(static_cast<uint16_t *>(ptr)) = 0xBC00;
        return success();
      })
      .whenCXXInt([&](auto *ptr) { // Standard C++ integers.
        *ptr = -1;
        return success();
      })
      .whenCXXFP([&](auto *ptr) { // float and double.
        *ptr = -1.0;
        return success();
      })
      .otherwise([&]() {
        return Error("getScalarNegativeOne: cannot initialize " +
                     eltType.getAsString() + " to -1");
      });
}

/// This kernel fills the specified generic buffer with a constant value
/// specified by "element".  This returns a non-empty error on failure.
ErrorOrSuccess M::fillHomogeneous(void *destPtr, size_t numElements,
                                  DType eltType, const void *elementPtr) {
  ssize_t bufferSize = eltType.getSizeInBytes(numElements);
  if (bufferSize < 0)
    return Error("fillHomogeneous: " + eltType.getAsString() +
                 " has unknown width");
  if (size_t(bufferSize) < numElements)
    return Error("fillHomogeneous: not support for sub-byte type " +
                 eltType.getAsString() + " (yet)");

  size_t bytesPerElement = eltType.getWidthInBits() >> 3;

  // Fill `chunk` with the data repeated a few times, then memcpy it into the
  // destination.
  uint8_t chunk[64];
  static_assert(sizeof(chunk) >= DType::kMaxElementSizeInBytes,
                "chunk size cannot be smaller than the max dtype element size");

  /// Fill the output destPtr with data from 'chunk'.  The 'chunkSize' variable
  /// indicates how many bytes of chunk are valid to use.  This can be less than
  /// 64 in the case of things like f80 (where it will be 60).
  auto fillFromChunk = [&](size_t chunkSize) {
    while (size_t(bufferSize) >= chunkSize) {
      memcpy(destPtr, chunk, chunkSize);
      destPtr = (char *)destPtr + chunkSize;
      bufferSize -= chunkSize;
    }
    if (bufferSize)
      memcpy(destPtr, chunk, bufferSize);
  };

  switch (bytesPerElement) {
  case 1:
    memset(destPtr, *(const uint8_t *)elementPtr, numElements);
    return success();
#ifdef __APPLE__
  case 2:
    // Each of these memcpy's will be compiled into an unaligned load/store.
    memcpy(chunk, elementPtr, 2);
    memcpy(chunk + 2, elementPtr, 2);
    elementPtr = chunk;
    [[fallthrough]];
  case 4:
    memset_pattern4(destPtr, elementPtr, bufferSize);
    return success();
  case 8:
    memset_pattern8(destPtr, elementPtr, bufferSize);
    return success();
  case 16:
    memset_pattern16(destPtr, elementPtr, bufferSize);
    return success();
#else
  case 2:
    // Each of these memcpy's will be compiled into an unaligned load/store.
    memcpy(chunk, elementPtr, 2);
    memcpy(chunk + 2, elementPtr, 2);
    memcpy(chunk + 4, chunk, 4);
    memcpy(chunk + 8, chunk, 8);
    break;
  case 4:
    memcpy(chunk, elementPtr, 4);
    memcpy(chunk + 4, elementPtr, 4);
    memcpy(chunk + 8, chunk, 8);
    break;
  case 8:
    memcpy(chunk, elementPtr, 8);
    memcpy(chunk + 8, elementPtr, 8);
    break;
  case 16:
    memcpy(chunk, elementPtr, 16);
    break;
#endif
  default: {
    // This is for the weird case, things like f80 which isn't a power of two in
    // size.
    assert(bytesPerElement <= sizeof(chunk) && "giant element?");
    memcpy(chunk, elementPtr, bytesPerElement);
    unsigned chunkSize = bytesPerElement;
    while (chunkSize * 2 <= sizeof(chunk)) {
      memcpy(chunk + chunkSize, chunk, chunkSize);
      chunkSize *= 2;
    }
    // Finally, fill the target buffer with chunks until we reach the end.  This
    // is less efficient than the copy of this below, because the chunk size is
    // not constant here.
    fillFromChunk(chunkSize);
    return success();
  }
  }

  // Otherwise, we know we have a 16-byte chunk by now, fill out the rest of the
  // buffer.
  memcpy(chunk + 16, chunk, 16);
  memcpy(chunk + 32, chunk, 32);

  // Finally, fill the target buffer with 64-byte chunks until we reach the end.
  fillFromChunk(sizeof(chunk));
  return success();
}

/// Fill the given buffer with random floats in [-1, 1) with the given
/// semantic (e.g. IEEESingle, BFloat). Meant to use if no native C++ type.
template <typename StorageType>
inline ErrorOrSuccess handleSpecialFloats(void *destPtr, size_t numElements,
                                          const llvm::fltSemantics &semantic) {
  fillWithRandomSpecialFloats<StorageType>(
      MutableArrayRef(static_cast<StorageType *>(destPtr), numElements), -1.0,
      1.0, semantic);
  return success();
}

/// This kernel fills the specified generic buffer with a random values.
/// This returns a non-empty error on failure.
///
/// TODO: This is not implemented in a very general way, the bounds should be
/// passed it or something.
ErrorOrSuccess M::fillRandom(void *destPtr, size_t numElements, DType eltType) {

  return eltType.dispatch<ErrorOrSuccess>(destPtr)
      .when([&](bool *destPtr) {
        fillWithRandomDistribution(MutableArrayRef(destPtr, numElements),
                                   std::bernoulli_distribution());
        return success();
      })
      .when<DType::f16>([&](void *destPtr) {
        return handleSpecialFloats<uint16_t>(destPtr, numElements,
                                             llvm::APFloat::IEEEhalf());
      })
      .when<DType::bf16>([&](void *destPtr) {
        return handleSpecialFloats<uint16_t>(destPtr, numElements,
                                             llvm::APFloat::BFloat());
      })
      .whenCXXInt([&](auto *destPtr) {
        fillWithRandomInts<std::remove_pointer_t<decltype(destPtr)>>(
            MutableArrayRef(destPtr, numElements), eltType.isSInt() ? -10 : 0,
            10);
        return success();
      })
      .whenCXXFP([&](auto *destPtr) {
        fillWithRandomFloats<std::remove_pointer_t<decltype(destPtr)>>(
            MutableArrayRef(destPtr, numElements), -1.0, 1.0);
        return success();
      })
      .otherwise([&]() {
        return Error("cannot random initialize " + eltType.getAsString());
      });
}
