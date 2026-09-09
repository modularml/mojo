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

#include "Support/ML/DebugPrint.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"
#include "llvm/Support/FileSystem.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>

using namespace M;

namespace {
/// We're using raw_ostream to print the tensor (which is probably a bad idea
/// by itself :), which auto formats "char" as a character instead of printing
/// out the byte value.  Use this template to cast to an element type that will
/// cooperate when printing.
template <typename EltType>
struct ElementPrintType {
  using Type = EltType;
};

template <>
struct ElementPrintType<int8_t> {
  using Type = int32_t;
};

template <>
struct ElementPrintType<uint8_t> {
  using Type = uint32_t;
};

static constexpr auto kStartTensorMarker = '[';
static constexpr auto kEndTensorMarker = ']';
static constexpr auto kTensorFiller = " ..., ";
static constexpr size_t kCompactMaxElemsToPrint = 7;
static constexpr size_t kCompactElemPerSide = kCompactMaxElemsToPrint / 2;

} // namespace

template <typename uintT>
static void printElement(raw_ostream &os, const uintT &elem, const DType &dtype,
                         [[maybe_unused]] llvm::FloatStyle floatStyle,
                         [[maybe_unused]] size_t precision) {
  constexpr int numBits = sizeof(elem) * CHAR_BIT;
  static_assert(std::is_unsigned_v<uintT> && std::is_integral_v<uintT>,
                "Use uintXX_t for storage type.");
  static_assert(numBits <= 64,
                "Only support printing up to 64 bit types for now.");

  APInt asAPInt = llvm::APInt(numBits, elem);
  if (dtype.isUInt()) {
    os << asAPInt.getZExtValue();
    return;
  }

  if (dtype.isSInt()) {
    os << asAPInt.getSExtValue();
    return;
  }

  if (dtype.isFloat()) {
    auto writeFloat = [&](const llvm::fltSemantics &semantic) {
      double value = APFloat(semantic, asAPInt).convertToDouble();
      llvm::write_double(os, value, floatStyle, precision);
    };
    dtype.dispatch<void>()
        .template when<DType::f16>([&]() { writeFloat(APFloat::IEEEhalf()); })
        .template when<DType::bf16>([&]() { writeFloat(APFloat::BFloat()); })
        .template when<DType::f32>([&]() { writeFloat(APFloat::IEEEsingle()); })
        .template when<DType::f64>([&]() { writeFloat(APFloat::IEEEdouble()); })
        .otherwise([&]() { os << dtype.getAsString(); });
    return;
  }

  if (dtype.isBool()) {
    os << (asAPInt.getZExtValue() ? "True" : "False");
    return;
  }

  os << dtype.getAsString();
}

template <typename T, typename ElementFunctor>
static ErrorOrSuccess printElementsComplete(const T *ptr, size_t numElements,
                                            ElementFunctor printFn,
                                            raw_ostream &os) {
  auto elts = ArrayRef(ptr, numElements);
  llvm::interleaveComma(elts, os, printFn);
  return success();
}

template <typename T, typename ElementFunctor>
static ErrorOrSuccess printElementsCompact(const T *ptr, size_t numElements,
                                           ElementFunctor printFn,
                                           raw_ostream &os) {
  os << kStartTensorMarker;
  if (numElements < kCompactMaxElemsToPrint) {
    ErrorOrSuccess wasSuccess =
        printElementsComplete(ptr, numElements, printFn, os);
    os << kEndTensorMarker;
    return wasSuccess;
  }

  auto elts = ArrayRef(ptr, numElements);
  llvm::interleaveComma(elts.take_front(kCompactElemPerSide), os, printFn);

  /// Add ... to fill the intermediate values.
  os << ",";
  os << kTensorFiller;

  llvm::interleaveComma(elts.take_back(kCompactElemPerSide), os, printFn);
  os << kEndTensorMarker;
  return success();
}

template <typename FormattingFunc>
static ErrorOrSuccess printElementsWithFormatting(
    raw_ostream &os, const void *srcPtr, const DType &dtype,
    const size_t &numElements, const llvm::FloatStyle &floatStyle,
    const size_t &precision, FormattingFunc &&printWithFormat) {
  auto printWithStorageWidth = [&](const auto *a) -> ErrorOrSuccess {
    return printWithFormat(
        a, numElements,
        [&](auto elem) {
          printElement(os, elem, dtype, floatStyle, precision);
        },
        os);
  };

  switch (dtype.getWidthInBits()) {
  case 8:
    return printWithStorageWidth(static_cast<const uint8_t *>(srcPtr));
  case 16:
    return printWithStorageWidth(static_cast<const uint16_t *>(srcPtr));
  case 32:
    return printWithStorageWidth(static_cast<const uint32_t *>(srcPtr));
  case 64:
    return printWithStorageWidth(static_cast<const uint64_t *>(srcPtr));
  }

  return Error("Unhandled bit-width " + Twine(dtype.getWidthInBits()));
}

static ErrorOrSuccess printTensorComplete(raw_ostream &os, const void *srcPtr,
                                          const TensorSpec &spec,
                                          llvm::FloatStyle floatStyle,
                                          size_t precision) {
  os << "tensor<";
  for (auto elt : spec)
    os << elt << "x";

  auto eltType = spec.getEltType();
  os << eltType.getAsString() << "> [";

  size_t numElements = spec.getNumElements();

  ErrorOrSuccess didPrint = printElementsWithFormatting(
      os, srcPtr, eltType, numElements, floatStyle, precision,
      [&](auto ptr, auto num_elements, auto printFn, auto &os) {
        return printElementsComplete(ptr, num_elements, printFn, os);
      });

  os << "]\n";
  os.flush();

  if (didPrint.isError()) {
    std::string errorMsg;
    llvm::raw_string_ostream errStream(errorMsg);
    errStream << "Printing-Error";
    errStream << "Could not print tensor with dtype " << eltType.getAsString()
              << " with error: \n"
              << didPrint.getError() << "\n";
    return Error(errorMsg);
  }

  return success();
}

static ErrorOrSuccess printTensorCompact(raw_ostream &os, const void *srcPtr,
                                         const TensorSpec &spec,
                                         llvm::FloatStyle floatStyle,
                                         size_t precision) {

  /// What we are doing is printing a series of 2D arrays if the dimension
  /// is greater than 2. We limit print depth to 7 in height, width and depth
  /// Assume the dimension is [1, 9, 100, 100]. We would print pick first 3
  /// [1, 100, 100] matrices, and last 3 [1, 100, 100] matrices and print
  /// them. In each of those we only print first, and last 3 rows and
  /// first and last 3 columns. The intermediaries are filled with '...'
  /// to indicate something is here but we are not displaying it.

  auto dims = spec.getDimsCopy();
  size_t columnElemCount = dims.empty() ? 1 : dims.pop_back_val();
  // If the tensor is a rank-1 vector, then the number of rows is 1.
  size_t rowElemCount = dims.empty() ? 1 : dims.pop_back_val();
  size_t matrixElemCount = columnElemCount * rowElemCount;
  std::string errorMsg;
  llvm::raw_string_ostream errStream(errorMsg);
  os << "tensor(";
  auto eltType = spec.getEltType();

  // Open parens for every other dimension other than rowElemCount &
  // columnElemCount
  for (size_t i = 2, e = spec.getRank(); i < e; ++i)
    os << kStartTensorMarker;

  /// We are basically printing a bunch of 2D tensors in succession
  /// So if a tensor is of numMatrices * rowElemCount * columnElemCount
  /// dimension, we print numMatrices tensors of rowElemCount * columnElemCount
  /// dimension. numMatrices equals the product of all dims other than last two

  /// We have popped out all dims other than those relevant to numMatrices
  /// already.
  size_t numMatrices =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());

  for (size_t matrixIdx = 0; matrixIdx < numMatrices;) {

    /// This is the marker between a rowElemCount*columnElemCount tensor and
    /// next. We don't need marker for the first time
    if (matrixIdx > 0)
      os << ",\n\n";
    os << kStartTensorMarker;

    /// Print row
    for (size_t rowIdx = 0; rowIdx < rowElemCount;) {

      /// We print the next row in a new line
      if (rowIdx > 0)
        os << "\n";

      ErrorOrSuccess didPrint = printElementsWithFormatting(
          os, srcPtr, eltType, columnElemCount, floatStyle, precision,
          [&](auto ptr, auto num_elements, auto printFn, auto &os) {
            return printElementsCompact(ptr + matrixIdx * matrixElemCount +
                                            rowIdx * columnElemCount,
                                        num_elements, printFn, os);
          });

      if (didPrint.isError()) {
        os << "Printing-Error";
        errStream << "Could not print tensor with dtype "
                  << eltType.getAsString() << " with error: \n"
                  << didPrint.getError() << "\n";
      }

      ++rowIdx;

      /// We are skipping printing comma after the last bracket.
      if (rowIdx != rowElemCount)
        os << ",";

      // Intermediate rows are filled with "..." and rowIdx is advanced to third
      // from last
      if (rowElemCount >= kCompactMaxElemsToPrint &&
          rowIdx == kCompactElemPerSide) {
        os << "\n" << kTensorFiller << "\n";
        rowIdx = rowElemCount - kCompactElemPerSide;
      }
    }
    os << kEndTensorMarker;
    ++matrixIdx;

    // Again intermediate matrices are skipped for compactness
    if (numMatrices >= kCompactMaxElemsToPrint &&
        matrixIdx == kCompactElemPerSide) {
      os << "\n" << kTensorFiller << "\n";
      matrixIdx = numMatrices - kCompactElemPerSide;
    }
  }

  // Now every element is printed. We just have to close all open parans.
  for (size_t i = 2, rank = spec.getRank(); i < rank; ++i)
    os << kEndTensorMarker;

  os << ", dtype=";
  os << eltType.getAsString();
  os << ", shape=[";
  llvm::interleave(spec, os, ",");
  os << "])\n";

  if (!errorMsg.empty())
    os << errorMsg << "\n";
  os.flush();
  return success();
}

/// This kernel prints the specified tensor data to stdout.
ErrorOrSuccess M::printTensor(raw_ostream &os, const void *srcPtr,
                              const TensorSpec &spec, ResultOutputStyle style,
                              unsigned precision) {
  // TODO: Add print support for quantization.
  switch (style) {
  case ResultOutputStyle::kFull:
    return printTensorComplete(os, srcPtr, spec, llvm::FloatStyle::Exponent,
                               precision);
  case ResultOutputStyle::kCompact:
    return printTensorCompact(os, srcPtr, spec, llvm::FloatStyle::Fixed, 4);
  case ResultOutputStyle::kBinary:
  case ResultOutputStyle::kBinaryMaxCheckpoint:
  case ResultOutputStyle::kNone:

    // No-op.
    return success();
  }
  return Error("unknown ResultOutputStyle style");
}

ErrorOrSuccess DebugTensorPrintOptions::printTensor(const void *buffer,
                                                    const M::TensorSpec &spec,
                                                    StringRef label) {
  if (style == ResultOutputStyle::kNone)
    return success();

  if (style != ResultOutputStyle::kBinary &&
      style != ResultOutputStyle::kBinaryMaxCheckpoint) {
    auto &os = llvm::outs();
    if (!label.empty())
      os << label << " = ";
    if (auto errOr = M::printTensor(os, buffer, spec, style, precision))
      return errOr.takeError();
    os.flush();
    return success();
  }

  if (label.empty())
    return Error("Debug printing of tensors in binary format requires a label");

  if (!binaryDir.empty()) {
    // Create the output directory. If the output directory is not there, then
    // the directory is created along with all of its parents. If the directory
    // does exist, then the following is a no-op.
    if (auto ec = llvm::sys::fs::create_directories(binaryDir,
                                                    /*ignoreExisting=*/true))
      return Error(Twine("Unable to create output directory for writing debug "
                         "tensor binary: ") +
                   ec.message());
  }

  // If the output name has illegal symbols, replace them with '_'.
  const std::string illegalChars = " \\/:?\"<>|";
  std::string sanitizedName = label.str();
  std::replace_if(
      sanitizedName.begin(), sanitizedName.end(),
      [&](char c) { return illegalChars.find(c) != std::string::npos; }, '_');

  std::string filename = sanitizedName;
  if (style == ResultOutputStyle::kBinaryMaxCheckpoint) {
    filename += ".max";
  }

  std::filesystem::path outputPath =
      binaryDir.empty() ? std::filesystem::path(filename)
                        : std::filesystem::path(binaryDir) / filename;
  llvm::dbgs() << "Writing debug binary tensor " << spec << " of "
               << spec.getSizeInBytes() << " bytes to '" << outputPath.string()
               << "'\n";
  std::ofstream outputFile(outputPath, std::ofstream::binary);

  if (style == ResultOutputStyle::kBinaryMaxCheckpoint) {
    // Write to the MAX checkpoint format, which prepends the output file with
    // a header, version numbers, and metadata.
    // This must match the header in max/graph/checkpoint/metadata.mojo.
    uint8_t serializationHeader[] = {0x93, 0xF0, 0x9F, 0x94,
                                     0xA5, 0x2B, 0x2B, 0x93};
    uint32_t serializationMajorFormat = 0;
    uint32_t serializationMinorFormat = 1;
    outputFile.write((char *)&serializationHeader, sizeof(serializationHeader));
    outputFile.write((char *)&serializationMajorFormat,
                     sizeof(serializationMajorFormat));
    outputFile.write((char *)&serializationMinorFormat,
                     sizeof(serializationMinorFormat));

    // Metadata consists of:
    //  - Metadata size
    //  - key length (4 bytes)
    //  - string key (variable size)
    //  - dtype (1 byte)
    //  - rank (1 byte)
    //  - dimensions (4 * rank bytes)
    //  - offset (8 bytes)
    const char *key = sanitizedName.c_str();
    uint32_t keylen = strlen(key);
    uint8_t dtype = spec.getEltType().getValue();
    uint8_t rank = spec.getRank();
    uint64_t metadataSize = sizeof(keylen) + keylen + sizeof(dtype) +
                            sizeof(rank) + sizeof(uint32_t) * rank +
                            sizeof(uint64_t);
    outputFile.write((char *)&metadataSize, sizeof(metadataSize));
    outputFile.write((char *)&keylen, sizeof(keylen));
    outputFile.write(key, strlen(key));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"

    outputFile.write((char *)&dtype, sizeof(dtype));
    outputFile.write((char *)&rank, sizeof(rank));

#pragma GCC diagnostic pop

    for (uint8_t i = 0; i < rank; ++i) {
      uint32_t dim = spec[i];
      outputFile.write((char *)&dim, sizeof(dim));
    }
    uint64_t tensorOffset =
        sizeof(serializationHeader) + sizeof(serializationMajorFormat) +
        sizeof(serializationMinorFormat) + sizeof(metadataSize) + metadataSize;
    outputFile.write((char *)&tensorOffset, sizeof(tensorOffset));
  }

  outputFile.write(static_cast<const char *>(buffer), spec.getSizeInBytes());
  if (!outputFile)
    return Error(Twine("Unable to write to debug tensor binary '") +
                 outputPath.string() + "'");

  return success();
}
