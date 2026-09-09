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
//
// Helpers for working with size, alignment and dimension values at both
// compile time and runtime using various encoding strategies.
//
// At compile time we have four (!) representations:
//  - uint64_t (aka 'plain form'), where the value cannot be zero or the
//    sentinel used by the raw form. This is the representation to use both
//    at compile time and at runtime when you just want a size_t like value
//    which can never be 'unknown'. It can transition to MEF attributes
//    directly.
//  - std::optional<uint64_t> (aka 'optional form'), where none denotes
//  'unknown',
//    and the value cannot be zero or the sentinel used by the raw form.
//    This is the representation to use at compile time when you need a size_t
//    which could also be 'unknown'. However, it cannot transition to MEF
//    attributes, which requires the use of the 'raw form' below.
//  - uint64_t (aka 'raw form'), where +max denotes 'unknown', and the value
//    cannot be zero. This is the representation to use when transitioning to
//    MEF attributes since they cannot represent the std::optional none value.
//  - int64_t (aka 'raw signed form'), where the MLIR ShapedType::kDynamicSize
//    value denotes 'unknown', and the value must otherwise be strictly
//    positive. This is the representation already chosen by the MLIR
//    ShapedType interface, and is ubiquitous in MLIR dialects. Use this
//    representation when interfacing with other dialects via ShapedType.
//    Note that many runtimes also use a signed integer for dimensions, however
//    tend to use -1 to denote 'unknown', so be careful.
//
// At run time we have the single representation:
//  - size_t (aka 'runtime form'), where +max denotes 'unknown' and the value
//    cannot be zero.
// (However note the MLIR index type is represented as ssize_t at runtime.)
//
// The utilities here help validate and translate these encodings.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_SIZEUTILS_H
#define SUPPORT_ML_SIZEUTILS_H

#include "llvm/Support/Alignment.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace M {

/// Denotes an unknown size in the 'raw form' encoding.
constexpr uint64_t kUnknownSize = std::numeric_limits<uint64_t>::max();

/// Denotes an unknown size in the 'raw signed form' encoding.
/// Copied from ShapedType::kDynamicSize so as to avoid dependency.
constexpr int64_t kUnknownSignedSize = std::numeric_limits<int64_t>::min();

/// Denotes an unknown size in the 'runtime form' encoding.
constexpr size_t kRuntimeUnknownSize = std::numeric_limits<size_t>::max();

/// Returns true if size in plain form is valid.
inline bool isValidPlainSize(uint64_t size) { return size != kUnknownSize; }

/// Returns true if size in optional form is specified.
inline bool hasOptSize(std::optional<uint64_t> optSize) {
  return optSize.has_value();
}

/// Returns true if size in optional form is valid.
inline bool isValidOptSize(std::optional<uint64_t> optSize) {
  return !optSize || isValidPlainSize(*optSize);
}

/// Returns true if size in raw form is specified.
inline bool hasRawSize(uint64_t rawSize) { return rawSize != kUnknownSize; }

/// Returns true if size in raw signed form is specified.
inline bool hasRawSignedSize(int64_t rawSignedSize) {
  return rawSignedSize != kUnknownSignedSize;
}

/// Returns true if size in raw signed form is valid.
inline bool isValidRawSignedSize(int64_t rawSignedSize) {
  return rawSignedSize == kUnknownSignedSize || rawSignedSize >= 0;
}

/// Returns true if size in runtime form is specified.
inline bool hasRuntimeSize(size_t runtimeSize) {
  return runtimeSize != kRuntimeUnknownSize;
}

/// Translates size from optional form to raw form.
/// We assert check for validity, so there's no checking in release builds.
inline uint64_t optSizeToRawSize(std::optional<uint64_t> optSize) {
  assert(isValidOptSize(optSize) && "invalid optional size");
  if (!optSize)
    return kUnknownSize;
  assert(*optSize != kUnknownSize &&
         "cannot represent optional size as raw size");
  return *optSize;
}

/// Translates size from optional form to raw signed form.
/// We assert check for validity, so there's no checking in release builds.
inline int64_t optSizeToRawSignedSize(std::optional<uint64_t> optSize) {
  assert(isValidOptSize(optSize) && "invalid optional size");
  if (!optSize)
    return kUnknownSignedSize;
  assert(*optSize != kUnknownSize &&
         "cannot represent optional size as raw size");
  assert(*optSize <=
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
         "optional size is too large to represent in signed form");
  return static_cast<int64_t>(*optSize);
}

/// Translates size from raw form to optional form.
/// We assert check for validity, so there's no checking in release builds.
inline std::optional<uint64_t> rawSizeToOptSize(uint64_t rawSize) {
  return rawSize == kUnknownSize ? std::optional<uint64_t>() : rawSize;
}

/// Translates size from raw signed form to clean form.
/// We assert check for validity, so there's no checking in release builds.
inline std::optional<uint64_t> rawSignedSizeToOptSize(int64_t rawSignedSize) {
  return rawSignedSize == kUnknownSignedSize
             ? std::optional<uint64_t>()
             : static_cast<uint64_t>(rawSignedSize);
}

/// Translates size from plain form to runtime form.
/// We assert check for validity and no overflow, so there's no checking in
/// release builds.
inline size_t plainSizeToRuntimeSize(uint64_t size) {
  assert(isValidPlainSize(size) && "invalid plain form size");
  assert(size <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()) &&
         "size in plain form is too large to represent in runtime form");
  size_t runtimeSize = static_cast<size_t>(size);
  assert(runtimeSize != kRuntimeUnknownSize &&
         "cannot represent plain size in runtime form");
  return runtimeSize;
}

/// Translates size from raw form to runtime form.
/// We assert check for validity and no overflow, so there's no checking in
/// release builds.
inline size_t rawSizeToRuntimeSize(uint64_t rawSize) {
  if (rawSize == kUnknownSize)
    return kRuntimeUnknownSize;
  assert(rawSize <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()) &&
         "size in raw form is too large to represent in runtime form");
  size_t runtimeSize = static_cast<size_t>(rawSize);
  assert(runtimeSize != kRuntimeUnknownSize &&
         "cannot represent raw size in runtime form");
  return runtimeSize;
}

/// Translates size from raw form to runtime form. If the raw form is the
/// distinguished unknown value the defaultSize is returned.
/// We assert check for validity, so there's no checking in release builds.
inline size_t rawSizeToRuntimeSizeOrDefault(uint64_t rawSize,
                                            size_t defaultSize) {
  if (rawSize == kUnknownSize)
    return defaultSize;
  assert(rawSize <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()) &&
         "size in raw form too large to represent in runtime form");
  assert(static_cast<size_t>(rawSize) != kRuntimeUnknownSize &&
         "cannot represent raw size in runtime form");
  return static_cast<size_t>(rawSize);
}

/// Maximum specifiable alignment attribute.
/// TODO(5815): change this to make MEF file creation compatible across
/// host/target systems with different page sizes.
inline llvm::Align maxAttributeAlignment() {
  return llvm::Align(llvm::sys::Process::getPageSizeEstimate());
}

/// Allocates a MemoryBuffer with the maximum attribute alignment and
/// initializes it from the provided iterator.
inline std::unique_ptr<llvm::MemoryBuffer>
createAndCopyToAlignedBuffer(const char *BufStart, const char *BufEnd) {
  auto alignedMemBuffer = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
      std::distance(BufStart, BufEnd), /*BufferName=*/"",
      /*Alignment=*/maxAttributeAlignment());
  std::copy(BufStart, BufEnd, alignedMemBuffer->getBufferStart());
  return std::move(alignedMemBuffer);
}

/// Returns true if align in raw form is valid.
inline bool isValidRawAlign(uint64_t rawAlign) {
  return rawAlign == kUnknownSize ||
         (llvm::isPowerOf2_64(rawAlign) &&
          (rawAlign <= maxAttributeAlignment().value()));
}

/// Returns true if align in optional form is valid.
inline bool isValidOptAlign(std::optional<uint64_t> optAlign) {
  return !optAlign || isValidRawAlign(*optAlign);
}

/// Translates align in optional form to it's llvm::MaybeAlign
/// representation. We assert check for validity (inside llvm::Align), so
/// there's no checking in release builds.
inline llvm::MaybeAlign optAlignToMaybeAlign(std::optional<uint64_t> optAlign) {
  return llvm::MaybeAlign(optAlign.value_or(0));
}

/// Translates align in llvm::MaybeAlign form to it's optional form.
inline std::optional<uint64_t> maybeAlignToOptAlign(llvm::MaybeAlign align) {
  return align ? align->value() : std::optional<uint64_t>();
}

} // namespace M

#endif // SUPPORT_ML_SIZEUTILS_H
