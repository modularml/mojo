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

#ifndef SUPPORT_COMPILER_BYTECODE_H
#define SUPPORT_COMPILER_BYTECODE_H

#include "Support/LogicalResult.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace M {

//===----------------------------------------------------------------------===//
// Readers and Writers
//===----------------------------------------------------------------------===//

/// ODS helper for parsing an enum.
template <typename T>
LogicalResult readIntegral(mlir::DialectBytecodeReader &reader, T &result) {
  uint64_t value;
  if (failed(reader.readVarInt(value)))
    return failure();
  result = static_cast<T>(value);
  return success();
}

/// ODS helper for printing an enum.
template <typename T>
void writeIntegral(mlir::DialectBytecodeWriter &writer, T value) {
  writer.writeVarInt(static_cast<uint64_t>(value));
}

template <typename T, typename ReadElementFn>
LogicalResult readOptional(mlir::DialectBytecodeReader &reader,
                           std::optional<T> &result, ReadElementFn &&read) {
  FailureOr<llvm::APInt> present = reader.readAPIntWithKnownWidth(1);
  if (failed(present))
    return failure();
  if (present->getLimitedValue() == 0)
    return success();
  result.emplace();
  return std::forward<ReadElementFn>(read)(reader, *result);
}

template <typename T, typename WriteElementFn>
void writeOptional(mlir::DialectBytecodeWriter &writer,
                   const std::optional<T> &result, WriteElementFn &&write) {
  writer.writeAPIntWithKnownWidth(APInt(1, result.has_value()));
  if (result)
    std::forward<WriteElementFn>(write)(writer, *result);
}

/// ODS helper for parsing an array of enums.
template <typename T>
LogicalResult readIntegralArray(mlir::DialectBytecodeReader &reader,
                                llvm::SmallVectorImpl<T> &result) {
  return reader.readList(
      result, [&](T &value) { return M::readIntegral<T>(reader, value); });
}

/// ODS helper for printing an array of enums.
template <typename T>
void writeIntegralArray(mlir::DialectBytecodeWriter &writer,
                        llvm::ArrayRef<T> values) {
  writer.writeList(values, [&](T value) { writeIntegral(writer, value); });
}

//===----------------------------------------------------------------------===//
// WrappedAttrType
//===----------------------------------------------------------------------===//

/// This class provides a bytecode specific wrapper that invokes a special
/// bytecode `get` method of an attribute or type. This is useful for types that
/// override their `get` methods to perform additional logic (which we want to
/// avoid for bytecode, where we already know the values are in the canonical
/// form).
template <typename T>
struct WrappedAttrType : public T {
  using T::T;

  template <typename... Ts>
  static T get(Ts &&...ts) {
    return T::getFromBytecode(std::forward<Ts>(ts)...);
  }
};

} // namespace M

#endif // SUPPORT_COMPILER_BYTECODE_H
