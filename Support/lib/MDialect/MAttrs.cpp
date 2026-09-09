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

#include "Support/MDialect/MAttrs.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/DeviceSpecs.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "Support/MDialect/MDialect.h"
#include "Support/MDialect/MTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace M;

//===----------------------------------------------------------------------===//
// MDialect
//===----------------------------------------------------------------------===//

void MDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Support/MDialect/MAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ArithmeticType
//===----------------------------------------------------------------------===//

namespace {
/// A helper class for manipulating float, integer, or index element types.
class ArithmeticType : public Type {
public:
  using Type::Type;

  /// Support type inquiry.
  static bool classof(Type type) { return type.isIntOrIndexOrFloat(); }

  /// Get the bitwidth.
  unsigned getWidth() const {
    if (isIndex())
      return IndexType::kInternalStorageBitWidth;
    return getIntOrFloatBitWidth();
  }

  /// Get the type size in bytes rounded up to the nearest byte boundary.
  unsigned getNearestByteSize() const {
    return llvm::divideCeil(getWidth(), CHAR_BIT);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PrimitiveArray
//===----------------------------------------------------------------------===//

/// Require the element type to be a float, integer, or index.
LogicalResult
PrimitiveArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           ArrayRef<uint8_t> data, Type elementType) {
  auto intOrFp = llvm::dyn_cast<ArithmeticType>(elementType);
  if (!intOrFp)
    return emitError() << "expected integer, index, or float element type";
  // Disallow s/ui0.
  if (intOrFp.getWidth() == 0)
    return emitError() << "zero-width element type unsupported";
  // Sanity check the provided data.
  unsigned extraBytes = data.size() % intOrFp.getNearestByteSize();
  if (extraBytes)
    return emitError() << "provided raw data has " << extraBytes
                       << " extra bytes";
  return success();
}

/// Copy the data into a byte buffer aligned to the nearest power-of-2 byte
/// boundary.
static ArrayRef<uint8_t>
copyIntoAlignedBuffer(mlir::StorageUniquer::StorageAllocator &allocator,
                      ArrayRef<uint8_t> data, Type elementType) {
  unsigned byteSize =
      llvm::divideCeil(cast<ArithmeticType>(elementType).getWidth(), CHAR_BIT);
  auto *ptr = static_cast<uint8_t *>(
      allocator.allocate(data.size(), llvm::PowerOf2Ceil(byteSize)));
  std::uninitialized_copy(data.begin(), data.end(), ptr);
  return {ptr, data.size()};
}

namespace {
/// Helper for parsing arbitrary integers and floats.
class PrimitiveElementParser {
public:
  PrimitiveElementParser(ArithmeticType type)
      : type(type), byteSize(type.getNearestByteSize()) {}

  /// Take the parsed data.
  std::vector<uint8_t> takeData() { return std::move(data); }

  /// Parse a single integer.
  ParseResult parseSingleInteger(AsmParser &p) {
    APInt apInt(type.getWidth(), 0, !type.isUnsignedInteger());
    if (p.parseInteger(apInt))
      return failure();
    append(apInt.sextOrTrunc(type.getWidth()));
    return success();
  }

  /// Parse a single float.
  ParseResult parseSingleFloat(AsmParser &p) {
    FloatAttr fpAttr;
    if (p.parseAttribute(fpAttr, type))
      return failure();
    append(fpAttr.getValue().bitcastToAPInt());
    return success();
  }

  /// Parse a comma-separated list of integers or floats.
  ParseResult parseElements(AsmParser &p) {
    if (isa<FloatType>(type))
      return p.parseCommaSeparatedList([&] { return parseSingleFloat(p); });
    return p.parseCommaSeparatedList([&] { return parseSingleInteger(p); });
  }

private:
  /// Append one element.
  void append(const APInt &value) {
    size_t offset = data.size();
    data.insert(data.end(), byteSize, 0);
    llvm::StoreIntToMemory(value, data.data() + offset, byteSize);
  }

  ArithmeticType type;
  unsigned byteSize;
  std::vector<uint8_t> data;
};

/// Helper for printing arbitrary integers and floats.
class PrimitiveElementPrinter {
public:
  PrimitiveElementPrinter(ArithmeticType type, ArrayRef<uint8_t> data)
      : type(type), byteSize(type.getNearestByteSize()), data(data) {}

  /// Print a single integer.
  void printSingleInteger(AsmPrinter &p, unsigned i) {
    APInt intVal = getValue(i);
    // Print i1 as 'true' or 'false'.
    if (type.isInteger(1))
      p << (intVal.isOne() ? "true" : "false");
    else
      intVal.print(p.getStream(), !type.isUnsignedInteger());
  }

  /// Print a single float.
  void printSingleFloat(AsmPrinter &p, unsigned i) {
    APInt intVal = getValue(i);
    APFloat fpVal(cast<FloatType>(type).getFloatSemantics(), intVal);
    p.printFloat(fpVal);
  }

  /// Print the elements.
  void printElements(AsmPrinter &p) {
    unsigned size = data.size() / byteSize;
    if (auto fpType = dyn_cast<FloatType>(type)) {
      llvm::interleaveComma(llvm::seq<unsigned>(0, size), p,
                            [&](unsigned i) { printSingleFloat(p, i); });
    } else {
      llvm::interleaveComma(llvm::seq<unsigned>(0, size), p,
                            [&](unsigned i) { printSingleInteger(p, i); });
    }
  }

  /// Load a single element.
  APInt getValue(unsigned i) {
    APInt value(type.getWidth(), 0, !type.isUnsignedInteger());
    llvm::LoadIntFromMemory(value, data.data() + i * byteSize, byteSize);
    return value;
  }

private:
  ArithmeticType type;
  unsigned byteSize;
  ArrayRef<uint8_t> data;
};
} // namespace

/// Parse the elements of a primitive array.
static ParseResult parsePrimitiveArray(AsmParser &p,
                                       std::vector<uint8_t> &values,
                                       Type elementType) {
  auto intOrFp = dyn_cast<ArithmeticType>(elementType);
  if (!intOrFp)
    return p.emitError(p.getCurrentLocation(),
                       "expected integer, index, or float element type");

  // The array is empty if there are no colons.
  if (p.parseOptionalColon())
    return success();

  PrimitiveElementParser handler(intOrFp);
  if (handler.parseElements(p))
    return failure();
  values = handler.takeData();
  return success();
}

/// Print the elements of a primitive array.
static void printPrimitiveArray(AsmPrinter &p, ArrayRef<uint8_t> values,
                                Type elementType) {
  // Skip the colon if the array is empty.
  if (values.empty())
    return;
  p << ": ";
  PrimitiveElementPrinter handler(cast<ArithmeticType>(elementType), values);
  handler.printElements(p);
}

int64_t PrimitiveArrayAttr::size() const {
  return getData().size() /
         llvm::cast<ArithmeticType>(getElementType()).getNearestByteSize();
}

PrimitiveArrayAttr PrimitiveArrayAttr::get(ArrayRef<uint8_t> data,
                                           Type elementType) {
  return get(elementType.getContext(), data, elementType);
}

PrimitiveArrayAttr
PrimitiveArrayAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<uint8_t> data, Type elementType) {
  if (failed(verify(emitError, data, elementType)))
    return {};
  return get(data, elementType);
}

//===----------------------------------------------------------------------===//
// ArrayElementsAttr
//===----------------------------------------------------------------------===//

/// Verify that the shaped type elements count matches the size of the array.
LogicalResult
ArrayElementsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          PrimitiveArrayAttr data, ShapedType type) {
  if (!type.hasStaticShape())
    return emitError() << "shaped type must have static shape";
  if (type.getNumElements() != data.size())
    return emitError() << "attribute type indicates " << type.getNumElements()
                       << " elements, but array has " << data.size();
  return success();
}

/// Parse the elements of an array elements attribute.
Attribute ArrayElementsAttr::parse(AsmParser &p, Type attrType) {
  // Validate the self type.
  auto type = dyn_cast_if_present<ShapedType>(attrType);
  if (!type) {
    p.emitError(p.getCurrentLocation(), "expected a shaped type");
    return {};
  }
  auto elementType = llvm::dyn_cast<ArithmeticType>(type.getElementType());
  if (!elementType) {
    p.emitError(p.getCurrentLocation(),
                "expected integer, index, or float element type");
    return {};
  }

  auto emitError = [&] { return p.emitError(p.getCurrentLocation()); };

  if (p.parseLess())
    return {};
  // Check for an empty attribute.
  if (succeeded(p.parseOptionalGreater()))
    return getChecked(emitError, p.getContext(),
                      PrimitiveArrayAttr::get({}, elementType), type);

  FailureOr<std::vector<uint8_t>> result;
  PrimitiveElementParser handler(elementType);
  if (handler.parseElements(p) || p.parseGreater())
    return {};
  return getChecked(emitError, p.getContext(),
                    PrimitiveArrayAttr::get(handler.takeData(), elementType),
                    type);
}

/// Print the elements of an array elements attribute.
void ArrayElementsAttr::print(AsmPrinter &p) const {
  p << '<';
  PrimitiveElementPrinter handler(llvm::cast<ArithmeticType>(getElementType()),
                                  getData().getData());
  handler.printElements(p);
  p << '>';
}

ArrayElementsAttr ArrayElementsAttr::get(ArrayRef<uint8_t> data,
                                         ShapedType type) {
  return get(type.getContext(),
             PrimitiveArrayAttr::get(data, type.getElementType()), type);
}

ArrayElementsAttr
ArrayElementsAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<uint8_t> data, ShapedType type) {

  PrimitiveArrayAttr dataAttr =
      PrimitiveArrayAttr::get(data, type.getElementType());
  if (failed(verify(emitError, dataAttr, type)))
    return {};
  return get(data, type);
}

ArrayRef<uint8_t> ArrayElementsAttr::getRawData() const {
  return getData().getData();
}

FailureOr<detail::AttrIterator>
ArrayElementsAttr::try_value_begin_impl(OverloadToken<Attribute>) const {
  return detail::AttrIterator(getRawData().data(), 0, getElementType());
}

Attribute detail::AttrIterator::operator*() const {
  auto type = cast<ArithmeticType>(elementType);
  APInt val(type.getWidth(), 0);
  unsigned byteSize = type.getNearestByteSize();
  llvm::LoadIntFromMemory(val, getBase() + getIndex() * byteSize, byteSize);
  if (type.isIntOrIndex())
    return IntegerAttr::get(type, val);
  APFloat fpVal(cast<FloatType>(type).getFloatSemantics(), val);
  return FloatAttr::get(cast<FloatType>(type), fpVal);
}

/// HasAlignedBytesInterface::getAlignedBytesTypes
AlignedBytesType ArrayElementsAttr::getAlignedBytesType() const {
  if (auto alignedBytesType = llvm::dyn_cast<AlignedBytesType>(getType()))
    return alignedBytesType;
  uint64_t elementByteSize = static_cast<uint64_t>(llvm::divideCeil(
      llvm::cast<ArithmeticType>(getElementType()).getWidth(), CHAR_BIT));
  uint64_t byteSize = elementByteSize * static_cast<uint64_t>(size());
  uint64_t align = llvm::PowerOf2Ceil(elementByteSize);
  return AlignedBytesType::get(getContext(), byteSize, align);
}

//===----------------------------------------------------------------------===//
// Shared Logic
//===----------------------------------------------------------------------===//

/// Pack the integer values into a byte array; The input template argument is
/// expected to be either an APInt or an APSInt.
template <typename Int>
static std::vector<uint8_t> packIntegerValues(unsigned width,
                                              ArrayRef<Int> values) {
  static_assert(std::is_same_v<Int, APInt> || std::is_same_v<Int, APSInt>,
                "unexpected integer type");
  unsigned byteSize = llvm::divideCeil(width, CHAR_BIT);
  std::vector<uint8_t> data(values.size() * byteSize, 0);
  for (auto [index, value] : llvm::enumerate(values))
    llvm::StoreIntToMemory(value, data.data() + (index * byteSize), byteSize);

  return data;
}

//===----------------------------------------------------------------------===//
// IntArrayElementsAttr
//===----------------------------------------------------------------------===//

IntArrayElementsAttr IntArrayElementsAttr::get(ShapedType type,
                                               ArrayRef<APInt> values) {
  std::vector<uint8_t> data =
      packIntegerValues(type.getElementTypeBitWidth(), values);
  return llvm::cast<IntArrayElementsAttr>(ArrayElementsAttr::get(data, type));
}

IntArrayElementsAttr IntArrayElementsAttr::get(ShapedType type,
                                               ArrayRef<APSInt> values) {
  std::vector<uint8_t> data =
      packIntegerValues(type.getElementTypeBitWidth(), values);
  return llvm::cast<IntArrayElementsAttr>(ArrayElementsAttr::get(data, type));
}

APInt IntArrayElementsAttr::Iterator::operator*() const {
  unsigned byteWidth = llvm::divideCeil(type.getWidth(), CHAR_BIT);
  APInt value(type.getWidth(), 0, !type.isUnsigned());
  llvm::LoadIntFromMemory(value, (const uint8_t *)base + index * byteWidth,
                          byteWidth);
  return value;
}

auto IntArrayElementsAttr::begin() const -> Iterator {
  return Iterator(llvm::cast<IntegerType>(getElementType()),
                  getRawData().data(), 0);
}

auto IntArrayElementsAttr::end() const -> Iterator {
  return Iterator(llvm::cast<IntegerType>(getElementType()),
                  getRawData().data(), size());
}

bool IntArrayElementsAttr::classof(Attribute attr) {
  if (auto arr = llvm::dyn_cast<ArrayElementsAttr>(attr))
    return llvm::isa<IntegerType>(arr.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// custom<DenseIntArray>
//===----------------------------------------------------------------------===//

ParseResult M::parseDenseIntArray(AsmParser &p, IntArrayElementsAttr &result,
                                  unsigned width,
                                  IntegerType::SignednessSemantics signedness) {
  auto elementType = IntegerType::get(p.getContext(), width, signedness);
  APInt value;
  mlir::OptionalParseResult maybeEmpty = p.parseOptionalInteger(value);
  // Check for an empty array.
  if (!maybeEmpty.has_value()) {
    result = IntArrayElementsAttr::get(ArrayType::get(0, elementType),
                                       ArrayRef<APInt>());
    return success();
  }
  if (maybeEmpty.value())
    return failure();

  SmallVector<APInt> values;
  auto addValue = [&](const APInt &value) {
    values.push_back(value.sextOrTrunc(elementType.getWidth()));
  };
  addValue(value);

  while (succeeded(p.parseOptionalComma())) {
    if (p.parseInteger(value))
      return failure();
    addValue(value);
  }
  result = IntArrayElementsAttr::get(ArrayType::get(values.size(), elementType),
                                     values);
  return success();
}

void M::printDenseIntArray(AsmPrinter &p, Operation *op,
                           IntArrayElementsAttr result, unsigned width,
                           IntegerType::SignednessSemantics) {
  llvm::interleaveComma(result, p);
}

//===----------------------------------------------------------------------===//
// FloatArrayElementsAttr
//===----------------------------------------------------------------------===//

FloatArrayElementsAttr FloatArrayElementsAttr::get(ShapedType type,
                                                   ArrayRef<APFloat> values) {
  SmallVector<APInt> intVals;
  intVals.reserve(values.size());

  unsigned bitWidth = type.getElementTypeBitWidth();
  unsigned bitsToShift = 0;

  auto floatType = llvm::cast<FloatType>(type.getElementType());
  unsigned semBitWidth =
      llvm::APFloat::getSizeInBits(floatType.getFloatSemantics());
  if (bitWidth < semBitWidth)
    bitWidth = semBitWidth;

  bitsToShift = bitWidth - semBitWidth;

  for (const APFloat &value : values) {
    if (floatType.isBF16()) {
      // TODO we need to investigate why BF16 returns a 32-bit width int #33856
      intVals.push_back(value.bitcastToAPInt());
    } else {
      // Extend FloatTF32 bits to 32 and left shift 32 - 19 = 13 bits; Other
      // types, this is no-op.
      intVals.push_back(value.bitcastToAPInt().zext(bitWidth).shl(bitsToShift));
    }
  }

  std::vector<uint8_t> rawData =
      packIntegerValues(type.getElementTypeBitWidth(), ArrayRef(intVals));
  return llvm::cast<FloatArrayElementsAttr>(
      ArrayElementsAttr::get(rawData, type));
}

FloatArrayElementsAttr FloatArrayElementsAttr::get(ArrayRef<APFloat> values,
                                                   Type elementType) {
  auto shapedType = ArrayType::get(values.size(), elementType);
  return FloatArrayElementsAttr::get(shapedType, values);
}

APFloat FloatArrayElementsAttr::Iterator::operator*() const {
  FloatType type = this->type;
  unsigned byteWidth = llvm::divideCeil(type.getWidth(), CHAR_BIT);
  APInt intVal(type.getWidth(), 0);
  llvm::LoadIntFromMemory(intVal, (const uint8_t *)base + index * byteWidth,
                          byteWidth);
  return APFloat(type.getFloatSemantics(), intVal);
}

auto FloatArrayElementsAttr::begin() const -> Iterator {
  return Iterator(llvm::cast<FloatType>(getElementType()), getRawData().data(),
                  0);
}

auto FloatArrayElementsAttr::end() const -> Iterator {
  return Iterator(llvm::cast<FloatType>(getElementType()), getRawData().data(),
                  size());
}

bool FloatArrayElementsAttr::classof(Attribute attr) {
  if (auto arr = llvm::dyn_cast<ArrayElementsAttr>(attr))
    return llvm::isa<FloatType>(arr.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// IndexArrayElementsAttr
//===----------------------------------------------------------------------===//

IndexArrayElementsAttr IndexArrayElementsAttr::get(ShapedType type,
                                                   ArrayRef<int64_t> values) {
  ArrayRef<uint8_t> data(reinterpret_cast<const uint8_t *>(values.data()),
                         values.size() * sizeof(int64_t));
  return llvm::cast<IndexArrayElementsAttr>(ArrayElementsAttr::get(data, type));
}

bool IndexArrayElementsAttr::classof(Attribute attr) {
  if (auto arr = llvm::dyn_cast<ArrayElementsAttr>(attr))
    return llvm::isa<IndexType>(arr.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Convert a `DenseElementsAttr` to an `ArrayElementsAttr`. Pass through any
/// other kind of attribute. This should be the only place where the splatness
/// and bitpacked-ness of the attribute are handled.
Attribute M::convertDenseElements(Attribute attr) {
  auto denseElements = dyn_cast<DenseElementsAttr>(attr);
  if (!denseElements || !denseElements.getElementType().isIntOrFloat())
    return attr;
  if (denseElements.getType().getElementTypeBitWidth() % 8 == 0) {
    ArrayRef<char> charData = denseElements.getRawData();
    ArrayRef<uint8_t> data(reinterpret_cast<const uint8_t *>(charData.data()),
                           charData.size());
    // If the data is byte-aligned and is not splat, just pass it along.
    if (!denseElements.isSplat())
      return ArrayElementsAttr::get(data, denseElements.getType());

    // Replicate the splat.
    std::vector<uint8_t> replicated(data.size() * denseElements.size(), 0);
    for (unsigned i = 0, e = denseElements.size(); i < e; ++i)
      memcpy(replicated.data() + i * data.size(), data.data(), data.size());
    return ArrayElementsAttr::get(replicated, denseElements.getType());
  }

  // Unpack the data.
  if (llvm::isa<FloatType>(denseElements.getElementType())) {
    auto values = llvm::to_vector(denseElements.getValues<APFloat>());
    return FloatArrayElementsAttr::get(denseElements.getType(), values);
  }
  auto values = llvm::to_vector(denseElements.getValues<APInt>());
  return IntArrayElementsAttr::get(denseElements.getType(), values);
}

ElementsAttr M::getInlineAttrForTensorDataCopy(ShapedType type,
                                               ArrayRef<char> data) {
  return ArrayElementsAttr::get(
      {reinterpret_cast<const uint8_t *>(data.data()), data.size()}, type);
}

ElementsAttr M::getAttrForTensorDataCopy(
    ShapedType type, StringRef bufferName, ArrayRef<char> data,
    DenseResourceElementsHandleManager &resourceManager) {
  if (!shouldUseOutOfLineAttrStorage(type.getNumElements()))
    return getInlineAttrForTensorDataCopy(type, data);

  // Use a default alignment based on the element type.
  size_t elementByteAlign = llvm::PowerOf2Ceil(
      llvm::divideCeil(type.getElementTypeBitWidth(), CHAR_BIT));
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
      data, elementByteAlign);
  return DenseResourceElementsAttr::get(
      type, resourceManager.insert(bufferName, std::move(blob)));
}

SmallVector<int64_t> M::getIntBlob(IntArrayElementsAttr intElemsAttr) {
  return llvm::map_to_vector(
      intElemsAttr.getValues(),
      [](const llvm::APInt &dim) { return dim.getSExtValue(); });
}

SmallVector<float> M::getFloatBlob(FloatArrayElementsAttr floatElemsAttr) {
  return llvm::map_to_vector(
      floatElemsAttr.getValues(),
      [](const llvm::APFloat &dim) { return dim.convertToFloat(); });
}

//===----------------------------------------------------------------------===//
// AlignedBytesAttr
//===----------------------------------------------------------------------===//

/// Interns data into allocator with align (or 1 if no align constraint given).
/// The AlignedBytesAttr's getData will thus return a pointer respecting the
/// requested alignment.
static ArrayRef<uint8_t>
copyIntoBytes(mlir::StorageUniquer::StorageAllocator &allocator,
              ArrayRef<uint8_t> data, std::optional<uint64_t> align) {
  auto *ptr = static_cast<uint8_t *>(
      allocator.allocate(data.size(), align ? *align : sizeof(uint8_t)));
  std::uninitialized_copy(data.begin(), data.end(), ptr);
  return {ptr, data.size()};
}

/// (Just to make clear we'll be converting from decoded hex strings directly
///  to vectors of uint8_t below.)
static_assert(sizeof(uint8_t) == sizeof(char));

/// Parses a hex string into data. The empty string denotes no data.
static ParseResult parseAlignedBytesData(AsmParser &p,
                                         SmallVector<uint8_t> &data) {
  std::string encoded;
  if (p.parseString(&encoded))
    return failure();
  if (encoded.empty())
    return success();
  StringRef encodedRef = encoded;
  std::string decoded;
  auto startLoc = p.getCurrentLocation();
  if (!encodedRef.consume_front("0x") || encodedRef.empty() ||
      (encodedRef.size() & 1) || !llvm::tryGetFromHex(encodedRef, decoded)) {
    return p.emitError(startLoc, "invalid hex string for aligned_bytes");
  }
  data.resize(decoded.size());
  memcpy(data.data(), decoded.data(), decoded.size());
  return success();
}

/// Print data as a hex string. The empty data array is printed as the empty
/// string.
static void printAlignedBytesData(AsmPrinter &p, ArrayRef<uint8_t> data) {
  if (data.empty()) {
    p << "\"\"";
    return;
  }
  p << "\"0x";
  StringRef decodedRef(reinterpret_cast<const char *>(data.data()),
                       data.size());
  p << llvm::toHex(decodedRef);
  p << "\"";
}

/// Verifies the attribute's align constraint is sensible.
LogicalResult
AlignedBytesAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<uint8_t> data, uint64_t align) {
  if (!llvm::isPowerOf2_64(align))
    return emitError() << "alignment must be a power of two.";
  return success();
}

/// ElementsAttrInterface: returns the underlying data.
FailureOr<const uint8_t *>
AlignedBytesAttr::try_value_begin_impl(OverloadToken<uint8_t>) const {
  return getData().data();
}

/// ElementsAttrInterface: returns the implied buffer type.
ShapedType AlignedBytesAttr::getType() const {
  return ArrayType::get(
      static_cast<int64_t>(getData().size()),
      IntegerType::get(getContext(), 8, IntegerType::Unsigned));
}

/// HasAlignedBytesInterface::getAlignedBytesType
AlignedBytesType AlignedBytesAttr::getAlignedBytesType() const {
  return AlignedBytesType::get(
      getContext(), static_cast<uint64_t>(getData().size()), getAlign());
}

//===----------------------------------------------------------------------===//
// DataLayout
//===----------------------------------------------------------------------===//

/// Construct the default data layout and then overwrite the entries when
/// parsing the data layout string.
DataLayout::DataLayout(StringRef dlSpecStr)
    : intAbiAlign{{1, 1}, {8, 1}, {16, 2}, {32, 4}, {64, 4}},
      fpAbiAlign{{16, 2}, {32, 4}, {64, 8}, {128, 16}},
      vecAbiAlign{{64, 8}, {128, 16}}, dlSpecStr(dlSpecStr),
      ptrInfos(kNumAddrSpaces) {
  ptrInfos[0] = {.ptrWidth = 64, .ptrAbiAlign = 8};
}

/// Checked version of split to ensure mandatory subparts.
static ErrorOr<std::pair<StringRef, StringRef>> checkedSplit(StringRef str,
                                                             char sep) {
  assert(!str.empty() && "expected non-empty string");
  std::pair<StringRef, StringRef> result = str.split(sep);
  if (result.second.empty() && result.first != str)
    return Error("trailing separator in datalayout string");
  if (!result.second.empty() && result.first.empty())
    return Error("expected token before separator in data layout string");
  return result;
}

/// Get an integer, including error checks.
static ErrorOr<int32_t> getInt(StringRef repr) {
  int32_t result;
  if (repr.getAsInteger(10, result))
    return Error("not an integer");
  return result;
}

/// Get an integer representing the number of bits and convert it into bytes.
/// Error out of not a byte width multiple.
static ErrorOr<int32_t> getIntInBytes(StringRef repr) {
  ErrorOr<int32_t> result = getInt(repr);
  if (result.isError())
    return result.takeError();
  if (*result % CHAR_BIT)
    return Error("number of bits must be a byte width multiple");
  return *result / CHAR_BIT;
}

ErrorOr<DataLayout> DataLayout::parse(StringRef desc) {
  DataLayout dl(desc);
  if (auto err = dl.parse())
    return err.takeError();

  return dl;
}

/// Set the ABI alignment in the data layout entry list, creating an entry or
/// overwriting an existing entry.
static void
setABIAlignment(SmallVectorImpl<std::pair<int32_t, int32_t>> &entries,
                int32_t size, int32_t abiAlign) {
  auto it = llvm::partition_point(
      entries, [&](const std::pair<int32_t, int32_t> &entry) {
        return entry.first < size;
      });
  if (it != entries.end() && it->first == size) {
    // Update the alignment in-place.
    it->second = abiAlign;
  } else {
    // Perform a sorted insert.
    entries.insert(it, {size, abiAlign});
  }
}

/// Parse an LLVM data layout specifier. Drop information that our `DataLayout`
/// specification does not track and take the information it needs. This is a
/// copy of the LLVM data layout parser found in `llvm/lib/DataLayout.cpp`.
///
/// See also https://llvm.org/docs/LangRef.html#langref-datalayout.
ErrorOrSuccess DataLayout::parse() {
  StringRef desc = dlSpecStr;
  std::pair<StringRef, StringRef> split;

  while (!desc.empty()) {
    // Split at '-'.
    {
      auto splitOr = checkedSplit(desc, '-');
      if (splitOr.isError())
        return splitOr.takeError();
      split = std::move(*splitOr);
    }
    desc = split.second;

    // Split at ':'.
    {
      auto splitOr = checkedSplit(split.first, ':');
      if (splitOr.isError())
        return splitOr.takeError();
      split = std::move(*splitOr);
    }

    // Aliases used below.
    StringRef &tok = split.first;   // Current token.
    StringRef &rest = split.second; // The rest of the string.

    if (tok == "ni") {
      // Skip non-integral address spaces.
      continue;
    }

    char specifier = tok.front();
    tok = tok.substr(1);

    switch (specifier) {
    case 'E':
      isLittleEndian = false;
      break;
    case 'e':
      isLittleEndian = true;
      break;
    case 'p': {
      // Address space.
      unsigned addrSpace = 0;
      if (!tok.empty()) {
        auto addrSpaceOr = getInt(tok);
        if (addrSpaceOr.isError())
          return addrSpaceOr.takeError();
        addrSpace = std::move(*addrSpaceOr);
      }

      // Size.
      if (rest.empty())
        return Error("missing pointer size specification");
      {
        auto splitOr = checkedSplit(rest, ':');
        if (splitOr.isError())
          return splitOr.takeError();
        split = std::move(*splitOr);
      }
      int32_t pWidth, pAbiAlign;

      {
        auto ptrWidthOr = getInt(tok);
        if (ptrWidthOr.isError())
          return ptrWidthOr.takeError();
        pWidth = std::move(*ptrWidthOr);
      }
      if (!pWidth)
        return Error("invalid pointer size of 0 bytes");

      // ABI alignment.
      if (rest.empty())
        return Error("missing pointer ABI alignment specification");
      {
        auto splitOr = checkedSplit(rest, ':');
        if (splitOr.isError())
          return splitOr.takeError();
        split = std::move(*splitOr);
      }
      {
        auto ptrAbiAlignOr = getIntInBytes(tok);
        if (ptrAbiAlignOr.isError())
          return ptrAbiAlignOr.takeError();
        pAbiAlign = std::move(*ptrAbiAlignOr);
      }
      if (!llvm::isPowerOf2_32(pAbiAlign))
        return Error("pointer ABI alignment must be a power of 2");

      if (addrSpace >= ptrInfos.size())
        ptrInfos.resize(addrSpace + 1);

      ptrInfos[addrSpace] =
          PointerLayout{.ptrWidth = pWidth, .ptrAbiAlign = pAbiAlign};

      // Skip pointer preferred alignment and index size.
      break;
    }
    case 'i':
    case 'v':
    case 'f':
    case 'a': {
      // Bit size.
      unsigned size = 0;
      if (!tok.empty()) {
        if (specifier == 'a')
          return Error("incorrect aggregate alignment specification");
        auto sizeOr = getInt(tok);
        if (sizeOr.isError())
          return sizeOr.takeError();
        size = std::move(*sizeOr);
      }

      // ABI alignment.
      if (rest.empty())
        return Error("missing alignment specification");
      {
        auto splitOr = checkedSplit(rest, ':');
        if (splitOr.isError())
          return splitOr.takeError();
        split = std::move(*splitOr);
      }
      unsigned abiAlign;
      {
        auto abiAlignOr = getIntInBytes(tok);
        if (abiAlignOr.isError())
          return abiAlignOr.takeError();
        abiAlign = std::move(*abiAlignOr);
      }
      // For aggregates, a value of zero means a one-byte alignment
      if (specifier == 'a' && !abiAlign)
        abiAlign = 1;
      if (!abiAlign)
        return Error("ABI alignment specification cannot be zero");

      if (!llvm::isPowerOf2_64(abiAlign))
        return Error("ABI alignment must be a power of 2");
      if (specifier == 'i' && size == 8 && abiAlign != 1)
        return Error("i8 must be naturally aligned");

      // Skip preferred alignment.

      switch (specifier) {
      case 'i':
        setABIAlignment(intAbiAlign, size, abiAlign);
        break;
      case 'f':
        setABIAlignment(fpAbiAlign, size, abiAlign);
        break;
      case 'v':
        setABIAlignment(vecAbiAlign, size, abiAlign);
        break;
      case 'a':
        structAbiAlign = abiAlign;
        break;
      default:
        llvm_unreachable("unknown specifier");
      }
      break;
    }
    case 'n':
      // Skip native integer types.
      break;
    case 'S':
      // Skip stack natural alignment.
      break;
    case 'F':
      // Skip function pointer alignment.
      break;
    case 'P':
      // Skip function address space.
      break;
    case 'A': {
      // Default alloca address space.
      auto allocaAddrSpaceOr = getInt(tok);
      if (allocaAddrSpaceOr.isError())
        return allocaAddrSpaceOr.takeError();
      allocaAddrSpace = std::move(*allocaAddrSpaceOr);
      break;
    }
    case 'G':
      // Skip default address space for global variables.
      break;
    case 'm':
      // Skip mangling mode.
      break;
    default:
      return Error("unknown specifier in data layout string");
    }
  }

  if (intAbiAlign.empty())
    return Error(
        "data layout specification requires at least one integer entry");

  return success();
}

namespace M {
/// Provide the ability to hash data layout specifications.
static llvm::hash_code hash_value(const DataLayout &dl) {
  return hash_value(dl.toString());
}

/// Allow the attribute parser to print a data layout specification.
static raw_ostream &operator<<(raw_ostream &os, const DataLayout &dl) {
  return os << '"' << dl.toString() << '"';
}

/// Compare data layouts by their string specification. Note that multiple
/// strings can produce the same data layout.
static bool operator==(const DataLayout &lhs, const DataLayout &rhs) {
  return lhs.toString() == rhs.toString();
}
} // namespace M

int32_t DataLayout::getIntegerABIAlign(int32_t bitwidth) const {
  // Binary search for a corresponding entry by bitwidth. This returns the first
  // entry whose bitwidth is greater than or equal to the type bitwidth.
  auto it = llvm::partition_point(
      intAbiAlign, [&](const std::pair<int32_t, int32_t> &entry) {
        return entry.first < bitwidth;
      });
  assert(!intAbiAlign.empty() && "expected at least one integer entry");

  // Use the alignment of the next largest integer type. If there wasn't one,
  // use the alignment of the largest integer type.
  if (it == intAbiAlign.end())
    --it;
  return it->second;
}

int32_t DataLayout::getFloatABIAlign(int32_t bitwidth) const {
  assert(bitwidth != 0 && "Zero bit float");

  // Binary search for a corresponding entry by float bitwidth.
  auto it = llvm::partition_point(
      fpAbiAlign, [&](const std::pair<int32_t, int32_t> &entry) {
        return entry.first < bitwidth;
      });

  // If we found an entry, use it.
  if (it != fpAbiAlign.end() && it->first == bitwidth)
    return it->second;

  // Otherwise, the default alignment is the power of 2 equal to or greater than
  // the size rounded up to the nearest byte.
  return llvm::PowerOf2Ceil(llvm::divideCeil(bitwidth, CHAR_BIT));
}

int32_t DataLayout::getVectorABIAlign(int32_t numElts,
                                      int32_t eltBitWidth) const {
  int32_t size = numElts * eltBitWidth;
  // Binary search for a corresponding entry by vector bitwidth.
  auto it = llvm::partition_point(
      vecAbiAlign, [&](const std::pair<int32_t, int32_t> &entry) {
        return entry.first < size;
      });

  // If we found an entry for the alignment of a vector of this size, use it.
  if (it != vecAbiAlign.end() && it->first == size)
    return it->second;

  // Otherwise, the default alignment is the power of 2 equal to or greater than
  // the size rounded up to the nearest byte.
  return llvm::PowerOf2Ceil(llvm::divideCeil(size, CHAR_BIT));
}

int32_t DataLayout::getStructABIAlign() const { return structAbiAlign; }

//===----------------------------------------------------------------------===//
// TargetInfoAttr
//===----------------------------------------------------------------------===//

/// The dialect attribute name used to attached target info to a module.
static constexpr llvm::StringLiteral targetInfoAttrName = "M.target_info";

StringLiteral M::stringifyRelocationModel(llvm::Reloc::Model model) {
  switch (model) {
  case llvm::Reloc::Static:
    return "static";
  case llvm::Reloc::PIC_:
    return "pic";
  case llvm::Reloc::DynamicNoPIC:
    return "dynamic-no-pic";
  case llvm::Reloc::ROPI:
    return "ropi";
  case llvm::Reloc::RWPI:
    return "rwpi";
  case llvm::Reloc::ROPI_RWPI:
    return "ropi-rwpi";
  }
  llvm_unreachable("invalid relocation model");
}

ErrorOr<llvm::Reloc::Model> M::symbolizeRelocationModel(StringRef str) {
  return llvm::StringSwitch<ErrorOr<llvm::Reloc::Model>>(str)
      .Case("static", llvm::Reloc::Static)
      .Case("pic", llvm::Reloc::PIC_)
      .Case("dynamic-no-pic", llvm::Reloc::DynamicNoPIC)
      .Case("ropi", llvm::Reloc::ROPI)
      .Case("rwpi", llvm::Reloc::RWPI)
      .Case("ropi-rwpi", llvm::Reloc::ROPI_RWPI)
      .Default(Error("invalid relocation-model '" + str +
                     "', expected one of: `static`, `pic`, `dynamic-no-pic`, "
                     "`ropi`, `rwpi`, or `ropi-rwpi`"));
}

TargetInfoAttr M::getTargetInfo(ModuleOp module) {
  return module->getAttrOfType<TargetInfoAttr>(targetInfoAttrName);
}

void M::setTargetInfo(ModuleOp module, TargetInfoAttr target) {
  assert(!getTargetInfo(module) && "module already has a target specification");
  module->setAttr(targetInfoAttrName, target);
}

TargetInfoAttr M::lookupTargetInfo(Operation *from) {
  if (auto module = dyn_cast<ModuleOp>(from))
    return getTargetInfo(module);
  auto module = from->getParentOfType<ModuleOp>();
  if (!module)
    return {};
  return getTargetInfo(module);
}

void M::eraseTargetInfo(ModuleOp module) {
  [[maybe_unused]] Attribute target = module->removeAttr(targetInfoAttrName);
  assert(target && "module did not have a target to remove");
}

ErrorOr<TargetInfoAttr>
M::getTargetInfoFor(MLIRContext *ctx, StringRef targetTriple, StringRef arch,
                    StringRef features, StringRef tuneCpu,
                    StringRef acceleratorArch, llvm::Reloc::Model relocModel,
                    StringRef abi) {
  std::string errorMessage;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(targetTriple), errorMessage);
  if (!target)
    return Error("could not construct host target info: " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(targetTriple), arch, features, /*Options=*/{},
      /*RM=*/relocModel));
  if (!machine)
    return Error("failed to create target machine for data layout lookup");

  ErrorOr<DataLayout> dl =
      DataLayout::parse(machine->createDataLayout().getStringRepresentation());
  assert(!dl.isError() && "failed to parse LLVM data layout?");

  // Expand the explicit feature delta against the CPU model so TargetInfoAttr
  // stores the fully resolved feature set. Without this, hasFeature() returns
  // false for features that are enabled by CPU model defaults but absent from
  // the explicit delta (e.g. avx512f on znver4 when -avx512f is not passed).
  std::string resolvedFeatures = features.str();
  if (!arch.empty()) {
    ErrorOr<std::vector<std::string>> baseOr =
        M::getFeatures(targetTriple, arch);
    ErrorOr<DecodedFeatures> deltaOr = M::decodeFeatures(features);
    if (!baseOr.isError() && !deltaOr.isError()) {
      // Build the resolved feature set while preserving the user's ordering:
      // 1. Start with the user's explicitly enabled features (in their order).
      // 2. Append CPU model defaults not mentioned in the user's delta — these
      //    are features LLVM enables silently (e.g. avx512f on znver4) that
      //    must be visible to hasFeature().
      // 3. Disabled features go at the end (handled by encodeFeatures).
      auto isMentioned = [&](StringRef f) {
        return llvm::is_contained(deltaOr->enabled, f) ||
               llvm::is_contained(deltaOr->disabled, f);
      };

      std::vector<std::string> resolved = deltaOr->enabled;
      for (StringRef f : *baseOr)
        if (!isMentioned(f))
          resolved.push_back(f.str());

      TargetInfo ti(llvm::Triple(targetTriple), arch.str(), std::move(resolved),
                    std::move(deltaOr->disabled));
      resolvedFeatures = encodeFeatures(ti);
    }
  }

  int32_t pointerBitWidth = dl->getPointerBitWidth();
  // The plugin name is only set on targets built from Mojo; targets built here
  // get the default.
  return TargetInfoAttr::get(ctx, llvm::Triple(targetTriple), arch, "default",
                             resolvedFeatures, std::move(*dl),
                             machine->getRelocationModel(),
                             simdWidthFromFeatures(StringRef(resolvedFeatures)),
                             pointerBitWidth, tuneCpu, acceleratorArch, abi);
}

bool M::isKnownTargetFeature(StringRef feature, StringRef targetTriple) {
  // LLVM has no combined feature table, so union the current target (covers
  // whatever we compile for, GPU included) with host x86-64 + aarch64 + riscv64
  // (cover the cross-arch CPU queries the stdlib `has_*` wrappers make).  The
  // OS/environment does not affect the feature table, so use `unknown-unknown`.
  // riscv64 subsumes riscv32's extension names apart from the `32bit`/`64bit`
  // pair, and `CompilationTarget` exposes those as `is_rv32()`/`is_rv64()`.
  static constexpr StringLiteral kUnionTriples[] = {"x86_64-unknown-unknown",
                                                    "aarch64-unknown-unknown",
                                                    "riscv64-unknown-unknown"};

  static llvm::sys::SmartMutex<true> mu;
  // triple -> its full set of valid feature names. An entry that failed to
  // build (backend not linked) is present but empty.
  static llvm::StringMap<llvm::StringSet<>> cache;

  llvm::sys::SmartScopedLock<true> lock(mu);

  auto namesFor = [&](StringRef triple) -> const llvm::StringSet<> & {
    auto [it, inserted] = cache.try_emplace(triple);
    if (!inserted)
      return it->second;
    llvm::StringSet<> &names = it->second;
    std::string err;
    llvm::Triple tt(triple);
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(tt, err);
    if (!target)
      return names; // Backend not linked; leave empty.
    std::unique_ptr<llvm::MCSubtargetInfo> sti(
        target->createMCSubtargetInfo(tt, /*CPU=*/"", /*Features=*/""));
    if (!sti)
      return names;
    for (const llvm::SubtargetFeatureKV &kv : sti->getAllProcessorFeatures())
      names.insert(kv.key());
    return names;
  };

  bool anyTableAvailable = false;
  auto check = [&](StringRef triple) -> bool {
    const llvm::StringSet<> &names = namesFor(triple);
    if (names.empty())
      return false;
    anyTableAvailable = true;
    return names.contains(feature);
  };

  if (check(targetTriple))
    return true;
  for (StringLiteral triple : kUnionTriples) {
    if (check(triple))
      return true;
  }

  // If no backend was available to validate against, don't claim the name is
  // unknown - ex falso sequitur quodlibet, "from falsehood, anything follows".
  return !anyTableAvailable;
}

ErrorOr<TargetInfo> M::toRuntimeTargetInfo(TargetInfoAttr targetInfoAttr) {
  auto errOr = decodeFeatures(targetInfoAttr.getFeatures());
  if (errOr)
    return errOr.takeError();
  return TargetInfo(targetInfoAttr.getTriple(),
                    std::string(targetInfoAttr.getArch()),
                    std::move(errOr->enabled), std::move(errOr->disabled),
                    std::string(targetInfoAttr.getAbi()));
}

/// Returns attribute representing runtime target info.
TargetInfoAttr M::fromRuntimeTargetInfo(MLIRContext *ctx,
                                        const TargetInfo &runtimeTargetInfo) {
  return TargetInfoAttr::get(
      ctx, runtimeTargetInfo.triple, runtimeTargetInfo.arch,
      /*stdlib_plugin=*/"default", encodeFeatures(runtimeTargetInfo),
      /*data_layout=*/{}, /*relocation_model=*/llvm::Reloc::Static,
      /*simd_bit_width=*/0, /*index_width=*/std::nullopt,
      /*tune_cpu=*/{}, /*accelerator_arch=*/{}, runtimeTargetInfo.abi);
}

namespace mlir {
/// Allow target triples to be parsed by MLIR.
template <>
struct FieldParser<llvm::Triple> {
  static FailureOr<llvm::Triple> parse(AsmParser &p) {
    std::string tripleStr;
    if (failed(p.parseString(&tripleStr)))
      return failure();
    return llvm::Triple(tripleStr);
  }
};

/// Allow data layout specifications to be parsed.
template <>
struct FieldParser<M::DataLayout> {
  static FailureOr<M::DataLayout> parse(AsmParser &p) {
    std::string dlSpecStr;
    llvm::SMLoc loc = p.getCurrentLocation();
    if (failed(p.parseString(&dlSpecStr)))
      return failure();
    ErrorOr<M::DataLayout> dl = M::DataLayout::parse(dlSpecStr);
    if (dl.isError())
      return p.emitError(loc, dl.getError());
    return dl.takeValue();
  }
};

/// Allow `llvm::Reloc::Model` to be parsed by MLIR.
template <>
struct FieldParser<llvm::Reloc::Model> {
  static FailureOr<llvm::Reloc::Model> parse(AsmParser &p) {
    std::string str;
    if (failed(p.parseString(&str)))
      return failure();

    ErrorOr<llvm::Reloc::Model> model = symbolizeRelocationModel(str);
    if (model.isError())
      return failure();
    return *model;
  }
};
} // namespace mlir

namespace llvm {
/// Provide the ability to hash triples for attribute uniquing.
static hash_code hash_value(const llvm::Triple &triple) {
  return hash_value(triple.normalize());
}

/// Allow the attribute printer to print a target triple.
static raw_ostream &operator<<(raw_ostream &os, const llvm::Triple &triple) {
  return os << '"' << triple.normalize() << '"';
}

/// Allow the attribute printer to print a relocation model.
static raw_ostream &operator<<(raw_ostream &os, llvm::Reloc::Model model) {
  return os << '"' << stringifyRelocationModel(model) << '"';
}
} // namespace llvm

//===----------------------------------------------------------------------===//
// DeviceRefAttr
//===----------------------------------------------------------------------===//

std::string DeviceRefAttr::toString() const {
  return toRuntimeDeviceRef(*this)->toString();
}

ErrorOr<DeviceRef> M::toRuntimeDeviceRef(DeviceRefAttr deviceRefAttr) {
  DeviceRef result;
  result.label = deviceRefAttr.getLabel();
  result.id = deviceRefAttr.getId();
  return result;
}

DeviceRefAttr M::fromRuntimeDeviceRef(MLIRContext *ctx,
                                      const DeviceRef &runtimeDeviceRef) {
  return DeviceRefAttr::get(ctx, runtimeDeviceRef.label, runtimeDeviceRef.id);
}

DeviceRefAttr DeviceRefAttr::getDefault(MLIRContext *ctx) {
  return DeviceRefAttr::get(ctx, kCPULabel, 0);
}

DeviceRefAttr M::getHostDevice(MLIRContext *ctx) {
  return DeviceRefAttr::getDefault(ctx);
}

DeviceRefAttr M::getHostDevice(Operation *op) {
  return getHostDevice(op->getContext());
}

bool M::isHostPlaced(DeviceRefAttr device) {
  return device.getLabel() == kCPULabel;
}

//===----------------------------------------------------------------------===//
// DeviceSpecAttr
//===----------------------------------------------------------------------===//

ErrorOr<DeviceSpec> M::toRuntimeDeviceSpec(DeviceSpecAttr deviceAttr) {
  DeviceSpec result;
  auto refOr = toRuntimeDeviceRef(deviceAttr.getRef());
  if (refOr)
    return refOr.takeError();
  result.ref = *refOr;
  auto targetOr = toRuntimeTargetInfo(deviceAttr.getTarget());
  if (targetOr)
    return targetOr.takeError();
  result.target = *targetOr;
  return result;
}

DeviceSpecAttr M::fromRuntimeDeviceSpec(MLIRContext *ctx,
                                        const DeviceSpec &runtimeDeviceSpec) {
  return DeviceSpecAttr::get(
      ctx, fromRuntimeDeviceRef(ctx, runtimeDeviceSpec.ref),
      fromRuntimeTargetInfo(ctx, runtimeDeviceSpec.target));
}

//===----------------------------------------------------------------------===//
// DeviceSpecsAttr
//===----------------------------------------------------------------------===//

static ParseResult
parseDeviceSpecAttrArray(AsmParser &p, SmallVector<DeviceSpecAttr> &devices) {
  return p.parseCommaSeparatedList(
      AsmParser::Delimiter::Square, [&]() -> ParseResult {
        auto deviceSpecAttr = llvm::dyn_cast_if_present<DeviceSpecAttr>(
            DeviceSpecAttr::parse(p, /*odsType=*/{}));
        if (!deviceSpecAttr)
          return failure();
        devices.emplace_back(deviceSpecAttr);
        return success();
      });
}

static void printDeviceSpecAttrArray(AsmPrinter &p,
                                     ArrayRef<DeviceSpecAttr> devices) {
  p << "[";
  llvm::interleaveComma(devices, p.getStream(),
                        [&p](const DeviceSpecAttr &attr) { attr.print(p); });
  p << "]";
}

ErrorOr<DeviceSpecAttr>
DeviceSpecCollectionAttr::findDeviceSpec(DeviceRefAttr deviceReference) const {
  auto itr =
      llvm::find_if(getDevices(), [&deviceReference](DeviceSpecAttr device) {
        return device.getRef() == deviceReference;
      });
  if (itr == getDevices().end())
    return Error(Twine("no such device spec for reference '" +
                       deviceReference.toString() + "'"));
  return *itr;
}

DeviceSpecAttr DeviceSpecCollectionAttr::getHostDeviceSpec() const {
  auto specOr = findDeviceSpec(getHost());
  assert(!specOr.isError() && "no such host device spec");
  return *specOr;
}

LogicalResult DeviceSpecCollectionAttr::verify(
    function_ref<mlir::InFlightDiagnostic()> emitError, DeviceRefAttr host,
    ArrayRef<M::DeviceSpecAttr> devices) {
  llvm::DenseSet<DeviceRefAttr> deviceReferences;
  for (const auto &deviceSpec : devices) {
    if (!deviceReferences.insert(deviceSpec.getRef()).second)
      return emitError()
             << "#M.device_spec_collection contains duplicate device specs "
                "for the device reference '" +
                    deviceSpec.getRef().toString() + "'.";
  }
  if (!deviceReferences.contains(host))
    return emitError()
           << "#M.device_spec_collection does not contain a device spec with "
              "the host device reference '" +
                  host.toString() + "'.";
  return success();
}

ErrorOr<DeviceSpecCollection>
M::toRuntimeDeviceSpecs(DeviceSpecCollectionAttr deviceSpecCollectionAttr) {
  DeviceSpecCollection result;
  auto refOr = toRuntimeDeviceRef(deviceSpecCollectionAttr.getHost());
  if (refOr)
    return refOr.takeError();
  result.host = *refOr;

  result.devices.reserve(deviceSpecCollectionAttr.getDevices().size());
  for (auto &deviceAttr : deviceSpecCollectionAttr.getDevices()) {
    auto deviceOr = toRuntimeDeviceSpec(deviceAttr);
    if (deviceOr)
      return deviceOr.takeError();
    result.devices.emplace_back(*deviceOr);
  }
  return result;
}

DeviceSpecCollectionAttr M::fromRuntimeDeviceSpecs(
    MLIRContext *ctx, const DeviceSpecCollection &runtimeDeviceSpecCollection) {
  std::vector<DeviceSpecAttr> deviceAttrs;
  deviceAttrs.reserve(runtimeDeviceSpecCollection.devices.size());
  for (auto &device : runtimeDeviceSpecCollection.devices)
    deviceAttrs.emplace_back(fromRuntimeDeviceSpec(ctx, device));
  return DeviceSpecCollectionAttr::get(
      ctx, fromRuntimeDeviceRef(ctx, runtimeDeviceSpecCollection.host),
      deviceAttrs);
}

//===----------------------------------------------------------------------===//
// InOutSignatureAttr
//===----------------------------------------------------------------------===//

std::string
signatureString(ArrayRef<InOutSignatureAttr::InOutSemantics> signature) {
  std::string str;
  str.resize(signature.size());
  for (size_t i = 0, n = signature.size(); i < n; ++i)
    str[i] = static_cast<char>(signature[i]);
  return str;
}

InOutSignatureAttr InOutSignatureAttr::get(MLIRContext *context,
                                           ArrayRef<InOutSemantics> signature) {
  return InOutSignatureAttr::get(context, signatureString(signature));
}

InOutSignatureAttr
InOutSignatureAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context,
                               ArrayRef<InOutSemantics> signature) {
  std::string str = signatureString(signature);
  if (failed(verify(emitError, str)))
    return {};
  return InOutSignatureAttr::get(context, str);
}

LogicalResult
InOutSignatureAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           StringRef signature) {
  for (size_t i = 0, n = signature.size(); i < n; ++i) {
    switch (signature[i]) {
    case kNone:
    case kIn:
    case kOut:
    case kMut:
      break;
    default:
      return emitError() << "invalid #M.inout_sig at operand " << i << ".";
    }
  }
  return success();
}

InOutSignatureAttr InOutSignatureAttr::remove(const llvm::BitVector &toRemove) {
  SmallVector<InOutSignatureAttr::InOutSemantics> newSemantics;

  for (size_t i : llvm::seq(size()))
    if (!toRemove.test(i))
      newSemantics.push_back((*this)[i]);

  return InOutSignatureAttr::get(getContext(), newSemantics);
}

InOutSignatureAttr
InOutSignatureAttr::append(ArrayRef<InOutSemantics> newArgs) {
  SmallVector<InOutSignatureAttr::InOutSemantics> newSemantics;
  newSemantics.reserve(size() + newArgs.size());

  for (size_t i : llvm::seq(size()))
    newSemantics.push_back((*this)[i]);

  newSemantics.append(newArgs.begin(), newArgs.end());
  return InOutSignatureAttr::get(getContext(), newSemantics);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "Support/MDialect/MAttrs.cpp.inc"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
