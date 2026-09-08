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

#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Interpreter/InterpreterState.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

LogicalResult
POP::ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                       TypedAttr size, Type elementType) {
  if (!llvm::isa<IndexType>(size.getType()))
    return emitError() << "expected size expression to be index type";
  return success();
}

std::optional<int64_t> POP::ArrayType::getResolvedSize() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

POP::ArrayType POP::ArrayType::get(TypedAttr size, Type elementType) {
  MLIRContext *ctx = size.getContext();
  return get(ctx, size, elementType);
}

POP::ArrayType POP::ArrayType::get(int64_t size, Type elementType) {
  return get(Builder(elementType.getContext()).getIndexAttr(size), elementType);
}

POP::ArrayType POP::ArrayType::get(ValueRange elements) {
  assert(!elements.empty() && "expected non-empty elements");
  auto firstElement = elements.front();
  assert(llvm::all_of(elements,
                      [firstType = firstElement.getType()](Value v) {
                        return v.getType() == firstType;
                      }) &&
         "expected same element types");
  return get(elements.size(), firstElement.getType());
}

/// The size of the array is the number of elements times the size of each
/// aligned element.
std::optional<int64_t>
POP::ArrayType::getTypeSize(TargetInfoAttr target) const {
  std::optional<int64_t> size = getResolvedSize();
  if (!size)
    return {};

  Type elementType = getElementType();
  std::optional<int64_t> elementAllocSize =
      DataLayoutInterface::getTypeAllocSize(target, elementType);
  if (!elementAllocSize)
    return {};

  return *size * *elementAllocSize;
}

/// The alignment of the array is the alignment of the element type.
std::optional<int64_t>
POP::ArrayType::getTypeAlign(TargetInfoAttr target) const {
  Type elementType = getElementType();
  return DataLayoutInterface::getTypeABIAlign(target, elementType);
}

ErrorOrSuccess POP::ArrayType::writeTo(TypedAttr value, int64_t addr,
                                       InterpreterState &state) const {

  // Store each element spaced apart by padding according to its alignment.
  std::optional<int64_t> stride = DataLayoutInterface::getTypeAllocSize(
      state.getTarget(), getElementType());
  if (!stride)
    return Error("failed to get array element stride");
  int64_t offset = *stride;

  auto tv = dyn_cast_if_present<POP::ArrayAttr>(value);
  if (!tv) {
    return Error("array not a writeable type, got " + mlir::debugString(value) +
                 " instead");
  }

  for (TypedAttr value : tv.getValues()) {
    ErrorOrSuccess result = state.writeAttributeToMemory(addr, value);
    if (result.isError())
      return result.takeError();
    addr += offset;
  }
  return success();
}

ErrorOr<TypedAttr> POP::ArrayType::readFrom(int64_t addr,
                                            InterpreterState &state) const {
  Type elemType = getElementType();
  std::optional<int64_t> stride =
      DataLayoutInterface::getTypeAllocSize(state.getTarget(), elemType);
  if (!stride)
    return Error("failed to get array element stride");
  int64_t offset = *stride;
  SmallVector<TypedAttr> values;
  for (int64_t i = 0, e = *getResolvedSize(); i != e; ++i, addr += offset) {
    ErrorOr<TypedAttr> result = state.readAttributeFromMemory(addr, elemType);
    if (result.isError())
      return result.takeError();
    values.push_back(result.takeValue());
  }
  return POP::ArrayAttr::get(values, *this);
}

//===----------------------------------------------------------------------===//
// UnionType
//===----------------------------------------------------------------------===//

Type UnionType::parse(AsmParser &p) {
  if (p.parseLess())
    return {};

  auto metatype = TypeType::get(p.getContext());
  auto variadicType = ParamListType::get(metatype);
  TypedAttr variadic;

  // Special case `[<variadic>]` to parse the variadic parameter directly.
  if (succeeded(p.parseOptionalLSquare())) {
    if (parseParamValue(p, variadic, variadicType) || p.parseRSquare() ||
        p.parseGreater())
      return {};
    return get(p.getContext(), variadic);
  }

  // Empty union: !pop.union<> - check for '>' before trying to parse types.
  if (succeeded(p.parseOptionalGreater()))
    return get(p.getContext(), ArrayRef<Type>{});

  // Parse a non-empty list of concrete types.
  SmallVector<Type> values;
  if (parseParamTypes(p, values) || p.parseGreater())
    return {};

  return get(p.getContext(), values);
}

void UnionType::print(AsmPrinter &p) const {
  p << '<';
  auto attr = dyn_cast<ParamListAttr>(getVariadic());

  // Unresolved (parametric) variadic - print with brackets.
  if (!attr) {
    p << '[';
    printParamValue(p, getVariadic());
    p << ']';
  } else if (!attr.getValues().empty()) {
    // Resolved non-empty union - print concrete types.
    SmallVector<Type> values;
    for (TypedAttr value : attr.getValues())
      values.push_back(ParamType::get(value));
    printParamTypes(p, values);
  }
  // Empty union - prints nothing between < and >.
  p << '>';
}

UnionType UnionType::get(MLIRContext *ctx, ArrayRef<Type> types) {
  auto metatype = TypeType::get(ctx);
  auto variadicType = ParamListType::get(metatype);
  SmallVector<TypedAttr> elements;
  for (Type type : types)
    elements.push_back(TypeParamAttr::get(type, metatype));
  return get(ctx, ParamListAttr::get(elements, variadicType));
}

UnionType UnionType::get(ArrayRef<Type> types) {
  assert(!types.empty() && "use get(MLIRContext*, ArrayRef<Type>) for "
                           "potentially empty unions");
  return get(types.front().getContext(), types);
}

bool UnionType::isResolved() const { return isa<ParamListAttr>(getVariadic()); }

Type UnionType::getType(unsigned index) const {
  auto attr = dyn_cast<ParamListAttr>(getVariadic());
  if (!attr)
    return {};
  return ParamType::get(attr.getValues()[index]);
}

size_t UnionType::getNumTypes() const {
  auto attr = dyn_cast<ParamListAttr>(getVariadic());
  if (!attr)
    return 0;
  return attr.getValues().size();
}

static Type unwrapTypeAttr(TypedAttr attr) { return ParamType::get(attr); }

llvm::iterator_range<
    llvm::mapped_iterator<ArrayRef<TypedAttr>::iterator, Type (*)(TypedAttr)>>
UnionType::getTypes() const {
  auto attr = dyn_cast<ParamListAttr>(getVariadic());
  if (!attr) {
    // Return empty range for non-resolved variadics.
    return llvm::map_range(ArrayRef<TypedAttr>{}, unwrapTypeAttr);
  }
  return llvm::map_range(attr.getValues(), unwrapTypeAttr);
}

OptionalParseResult UnionType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  if (failed(p.parseOptionalLBrace()))
    return {};
  TypedAttr element;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (parseColonTypeParamValue(p, element) || p.parseRBrace())
    return failure();
  value =
      UnionAttr::getChecked([&] { return p.emitError(loc); }, element, *this);
  return mlir::success((bool)value);
}

LogicalResult UnionType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto attr = ::dyn_cast<UnionAttr>(value);
  if (!attr)
    return failure();
  p << '{';
  printColonTypeParamValue(p, attr.getValue());
  p << '}';
  return success();
}

std::optional<int64_t> UnionType::getTypeSize(TargetInfoAttr target) const {
  assert(isResolved() && "cannot compute size of unresolved union type");
  int64_t maxSize = 0;
  for (Type type : getTypes()) {
    std::optional<int64_t> size =
        DataLayoutInterface::getTypeAllocSize(target, type);
    if (!size)
      return std::nullopt;
    maxSize = std::max(maxSize, *size);
  }
  std::optional<int64_t> align = getTypeAlign(target);
  if (!align)
    return std::nullopt;
  return llvm::alignTo(maxSize, *align);
}

std::optional<int64_t> UnionType::getTypeAlign(TargetInfoAttr target) const {
  assert(isResolved() && "cannot compute alignment of unresolved union type");
  // The alignment of the union type is the max alignment of all variant types.
  // This respects @align decorators on variant types.
  int64_t maxAlign = 1;
  for (Type type : getTypes()) {
    std::optional<int64_t> align =
        DataLayoutInterface::getTypeABIAlign(target, type);
    if (!align)
      return std::nullopt;
    maxAlign = std::max(maxAlign, *align);
  }
  return maxAlign;
}

ErrorOrSuccess UnionType::writeTo(TypedAttr value, int64_t addr,
                                  InterpreterState &state) const {
  return state.writeAttributeToMemory(addr,
                                      ::cast<UnionAttr>(value).getValue());
}

ErrorOr<TypedAttr> UnionType::readFrom(int64_t addr,
                                       InterpreterState &state) const {
  return Error("cannot read a union-typed value");
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Mojo/POPDialect/POPTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

void POPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Mojo/POPDialect/POPTypes.cpp.inc"
      >();

  auto *dialect = getContext()->getOrLoadDialect<KGENDialect>();
  dialect->registerMnemonicType<ArrayType>();
  dialect->registerMnemonicType<UnionType>();
}

//===----------------------------------------------------------------------===//
// IntLiteralType
//===----------------------------------------------------------------------===//

OptionalParseResult IntLiteralType::parseValue(AsmParser &p,
                                               TypedAttr &value) const {
  APInt resultAP;
  OptionalParseResult parseResult = p.parseOptionalInteger(resultAP);
  if (!parseResult.has_value())
    return {};
  if (failed(*parseResult))
    return failure();
  value = IntLiteralAttr::get(p.getContext(), IPInt(resultAP));
  return mlir::success();
}

LogicalResult IntLiteralType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto v = ::dyn_cast<IntLiteralAttr>(value);
  if (!v)
    return failure();
  p.getStream() << v.getValue();
  return success();
}

std::optional<int64_t>
IntLiteralType::getTypeSize(TargetInfoAttr target) const {
  // We use this size to allocate in the interpreter memory space
  // to store an opaque pointer which we always assume is of 64bit,
  // so we should allocate space for 64bit integers instead of
  // target specific pointer size which can be 32bit.
  return target.getDefaultPointerSize();
}

std::optional<int64_t>
IntLiteralType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

/// Write an opaque symbolic attribute to memory.
static ErrorOrSuccess
writeSymbolicAttribute(DataLayoutInterface type, TypedAttr value, int64_t addr,
                       InterpreterState &state,
                       RegionMark regionMark = RegionMark::None) {
  unsigned size = *type.getTypeSize(state.getTarget());

  // Get the default pointer size which is always 8 bytes since we are going to
  // write a uint64_t to the interpreter memory.
  unsigned ptrSize = state.getTarget().getDefaultPointerSize();
  // The ptr to the symbol is written.
  if (size != ptrSize && regionMark == RegionMark::Symbol)
    size = ptrSize;
  ErrorOr<void *> mem = state.getWritableMemory(addr, size, regionMark);
  if (mem)
    return mem.takeError();

  // Without a concrete runtime representation, just make sure the value can be
  // roundtripped.
  llvm::StoreIntToMemory(
      APInt(ptrSize * 8, (uint64_t)value.getAsOpaquePointer()), (uint8_t *)*mem,
      ptrSize);
  return success();
}

/// Read an opaque symbolic attribute from memory.
static ErrorOr<TypedAttr> readSymbolicAttribute(DataLayoutInterface type,
                                                int64_t addr,
                                                InterpreterState &state) {
  ErrorOr<const void *> mem =
      state.getReadableMemory(addr, *type.getTypeSize(state.getTarget()));
  if (mem)
    return mem.takeError();

  // Without a concrete runtime representation, just make sure the value can be
  // roundtripped.
  // Get the default pointer size which is always 8 bytes since we are going to
  // get a uint64_t from the interpreter memory which was stored by
  // writeSymbolicAttribute.
  unsigned ptrSize = state.getTarget().getDefaultPointerSize();
  APInt opaque(ptrSize * 8, 0);
  llvm::LoadIntFromMemory(opaque, (const uint8_t *)*mem, ptrSize);
  return ::cast<TypedAttr>(
      Attribute::getFromOpaquePointer((const void *)opaque.getLimitedValue()));
}

ErrorOrSuccess IntLiteralType::writeTo(TypedAttr value, int64_t addr,
                                       InterpreterState &state) const {
  return writeSymbolicAttribute(*this, value, addr, state);
}

ErrorOr<TypedAttr> IntLiteralType::readFrom(int64_t addr,
                                            InterpreterState &state) const {
  return readSymbolicAttribute(*this, addr, state);
}

//===----------------------------------------------------------------------===//
// FloatLiteralType
//===----------------------------------------------------------------------===//

std::optional<int64_t>
FloatLiteralType::getTypeSize(TargetInfoAttr target) const {
  // We use this size to allocate in the interpreter memory space
  // to store an opaque pointer which we always assume is of 64bit,
  // so we should allocate space for 64bit integers instead of
  // target specific pointer size which can be 32bit.
  return target.getDefaultPointerSize();
}

std::optional<int64_t>
FloatLiteralType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

ErrorOrSuccess FloatLiteralType::writeTo(TypedAttr value, int64_t addr,
                                         InterpreterState &state) const {
  return writeSymbolicAttribute(*this, value, addr, state);
}

ErrorOr<TypedAttr> FloatLiteralType::readFrom(int64_t addr,
                                              InterpreterState &state) const {
  return readSymbolicAttribute(*this, addr, state);
}
