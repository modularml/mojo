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

#include "LLVMLoweringUtils.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/xxhash.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

/// Since !kgen.string type is not parameterized on the size of the string,
/// we lower it as struct with a pointer field holding the data and a index
/// field holding the string size.
static Type getLLVMTypeForKGENStringType(MLIRContext *ctx, Type strSizeType) {
  SmallVector<Type> elementTypes{LLVM::LLVMPointerType::get(ctx), strSizeType};
  return LLVM::LLVMStructType::getLiteral(ctx, elementTypes);
}

//===----------------------------------------------------------------------===//
// LLVMDataLayout
//===----------------------------------------------------------------------===//

int64_t LLVMDataLayout::getTypeSizeInBits(Type type) const {
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();

  // Assert here to let fp8 types passes above.
  assert(LLVM::isCompatibleType(type) && "expected an LLVM type");

  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type)) {
    return target.getDataLayout().getPointerBitWidth(ptrType.getAddressSpace());
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return target.getDataLayout().getVectorBitWidth(
        vecType.getNumElements(), getTypeSizeInBits(vecType.getElementType()));
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type)) {
    return arrayType.getNumElements() *
           getTypeStoreSize(arrayType.getElementType()) * CHAR_BIT;
  }
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    int64_t size = 0;
    int64_t strictest = 1;
    for (Type type : structType.getBody()) {
      int64_t eltABIAlign = getTypeABIAlign(type);
      size = llvm::alignTo(size, eltABIAlign) + getTypeAllocSize(type);
      strictest = std::max(strictest, eltABIAlign);
    }
    return llvm::alignTo(size, strictest) * CHAR_BIT;
  }
  llvm::report_fatal_error("unsupported LLVM dialect type");
}

std::pair<int64_t, Type>
LLVMDataLayout::getTypeABIAlignAndType(Type type) const {
  if (auto intType = dyn_cast<IntegerType>(type))
    return {target.getDataLayout().getIntegerABIAlign(intType.getWidth()),
            intType};
  if (auto fpType = dyn_cast<FloatType>(type))
    return {target.getDataLayout().getFloatABIAlign(fpType.getWidth()), fpType};

  // Assert here to let fp8 types passes above.
  assert(LLVM::isCompatibleType(type) && "expected an LLVM type");

  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type))
    return {target.getDataLayout().getPointerABIAlign(), ptrType};
  // Use the natural alignment for vector to be conservative
  if (auto vecType = dyn_cast<VectorType>(type))
    return getTypeABIAlignAndType(vecType.getElementType());
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type))
    return getTypeABIAlignAndType(arrayType.getElementType());
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    std::pair<int64_t, Type> strictestPair(1, nullptr);
    for (Type type : structType.getBody()) {
      auto cur = getTypeABIAlignAndType(type);
      if (cur.first >= strictestPair.first)
        strictestPair = cur;
    }

    // Aggregates/structs can be ABI-aligned as a whole, as per the DL.
    strictestPair.first = std::max(
        strictestPair.first,
        static_cast<int64_t>(target.getDataLayout().getStructABIAlign()));
    return strictestPair;
  }
  llvm::report_fatal_error("unsupported LLVM dialect type");
}

/// Whether `type`'s value fills its own storage, e.g. false for `i1` (1 bit
/// in a whole byte). Reusing an unsafe type as a union's representative
/// field truncates a sibling's data.
static bool isSafeUnionReprType(const LLVMDataLayout &dl, Type type) {
  return dl.getTypeSizeInBits(type) == 8 * dl.getTypeStoreSize(type);
}

namespace {
/// A union member paired with the LLVM type it converts to. The payload size
/// and the representative type are separate readings of the same set, so they
/// are derived from one conversion rather than accumulated side by side.
struct UnionMember {
  Type kgenType;
  Type llvmType;
};
} // namespace

/// Bytes the union's payload must span: the widest member, measured in KGEN's
/// data layout where it knows the type and LLVM's otherwise (MOCO-4689). A
/// simd<N, bool> is N bytes to KGEN but converts to vector<N x i1>, which LLVM
/// allocates in ceil(N/8). Types outside KGEN's layout interface, such as f80,
/// are sized by LLVM alone.
static int64_t getUnionPayloadSize(const LLVMDataLayout &dl,
                                   TargetInfoAttr target,
                                   ArrayRef<UnionMember> members) {
  int64_t payloadSize = 0;
  for (const UnionMember &member : members) {
    int64_t memberSize = dl.getTypeAllocSize(member.llvmType);
    if (std::optional<int64_t> kgenSize =
            DataLayoutInterface::getTypeAllocSize(target, member.kgenType))
      memberSize = std::max(memberSize, *kgenSize);
    payloadSize = std::max(payloadSize, memberSize);
  }
  return payloadSize;
}

/// The type heading the lowered struct, chosen to carry the union's alignment.
/// It is sized independently of the payload: whatever it does not cover is
/// made up by the trailing byte array.
static Type getUnionReprType(const LLVMDataLayout &dl, TargetInfoAttr target,
                             POP::UnionType unionType,
                             ArrayRef<UnionMember> members) {
  // The most-aligned member, skipping null candidates (MOCO-3275) and types
  // that do not fill their own storage (MOCO-3900).
  std::pair<int64_t, Type> maxAlignAndType(1, nullptr);
  for (const UnionMember &member : members) {
    auto curAlignAndMember = dl.getTypeABIAlignAndType(member.llvmType);
    if (curAlignAndMember.second &&
        isSafeUnionReprType(dl, curAlignAndMember.second) &&
        curAlignAndMember.first >= maxAlignAndType.first)
      maxAlignAndType = curAlignAndMember;
  }

  // Fall back to i8 if no member offered a safe candidate.
  MLIRContext *ctx = unionType.getContext();
  Type reprType = maxAlignAndType.second ? maxAlignAndType.second
                                         : IntegerType::get(ctx, 8);

  // That candidate can still under-report the union's true alignment, e.g.
  // when a wider member was excluded as unsafe. Try to synthesize a plain
  // integer matching the true alignment in that case.
  if (auto trueAlign = unionType.getTypeAlign(target);
      trueAlign && *trueAlign > dl.getTypeABIAlign(reprType)) {
    Type widened = IntegerType::get(ctx, 8 * *trueAlign);
    // FIXME(MOCO-4441): Depending on the exact target, the integer may or may
    // not have the desired alignment. Use widened only when it succeeds.
    if (dl.getTypeABIAlign(widened) >= *trueAlign)
      reprType = widened;
  }
  return reprType;
}

//===----------------------------------------------------------------------===//
// TargetInfoAttr
//===----------------------------------------------------------------------===//

ArrayAttr KGEN::attachTargetPassthroughAttrs(OpBuilder &b,
                                             TargetInfoAttr target,
                                             ArrayAttr passthrough) {
  SmallVector<Attribute> attrs;
  if (passthrough)
    llvm::append_range(attrs, passthrough);
  // Attach the target info attributes.
  attrs.push_back(b.getArrayAttr(
      {b.getStringAttr("target-cpu"), b.getStringAttr(target.getArch())}));
  attrs.push_back(b.getArrayAttr({b.getStringAttr("target-features"),
                                  b.getStringAttr(target.getFeatures())}));
  if (!target.getTuneCpu().empty())
    attrs.push_back(b.getArrayAttr(
        {b.getStringAttr("tune-cpu"), b.getStringAttr(target.getTuneCpu())}));
  // `abi` is recorded as a module flag in LowerKGENToLLVMPass instead.
  return b.getArrayAttr(attrs);
}

//===----------------------------------------------------------------------===//
// POPToLLVMTypeConverter
//===----------------------------------------------------------------------===//

bool KGEN::isFP8(Type fpType) {
  return isa<Float8E5M2Type, Float8E5M2FNUZType, Float8E4M3FNType,
             mlir::Float8E3M4Type, Float8E4M3FNUZType, Float8E8M0FNUType>(
      fpType);
}

bool KGEN::isFP6(Type fpType) {
  return isa<mlir::Float6E2M3FNType, mlir::Float6E3M2FNType>(fpType);
}

bool KGEN::isFP4(Type fpType) { return isa<mlir::Float4E2M1FNType>(fpType); }

// If the type is a floating-point type lowered to an integer, return the
// bitwidth of the integer width. Else return std::nullopt.
std::optional<int> KGEN::isFPTyLoweredAsInt(Type fpType) {
  if (isFP4(fpType))
    return 4;
  if (isFP6(fpType))
    return 6;
  if (isFP8(fpType))
    return 8;
  return std::nullopt;
}

std::optional<Type> M::KGEN::getMLIRTypeForDType(MLIRContext *ctx,
                                                 KGENDType dtype,
                                                 size_t indexBitwidth) {
  if (dtype.isBool())
    return IntegerType::get(ctx, 1);

  if (dtype.isAddress())
    return LLVM::LLVMPointerType::get(ctx);

  if (dtype.isIndex() || dtype.isUIndex())
    return IntegerType::get(ctx, indexBitwidth);

  // This intentionally discards signed-ness because LLVM is signless.
  if (dtype.isInt())
    return IntegerType::get(ctx, dtype.getIntegerWidthInBits());

  if (dtype.isFloat()) {
    if (FloatType fpType = getEquivalentFloatType(ctx, dtype)) {
      if (auto intWidth = isFPTyLoweredAsInt(fpType))
        return IntegerType::get(ctx, *intWidth);
      return fpType;
    }
  }

  return {};
}

/// Build LLVM lowering options for a target.
static mlir::LowerToLLVMOptions buildLLVMLoweringOpts(TargetInfoAttr target) {
  mlir::LowerToLLVMOptions opts(target.getContext());
  opts.overrideIndexBitwidth(target.resolveIndexBitWidth());
  opts.dataLayout = llvm::DataLayout(target.getDataLayout().toString());
  return opts;
}

POPToLLVMTypeConverter::POPToLLVMTypeConverter(TargetInfoAttr target)
    : LLVMTypeConverter(target.getContext(), buildLLVMLoweringOpts(target)),
      LLVMDataLayout(target) {

  //===--------------------------------------------------------------------===//
  // KGEN

  // Convert `!kgen.none` to an empty struct.
  addConversion([this](KGEN::NoneType) {
    return LLVM::LLVMStructType::getLiteral(&getContext(), {});
  });

  addConversion([this](KGEN::NeverType) {
    return LLVM::LLVMStructType::getLiteral(&getContext(), {});
  });

  // Convert string types to LLVM literal structs: struct{ptr, size} of type
  // !llvm.struct<(ptr<i8>, index).
  addConversion([this](KGEN::StringType stringType) -> std::optional<Type> {
    return getLLVMTypeForKGENStringType(stringType.getContext(),
                                        getIndexType());
  });

  addConversion([this](FuncType signatureType) -> std::optional<Type> {
    MLIRContext *ctx = signatureType.getContext();
    if (signatureType.isCapturing()) {
      auto pointerTy = LLVM::LLVMPointerType::get(ctx);
      return LLVM::LLVMStructType::getLiteral(ctx, {pointerTy, pointerTy});
    } else {
      return convertType(signatureType.getValues());
    }
  });

  // TODO(MOCO-1253): GeneratorType should not be allowed during LLVM lowering.
  addConversion(
      [this](FuncTypeGeneratorType sigGenType) -> std::optional<Type> {
        return convertType(sigGenType.getBody());
      });

  // Convert pointer types to LLVM pointer types.
  addConversion([](PointerType pointer) -> std::optional<Type> {
    unsigned addressSpace =
        cast<IntegerAttr>(pointer.getAddressSpace()).getInt();
    return LLVM::LLVMPointerType::get(pointer.getContext(), addressSpace);
  });

  //===--------------------------------------------------------------------===//
  // POP

  // Convert array types to LLVM array types.
  addConversion([this](POP::ArrayType array) -> std::optional<Type> {
    std::optional<int64_t> size = array.getResolvedSize();
    if (!size)
      return {};
    Type elementType = convertType(array.getElementType());
    if (!elementType)
      return {};
    return LLVM::LLVMArrayType::get(elementType, *size);
  });

  // Convert struct types to LLVM literal structs, adding padding fields as
  // needed to satisfy field alignment requirements from @align decorators.
  //
  // We add explicit padding when the KGEN type's alignment exceeds the LLVM
  // type's alignment. This happens when a field has @align(N) where N is
  // greater than the type's natural alignment.
  //
  // IMPORTANT: We must use KGEN's type size information consistently, not
  // LLVM's, because LLVM struct layout may differ from KGEN's expectations
  // (e.g., for unions that become LLVM structs with different member order).
  addConversion([this](StructType structType) -> std::optional<Type> {
    std::optional<SmallVector<Type>> elementTypes =
        structType.getElementTypes();
    if (!elementTypes)
      return {};

    SmallVector<Type> llvmTypes;
    int64_t offset = 0;
    Type i8Type = IntegerType::get(&getContext(), 8);

    for (auto [logicalIdx, kgenType] : llvm::enumerate(*elementTypes)) {
      Type llvmType = convertType(kgenType);
      if (!llvmType)
        return {};

      // Get alignment and size from KGEN type (respects @align decorator).
      std::optional<int64_t> kgenAlign =
          DataLayoutInterface::getTypeABIAlign(getTarget(), kgenType);
      std::optional<int64_t> kgenAllocSize =
          DataLayoutInterface::getTypeAllocSize(getTarget(), kgenType);
      if (!kgenAlign || !kgenAllocSize)
        return {};

      // Get LLVM's natural alignment for this type.
      // We always use LLVM's actual alignment to compare against KGEN's
      // alignment, since LLVM doesn't know about KGEN's @align decorators
      // or alignment inheritance from fields.
      int64_t llvmNaturalAlign = getTypeABIAlign(llvmType);

      // Compute where LLVM would naturally place this field and where KGEN
      // requires it.
      int64_t llvmOffset = llvm::alignTo(offset, llvmNaturalAlign);
      int64_t kgenOffset = llvm::alignTo(offset, *kgenAlign);

      // Only add explicit padding if KGEN requires stricter alignment than
      // LLVM's natural alignment.
      if (kgenOffset > llvmOffset) {
        int64_t paddingNeeded = kgenOffset - offset;
        llvmTypes.push_back(LLVM::LLVMArrayType::get(i8Type, paddingNeeded));
      }

      // Cache the mapping from logical index to LLVM index.
      auto cacheKey = std::make_pair(Type(structType), (int64_t)logicalIdx);
      auto [it, inserted] =
          structFieldIndexCache.try_emplace(cacheKey, llvmTypes.size());
      // If already cached, verify consistency (type conversion is
      // deterministic).
      assert((inserted || it->second == (int64_t)llvmTypes.size()) &&
             "inconsistent field index mapping for struct type");
      llvmTypes.push_back(llvmType);

      // Advance offset using KGEN alloc size.
      offset = kgenOffset + *kgenAllocSize;
    }

    auto llvmStructType =
        LLVM::LLVMStructType::getLiteral(&getContext(), llvmTypes);

    // Pad the struct up to its required alignment, if LLVM's data layout
    // wouldn't implicitly do it for us.
    std::optional<int64_t> kgenTotalAlign =
        DataLayoutInterface::getTypeABIAlign(getTarget(), structType);
    if (!kgenTotalAlign)
      return {};

    int64_t llvmOffset = llvm::alignTo(offset, getTypeABIAlign(llvmStructType));
    int64_t kgenOffset = llvm::alignTo(offset, *kgenTotalAlign);
    if (kgenOffset > llvmOffset) {
      llvmTypes.push_back(
          LLVM::LLVMArrayType::get(i8Type, kgenOffset - offset));
      llvmStructType =
          LLVM::LLVMStructType::getLiteral(&getContext(), llvmTypes);
    }

    return llvmStructType;
  });

  // Convert SIMD types to vector types.
  addConversion([this](SIMDType simd) -> std::optional<Type> {
    std::optional<KGENDType> dtype = simd.getResolvedDType();
    std::optional<uint64_t> size = simd.getResolvedSize();
    if (!dtype || !size)
      return {};
    std::optional<Type> type = getMLIRTypeForDType(
        simd.getContext(), *dtype, getOptions().getIndexBitwidth());
    if (!type)
      return {};

    // Scalar case, size = 1
    if (*size == 1)
      return *type;

    // Vector case, size != 1
    return VectorType::get(*size, *type);
  });

  // Convert data type types to `i8`.
  addConversion([this](DTypeType dtype) -> std::optional<Type> {
    return Builder(&getContext()).getI8Type();
  });

  // Convert union types to an array with enough space to contain the largest
  // union element type.
  addConversion([this](POP::UnionType unionType) -> std::optional<Type> {
    // Unresolved (parameterized) unions must be specialized before lowering.
    // They are resolved during monomorphization in the KGEN elaboration pass.
    assert(unionType.isResolved() &&
           "cannot lower unresolved union type to LLVM");
    // TODO: The generated assembly is sensitive to the content type of the
    // union type. This needs to be optimized. For now, use an array of
    // word-size integers.
    SmallVector<UnionMember> members;
    members.reserve(unionType.getNumTypes());
    for (Type memberType : unionType.getTypes()) {
      Type llvmType = convertType(memberType);
      if (!llvmType)
        return {};
      members.push_back({memberType, llvmType});
    }

    int64_t payloadSize = getUnionPayloadSize(*this, getTarget(), members);
    if (payloadSize == 0)
      return LLVM::LLVMStructType::getLiteral(&getContext(), {});

    // Lower union to {repr, [(payloadSize - sizeof(repr)) x i8]}. The
    // representative gives the whole structure its alignment, the trailing
    // array covers the payload it does not span.
    Type reprType = getUnionReprType(*this, getTarget(), unionType, members);

    SmallVector<Type, 2> structElemTp;
    structElemTp.push_back(reprType);

    if (int64_t tailLen = payloadSize - getTypeAllocSize(reprType);
        tailLen != 0) {
      structElemTp.push_back(LLVM::LLVMArrayType::get(
          IntegerType::get(&getContext(), 8), tailLen));
    }

    auto ret = LLVM::LLVMStructType::getLiteral(&getContext(), structElemTp);
    assert(payloadSize == getTypeAllocSize(ret) &&
           "expect lowered UnionType to have the same size as the biggest "
           "union variant.");
    return ret;
  });

  // Coroutine handles are always lowered to opaque pointers.
  addConversion([](CO::CoroutineType coro) {
    return LLVM::LLVMPointerType::get(coro.getContext());
  });
}

int64_t
POPToLLVMTypeConverter::getRemappedFieldIndex(StructType structType,
                                              int64_t logicalIndex) const {
  auto it = structFieldIndexCache.find({structType, logicalIndex});
  if (it != structFieldIndexCache.end())
    return it->second;
  // The struct type must be converted by this type converter instance before
  // calling getRemappedFieldIndex. The cache is populated during type
  // conversion, so a cache miss indicates a bug in the calling code.
  llvm_unreachable("struct type was not converted by this type converter");
}

//===----------------------------------------------------------------------===//
// VariantHelper
//===----------------------------------------------------------------------===//

/// Advance the variant storage pointer by a set amount no more than the current
/// amount of remaining space in the current storage element.
template <typename ItType>
static unsigned advanceStoragePtr(ItType &valueIt, unsigned &storageOffset,
                                  unsigned amt) {
  auto curStorageSize = cast<IntegerType>(valueIt->getType()).getWidth();
  unsigned advanceBy = std::min(amt, curStorageSize - storageOffset);
  storageOffset += advanceBy;
  if (storageOffset == curStorageSize) {
    ++valueIt;
    storageOffset = 0;
  }
  return advanceBy;
}

/// Pad the variant storage by a bit amount. This is used to add padding to the
/// variant layout.
template <typename ItType>
static void addStoragePadding(ItType &valueIt, unsigned &storageOffset,
                              unsigned &offset, unsigned alignment) {
  unsigned padding = llvm::alignTo(offset, alignment * CHAR_BIT) - offset;
  for (unsigned added = 0; added != padding;)
    added += advanceStoragePtr(valueIt, storageOffset, padding - added);
  offset += padding;
}

void VariantHelper::walkAndCreateVariant(
    MutableArrayRef<Value>::iterator &valueIt, unsigned &storageOffset,
    unsigned &offset, Value value) {
  // Align the storage pointer to the current value being stored.
  addStoragePadding(valueIt, storageOffset, offset,
                    dl.getTypeABIAlign(value.getType()));

  // Aggregate types like structs and arrays are flattened to their leaf types.
  // Leaf types are integers, floats, and pointers.
  if (isa<IntegerType, FloatType, LLVM::LLVMPointerType>(value.getType())) {
    Value normalizedValue;
    // Normalize the value to store to an integer.
    if (isa<IntegerType>(value.getType()))
      normalizedValue = value;
    else if (auto fpType = dyn_cast<FloatType>(value.getType()))
      normalizedValue = LLVM::BitcastOp::create(
          b, b.getIntegerType(fpType.getWidth()), value);
    else
      normalizedValue = LLVM::PtrToIntOp::create(
          b, b.getIntegerType(dl.getTypeSizeInBits(value.getType())), value);

    unsigned curValueSize =
        cast<IntegerType>(normalizedValue.getType()).getWidth();
    offset += curValueSize;
    unsigned curValueOffset = 0;
    while (curValueOffset != curValueSize) {
      // Compute the remaining space.
      auto curStorageType = cast<IntegerType>(valueIt->getType());

      // Ignore the bits of the value that has already been stored.
      Value valueToStore = LLVM::LShrOp::create(
          b, normalizedValue,
          LLVM::ConstantOp::create(b, normalizedValue.getType(),
                                   curValueOffset));
      // Match the type with the storage type.
      if (curValueSize < curStorageType.getWidth())
        valueToStore = LLVM::ZExtOp::create(b, curStorageType, valueToStore);
      else
        valueToStore = LLVM::TruncOp::create(b, curStorageType, valueToStore);
      // Shift the current value to store to the current storage offset.
      valueToStore = LLVM::ShlOp::create(
          b, valueToStore,
          LLVM::ConstantOp::create(b, curStorageType, storageOffset));
      // Set the bits of the current value to store.
      *valueIt = LLVM::OrOp::create(b, *valueIt, valueToStore);

      curValueOffset += advanceStoragePtr(valueIt, storageOffset,
                                          curValueSize - curValueOffset);
    }

    // The value has been stored.
    return;
  }

  // This is an aggregate type. Extract the next elements and recurse.
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(value.getType())) {
    for (unsigned i = 0, e = arrayType.getNumElements(); i < e; ++i) {
      Value nestedValue = LLVM::ExtractValueOp::create(b, value, i);
      walkAndCreateVariant(valueIt, storageOffset, offset, nestedValue);
    }
    return;
  }
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(value.getType())) {
    for (unsigned i = 0, e = structType.getBody().size(); i < e; ++i) {
      Value nestedValue = LLVM::ExtractValueOp::create(b, value, i);
      walkAndCreateVariant(valueIt, storageOffset, offset, nestedValue);
    }
    return;
  }
  auto vectorType = dyn_cast<VectorType>(value.getType());
  assert(vectorType && !vectorType.isScalable() &&
         "Expecting a fixed vector type");
  for (unsigned i = 0, e = vectorType.getNumElements(); i < e; ++i) {
    Value nestedValue = LLVM::ExtractElementOp::create(
        b, value, LLVM::ConstantOp::create(b, b.getI32Type(), i));
    walkAndCreateVariant(valueIt, storageOffset, offset, nestedValue);
  }
}

Value VariantHelper::materializeLLVMUnion(
    mlir::LLVM::LLVMStructType unionStructTp, Value value) {
  if (unionStructTp.getBody().empty())
    return LLVM::UndefOp::create(b, unionStructTp);

  SmallVector<Value> storageValues;
  auto maxAlignTp = unionStructTp.getBody().front();

  // Normalize the max_align_t to an (potentially array of) integers.
  if (auto t = dyn_cast<VectorType>(maxAlignTp)) {
    // Flatten the vector.
    for (int i = 0, e = t.getNumElements(); i < e; ++i) {
      storageValues.push_back(LLVM::ConstantOp::create(
          b, b.getIntegerType(dl.getTypeSizeInBits(t.getElementType())), 0));
    }
  } else if (maxAlignTp.isIntOrFloat() ||
             isa<LLVM::LLVMPointerType>(maxAlignTp)) {
    storageValues.push_back(LLVM::ConstantOp::create(
        b, b.getIntegerType(dl.getTypeSizeInBits(maxAlignTp)), 0));
  } else {
    llvm_unreachable(
        "The first type in lowered union type must be non-aggregated.");
  }

  LLVM::LLVMArrayType tailingMem = nullptr;
  if (unionStructTp.getBody().size() == 2) {
    tailingMem = cast<LLVM::LLVMArrayType>(unionStructTp.getBody().back());
    for (unsigned i = 0, e = tailingMem.getNumElements(); i < e; ++i)
      storageValues.push_back(
          LLVM::ConstantOp::create(b, tailingMem.getElementType(), 0));
  }

  MutableArrayRef<Value>::iterator valueIt = storageValues.begin();
  unsigned storageOffset = 0;
  unsigned offset = 0;
  walkAndCreateVariant(valueIt, storageOffset, offset, value);

  ArrayRef<Value> toPack = storageValues;
  Value content = LLVM::UndefOp::create(b, unionStructTp);

  Value maxAlignV;
  if (auto vecTp = dyn_cast<VectorType>(maxAlignTp)) {
    // Aggregate the vector.
    maxAlignV = LLVM::UndefOp::create(b, vecTp);
    for (int i = 0, e = vecTp.getNumElements(); i < e; ++i) {
      auto element = toPack.front();
      maxAlignV = LLVM::InsertElementOp::create(
          b, maxAlignV,
          LLVM::BitcastOp::create(b, vecTp.getElementType(), element),
          LLVM::ConstantOp::create(b, b.getI32Type(), i));

      toPack = toPack.drop_front();
    }
  } else if (isa<LLVM::LLVMPointerType>(maxAlignTp)) {
    maxAlignV = LLVM::IntToPtrOp::create(b, maxAlignTp, toPack.front());
    toPack = toPack.drop_front();
  } else if (maxAlignTp.isIntOrFloat()) {
    maxAlignV = LLVM::BitcastOp::create(b, maxAlignTp, toPack.front());
    toPack = toPack.drop_front();
  } else {
    llvm_unreachable(
        "The first type in lowered union type must be non-aggregated.");
  }
  content = LLVM::InsertValueOp::create(b, content, maxAlignV,
                                        static_cast<int64_t>(0));

  if (tailingMem) {
    Value arrayV = LLVM::UndefOp::create(b, tailingMem);
    for (auto [idx, value] : llvm::enumerate(toPack))
      arrayV = LLVM::InsertValueOp::create(b, arrayV, value, idx);
    content = LLVM::InsertValueOp::create(b, content, arrayV, 1);
  }

  return content;
}

//===----------------------------------------------------------------------===//
// Interpreter Memory Conversion
//===----------------------------------------------------------------------===//

ErrorOr<Value> InterpreterMemoryConverter::MaterializationScope::convertMemRef(
    ImplicitLocOpBuilder &b, MemRefAttr ref) {
  ErrorOr<MaterializedBlob &> materialized =
      getOrMaterialize(b, ref.getModel().getMemory(), ref.getIndex());

  if (materialized.isError())
    return materialized.takeError();

  Value ptr = getBlobPointer(b, imc.tc.convertType(ref.getType()),
                             *materialized, ref.getIndex(), ref.getOffset());
  return LLVM::BitcastOp::create(b, imc.tc.convertType(ref.getType()), ptr);
}

Value InterpreterMemoryConverter::MaterializationScope::getBlobPointer(
    ImplicitLocOpBuilder &b, Type ptrType, MaterializedBlob &value,
    int64_t index, int64_t offset) {
  Value ptr = dyn_cast<Value>(value);
  if (!ptr) {
    ptr = LLVM::BitcastOp::create(
        b, ptrType,
        LLVM::AddressOfOp::create(
            b, cast<LLVM::GlobalOp>(cast<Operation *>(value))));
  }
  return LLVM::GEPOp::create(b, ptrType, b.getI8Type(), ptr,
                             LLVM::GEPArg(offset),
                             LLVM::GEPNoWrapFlags::inbounds);
}

Operation *InterpreterMemoryConverter::getOrCreateGlobal(Location loc,
                                                         MemoryBlobAttr blob) {
  // Lookup an existing global for this handle.
  if (Operation *global = globals.lookup(blob))
    return global;

  // If not, create it.
  OpBuilder b(loc.getContext());
  Attribute value;
  MemoryHandleAttr hdl = blob.getHandle();
  if (hdl.isString()) {
    // Create a string attribute for readability.
    value = b.getStringAttr(StringRef(hdl.getData(), hdl.getSize()));
  } else {
    // Store the raw bytes into an elements attribute.
    value = IntArrayElementsAttr::get(b.getContext(), hdl.getMemory().data,
                                      IntegerType::Signless);
  }

  std::string hashKey = llvm::utohexstr(
      llvm::xxh3_64bits({(const uint8_t *)hdl.getData(), hdl.getSize()}),
      /*LowerCase=*/true,
      /*Width=*/16);
  std::string key =
      (hdl.isString() ? "static_string_" : "memory_blob_") + hashKey;

  auto global = LLVM::GlobalOp::create(
      b, loc, LLVM::LLVMArrayType::get(b.getI8Type(), hdl.getSize()),
      blob.getKind() == MemoryKind::ConstGlobal, LLVM::Linkage::Internal, key,
      value, hdl.getAlign(), blob.getAddressSpace());
  symtab.insert(global);

  globals.try_emplace(blob, global);
  return global;
}

/// Store `size` worth of data to offset `idx` into `ptr`, reading from `data`.
/// Use vector stores, and progressively smaller ones if `size` is not a
/// multiple of 2.
static void materializeVectorStores(int64_t idx, int64_t size, Value ptr,
                                    const char *data, ImplicitLocOpBuilder &b,
                                    Type ptrType, size_t align) {
  // Nothing to do.
  if (size == 0)
    return;

  // GEP to the current offset.
  Value gep =
      LLVM::GEPOp::create(b, ptrType, b.getI8Type(), ptr, LLVM::GEPArg(idx),
                          LLVM::GEPNoWrapFlags::inbounds);
  // Emit a scalar store.
  if (size == 1) {
    LLVM::StoreOp::create(
        b, LLVM::ConstantOp::create(b, b.getI8Type(), data[idx]), gep, align);
    return;
  }

  // Round down to the nearest power of 2, inclusive.
  int64_t curSize = llvm::NextPowerOf2(size) / 2;
  int64_t remaining = size - curSize;

  ArrayRef<uint8_t> slice((const uint8_t *)data + idx, curSize);
  auto value =
      ArrayElementsAttr::get(slice, VectorType::get(curSize, b.getI8Type()));
  LLVM::StoreOp::create(b, LLVM::ConstantOp::create(b, value), gep, align);
  materializeVectorStores(idx + curSize, remaining, ptr, data, b, ptrType,
                          align);
}

ErrorOr<InterpreterMemoryConverter::MaterializedBlob &>
InterpreterMemoryConverter::MaterializationScope::getOrMaterialize(
    ImplicitLocOpBuilder &b, MemorySpaceAttr space, size_t refIndex) {

  auto iter = blobs.find(space);
  if (iter != blobs.end()) {
    // We've already materialized the stack, const_global and persist blobs in
    // this space.
    auto refIter = iter->second.find(refIndex);
    if (refIter != iter->second.end())
      return refIter->second;
  }

  MaterializedBlobs materialized;
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());

  auto materializeBlob = [&](const MemoryBlobAttr &blob, size_t idx) {
    if (materialized.contains(idx))
      return;
    if (isGlobalBlob(blob)) {
      materialized.insert({idx, imc.getOrCreateGlobal(b.getLoc(), blob)});
      return;
    }

    // Create the relevant allocation.
    Value popAlloc;
    MemoryHandleAttr hdl = blob.getHandle();
    if (blob.getKind() == MemoryKind::Stack ||
        // FIXME(#32052): Persistent memory requires planning, but downcast to a
        // stack allocation for now.
        blob.getKind() == MemoryKind::Persistent) {
      popAlloc = POP::StackAllocationOp::create(
          b, PointerType::get(b.getI8Type()), hdl.getSize(),
          b.getIndexAttr(hdl.getAlign()));

      Value ptr = mlir::UnrealizedConversionCastOp::create(b, ptrType, popAlloc)
                      .getResult(0);
      materialized.insert(
          {idx, Value(LLVM::BitcastOp::create(b, ptrType, ptr))});

    } else {
      popAlloc = POP::AlignedAllocOp::create(
          b, PointerType::get(b.getI8Type()),
          mlir::index::ConstantOp::create(b, hdl.getAlign()),
          mlir::index::ConstantOp::create(b, hdl.getSize()));

      Value ptr = mlir::UnrealizedConversionCastOp::create(b, ptrType, popAlloc)
                      .getResult(0);
      materialized.insert(
          {idx, Value(LLVM::BitcastOp::create(b, ptrType, ptr))});
    }
  };

  std::function<ErrorOrSuccess(const MemoryBlobAttr &)>
      materializePtrRegionBlobs =
          [&](const MemoryBlobAttr &blob) -> ErrorOrSuccess {
    auto ptrIt = blob.getPointerRegions().begin();
    auto ptrEnd = blob.getPointerRegions().end();
    for (; ptrIt != ptrEnd; ++ptrIt) {
      if (materialized.contains(ptrIt->blobIndex))
        continue;

      const MemoryBlobAttr &ptrBlob = space[ptrIt->blobIndex];
      if (ptrBlob.getKind() == MemoryKind::Stack) {
        return Error("indirect access to interpreter stack memory");
      }
      materializeBlob(ptrBlob, ptrIt->blobIndex);
      ErrorOrSuccess result = materializePtrRegionBlobs(ptrBlob);
      if (result.isError())
        return result.takeError();
    }
    return success();
  };

  // Get the current blob for the memref.
  const MemoryBlobAttr &blob = space[refIndex];
  // Materialize the blob
  materializeBlob(blob, refIndex);
  // Materialize pointer region blobs and recurse to
  // materialize each blob's pointer region.
  ErrorOrSuccess mprbResult = materializePtrRegionBlobs(blob);
  if (mprbResult.isError())
    return mprbResult.takeError();

  // Perform memcpy of non-global blobs while remapping pointer regions.
  int64_t pointerSize = imc.tc.getTarget().getDataLayout().getPointerSize();
  int64_t simdWidth = imc.tc.getTarget().getSimdBitWidth() / 8;

  for (auto [idx, blob] : llvm::enumerate(space)) {
    // Globals don't have pointer regions.
    if (isGlobalBlob(blob))
      continue;

    auto iter = materialized.find(idx);
    if (iter == materialized.end())
      continue;

    auto ptr = cast<Value>(iter->second);
    MemoryHandleAttr hdl = blob.getHandle();
    ArrayRef<char> data = hdl.getMemory().data;
    auto materializeStoreImpl = [&, align = hdl.getAlign()](int64_t idx,
                                                            int64_t size) {
      materializeVectorStores(idx, size, ptr, data.data(), b, ptrType, align);
    };

    // For large, contiguous chunks of memory with the same byte value,
    // "compress" the generated IR by emitting a memset instead of a huge number
    // of SIMD stores. This struct tracks the current compression state, which
    // has to be "committed". This will prevent large materialized blobs from
    // destroying compile time. However, if the user fills a large blob with
    // "random" data, not much can be done.
    struct CompressionState {
      int64_t startIdx;
      char value;
      int64_t numReps;
    };
    std::optional<CompressionState> compressionState;
    auto commitCompressedStores = [&] {
      if (!compressionState)
        return;
      auto [startIdx, value, numReps] = std::move(*compressionState);
      compressionState.reset();
      // Simple heuristic: if the compressed size is more than 8 times the
      // preferred SIMD width, then use a memset instead.
      if (numReps <= 8 * simdWidth) {
        for (int64_t i = 0; i < numReps; i += simdWidth)
          materializeStoreImpl(startIdx + i, std::min(simdWidth, numReps - i));
        return;
      }
      // Emit a memset.
      Value gep = LLVM::GEPOp::create(b, ptrType, b.getI8Type(), ptr,
                                      LLVM::GEPArg(startIdx),
                                      LLVM::GEPNoWrapFlags::inbounds);
      LLVM::MemsetOp::create(
          b, gep, LLVM::ConstantOp::create(b, b.getI8Type(), value),
          LLVM::ConstantOp::create(b, b.getI64Type(), numReps),
          /*isVolatile=*/false);
    };

    auto materializeStore = [&](int64_t idx, int64_t size) {
      if (size == 0)
        return;
      if (!llvm::all_equal(data.slice(idx, size))) {
        // No compression possible. Commit the current state and materialize the
        // next chunk.
        commitCompressedStores();
        materializeStoreImpl(idx, size);
      } else if (!compressionState) {
        // Start tracking a new compression state.
        compressionState = CompressionState{idx, data[idx], size};
      } else if (compressionState->value == data[idx]) {
        // Increase the size of the compressed chunk.
        compressionState->numReps += size;
      } else {
        // Commit the previous state.
        commitCompressedStores();
        compressionState = CompressionState{idx, data[idx], size};
      }
    };

    // Store the memory blob in chunks of the preferred SIMD width.
    auto ptrIt = blob.getPointerRegions().begin();
    auto ptrEnd = blob.getPointerRegions().end();
    for (int64_t i = 0, e = data.size(); i < e;) {
      // Check if the current chunk contains a pointer.
      if (ptrIt != ptrEnd && i <= ptrIt->offset &&
          ptrIt->offset < (i + simdWidth)) {
        // Store up to the pointer region.
        int64_t partSize = ptrIt->offset - i;
        materializeStore(i, partSize);
        i += partSize;

        // Store the pointer value to the current offset.
        commitCompressedStores();
        Value gep =
            LLVM::GEPOp::create(b, ptrType, b.getI8Type(), ptr, LLVM::GEPArg(i),
                                LLVM::GEPNoWrapFlags::inbounds);
        auto [_, index, offset] = *ptrIt++;
        LLVM::StoreOp::create(
            b, getBlobPointer(b, ptrType, materialized[index], index, offset),
            gep, hdl.getAlign());
        i += pointerSize;
        continue;
      }
      materializeStore(i, std::min(simdWidth, e - i));
      i += simdWidth;
    }
    commitCompressedStores();
  }

  // Put materialized blob into a hashmap.
  if (iter == blobs.end()) {
    iter = blobs.try_emplace(space, std::move(materialized)).first;
  } else {
    for (auto &[idx, blob] : materialized)
      iter->second.insert({idx, std::move(blob)});
  }
  return iter->second[refIndex];
}

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Convert a SIMD vector constant.
static Value convertSIMDAttr(ImplicitLocOpBuilder &b,
                             const mlir::LLVMTypeConverter &tc,
                             KGEN::SIMDAttr simd) {
  KGENDType dtype = *simd.getType().getResolvedDType();
  auto asConst = [&](TypedAttr value) {
    return LLVM::ConstantOp::create(b, value);
  };

  // Handle scalar constants.
  if (simd.getValues().size() == 1) {
    const KGEN::DTypeValue &value = simd.getValues().front();
    if (dtype.isBool())
      return asConst(b.getBoolAttr(value.getBoolVal()));
    if (dtype.isInt())
      return asConst(b.getIntegerAttr(
          b.getIntegerType(dtype.getIntegerWidthInBits()), value.getIntVal()));
    if (dtype.isIndex() || dtype.isUIndex() || dtype.isAddress()) {
      Value addr =
          asConst(b.getIntegerAttr(tc.getIndexType(), value.getIndexVal()));
      if (dtype.isIndex() || dtype.isUIndex())
        return addr;
      return LLVM::IntToPtrOp::create(
          b, LLVM::LLVMPointerType::get(b.getContext()), addr);
    }

    FloatType fpType = getEquivalentFloatType(b.getContext(), dtype);
    FloatAttr attrVal = b.getFloatAttr(fpType, value.getFloatVal());
    if (auto intWidth = isFPTyLoweredAsInt(fpType))
      return LLVM::ConstantOp::create(b, b.getIntegerType(*intWidth), attrVal);

    return asConst(attrVal);
  }

  // Handle vector constants.
  if (dtype.isBool()) {
    SmallVector<APInt> values;
    for (const KGEN::DTypeValue &value : simd.getValues())
      values.emplace_back(1, value.getBoolVal());
    return asConst(cast<TypedAttr>(IntArrayElementsAttr::get(
        VectorType::get(values.size(), b.getI1Type()), values)));
  }
  if (dtype.isInt()) {
    SmallVector<APInt> values;
    for (const KGEN::DTypeValue &value : simd.getValues())
      values.push_back(value.getIntVal());
    return asConst(cast<TypedAttr>(IntArrayElementsAttr::get(
        VectorType::get(values.size(),
                        b.getIntegerType(dtype.getIntegerWidthInBits())),
        values)));
  }
  if (dtype.isIndex() || dtype.isUIndex() || dtype.isAddress()) {
    SmallVector<APInt> values;
    auto indexType = cast<IntegerType>(tc.getIndexType());
    for (const KGEN::DTypeValue &value : simd.getValues())
      values.push_back(APInt(indexType.getWidth(), value.getIndexVal()));
    Value addr = asConst(cast<TypedAttr>(IntArrayElementsAttr::get(
        VectorType::get(values.size(), indexType), values)));
    if (dtype.isIndex() || dtype.isUIndex())
      return addr;
    return LLVM::IntToPtrOp::create(
        b,
        VectorType::get(values.size(),
                        LLVM::LLVMPointerType::get(b.getContext())),
        addr);
  }
  SmallVector<APFloat> values;
  for (const KGEN::DTypeValue &value : simd.getValues())
    values.push_back(value.getFloatVal());

  auto fpType = getEquivalentFloatType(b.getContext(), dtype);
  auto attrVal = cast<TypedAttr>(FloatArrayElementsAttr::get(
      VectorType::get(values.size(), fpType), values));

  if (auto intWidth = isFPTyLoweredAsInt(fpType)) {
    return LLVM::ConstantOp::create(
        b, VectorType::get(values.size(), b.getIntegerType(*intWidth)),
        attrVal);
  }

  return asConst(attrVal);
}

/// Lower the string to a pop.global_constant and create a llvm struct of type
/// !llvm.struct<(ptr<i8>, i64)> holding the pointer to the global string and
/// its size.
static Value lowerStringToGlobalConstant(StringAttr strAttr,
                                         ImplicitLocOpBuilder &b,
                                         const POPToLLVMTypeConverter &tc,
                                         InterpreterMemoryConverter &imc) {
  StringRef strAttrRef = strAttr.getValue();
  // This is safe because StringAttr always stores a null terminator.
  // Make sure we have a null terminator, even for empty strings.
  StringRef str(strAttrRef.data(), strAttrRef.size() + 1);
  if (strAttrRef.empty())
    str = StringRef("\0", 1);

  // Add the string to the global string table.
  MemoryHandleAttr hdl = MemoryHandleAttr::get(strAttr.getContext(), str);
  auto global = cast<LLVM::GlobalOp>(imc.getOrCreateGlobal(
      b.getLoc(), MemoryBlobAttr::get(hdl, MemoryKind::ConstGlobal, {}, {},
                                      /*addressSpace=*/0)));
  // The actual string size does not include \0.
  auto sizeType = cast<IntegerType>(tc.getIndexType());
  Value sizeVal = LLVM::ConstantOp::create(
      b, b.getLoc(), IntegerAttr::get(sizeType, strAttr.size()));
  Value undefOp = LLVM::UndefOp::create(
      b, b.getLoc(), getLLVMTypeForKGENStringType(b.getContext(), sizeType));
  Value llvmString =
      LLVM::BitcastOp::create(b, LLVM::LLVMPointerType::get(b.getContext()),
                              LLVM::AddressOfOp::create(b, global));
  Value structVal0 = LLVM::InsertValueOp::create(
      b, b.getLoc(), undefOp, llvmString, static_cast<int64_t>(0));
  return LLVM::InsertValueOp::create(b, b.getLoc(), structVal0, sizeVal, 1);
}

Value KGEN::materializeLLVMStruct(ImplicitLocOpBuilder &b, Type structType,
                                  ValueRange elements) {
  Value container = LLVM::UndefOp::create(b, structType);
  for (auto [index, element] : llvm::enumerate(elements))
    container = LLVM::InsertValueOp::create(b, container, element, index);

  return container;
}

void KGEN::replaceCallWithLLVMCall(mlir::RewriterBase &b, Operation *op,
                                   LLVM::CallOp call) {
  SmallVector<Value> results;

  // Unpack the struct if necessary.
  if (op->getNumResults() <= 1) {
    llvm::append_range(results, call.getResults());
  } else {
    results.reserve(op->getNumResults());
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
      results.push_back(
          LLVM::ExtractValueOp::create(b, op->getLoc(), call.getResult(), i));
    }
  }

  // Replace the call operation.
  b.replaceOp(op, results);
}

ErrorOr<Value> KGEN::convertParameterToLLVM(
    ImplicitLocOpBuilder &b, const POPToLLVMTypeConverter &tc,
    InterpreterMemoryConverter *imc,
    InterpreterMemoryConverter::MaterializationScope *scope, TypedAttr attr) {

  //===--------------------------------------------------------------------===//
  // builtin

  // Convert valueless attributes to undef. A singleton's type has a single
  // inhabitant, so it lowers to a zero-sized aggregate and undef is exact.
  if (isa<SingletonAttr, UninitMemAttr>(attr)) {
    Type type = tc.convertType(attr.getType());
    if (!type)
      return Error("unknown type lowering uninitialized memory");
    return LLVM::UndefOp::create(b, type);
  }

  if (auto intCst = dyn_cast<IntegerAttr>(attr)) {
    // Check for index types a truncate index constants if required.
    if (isa<IndexType>(attr.getType())) {
      return LLVM::ConstantOp::create(
          b,
          b.getIntegerAttr(cast<IntegerType>(tc.getIndexType()),
                           intCst.getValue().trunc(tc.getIndexTypeBitwidth())));
    }

    // Drop the sign on integer attributes; LLVM is signless.
    return LLVM::ConstantOp::create(
        b, b.getIntegerAttr(cast<IntegerType>(tc.convertType(intCst.getType())),
                            intCst.getValue()));
  }

  if (auto fltCst = dyn_cast<FloatAttr>(attr)) {
    Type type = fltCst.getType();
    if (isFPTyLoweredAsInt(type)) {
      return LLVM::ConstantOp::create(b, tc.convertType(type),
                                      fltCst.getValue().bitcastToAPInt());
    }

    // Float attributes are fine as-is.
    return LLVM::ConstantOp::create(b, fltCst);
  }

  // Convert DType constants to `i8` constants of the DType's enum value.
  if (auto dtypeCst = dyn_cast<DTypeConstantAttr>(attr))
    return LLVM::ConstantOp::create(
        b, b.getI8IntegerAttr(dtypeCst.getDType().getValue()));

  // Convert `#kgen.none` to an empty struct.
  if (isa<NoneAttr>(attr))
    return LLVM::UndefOp::create(
        b, LLVM::LLVMStructType::getLiteral(b.getContext(), {}));

  // Convert pointer attributes (usually null pointers).
  if (auto ptr = dyn_cast<PointerAttr>(attr)) {
    return LLVM::IntToPtrOp::create(
        b, tc.convertType(ptr.getType()),
        LLVM::ConstantOp::create(
            b, b.getIntegerAttr(tc.getIndexType(), ptr.getAddr())),
        LLVM::DereferenceableAttr{});
  }

  // We can lower `StoreToMemAttr` by writing the underlying value into a
  // stack allocation.
  if (auto store = dyn_cast<StoreToMemAttr>(attr)) {
    ErrorOr<Value> loweredValue =
        convertParameterToLLVM(b, tc, imc, scope, store.getValue());
    if (loweredValue.isError())
      return loweredValue;
    auto value = loweredValue.get();
    unsigned align = tc.getTypeABIAlign(value.getType());
    Value ptr = LLVM::AllocaOp::create(
        b, tc.convertType(attr.getType()), value.getType(),
        LLVM::ConstantOp::create(b, b.getI64IntegerAttr(1)), align);
    LLVM::StoreOp::create(b, value, ptr, align);
    return ptr;
  }

  // Materialize memrefs from the interpreter.
  if (scope)
    if (auto ref = dyn_cast<MemRefAttr>(attr))
      return scope->convertMemRef(b, ref);

  // Convert string constant to a struct{ptr, size} of type
  // !llvm.struct<(ptr<i8>, index).
  if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    if (!imc)
      return Error("cannot lower string constant without target memory model");
    return lowerStringToGlobalConstant(strAttr, b, tc, *imc);
  }

  if (auto callable = dyn_cast<CallableSymbolAttrInterface>(attr)) {
    if (cast<FuncTypeGeneratorType>(callable.getType()).getBody().isCapturing())
      return Error("TODO: capturing closures cannot be materialized as runtime "
                   "values");
    return LLVM::AddressOfOp::create(
        b, tc.convertType(callable.getType()),
        cast<FlatSymbolRefAttr>(callable.getSymbol()));
  }

  //===--------------------------------------------------------------------===//
  // POP

  // Convert SIMD constants to an array of integer or float constants.
  if (auto simd = dyn_cast<KGEN::SIMDAttr>(attr))
    return convertSIMDAttr(b, tc, simd);

  // Convert array constants to LLVM array constants.
  if (auto arrayAttr = dyn_cast<POP::ArrayAttr>(attr)) {
    Type type = tc.convertType(arrayAttr.getType());
    if (!type)
      return Error("cannot lower array constant with unknown type");
    Value aggregate = LLVM::UndefOp::create(b, type);
    for (auto [idx, value] : llvm::enumerate(arrayAttr.getValues())) {
      ErrorOr<Value> loweredValue =
          convertParameterToLLVM(b, tc, imc, scope, value);
      if (loweredValue.isError())
        return loweredValue;
      aggregate =
          LLVM::InsertValueOp::create(b, aggregate, loweredValue.get(), idx);
    }
    return aggregate;
  }

  // Convert struct constants to LLVM struct constants.
  if (auto structAttr = dyn_cast<StructAttr>(attr)) {
    auto kgenStructType = cast<StructType>(structAttr.getType());
    Type type = tc.convertType(kgenStructType);
    if (!type)
      return Error("cannot lower struct constant with unknown type");
    Value aggregate = LLVM::UndefOp::create(b, type);
    for (auto [logicalIdx, value] : llvm::enumerate(structAttr.getValues())) {
      ErrorOr<Value> loweredValue =
          convertParameterToLLVM(b, tc, imc, scope, value);
      if (loweredValue.isError())
        return loweredValue;
      // Use remapped field index to account for alignment padding.
      int64_t llvmIdx = tc.getRemappedFieldIndex(kgenStructType, logicalIdx);
      aggregate = LLVM::InsertValueOp::create(b, aggregate, loweredValue.get(),
                                              llvmIdx);
    }
    return aggregate;
  }

  // Bitpack union constants.
  if (auto unionAttr = dyn_cast<POP::UnionAttr>(attr)) {
    ErrorOr<Value> loweredValue =
        convertParameterToLLVM(b, tc, imc, scope, unionAttr.getValue());
    if (loweredValue.isError())
      return loweredValue;
    auto value = loweredValue.get();

    auto contentType =
        cast_or_null<LLVM::LLVMStructType>(tc.convertType(unionAttr.getType()));
    if (!contentType)
      return Error("cannot lower union constant with unknown type");
    VariantHelper helper(b, b.getLoc(), tc);
    return helper.materializeLLVMUnion(contentType, value);
  }

  // Otherwise we have a failure, try to report a useful error message.

  // If this is an param attribute that refused to fold, see if we're able to
  // get a custom error message from it to explain what is going on.
  if (auto itf = ::dyn_cast<ParameterAttr>(attr)) {
    auto errorMessage = itf.validateForElaborator();
    if (errorMessage.isError())
      return errorMessage.takeError();
  }

  // Unknown attribute to convert.
  return Error("cannot lower unknown attribute to LLVM: " +
               getParamAsString(attr));
}

//===----------------------------------------------------------------------===//
// getWorkGroupSizeRange
//===----------------------------------------------------------------------===//

namespace {
/// Helper to extract the min and max values of a work-group-size attribute.
template <typename AttrT>
ErrorOr<std::pair<int64_t, int64_t>> getWorkGroupSizeRangeHelper(AttrT attr) {
  SmallVector<int64_t> values;
  if constexpr (std::is_same_v<POP::ArrayAttr, AttrT>) {
    if (attr.getValues().size() != 1 && attr.getValues().size() != 2)
      return Error("ArrayAttr must contain exactly one or two values");

    values =
        llvm::map_to_vector(attr.getValues(), [](Attribute attr) -> int64_t {
          if (auto integerAttr = ::dyn_cast<IntegerAttr>(attr))
            return static_cast<int64_t>(integerAttr.getInt());
          return static_cast<int64_t>(::cast<KGEN::SIMDAttr>(attr)
                                          .getValues()
                                          .front()
                                          .getIntVal()
                                          .getSExtValue());
        });
  } else if constexpr (std::is_same_v<IntegerAttr, AttrT>) {
    values.push_back(static_cast<int64_t>(attr.getInt()));
  } else {
    llvm_unreachable(
        "must be either an ArrayAttr of 1 element or an IntegerAttr");
  }
  if (values.size() == 1)
    values.insert(values.begin(), 1);

  return std::make_pair(values[0], values[1]);
}
} // namespace

ErrorOr<std::pair<int64_t, int64_t>>
KGEN::getWorkGroupSizeRange(Attribute value) {
  if (auto array = dyn_cast<POP::ArrayAttr>(value)) {
    return getWorkGroupSizeRangeHelper(array);
  }
  if (auto integer = dyn_cast<IntegerAttr>(value)) {
    return getWorkGroupSizeRangeHelper(integer);
  }
  return Error("attribute type must be either an ArrayAttr of 1 or 2 elements "
               "or an IntegerAttr");
}

//===----------------------------------------------------------------------===//
// squashPointlessCasts
//===----------------------------------------------------------------------===//

mlir::Value KGEN::squashPointlessCasts(mlir::Value v) {
  auto cast1Op = v.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast1Op || cast1Op.getNumOperands() != 1 || cast1Op.getNumResults() != 1)
    return v;

  auto cast2Op =
      cast1Op.getOperand(0).getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast2Op || cast1Op.getNumOperands() != 1 ||
      cast1Op.getNumResults() != 1 ||
      cast2Op.getOperand(0).getType() != v.getType())
    return v;

  return squashPointlessCasts(cast2Op.getOperand(0));
}
