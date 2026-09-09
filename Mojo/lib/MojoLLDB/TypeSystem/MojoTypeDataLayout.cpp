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

#include "MojoTypeDataLayout.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/POPDialect/POPTypes.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;

//===----------------------------------------------------------------------===//
// MojoTypeDataLayoutContext::Impl
//===----------------------------------------------------------------------===//

struct MojoTypeDataLayoutContext::Impl {
  Impl(MojoParserContext &context, TargetInfoAttr targetInfo)
      : context(context), targetInfo(targetInfo) {}

  /// Implementation of `MojoTypeDataLayoutContext::getOrCalculate`
  const std::optional<MojoTypeDataLayout> &
  getOrCalculate(MojoASTTypeRef typeRef);

  /// Calculate the data layout of the given type.
  std::optional<MojoTypeDataLayout> calculate(MojoASTTypeRef type);

  /// Calculate the data layout of the given struct.
  std::optional<MojoTypeDataLayout>
  calculateForStruct(MojoASTTypeRef type, LIT::StructDeclOp structOp);

  /// Calculate the data layout of the given struct.
  std::optional<MojoTypeDataLayout>
  calculateForStruct(MojoASTTypeRef typeRef, KGEN::StructType structType);

  /// Calculate the data layout of the given variant.
  std::optional<MojoTypeDataLayout>
  calculateForVariant(MojoASTTypeRef typeRef, VariantType variantType);

  /// Calculate the data layout of the given union. Each member sits at
  /// offset 0; the type's overall size is the max member size, rounded up
  /// to the union's alignment.
  std::optional<MojoTypeDataLayout> calculateForUnion(MojoASTTypeRef typeRef,
                                                      POP::UnionType unionType);

  /// Calculate the data layout of a struct-like collection of types.
  std::optional<MojoTypeDataLayout>
  calculateForStructLike(ArrayRef<MojoASTTypeRef> fieldTypes);

  DenseMap<Type, std::optional<MojoTypeDataLayout>> cache;
  MojoParserContext &context;
  TargetInfoAttr targetInfo;
};

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForStructLike(
    ArrayRef<MojoASTTypeRef> fieldTypes) {
  uint64_t size = 0, align = 1;
  MojoTypeDataLayout layout;

  for (MojoASTTypeRef fieldType : fieldTypes) {
    auto &fieldLayout = getOrCalculate(fieldType);
    // If we cannot calculate the layout of a field, we bail because reporting
    // an incorrect size for this struct might result in miscalculating the
    // locations of neighboring variables.
    if (!fieldLayout)
      return {};

    uint64_t fieldOffset = llvm::alignTo(size, fieldLayout->getAlignment());
    layout.addField({fieldOffset, fieldLayout->getByteSize(),
                     fieldLayout->getAlignment(), fieldType});
    size = fieldOffset + fieldLayout->getByteSize();
    align = std::max(align, fieldLayout->getAlignment());
  }
  layout.setByteSize(llvm::alignTo(size, align));
  layout.setAlignment(align);
  return layout;
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForStruct(
    MojoASTTypeRef typeRef, LIT::StructDeclOp structOp) {
  auto refType = sugarCast<LIT::StructType>(typeRef.getMLIRType());
  return calculateForStructLike(llvm::map_to_vector(
      structOp.getFieldDecls(), [&](LIT::StructFieldOp field) {
        MojoASTTypeRef fieldType =
            MojoASTTypeRef(field.getTypeAttr().getValue());
        // If the StructType has parameters, try to evaluate and substitute
        // them into the type.
        if (!refType.getParamValues().empty()) {
          fieldType = context.concretizeType(typeRef, refType.getParamValues(),
                                             fieldType);
        }
        return fieldType;
      }));
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForStruct(
    MojoASTTypeRef typeRef, KGEN::StructType structType) {
  std::optional<SmallVector<Type>> elementTypes = structType.getElementTypes();
  if (!elementTypes)
    return {};
  return calculateForStructLike(llvm::map_to_vector(
      *elementTypes, [&](Type value) { return MojoASTTypeRef(value); }));
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForVariant(MojoASTTypeRef typeRef,
                                                     VariantType variantType) {
  // FIXME(35592): We are disabling printing of non-concrete variants for the
  // time being, because they can't be introspected.
  if (!isa<ParamListAttr>(variantType.getVariadic()))
    return {};

  std::optional<uint64_t> overallAlign = variantType.getTypeAlign(targetInfo);
  if (!overallAlign)
    return {};

  MojoTypeDataLayout layout;
  uint64_t maxVariantSize = 0;
  for (MojoASTTypeRef fieldType : variantType.getTypes()) {
    auto &fieldLayout = getOrCalculate(fieldType);
    if (!fieldLayout)
      return {};

    layout.addField({0, fieldLayout->getByteSize(), fieldLayout->getAlignment(),
                     fieldType});
    maxVariantSize = std::max(maxVariantSize, fieldLayout->getByteSize());
  }
  maxVariantSize = llvm::alignTo(maxVariantSize, *overallAlign);

  uint64_t discrByteSize =
      llvm::divideCeil(variantType.getDiscrSizeInBits(), CHAR_BIT);
  Type discrType = IntegerType::get(variantType.getContext(),
                                    variantType.getDiscrSizeInBits());
  layout.addField({maxVariantSize, discrByteSize, *overallAlign, discrType});

  layout.setByteSize(
      llvm::alignTo(maxVariantSize + discrByteSize, *overallAlign));
  layout.setAlignment(*overallAlign);
  return layout;
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForUnion(MojoASTTypeRef typeRef,
                                                   POP::UnionType unionType) {
  if (!unionType.isResolved())
    return {};

  std::optional<int64_t> overallSize = unionType.getTypeSize(targetInfo);
  std::optional<int64_t> overallAlign = unionType.getTypeAlign(targetInfo);
  if (!overallSize || !overallAlign)
    return {};

  MojoTypeDataLayout layout;
  for (Type memberType : unionType.getTypes()) {
    MojoASTTypeRef memberRef(memberType);
    auto &memberLayout = getOrCalculate(memberRef);
    if (!memberLayout)
      return {};
    layout.addField({/*byteOffset=*/0, memberLayout->getByteSize(),
                     memberLayout->getAlignment(), memberRef});
  }
  layout.setByteSize(*overallSize);
  layout.setAlignment(*overallAlign);
  return layout;
}

const std::optional<MojoTypeDataLayout> &
MojoTypeDataLayoutContext::Impl::getOrCalculate(MojoASTTypeRef type) {
  auto it = cache.find(type.getMLIRType());
  if (it != cache.end())
    return it->second;
  auto ret = cache.try_emplace(type.getMLIRType(), calculate(type));
  return ret.first->second;
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculate(MojoASTTypeRef type) {
  // A REPLResultRefType is effectively a pointer, so we transform it into one.
  if (auto replType = sugarDynCast<LIT::REPLResultRefType>(type.getMLIRType()))
    return calculate(KGEN::PointerType::get(replType.getElementType()));

  if (MojoASTDeclRef declRef = context.getDecl(type)) {
    if (LIT::StructDeclOp structDeclOp =
            dyn_cast_if_present<LIT::StructDeclOp>(declRef.getIfOperation())) {
      return calculateForStruct(type, structDeclOp);
    }
  }

  if (auto structType = dyn_cast<KGEN::StructType>(type))
    return calculateForStruct(type, structType);

  if (auto variantType = dyn_cast<VariantType>(type))
    return calculateForVariant(type, variantType);

  if (auto unionType = dyn_cast<POP::UnionType>(type))
    return calculateForUnion(type, unionType);

  std::optional<uint64_t> size =
      DataLayoutInterface::getTypeStoreSize(targetInfo, type.getMLIRType());
  std::optional<uint64_t> alignment =
      DataLayoutInterface::getTypeABIAlign(targetInfo, type.getMLIRType());

  if (!size || !alignment)
    return {};

  // If a type has -1 size, it will crash the debugger as it'll attempt to read
  // too much memory, so we change the size to 0, because no data should be
  // read. We also need to report these cases at least with an error message.
  if (size == std::numeric_limits<uint64_t>::max()) {
    llvm::errs() << "Error: MLIR type '"
                 << type.getAsString(context.getSharedState())
                 << "' has an invalid size of " << size << " (-1).\n";
    size = 0;
    alignment = 1;
  }
  return MojoTypeDataLayout(*size, *alignment);
}

//===----------------------------------------------------------------------===//
// MojoTypeDataLayoutContext
//===----------------------------------------------------------------------===//

MojoTypeDataLayoutContext::~MojoTypeDataLayoutContext() = default;

MojoTypeDataLayoutContext::MojoTypeDataLayoutContext(MojoParserContext &context,
                                                     TargetInfoAttr targetInfo)
    : impl(std::make_unique<MojoTypeDataLayoutContext::Impl>(context,
                                                             targetInfo)) {}

const std::optional<MojoTypeDataLayout> &
MojoTypeDataLayoutContext::getOrCalculate(MojoASTTypeRef type) {
  return impl->getOrCalculate(type);
}

void MojoTypeDataLayoutContext::invalidateCache(MojoASTTypeRef typeRef) {
  impl->cache.erase(typeRef.getMLIRType());
}
