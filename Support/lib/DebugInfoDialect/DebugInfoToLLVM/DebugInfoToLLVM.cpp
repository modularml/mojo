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

#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"
#include "Target/TargetAdapter.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

using namespace M;
using namespace M::DebugInfo;

namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// MetadataConversion
//===----------------------------------------------------------------------===//

namespace {
/// This class handles the conversion of DebugInfo metadata to the LLVM IR
/// metadata equivalent.
struct MetadataConverter {
  MetadataConverter(DebugInfo::DebugInfoTypeConverter &typeConverter,
                    M::TargetInfoAttr target)
      : typeConverter(typeConverter), target(target) {}

  /// Convert the given derived debug info attribute to LLVM.
  template <typename T>
  auto convertAttr(T attr) {
    // Infer the LLVM type from the attribute kind.
    using LLVMTypeT = std::remove_pointer_t<decltype(convertAttrImpl(attr))>;
    return cast_or_null<LLVMTypeT>(convertAttrImpl(DIAttr(attr)));
  }

  /// Convert the given derived debug info type to LLVM.
  template <typename T>
  auto convertType(T type) {
    // Infer the LLVM type from the Type kind.
    using LLVMTypeT = std::remove_pointer_t<decltype(convertTypeImpl(type))>;
    return cast_or_null<LLVMTypeT>(convertTypeImpl(DIType(type)));
  }

private:
  Attribute convertAttrImpl(DIAttr attr);
  LLVM::DICompileUnitAttr convertAttrImpl(DICompileUnitAttr attr);
  LLVM::DIFileAttr convertAttrImpl(DIFileAttr attr);
  LLVM::DILexicalBlockAttr convertAttrImpl(DILexicalBlockAttr attr);
  LLVM::DILocalVariableAttr convertAttrImpl(DILocalVariableAttr attr);
  LLVM::DIScopeAttr convertAttrImpl(DIScopeAttr attr);
  LLVM::DISubprogramAttr convertAttrImpl(DISubprogramAttr attr);
  LocationAttr convertAttrImpl(DICallLocAttr attr);

  LLVM::DIExpressionAttr convertAttrImpl(DIAggregatesIntoExprAttr attr);
  LLVM::DIExpressionAttr convertAttrImpl(DIRefOfExprAttr attr);
  LLVM::DIExpressionAttr convertAttrImpl(DIDerefExprAttr attr);
  LLVM::DIExpressionAttr convertAttrImpl(DIIRValueExprAttr attr);

  LLVM::DITypeAttr convertTypeImpl(DIType type);
  LLVM::DITypeAttr convertTypeImpl(DIArrayType type);
  LLVM::DIBasicTypeAttr convertTypeImpl(DIBasicType type);
  LLVM::DITypeAttr convertTypeImpl(DIPointerType type);
  LLVM::DITypeAttr convertTypeImpl(DIStructType type);
  LLVM::DISubroutineTypeAttr convertTypeImpl(DISubroutineType type);
  LLVM::DITypeAttr convertTypeImpl(DIUnresolvedMLIRType type);
  LLVM::DIBasicTypeAttr convertTypeImpl(DIUnspecifiedType type);
  LLVM::DITypeAttr convertTypeImpl(DIVariantType type);
  LLVM::DITypeAttr convertTypeImpl(DIVectorType type);

  DebugInfo::DebugInfoTypeConverter &typeConverter;
  M::TargetInfoAttr target;
  DenseMap<Attribute, Attribute> convertedAttrs;
  DenseMap<Type, LLVM::DITypeAttr> convertedTypes;
};
} // namespace

//===----------------------------------------------------------------------===//
// Attributes

Attribute MetadataConverter::convertAttrImpl(DIAttr attr) {
  if (!attr)
    return nullptr;
  if (Attribute converted = convertedAttrs.lookup(attr))
    return converted;

  Attribute result =
      TypeSwitch<DIAttr, Attribute>(attr)
          .Case<DIAggregatesIntoExprAttr, DICompileUnitAttr, DIDerefExprAttr,
                DIFileAttr, DIIRValueExprAttr, DILexicalBlockAttr,
                DILocalVariableAttr, DIRefOfExprAttr, DISubprogramAttr,
                DICallLocAttr>(
              [&](auto attr) { return convertAttrImpl(attr); });
  return convertedAttrs[attr] = result;
}

LLVM::DICompileUnitAttr
MetadataConverter::convertAttrImpl(DICompileUnitAttr attr) {
  return LLVM::DICompileUnitAttr::get(
      mlir::DistinctAttr::create(mlir::UnitAttr::get(attr.getContext())),
      attr.getSourceLanguage(), convertAttr(attr.getFile()), attr.getProducer(),
      attr.getIsOptimized(),
      static_cast<LLVM::DIEmissionKind>(attr.getEmissionKind()),
      /*isDebugInfoForProfiling=*/false,
      static_cast<LLVM::DINameTableKind>(attr.getNameTableKind()));
}

LLVM::DIFileAttr MetadataConverter::convertAttrImpl(DIFileAttr attr) {
  return LLVM::DIFileAttr::get(attr.getContext(), attr.getName(),
                               attr.getDirectory());
}

LLVM::DILexicalBlockAttr
MetadataConverter::convertAttrImpl(DILexicalBlockAttr attr) {
  return LLVM::DILexicalBlockAttr::get(convertAttr(attr.getScope()),
                                       convertAttr(attr.getFile()),
                                       attr.getLine(), attr.getColumn());
}

LLVM::DILocalVariableAttr
MetadataConverter::convertAttrImpl(DILocalVariableAttr attr) {
  return LLVM::DILocalVariableAttr::get(
      convertAttr(attr.getScope()), attr.getName(), convertAttr(attr.getFile()),
      attr.getLine(), attr.getArg(), attr.getAlignInBits(),
      convertType(attr.getType()), (LLVM::DIFlags)attr.getFlags());
}

LLVM::DISubprogramAttr
MetadataConverter::convertAttrImpl(DISubprogramAttr attr) {
  SmallVector<LLVM::DINodeAttr> annotations;
  auto annotation = mlir::LLVM::DIAnnotationAttr::get(
      attr.getContext(), StringAttr::get(attr.getContext(), "mojo_source_name"),
      attr.getSourceName().encode());
  annotations.push_back(annotation);

  return LLVM::DISubprogramAttr::get(
      attr.getContext(),
      mlir::DistinctAttr::create(mlir::UnitAttr::get(attr.getContext())),
      convertAttr(attr.getCompileUnit()), convertAttr(attr.getScope()),
      attr.getSourceName().getName(), attr.getLinkageName(),
      convertAttr(attr.getFile()), attr.getLine(), attr.getScopeLine(),
      static_cast<LLVM::DISubprogramFlags>(attr.getSubprogramFlags()),
      convertType(attr.getType()), /*retainedNodes=*/{}, annotations);
}

LocationAttr MetadataConverter::convertAttrImpl(DICallLocAttr attr) {
  return attr.getCallLoc();
}

//===----------------------------------------------------------------------===//
// DIExpression Attributes

LLVM::DIExpressionAttr
MetadataConverter::convertAttrImpl(DIAggregatesIntoExprAttr attr) {
  auto prefix = llvm::dyn_cast_or_null<LLVM::DIExpressionAttr>(
      convertAttr(cast<DIAttr>(attr.getFieldExpr())));
  if (!prefix)
    return {};

  // Struct and array aggregates both lower to a DICompositeTypeAttr.
  auto llvmAggregate = cast<LLVM::DICompositeTypeAttr>(
      convertType(cast<DIType>(attr.getType())));

  uint64_t prefixSize;
  uint64_t fieldSize;
  if (auto arrayType = dyn_cast<DIArrayType>(attr.getType())) {
    // An array composite carries DISubranges, not per-field members, so
    // element `i` is a uniform, tightly-packed fragment.
    uint64_t count = arrayType.getElementCount();
    if (count == 0)
      return prefix;
    fieldSize = llvmAggregate.getSizeInBits() / count;
    prefixSize = attr.getFieldIndex() * fieldSize;
  } else {
    if (llvmAggregate.getElements().size() == 1)
      return prefix;

    auto targetMember = cast<LLVM::DIDerivedTypeAttr>(
        llvmAggregate.getElements()[attr.getFieldIndex()]);
    fieldSize = targetMember.getSizeInBits();

    prefixSize = 0;
    for (LLVM::DINodeAttr member :
         llvmAggregate.getElements().take_front(attr.getFieldIndex())) {
      auto memberType = cast<LLVM::DIDerivedTypeAttr>(member);
      uint64_t sizeInBits = memberType.getSizeInBits();
      uint32_t alignInBits = memberType.getAlignInBits();
      prefixSize =
          llvm::alignTo(prefixSize, std::max(1u, alignInBits)) + sizeInBits;
    }

    if (uint32_t fieldAlignment = targetMember.getAlignInBits())
      prefixSize = llvm::alignTo(prefixSize, fieldAlignment);
  }

  // A fragment covering the whole aggregate is elided.
  if (fieldSize == llvmAggregate.getSizeInBits())
    return prefix;

  SmallVector<LLVM::DIExpressionElemAttr> expr(prefix.getOperations());
  expr.push_back(LLVM::DIExpressionElemAttr::get(
      attr.getContext(), llvm::dwarf::DW_OP_LLVM_fragment,
      {prefixSize, fieldSize}));
  return LLVM::DIExpressionAttr::get(attr.getContext(), expr);
}

LLVM::DIExpressionAttr
MetadataConverter::convertAttrImpl(DIRefOfExprAttr attr) {
  auto prefix = dyn_cast_or_null<LLVM::DIExpressionAttr>(
      convertAttr(cast<DIAttr>(attr.getValueExpr())));
  if (!prefix)
    return {};

  SmallVector<LLVM::DIExpressionElemAttr> expr(prefix.getOperations());
  expr.push_back(LLVM::DIExpressionElemAttr::get(
      attr.getContext(), llvm::dwarf::DW_OP_LLVM_implicit_pointer, {}));
  return LLVM::DIExpressionAttr::get(attr.getContext(), expr);
}

LLVM::DIExpressionAttr
MetadataConverter::convertAttrImpl(DIDerefExprAttr attr) {
  auto prefix = dyn_cast_or_null<LLVM::DIExpressionAttr>(
      convertAttr(cast<DIAttr>(attr.getPtrExpr())));
  if (!prefix)
    return {};

  SmallVector<LLVM::DIExpressionElemAttr> expr(prefix.getOperations());
  expr.push_back(LLVM::DIExpressionElemAttr::get(attr.getContext(),
                                                 llvm::dwarf::DW_OP_deref, {}));
  return LLVM::DIExpressionAttr::get(attr.getContext(), expr);
}

LLVM::DIExpressionAttr
MetadataConverter::convertAttrImpl(DIIRValueExprAttr attr) {
  // The base case is just an empty/trivial location list.
  return LLVM::DIExpressionAttr::get(attr.getContext(), {});
}

//===----------------------------------------------------------------------===//
// Types

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIType type) {
  if (!type)
    return {};
  if (LLVM::DITypeAttr converted = convertedTypes.lookup(type))
    return converted;

  // Run the type through the type converter to resolve any lingering types.
  type = typeConverter.convertDebugType(type);

  // Dispatch to the right metadata converter.
  LLVM::DITypeAttr result =
      TypeSwitch<DIType, LLVM::DITypeAttr>(type)
          .Case<DIArrayType, DIBasicType, DIPointerType, DIStructType,
                DISubroutineType, DIUnresolvedMLIRType, DIUnspecifiedType,
                DIVariantType, DIVectorType>(
              [&](auto type) { return convertTypeImpl(type); });
  return convertedTypes[type] = result;
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIArrayType type) {
  Builder builder(type.getContext());
  auto element = LLVM::DISubrangeAttr::get(
      type.getContext(), builder.getI64IntegerAttr(type.getElementCount()),
      /*lowerBound=*/nullptr, /*upperBound=*/nullptr, /*stride=*/nullptr);
  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_array_type,
      StringAttr::get(type.getContext()), nullptr, /*line=*/0,
      /*scope=*/nullptr, convertType(type.getElementType()),
      LLVM::DIFlags::Zero, type.getSizeInBits(), /*alignInBits=*/0,
      /*dataLocation=*/{}, /*rank=*/{}, /*allocated=*/{}, /*associated=*/{},
      /*identifier=*/{}, /*discriminator=*/{}, element);
}

LLVM::DIBasicTypeAttr MetadataConverter::convertTypeImpl(DIBasicType type) {
  return LLVM::DIBasicTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_base_type, type.getName(),
      type.getSizeInBits(), type.getEncoding());
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIPointerType type) {
  return LLVM::DIDerivedTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_pointer_type,
      /*name=*/nullptr, /*file=*/{}, /*line=*/0, /*scope=*/{},
      convertType(type.getElementType()), type.getSizeInBits(),
      type.getAlignInBits(), /*offsetInBits=*/0, type.getAddressSpace(),
      /*flags=*/{}, /*extraData=*/{});
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIStructType type) {
  SmallVector<LLVM::DINodeAttr> elementTypes;

  // Convert each of the members.
  uint64_t structSize = 0;
  uint32_t structAlign = 0;
  for (DIMemberType member : type.getMembers()) {
    // Compute the offset/align of the element.
    uint64_t sizeInBits = member.getSizeInBits();
    uint32_t alignInBits = member.getAlignInBits();
    uint64_t offsetInBits =
        llvm::alignTo(structSize, std::max(1u, alignInBits));
    structSize = offsetInBits + sizeInBits;
    structAlign = std::max(structAlign, alignInBits);

    elementTypes.push_back(LLVM::DIDerivedTypeAttr::get(
        member.getContext(), llvm::dwarf::DW_TAG_member, member.getName(),
        /*file=*/{}, /*line=*/0, /*scope=*/{}, convertType(member.getType()),
        sizeInBits, alignInBits, offsetInBits, /*dwarfAddressSpace=*/{},
        /*flags=*/{}, /*extraData=*/{}));
  }

  // Pad the struct size to the largest element alignment.
  if (structAlign)
    structSize = llvm::alignTo(structSize, structAlign);

  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_structure_type, type.getName(),
      /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, /*baseType=*/nullptr,
      LLVM::DIFlags::Zero, structSize, structAlign,
      /*dataLocation=*/{}, /*rank=*/{},
      /*allocated=*/{}, /*associated=*/{}, /*identifier=*/{},
      /*discriminator=*/{}, elementTypes);
}

LLVM::DISubroutineTypeAttr
MetadataConverter::convertTypeImpl(DISubroutineType type) {
  // Grab the result type if we have one.
  SmallVector<LLVM::DITypeAttr> convertedTypes;
  if (type.getResultTypes().size() == 1)
    convertedTypes.push_back(convertType(type.getResultTypes()[0]));
  else
    convertedTypes.push_back(LLVM::DINullTypeAttr::get(type.getContext()));

  for (auto argType : type.getArgumentTypes())
    convertedTypes.push_back(convertType(argType));
  return LLVM::DISubroutineTypeAttr::get(
      type.getContext(), type.getCallingConvention(), convertedTypes);
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIUnresolvedMLIRType type) {
  // TODO: We could choose to fail here if we get an unresolved type, as opposed
  // to what's described here (i.e. just replace with an unspecified type and
  // use the string representation of the type).
  return LLVM::DIBasicTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_unspecified_type,
      mlir::debugString(type.getType()), /*sizeInBits=*/0, /*encoding=*/0);
}

LLVM::DIBasicTypeAttr
MetadataConverter::convertTypeImpl(DIUnspecifiedType type) {
  return LLVM::DIBasicTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_unspecified_type, type.getName(),
      /*sizeInBits=*/0, /*encoding=*/0);
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIVariantType type) {
  MLIRContext *context = type.getContext();
  SmallVector<LLVM::DINodeAttr> variantTypes;

  // Convert each of the members.
  for (DIMemberType member : type.getVariants()) {
    // TODO(#30619): add discriminator value to the DW_TAG_variant entry once
    // upstream is ready.
    LLVM::DITypeAttr memberType = LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_member, member.getName(),
        /*file=*/{}, /*line=*/0, /*scope=*/{}, convertType(member.getType()),
        member.getSizeInBits(), member.getAlignInBits(), 0,
        /*dwarfAddressSpace=*/{}, /*flags=*/{}, /*extraData=*/{});
    variantTypes.push_back(memberType);
  }

  // TODO(#30619): add discriminator field to the DW_TAG_variant_part entry once
  // upstream is ready.
  return LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_variant_part, StringAttr::get(context),
      nullptr, 0, nullptr, nullptr, LLVM::DIFlags::Zero, type.getSizeInBits(),
      type.getAlignInBits(), /*dataLocation=*/{}, /*rank=*/{},
      /*allocated=*/{}, /*associated=*/{}, /*identifier=*/{},
      /*discriminator=*/{}, variantTypes);
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIVectorType type) {
  Builder builder(type.getContext());
  auto element = LLVM::DISubrangeAttr::get(
      type.getContext(), builder.getI64IntegerAttr(type.getElementCount()),
      /*lowerBound=*/nullptr, /*upperBound=*/nullptr, /*stride=*/nullptr);
  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_array_type, type.getName(),
      nullptr, /*line=*/0, /*scope=*/nullptr,
      convertType(type.getElementType()), LLVM::DIFlags::Vector,
      type.getSizeInBits(), /*alignInBits=*/0, /*dataLocation=*/{},
      /*rank=*/{},
      /*allocated=*/{}, /*associated=*/{}, /*identifier=*/{},
      /*discriminator=*/{}, element);
}

//===----------------------------------------------------------------------===//
// KillOp
//===----------------------------------------------------------------------===//

namespace {
struct ConvertKillOp : public OpRewritePattern<KillOp> {
  ConvertKillOp(MLIRContext *ctx, DIAttrTypeReplacer &replacer)
      : OpRewritePattern<KillOp>(ctx), replacer(replacer) {}

  LogicalResult matchAndRewrite(KillOp op,
                                PatternRewriter &rewriter) const override {
    auto undef = LLVM::UndefOp::create(
        rewriter, op.getLoc(),
        LLVM::LLVMStructType::getLiteral(getContext(), {}));
    LLVM::DbgValueOp::create(
        rewriter, replacer.replace<LocationAttr>(op.getLoc()), undef,
        replacer.replace<LLVM::DILocalVariableAttr>(op.getValueInfo()));
    rewriter.eraseOp(op);
    return success();
  }

  /// The replacer used to update attributes.
  DIAttrTypeReplacer &replacer;
};
} // namespace

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

namespace {
struct ConvertValueOp : public OpRewritePattern<ValueOp> {
  ConvertValueOp(MLIRContext *ctx, DIAttrTypeReplacer &replacer)
      : OpRewritePattern<ValueOp>(ctx), replacer(replacer) {}

  LogicalResult matchAndRewrite(ValueOp op,
                                PatternRewriter &rewriter) const override {
    LLVM::DbgValueOp::create(
        rewriter, replacer.replace<LocationAttr>(op.getLoc()), op.getValue(),
        replacer.replace<LLVM::DILocalVariableAttr>(op.getValueInfo()),
        replacer.replace<LLVM::DIExpressionAttr>(op.getConversionExpr()));
    rewriter.eraseOp(op);
    return success();
  }

  /// The replacer used to update attributes.
  DIAttrTypeReplacer &replacer;
};
} // namespace

//===----------------------------------------------------------------------===//
// OpLocations
//===----------------------------------------------------------------------===//

namespace {
/// This pattern handles converting the debug information for non-debuginfo
/// operations.
struct ConvertOpLocations : public mlir::RewritePattern {
  ConvertOpLocations(MLIRContext *ctx, DIAttrTypeReplacer &replacer)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        replacer(replacer) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] {
      // Update the debug info attributes within the locations of this operation
      // to use the LLVM equivalent.
      replacer.replaceElementsIn(op);
    });
    return success();
  }

  /// The replacer used to update attributes.
  DIAttrTypeReplacer &replacer;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateDebugInfoToLLVMPatterns(DIAttrTypeReplacer &replacer,
                                            RewritePatternSet &patterns) {
  patterns.add<ConvertKillOp, ConvertValueOp, ConvertOpLocations>(
      patterns.getContext(), replacer);
}

//===----------------------------------------------------------------------===//
// DebugInfoToLLVMTypeConverter
//===----------------------------------------------------------------------===//

namespace {
struct DebugInfoToLLVMTypeConverter : public DebugInfo::DebugInfoTypeConverter {
  DebugInfoToLLVMTypeConverter(mlir::LLVMTypeConverter &typeConverter) {
    addUnresolvedConverter(typeConverter);

    // TODO: Cover more LLVM types here as needed.
    const llvm::DataLayout &dataLayout = typeConverter.getDataLayout();
    addConversion([&](LLVM::LLVMPointerType type) -> DebugInfo::DIType {
      // Convert the pointer element type.
      DIType diEltType =
          DebugInfo::DIUnspecifiedType::get(type.getContext(), "opaque");

      size_t size = dataLayout.getPointerSizeInBits();
      llvm::Align align =
          dataLayout.getPointerPrefAlignment(type.getAddressSpace());
      return DebugInfo::DIPointerType::get(
          diEltType, size, align.value() * CHAR_BIT, type.getAddressSpace());
    });
    addConversion([&](LLVM::LLVMStructType structType) {
      MLIRContext *ctx = structType.getContext();

      SmallVector<DebugInfo::DIMemberType> elementTypes;
      for (auto [index, type] : llvm::enumerate(structType.getBody())) {
        // Build the member using a somewhat reasonable name given we don't have
        // a better one here.
        elementTypes.push_back(DebugInfo::DIMemberType::get(
            StringAttr::get(ctx, "field_" + Twine(index)),
            convertDebugType(type)));
      }

      StringRef name = structType.isIdentified() ? structType.getName() : "";
      return DebugInfo::DIStructType::get(StringAttr::get(ctx, name),
                                          elementTypes);
    });
    addConversion([&](LLVM::LLVMArrayType arrayType) {
      return DebugInfo::DIArrayType::get(
          convertDebugType(arrayType.getElementType()),
          arrayType.getNumElements());
    });
    addConversion([&](VectorType vecType) {
      return DebugInfo::DIVectorType::get(
          convertDebugType(vecType.getElementType()), vecType.getNumElements());
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
#define GEN_PASS_DEF_DEBUGINFOTOLLVM
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h.inc"
} // namespace M::DebugInfo

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct DebugInfoToLLVMPass
    : public impl::DebugInfoToLLVMBase<DebugInfoToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

/// Returns whether this debug value is unsupported by LLVM.
static bool isUnsupportedDebugValue(DebugInfo::ValueOp op) {
  /// LLVM does not yet support emitting DW_OP_LLVM_implicit_pointer to asm. If
  /// it is not yet optimized out by the time we emit to LLVM, it has to be
  /// removed.
  auto walkResult = op.getConversionExprAttr().walk(
      [](DebugInfo::DIRefOfExprAttr refof) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Remove debug values that are unsupported by LLVM.
static void removeUnsupportedDebugValues(Operation *op) {
  op->walk([](DebugInfo::ValueOp value) {
    if (isUnsupportedDebugValue(value))
      value->erase();
  });
}

void DebugInfoToLLVMPass::runOnOperation() {
  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<DebugInfoDialect>();
  target.addLegalOp<LLVM::DbgValueOp>();

  // Unknown operations are legal if they don't have debug info attached.
  target.markUnknownOpDynamicallyLegal([](Operation *op) -> bool {
    auto hasDIAttr = [](Location loc) -> bool {
      return !!loc->findInstanceOf<mlir::FusedLocWith<DebugInfo::DIAttr>>();
    };
    if (hasDIAttr(op->getLoc()))
      return false;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (hasDIAttr(arg.getLoc()))
            return false;
    return true;
  });

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  mlir::LLVMTypeConverter typeConverter(&getContext(), options);

  // Configure the metadata converter.
  DebugInfoToLLVMTypeConverter debugTypeConverter(typeConverter);
  MetadataConverter metadataConverter(debugTypeConverter,
                                      getTargetInfo(getOperation()));
  DIAttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](DIAttr attr) { return metadataConverter.convertAttr(attr); });

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateDebugInfoToLLVMPatterns(replacer, patterns);

  TargetAdapter targetAdapter = getTargetAdapter(getTargetInfo(getOperation()),
                                                 tradeoffPerfForVariableDI);
  targetAdapter.populateConversionPatterns(replacer, patterns);

  // Massage DebugInfo before conversion.
  targetAdapter.preTranslationAdapter(getOperation());

  // Legalize DebugValues.
  removeUnsupportedDebugValues(getOperation());

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();

  // Massage DebugInfo after conversion.
  targetAdapter.postTranslationAdapter(getOperation());
}
