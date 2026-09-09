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

#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <optional>
#include <utility>

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoTypeConverter
//===----------------------------------------------------------------------===//

/// Return the size and alignment to use for an integer or float type with the
/// given width.
static std::pair<uint64_t, uint32_t> getIntFloatDebugSizeAlign(unsigned width) {
  // Infer a reasonable default alignment from the type width.
  // TODO: We should really drive this through target info abstractions when
  // available.
  uint32_t align = llvm::PowerOf2Ceil(llvm::divideCeil(width, CHAR_BIT));
  align = std::min(align, 8u) * CHAR_BIT;

  return std::make_pair(llvm::alignTo(width, align), align);
}

template <bool isCyclic>
DebugInfoTypeConverterBase<isCyclic>::DebugInfoTypeConverterBase() {
  // Fallback that handle unresolved types.
  replacer.addReplacement(
      [&](Type type) -> std::optional<std::pair<Type, WalkResult>> {
        if (isa<DIType>(type))
          return std::nullopt;
        return std::pair<Type, WalkResult>(DIUnresolvedMLIRType::get(type),
                                           WalkResult::skip());
      });

  // Add conversions for known builtin types.
  replacer.addReplacement([&](IntegerType intType) {
    MLIRContext *ctx = intType.getContext();
    unsigned width = intType.getWidth();
    auto [size, align] = getIntFloatDebugSizeAlign(width);
    if (intType.isUnsigned())
      return DIBasicUIntType::get(ctx, "ui" + Twine(width), size, align);
    if (intType.isSigned())
      return DIBasicSIntType::get(ctx, "si" + Twine(width), size, align);
    // Treat signless integers as unsigned.
    return DIBasicUIntType::get(ctx, "i" + Twine(width), size, align);
  });
  replacer.addReplacement([&](FloatType floatTy) {
    auto [size, align] = getIntFloatDebugSizeAlign(floatTy.getWidth());
    return DIBasicFloatType::get(floatTy.getContext(),
                                 mlir::debugString(floatTy), size, align);
  });
  replacer.addReplacement([&](VectorType vecType) {
    return DIVectorType::get(convertDebugType(vecType.getElementType()),
                             vecType.getNumElements());
  });

  // Try to finalize unresolved types.
  replacer.addReplacement(
      [&](DIUnresolvedMLIRType type) -> std::pair<Type, WalkResult> {
        auto result = replacer.replace(type.getType());
        return {result ? result : type, WalkResult::skip()};
      });
}

template <bool isCyclic>
DIType DebugInfoTypeConverterBase<isCyclic>::convertDebugType(Type type) {
  if (!type)
    return {};
  return dyn_cast_or_null<DIType>(replacer.replace(type));
}

template <bool isCyclic>
Attribute
DebugInfoTypeConverterBase<isCyclic>::convertDebugTypesIn(Attribute attr) {
  auto result = replacer.replace(attr);
  if (auto expr = dyn_cast<DIAggregatesIntoExprAttr>(result))
    assert((isa<DIStructType, DIArrayType>(expr.getType())) &&
           "aggregate DI expr must resolve to a struct or array type");
  return result;
}

template <bool isCyclic>
void DebugInfoTypeConverterBase<isCyclic>::addUnresolvedConverter(
    TypeConverter &converter) {
  replacer.addReplacement(
      [&](Type type) -> std::optional<std::pair<Type, WalkResult>> {
        if (isa<DIType>(type))
          return std::nullopt;

        // Update the type using the provided converter.
        Type result = converter.convertType(type);
        if (!result || result == type)
          return std::nullopt;

        // If we succeeded, generate debug info for the new type.
        return std::pair<Type, WalkResult>(replacer.replace(result),
                                           WalkResult::skip());
      });
}

template <bool isCyclic>
void DebugInfoTypeConverterBase<isCyclic>::applyRecursively(Operation *op) {
  DIAttrTypeReplacer opReplacer;
  opReplacer.addReplacement(
      [&](DIType type) { return replacer.replace(type); });
  opReplacer.addReplacement(
      [&](DIExprAttr attr) { return replacer.replace(attr); });
  opReplacer.recursivelyReplaceElementsIn(op);
}

template class DebugInfo::DebugInfoTypeConverterBase<true>;
template class DebugInfo::DebugInfoTypeConverterBase<false>;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
class ConvertDebugKill : public mlir::OpConversionPattern<KillOp> {
public:
  ConvertDebugKill(TypeConverter &tc, DebugInfoTypeConverter &ditc,
                   MLIRContext *ctx)
      : OpConversionPattern(tc, ctx), ditc(ditc) {}

  LogicalResult
  matchAndRewrite(KillOp op, KillOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DIType diType = ditc.convertDebugType(op.getValueInfo().getType());
    if (!diType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert debuginfo type");
    }
    rewriter.modifyOpInPlace(op, [&] {
      DILocalVariableAttr info = op.getValueInfo();
      op.setValueInfoAttr(DILocalVariableAttr::get(
          info.getScope(), info.getName(), info.getFile(), info.getLine(),
          info.getArg(), info.getAlignInBits(), diType, info.getFlags()));
    });
    return success();
  }

private:
  /// The converter for the local variable type.
  DebugInfoTypeConverter &ditc;
};

class ConvertDebugValue : public mlir::OpConversionPattern<ValueOp> {
public:
  ConvertDebugValue(TypeConverter &tc, DebugInfoTypeConverter &ditc,
                    MLIRContext *ctx)
      : OpConversionPattern(tc, ctx), ditc(ditc) {}

  LogicalResult
  matchAndRewrite(ValueOp op, ValueOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DIType diType = ditc.convertDebugType(op.getValueInfo().getType());
    if (!diType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert debuginfo type");
    }
    DIExprAttr diExpr =
        cast<DIExprAttr>(ditc.convertDebugTypesIn(op.getConversionExprAttr()));
    if (!diExpr) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert debug value conversion expr");
    }
    rewriter.modifyOpInPlace(op, [&] {
      op.setOperand(adaptor.getValue());
      DILocalVariableAttr info = op.getValueInfo();
      op.setValueInfoAttr(DILocalVariableAttr::get(
          info.getScope(), info.getName(), info.getFile(), info.getLine(),
          info.getArg(), info.getAlignInBits(), diType, info.getFlags()));
      op.setConversionExprAttr(diExpr);
    });
    return success();
  }

private:
  /// The converter for the local variable type.
  DebugInfoTypeConverter &ditc;
};
} // namespace

void DebugInfo::populateTypeConversionPatterns(
    RewritePatternSet &patterns, DebugInfoTypeConverter &diConverter,
    TypeConverter &converter) {
  patterns.add<ConvertDebugKill>(converter, diConverter, patterns.getContext());
  patterns.add<ConvertDebugValue>(converter, diConverter,
                                  patterns.getContext());
}
