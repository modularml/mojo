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

#ifndef KGEN_LLVM_LOWERING_UTILS_H
#define KGEN_LLVM_LOWERING_UTILS_H

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace mlir::LLVM {
class CallOp;
class LLVMFuncOp;
} // namespace mlir::LLVM

namespace M::KGEN {
class KGENDType;
class GeneratorType;
class ParamType;
class PointerType;
class FuncType;
class NoneType;
class StringType;
class StructType;
class StructInstanceType;
class TypeParamAttr;
class TypeInstanceRefAttr;
class TypeValueType;
class SIMDType;

namespace POP {
class ArrayType;
class UnionType;
} // namespace POP

namespace CO {
class CoroutineType;
} // namespace CO

//===----------------------------------------------------------------------===//
// LLVMDataLayout
//===----------------------------------------------------------------------===//

/// This class is a helper to compute size and alignment of LLVM-compatible MLIR
/// types using a data layout specification.
class LLVMDataLayout {
public:
  explicit LLVMDataLayout(TargetInfoAttr target) : target(target) {}

  /// Get the size of the LLVM type in bits.
  int64_t getTypeSizeInBits(Type type) const;
  /// Get the maximum number of bytes that can be overwritten by storing the
  /// type. This is the type size in bits rounded up to the nearest byte.
  int64_t getTypeStoreSize(Type type) const {
    return llvm::divideCeil(getTypeSizeInBits(type), CHAR_BIT);
  }
  /// Get the alloc size of the type. This is the size of the type plus the
  /// required alignment padding.
  int64_t getTypeAllocSize(Type type) const {
    return llvm::alignTo(getTypeStoreSize(type), getTypeABIAlign(type));
  }
  /// Get the ABI alignment of the LLVM type and the element type that imposes
  /// the alignment requirement, it is typically the element type with maximum
  /// bitwidth.
  std::pair<int64_t, Type> getTypeABIAlignAndType(Type type) const;
  /// Get the ABI alignment of the LLVM type.
  int64_t getTypeABIAlign(Type type) const {
    return getTypeABIAlignAndType(type).first;
  }

  /// Get the target info.
  TargetInfoAttr getTarget() const { return target; }

private:
  /// The target info with the data layout to use.
  TargetInfoAttr target;
};

//===----------------------------------------------------------------------===//
// POPToLLVMTypeConverter
//===----------------------------------------------------------------------===//

/// Get the MLIR type for a data type.
std::optional<Type> getMLIRTypeForDType(MLIRContext *ctx, KGENDType dtype,
                                        size_t indexBitwidth);

/// This type converter maps fully-specified pop dialect parametric types and
/// built-in MLIR types to LLVM types.
struct POPToLLVMTypeConverter : public mlir::LLVMTypeConverter,
                                public LLVMDataLayout {
  POPToLLVMTypeConverter(TargetInfoAttr target);

  /// Get the LLVM field index for a logical KGEN struct field.
  /// When struct types are converted to LLVM, padding fields may be inserted
  /// to satisfy field alignment requirements (e.g., from @align decorators).
  /// This method returns the remapped index that accounts for these padding
  /// fields. Returns the logical index if no padding was added.
  int64_t getRemappedFieldIndex(StructType structType,
                                int64_t logicalIndex) const;

private:
  /// Cache mapping (StructType, logicalIndex) -> llvmIndex.
  /// Populated during struct type conversion when padding is inserted.
  mutable DenseMap<std::pair<Type, int64_t>, int64_t> structFieldIndexCache;
};

//===----------------------------------------------------------------------===//
// LLVMBuilder
//===----------------------------------------------------------------------===//

/// Create an `LLVM::CallOp`. Fast-math flags are governed by the `-fp-mode`
/// pass on the POP ops, so none are attached here.
template <typename... Args>
auto createLLVMCall(OpBuilder &b, Location loc, Args &&...args) {
  return mlir::LLVM::CallOp::create(b, loc, std::forward<Args>(args)...);
}

/// Attach `target-cpu` and `target-features` to the LLVM function attributes,
/// even if null. These attributes are attached to `LLVMFuncOp` and passed on
/// to LLVM IR function attributes.
ArrayAttr attachTargetPassthroughAttrs(OpBuilder &b, TargetInfoAttr target,
                                       ArrayAttr passthrough);

/// Create an `LLVMFuncOp` with the target info attributes.
template <typename... Args>
auto createLLVMFunc(OpBuilder &b, TargetInfoAttr target, Location loc,
                    Args &&...args) {
  auto func =
      mlir::LLVM::LLVMFuncOp::create(b, loc, std::forward<Args>(args)...);
  func.setPassthroughAttr(
      attachTargetPassthroughAttrs(b, target, func.getPassthroughAttr()));
  return func;
}

/// This class is a builder, type converter, and data layout bundled together.
struct LLVMBuilder : public ImplicitLocOpBuilder,
                     public POPToLLVMTypeConverter {
  LLVMBuilder(ImplicitLocOpBuilder &b, TargetInfoAttr target)
      : ImplicitLocOpBuilder(b), POPToLLVMTypeConverter(target) {}

  using ImplicitLocOpBuilder::getContext;
  using POPToLLVMTypeConverter::getIndexType;

  /// Create an `LLVMFuncOp` with the target info attributes.
  template <typename... Args>
  auto createFunc(Args &&...args) {
    return createLLVMFunc(*this, getTarget(), getLoc(),
                          std::forward<Args>(args)...);
  }

  /// Create an `unrealized_conversion_cast` operation.
  Value createConversion(Type type, Value src) {
    if (src.getType() == type)
      return src;
    return create<mlir::UnrealizedConversionCastOp>(type, src).getResult(0);
  }
  /// Lower a value's type and cast it if necessary.
  Value createConversion(Value src) {
    if (Type type = convertType(src.getType()))
      return createConversion(type, src);
    return {};
  }

  /// Get the pointer width in bytes.
  size_t getPointerByteWidth() {
    return llvm::divideCeil(getIndexTypeBitwidth(), CHAR_BIT);
  }
};

//===----------------------------------------------------------------------===//
// VariantHelper
//===----------------------------------------------------------------------===//

/// A helper for creating variants and extracting from them.
class VariantHelper {
public:
  VariantHelper(OpBuilder &b, Location loc, const LLVMDataLayout &dl)
      : b(loc, b), dl(dl) {}

  /// Generate the code required to materialize the provided value as a union
  /// of the given LLVM type.
  Value materializeLLVMUnion(mlir::LLVM::LLVMStructType type, Value value);

  /// Walk a simple or aggregate LLVM type and generate the code to insert its
  /// elements into a variant's content type. This tightly packs the element
  /// types within the content type. The first argument is an iterator to the
  /// current content element values. It is initialized with zeroes. The second
  /// is an iterator to the content element types.
  void walkAndCreateVariant(MutableArrayRef<Value>::iterator &valueIt,
                            unsigned &storageOffset, unsigned &offset,
                            Value value);

private:
  /// The builder to use.
  ImplicitLocOpBuilder b;
  /// The data layout to use.
  LLVMDataLayout dl;
};

//===----------------------------------------------------------------------===//
// Interpreter Memory Conversion
//===----------------------------------------------------------------------===//

/// This is a utility class for deduplicating memory instantiations from the
/// interpreter.
class InterpreterMemoryConverter {
public:
  /// Each materialized blob will have a corresponding SSA value representing
  /// the pointer to the beginning of the blob or an LLVM global for
  /// `const_global` blobs.
  using MaterializedBlob = PointerUnion<Operation *, Value>;
  using MaterializedBlobs = DenseMap<size_t, MaterializedBlob>;

  /// Create a converter instance. A single instance is held for an entire
  /// module to ensure globals are deduplicated.
  InterpreterMemoryConverter(SymbolTable &symtab, POPToLLVMTypeConverter &tc)
      : symtab(symtab), tc(tc) {}

  /// A conversion scope represents the range of IR in which identity is
  /// uniquely bestowed upon memory space attributes with the same value.
  /// Outside a scope, the same memory space attribute, where equality is
  /// determined by the contents, resolve to different actual addresses.
  ///
  /// FIXME: Once the parser has a correct model for lifetimes and identity in
  /// parameter expressions, the scope struct can be removed and materialization
  /// can be globally-scoped again.
  class MaterializationScope {
  public:
    /// Convert a single memory reference.
    ErrorOr<Value> convertMemRef(ImplicitLocOpBuilder &b, MemRefAttr ref);

    /// Get the parent converter.
    InterpreterMemoryConverter &getParent() { return imc; }

  private:
    explicit MaterializationScope(InterpreterMemoryConverter &imc) : imc(imc) {}

    /// Ensure the blobs within the memory space have been materialized and
    /// then return them.
    ErrorOr<MaterializedBlob &> getOrMaterialize(ImplicitLocOpBuilder &b,
                                                 MemorySpaceAttr space,
                                                 size_t refIndex);
    /// Get a pointer into the blob at the given offset.
    static Value getBlobPointer(ImplicitLocOpBuilder &b, Type ptrType,
                                MaterializedBlob &value, int64_t index,
                                int64_t offset);

    /// The interpreter memory converter.
    InterpreterMemoryConverter &imc;

    /// Lazily materialized memory spaces.
    DenseMap<MemorySpaceAttr, MaterializedBlobs> blobs;

    friend class InterpreterMemoryConverter;
  };

  friend class MaterializationScope;

  /// Create a new identity scope for converting memory values.
  MaterializationScope createScope() { return MaterializationScope(*this); }

  /// Get or add a global for the handle. It must be a `const_global` region.
  Operation *getOrCreateGlobal(Location loc, MemoryBlobAttr blob);

private:
  /// The symbol table to use for globals.
  SymbolTable &symtab;
  /// The type converter to use.
  POPToLLVMTypeConverter &tc;

  /// Lazily materialized globals.
  DenseMap<MemoryBlobAttr, Operation *> globals;
};

//===----------------------------------------------------------------------===//
// Struct Conversion
//===----------------------------------------------------------------------===//

/// Generate the LLVM IR to materialize a struct of the given LLVM struct type,
/// and insert the given element values into the struct.
Value materializeLLVMStruct(ImplicitLocOpBuilder &b, Type structType,
                            ValueRange elements);

/// Replace a KGEN call with an LLVM call, handling unpacking call results if
/// necessary.
void replaceCallWithLLVMCall(mlir::RewriterBase &b, Operation *op,
                             mlir::LLVM::CallOp call);

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Generate the LLVM IR to materialize a constant of the given value. This is
/// used to convert attribute values in `kgen.param.constant`.
ErrorOr<Value> convertParameterToLLVM(
    ImplicitLocOpBuilder &b, const POPToLLVMTypeConverter &tc,
    InterpreterMemoryConverter *imc,
    InterpreterMemoryConverter::MaterializationScope *scope, TypedAttr attr);

//===----------------------------------------------------------------------===//
// DebugInfoTypeConverter
//===----------------------------------------------------------------------===//

/// A specialized debug info type converter for converting from POP types to
/// LLVM.
/// An optional SymbolTable can be provided to lower TypeInstanceRefAttrs.
class DebugInfoTypeConverter : public DebugInfo::DebugInfoTypeConverter {
public:
  DebugInfoTypeConverter(POPToLLVMTypeConverter &tc, TargetInfoAttr targetInfo,
                         SymbolTable &symtab);

  /// Returns the first error encountered during type conversion, if any.
  /// Converter lambdas have no way to signal failure directly, so callers
  /// should check this after conversion is complete.
  std::optional<std::string> getError() const { return error; }

private:
  POPToLLVMTypeConverter &tc;
  SymbolTable &symtab;
  TargetInfoAttr targetInfo;

  /// Holds the first error message encountered during conversion, if any.
  std::optional<std::string> error;

  /// Build the debug type for a struct-like type.
  DebugInfo::DIType buildDebugStructTypeFromTypeAttrs(ArrayRef<Type> attrs,
                                                      StringAttr name);
  /// Build the debug type for a function type.
  DebugInfo::DIType buildDebugSubroutineType(FunctionType type);
  /// Build a pointer type.
  DebugInfo::DIType buildPointerType(DebugInfo::DIType type);
  DebugInfo::DIType buildPointerType(DebugInfo::DIType type,
                                     std::optional<unsigned> addressSpace);

  /// Build fully resolved debug type from partially resolved ones.
  DebugInfo::DIType
  buildDebugType(DebugInfo::DITargetIndependentPointerType type);

  /// Build fully resolved debug type from kgen/pop types.
  DebugInfo::DIType buildDebugType(GeneratorType type);
  DebugInfo::DIType buildDebugType(IndexType type);
  DebugInfo::DIType buildDebugType(ParamType type);
  DebugInfo::DIType buildDebugType(StringType type);
  DebugInfo::DIType buildDebugType(FuncType type);
  DebugInfo::DIType buildDebugType(POP::UnionType type);
  DebugInfo::DIType buildDebugType(KGEN::NoneType type);
  DebugInfo::DIType buildDebugType(POP::ArrayType type);
  DebugInfo::DIType buildDebugType(CO::CoroutineType type);
  DebugInfo::DIType buildDebugType(PointerType type);
  DebugInfo::DIType buildDebugType(SIMDType type);
  DebugInfo::DIType buildDebugType(StructType type);
  DebugInfo::DIType buildDebugType(StructInstanceType type);
  DebugInfo::DIType buildDebugType(TypeValueType type);

  /// Build fully resolved debug type from kgen type values.
  DebugInfo::DIType buildDebugType(TypeParamAttr attr);
  DebugInfo::DIType buildDebugType(TypeInstanceRefAttr attr);
};

//===----------------------------------------------------------------------===//
// ConvertPOPToLLVMPattern
//===----------------------------------------------------------------------===//

/// This is a templated instance of the wrapper class to rewrite a specific op.
template <typename OpT>
struct ConvertPOPToLLVMPattern : public mlir::ConvertOpToLLVMPattern<OpT> {
  using mlir::ConvertOpToLLVMPattern<OpT>::ConvertOpToLLVMPattern;

  /// Get the type converter.
  const POPToLLVMTypeConverter *getTypeConverter() const {
    return static_cast<const POPToLLVMTypeConverter *>(
        mlir::ConvertOpToLLVMPattern<OpT>::getTypeConverter());
  }

  /// Convert a type. Return null if the type conversion failed.
  Type convertType(Type type) const {
    return getTypeConverter()->convertType(type);
  }

  /// Builds an LLVM inline-asm op (ATT dialect, no side effects).
  mlir::LLVM::InlineAsmOp
  createInlineAsm(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  llvm::StringRef asmStr, llvm::StringRef asmConstraints,
                  Type resultType, llvm::SmallVector<Value> operands) const {
    const auto asmDialectAttr = mlir::LLVM::AsmDialectAttr::get(
        rewriter.getContext(), mlir::LLVM::AsmDialect::AD_ATT);
    return mlir::LLVM::InlineAsmOp::create(
        rewriter, loc, resultType,
        /*operands=*/operands,
        /*asm_string=*/asmStr,
        /*constraints=*/asmConstraints, /*has_side_effects=*/false,
        /*is_align_stack=*/false, mlir::LLVM::TailCallKind::None,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/mlir::ArrayAttr());
  }

  /// Creates a signless integer constant sized like `intType`.
  template <typename intType>
  Value createConstant(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Location loc, uint64_t value) const {
    return mlir::LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerType(sizeof(intType) * 8), value);
  }

  /// Creates an f32 floating-point constant.
  Value createConstant(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Location loc, llvm::APFloat value) const {
    return mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getF32Type(),
                                          value);
  }

  /// Extracts element `index` of a vector `value`.
  Value extractElement(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Location loc, Type resType, Value value,
                       unsigned index) const {
    return mlir::LLVM::ExtractElementOp::create(
        rewriter, loc, resType, value,
        createConstant<uint32_t>(rewriter, loc, index));
  }
};

//===----------------------------------------------------------------------===//
// OneToOneFloatOrIntConversion
//===----------------------------------------------------------------------===//

/// Compile-time detection of an op accessor `getFastmathFlags()`.
template <typename, typename = void>
struct has_getFastmathFlags : std::false_type {};

template <typename T>
struct has_getFastmathFlags<
    T, std::void_t<decltype(std::declval<T>().getFastmathFlags())>>
    : std::true_type {};

template <typename OpT>
static inline mlir::LLVM::FastmathFlags fastmathFlagsOrDefault(OpT &op) {
  if constexpr (has_getFastmathFlags<OpT>::value)
    return static_cast<mlir::LLVM::FastmathFlags>(op.getFastmathFlags());
  else
    return mlir::LLVM::FastmathFlags::none;
}

/// This patterns converts a scalar POP dialect operation to either an integer
/// or floating point LLVM operation one-to-one.
template <typename Op, typename FloatOp, typename SIntOp,
          typename UIntOp = SIntOp>
struct OneToOneFloatOrIntConversion : public ConvertPOPToLLVMPattern<Op> {
  using ConvertPOPToLLVMPattern<Op>::ConvertPOPToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    Type type = this->convertType(op.getType());

    if (dtype.isBool() || dtype.isInt() || dtype.isIndex() ||
        dtype.isUIndex()) {
      if (std::is_same_v<SIntOp, UIntOp> || dtype.isSInt() || dtype.isIndex()) {
        if constexpr (std::is_same_v<SIntOp, mlir::LLVM::SDivOp>)
          rewriter.replaceOpWithNewOp<SIntOp>(
              op, type, adaptor.getLhs(), adaptor.getRhs(), op.getIsExact());
        else
          rewriter.replaceOpWithNewOp<SIntOp>(op, type, adaptor.getLhs(),
                                              adaptor.getRhs());
      } else {
        if constexpr (std::is_same_v<UIntOp, mlir::LLVM::UDivOp>)
          rewriter.replaceOpWithNewOp<UIntOp>(
              op, type, adaptor.getLhs(), adaptor.getRhs(), op.getIsExact());
        else
          rewriter.replaceOpWithNewOp<UIntOp>(op, type, adaptor.getLhs(),
                                              adaptor.getRhs());
      }
    } else {
      // Take flags from a `getFastmathFlags()` accessor if present, else none.
      mlir::LLVM::FastmathFlags fastmathFlags = fastmathFlagsOrDefault(op);
      rewriter.replaceOpWithNewOp<FloatOp>(op, type, adaptor.getLhs(),
                                           adaptor.getRhs(), fastmathFlags);
    }

    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertSymbolOpToLLVM
//===----------------------------------------------------------------------===//

/// This pattern is used to rewrite symbol operations while keeping the symbol
/// table up-to-date.
template <typename OpT>
class ConvertSymbolOpToLLVM : public ConvertPOPToLLVMPattern<OpT> {
public:
  ConvertSymbolOpToLLVM(mlir::LLVMTypeConverter &typeConverter,
                        SymbolTable &symtab)
      : ConvertPOPToLLVMPattern<OpT>(typeConverter), symtab(symtab) {}

protected:
  /// The symbol table.
  SymbolTable &symtab;
};

/// Return true if fpType is an fp8 type.
bool isFP8(Type fpType);

/// Return true if fpType is an fp6 type.
bool isFP6(Type fpType);

/// Return true if fpType is an fp4 type.
bool isFP4(Type fpType);

/// If the type is a floating-point type lowered to an integer, return the
/// bitwidth of the integer width. Else return std::nullopt.
std::optional<int> isFPTyLoweredAsInt(Type fpType);

/// Parse a work-group-size metadata attribute into a `(min, max)` pair. Accepts
/// a `pop.array` of one or two elements or a scalar integer; a lone value is
/// treated as the max with min defaulting to 1. Used by GPU target lowerings to
/// translate `@__llvm_metadata` work-group-size annotations.
ErrorOr<std::pair<int64_t, int64_t>> getWorkGroupSizeRange(Attribute value);

/// Recursively squash pairs of unrealized_conversion_cast ops that cancel out.
/// For example, cast<A→B>(cast<B→A>(v)) reduces to v.
/// Used to clean up residual casts before generating calls.
mlir::Value squashPointlessCasts(mlir::Value v);

/// Return the byte alignment for a KGEN pointer type. Uses the explicit
/// alignment attribute when present, otherwise falls back to the ABI alignment
/// of the pointee type derived from the type converter.
inline unsigned getAlignment(const POPToLLVMTypeConverter *tc,
                             PointerType ptrType,
                             mlir::TypedAttr alignmentAttr = {}) {
  if (alignmentAttr)
    return mlir::cast<mlir::IntegerAttr>(alignmentAttr).getInt();
  return tc->getTypeABIAlign(tc->convertType(ptrType.getElementType()));
}

} // namespace M::KGEN

#endif // KGEN_LLVM_LOWERING_UTILS_H
