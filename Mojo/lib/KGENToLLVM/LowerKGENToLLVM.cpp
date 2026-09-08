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

#include "CABICallHelpers.h"
#include "LLVMLoweringUtils.h"
#include "LowerKGENToLLVMRewriteCABIFns.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/Threading.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Target/TargetLowering.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

/// Get the LLVM linkage kind for an export kind.
static LLVM::Linkage getLinkageKind(ExportKind exportKind) {
  switch (exportKind) {
  case ExportKind::NotExported:
    return LLVM::Linkage::Internal;
  case ExportKind::Exported:
    return LLVM::Linkage::External;
  }
  llvm_unreachable("invalid export kind");
}

namespace {

template <typename T>
static Attribute arrayAttrToDenseArrayAttr(Builder builder,
                                           POP::ArrayAttr array) {
  SmallVector<T> values =
      llvm::map_to_vector(array.getValues(), [](Attribute attr) -> T {
        if (auto integerAttr = ::dyn_cast<IntegerAttr>(attr))
          return static_cast<T>(integerAttr.getInt());
        return static_cast<T>(::cast<KGEN::SIMDAttr>(attr)
                                  .getValues()
                                  .front()
                                  .getIntVal()
                                  .getSExtValue());
      });
  if constexpr (std::is_same_v<T, int8_t>)
    return builder.getDenseI8ArrayAttr(values);
  else if constexpr (std::is_same_v<T, int16_t>)
    return builder.getDenseI16ArrayAttr(values);
  else if constexpr (std::is_same_v<T, int32_t>)
    return builder.getDenseI32ArrayAttr(values);
  else
    return builder.getDenseI64ArrayAttr(values);
}

static ErrorOrSuccess addArrayAttrToDict(Builder builder, NamedAttrList &attrs,
                                         StringRef name, POP::ArrayAttr array,
                                         Type elementType) {
  if (auto type = dyn_cast<SIMDType>(elementType)) {
    if (type.getResolvedSize().value_or(-1) != 1)
      return Error("ArrayAttr elements must be a scalar");

    std::optional<KGENDType> dtype = type.getResolvedDType();

    if (!dtype)
      return Error("unable to resolve the dtype for the SIMD value");

    if (!dtype->isInt())
      return Error("ArrayAttr must be an integral dtype");

    return addArrayAttrToDict(
        builder, attrs, name, array,
        getEquivalentIntegerType(builder.getContext(), *dtype));
  }

  if (auto type = dyn_cast<IntegerType>(elementType);
      type && llvm::is_contained({8, 16, 32, 64}, type.getWidth())) {
    if (type.getWidth() == 8)
      attrs.append(name, arrayAttrToDenseArrayAttr<int8_t>(builder, array));
    else if (type.getWidth() == 16)
      attrs.append(name, arrayAttrToDenseArrayAttr<int16_t>(builder, array));
    else if (type.getWidth() == 32)
      attrs.append(name, arrayAttrToDenseArrayAttr<int32_t>(builder, array));
    else
      attrs.append(name, arrayAttrToDenseArrayAttr<int64_t>(builder, array));
    return success();
  }

  return Error("non-integral dtypes are not supported");
}

//===----------------------------------------------------------------------===//
// ConvertKGENFunc
//===----------------------------------------------------------------------===//

namespace {
/// Cached attribute identifiers.
struct AttributeIdentifiers {
  StringAttr noalias, noundef, nonnull;
};
} // namespace

/// Return true if the function's borrowed pointer arguments must be passed by
/// value, as determined by the target lowering (e.g. exported kernels on
/// targets that require it).
static bool functionRequiresByVal(LLVM::LLVMFuncOp func,
                                  TargetInfoAttr target) {
  if (func.getLinkage() != LLVM::Linkage::External)
    return false;

  ErrorOr<const TargetLowering *> loweringOr =
      TargetLoweringRegistry::get().lookup(target.getTriple());
  return !loweringOr.isError() && (*loweringOr)->isExportedKernel(func);
}

/// Map `llvm.` attribute passed via `@__llvm_metadata` decorator to either
/// passthrough or use special LLVM Dialect's attribute.
static ErrorOrSuccess
mapToTypedLLVMFuncAttr(LLVM::LLVMFuncOp func, NamedAttrList &attrs,
                       SmallVectorImpl<Attribute> &passthrough, Builder &b,
                       StringRef suffix, Attribute value) {
  MLIRContext *ctx = func.getContext();

  // Attributes that user should not be allowed to override.
  // `dso_local` is derived from linkage by LLVMFuncOp's builder; the
  // `target_*` / `tune_*` knobs come from the codegen target.
  static const llvm::StringSet<> kDisallowedAttrs = {
      "dso_local",
      "target_cpu",
      "target_features",
      "tune_cpu",
  };
  if (kDisallowedAttrs.contains(suffix)) {
    return Error(Twine("'llvm.") + suffix +
                 "' is not allowed to be set via @__llvm_metadata");
  }

  // Do not allow following attributes without concrete need.
  static const llvm::StringSet<> kUnsupportedAttrs = {
      "alloc_family",
      "alloc_variant_zeroed",
      "dontcall_error",
      "dontcall_warn",
      "indirect_tls_seg_refs",
      "no_jump_tables",
      "patchable_function",
      "patchable_function_entry",
      "patchable_function_entry_section",
      "patchable_function_prefix",
      "vector_function_abi_variant",
  };
  if (kUnsupportedAttrs.contains(suffix)) {
    return Error(Twine("'llvm.") + suffix +
                 "' is temporarily not supported via @__llvm_metadata");
  }

  static llvm::StringMap<llvm::StringLiteral> allowedAttrsMap = {
      {"denormal_fp_math", "denormal-fp-math"},
      {"denormal_fp_math_f32", "denormal-fp-math-f32"},
      {"fp_contract", "fp-contract"},
      {"unsafe_fp_math", "unsafe-fp-math"},
      {"no_infs_fp_math", "no-infs-fp-math"},
      {"no_nans_fp_math", "no-nans-fp-math"},
      {"approx_func_fp_math", "approx-func-fp-math"},
      {"no_signed_zeros_fp_math", "no-signed-zeros-fp-math"},
      {"no_inline_line_tables", "no-inline-line-tables"},
      {"probe_stack", "probe-stack"},
      {"stack_probe_size", "stack-probe-size"},
      {"no_stack_arg_probe", "no-stack-arg-probe"},
      {"warn_stack_size", "warn-stack-size"},
  };

  if (allowedAttrsMap.contains(suffix)) {
    auto addPassthroughAttr = [&](StringRef key, StringRef val) {
      passthrough.push_back(
          b.getArrayAttr({b.getStringAttr(key), b.getStringAttr(val)}));
    };
    llvm::StringLiteral llvmIRName = allowedAttrsMap.at(suffix);
    if (auto str = dyn_cast<StringAttr>(value)) {
      addPassthroughAttr(llvmIRName, str.getValue());
      return success();
    }
    if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
      addPassthroughAttr(llvmIRName, boolAttr.getValue() ? "true" : "false");
      return success();
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      // LLVM IR encodes integer-valued function attributes as decimal strings.
      // Honor the IntegerType's explicit signedness; signless ints (the MLIR
      // default) print as unsigned so e.g. `4096 : i64` becomes "4096" rather
      // than potentially negative if interpreted as signed.
      if (intAttr.getType().isInteger(1)) {
        addPassthroughAttr(
            llvmIRName, intAttr.getValue().getBoolValue() ? "true" : "false");
      } else {
        llvm::SmallString<32> buf;
        intAttr.getValue().toString(
            buf, /*Radix=*/10,
            /*Signed=*/intAttr.getType().isSignedInteger());
        addPassthroughAttr(llvmIRName, buf);
      }
      return success();
    }
    return Error(Twine("'llvm.") + suffix +
                 "' expects a string, integer, or boolean value");
  }

  auto setOnFunc = [&](Attribute v) { attrs.set(suffix, v); };

  auto requireUnit = [&]() -> ErrorOrSuccess {
    if (!isa<UnitAttr>(value))
      return Error(Twine("'llvm.") + suffix + "' expects no value (unit flag)");
    setOnFunc(b.getUnitAttr());
    return success();
  };

  auto requireString = [&]() -> ErrorOrSuccess {
    auto str = dyn_cast<StringAttr>(value);
    if (!str)
      return Error(Twine("'llvm.") + suffix + "' expects a string value");
    setOnFunc(b.getStringAttr(str.getValue()));
    return success();
  };

  auto requireInt = [&](unsigned bitWidth) -> ErrorOrSuccess {
    auto intAttr = dyn_cast<IntegerAttr>(value);
    if (!intAttr)
      return Error(Twine("'llvm.") + suffix + "' expects an integer value");
    setOnFunc(IntegerAttr::get(IntegerType::get(ctx, bitWidth),
                               intAttr.getValue().getSExtValue()));
    return success();
  };

  auto requireDenseI32Array = [&]() -> ErrorOrSuccess {
    if (auto denseArr = dyn_cast<mlir::DenseI32ArrayAttr>(value)) {
      setOnFunc(denseArr);
      return success();
    }
    if (auto array = dyn_cast<POP::ArrayAttr>(value)) {
      Type elementType = array.getType().getElementType();
      if (auto simd = dyn_cast<SIMDType>(elementType))
        elementType = getEquivalentIntegerType(ctx, *simd.getResolvedDType());
      if (auto intTy = dyn_cast<IntegerType>(elementType);
          intTy && intTy.getWidth() == 32) {
        setOnFunc(arrayAttrToDenseArrayAttr<int32_t>(b, array));
        return success();
      }
    }
    return Error(
        Twine("'llvm.") + suffix +
        "' expects a dense i32 array (e.g. StaticTuple[Int32, N](...))");
  };

  // Every UnitAttr / OptionalAttr<UnitAttr> declared on LLVMFuncOp in
  // mlir/Dialect/LLVMIR/LLVMOps.td. Keep in sync if upstream MLIR grows new
  // flag-style function attributes.
  static const llvm::StringSet<> kUnitAttrs = {
      "always_inline",
      "arm_in_za",
      "arm_inout_za",
      "arm_locally_streaming",
      "arm_new_za",
      "arm_out_za",
      "arm_preserves_za",
      "arm_streaming",
      "arm_streaming_compatible",
      "cold",
      "convergent",
      "hot",
      "inline_hint",
      "minsize",
      "no_caller_saved_registers",
      "no_inline",
      "no_unwind",
      "nocallback",
      "noduplicate",
      "noreturn",
      "optimize_none",
      "optsize",
      "returns_twice",
      "save_reg_params",
      "will_return",
  };
  if (kUnitAttrs.contains(suffix))
    return requireUnit();

  if (suffix == "section" || suffix == "garbageCollector")
    return requireString();

  if (suffix == "alignment")
    return requireInt(/*bitWidth=*/64);

  if (suffix == "function_entry_count") {
    auto intAttr = dyn_cast<IntegerAttr>(value);
    if (!intAttr)
      return Error(Twine("'llvm.") + suffix + "' expects an integer value");
    setOnFunc(LLVM::FunctionEntryCountAttr::get(
        ctx, intAttr.getValue().getZExtValue(), LLVM::ProfileCountType::Real,
        /*imports=*/{}));
    return success();
  }

  if (suffix == "intel_reqd_sub_group_size")
    return requireInt(/*bitWidth=*/32);

  if (suffix == "work_group_size_hint" || suffix == "reqd_work_group_size")
    return requireDenseI32Array();

  if (suffix == "frame_pointer") {
    auto str = dyn_cast<StringAttr>(value);
    if (!str)
      return Error("'llvm.frame_pointer' expects a string value: "
                   "\"none\", \"non-leaf\", \"all\", or \"reserved\"");
    std::optional<LLVM::framePointerKind::FramePointerKind> kind =
        LLVM::framePointerKind::symbolizeFramePointerKind(str.getValue());
    if (!kind)
      return Error(
          Twine("invalid 'llvm.frame_pointer' value '") + str.getValue() +
          "'; expected \"none\", \"non-leaf\", \"all\", or \"reserved\"");
    setOnFunc(LLVM::FramePointerKindAttr::get(ctx, *kind));
    return success();
  }

  if (suffix == "vscale_range") {
    auto setRange = [&](int64_t min, int64_t max) {
      auto i32 = IntegerType::get(ctx, 32);
      setOnFunc(LLVM::VScaleRangeAttr::get(ctx, IntegerAttr::get(i32, min),
                                           IntegerAttr::get(i32, max)));
    };
    if (auto array = dyn_cast<POP::ArrayAttr>(value)) {
      if (array.getValues().size() != 2)
        return Error(
            "'llvm.vscale_range' expects a 2-element array (min, max)");
      auto getInt = [](Attribute a) -> int64_t {
        if (auto i = dyn_cast<IntegerAttr>(a))
          return i.getInt();
        return cast<KGEN::SIMDAttr>(a)
            .getValues()
            .front()
            .getIntVal()
            .getSExtValue();
      };
      setRange(getInt(array.getValues()[0]), getInt(array.getValues()[1]));
      return success();
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      setRange(intAttr.getInt(), 0);
      return success();
    }
    return Error(
        "'llvm.vscale_range' expects an Int or a 2-element array of Int32");
  }

  // Unknown `llvm.<suffix>`: forward as a generic LLVM passthrough attribute
  // (flag for UnitAttr; key/value for IntegerAttr/StringAttr). LLVM itself
  // promotes well-known parametric names like `alignstack`, `memory`, etc.
  StringAttr name = b.getStringAttr(suffix);
  if (isa<UnitAttr>(value)) {
    passthrough.push_back(name);
    return success();
  }
  if (auto intVal = dyn_cast<IntegerAttr>(value)) {
    SmallVector<char> str;
    intVal.getValue().toString(str, /*Radix=*/10, /*Signed=*/true);
    passthrough.push_back(b.getArrayAttr(
        {name, b.getStringAttr(StringRef(str.data(), str.size()))}));
    return success();
  }
  if (auto str = dyn_cast<StringAttr>(value)) {
    passthrough.push_back(
        b.getArrayAttr({name, b.getStringAttr(str.getValue())}));
    return success();
  }
  return Error("unsupported LLVM passthrough attribute kind");
}

/// Convert LLVM metadata expressed in KGEN attributes to an LLVM dialect
/// compatible representation. Unsupported metadata values are rejected.
static LogicalResult convertLLVMMetadata(LLVM::LLVMFuncOp func, FuncType sig,
                                         DictionaryAttr metadata,
                                         ArrayAttr argMetadata,
                                         const AttributeIdentifiers &ids,
                                         const TypeConverter *tc,
                                         TargetInfoAttr target) {
  NamedAttrList attrs = func->getAttrDictionary();
  SmallVector<Attribute> passthrough =
      llvm::to_vector(func.getPassthroughAttr());
  Builder b(func.getContext());
  ErrorOr<const TargetLowering *> loweringOr =
      TargetLoweringRegistry::get().lookup(target.getTriple());
  const TargetLowering *lowering = loweringOr.isError() ? nullptr : *loweringOr;

  for (const NamedAttribute &attr : metadata) {
    // Treat `llvm.*` metadata attributes as passthrough function attributes.
    Attribute value = attr.getValue();
    Dialect *nameDialect = attr.getNameDialect();
    if (!nameDialect) {
      return mlir::emitError(func.getLoc(), "'")
             << attr.getName() << "' is not defined";
    }
    if (isa<LLVM::LLVMDialect>(nameDialect)) {
      StringRef suffix =
          attr.getName().strref().drop_front(StringRef("llvm.").size());
      if (ErrorOrSuccess err = mapToTypedLLVMFuncAttr(func, attrs, passthrough,
                                                      b, suffix, value);
          err.isError())
        return mlir::emitError(func.getLoc()) << err.takeError();
      continue;
    }

    // Target-dialect function metadata is translated by the target lowering;
    // the annotation's dialect and the active target always correspond.
    if (lowering) {
      bool handled = false;
      if (failed(lowering->mapFuncMetadata(func, attr, attrs, passthrough,
                                           handled)))
        return failure();
      if (handled)
        continue;
    }

    // For anything else, forward them as function attributes.
    if (isa<UnitAttr, IntegerAttr>(value)) {
      // HACK make kgen.offload.kernelid as passthrough attribute so that
      // it gets kept during llvm lowering.
      if (attr.getName() == "kgen.offload.kernelid") {
        passthrough.push_back(b.getArrayAttr(
            {attr.getName(), b.getStringAttr(std::to_string(
                                 cast<IntegerAttr>(value).getInt()))}));
      }
      // Propagate unit and integer attribute.
      attrs.append(attr.getName(), value);
    } else if (auto str = dyn_cast<StringAttr>(value)) {
      // Strip the type from string attributes.
      attrs.append(attr.getName(), b.getStringAttr(str.getValue()));
    } else if (auto array = dyn_cast<POP::ArrayAttr>(value)) {
      Type elementType = array.getType().getElementType();
      if (ErrorOrSuccess err =
              addArrayAttrToDict(b, attrs, attr.getName(), array, elementType);
          err.isError())
        return mlir::emitError(func.getLoc(), "unsupported array type: ")
               << array << " because " << err.takeError();
    } else if (auto array = dyn_cast<mlir::DenseI32ArrayAttr>(value)) {
      attrs.append(attr.getName(), array);
    } else {
      return mlir::emitError(func.getLoc(),
                             "unsupported LLVM metadata attribute kind: ")
             << value;
    }
  }

  // Some targets pass borrowed arguments by value for exported functions.
  bool needsByVal = functionRequiresByVal(func, target);

  // Attach the by-value argument attribute per the target's argument-passing
  // convention.
  auto addByValAttribute = [tc, lowering](NamedAttrList &attrs, Type type,
                                          LLVM::LLVMFuncOp func) {
    assert(lowering && "by-val kernel args require a target lowering");
    MLIRContext *ctx = func.getContext();
    StringAttr byValAttr =
        StringAttr::get(ctx, lowering->getKernelByValArgAttrName());
    attrs.set(byValAttr, TypeAttr::get(tc->convertType(
                             cast<PointerType>(type).getElementType())));
  };

  // Helper to apply nonnull attribute for non-null pointer types.
  auto applyNonNullAttributes = [&ids, &b](NamedAttrList &attrs, Type type) {
    if (auto ptrType = dyn_cast<PointerType>(type);
        ptrType && ptrType.getIsNonNull()) {
      attrs.set(ids.noundef, b.getUnitAttr());
      attrs.set(ids.nonnull, b.getUnitAttr());
    }
  };

  // For each argument and result, leverage signature information to generate
  // the corresponding LLVM argument and result attributes.
  SmallVector<Attribute> argAttrs;
  for (auto [i, conv, type] :
       llvm::enumerate(sig.getArgConventions(), sig.getArguments())) {
    NamedAttrList list;

    if (!argMetadata.empty()) {
      // `argMetadata` is legalized to either be empty or have same size as the
      // number of function arguments.
      for (const NamedAttribute &attr : cast<DictionaryAttr>(argMetadata[i])) {
        Attribute value = attr.getValue();
        Dialect *nameDialect = attr.getNameDialect();
        if (!nameDialect) {
          return mlir::emitError(
                     func.getLoc(),
                     "dialect not loaded for LLVM passthrough attribute: ")
                 << attr.getName() << '=' << value;
        }

        // The target lowering applies the kernel-argument annotations it
        // recognizes and drops the rest, so the same mojo code stays portable
        // across backends.
        // TODO: build a more general language mechanism to parameterize this
        // annotation by target at the mojo level.
        if (lowering)
          lowering->mapKernelArgMetadata(func, attr, list);
      }
    }

    applyNonNullAttributes(list, type);

    switch (conv) {
    case ArgConvention::OwnedMem:
    case ArgConvention::DeinitMem:
      list.set(ids.noalias, b.getUnitAttr());
      [[fallthrough]];
    case ArgConvention::ImmMem:
      if (needsByVal)
        addByValAttribute(list, type, func);
      list.set(ids.nonnull, b.getUnitAttr());
      list.set(ids.noundef, b.getUnitAttr());
      break;
    case ArgConvention::Mut:
    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
    case ArgConvention::MutRef:
      // The compiler enforces that each function can only have one mutable
      // reference to an object at a time. Thus, we know the pointers that back
      // mutable in-memory arguments are noalias.
      list.set(ids.noalias, b.getUnitAttr());
      [[fallthrough]];
    case ArgConvention::Ref:
      // We know the pointers that back in-memory arguments are nonnull.
      list.set(ids.nonnull, b.getUnitAttr());
      [[fallthrough]];

    case ArgConvention::ImmReg:
    case ArgConvention::OwnedReg:
      // The only thing we can say about values passed in-register is `noundef`,
      // which is equivalent to saying that they are known initialized. This
      // also applies to all the pointers passed for in-memory arguments.
      list.set(ids.noundef, b.getUnitAttr());
      break;
    }

    argAttrs.push_back(list.getDictionary(b.getContext()));
  }

  // Handle result attributes for pointer return types.
  // Only apply result attributes for single-result functions, as multiple
  // KGEN results are packed into a single LLVM struct, and LLVM expects
  // result attributes to match the number of LLVM-level results.
  SmallVector<Attribute> resAttrs;
  if (sig.getResults().size() == 1) {
    NamedAttrList list;
    applyNonNullAttributes(list, sig.getResults()[0]);
    resAttrs.push_back(list.getDictionary(b.getContext()));
  }

  // Update the attributes.
  attrs.set(func.getArgAttrsAttrName(), b.getArrayAttr(argAttrs));
  if (!resAttrs.empty())
    attrs.set(func.getResAttrsAttrName(), b.getArrayAttr(resAttrs));
  func->setAttrs(attrs.getDictionary(func.getContext()));
  func.setPassthroughAttr(b.getArrayAttr(passthrough));
  return success();
}

/// Convert inline level to an LLVM passthrough attribute.
/// compatible representation. Unsupported metadata values are rejected.
static void convertInlineLevel(LLVM::LLVMFuncOp func, InlineLevel inlineLevel) {
  if (inlineLevel == InlineLevel::Automatic)
    return;

  SmallVector<Attribute> passthrough =
      llvm::to_vector(func.getPassthroughAttr());
  Builder b(func.getContext());

  const char *attrName;
  switch (inlineLevel) {
  case InlineLevel::Always:
  case InlineLevel::AlwaysNoDebug:
  case InlineLevel::AlwaysBuiltin:
    attrName = "alwaysinline";
    break;
  case InlineLevel::Never:
    attrName = "noinline";
    break;
  default:
    llvm_unreachable("invalid InlineLevel enum");
  }
  passthrough.push_back(b.getStringAttr(attrName));
  func.setPassthroughAttr(b.getArrayAttr(passthrough));
}

/// Returns true if type is an empty !llvm.struct type, or an array of empty
/// types e.g !llvm.array<0 x ..any type..>, !llvm.array<N x empty_struct>
/// TODO: Consider querying size from DataLayout instead.
static bool isEmptyType(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case([](LLVM::LLVMArrayType arrayType) {
        if (arrayType.getNumElements() == 0)
          return true;
        return isEmptyType(arrayType.getElementType());
      })
      .Case([](LLVM::LLVMStructType structType) {
        bool emptyType = true;
        for (Type innerType : structType.getBody())
          emptyType &= isEmptyType(innerType);
        return emptyType;
      })
      .Default([](Type /* default */) { return false; });
}

/// Drops empty struct arguments from funcOp and replace usage with an undef
/// struct.
static void dropEmptyStructArguments(LLVM::LLVMFuncOp &func,
                                     ConversionPatternRewriter &rewriter) {
  SmallVector<unsigned> emptyArgIdx, nonEmptyArgIdx;
  SmallVector<Type> emptyArgType, nonEmptyArgTypes;
  for (auto [idx, argType] : enumerate(func.getArgumentTypes())) {
    if (isEmptyType(argType)) {
      emptyArgIdx.push_back(idx);
      emptyArgType.push_back(argType);
    } else {
      nonEmptyArgIdx.push_back(idx);
      nonEmptyArgTypes.push_back(argType);
    }
  }

  if (emptyArgIdx.empty())
    return;

  // If it has a body block erase empty struct function arguments and
  // replace their inner usage with undef empty struct types.
  if (!func.getBody().empty()) {
    Block *entryBlock = &func.getBody().front();
    rewriter.setInsertionPointToStart(entryBlock);
    TypeConverter::SignatureConversion sigConverter(func.getNumArguments());
    for (auto [idx, type] : zip(nonEmptyArgIdx, nonEmptyArgTypes))
      sigConverter.addInputs(idx, type);

    for (auto [idx, type] : zip(emptyArgIdx, emptyArgType)) {
      Value emptyStruct = LLVM::UndefOp::create(rewriter, func->getLoc(), type);
      sigConverter.remapInput(idx, emptyStruct);
    }
    rewriter.applySignatureConversion(&func.getBody().front(), sigConverter);
  }

  // Recreate the args attr
  auto argsAttr = func.getArgAttrsAttr();
  SmallVector<Attribute> newEntries;
  for (auto i : nonEmptyArgIdx)
    newEntries.push_back(argsAttr[i]);
  func.setArgAttrsAttr(ArrayAttr::get(func.getContext(), newEntries));

  // Update funcOp type.
  rewriter.modifyOpInPlace(func, [&]() {
    func.setType(LLVM::LLVMFunctionType::get(
        func.getFunctionType().getReturnType(), nonEmptyArgTypes));
  });
}

/// Attribute stamped onto llvm.func ops that were converted from kgen.func ops
/// with the abi("C") effect. Used by ConvertKGENCall to identify C-ABI
/// callees regardless of whether the callee was converted before or after the
/// call site.
static constexpr StringLiteral kCABIFuncAttr = "kgen.c_abi";

/// Returns true if the callee identified by `sym` has the abi("C") effect.
/// The callee may be a kgen.func (not yet converted) or an llvm.func already
/// stamped with kCABIFuncAttr by ConvertKGENFunc, depending on conversion
/// order.
static bool isCABICallee(Operation *callOp, FlatSymbolRefAttr sym) {
  auto module = callOp->getParentOfType<mlir::ModuleOp>();
  auto *calleeOp = mlir::SymbolTable::lookupSymbolIn(module, sym);
  if (auto kgenFunc = dyn_cast_or_null<FuncOp>(calleeOp))
    return kgenFunc.getFuncTypeGenerator().getBody().getFnEffects().isCABI();
  if (auto llvmFunc = dyn_cast_or_null<LLVM::LLVMFuncOp>(calleeOp))
    return llvmFunc->hasAttr(kCABIFuncAttr);
  return false;
}

class ConvertKGENFunc : public ConvertSymbolOpToLLVM<FuncOp> {
public:
  ConvertKGENFunc(mlir::LLVMTypeConverter &tc, SymbolTable &symtab,
                  const AttributeIdentifiers &ids)
      : ConvertSymbolOpToLLVM(tc, symtab), ids(ids) {}

  LogicalResult matchAndRewrite(FuncOp func, FuncOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    // Convert the func signature.
    TypeConverter::SignatureConversion result(func.getNumArguments());
    Type funcType = getTypeConverter()->convertFunctionSignature(
        func.getFunctionType(), /*isVariadic=*/false,
        getTypeConverter()->getOptions().useBarePtrCallConv, result);
    if (!funcType)
      return emitError(func.getLoc(), "failed to convert func signature");

    TargetInfoAttr target = getTypeConverter()->getTarget();

    // Mark all functions as internal for now - we'll clean this up later.
    LLVM::Linkage linkage = func.isExternal()
                                ? LLVM::Linkage::External
                                : getLinkageKind(func.getExportKind());
    auto funcOp = createLLVMFunc(b, target, func.getLoc(), func.getNameAttr(),
                                 funcType, linkage);
    if (func.isExported()) {
      funcOp.setDsoLocal(true);

      // Let the target lowering apply any target-specific marking to exported
      // functions so the backend can recognize them.
      if (ErrorOr<const TargetLowering *> loweringOr =
              TargetLoweringRegistry::get().lookup(target.getTriple());
          !loweringOr.isError())
        (*loweringOr)->markExportedKernel(funcOp);
    }
    if (failed(convertLLVMMetadata(
            funcOp, func.getFuncTypeGenerator().getBody(),
            func.getLLVMMetadataAttr(), func.getLLVMArgMetadata(), ids,
            typeConverter, target)))
      return failure();

    if (func.getCoroutineType()) {
      Type coroType =
          typeConverter->convertType(func.getCoroutineType().value());
      funcOp->setAttr(func.getCoroutineTypeAttrName(), TypeAttr::get(coroType));
    }

    if (func.isConvergent())
      funcOp.setConvergent(true);

    // Propagate InlineLevel as a passthrough LLVM attribute.
    convertInlineLevel(funcOp, func.getInlineLevel());

    // Mark abi("C") functions so ConvertKGENCall can identify them via
    // symbol lookup regardless of conversion order.
    if (func.getFuncTypeGenerator().getBody().getFnEffects().isCABI())
      funcOp->setAttr(kCABIFuncAttr, b.getUnitAttr());

    // And move the func's body into the new function.
    if (!func.isExternal()) {
      b.inlineRegionBefore(func.getBodyRegion(), funcOp.getBody(),
                           funcOp.end());
      (void)b.convertRegionTypes(&funcOp.getBody(), *getTypeConverter());
    }

    // Drop empty struct arguments.
    dropEmptyStructArguments(funcOp, b);

    // Remove the function.
    symtab.remove(func);
    Block::iterator insertPt(func->getNextNode());
    funcOp->remove();
    symtab.insert(funcOp, insertPt);
    b.eraseOp(func);
    return success();
  }

private:
  const AttributeIdentifiers &ids;
};

//===----------------------------------------------------------------------===//
// ConvertKGENCall
//===----------------------------------------------------------------------===//

/// Convert `kgen.call` to `llvm.call`, unpacking results if necessary.
/// For callees marked with the `abi("C")` effect, C ABI coercion is applied
/// via CABICallHelper (same mechanism as for abi("C") indirect calls).
struct ConvertKGENCall : public ConvertPOPToLLVMPattern<CallOp> {
  ConvertKGENCall(mlir::LLVMTypeConverter &tc) : ConvertPOPToLLVMPattern(tc) {}

  LogicalResult
  matchAndRewrite(CallOp op, CallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the result types.
    SmallVector<Type> types = llvm::to_vector(op.getResultTypes());
    if (!types.empty()) {
      types.assign({getTypeConverter()->packFunctionResults(types)});
      if (!types.back())
        return emitError(op.getLoc(), "failed to convert call result type");
    }

    auto flatSymbol = dyn_cast<FlatSymbolRefAttr>(op.getCalleeSymbol());
    if (!flatSymbol)
      return emitError(op.getLoc(),
                       "cannot lower call to nested symbol to LLVM");

    // Drop empty struct arguments.
    auto filteredOperands = to_vector(
        llvm::make_filter_range(adaptor.getOperands(), [](Value operand) {
          return !isEmptyType(operand.getType());
        }));

    // If the callee is a abi("C") function definition, apply C ABI coercion
    // at the call site so that struct args/returns are passed correctly.
    if (isCABICallee(op, flatSymbol)) {
      CABICallHelper cabi(getTypeConverter(), getContext(), op.getOperation());
      Location loc = op.getLoc();
      Type origRetTy = types.empty() ? Type{} : types.front();
      auto prep =
          cabi.prepareCall(mlir::ValueRange(filteredOperands).getTypes(),
                           filteredOperands, origRetTy, loc, rewriter);
      LLVM::CallOp llvmCall = LLVM::CallOp::create(
          rewriter, loc, prep.signature, flatSymbol, prep.callArgs);
      CABICallHelper::applySRetAttrIfNeeded(llvmCall, origRetTy, prep.usesSRet,
                                            rewriter);
      cabi.applyByvalAttrsToCall(llvmCall, prep.argClass,
                                 mlir::ValueRange(filteredOperands).getTypes(),
                                 prep.usesSRet, rewriter);
      // A tail call is only safe when the callee does not write through any
      // pointer that lives in the caller's stack frame.  The sret alloca
      // created by prepareCall lives in the caller's frame, so a tail call
      // would free that frame before the callee stores its return value,
      // producing a silent memory corruption (MOCO-3841).  Suppress the tail
      // marker whenever sret is in use; non-sret returns are register-passed
      // and remain eligible for tail-call optimization.
      if (!prep.usesSRet)
        applyTailKind(llvmCall, op.getTailKind());
      if (op.getNumResults() == 0) {
        rewriter.eraseOp(op);
        return success();
      }
      Value rawResult = prep.usesSRet ? Value{} : llvmCall.getResult();
      rewriter.replaceOp(op, cabi.extractReturn(prep.retClass, rawResult,
                                                prep.sretPtr, origRetTy, loc,
                                                rewriter));
      return success();
    }

    // Create the LLVM call operation (Mojo ABI).
    LLVM::CallOp llvmCall = createLLVMCall(rewriter, op.getLoc(), types,
                                           flatSymbol, filteredOperands);
    applyTailKind(llvmCall, op.getTailKind());

    replaceCallWithLLVMCall(rewriter, op, llvmCall);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENReturn
//===----------------------------------------------------------------------===//

/// Convert `kgen.return` to `llvm.return`, packing the results if necessary.
struct ConvertKGENReturn : public ConvertPOPToLLVMPattern<ReturnOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();

    // If the results don't need to be packed, create the LLVM return.
    if (op->getNumOperands() <= 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), operands);
      return success();
    }

    // Pack the function results in a struct.
    Type type = getTypeConverter()->packFunctionResults(op->getOperandTypes());
    if (!type)
      return emitError(op->getLoc(), "failed to convert return types");
    Value result = LLVM::UndefOp::create(rewriter, op->getLoc(), type);
    for (auto [index, operand] : llvm::enumerate(operands)) {
      result = LLVM::InsertValueOp::create(rewriter, op->getLoc(), result,
                                           operand, index);
    }

    // Create the LLVM return.
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENUnreachable
//===----------------------------------------------------------------------===//

/// Convert `kgen.unreachable` to `llvm.unreachable`.
struct ConvertKGENUnreachable : public ConvertPOPToLLVMPattern<UnreachableOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnreachableOp op, UnreachableOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the llvm.trap + llvm.unreachable ops.
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    LLVM::CallIntrinsicOp::create(
        rewriter, op.getLoc(), TypeRange{voidTy},
        rewriter.getStringAttr("llvm.trap"),
        /*args=*/ValueRange(),
        rewriter.getAttr<LLVM::FastmathFlagsAttr>(LLVM::FastmathFlags::none));
    rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENParamConstant
//===----------------------------------------------------------------------===//

/// Handle errors when lowering constants to LLVM.  Return failure if this is
/// unrecoverable.
static LogicalResult handleConstantLoweringError(
    ErrorOr<Value> &value, Operation *op, ConversionPatternRewriter &rewriter,
    bool &passFailed, const mlir::TypeConverter &typeConverter) {
  // If not an error, do nothing.
  if (!value.isError())
    return success();
  auto loc = op->getLoc();

  // MLIR is very lossy with respect to constants; FoldUtils massively kills
  // them, which is really bad for QoI.  To try to recover SOMETHING, walk
  // the use chain of the constant and grab the first thing with a relevant
  // location.
  if (isa<UnknownLoc>(loc)) {
    for (auto user : op->getResult(0).getUsers()) {
      if (!isa<UnknownLoc>(user->getLoc())) {
        loc = user->getLoc();
        break;
      }
    }
  }

  // Report a nice error to the user instead of a generic "failed to
  // legalize operation" error that the lowering machinery would produce for
  // us. To do this, we emit the error, but return success (so we don't get
  // the generic error)then signal a pass failure so that
  // the pass manager stops executing passes.
  mlir::emitError(loc, value.getError());
  passFailed = true;

  // FIXME: the pattern rewrite infra really sucks, we should move off it.
  auto type = typeConverter.convertType(op->getResult(0).getType());
  // Type lowering can fail with things like !kgen.intliteral which should never
  // have been a runtime type in the first place (MOCO-1628)
  if (!type)
    return failure();

  value = LLVM::UndefOp::create(rewriter, loc, type);
  return success();
}

class ConvertKGENParamConstant
    : public ConvertPOPToLLVMPattern<ParamConstantOp> {
public:
  ConvertKGENParamConstant(mlir::LLVMTypeConverter &tc,
                           InterpreterMemoryConverter &imc, bool &passFailed)
      : ConvertPOPToLLVMPattern(tc), imc(imc), passFailed(passFailed) {}

  LogicalResult
  matchAndRewrite(ParamConstantOp op, ParamConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    InterpreterMemoryConverter::MaterializationScope scope = imc.createScope();
    ErrorOr<Value> value = convertParameterToLLVM(b, *getTypeConverter(), &imc,
                                                  &scope, op.getValue());
    if (failed(handleConstantLoweringError(value, op, rewriter, passFailed,
                                           *getTypeConverter())))
      return failure();

    rewriter.replaceOp(op, value.get());
    return success();
  }

private:
  /// Convert for global memory references.
  InterpreterMemoryConverter &imc;
  bool &passFailed;
};

//===----------------------------------------------------------------------===//
// ConvertKGENParamMaterialize
//===----------------------------------------------------------------------===//

class ConvertKGENParamMaterialize
    : public ConvertPOPToLLVMPattern<ParamMaterializeOp> {
public:
  ConvertKGENParamMaterialize(mlir::LLVMTypeConverter &tc,
                              InterpreterMemoryConverter &imc, bool &passFailed)
      : ConvertPOPToLLVMPattern(tc), imc(imc), passFailed(passFailed) {}

  LogicalResult
  matchAndRewrite(ParamMaterializeOp op, ParamMaterializeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    InterpreterMemoryConverter::MaterializationScope scope = imc.createScope();
    ErrorOr<Value> value = convertParameterToLLVM(b, *getTypeConverter(), &imc,
                                                  &scope, op.getValue());
    if (failed(handleConstantLoweringError(value, op, rewriter, passFailed,
                                           *getTypeConverter())))
      return failure();

    rewriter.replaceOp(op, value.get());
    return success();
  }

private:
  /// Convert for interpreter memory references.
  InterpreterMemoryConverter &imc;
  bool &passFailed;
};

} // namespace

//===----------------------------------------------------------------------===//
// ConvertKGENSourceLoc
//===----------------------------------------------------------------------===//

struct ConvertKGENSourceLoc : ConvertPOPToLLVMPattern<SourceLocOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  // An op reaching here was not inlined enough to reach its requested caller
  // frame (`SourceLocOp::fold` resolves the rest). Inlining is done, so degrade
  // to the outermost caller in its call-site location rather than failing.
  LogicalResult
  matchAndRewrite(SourceLocOp op, SourceLocOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<IntegerAttr>(op.getInlineCount()))
      return op.emitError(
          "failed to materialize inline count value for call location");

    LocationAttr bestFrame;
    DebugInfo::walkLocation(op.getLoc(),
                            DebugInfo::LocWalkPolicy::CallerPriority,
                            [&](Location loc) -> WalkResult {
                              if (isa<mlir::CallSiteLoc>(loc))
                                return WalkResult::advance();
                              bestFrame = loc;
                              return WalkResult::interrupt();
                            });

    Location loc = op.getLoc();
    FileLineColLoc frame =
        bestFrame ? DebugInfo::extractSourceLoc(bestFrame) : FileLineColLoc();

    Value line = ParamConstantOp::create(
        rewriter, loc, op.getLine().getType(),
        rewriter.getIndexAttr(frame ? frame.getLine() : 0));
    Value col = ParamConstantOp::create(
        rewriter, loc, op.getCol().getType(),
        rewriter.getIndexAttr(frame ? frame.getColumn() : 0));
    Value file = ParamConstantOp::create(
        rewriter, loc, op.getFileName().getType(),
        StringAttr::get(frame ? frame.getFilename().getValue()
                              : StringRef("<unknown location>"),
                        rewriter.getType<StringType>()));
    rewriter.replaceOp(op, {line, col, file});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructCreate
//===----------------------------------------------------------------------===//

struct ConvertKGENStructCreate : ConvertPOPToLLVMPattern<StructCreateOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructCreateOp op, StructCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kgenStructType = cast<StructType>(op.getType());
    Type llvmStructType = convertType(kgenStructType);
    if (!llvmStructType)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "failed to convert struct type");
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value container = LLVM::UndefOp::create(b, llvmStructType);

    // Use remapped field indices to account for padding fields.
    for (auto [logicalIdx, element] : llvm::enumerate(adaptor.getOperands())) {
      int64_t llvmIdx =
          getTypeConverter()->getRemappedFieldIndex(kgenStructType, logicalIdx);
      container = LLVM::InsertValueOp::create(b, container, element, llvmIdx);
    }

    rewriter.replaceOp(op, container);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructReplace
//===----------------------------------------------------------------------===//

struct ConvertKGENStructReplace : ConvertPOPToLLVMPattern<StructReplaceOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructReplaceOp op, StructReplaceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kgenStructType = cast<StructType>(op.getContainer().getType());
    int64_t llvmIdx = getTypeConverter()->getRemappedFieldIndex(
        kgenStructType, op.getIndexAttr().getInt());
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getContainer(), adaptor.getValue(), llvmIdx);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructGet
//===----------------------------------------------------------------------===//

struct ConvertKGENStructGet : ConvertPOPToLLVMPattern<StructExtractOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructExtractOp op, StructExtractOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto indexAttr = dyn_cast<IntegerAttr>(op.getIndexAttr());
    if (!indexAttr)
      return op.emitOpError("expected constant index for LLVM lowering");
    auto kgenStructType = cast<StructType>(op.getContainer().getType());
    int64_t llvmIdx = getTypeConverter()->getRemappedFieldIndex(
        kgenStructType, indexAttr.getInt());
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getContainer(), llvmIdx);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructGEP
//===----------------------------------------------------------------------===//

struct ConvertKGENStructGEP : ConvertPOPToLLVMPattern<StructGEPOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructGEPOp op, StructGEPOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PointerType ptrType = cast<PointerType>(op.getContainer().getType());
    auto kgenStructType = cast<StructType>(ptrType.getElementType());
    Type elementType = convertType(kgenStructType);
    if (!elementType)
      return op.emitError("failed to convert result type");
    auto indexAttr = cast<IntegerAttr>(op.getIndex());
    int64_t llvmIdx = getTypeConverter()->getRemappedFieldIndex(
        kgenStructType, indexAttr.getInt());
    unsigned addrSpace = ptrType.getAddrSpaceOrZero();
    auto opaquePtr = LLVM::LLVMPointerType::get(getContext(), addrSpace);
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, opaquePtr, elementType, adaptor.getContainer(),
        ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(llvmIdx)});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns,
                                       SymbolTable &symtab,
                                       InterpreterMemoryConverter &imc,
                                       bool &passFailed,
                                       const AttributeIdentifiers &ids) {
  patterns.insert<
      // clang-format off
      ConvertKGENCall,
      ConvertKGENSourceLoc,
      ConvertKGENStructCreate,
      ConvertKGENStructGEP,
      ConvertKGENStructGet,
      ConvertKGENStructReplace,
      ConvertKGENReturn,
      ConvertKGENUnreachable
      // clang-format on
      >(typeConverter);
  patterns.insert<ConvertKGENFunc>(typeConverter, symtab, ids);
  patterns.insert<ConvertKGENParamConstant, ConvertKGENParamMaterialize>(
      typeConverter, imc, passFailed);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENTOLLVM
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class LowerKGENToLLVMPass
    : public KGEN::impl::LowerKGENToLLVMBase<LowerKGENToLLVMPass> {
public:
  using LowerKGENToLLVMBase::LowerKGENToLLVMBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    using LLVM::LLVMDialect;
    auto id = [&](StringRef name) { return StringAttr::get(ctx, name); };

    ids.noalias = id(LLVMDialect::getNoAliasAttrName());
    ids.noundef = id(LLVMDialect::getNoUndefAttrName());
    ids.nonnull = id(LLVMDialect::getNonNullAttrName());

    return success();
  }

  void runOnOperation() override;

private:
  AttributeIdentifiers ids;
};
} // namespace

void LowerKGENToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<KGENDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<POP::POPDialect>();
  target.addLegalDialect<mlir::index::IndexDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addLegalOp<KGEN::CallIndirectOp>();
  target.addLegalOp<KGEN::CreateClosureOp>();
  target.addLegalOp<KGEN::StructInstanceOp>();

  // Configure the type converter.
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  POPToLLVMTypeConverter typeConverter(targetInfo);

  // Attach the LLVM data layout and target triple strings to the module so they
  // are present when exporting to LLVMIR.
  NamedAttrList moduleAttrs(theModule->getAttrDictionary());
  moduleAttrs.set(LLVM::LLVMDialect::getTargetTripleAttrName(),
                  StringAttr::get(&getContext(), targetInfo.getTripleStr()));
  moduleAttrs.set(
      LLVM::LLVMDialect::getDataLayoutAttrName(),
      StringAttr::get(&getContext(), targetInfo.getDataLayout().toString()));
  moduleAttrs.erase(EnvAttr::getEnvAttrName());
  theModule->setAttrs(moduleAttrs.getDictionary(&getContext()));

  // Recorded as a module flag, mirroring clang, rather than a function
  // attribute like `target-cpu`/`target-features`/`tune-cpu` below.
  if (!targetInfo.getAbi().empty()) {
    auto flag = LLVM::ModuleFlagAttr::get(
        &getContext(), LLVM::ModFlagBehavior::Error,
        StringAttr::get(&getContext(), "target-abi"),
        StringAttr::get(&getContext(), targetInfo.getAbi()));
    OpBuilder b(&getContext());
    b.setInsertionPointToStart(theModule.getBody());
    LLVM::ModuleFlagsOp::create(b, theModule.getLoc(),
                                ArrayAttr::get(&getContext(), {flag}));
  }

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  auto &symtabAnalysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTable &symtab = symtabAnalysis.getTopLevelSymbolTable();
  InterpreterMemoryConverter imc(symtab, typeConverter);
  bool passFailed = false;
  populateKGENToLLVMPatterns(typeConverter, patterns, symtab, imc, passFailed,
                             ids);

  DebugInfoTypeConverter debugTypeConverter(typeConverter, targetInfo, symtab);
  DebugInfo::populateTypeConversionPatterns(patterns, debugTypeConverter,
                                            typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Let the target lowering apply any target-specific finalization to each
  // converted function.
  if (ErrorOr<const TargetLowering *> loweringOr =
          TargetLoweringRegistry::get().lookup(
              typeConverter.getTarget().getTriple());
      !loweringOr.isError()) {
    const TargetLowering *lowering = *loweringOr;
    for (auto funcOp : theModule.getOps<LLVM::LLVMFuncOp>())
      lowering->finalizeConvertedFunction(funcOp);
  }

  // Apply C ABI to abi("C") function definitions. This rewrites the function
  // entry (C ABI args → Mojo types) and exits (Mojo return → C ABI), making
  // the function directly callable from C. Call sites were already patched
  // during conversion by ConvertKGENCall.
  {
    auto abiHandler =
        createCABIInfo(typeConverter.getTarget().getTriple(), &getContext(),
                       static_cast<const LLVMDataLayout &>(typeConverter));
    for (auto func : theModule.getOps<LLVM::LLVMFuncOp>())
      if (func->hasAttr(kCABIFuncAttr))
        processCABIFunctionDefinition(func, *abiHandler);
  }

  // Convert the debug info within the IR.
  assert(!debugTypeConverter.getError() &&
         "DebugInfoTypeConverter already has an error before applyRecursively; "
         "the converter must not be reused across calls");
  debugTypeConverter.applyRecursively(theModule);

  // Check for errors from the debug type converter. Converter lambdas have no
  // way to signal failure directly, so errors are recorded and checked here.
  if (std::optional<std::string> err = debugTypeConverter.getError()) {
    theModule.emitError("debug info type conversion failed: ") << *err;
    return signalPassFailure();
  }

  // All type symbols should be inaccessible now (they do not yet lower to
  // runtime). Erase them.
  for (auto structInstance :
       llvm::make_early_inc_range(theModule.getOps<StructInstanceOp>()))
    structInstance.erase();

  if (passFailed)
    signalPassFailure();
}
