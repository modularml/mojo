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

#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/ML/DType.h"
#include "Support/STLExtras.h"
#include "llvm/Support/FormatAdapters.h"
#include <filesystem>

using namespace M;
using namespace KGEN;

/// Get the C type string for the dtype.
static LogicalResult getCTypeForDType(FuncOp func, KGENDType dt,
                                      SmallVectorImpl<std::string> &types) {
  if (dt.isFloat()) {
    switch (dt.getValue()) {
    case DType::f32:
      types.push_back("float");
      return success();
    case DType::f64:
      types.push_back("double");
      return success();
    }
    return func.emitError("unhandled floating point dtype: ")
           << dt.getAsString();
  }
  if (dt.isInt()) {
    types.push_back(((dt.isUInt() ? "u" : "") + StringRef("int") +
                     Twine(dt.getWidthInBits()) + "_t")
                        .str());
    return success();
  }
  if (dt == DType::invalid) {
    types.push_back("void");
    return success();
  }
  if (dt.isBool()) {
    types.push_back("bool");
    return success();
  }
  if (dt.isIndex() || dt.isUIndex()) {
    types.push_back("ssize_t");
    return success();
  }
  return func.emitError("unhandled dtype for header generation ")
         << dt.getAsString();
}

/// Get the C type for an elementary scalar type.
static ErrorOr<std::string> getCTypeForElementary(Type t) {
  if (isa<IntegerType>(t)) {
    if (auto bitWidth = t.getIntOrFloatBitWidth(); bitWidth > 1)
      return ("int" + Twine(bitWidth) + "_t").str();
    return "bool";
  }
  if (isa<IndexType>(t))
    return "ssize_t";
  if (t.isF16())
    llvm::report_fatal_error("no support for fp16 yet");
  if (t.isF32())
    return "float";
  if (t.isF64())
    return "double";

  SmallString<128> str;
  llvm::raw_svector_ostream os(str);
  t.print(os);
  return Error("unhandled elementary type: '" + str + "'");
}

/// Get the C types for the given type.
static LogicalResult getCTypeForType(FuncOp func, Type t,
                                     SmallVectorImpl<std::string> &types) {
  if (isa<KGEN::NoneType>(t)) {
    types.push_back("void");
    return success();
  }

  if (auto simd = dyn_cast<SIMDType>(t)) {
    // Since the vector_size attribute only works on GNU and CLANG compilers,
    // we pass in an array.
    if (failed(getCTypeForDType(func, *simd.getResolvedDType(), types)))
      return failure();
    auto size = *simd.getResolvedSize();
    // size == 1 is a scalar
    if (size != 1)
      types.back() += ("[" + Twine(size) + "]").str();
    return success();
  }

  if (auto ptr = dyn_cast<PointerType>(t)) {
    ErrorOr<std::string> elementaryType =
        getCTypeForElementary(ptr.getElementType());
    // If the type is not elementary, then pass it as an opaque pointer.
    if (elementaryType.isError())
      types.push_back("void *");
    else
      types.push_back(elementaryType.takeValue() + "*");
    return success();
  }

  if (auto array = dyn_cast<POP::ArrayType>(t)) {
    if (!*array.getResolvedSize())
      return success();

    if (failed(getCTypeForType(func, array.getElementType(), types)))
      return failure();
    // Size-1 arrays are ABI-compatible with the scalar element type,
    // so omit the [1] suffix (e.g. Optional[UnsafePointer] lowers
    // to array<1, pointer> but should emit as `void *` in the C header).
    if (*array.getResolvedSize() != 1)
      types.back() += ("[" + Twine(*array.getResolvedSize()) + "]").str();
    return success();
  }

  if (auto structType = dyn_cast<StructType>(t)) {
    std::optional<SmallVector<Type>> elementTypes =
        structType.getElementTypes();
    if (!elementTypes)
      return func.emitError("cannot generate C type for parametric struct");
    for (Type elTy : *elementTypes)
      if (failed(getCTypeForType(func, elTy, types)))
        return failure();
    return success();
  }

  if (auto variadic = dyn_cast<ParamListType>(t)) {
    types.push_back("void *");
    types.push_back("ssize_t");
    return success();
  }

  if (isa<DTypeType>(t)) {
    types.push_back("uint8_t");
    return success();
  }

  if (!isa<IndexType, IntegerType, FloatType>(t))
    return func.emitError("unsupported argument type: ") << t;
  if (!t.isIndex() && !llvm::isPowerOf2_64(t.getIntOrFloatBitWidth()))
    return func.emitError("integer or float bitwidth must be a power of 2");

  ErrorOr<std::string> elementaryTypeName = getCTypeForElementary(t);
  if (elementaryTypeName.isError())
    return func->emitError(elementaryTypeName.takeError().get());
  types.push_back(elementaryTypeName.takeValue());
  return success();
}

/// Collapse a chain of single-field structs to the field they wrap. A pointer
/// wrapper such as `UnsafePointer` arrives here nested several deep, and one C
/// parameter of the innermost field is ABI-identical to the whole chain.
static Type collapseSingleFieldStructs(Type t) {
  while (auto structType = dyn_cast<StructType>(t)) {
    std::optional<SmallVector<Type>> elts = structType.getElementTypes();
    if (!elts || elts->size() != 1)
      break;
    t = (*elts)[0];
  }
  return t;
}

/// True if `t` fills exactly one eightbyte, so that spelling it as its own C
/// parameter puts it in the same register the platform ABI would give it.
static bool fillsOneEightbyte(Type t) {
  t = collapseSingleFieldStructs(t);
  if (isa<PointerType>(t) || isa<IndexType>(t))
    return true;
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth() == 64;
  if (auto floatTy = dyn_cast<FloatType>(t))
    return floatTy.getWidth() == 64;
  // A Mojo scalar field is a one-element SIMD type, e.g. `scalar<index>`.
  if (auto simd = dyn_cast<SIMDType>(t)) {
    std::optional<int64_t> size = simd.getResolvedSize();
    std::optional<KGENDType> dt = simd.getResolvedDType();
    if (!size || *size != 1 || !dt)
      return false;
    return dt->isIndex() || dt->isUIndex() || dt->getWidthInBits() == 64;
  }
  return false;
}

/// Reject structs whose flattened "one C param per field" spelling would not
/// match the platform ABI. Headergen has no struct name (LowerLIT drops it), so
/// it can only safely emit: a single collapsed field, or two fields that each
/// fill an eightbyte. Multi-field returns, >2 fields, and narrow fields are
/// refused — including some ABI-correct shapes (e.g. {float, double}) because
/// we lack DataLayout to prove padding. Workaround: pass a pointer.
/// TODO(MOCO-4513): emit a named `typedef struct` and drop this check.
static LogicalResult checkStructSpellingMatchesABI(FuncOp func, Type t,
                                                   bool isResult) {
  auto structType = dyn_cast<StructType>(collapseSingleFieldStructs(t));
  if (!structType)
    return success();

  // A parameter pack is spelled by the ParamListType path, not as a struct.
  if (structType.getIsParamPack())
    return success();

  std::optional<SmallVector<Type>> elts = structType.getElementTypes();
  if (!elts)
    return success(); // Parametric; getCTypeForType reports it.

  if (isResult)
    return func.emitError("cannot declare a C prototype returning struct ")
           << t
           << ": it is returned in registers, but the header would declare one "
              "out-parameter per field. Return a pointer instead, or see "
              "MOCO-4513";

  if (elts->size() > 2 || !llvm::all_of(*elts, fillsOneEightbyte))
    return func.emitError("cannot declare a C prototype taking struct ")
           << t
           << " by value: one parameter per field would not match the "
              "registers the platform ABI assigns it. Pass a pointer "
              "instead, or see MOCO-4513";

  return success();
}

/// Emit the C signature of a KGEN func.
static LogicalResult emitSignature(raw_ostream &os, FuncOp func) {
  SmallVector<std::string> argTys, resTys;
  for (Type type : func.getArgumentTypes()) {
    if (failed(checkStructSpellingMatchesABI(func, type, /*isResult=*/false)))
      return failure();
    if (failed(getCTypeForType(func, type, argTys)))
      return failure();
  }
  for (Type type : func.getResultTypes()) {
    if (failed(checkStructSpellingMatchesABI(func, type, /*isResult=*/true)))
      return failure();
    if (failed(getCTypeForType(func, type, resTys)))
      return failure();
  }

  // Print the function declaration.
  os << "extern ";

  // If there is exactly one result type, return it. If there are multiple,
  // return them as pointers.
  if (resTys.size() == 1)
    os << resTys.front();
  else
    os << "void";

  // FIXME: This assumes the C wrapper that eventually gets generated is not
  // renamed due to a symbol name conflict. Header emission happens too early
  // in the pipeline.
  os << ' ' << func.getSymName() << '(';
  llvm::interleaveComma(argTys, os);
  if (resTys.size() > 1) {
    if (!argTys.empty())
      os << ", ";
    llvm::interleaveComma(resTys, os,
                          [&](StringRef type) { os << type << " *"; });
  }
  os << ");";
  return success();
}

LogicalResult ObjectCompiler::emitCXXHeader(ModuleOp module, StringRef filename,
                                            llvm::raw_ostream &os) {
  const char *headerFmtStart = R"literal(//===-{0}-===//
//
// This file is Modular Inc proprietary.
//
//==={0}===//
// THIS FILE IS AUTOGENERATED BY `kgen`, DO NOT EDIT!
//==={0}===//

#ifndef {1}
#define {1}

#ifdef __cplusplus
extern "C" {{
#endif

#if defined(_WIN64) || defined(_WIN32)
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#else // defined(_WIN64) || defined(_WIN32)
#include <sys/types.h>
#endif // defined(_WIN64) || defined(_WIN32)

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

)literal";

  const char *headerFmtEnd = R"literal(
#ifdef __cplusplus
} // extern "C"
#endif

#endif // {0}
)literal";

  std::filesystem::path filepath(filename.str());
  std::string headerGuard =
      "__KGEN_" + StringRef(filepath.stem().string()).upper() + "_H";
  os << llvm::formatv(headerFmtStart,
                      llvm::fmt_repeat('-', 80 - 2 * strlen("//===")),
                      headerGuard);

  // Emit the function decls into the header.
  for (auto f : module.getOps<FuncOp>()) {
    if (!f.getFuncTypeGenerator().getBody().getFnEffects().isCABI())
      continue;
    // The symbol was exported, use its alias name.
    if (failed(emitSignature(os, f)))
      return mlir::emitError(f.getLoc(),
                             "during header emission for this function");
    os << "\n";
  }

  os << llvm::formatv(headerFmtEnd, headerGuard);
  return success();
}
